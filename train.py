import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import wandb
import random
import numpy as np
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from transformers import (
    AutoModel,
    AutoConfig,
    get_cosine_schedule_with_warmup
)
from tqdm import tqdm

from dataset import CompetitionDataset
from custom_create_submission import evaluate_knn_val_accuracy

load_dotenv()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_state_dict(model):
    """Get state dict from model, handling compiled models."""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod.state_dict()
    return model.state_dict()


def main():
    with open('cfg.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    set_seeds(cfg['training']['seed'])
    
    device = torch.device('cuda')
    
    wandb.init(
        entity='sposiboh',
        project='nyu_dl_competition',
        name=cfg['run_name'],
        config=cfg,
        mode='disabled' if cfg['run_name'] == 'debug' else 'online'
    )
    
    # Load dataset
    ds = CompetitionDataset(image_dir='/workspace/images/train', image_size=96, cache_in_memory=True)
    print(f"Dataset loaded with {len(ds)} samples")
    
    # Load image processor (use teacher's processor for consistency)
    # teacher_model_name = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'
    teacher_model_name = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    student_model_name = 'facebook/dino-vits8'
    
    teacher_model = AutoModel.from_pretrained(teacher_model_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print(f"Teacher model loaded: {teacher_model_name}")
    
    # Initialize student model (untrained - random weights from config only)
    student_config = AutoConfig.from_pretrained(student_model_name)
    student_config.image_size = 96
    # print(student_config)
    student_model = AutoModel.from_config(student_config)
    student_model = student_model.to(device)
    student_model.train()
    print(f"Student model initialized with random weights: {student_model_name}")
    
    # Create dataloader
    num_workers = cfg['training']['num_workers']
    train_loader = DataLoader(
        ds,
        batch_size=cfg['training']['train_bs'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay']
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * cfg['training']['epochs']
    warmup_steps = round(total_steps * cfg['training']['part_warmup'])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    if cfg['training'].get('compile', False) and hasattr(torch, 'compile'):
        student_model = torch.compile(student_model)
        teacher_model = torch.compile(teacher_model)
        print("Model compiled with torch.compile")
    
    # Projection head to match dimensions if needed
    student_dim = 384 # student_model.config.hidden_size
    teacher_dim = 768 # teacher_model.config.hidden_size
    

    projection_head = nn.Linear(student_dim, teacher_dim).to(device)
    mlp_prototype = nn.Sequential(
        nn.Linear(teacher_dim, cfg['prot']['hidden_prototype']),
        nn.GELU(),
        nn.LayerNorm(cfg['prot']['hidden_prototype']),
        nn.Linear(cfg['prot']['hidden_prototype'], cfg['prot']['dim_prototype'])
    ).to(device)
    center = torch.zeros(cfg['prot']['dim_prototype']).to(device)


    optimizer.add_param_group({'params': projection_head.parameters()})
    optimizer.add_param_group({'params': mlp_prototype.parameters(), 'weight_decay': cfg['prot']['mlp_prototype_wd']})
    print(f"Projection head added: {student_dim} -> {teacher_dim}")

    for epoch in range(cfg['training']['epochs']):
        student_model.train()
        if projection_head:
            projection_head.train()
        
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        
        for images in pbar:
            images = images.to(device)
            view1 = ds.transform(images)
            view2 = ds.transform(images)
            
            # inference backbone - student gets view1, teacher gets view2
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                student_out = student_model(view1).pooler_output.to(dtype=torch.float32)
                with torch.no_grad():
                    teacher_out = teacher_model(view2).pooler_output.to(dtype=torch.float32)
            # do loss stuff
            student_out = projection_head(student_out)
            student_prot = mlp_prototype(student_out)
            student_probs = F.softmax(student_prot / cfg['prot']['student_temp'], dim=-1)
            
            with torch.no_grad():
                teacher_prot = mlp_prototype(teacher_out)
                teacher_prot_centered = teacher_prot - center.unsqueeze(0)
                center = center * cfg['prot']['center_momentum'] + teacher_prot.mean(0) * (1 - cfg['prot']['center_momentum'])
                teacher_probs = F.softmax(teacher_prot_centered / cfg['prot']['teacher_temp'], dim=-1)
                
                teacher_ent = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(-1).mean()
                student_ent = -(student_probs * torch.log(student_probs + 1e-8)).sum(-1).mean()
                
                
            student_logprobs = torch.log(student_probs + 1e-8)
            prototype_loss = -(teacher_probs * student_logprobs).sum(dim=1).mean()
            cossim_loss = F.mse_loss(F.normalize(student_out, dim=-1), F.normalize(teacher_out.detach(), dim=-1))
            
            loss = cfg['prot']['prototype_coef'] * prototype_loss + cfg['prot']['cossim_coef'] * cossim_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), cfg['training']['grad_clip'])
            torch.nn.utils.clip_grad_norm_(projection_head.parameters(), cfg['training']['grad_clip'])
            torch.nn.utils.clip_grad_norm_(mlp_prototype.parameters(), cfg['training']['grad_clip'])
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            
            # Log to wandb
            wandb.log({
                'train/loss': loss.item(),
                'train/prototype_loss': prototype_loss.item(),
                'train/cossim_loss': cossim_loss.item(),
                'train/lr': lr_scheduler.get_last_lr()[0],
                'teacher_ent': teacher_ent.item(),
                'student_ent': student_ent.item(),
            })
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
        
        wandb.log({
            'train/epoch_loss': avg_epoch_loss,
            'train/epoch': epoch + 1,
        })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'student_model_state_dict': get_state_dict(student_model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': avg_epoch_loss,
        }
        if projection_head:
            checkpoint['projection_head_state_dict'] = get_state_dict(projection_head)
        
        checkpoint_path = f'{cfg["run_name"]}_checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        if (epoch + 1) % 3 == 0:
            val_acc, _ = evaluate_knn_val_accuracy(
                checkpoint_path=checkpoint_path,
                data_dir='/root/dl_nyu_competition/kaggle_data',
                k=5,
                batch_size=cfg['training']['train_bs'],
                num_workers=cfg['training']['num_workers'],
                device='cuda'
            )
            wandb.log({
                'knn/val_accuracy': val_acc,
            })
            
    
    val_acc, _ = evaluate_knn_val_accuracy(
        checkpoint_path=checkpoint_path,
        data_dir='/root/dl_nyu_competition/kaggle_data',
        k=5,
        batch_size=cfg['training']['train_bs'],
        num_workers=cfg['training']['num_workers'],
        device='cuda'
    )
    
    wandb.log({
        'knn/val_accuracy': val_acc,
    })
    
    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()

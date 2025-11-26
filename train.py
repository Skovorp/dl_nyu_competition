import torch
import torch.nn as nn
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

load_dotenv()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    ds = CompetitionDataset(split='train', image_size=96)
    print(f"Dataset loaded with {len(ds)} samples")
    
    # Load image processor (use teacher's processor for consistency)
    teacher_model_name = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'
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
    train_loader = DataLoader(
        ds,
        batch_size=cfg['training']['train_bs'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        drop_last=True
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
    teacher_dim = 1280 # teacher_model.config.hidden_size
    

    projection_head = nn.Linear(student_dim, teacher_dim).to(device)
    # Add projection head parameters to optimizer
    optimizer.add_param_group({'params': projection_head.parameters()})
    print(f"Projection head added: {student_dim} -> {teacher_dim}")

    for epoch in range(cfg['training']['epochs']):
        student_model.train()
        # if projection_head:
        #     projection_head.train()
        
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        
        for images in pbar:
            images = images.to(device)
            
            student_out = projection_head(student_model(images).pooler_output)
            with torch.no_grad():
                teacher_out = teacher_model(images).pooler_output
            
            loss = nn.functional.mse_loss(student_out, teacher_out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            
            # Log to wandb
            wandb.log({
                'train/loss': loss.item(),
                'train/lr': lr_scheduler.get_last_lr()[0],
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
            'student_model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': avg_epoch_loss,
        }
        if projection_head:
            checkpoint['projection_head_state_dict'] = projection_head.state_dict()
        
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')
        print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pt")
    
    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()

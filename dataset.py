import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image


class CompetitionDataset(Dataset):
    def __init__(self, image_dir='/workspace/images/train', image_size=96):
        self.image_paths = sorted([
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if f.endswith('.jpg')
        ])
        # BYOL/DINO-style augmentations
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.4, 1.0),
                ratio=(0.75, 1.33),
                antialias=True
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            v2.RandomSolarize(threshold=128, p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2
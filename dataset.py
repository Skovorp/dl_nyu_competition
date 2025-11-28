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
        self.transform = self._make_transform(image_size)
    
    def _make_transform(self, resize_size):
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
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
        return self.transform(image)
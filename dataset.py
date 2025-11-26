import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision.transforms import v2


class CompetitionDataset(Dataset):
    def __init__(self, split='train', image_size=96):
        self.ds = load_dataset('tsbpp/fall2025_deeplearning', split=split)
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
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item['image'].convert('RGB')
        image = self.transform(image)
        return image


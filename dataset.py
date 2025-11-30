import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def _load_single_image(path):
    """Load a single image - used for parallel loading."""
    return read_image(path, mode=ImageReadMode.RGB)


class CompetitionDataset(Dataset):
    def __init__(self, image_dir='/workspace/images/train', image_size=96, cache_in_memory=True, num_loading_workers=32):
        self.image_paths = sorted([
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if f.endswith('.jpg')
        ])
        self.cache_in_memory = cache_in_memory
        
        # Pre-load all images into memory using parallel workers
        if cache_in_memory:
            print(f"Pre-loading {len(self.image_paths)} images into memory with {num_loading_workers} workers...")
            self.images = [None] * len(self.image_paths)
            
            with ThreadPoolExecutor(max_workers=num_loading_workers) as executor:
                # Submit all loading tasks
                future_to_idx = {
                    executor.submit(_load_single_image, path): idx 
                    for idx, path in enumerate(self.image_paths)
                }
                # Collect results with progress bar
                for future in tqdm(as_completed(future_to_idx), total=len(self.image_paths), desc="Loading images"):
                    idx = future_to_idx[future]
                    self.images[idx] = future.result()
            
            print(f"Cached {len(self.images)} images in memory")
        
        # BYOL/DINO-style augmentations (no ToImage needed - already tensor)
        self.transform = v2.Compose([
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
        if self.cache_in_memory:
            return self.images[idx]
        return read_image(self.image_paths[idx], mode=ImageReadMode.RGB)
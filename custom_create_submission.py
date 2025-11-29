"""
Create Kaggle Submission with Custom Trained Model + KNN Classifier
====================================================================

This script uses:
- Custom trained student model (dino-vits8) from checkpoint
- KNN classifier for predictions

Usage:
    python custom_create_submission.py \
        --data_dir ./kaggle_data \
        --checkpoint checkpoint_epoch_1.pt \
        --output submission.csv \
        --k 5
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from transformers import AutoModel, AutoConfig
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import argparse


# ============================================================================
#                          MODEL SECTION
# ============================================================================

def strip_compiled_prefix(state_dict):
    """Strip _orig_mod. prefix from compiled model state dicts."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class StudentFeatureExtractor:
    """
    Feature extractor using the custom trained student model.
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize feature extractor from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint_epoch_X.pt
            device: 'cuda' or 'cpu'
        """
        print(f"Loading student model from checkpoint: {checkpoint_path}")
        
        # Initialize student model architecture (same as training)
        student_model_name = 'facebook/dino-vits8'
        student_config = AutoConfig.from_pretrained(student_model_name)
        student_config.image_size = 96
        
        self.model = AutoModel.from_config(student_config)
        
        # Load checkpoint (handle compiled model state dicts)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = strip_compiled_prefix(checkpoint['student_model_state_dict'])
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        
        # Create transform (same normalization as training)
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((96, 96), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        
        print(f"  Model loaded successfully (feature_dim=384)")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Checkpoint loss: {checkpoint.get('loss', 'unknown'):.4f}" if 'loss' in checkpoint else "")
        
    def extract_features(self, image):
        """
        Extract features from a single PIL Image.
        
        Args:
            image: PIL Image
        
        Returns:
            features: numpy array of shape (feature_dim,)
        """
        # Transform image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Use pooler_output (same as training)
        features = outputs.pooler_output
        
        return features.cpu().numpy()[0]
    
    def extract_batch_features(self, images):
        """
        Extract features from a batch of PIL Images.
        
        Args:
            images: List of PIL Images
        
        Returns:
            features: numpy array of shape (batch_size, feature_dim)
        """
        # Transform batch
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(batch)
        
        # Use pooler_output
        features = outputs.pooler_output
        
        return features.cpu().numpy()


# ============================================================================
#                          DATA SECTION
# ============================================================================

class ImageDataset(Dataset):
    """Simple dataset for loading images"""
    
    def __init__(self, image_dir, image_list, labels=None):
        """
        Args:
            image_dir: Directory containing images
            image_list: List of image filenames
            labels: List of labels (optional, for train/val)
        """
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        
        # Load image (transformation done in feature extractor)
        image = Image.open(img_path).convert('RGB')
        
        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name


def collate_fn(batch):
    """Custom collate function to handle PIL images"""
    if len(batch[0]) == 3:  # train/val (image, label, filename)
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        filenames = [item[2] for item in batch]
        return images, labels, filenames
    else:  # test (image, filename)
        images = [item[0] for item in batch]
        filenames = [item[1] for item in batch]
        return images, filenames


# ============================================================================
#                          FEATURE EXTRACTION
# ============================================================================

def extract_features_from_dataloader(feature_extractor, dataloader, split_name='train'):
    """
    Extract features from a dataloader.
    
    Args:
        feature_extractor: StudentFeatureExtractor instance
        dataloader: DataLoader
        split_name: Name of split (for progress bar)
    
    Returns:
        features: numpy array (N, feature_dim)
        labels: list of labels (or None for test)
        filenames: list of filenames
    """
    all_features = []
    all_labels = []
    all_filenames = []
    
    print(f"\nExtracting features from {split_name} set...")
    
    for batch in tqdm(dataloader, desc=f"{split_name} features"):
        if len(batch) == 3:  # train/val
            images, labels, filenames = batch
            all_labels.extend(labels)
        else:  # test
            images, filenames = batch
        
        # Extract features for batch
        features = feature_extractor.extract_batch_features(images)
        all_features.append(features)
        all_filenames.extend(filenames)
    
    features = np.concatenate(all_features, axis=0)
    labels = all_labels if all_labels else None
    
    print(f"  Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    
    return features, labels, all_filenames


# ============================================================================
#                          KNN CLASSIFIER
# ============================================================================

def train_knn_classifier(train_features, train_labels, val_features, val_labels, k=5):
    """
    Train KNN classifier on features.
    
    Args:
        train_features: Training features (N_train, feature_dim)
        train_labels: Training labels (N_train,)
        val_features: Validation features (N_val, feature_dim)
        val_labels: Validation labels (N_val,)
        k: Number of neighbors
    
    Returns:
        classifier: Trained KNN classifier
    """
    print(f"\nTraining KNN classifier (k={k})...")
    
    classifier = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',  # Weight by inverse distance
        metric='cosine',  # Cosine similarity for embeddings
        n_jobs=-1
    )
    
    classifier.fit(train_features, train_labels)
    
    # Evaluate
    train_acc = classifier.score(train_features, train_labels)
    val_acc = classifier.score(val_features, val_labels)
    
    print(f"\nKNN Results:")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return classifier


# ============================================================================
#                          SUBMISSION CREATION
# ============================================================================

def create_submission(test_features, test_filenames, classifier, output_path):
    """
    Create submission.csv for Kaggle.
    
    Args:
        test_features: Test features (N_test, feature_dim)
        test_filenames: List of test image filenames
        classifier: Trained KNN classifier
        output_path: Path to save submission.csv
    """
    print("\nGenerating predictions on test set...")
    predictions = classifier.predict(test_features)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'class_id': predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Submission file created: {output_path}")
    print(f"{'='*60}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    print(f"\nClass distribution in predictions:")
    print(submission_df['class_id'].value_counts().head(10))
    
    # Validate submission format
    print(f"\nValidating submission format...")
    assert list(submission_df.columns) == ['id', 'class_id'], "Invalid columns!"
    assert submission_df['class_id'].min() >= 0, "Invalid class_id < 0"
    assert submission_df['class_id'].max() <= 199, "Invalid class_id > 199"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")


# ============================================================================
#                          EVALUATION FUNCTION (for training)
# ============================================================================

def evaluate_knn_val_accuracy(
    checkpoint_path,
    data_dir='/workspace/kaggle_data',
    k=5,
    batch_size=64,
    num_workers=4,
    device='cuda'
):
    """
    Evaluate model checkpoint using KNN classifier and return validation accuracy.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Root directory containing train/val folders
        k: Number of neighbors for KNN
        batch_size: Batch size for feature extraction
        num_workers: Number of workers for data loading
        device: Device to use (cuda or cpu)
    
    Returns:
        val_acc: Validation accuracy (float)
        train_acc: Training accuracy (float)
    """
    device = device if torch.cuda.is_available() else 'cpu'
    data_dir = Path(data_dir)
    
    # Load CSV files
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    
    # Create datasets
    train_dataset = ImageDataset(
        data_dir / 'train',
        train_df['filename'].tolist(),
        train_df['class_id'].tolist()
    )
    
    val_dataset = ImageDataset(
        data_dir / 'val',
        val_df['filename'].tolist(),
        val_df['class_id'].tolist()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize feature extractor with checkpoint
    feature_extractor = StudentFeatureExtractor(
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # Extract features
    train_features, train_labels, _ = extract_features_from_dataloader(
        feature_extractor, train_loader, 'train'
    )
    val_features, val_labels, _ = extract_features_from_dataloader(
        feature_extractor, val_loader, 'val'
    )
    
    # Train KNN classifier
    classifier = train_knn_classifier(
        train_features, train_labels,
        val_features, val_labels,
        k=k
    )
    
    # Get accuracies
    train_acc = classifier.score(train_features, train_labels)
    val_acc = classifier.score(val_features, val_labels)
    
    return val_acc, train_acc


# ============================================================================
#                          MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Create Kaggle Submission with Custom Model + KNN')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_epoch_1.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output path for submission file')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors for KNN')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    
    # Load CSV files
    print("\nLoading dataset metadata...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    print(f"  Classes: {train_df['class_id'].nunique()}")
    
    # Create datasets
    print(f"\nCreating datasets...")
    train_dataset = ImageDataset(
        data_dir / 'train',
        train_df['filename'].tolist(),
        train_df['class_id'].tolist()
    )
    
    val_dataset = ImageDataset(
        data_dir / 'val',
        val_df['filename'].tolist(),
        val_df['class_id'].tolist()
    )
    
    test_dataset = ImageDataset(
        data_dir / 'test',
        test_df['filename'].tolist(),
        labels=None
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize feature extractor with custom checkpoint
    feature_extractor = StudentFeatureExtractor(
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    # Extract features
    train_features, train_labels, _ = extract_features_from_dataloader(
        feature_extractor, train_loader, 'train'
    )
    val_features, val_labels, _ = extract_features_from_dataloader(
        feature_extractor, val_loader, 'val'
    )
    test_features, _, test_filenames = extract_features_from_dataloader(
        feature_extractor, test_loader, 'test'
    )
    
    # Train KNN classifier
    classifier = train_knn_classifier(
        train_features, train_labels,
        val_features, val_labels,
        k=args.k
    )
    
    # Create submission
    create_submission(test_features, test_filenames, classifier, args.output)
    
    print("\n" + "="*60)
    print("DONE! Now upload your submission.csv to Kaggle.")
    print("="*60)


if __name__ == "__main__":
    main()


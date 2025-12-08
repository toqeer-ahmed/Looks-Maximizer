import os
import sys
import time
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# --- Configuration ---
CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- 1. Dataset Class ---

class CelebADataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Attribute Groups
        self.attr_groups = {
            "hair": ["Bald", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Straight_Hair", "Wavy_Hair"],
            "beard": ["No_Beard", "Mustache", "Goatee", "Sideburns"],
            "glasses": ["Eyeglasses"], # Corrected from Wearing_Glasses based on standard CelebA
            "attractive": ["Attractive"],
            "gender": ["Male"],
            "smiling": ["Smiling"]
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row.name # Index is filename
        img_path = self.img_dir / img_name
        
        # Load Image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Image {img_path} not found or corrupted.")
            image = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for transforms
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
            
        # Extract Labels
        labels = {}
        
        # Hair (Multi-label)
        hair_labels = [row[attr] for attr in self.attr_groups["hair"]]
        labels["hair"] = torch.tensor(hair_labels, dtype=torch.float32)
        
        # Beard (Multi-label)
        beard_labels = [row[attr] for attr in self.attr_groups["beard"]]
        labels["beard"] = torch.tensor(beard_labels, dtype=torch.float32)
        
        # Binary Attributes
        labels["glasses"] = torch.tensor([row["Eyeglasses"]], dtype=torch.float32)
        labels["attractive"] = torch.tensor([row["Attractive"]], dtype=torch.float32)
        labels["gender"] = torch.tensor([row["Male"]], dtype=torch.float32)
        labels["smiling"] = torch.tensor([row["Smiling"]], dtype=torch.float32)
        
        return image, labels

# --- 2. Model Architecture ---

class MultiTaskCelebAModel(nn.Module):
    def __init__(self):
        super(MultiTaskCelebAModel, self).__init__()
        # Backbone
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity() # Remove default head
        
        # Heads
        # Hair: 7 classes (Multi-label)
        self.head_hair = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 7)
        )
        
        # Beard: 4 classes (Multi-label)
        self.head_beard = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
        
        # Binary Heads
        self.head_glasses = nn.Linear(num_features, 1)
        self.head_attractive = nn.Linear(num_features, 1)
        self.head_gender = nn.Linear(num_features, 1)
        self.head_smiling = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return {
            "hair": self.head_hair(features),
            "beard": self.head_beard(features),
            "glasses": self.head_glasses(features),
            "attractive": self.head_attractive(features),
            "gender": self.head_gender(features),
            "smiling": self.head_smiling(features)
        }

# --- 3. Training & Evaluation ---

class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2)
        
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate Loss
            loss = 0
            loss += self.criterion(outputs['hair'], labels['hair'])
            loss += self.criterion(outputs['beard'], labels['beard'])
            loss += self.criterion(outputs['glasses'], labels['glasses'])
            loss += self.criterion(outputs['attractive'], labels['attractive'])
            loss += self.criterion(outputs['gender'], labels['gender'])
            loss += self.criterion(outputs['smiling'], labels['smiling'])
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}
                
                outputs = self.model(images)
                
                loss = 0
                loss += self.criterion(outputs['hair'], labels['hair'])
                loss += self.criterion(outputs['beard'], labels['beard'])
                loss += self.criterion(outputs['glasses'], labels['glasses'])
                loss += self.criterion(outputs['attractive'], labels['attractive'])
                loss += self.criterion(outputs['gender'], labels['gender'])
                loss += self.criterion(outputs['smiling'], labels['smiling'])
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)

# --- 4. Main Pipeline ---

def main():
    parser = argparse.ArgumentParser(description="CelebA Multi-Task Training Pipeline")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to ml_pipelines/celebA")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    # Robust Path Search
    possible_attr_paths = [
        dataset_path / "list_attr_celeba.txt",
        dataset_path / "Anno/list_attr_celeba.txt",
        dataset_path / "list_attr_celeba.csv",  # Kaggle version
        dataset_path / "Anno/list_attr_celeba.csv"
    ]
    
    attr_path = None
    for p in possible_attr_paths:
        if p.exists():
            attr_path = p
            break
            
    # Search for the folder that ACTUALLY contains the images
    possible_img_dirs = [
        dataset_path / "img_align_celeba",
        dataset_path / "Img/img_align_celeba",
        dataset_path / "img_align_celeba/img_align_celeba", # Double nested
        dataset_path # Maybe images are right in the root?
    ]
    
    img_dir = None
    for d in possible_img_dirs:
        if d.exists():
            # Check if it actually contains images
            if any(d.glob("*.jpg")):
                img_dir = d
                break
    
    if not attr_path or not img_dir:
        print(f"Error: Dataset files not found at {dataset_path}")
        print(f"Searched for attributes in: {[str(p) for p in possible_attr_paths]}")
        print(f"Searched for images in: {[str(d) for d in possible_img_dirs]}")
        return

    print(f"Found attributes at: {attr_path}")
    print(f"Found images at: {img_dir}")

    # 1. Load & Parse Data
    print(f"Loading attributes from {attr_path}...")
    
    if attr_path.suffix == '.csv':
        # Kaggle CSV format
        df = pd.read_csv(attr_path)
        df.set_index('image_id', inplace=True) # Kaggle CSV usually has 'image_id' column
    else:
        # Original TXT format
        df = pd.read_csv(attr_path, delim_whitespace=True, skiprows=1)
    
    df = df.replace(-1, 0) # Normalize -1 to 0
    
    print(f"Loaded {len(df)} images.")
    
    # 2. Split Data
    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 3. Transforms
    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Data Loaders
    train_ds = CelebADataset(train_df, img_dir, transform=transform)
    val_ds = CelebADataset(val_df, img_dir, transform=transform)
    test_ds = CelebADataset(test_df, img_dir, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    # 5. Initialize Model
    print(f"Initializing MobileNetV3 Large on {CONFIG['device']}...")
    model = MultiTaskCelebAModel()
    
    # 6. Train
    trainer = Trainer(model, train_loader, val_loader, CONFIG['device'])
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        trainer.scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), models_dir / "best_model.pth")
            patience_counter = 0
            print("Saved Best Model.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early Stopping.")
                break
                
    # 7. Evaluation (Basic)
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load(models_dir / "best_model.pth"))
    model.eval()
    model.to(CONFIG['device'])
    
    # Placeholder for detailed evaluation metrics
    # In a real run, we would iterate test_loader and compute F1 per attribute
    
    # 8. Export
    print("\nExporting Model...")
    export_dir = Path("export")
    export_dir.mkdir(exist_ok=True)
    
    dummy_input = torch.randn(1, 3, 224, 224).to(CONFIG['device'])
    torch.onnx.export(
        model,
        dummy_input,
        export_dir / "celeba_multitask.onnx",
        input_names=['input'],
        output_names=['hair', 'beard', 'glasses', 'attractive', 'gender', 'smiling'],
        opset_version=11
    )
    print(f"Model exported to {export_dir / 'celeba_multitask.onnx'}")

if __name__ == "__main__":
    main()

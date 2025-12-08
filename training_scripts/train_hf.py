import os
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import io
from PIL import Image

# --- Configuration ---
CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- 1. Dataset Class ---
class HFDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Decode Image
        # The image column contains a dict with 'bytes' key
        img_data = row['image']
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img_bytes = img_data['bytes']
        else:
            # Fallback if it's raw bytes or other format
            img_bytes = img_data
            
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"Error decoding image at idx {idx}: {e}")
            image = Image.new("RGB", (CONFIG['img_size'], CONFIG['img_size']))

        if self.transform:
            image = self.transform(image)
            
        # Labels
        labels = {
            "age": torch.tensor(row['age'], dtype=torch.long),
            "gender": torch.tensor(row['gender'], dtype=torch.float32), # Binary
            "race": torch.tensor(row['race'], dtype=torch.long)
        }
        
        return image, labels

# --- 2. Model Architecture ---
class MultiTaskHFModel(nn.Module):
    def __init__(self):
        super(MultiTaskHFModel, self).__init__()
        # Backbone
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        
        # Heads
        # Age: 9 classes
        self.head_age = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 9)
        )
        
        # Gender: Binary (0/1)
        self.head_gender = nn.Linear(num_features, 1)
        
        # Race: 7 classes
        self.head_race = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        features = self.backbone(x)
        return {
            "age": self.head_age(features),
            "gender": self.head_gender(features),
            "race": self.head_race(features)
        }

# --- 3. Training ---
class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2)
        
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_bin = nn.BCEWithLogitsLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            loss = 0
            loss += self.criterion_cls(outputs['age'], labels['age'])
            loss += self.criterion_bin(outputs['gender'].squeeze(), labels['gender'])
            loss += self.criterion_cls(outputs['race'], labels['race'])
            
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
                loss += self.criterion_cls(outputs['age'], labels['age'])
                loss += self.criterion_bin(outputs['gender'].squeeze(), labels['gender'])
                loss += self.criterion_cls(outputs['race'], labels['race'])
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to folder containing .parquet files")
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Load all parquet files
    print(f"Loading data from {data_dir}...")
    files = list(data_dir.rglob("*.parquet"))
    if not files:
        print("No .parquet files found!")
        return
        
    dfs = []
    for f in files:
        print(f"Reading {f.name}...")
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"ERROR reading {f.name}: {e}")
            print(f"File size: {f.stat().st_size} bytes")
            print("Skipping this file. Please re-upload it if needed.")
    
    # Filter out None values from failed reads
    dfs = [df for df in dfs if df is not None]
    
    if not dfs:
        print("No valid parquet files loaded. Exiting.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(full_df)}")
    
    # Split
    # We can't easily track which df came from which file anymore since we merged them.
    # So we will just do a random split on the full dataset.
    print("Splitting data 80/20...")
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)
        
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = HFDataset(train_df, transform=train_transform)
    val_ds = HFDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    # Model
    model = MultiTaskHFModel()
    trainer = Trainer(model, train_loader, val_loader, CONFIG['device'])
    
    # Train
    models_dir = Path("models_hf")
    models_dir.mkdir(exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        trainer.scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), models_dir / "best_model_hf.pth")
            print("Saved Best Model.")

    # Export ONNX
    print("\nExporting to ONNX...")
    model.load_state_dict(torch.load(models_dir / "best_model_hf.pth"))
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(CONFIG['device'])
    
    torch.onnx.export(
        model,
        dummy_input,
        "hf_model.onnx",
        input_names=['input'],
        output_names=['age', 'gender', 'race'],
        opset_version=11
    )
    print("Exported hf_model.onnx")

if __name__ == "__main__":
    main()

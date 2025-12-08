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
from PIL import Image

# --- Configuration ---
CONFIG = {
    "img_size": 256,
    "batch_size": 32,
    "num_workers": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- 1. Dataset Class ---
class SCUTDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Read labels file (Space separated: Filename Score)
        # Some lines might have multiple spaces, so we use regex delimiter '\s+'
        self.df = pd.read_csv(labels_file, sep=r"\s+", header=None, names=["filename", "score"])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        score = row['score']
        
        # Image path: root_dir / Images / img_name
        # Note: We need to find where the images are. 
        # Usually SCUT dataset has an 'Images' folder.
        # We will search for it.
        img_path = self.root_dir / "Images" / img_name
        
        if not img_path.exists():
            # Try searching recursively if not found directly
            found = list(self.root_dir.rglob(img_name))
            if found:
                img_path = found[0]
            else:
                # Create a black image if not found
                image = Image.new("RGB", (CONFIG['img_size'], CONFIG['img_size']))
                if self.transform:
                    image = self.transform(image)
                return image, torch.tensor(score, dtype=torch.float32)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new("RGB", (CONFIG['img_size'], CONFIG['img_size']))

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(score, dtype=torch.float32)

# --- 2. Model Architecture ---
class BeautyScorerModel(nn.Module):
    def __init__(self):
        super(BeautyScorerModel, self).__init__()
        # Backbone: ResNet50 (More powerful than MobileNet)
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output.squeeze()

# --- 3. Training ---
class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-5, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        
        self.criterion = nn.MSELoss() # Regression Loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for images, scores in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            scores = scores.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            loss = self.criterion(outputs, scores)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, scores in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                scores = scores.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, scores)
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to SCUT-FBP5500_v2 folder")
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    labels_file = data_dir / "train_test_files" / "All_labels.txt"
    
    if not labels_file.exists():
        # Try finding it recursively
        print(f"Searching for All_labels.txt in {data_dir}...")
        found = list(data_dir.rglob("All_labels.txt"))
        if found:
            labels_file = found[0]
        else:
            # Fallback: Check if it's in the current directory (root)
            if Path("All_labels.txt").exists():
                labels_file = Path("All_labels.txt")
            else:
                print(f"Error: All_labels.txt not found in {data_dir} or current directory.")
                # List contents to help debug
                print("Directory contents:")
                for p in data_dir.glob("*"):
                    print(f" - {p.name}")
                return

    print(f"Found labels at: {labels_file}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    full_ds = SCUTDataset(data_dir, labels_file, transform=None) # Transform applied later? No, usually in init.
    # Let's split indices first
    dataset_size = len(full_ds)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Re-init datasets with transforms
    train_ds = SCUTDataset(data_dir, labels_file, transform=train_transform)
    val_ds = SCUTDataset(data_dir, labels_file, transform=val_transform)
    
    train_ds = torch.utils.data.Subset(train_ds, train_indices)
    val_ds = torch.utils.data.Subset(val_ds, val_indices)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model
    model = BeautyScorerModel()
    trainer = Trainer(model, train_loader, val_loader, CONFIG['device'])
    
    # Train
    models_dir = Path("models_scut")
    models_dir.mkdir(exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()
        print(f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        
        trainer.scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), models_dir / "best_model_scut.pth")
            print("Saved Best Model.")

    # Export ONNX
    print("\nExporting to ONNX...")
    model.load_state_dict(torch.load(models_dir / "best_model_scut.pth"))
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256).to(CONFIG['device'])
    
    torch.onnx.export(
        model,
        dummy_input,
        "scut_model.onnx",
        input_names=['input'],
        output_names=['score'],
        opset_version=11
    )
    print("Exported scut_model.onnx")

if __name__ == "__main__":
    main()

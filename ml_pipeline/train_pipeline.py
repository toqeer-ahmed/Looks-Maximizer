import os
import sys
import time
import json
import argparse
import zipfile
import urllib.request
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Optional dependencies for specific tasks
try:
    import mediapipe as mp
except ImportError:
    print("Mediapipe not installed. Landmark extraction will fail.")

try:
    import kaggle
except ImportError:
    print("Kaggle API not installed. Dataset downloading via Kaggle will fail.")

# --- Configuration ---
CONFIG = {
    "dirs": {
        "data": "datasets",
        "processed": "datasets/processed",
        "models": "models",
        "results": "results",
        "web_model": "web_model"
    },
    "img_size": 224,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-4,
    "patience": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4
}

# --- 1. Dataset Downloading ---

class DatasetManager:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def download_celeba(self):
        print("Checking CelebA...")
        target_dir = self.base_dir / "celeba"
        if target_dir.exists():
            print("CelebA found.")
            return
        
        print("Downloading CelebA via Kaggle API...")
        # Requires ~/.kaggle/kaggle.json
        os.system(f"kaggle datasets download -d jessicali9530/celeba-dataset -p {self.base_dir}")
        
        zip_path = self.base_dir / "celeba-dataset.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_dir / "celeba")
            os.remove(zip_path)

    def download_utkface(self):
        print("Checking UTKFace...")
        target_dir = self.base_dir / "utkface"
        if target_dir.exists():
            print("UTKFace found.")
            return

        print("Downloading UTKFace via Kaggle API...")
        os.system(f"kaggle datasets download -d jangedoo/utkface-new -p {self.base_dir}")
        
        zip_path = self.base_dir / "utkface-new.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_dir / "utkface")
            os.remove(zip_path)

    def download_fairface(self):
        print("Checking FairFace...")
        target_dir = self.base_dir / "fairface"
        if target_dir.exists():
            print("FairFace found.")
            return
        
        # FairFace usually requires direct URL or Kaggle mirror
        print("Downloading FairFace via Kaggle API (Mirror)...")
        os.system(f"kaggle datasets download -d jessicali9530/fairface -p {self.base_dir}")
        
        zip_path = self.base_dir / "fairface.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_dir / "fairface")
            os.remove(zip_path)

    def download_fitzpatrick17k(self):
        print("Checking Fitzpatrick17k...")
        target_dir = self.base_dir / "fitzpatrick17k"
        if target_dir.exists():
            print("Fitzpatrick17k found.")
            return
        
        print("Please manually download Fitzpatrick17k due to licensing and place in datasets/fitzpatrick17k")
        # Placeholder for automatic download if a direct link was available and licensed

    def prepare_all(self):
        self.download_celeba()
        self.download_utkface()
        self.download_fairface()
        self.download_fitzpatrick17k()

# --- 2. Preprocessing & Label Harmonization ---

class Preprocessor:
    def __init__(self, raw_dir, processed_dir):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_face_shape(self, landmarks, width, height):
        """
        Estimate face shape based on geometric ratios of landmarks.
        Landmarks: 
        - Chin: 152
        - Forehead: 10
        - Left Cheek: 234
        - Right Cheek: 454
        - Left Jaw: 58
        - Right Jaw: 288
        """
        # Convert normalized landmarks to pixel coordinates
        coords = {}
        for idx in [10, 152, 234, 454, 58, 288]:
            lm = landmarks[idx]
            coords[idx] = np.array([lm.x * width, lm.y * height])

        # Calculate dimensions
        face_length = np.linalg.norm(coords[10] - coords[152])
        face_width_cheek = np.linalg.norm(coords[234] - coords[454])
        jaw_width = np.linalg.norm(coords[58] - coords[288])

        ratio_len_width = face_length / face_width_cheek
        ratio_jaw_cheek = jaw_width / face_width_cheek

        # Simplified Logic Rules
        if ratio_len_width > 1.5:
            return "oval" # or oblong
        elif ratio_len_width < 1.15:
            return "square" if ratio_jaw_cheek > 0.9 else "round"
        else:
            if ratio_jaw_cheek < 0.7:
                return "heart" # or diamond depending on forehead
            elif ratio_jaw_cheek > 0.9:
                return "square"
            else:
                return "oval" # Default

    def process_image(self, img_path, output_path):
        try:
            img = cv2.imread(str(img_path))
            if img is None: return None, None

            # RGB Conversion
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape

            # Landmark Extraction
            results = self.face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return None, None
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Face Shape Estimation
            face_shape = self.get_face_shape(landmarks, w, h)

            # Resize
            img_resized = cv2.resize(img_rgb, (CONFIG['img_size'], CONFIG['img_size']))
            
            # Save Processed Image
            cv2.imwrite(str(output_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

            # Return landmarks as list of tuples
            lm_list = [(lm.x, lm.y, lm.z) for lm in landmarks]
            return face_shape, lm_list

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, None

    def run_pipeline(self):
        # This function would iterate through all raw datasets, 
        # normalize labels, run process_image, and build a master CSV.
        # For brevity, we will simulate the metadata creation structure.
        
        metadata = []
        
        # Example: Walk through UTKFace
        utk_dir = self.raw_dir / "utkface/UTKFace"
        if utk_dir.exists():
            print("Processing UTKFace...")
            for img_file in tqdm(list(utk_dir.glob("*.jpg"))[:100]): # Limit for demo
                # UTK Filename format: [age]_[gender]_[race]_[date].jpg
                parts = img_file.name.split('_')
                if len(parts) < 4: continue
                
                try:
                    age = int(parts[0])
                    gender = int(parts[1]) # 0: Male, 1: Female
                    race = int(parts[2]) # 0: White, 1: Black, 2: Asian, 3: Indian, 4: Others
                except: continue

                out_name = f"utk_{img_file.name}"
                out_path = self.processed_dir / out_name
                
                shape, lms = self.process_image(img_file, out_path)
                if shape:
                    metadata.append({
                        "filename": out_name,
                        "source": "utkface",
                        "age": age,
                        "gender": gender, # 0: Male, 1: Female
                        "ethnicity": race, # Need to map to unified 7 classes
                        "face_shape": shape,
                        "landmarks": lms
                        # Add placeholders for others
                    })

        # Save Master CSV
        df = pd.DataFrame(metadata)
        df.to_csv(self.processed_dir / "master_labels.csv", index=False)
        print(f"Preprocessing complete. {len(df)} images processed.")

# --- 3. Data Loader ---

class FaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Mappings
        self.shape_map = {k: v for v, k in enumerate(["oval", "round", "square", "triangle", "heart", "diamond"])}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = self.root_dir / row['filename']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        # Prepare Labels
        labels = {
            "face_shape": self.shape_map.get(row['face_shape'], 0),
            "gender": int(row['gender']),
            "age": float(row['age']),
            "ethnicity": int(row['ethnicity']) if 'ethnicity' in row else 0,
            # Add other heads...
        }
        
        return image, labels

# --- 4. Model Architecture ---

class MultiTaskFaceModel(nn.Module):
    def __init__(self):
        super(MultiTaskFaceModel, self).__init__()
        # Backbone
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity() # Remove default head
        
        # Heads
        self.head_shape = nn.Linear(num_features, 6)      # Face Shape
        self.head_gender = nn.Linear(num_features, 2)     # Gender
        self.head_age = nn.Linear(num_features, 1)        # Age
        self.head_ethnicity = nn.Linear(num_features, 7)  # Ethnicity
        # Add heads for hair, beard, glasses, skin_tone...

    def forward(self, x):
        features = self.backbone(x)
        return {
            "face_shape": self.head_shape(features),
            "gender": self.head_gender(features),
            "age": self.head_age(features),
            "ethnicity": self.head_ethnicity(features)
        }

# --- 5. Training Loop ---

class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(CONFIG['device'])
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        
        # Loss Functions
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(CONFIG['device'])
            labels = {k: v.to(CONFIG['device']) for k, v in labels.items()}
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Multi-task Loss Calculation
            loss_shape = self.criterion_ce(outputs['face_shape'], labels['face_shape'])
            loss_gender = self.criterion_ce(outputs['gender'], labels['gender'])
            loss_ethnicity = self.criterion_ce(outputs['ethnicity'], labels['ethnicity'])
            loss_age = self.criterion_mse(outputs['age'].squeeze(), labels['age'].float())
            
            loss = loss_shape + loss_gender + loss_ethnicity + (loss_age * 0.1) # Weight age loss
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        # Metrics storage could go here
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(CONFIG['device'])
                labels = {k: v.to(CONFIG['device']) for k, v in labels.items()}
                
                outputs = self.model(images)
                
                loss_shape = self.criterion_ce(outputs['face_shape'], labels['face_shape'])
                loss_gender = self.criterion_ce(outputs['gender'], labels['gender'])
                loss_ethnicity = self.criterion_ce(outputs['ethnicity'], labels['ethnicity'])
                loss_age = self.criterion_mse(outputs['age'].squeeze(), labels['age'].float())
                
                loss = loss_shape + loss_gender + loss_ethnicity + (loss_age * 0.1)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)

    def run(self):
        best_loss = float('inf')
        patience_counter = 0
        
        Path(CONFIG['dirs']['models']).mkdir(exist_ok=True)
        
        for epoch in range(CONFIG['epochs']):
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            self.scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), f"{CONFIG['dirs']['models']}/best_model.pth")
                patience_counter = 0
                print("Saved Best Model.")
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['patience']:
                    print("Early Stopping.")
                    break

# --- 6. Evaluation & Export ---

class Evaluator:
    def evaluate(self, model, test_loader):
        model.eval()
        # Implement detailed metric calculation (Accuracy, F1, MAE) here
        # and save to results.json
        pass

class Exporter:
    def export_onnx(self, model, path):
        dummy_input = torch.randn(1, 3, 224, 224).to(CONFIG['device'])
        torch.onnx.export(
            model, 
            dummy_input, 
            path, 
            input_names=['input'], 
            output_names=['face_shape', 'gender', 'age', 'ethnicity'],
            opset_version=11
        )
        print(f"Exported to ONNX: {path}")

    def export_tfjs(self, onnx_path, output_dir):
        # This requires 'onnx-tf' and 'tensorflowjs' installed
        # 1. ONNX -> TF SavedModel
        # 2. TF SavedModel -> TFJS
        print("To complete export:")
        print(f"1. onnx-tf convert -i {onnx_path} -o tf_model")
        print(f"2. tensorflowjs_converter --input_format=tf_saved_model tf_model {output_dir}")

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="LooksMaximizer AI Training Pipeline")
    parser.add_argument('--mode', type=str, default='train', choices=['download', 'preprocess', 'train', 'export'])
    args = parser.parse_args()

    if args.mode == 'download':
        dm = DatasetManager(CONFIG['dirs']['data'])
        dm.prepare_all()
        
    elif args.mode == 'preprocess':
        prep = Preprocessor(CONFIG['dirs']['data'], CONFIG['dirs']['processed'])
        prep.run_pipeline()
        
    elif args.mode == 'train':
        # Load Data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        csv_path = Path(CONFIG['dirs']['processed']) / "master_labels.csv"
        if not csv_path.exists():
            print("Labels not found. Run preprocess first.")
            return

        dataset = FaceDataset(csv_path, CONFIG['dirs']['processed'], transform=transform)
        
        # Split
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
        
        # Train
        model = MultiTaskFaceModel()
        trainer = Trainer(model, train_loader, val_loader)
        trainer.run()
        
    elif args.mode == 'export':
        model = MultiTaskFaceModel()
        model.load_state_dict(torch.load(f"{CONFIG['dirs']['models']}/best_model.pth"))
        model.to(CONFIG['device'])
        
        exp = Exporter()
        exp.export_onnx(model, f"{CONFIG['dirs']['models']}/model.onnx")
        exp.export_tfjs(f"{CONFIG['dirs']['models']}/model.onnx", CONFIG['dirs']['web_model'])

if __name__ == "__main__":
    main()


import torch
import torch.nn as nn
from torchvision import models
import os
import shutil

# --- MODEL DEFINITIONS ---

# 1. SCUT (Beauty)
class BeautyScorerModel(nn.Module):
    def __init__(self):
        super(BeautyScorerModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
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

# 2. CelebA (Attributes)
class MultiTaskCelebAModel(nn.Module):
    def __init__(self):
        super(MultiTaskCelebAModel, self).__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=False)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.head_hair = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 7))
        self.head_beard = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 4))
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

# 3. HF (Age/Gender/Race)
class MultiTaskHFModel(nn.Module):
    def __init__(self):
        super(MultiTaskHFModel, self).__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=False)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.head_age = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 9))
        self.head_gender = nn.Linear(num_features, 1)
        self.head_race = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 7))

    def forward(self, x):
        features = self.backbone(x)
        return {
            "age": self.head_age(features),
            "gender": self.head_gender(features),
            "race": self.head_race(features)
        }

# --- EXPORT FUNCTION ---

def export_model(name, model_class, pth_path, onnx_path, input_shape, output_names):
    print(f"\n--- Exporting {name} ---")
    if not os.path.exists(pth_path):
        print(f"❌ .pth file not found: {pth_path}")
        return False

    try:
        device = torch.device("cpu")
        model = model_class()
        state_dict = torch.load(pth_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=output_names,
            opset_version=12,
            do_constant_folding=True,
            export_params=True # Embed weights
        )
        size_mb = os.path.getsize(onnx_path) / (1024*1024)
        print(f"✅ Success! {name} -> {onnx_path} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"❌ Export failed for {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- MAIN ---

if __name__ == "__main__":
    ml_pipeline = r"d:\Looks Maximizer\ml_pipeline"
    
    # SCUT
    export_model(
        "SCUT", 
        BeautyScorerModel, 
        os.path.join(ml_pipeline, "best_model_scut.pth"),
        os.path.join(ml_pipeline, "scut_model.onnx"),
        (256, 256),
        ['score']
    )
    
    # CelebA
    export_model(
        "CelebA", 
        MultiTaskCelebAModel, 
        os.path.join(ml_pipeline, "best_model.pth"),
        os.path.join(ml_pipeline, "celeba_multitask.onnx"),
        (224, 224),
        ['hair', 'beard', 'glasses', 'attractive', 'gender', 'smiling']
    )
    
    # HF
    export_model(
        "HF", 
        MultiTaskHFModel, 
        os.path.join(ml_pipeline, "best_model_hf.pth"),
        os.path.join(ml_pipeline, "hf_model.onnx"),
        (224, 224),
        ['age', 'gender', 'race']
    )
    
    print("\nCheck complete. Proceed to run backend.")

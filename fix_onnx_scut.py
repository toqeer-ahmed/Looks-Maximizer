
import torch
import torch.nn as nn
from torchvision import models
import os

# --- Model Definition (must match train_scut.py) ---
class BeautyScorerModel(nn.Module):
    def __init__(self):
        super(BeautyScorerModel, self).__init__()
        # Backbone: ResNet50
        self.backbone = models.resnet50(pretrained=False) # Do not download pretrained weights, we load ours
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

# --- Export Logic ---
def export():
    pth_path = r"d:\Looks Maximizer\ml_pipeline\best_model_scut.pth"
    onnx_path = r"d:\Looks Maximizer\ml_pipeline\scut_model.onnx"
    
    if not os.path.exists(pth_path):
        print(f"❌ .pth file not found: {pth_path}")
        return

    print(f"Loading weights from {pth_path}...")
    try:
        device = torch.device("cpu")
        model = BeautyScorerModel()
        # Ensure we load nicely even if keys mismatch slightly (though they shouldn't)
        state_dict = torch.load(pth_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 256, 256)
        
        print(f"Exporting to {onnx_path}...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['score'],
            opset_version=12,
            do_constant_folding=True,
            export_params=True # CRITICAL: This embeds weights into the ONNX file
        )
        print("✅ SCUT Export successful!")
        print(f"New size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    export()

import torch
import os
from train_scut import BeautyScorerModel, CONFIG

# 1. Initialize Model
device = torch.device("cpu") # Export on CPU to avoid complications
model = BeautyScorerModel()

# 2. Load the trained weights
# Adjust path if needed
model_path = "models_scut/best_model_scut.pth" 
if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"Error: {model_path} not found. Make sure you trained the model first.")
    exit()

model.to(device)
model.eval()

# 3. Export to ONNX (Self-contained)
print("Exporting to ONNX...")
dummy_input = torch.randn(1, 3, 256, 256).to(device)

# Ensure the model is self-contained (no external data files) if possible
# or zip them together if too large (>2GB, which this is not).
output_file = "scut_model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    output_file,
    input_names=['input'],
    output_names=['score'],
    opset_version=12, # Use a stable opset
    do_constant_folding=True,
    export_params=True
)

print(f"Success! Model exported to {output_file}")
print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

# 4. Zip it (just in case it created external data, though unlikely for this size)
import shutil
zip_filename = "scut_model_export"
shutil.make_archive(zip_filename, 'zip', '.', output_file)
print(f"Created {zip_filename}.zip containing the model.")

# 5. Download instructions
print("\nTo download in Colab:")
print("from google.colab import files")
print(f"files.download('{zip_filename}.zip')")

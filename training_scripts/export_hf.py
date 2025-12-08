import torch
import os
from train_hf import MultiTaskHFModel, CONFIG

# 1. Initialize Model
device = torch.device("cpu")
model = MultiTaskHFModel()

# 2. Load the trained weights
model_path = "models_hf/best_model_hf.pth"
if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"Error: {model_path} not found.")
    exit()

model.to(device)
model.eval()

# 3. Export to ONNX
print("Exporting to ONNX...")
dummy_input = torch.randn(1, 3, 224, 224).to(device)
output_file = "hf_model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    output_file,
    input_names=['input'],
    output_names=['age', 'gender', 'race'],
    opset_version=12,
    do_constant_folding=True,
    export_params=True
)

print(f"Success! Model exported to {output_file}")
print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

# 4. Zip
import shutil
zip_filename = "hf_model_export"
shutil.make_archive(zip_filename, 'zip', '.', output_file)
print(f"Created {zip_filename}.zip")

print("\nTo download in Colab:")
print("from google.colab import files")
print(f"files.download('{zip_filename}.zip')")

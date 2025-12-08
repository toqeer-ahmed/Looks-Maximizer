
import os
import onnxruntime as ort

model_dir = "ml_pipeline"
models = ["celeba_multitask.onnx", "hf_model.onnx", "scut_model.onnx"]

print(f"Checking models in {model_dir}...")

for m in models:
    path = os.path.join(model_dir, m)
    if not os.path.exists(path):
        print(f"❌ {m} NOT FOUND at {path}")
        continue
        
    try:
        sess = ort.InferenceSession(path)
        print(f"✅ {m} loaded successfully!")
    except Exception as e:
        print(f"❌ {m} FAILED to load.")
        print(f"   Error: {e}")

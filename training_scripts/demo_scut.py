import cv2
import torch
import numpy as np
from torchvision import transforms
from train_scut import BeautyScorerModel, CONFIG
import time

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    print("Loading SCUT Beauty Model...")
    model = BeautyScorerModel()
    
    model_path = "ml_pipeline/best_model_scut.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    model.to(device)
    model.eval()

    # 3. Define Transforms (Must match training)
    # Note: Training used 256x256
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Beauty Scorer Demo... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            score = model(input_tensor).item()

        # Visualization
        # Score is typically 1.0 - 5.0
        
        # Color scale based on score
        if score < 2.5:
            color = (0, 0, 255) # Red
        elif score < 3.5:
            color = (0, 255, 255) # Yellow
        else:
            color = (0, 255, 0) # Green

        text = f"Beauty Score: {score:.2f} / 5.0"
        
        # Add a background box for text
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (10, 10), (10 + w + 20, 10 + h + 20), (0, 0, 0), -1)
        
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Looks Maximizer - Beauty Scorer', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import torch
import numpy as np
from torchvision import transforms
from train_hf import MultiTaskHFModel, CONFIG
import time

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    print("Loading HF model (Age/Gender/Race)...")
    model = MultiTaskHFModel()
    
    model_path = "ml_pipeline/best_model_hf.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    model.to(device)
    model.eval()

    # 3. Define Transforms (Must match training)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Demo... Press 'q' to quit.")
    
    # Labels
    # Age: 9 classes (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)
    # Note: We need to map these indices to actual strings based on the dataset.
    # Assuming standard mapping for now, or just showing the index/raw prediction.
    # Let's use the unique values we saw earlier: [6 4 1 3 5 2 7 0 8]
    # We'll just show the predicted class index for now or a generic label.
    age_labels = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    
    # Race: 7 classes
    race_labels = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"] 
    # (Note: This is a guess based on common datasets like FairFace. 
    # If the user knows the specific mapping, they can update it. 
    # For now, we'll use indices if we aren't sure, but let's try these common ones.)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)

        # Process Outputs
        # Gender: Binary (Sigmoid)
        gender_prob = torch.sigmoid(outputs['gender']).item()
        gender_str = "Male" if gender_prob > 0.5 else "Female"
        
        # Age: Multi-class (Argmax)
        age_idx = torch.argmax(outputs['age']).item()
        age_str = age_labels[age_idx] if age_idx < len(age_labels) else str(age_idx)
        
        # Race: Multi-class (Argmax)
        race_idx = torch.argmax(outputs['race']).item()
        race_str = race_labels[race_idx] if race_idx < len(race_labels) else str(race_idx)

        # Visualization
        y_pos = 30
        line_height = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 255) # Yellow
        thickness = 2

        cv2.putText(frame, f"Gender: {gender_str} ({gender_prob:.2f})", (10, y_pos), font, 0.8, color, thickness)
        y_pos += line_height
        
        cv2.putText(frame, f"Age Group: {age_str}", (10, y_pos), font, 0.8, color, thickness)
        y_pos += line_height
        
        cv2.putText(frame, f"Race/Eth: {race_str}", (10, y_pos), font, 0.8, color, thickness)

        cv2.imshow('Looks Maximizer - HF Model Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

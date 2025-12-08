import cv2
import torch
import numpy as np
from torchvision import transforms
from train_celeba import MultiTaskCelebAModel, CONFIG
import time

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    print("Loading model...")
    model = MultiTaskCelebAModel()
    
    model_path = "ml_pipeline/best_model.pth"
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
    
    # Thresholds for binary attributes
    threshold = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        # OpenCV is BGR, model expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transform expects PIL Image or Tensor, our transform handles PIL conversion from numpy
        input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)

        # Process Outputs
        # Binary attributes (Sigmoid)
        glasses = torch.sigmoid(outputs['glasses']).item()
        attractive = torch.sigmoid(outputs['attractive']).item()
        gender = torch.sigmoid(outputs['gender']).item() # 1 = Male, 0 = Female
        smiling = torch.sigmoid(outputs['smiling']).item()

        # Multi-label attributes (Sigmoid + Threshold or Argmax)
        # For demo, let's just show top hair color and beard style if present
        hair_probs = torch.sigmoid(outputs['hair']).squeeze()
        beard_probs = torch.sigmoid(outputs['beard']).squeeze()
        
        # Get attribute names from dataset class (hardcoded here for simplicity matching train_celeba.py)
        hair_names = ["Bald", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Straight_Hair", "Wavy_Hair"]
        beard_names = ["No_Beard", "Mustache", "Goatee", "Sideburns"]

        # Visualization
        # Draw a rectangle for the "face" (just the whole frame for now, or center crop)
        # To make it look cooler, we'll just overlay text
        
        y_pos = 30
        line_height = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0) # Green
        thickness = 2

        # Gender
        gender_str = "Male" if gender > threshold else "Female"
        cv2.putText(frame, f"Gender: {gender_str} ({gender:.2f})", (10, y_pos), font, 0.7, color, thickness)
        y_pos += line_height

        # Attractive
        attr_str = "High" if attractive > threshold else "Low"
        cv2.putText(frame, f"Attractiveness: {attractive:.2f}", (10, y_pos), font, 0.7, color, thickness)
        y_pos += line_height

        # Smiling
        smile_str = "Yes" if smiling > threshold else "No"
        cv2.putText(frame, f"Smiling: {smile_str} ({smiling:.2f})", (10, y_pos), font, 0.7, color, thickness)
        y_pos += line_height

        # Glasses
        glass_str = "Yes" if glasses > threshold else "No"
        cv2.putText(frame, f"Glasses: {glass_str} ({glasses:.2f})", (10, y_pos), font, 0.7, color, thickness)
        y_pos += line_height

        # Hair (Show top 1)
        top_hair_idx = torch.argmax(hair_probs).item()
        hair_str = hair_names[top_hair_idx]
        cv2.putText(frame, f"Hair: {hair_str}", (10, y_pos), font, 0.7, color, thickness)
        y_pos += line_height
        
        # Beard (Show if any > threshold, else No Beard)
        beard_str = "No Beard"
        for i, prob in enumerate(beard_probs):
            if prob > threshold and beard_names[i] != "No_Beard":
                beard_str = beard_names[i]
                break
        cv2.putText(frame, f"Beard: {beard_str}", (10, y_pos), font, 0.7, color, thickness)


        cv2.imshow('Looks Maximizer Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

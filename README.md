# Looks Maximizer

Looks Maximizer is an AI-powered application that analyzes facial features to provide personalized recommendations for hairstyles, grooming, and fashion. It uses deep learning models to assess attributes like face shape, age, gender, and beauty score.

## Project Structure

*   **`backend/`**: Flask API that handles image processing and AI inference.
*   **`looks-maximizer/`**: React + Vite frontend application.
*   **`ml_pipeline/`**: Contains the trained ONNX models and inference logic.
*   **`training_scripts/`**: Python scripts and guides used to train the models (e.g., on Google Colab).

## Prerequisites

*   **Node.js** (v16 or higher)
*   **Python** (v3.8 or higher)

## Setup & Running

### 1. Backend Setup

1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Install Python dependencies:
    ```bash
    pip install flask flask-cors opencv-python numpy onnxruntime google-cloud-firestore google-auth
    ```
3.  Run the Flask server:
    ```bash
    python app.py
    ```
    The backend will start on `http://127.0.0.1:5000`.

### 2. Frontend Setup

1.  Open a new terminal and navigate to the frontend directory:
    ```bash
    cd looks-maximizer
    ```
2.  Install Node dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
    The app will open at `http://localhost:5173`.

## Usage

1.  Open the web app.
2.  **Sign Up** to create a profile with your details (Age, Height, etc.).
3.  **Upload a Photo** or use the **Live Camera**.
4.  Click **Analyze Now** to get your personalized report.

## Models

The application uses the following ONNX models in `ml_pipeline/`:
*   `hf_model.onnx`: Predicts Age, Gender, and Race.
*   `scut_model.onnx`: Predicts Beauty Score.
*   `celeba_multitask.onnx`: Predicts attributes like Glasses, Smiling, etc.

## Training

If you want to retrain the models, refer to the guides in `training_scripts/`:
*   `COLAB_HF_GUIDE.md`: Guide for Age/Gender/Race model.
*   `COLAB_SCUT_GUIDE.md`: Guide for Beauty Score model.

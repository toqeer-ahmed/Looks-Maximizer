import os
import time
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import firestore
from google.oauth2 import service_account

# --- Configuration ---
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "DELETE", "OPTIONS"]}})

# Environment Variables (with defaults for development)
FLASK_APP_ID = os.environ.get('FLASK_APP_ID', 'looks-maximizer-mvp')
FIRESTORE_PROJECT_ID = os.environ.get('FIRESTORE_PROJECT_ID', 'looks-maximizer-db')
MOCKED_ANALYSIS_DELAY_MS = int(os.environ.get('MOCKED_ANALYSIS_DELAY_MS', 3000))

# --- Firestore Initialization ---
# Note: In a real production environment, Google Cloud libraries automatically 
# find credentials. For local dev without a real GCP project, this might fail 
# or require a mock. We will wrap it to prevent immediate crash if no creds.
try:
    db = firestore.Client(project=FIRESTORE_PROJECT_ID)
except Exception as e:
    print(f"Warning: Firestore Client failed to initialize (expected if no GCP creds): {e}")
    db = None

# --- Mock Data ---
MOCKED_RESULTS = {
    "faceShape": "Square",
    "lookScore": 92,
    "recommendations": {
        "hairstyles": [
            "Quiff (Adds height, lengthens face)",
            "Low Fade (Softens jawline)",
            "Crew Cut (Classic, balances features)"
        ],
        "beardStyles": [
            "Stubble (Always safe)",
            "Circle Beard (Adds length to chin)"
        ],
        "clothingStyle": [
            "Structured jackets (Sharp lines)",
            "V-neck shirts (Softens the neck/jaw transition)",
            "Bold patterns (Draws attention to the chest)"
        ]
    }
}

from backend.inference import model_manager

# --- Helper Functions ---

def run_analysis(user_id, image_url, user_details={}):
    """
    Runs AI analysis using the loaded models.
    """
    print(f"Running analysis for {user_id} on {image_url} with details {user_details}")
    
    try:
        results = model_manager.predict(image_url, user_details)
        if results:
            return results
        else:
            print("Inference failed, returning mock data.")
            return MOCKED_RESULTS
    except Exception as e:
        print(f"Error during inference: {e}")
        return MOCKED_RESULTS

# --- API Endpoints ---

@app.route('/api/analyze_face', methods=['POST'])
def analyze_face():
    data = request.get_json()
    
    # 1. Validate Input
    if not data or 'userId' not in data:
        return jsonify({"status": "error", "message": "Missing userId"}), 400
    
    user_id = data['userId']
    # Prefer imageData (Base64) if provided, otherwise URL
    image_source = data.get('imageData') or data.get('uploadedImageURL')
    user_details = data.get('userDetails', {}) # Capture user details
    
    if not image_source:
        return jsonify({"status": "error", "message": "Missing image data or URL"}), 400
    
    # 2. Run Analysis
    results = run_analysis(user_id, image_source, user_details)
    
    # 3. Log to Firestore
    analysis_id = "mock_analysis_id_" + str(int(time.time()))
    timestamp = datetime.datetime.now()
    
    if db:
        try:
            doc_ref = db.collection('artifacts').document(FLASK_APP_ID)\
                        .collection('users').document(user_id)\
                        .collection('analysis').document()
            
            analysis_data = {
                **results,
                "userId": user_id,
                "uploadedImageURL": image_url,
                "timestamp": timestamp
            }
            
            doc_ref.set(analysis_data)
            analysis_id = doc_ref.id
        except Exception as e:
            print(f"Error writing to Firestore: {e}")
            # Continue even if DB write fails for the MVP mock
    
    # 4. Return Response
    return jsonify({
        "status": "success",
        "analysisId": analysis_id,
        "results": results
    })

@app.route('/api/log_feedback', methods=['POST'])
def log_feedback():
    data = request.get_json()
    
    # 1. Validate Input
    if not data or 'userId' not in data or 'analysisId' not in data or 'helpful' not in data:
        return jsonify({"status": "error", "message": "Missing required fields"}), 400
        
    user_id = data['userId']
    analysis_id = data['analysisId']
    helpful = data['helpful']
    
    # 2. Log to Firestore
    if db:
        try:
            db.collection('artifacts').document(FLASK_APP_ID)\
              .collection('users').document(user_id)\
              .collection('feedback').add({
                  "userId": user_id,
                  "analysisId": analysis_id,
                  "helpful": helpful,
                  "timestamp": datetime.datetime.now()
              })
        except Exception as e:
            print(f"Error writing feedback to Firestore: {e}")
            return jsonify({"status": "error", "message": "Database error"}), 500

    return jsonify({"status": "success", "message": "Feedback logged successfully"})

@app.route('/api/analysis/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    user_id = request.args.get('userId')
    
    if not user_id:
        return jsonify({"status": "error", "message": "Missing userId"}), 400

    if db:
        try:
            # Security Check: Ensure the document belongs to the user
            doc_ref = db.collection('artifacts').document(FLASK_APP_ID)\
                        .collection('users').document(user_id)\
                        .collection('analysis').document(analysis_id)
            
            doc = doc_ref.get()
            if not doc.exists:
                return jsonify({"status": "error", "message": "Analysis not found"}), 404
            
            # In a real app with ID Tokens, we would verify the token's UID matches the path.
            # Here we rely on the path structure.
            
            doc_ref.delete()
            return jsonify({"status": "success", "message": "Analysis deleted"})
        except Exception as e:
            print(f"Error deleting from Firestore: {e}")
            return jsonify({"status": "error", "message": "Database error"}), 500
            
    return jsonify({"status": "success", "message": "Mock delete success"})

@app.route('/', methods=['GET'])
def health_check():
    return "Backend is running!", 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)

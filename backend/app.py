import os
import time
import datetime
import json
import hashlib
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from google.cloud import firestore
from google.oauth2 import service_account

# --- Configuration ---
app = Flask(__name__)

# Enable CORS for all routes - allow all origins
CORS(app, 
     origins=["*"], 
     methods=["GET", "POST", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=False)

# Additional CORS headers for all responses
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

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

# --- Local Repositories ---
class UserRepository:
    def __init__(self, filename="local_users.json"):
        self.filename = filename
        self.users = self._load()

    def _load(self):
        if not os.path.exists(self.filename):
            return {}
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.users, f, indent=2)

    def create_user(self, email, password, details):
        if email in self.users:
            return False, "User already exists"
        
        # Simple hash for this MVP
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        
        user_id = "user_" + str(int(time.time())) + "_" + str(hash(email))
        self.users[email] = {
            "uid": user_id,
            "email": email,
            "password": pwd_hash,
            "details": details,
            "details": details,
            "plan": "free", # Default plan
            "created_at": str(datetime.datetime.now())
        }
        self._save()
        return True, self.users[email]

    def verify_user(self, email, password):
        if email not in self.users:
            return False, "User not found"
        
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        if self.users[email]["password"] == pwd_hash:
            return True, self.users[email]
        else:
            return False, "Invalid password"

class AnalysisRepository:
    def __init__(self, filename="local_data_analysis.json"):
        self.filename = filename

    def get_user_history(self, user_id):
        if not os.path.exists(self.filename):
            return []
        try:
            with open(self.filename, 'r') as f:
                all_data = json.load(f)
                # Filter by correct user_id
                user_data = [d for d in all_data if d.get('userId') == user_id]
                # Sort by timestamp desc
                user_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                return user_data
        except:
            return []

class SubscriptionManager:
    def __init__(self, usage_file="local_usage.json"):
        self.usage_file = usage_file
        self.usage = self._load()

    def _load(self):
        if not os.path.exists(self.usage_file):
            return {}
        try:
            with open(self.usage_file, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save(self):
        with open(self.usage_file, 'w') as f:
            json.dump(self.usage, f, indent=2)

    def check_limit(self, user_id, plan):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Initialize usage record if needed
        if user_id not in self.usage:
            self.usage[user_id] = {}
        
        if today not in self.usage[user_id]:
            self.usage[user_id][today] = 0
            
        current_usage = self.usage[user_id][today]
        
        # LIMITS
        if plan == 'free':
            limit = 1
        else:
            limit = 9999 # Unlimited for Pro/Elite

        if current_usage >= limit:
            return False, f"Daily limit reached for {plan} plan."
            
        return True, "Allowed"

    def increment_usage(self, user_id):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        if user_id in self.usage and today in self.usage[user_id]:
            self.usage[user_id][today] += 1
            self._save()

user_repo = UserRepository()
history_repo = AnalysisRepository()
sub_manager = SubscriptionManager()

# --- Mock Data --- 
# (Kept only for fallback if inference fails completely, but not for auth)
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

def log_to_local_file(data, log_type="analysis"):
    """
    Logs data to a local JSON file when Firestore is unavailable.
    """
    try:
        filename = f"local_data_{log_type}.json"
        
        # Ensure timestamp is serializable
        data_to_log = data.copy()
        if 'timestamp' in data_to_log and isinstance(data_to_log['timestamp'], datetime.datetime):
             data_to_log['timestamp'] = data_to_log['timestamp'].isoformat()
             
        # Read existing
        existing_data = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        existing_data.append(data_to_log)
        
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        print(f"Logged {log_type} to local file: {filename}")
    except Exception as e:
        print(f"Failed to log to local file: {e}")

# --- API Endpoints ---

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    details = data.get('details', {})

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password required"}), 400

    success, result = user_repo.create_user(email, password, details)
    if success:
        # Don't return password hash
        user_safe = result.copy()
        del user_safe['password']
        return jsonify({"status": "success", "user": user_safe})
    else:
        return jsonify({"status": "error", "message": result}), 400

@app.route('/api/auth/profile', methods=['GET'])
def get_profile():
    user_id = request.args.get('userId')
    if not user_id: return jsonify({'error': 'Missing userId'}), 400
    
    # Simple lookup in local repo
    # In real app, verify token
    for email, u in user_repo.users.items():
        if u['uid'] == user_id:
            user_safe = u.copy()
            del user_safe['password']
            return jsonify({'status': 'success', 'user': user_safe})
            
    return jsonify({'error': 'User not found'}), 404

@app.route('/api/subscription/upgrade', methods=['POST'])
def upgrade_plan():
    data = request.get_json()
    user_id = data.get('userId')
    new_plan = data.get('plan') # 'pro' or 'elite'
    
    if not user_id or new_plan not in ['pro', 'elite']:
        return jsonify({"status": "error", "message": "Invalid request"}), 400

    # Find and update user in local repo
    found = False
    for email, u in user_repo.users.items():
        if u['uid'] == user_id:
            u['plan'] = new_plan
            user_repo._save()
            found = True
            break
            
    if found:
        return jsonify({"status": "success", "message": f"Upgraded to {new_plan}"})
    else:
        return jsonify({"status": "error", "message": "User not found"}), 404

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password required"}), 400

    success, result = user_repo.verify_user(email, password)
    if success:
        user_safe = result.copy()
        del user_safe['password']
        return jsonify({"status": "success", "user": user_safe})
    else:
        return jsonify({"status": "error", "message": result}), 401

@app.route('/api/history', methods=['GET'])
def get_history():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({"status": "error", "message": "Missing userId"}), 400
    
    history = history_repo.get_user_history(user_id)
    return jsonify({"status": "success", "history": history})


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
    
    # 0. Determine Plan
    # If no userId provided, treat as "guest"
    # Guest shares "free" limits but we need a guest ID to track usage. 
    # For now, let's assume specific guest ID passed or generate one.
    
    current_plan = 'free'
    usage_id = user_id
    
    # Lookup real plan if user exists
    if user_id.startswith("user_"):
         # Find user in repo to get plan
         for email, u in user_repo.users.items():
             if u['uid'] == user_id:
                 current_plan = u.get('plan', 'free')
                 break

    # 1. Check Limits
    allowed, msg = sub_manager.check_limit(usage_id, current_plan)
    if not allowed:
        return jsonify({
            "status": "error", 
            "code": "LIMIT_REACHED",
            "message": "You have reached your daily limit for the Free plan. Upgrade to LooksMax Pro for unlimited analysis."
        }), 403

    # 2. Run Analysis
    # We pass 'mask_advanced' flag to inference or handle it here.
    # We will run full inference but remove keys before returning if free.
    full_results = run_analysis(user_id, image_source, user_details)
    
    # 3. Gate Content (Masking)
    # Check if this is the user's VERY FIRST analysis
    user_history = history_repo.get_user_history(user_id)
    is_first_time = len(user_history) == 0

    if current_plan == 'free' and not is_first_time:
        # Blur/Hide Advanced Metrics
        visible_results = {
            "lookScore": full_results.get("lookScore"),
            "faceShape": full_results.get("faceShape"),
            "age_group": full_results.get("age_group"),
            "gender_hf": full_results.get("gender_hf"),
            "recommendations": {
                 # Limit to 1 item
                 "hairstyles": full_results.get("recommendations", {}).get("hairstyles", [])[:1],
                 "beardStyles": full_results.get("recommendations", {}).get("beardStyles", [])[:1],
                 "clothingStyle": full_results.get("recommendations", {}).get("clothingStyle", [])[:1],
            },
            "is_premium": False,
            "preview_only": True
        }
        # Add "Locked" placeholders for advanced features
        visible_results["symmetry_analysis"] = "LOCKED"
        visible_results["skin_quality"] = "LOCKED"
        visible_results["evolution_tracker"] = "LOCKED"
    else:
        # Full Access (Pro/Elite OR First Time Trial)
        visible_results = full_results
        visible_results["is_premium"] = True
        
        # If it's a trial, maybe add a flag to UI?
        if is_first_time and current_plan == 'free':
             visible_results["is_trial"] = True
             
        # Mock Advanced Features for Pro
        visible_results["symmetry_analysis"] = "Excellent Symmetry (94%)"
        visible_results["skin_quality"] = "Clear, slight uneven tone"
        if current_plan == 'elite':
             visible_results["evolution_tracker"] = "Available"
        
    sub_manager.increment_usage(usage_id)

    # 3. Log to Firestore (or Local File)
    analysis_id = "mock_analysis_id_" + str(int(time.time()))
    timestamp = datetime.datetime.now()
    
    # Prepare data for logging
    analysis_data = {
        **full_results, # We save FULL results even if user can't see them yet
        "userId": user_id,
        "uploadedImageURL": image_source if len(str(image_source)) < 500 else "base64_image",
        "timestamp": timestamp,
        "userDetails": user_details
    }

    if db:
        try:
            doc_ref = db.collection('artifacts').document(FLASK_APP_ID)\
                        .collection('users').document(user_id)\
                        .collection('analysis').document()
            
            doc_ref.set(analysis_data)
            analysis_id = doc_ref.id
        except Exception as e:
            print(f"Error writing to Firestore: {e}")
            log_to_local_file(analysis_data, "analysis")
    else:
        log_to_local_file(analysis_data, "analysis")
    
    # 4. Return Response
    return jsonify({
        "status": "success",
        "analysisId": analysis_id,
        "results": visible_results
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
            log_to_local_file(data, "feedback")
            return jsonify({"status": "error", "message": "Database error"}), 500
    else:
        log_to_local_file(data, "feedback")

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

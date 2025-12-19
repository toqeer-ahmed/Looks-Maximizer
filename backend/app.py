import os
import time
import datetime
import json
import hashlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB

# --- Configuration ---
app = Flask(__name__)

# Enable CORS
CORS(app, 
     origins=["*"], 
     methods=["GET", "POST", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Database
# Use SQLite for local development if PostgreSQL URL is not provided
# Railway provides DATABASE_URL env var
db_url = os.environ.get('DATABASE_URL')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url or 'sqlite:///local_mvp.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Parameters ---
MOCKED_ANALYSIS_DELAY_MS = int(os.environ.get('MOCKED_ANALYSIS_DELAY_MS', 3000))

# --- Database Models ---

class User(db.Model):
    id = db.Column(db.String(128), primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    plan = db.Column(db.String(20), default='free')
    details = db.Column(db.JSON) # Stores age, gender, name
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # Usage tracking
    last_usage_date = db.Column(db.String(20)) # YYYY-MM-DD
    usage_count = db.Column(db.Integer, default=0)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(128), db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    image_url = db.Column(db.String(500))
    # Store full JSON results
    results = db.Column(db.JSON)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(128))
    analysis_id = db.Column(db.Integer)
    helpful = db.Column(db.Boolean)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Initialize DB (Create Tables)
with app.app_context():
    db.create_all()

# --- Repositories (Adapters) ---
# We simulate the old repository interface but use SQLAlchemy now

class UserRepository:
    def create_user(self, email, password, details):
        existing = User.query.filter_by(email=email).first()
        if existing:
            return False, "User already exists"
            
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        user_id = "user_" + str(int(time.time())) + "_" + str(hash(email))
        
        new_user = User(
            id=user_id,
            email=email,
            password_hash=pwd_hash,
            details=details,
            plan='free',
            usage_count=0
        )
        db.session.add(new_user)
        db.session.commit()
        return True, self._to_dict(new_user)

    def verify_user(self, email, password):
        user = User.query.filter_by(email=email).first()
        if not user:
            return False, "User not found"
            
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        if user.password_hash == pwd_hash:
            return True, self._to_dict(user)
        return False, "Invalid password"

    def update_plan(self, user_id, new_plan):
        user = User.query.get(user_id)
        if user:
            user.plan = new_plan
            db.session.commit()
            return True
        return False
        
    def get_user(self, user_id):
        user = User.query.get(user_id)
        if user:
            return self._to_dict(user)
        return None

    def _to_dict(self, user):
        return {
            "uid": user.id,
            "email": user.email,
            "details": user.details or {},
            "plan": user.plan,
            "created_at": str(user.created_at)
        }

class AnalysisRepository:
    def add_analysis(self, user_id, image_url, results):
        new_analysis = Analysis(
            user_id=user_id,
            image_url=str(image_url),
            results=results
        )
        db.session.add(new_analysis)
        db.session.commit()
        return new_analysis.id

    def get_user_history(self, user_id):
        analyses = Analysis.query.filter_by(user_id=user_id).order_by(Analysis.timestamp.desc()).all()
        history = []
        for a in analyses:
            item = a.results.copy()
            item["id"] = a.id
            item["timestamp"] = {"seconds": a.timestamp.timestamp()} # Match Firestore format roughly
            history.append(item)
        return history

    def delete_analysis(self, analysis_id, user_id):
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=user_id).first()
        if analysis:
            db.session.delete(analysis)
            db.session.commit()
            return True
        return False

class SubscriptionManager:
    def check_limit(self, user_id, plan):
        if not user_id: return True, "Guest"
        
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        user = User.query.get(user_id)
        
        if not user: return True, "Unknown User" # Should not happen ideally
        
        # Reset counter if new day
        if user.last_usage_date != today:
            user.last_usage_date = today
            user.usage_count = 0
            db.session.commit()
            
        # Check Limits
        limit = 1 if plan == 'free' else 9999
        
        if user.usage_count >= limit:
            return False, f"Daily limit reached for {plan} plan."
            
        return True, "Allowed"

    def increment_usage(self, user_id):
        user = User.query.get(user_id)
        if user:
            user.usage_count += 1
            db.session.commit()

user_repo = UserRepository()
history_repo = AnalysisRepository()
sub_manager = SubscriptionManager()

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
        return jsonify({"status": "success", "user": result})
    else:
        return jsonify({"status": "error", "message": result}), 400

@app.route('/api/auth/profile', methods=['GET'])
def get_profile():
    user_id = request.args.get('userId')
    if not user_id: return jsonify({'error': 'Missing userId'}), 400
    
    user = user_repo.get_user(user_id)
    if user:
         return jsonify({'status': 'success', 'user': user})
    return jsonify({'error': 'User not found'}), 404

@app.route('/api/subscription/upgrade', methods=['POST'])
def upgrade_plan():
    data = request.get_json()
    user_id = data.get('userId')
    new_plan = data.get('plan')
    
    if not user_id or new_plan not in ['pro', 'elite']:
        return jsonify({"status": "error", "message": "Invalid request"}), 400

    if user_repo.update_plan(user_id, new_plan):
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
        return jsonify({"status": "success", "user": result})
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
    
    if not data or 'userId' not in data:
        return jsonify({"status": "error", "message": "Missing userId"}), 400
    
    user_id = data['userId']
    image_source = data.get('imageData') or data.get('uploadedImageURL')
    user_details = data.get('userDetails', {})
    
    if not image_source:
        return jsonify({"status": "error", "message": "Missing image data"}), 400
    
    # 0. Get Plan
    user_data = user_repo.get_user(user_id)
    current_plan = user_data['plan'] if user_data else 'free'
    
    # 1. Check Limits
    allowed, msg = sub_manager.check_limit(user_id, current_plan)
    if not allowed:
        return jsonify({
            "status": "error", 
            "code": "LIMIT_REACHED",
            "message": "Limit Reached"
        }), 403

    # 2. Run Analysis
    full_results = run_analysis(user_id, image_source, user_details)
    
    # 3. Save to DB
    analysis_id = history_repo.add_analysis(user_id, image_source, full_results)
    
    # 4. Gate Content (Masking)
    # Check history count (excluding current one just added, is flawed logic if strict, 
    # but querying history now includes the one just added.
    # Trial Logic: If this was their first one (count == 1 now), give them access? 
    # Or query count before adding?
    # Let's simplify: check total count. If 1, it's the trial.
    user_history = history_repo.get_user_history(user_id)
    is_first_time = len(user_history) <= 1

    if current_plan == 'free' and not is_first_time:
        visible_results = {
            "lookScore": full_results.get("lookScore"),
            "faceShape": full_results.get("faceShape"),
            "age_group": full_results.get("age_group"),
            "gender": full_results.get("gender"), 
            "recommendations": {
                 "hairstyles": full_results.get("recommendations", {}).get("hairstyles", [])[:1],
                 "beardStyles": full_results.get("recommendations", {}).get("beardStyles", [])[:1],
                 "clothingStyle": full_results.get("recommendations", {}).get("clothingStyle", [])[:1],
            },
            "is_premium": False,
            "preview_only": True
        }
        visible_results["symmetry_analysis"] = "LOCKED"
        visible_results["skin_quality"] = "LOCKED"
        visible_results["evolution_tracker"] = "LOCKED"
    else:
        visible_results = full_results
        visible_results["is_premium"] = True
        
        if is_first_time and current_plan == 'free':
             visible_results["is_trial"] = True
             
        visible_results["symmetry_analysis"] = "Excellent Symmetry (94%)"
        visible_results["skin_quality"] = "Clear, slight uneven tone"
        if current_plan == 'elite':
             visible_results["evolution_tracker"] = "Available"
        
    sub_manager.increment_usage(user_id)
    
    return jsonify({
        "status": "success",
        "analysisId": analysis_id,
        "results": visible_results
    })

@app.route('/api/log_feedback', methods=['POST'])
def log_feedback():
    data = request.get_json()
    new_feedback = Feedback(
        user_id=data.get('user_id'),
        analysis_id=data.get('analysis_id'),
        helpful=data.get('helpful')
    )
    db.session.add(new_feedback)
    db.session.commit()
    return jsonify({"status": "success", "message": "Feedback logged"})

@app.route('/api/analysis/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    user_id = request.args.get('userId')
    if history_repo.delete_analysis(analysis_id, user_id):
         return jsonify({"status": "success", "message": "Analysis deleted"})
    return jsonify({"status": "error", "message": "Not found or forbidden"}), 404

@app.route('/', methods=['GET'])
def health_check():
    return "Backend is running!", 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)

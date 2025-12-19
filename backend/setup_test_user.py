import hashlib
import time
import datetime
import os
import sys

# Add the parent directory to sys.path so we can import backend.app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import app, db, User

EMAIL = "test@looksmax.com"
PASSWORD = "password123"

def create_or_update_admin():
    with app.app_context():
        # Ensure tables exist
        db.create_all()
        
        user = User.query.filter_by(email=EMAIL).first()
        pwd_hash = hashlib.sha256(PASSWORD.encode()).hexdigest()
        
        if user:
            print(f"Updating existing user {EMAIL} to Elite plan.")
            user.password_hash = pwd_hash
            user.plan = 'elite'
        else:
            print(f"Creating new test user {EMAIL} with Elite plan.")
            user_id = "user_" + str(int(time.time())) + "_test"
            new_user = User(
                id=user_id,
                email=EMAIL,
                password_hash=pwd_hash,
                details={"name": "Test Admin", "age": 25, "gender": "Male"},
                plan='elite',
                usage_count=0
            )
            db.session.add(new_user)
            
        try:
            db.session.commit()
            print("Success! User configured.")
        except Exception as e:
            print(f"Error saving user: {e}")
            db.session.rollback()

if __name__ == "__main__":
    create_or_update_admin()

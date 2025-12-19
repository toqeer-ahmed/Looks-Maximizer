import json
import hashlib
import time
import datetime
import os

FILENAME = "local_users.json"
EMAIL = "test@looksmax.com"
PASSWORD = "password123"

def create_or_update_admin():
    users = {}
    if os.path.exists(FILENAME):
        try:
            with open(FILENAME, 'r') as f:
                users = json.load(f)
        except:
            pass
            
    # Hash password
    pwd_hash = hashlib.sha256(PASSWORD.encode()).hexdigest()
    
    if EMAIL in users:
        print(f"Updating existing user {EMAIL} to Elite plan.")
        users[EMAIL]['password'] = pwd_hash
        users[EMAIL]['plan'] = 'elite'
    else:
        print(f"Creating new test user {EMAIL} with Elite plan.")
        user_id = "user_" + str(int(time.time())) + "_test"
        users[EMAIL] = {
            "uid": user_id,
            "email": EMAIL,
            "password": pwd_hash,
            "details": {
                "name": "Test Admin",
                "age": 25,
                "gender": "Male"
            },
            "plan": "elite",
            "created_at": str(datetime.datetime.now())
        }
        
    with open(FILENAME, 'w') as f:
        json.dump(users, f, indent=2)
        
    print("Success. Restart the backend to ensure it reloads the file.")

if __name__ == "__main__":
    create_or_update_admin()

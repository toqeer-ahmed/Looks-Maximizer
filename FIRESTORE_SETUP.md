# Getting Firestore Credentials for Railway

## Option A: If you have a Google Cloud Project (Recommended)

1. Go to https://console.cloud.google.com
2. Select your project
3. Go to **IAM & Admin → Service Accounts**
4. Click **Create Service Account**
5. Name: `looks-maximizer-railway`
6. Grant role: **Editor** (for simplicity) or **Firebase Admin**
7. Click **Create and Continue**
8. Go to **Keys** tab → **Add Key** → **Create new key**
9. Choose **JSON** → Download `serviceAccountKey.json`
10. Open the JSON file and copy the entire content
11. In Railway Variables, paste it as:
    ```
    GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
    ```
12. Create another variable:
    ```
    FIRESTORE_CREDS_JSON=<paste entire JSON content here>
    ```

## Option B: Using Firestore Emulator (Testing)

If you don't have GCP, Railway will use mock data (already handled in your app.py).
Skip Firestore setup and just deploy - it will work in demo mode.

## Step 4: Update app.py to Load Credentials

The updated app.py needs to decode credentials from the environment variable.
Add this code at the top of app.py (after imports):

```python
import json
import base64

# Decode Firestore credentials from Railway environment
firestore_creds_json = os.environ.get('FIRESTORE_CREDS_JSON')
if firestore_creds_json:
    try:
        creds_dict = json.loads(firestore_creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        db = firestore.Client(project=creds_dict['project_id'], credentials=credentials)
    except Exception as e:
        print(f"Firestore init error: {e}")
        db = None
```

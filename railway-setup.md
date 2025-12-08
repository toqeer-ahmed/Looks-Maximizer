# Looks Maximizer - Railway Deployment Guide

## Quick Start

### 1. **Create Railway Account**
   - Go to https://railway.app
   - Sign up with GitHub
   - Authorize Railway to access your repositories

### 2. **Deploy Your Project**
   - Click "New Project" 
   - Select "Deploy from GitHub repo"
   - Choose your `Looks-Maximizer` repository
   - Select `main` branch
   - Railway auto-detects Dockerfile and builds

### 3. **Set Environment Variables**
   Go to your Railway project → Variables tab and add:

   ```
   PORT=8000
   FLASK_APP_ID=looks-maximizer-mvp
   FIRESTORE_PROJECT_ID=your-gcp-project-id
   GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
   ```

   **For Firestore Credentials:**
   - Go to Google Cloud Console → Service Accounts
   - Create new service account with Firestore permissions
   - Download JSON key as `serviceAccountKey.json`
   - Base64 encode it: `cat serviceAccountKey.json | base64`
   - Add as Railway variable: `FIRESTORE_CREDENTIALS_B64=<base64-string>`

### 4. **Copy Credentials at Runtime**
   Railroad automatically handles this - add to your start command in `railway.json`

### 5. **Connect Frontend (Optional)**
   For full-stack hosting:
   - Deploy frontend separately on Vercel or Netlify
   - Update frontend API URL to: `https://your-railway-url.railway.app/api`

### 6. **Monitor Deployment**
   - Railway Dashboard shows logs in real-time
   - Check health: `https://your-railway-url.railway.app/`
   - View API: `https://your-railway-url.railway.app/api/analyze_face`

## Important Notes

- **Free Tier**: 500 hours/month (enough for always-on service)
- **Auto-sleep**: App sleeps after inactivity
- **Model Size**: Ensure ONNX models fit (Railway: ~5GB limit for build)
- **Cold Start**: First request takes 10-30 seconds after sleep
- **Logs**: Check deployment logs for errors

## Troubleshooting

**Build fails**: Check that `ml_pipeline/` models are in git LFS or reduce size
**Port issues**: Railway uses dynamic PORT env var - already handled in `railway.json`
**Firestore errors**: Verify credentials are properly base64 encoded

## Next Steps
1. Push these files to GitHub
2. Go to Railway and create new project
3. Watch the build complete
4. Test your API endpoints

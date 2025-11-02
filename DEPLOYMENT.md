# Deployment Guide - Streamlit Cloud

## Prerequisites

1. GitHub account
2. Streamlit Cloud account (free) - Sign up at https://streamlit.io/cloud

## Deployment Steps

### Step 1: Push Code to GitHub

1. **Initialize Git repository** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Student Performance Prediction Project"
   ```

2. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Create a new repository (e.g., `student-performance-prediction`)
   - **Don't** initialize with README, .gitignore, or license

3. **Push your code:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Click "New app"**

3. **Fill in the details:**
   - **Repository:** Select your GitHub repository
   - **Branch:** `main` (or `master`)
   - **Main file path:** `app.py`
   - **App URL:** Choose a custom subdomain (optional)

4. **Click "Deploy"**

5. **Wait for deployment** (usually 2-5 minutes)

### Step 3: First Run

- Streamlit Cloud will automatically:
  - Install dependencies from `requirements.txt`
  - Generate the dataset on first run
  - Train models automatically on first run
  - Cache models for subsequent runs

## Important Notes

### Files Included:
✅ `app.py` - Main application
✅ `requirements.txt` - Dependencies
✅ All Python modules (`.py` files)
✅ `README.md` - Documentation

### Files Excluded (by .gitignore):
❌ `venv/` - Virtual environment
❌ `*.pkl` - Model files (will be generated on cloud)
❌ `data/*.csv` - Dataset (will be generated on cloud)
❌ `results/` - Output files
❌ `__pycache__/` - Python cache

### First Run Behavior:
- The app will generate data automatically
- Models will train automatically (takes ~10-30 seconds)
- Everything is cached for faster subsequent loads

## Troubleshooting

### If deployment fails:
1. Check `requirements.txt` - all dependencies must be listed
2. Ensure `app.py` is in the root directory
3. Check Streamlit Cloud logs for errors

### If models take too long:
- Models train automatically on first run
- Subsequent runs use cached models
- Consider pre-training models locally and committing them (if needed)

## Custom Domain (Optional)
- Streamlit Cloud provides a free subdomain: `your-app.streamlit.app`
- You can also use your own custom domain

## Cost
- **Free tier:** Unlimited apps
- **No credit card required**
- Perfect for portfolio projects!


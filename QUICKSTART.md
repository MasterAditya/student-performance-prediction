# 🚀 Quick Start Guide - Streamlit Application

## Installation Steps

### Step 1: Create Virtual Environment (Recommended)

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> 💡 **Tip**: You should see `(venv)` in your terminal prompt when the virtual environment is active.

### Step 2: Install Required Packages

```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- streamlit, plotly

### Step 3: Launch the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

> **Note**: Make sure your virtual environment is activated before running the app!

## First Time Setup

1. **Start the app**: Run `streamlit run app.py`
2. **Go to Home page**: Click "🔄 Load/Generate Dataset" to create the dataset
3. **Explore the Dataset**: View the student data in the Dataset page
4. **Run EDA**: Check out interactive visualizations in the EDA page
5. **Train Models**: 
   - Go to "Train Models" page
   - Click "🔄 Preprocess Data" first
   - Then click "🚀 Train Model" (select "Both" to train both models)
6. **Make Predictions**: Use the Predict page to predict grades for new students
7. **View Clustering**: See student clusters in the Clustering page
8. **Get Insights**: Check recommendations in the Insights page

## Troubleshooting

- **Port already in use**: If port 8501 is busy, Streamlit will use the next available port
- **Module not found**: Make sure all dependencies are installed: `pip install -r requirements.txt`
- **Dataset not loading**: The app will generate the dataset automatically on first run

## Features Overview

- **Interactive Visualizations**: All plots are interactive using Plotly
- **Real-time Predictions**: Predict student grades with custom inputs
- **Model Training**: Train and evaluate ML models in the browser
- **Data Exploration**: Comprehensive EDA with interactive charts
- **Clustering Analysis**: Visualize student groups and patterns

Enjoy exploring the Student Performance Prediction System! 📚


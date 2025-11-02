"""
Pre-train models for the Student Performance Prediction System
Run this script once to train and save models before using the Streamlit app
"""

import pandas as pd
import numpy as np
import os
from generate_data import generate_student_data
from data_preprocessing import DataPreprocessor
from models import StudentPerformanceModels
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
import joblib

def train_and_save_models():
    """Train and save models."""
    print("=" * 80)
    print("Training Models for Student Performance Prediction")
    print("=" * 80)
    
    # Load or generate data
    data_path = 'data/student_performance.csv'
    if os.path.exists(data_path):
        print("Loading dataset...")
        df = pd.read_csv(data_path)
    else:
        print("Generating dataset...")
        df = generate_student_data(n_students=500, seed=42)
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
    
    print(f"Dataset loaded: {len(df)} students")
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor()
    X, y, feature_names = preprocessor.preprocess(
        df,
        target_column='Final_Grade',
        handle_missing=True,
        handle_outliers=True,
        encode_categorical=True,
        scale_features=True
    )
    print(f"Features: {len(feature_names)}")
    
    # Create model directory
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Train Linear Regression
    print("\nTraining Linear Regression model...")
    models = StudentPerformanceModels(X, y, feature_names)
    models.split_data()
    lr_model = LinearRegression()
    lr_model.fit(models.X_train, models.y_train)
    
    # Evaluate
    train_r2 = lr_model.score(models.X_train, models.y_train)
    test_r2 = lr_model.score(models.X_test, models.y_test)
    test_rmse = np.sqrt(mean_squared_error(models.y_test, lr_model.predict(models.X_test)))
    
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")
    print(f"  Testing RMSE: {test_rmse:.4f}")
    
    # Train K-Means
    print("\nTraining K-Means clustering model...")
    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X)
    
    print(f"  Number of clusters: {len(set(cluster_labels))}")
    print(f"  Inertia: {kmeans_model.inertia_:.2f}")
    
    # Save models
    print("\nSaving models...")
    joblib.dump(lr_model, f'{model_dir}/linear_regression_model.pkl')
    joblib.dump(kmeans_model, f'{model_dir}/kmeans_model.pkl')
    joblib.dump(preprocessor, f'{model_dir}/preprocessor.pkl')
    
    print("\n" + "=" * 80)
    print("Models trained and saved successfully!")
    print("=" * 80)
    print(f"\nSaved files:")
    print(f"  - {model_dir}/linear_regression_model.pkl")
    print(f"  - {model_dir}/kmeans_model.pkl")
    print(f"  - {model_dir}/preprocessor.pkl")

if __name__ == "__main__":
    train_and_save_models()


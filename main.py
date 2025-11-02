"""
Main execution script for Student Performance Prediction Project
This script runs the complete ML workflow from data generation to model evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np

# Import custom modules
from generate_data import generate_student_data
from data_preprocessing import DataPreprocessor
from eda import StudentPerformanceEDA
from models import StudentPerformanceModels
from visualizations import ModelVisualizations

def main():
    """Main execution function."""
    print("=" * 80)
    print("STUDENT PERFORMANCE PREDICTION PROJECT")
    print("ML & DS Internship - CipherSchools")
    print("=" * 80)
    
    # Create necessary directories
    directories = ['data', 'results', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Generate or load dataset
    print("\n" + "=" * 80)
    print("STEP 1: DATA GENERATION")
    print("=" * 80)
    
    data_path = 'data/student_performance.csv'
    if not os.path.exists(data_path):
        print("Generating synthetic student performance dataset...")
        df = generate_student_data(n_students=500, seed=42)
        df.to_csv(data_path, index=False)
        print(f"Dataset saved to {data_path}")
    else:
        print(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)
    
    print(f"Dataset loaded: {df.shape[0]} students, {df.shape[1]} features")
    
    # Step 2: Data Preprocessing
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 80)
    
    preprocessor = DataPreprocessor()
    
    # Display missing values before preprocessing
    print("\nMissing values before preprocessing:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Preprocess data
    X, y, feature_names = preprocessor.preprocess(
        df, 
        target_column='Final_Grade',
        handle_missing=True,
        handle_outliers=True,
        encode_categorical=True,
        scale_features=True
    )
    
    print(f"\nPreprocessed data shape: {X.shape}")
    print(f"Feature names: {feature_names}")
    
    # Step 3: Exploratory Data Analysis
    print("\n" + "=" * 80)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    eda = StudentPerformanceEDA(df)
    eda.run_complete_eda(save_results=True)
    
    # Step 4: Model Training - Linear Regression
    print("\n" + "=" * 80)
    print("STEP 4: MODEL TRAINING - LINEAR REGRESSION")
    print("=" * 80)
    
    models = StudentPerformanceModels(X, y, feature_names)
    lr_results = models.train_linear_regression()
    
    # Create visualizations for regression
    viz = ModelVisualizations()
    
    # Get predictions for visualization
    models.split_data()
    y_train_pred = models.lr_model.predict(models.X_train)
    y_test_pred = models.lr_model.predict(models.X_test)
    
    viz.plot_regression_results(models.y_test, y_test_pred)
    viz.plot_feature_importance(lr_results['coefficients'])
    
    # Step 5: Model Training - K-Means Clustering
    print("\n" + "=" * 80)
    print("STEP 5: MODEL TRAINING - K-MEANS CLUSTERING")
    print("=" * 80)
    
    # Find optimal number of clusters
    elbow_results = models.find_optimal_clusters(max_clusters=8)
    optimal_k = elbow_results['optimal_k']
    
    # Plot elbow method
    viz.plot_elbow_method(elbow_results['cluster_range'], elbow_results['inertias'])
    
    # Train K-Means with optimal k
    cluster_labels, inertia = models.train_kmeans(n_clusters=optimal_k, random_state=42)
    
    # Analyze clusters
    df_clustered, cluster_stats = models.analyze_clusters(cluster_labels, df)
    
    # Save clustered data
    df_clustered.to_csv('results/clustered_students.csv', index=False)
    print("\nClustered data saved to results/clustered_students.csv")
    
    # Create cluster visualizations
    viz.plot_clusters(X, cluster_labels, feature_names)
    viz.plot_cluster_analysis(df_clustered)
    viz.plot_cluster_grade_distribution(df_clustered)
    
    # Step 6: Save Models
    print("\n" + "=" * 80)
    print("STEP 6: SAVING MODELS")
    print("=" * 80)
    
    models.save_models()
    
    # Step 7: Generate Summary Report
    print("\n" + "=" * 80)
    print("STEP 7: GENERATING SUMMARY REPORT")
    print("=" * 80)
    
    generate_summary_report(lr_results, cluster_stats, df_clustered, optimal_k)
    
    print("\n" + "=" * 80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  - Data: data/student_performance.csv")
    print("  - Results: results/ (all visualizations and analysis)")
    print("  - Models: models/ (saved ML models)")
    print("  - Report: results/summary_report.txt")
    print("=" * 80)

def generate_summary_report(lr_results, cluster_stats, df_clustered, optimal_k):
    """Generate a summary report of the analysis."""
    report_path = 'results/summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STUDENT PERFORMANCE PREDICTION - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PROJECT OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write("This project predicts student academic performance using historical data\n")
        f.write("including attendance, assignment scores, and test results.\n\n")
        
        f.write("LINEAR REGRESSION MODEL RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training R² Score: {lr_results['train_r2']:.4f}\n")
        f.write(f"Training RMSE: {lr_results['train_rmse']:.4f}\n")
        f.write(f"Testing R² Score: {lr_results['test_r2']:.4f}\n")
        f.write(f"Testing RMSE: {lr_results['test_rmse']:.4f}\n\n")
        
        f.write("Top 5 Most Important Features:\n")
        top_features = lr_results['coefficients'].head(5)
        for idx, row in top_features.iterrows():
            f.write(f"  {idx+1}. {row['Feature']}: {row['Coefficient']:.4f}\n")
        f.write("\n")
        
        f.write("K-MEANS CLUSTERING RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Optimal Number of Clusters: {optimal_k}\n\n")
        
        f.write("Cluster Distribution:\n")
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            percentage = (count / len(df_clustered)) * 100
            avg_grade = df_clustered[df_clustered['Cluster'] == cluster]['Final_Grade'].mean()
            f.write(f"  Cluster {cluster}: {count} students ({percentage:.1f}%), "
                   f"Avg Grade: {avg_grade:.2f}\n")
        f.write("\n")
        
        f.write("KEY INSIGHTS\n")
        f.write("-" * 80 + "\n")
        f.write("1. The Linear Regression model shows good predictive performance.\n")
        f.write("2. Key factors affecting student performance have been identified.\n")
        f.write("3. Students have been grouped into distinct clusters based on performance patterns.\n")
        f.write("4. These insights can help identify students needing additional support.\n\n")
        
        f.write("ACTIONABLE RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Focus on improving attendance and study hours for underperforming students.\n")
        f.write("2. Provide targeted support to students in lower-performing clusters.\n")
        f.write("3. Monitor key factors identified as most important for academic success.\n")
        f.write("4. Use clustering to develop personalized intervention strategies.\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    main()


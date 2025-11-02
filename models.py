"""
Machine Learning Models Module
Implements Linear Regression for prediction and K-Means for clustering.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

class StudentPerformanceModels:
    """Class to implement ML models for student performance prediction."""
    
    def __init__(self, X, y, feature_names):
        """
        Initialize models.
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target variable
        feature_names : list
            List of feature names
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.lr_model = None
        self.kmeans_model = None
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Data split completed:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Testing set: {self.X_test.shape[0]} samples")
    
    def train_linear_regression(self):
        """Train Linear Regression model for grade prediction."""
        print("\n" + "=" * 80)
        print("TRAINING LINEAR REGRESSION MODEL")
        print("=" * 80)
        
        if self.X_train is None:
            self.split_data()
        
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train, self.y_train)
        
        # Training predictions
        y_train_pred = self.lr_model.predict(self.X_train)
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        
        # Testing predictions
        y_test_pred = self.lr_model.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        print(f"\nTraining Performance:")
        print(f"  R² Score: {train_r2:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        
        print(f"\nTesting Performance:")
        print(f"  R² Score: {test_r2:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        
        # Feature importance (coefficients)
        coefficients = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.lr_model.coef_
        })
        coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
        coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(coefficients.head(10).to_string(index=False))
        
        return {
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'coefficients': coefficients
        }
    
    def train_kmeans(self, n_clusters=4, random_state=42):
        """
        Train K-Means clustering model to group students.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        random_state : int
            Random state for reproducibility
        """
        print("\n" + "=" * 80)
        print(f"TRAINING K-MEANS CLUSTERING MODEL (n_clusters={n_clusters})")
        print("=" * 80)
        
        # Use full dataset for clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(self.X)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = self.kmeans_model.inertia_
        
        print(f"\nClustering Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Inertia (WCSS): {inertia:.2f}")
        
        # Cluster distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print(f"\nCluster Distribution:")
        for cluster, count in zip(unique, counts):
            percentage = (count / len(cluster_labels)) * 100
            print(f"  Cluster {cluster}: {count} students ({percentage:.1f}%)")
        
        return cluster_labels, inertia
    
    def find_optimal_clusters(self, max_clusters=10):
        """
        Find optimal number of clusters using elbow method.
        
        Parameters:
        -----------
        max_clusters : int
            Maximum number of clusters to test
        
        Returns:
        --------
        dict
            Dictionary with cluster numbers and their inertias
        """
        print("\n" + "=" * 80)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS (Elbow Method)")
        print("=" * 80)
        
        inertias = []
        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X)
            inertias.append(kmeans.inertia_)
            print(f"  k={k}: Inertia = {kmeans.inertia_:.2f}")
        
        # Find elbow point (simple method: largest decrease)
        if len(inertias) > 1:
            decreases = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            optimal_k = cluster_range[decreases.index(max(decreases))] + 1
            print(f"\nSuggested optimal k (elbow point): {optimal_k}")
        else:
            optimal_k = 3
        
        return {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'optimal_k': optimal_k
        }
    
    def analyze_clusters(self, cluster_labels, df_original):
        """
        Analyze clusters by examining characteristics of each cluster.
        
        Parameters:
        -----------
        cluster_labels : np.array
            Cluster labels for each student
        df_original : pd.DataFrame
            Original dataframe with feature names
        
        Returns:
        --------
        pd.DataFrame
            Cluster analysis summary
        """
        df_clustered = df_original.copy()
        df_clustered['Cluster'] = cluster_labels
        
        print("\n" + "=" * 80)
        print("CLUSTER ANALYSIS")
        print("=" * 80)
        
        # Group by cluster and calculate statistics
        numeric_cols = df_clustered.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Cluster']
        
        cluster_stats = df_clustered.groupby('Cluster')[numeric_cols].agg(['mean', 'std'])
        
        print("\nCluster Statistics (Mean ± Std):")
        print(cluster_stats)
        
        # Analyze each cluster
        n_clusters = len(np.unique(cluster_labels))
        print(f"\nDetailed Cluster Characteristics:")
        
        for cluster in range(n_clusters):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
            print(f"\nCluster {cluster} ({len(cluster_data)} students):")
            
            if 'Final_Grade' in cluster_data.columns:
                avg_grade = cluster_data['Final_Grade'].mean()
                print(f"  Average Final Grade: {avg_grade:.2f}")
                
                if 'Grade_Letter' in cluster_data.columns:
                    grade_dist = cluster_data['Grade_Letter'].value_counts()
                    print(f"  Grade Distribution: {dict(grade_dist)}")
            
            if 'Attendance' in cluster_data.columns:
                print(f"  Average Attendance: {cluster_data['Attendance'].mean():.2f}%")
            
            if 'Study_Hours_Per_Week' in cluster_data.columns:
                print(f"  Average Study Hours: {cluster_data['Study_Hours_Per_Week'].mean():.2f} hrs/week")
            
            if 'Avg_Assignment' in cluster_data.columns:
                print(f"  Average Assignment Score: {cluster_data['Avg_Assignment'].mean():.2f}")
            
            if 'Avg_Test' in cluster_data.columns:
                print(f"  Average Test Score: {cluster_data['Avg_Test'].mean():.2f}")
        
        return df_clustered, cluster_stats
    
    def save_models(self, model_dir='models'):
        """Save trained models."""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        if self.lr_model is not None:
            joblib.dump(self.lr_model, f'{model_dir}/linear_regression_model.pkl')
            print(f"\nLinear Regression model saved to {model_dir}/linear_regression_model.pkl")
        
        if self.kmeans_model is not None:
            joblib.dump(self.kmeans_model, f'{model_dir}/kmeans_model.pkl')
            print(f"K-Means model saved to {model_dir}/kmeans_model.pkl")
    
    def load_models(self, model_dir='models'):
        """Load trained models."""
        if self.lr_model is None:
            try:
                self.lr_model = joblib.load(f'{model_dir}/linear_regression_model.pkl')
                print(f"Linear Regression model loaded from {model_dir}/linear_regression_model.pkl")
            except FileNotFoundError:
                print("Linear Regression model not found.")
        
        if self.kmeans_model is None:
            try:
                self.kmeans_model = joblib.load(f'{model_dir}/kmeans_model.pkl')
                print(f"K-Means model loaded from {model_dir}/kmeans_model.pkl")
            except FileNotFoundError:
                print("K-Means model not found.")


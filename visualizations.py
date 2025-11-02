"""
Visualization Module for Model Results
Creates visualizations for model evaluation and cluster analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class ModelVisualizations:
    """Class to create visualizations for model results."""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """Setup plotting style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_regression_results(self, y_true, y_pred, save_path='results/regression_results.png'):
        """Plot regression model results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot: Predicted vs Actual
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=50)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Final Grade', fontweight='bold')
        axes[0].set_ylabel('Predicted Final Grade', fontweight='bold')
        axes[0].set_title('Predicted vs Actual Final Grade', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=50)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Final Grade', fontweight='bold')
        axes[1].set_ylabel('Residuals (Actual - Predicted)', fontweight='bold')
        axes[1].set_title('Residual Plot', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Regression results plot saved to {save_path}")
    
    def plot_feature_importance(self, coefficients, save_path='results/feature_importance.png'):
        """Plot feature importance from linear regression coefficients."""
        # Get top 10 features
        top_features = coefficients.head(10)
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
        plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors, edgecolor='black')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Coefficient Value', fontweight='bold')
        plt.title('Top 10 Feature Importance (Linear Regression Coefficients)', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {save_path}")
    
    def plot_elbow_method(self, cluster_range, inertias, save_path='results/elbow_method.png'):
        """Plot elbow method for optimal cluster selection."""
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, inertias, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontweight='bold')
        plt.ylabel('Inertia (WCSS)', fontweight='bold')
        plt.title('Elbow Method for Optimal Clusters', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Elbow method plot saved to {save_path}")
    
    def plot_clusters(self, X, cluster_labels, feature_names, save_path='results/clusters.png'):
        """Plot clusters using PCA for dimensionality reduction."""
        from sklearn.decomposition import PCA
        
        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                            cmap='viridis', s=50, alpha=0.6, edgecolors='black')
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})', 
                  fontweight='bold')
        plt.ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})', 
                  fontweight='bold')
        plt.title('Student Clusters (PCA Visualization)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cluster visualization saved to {save_path}")
    
    def plot_cluster_analysis(self, df_clustered, save_path='results/cluster_analysis.png'):
        """Plot cluster characteristics comparison."""
        n_clusters = df_clustered['Cluster'].nunique()
        
        # Select key features for visualization
        key_features = ['Final_Grade', 'Attendance', 'Avg_Assignment', 
                       'Avg_Test', 'Study_Hours_Per_Week', 'Previous_GPA']
        available_features = [f for f in key_features if f in df_clustered.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Cluster Characteristics Comparison', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(available_features[:6]):
            row = idx // 3
            col = idx % 3
            
            cluster_means = df_clustered.groupby('Cluster')[feature].mean()
            cluster_stds = df_clustered.groupby('Cluster')[feature].std()
            
            x_pos = np.arange(len(cluster_means))
            axes[row, col].bar(x_pos, cluster_means.values, yerr=cluster_stds.values, 
                              capsize=5, edgecolor='black', alpha=0.7)
            axes[row, col].set_xlabel('Cluster', fontweight='bold')
            axes[row, col].set_ylabel(feature, fontweight='bold')
            axes[row, col].set_title(f'Average {feature} by Cluster', fontweight='bold')
            axes[row, col].set_xticks(x_pos)
            axes[row, col].set_xticklabels([f'Cluster {i}' for i in cluster_means.index])
            axes[row, col].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cluster analysis plot saved to {save_path}")
    
    def plot_cluster_grade_distribution(self, df_clustered, save_path='results/cluster_grades.png'):
        """Plot grade distribution within each cluster."""
        if 'Grade_Letter' not in df_clustered.columns:
            print("Grade_Letter column not found. Skipping grade distribution plot.")
            return
        
        n_clusters = df_clustered['Cluster'].nunique()
        grades = sorted(df_clustered['Grade_Letter'].unique())
        
        fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 6))
        if n_clusters == 1:
            axes = [axes]
        
        fig.suptitle('Grade Distribution by Cluster', fontsize=16, fontweight='bold')
        
        for cluster in range(n_clusters):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
            grade_counts = cluster_data['Grade_Letter'].value_counts()
            grade_counts = grade_counts.reindex(grades, fill_value=0)
            
            axes[cluster].bar(grade_counts.index, grade_counts.values, 
                             color='steelblue', edgecolor='black')
            axes[cluster].set_title(f'Cluster {cluster}\n({len(cluster_data)} students)', 
                                   fontweight='bold')
            axes[cluster].set_xlabel('Grade Letter', fontweight='bold')
            axes[cluster].set_ylabel('Number of Students', fontweight='bold')
            axes[cluster].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cluster grade distribution plot saved to {save_path}")


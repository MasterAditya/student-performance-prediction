"""
Exploratory Data Analysis (EDA) Module
Performs comprehensive EDA with visualizations and correlation analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StudentPerformanceEDA:
    """Class to perform EDA on student performance data."""
    
    def __init__(self, df):
        self.df = df
        self.setup_style()
    
    def setup_style(self):
        """Setup plotting style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def basic_info(self):
        """Display basic information about the dataset."""
        print("=" * 80)
        print("BASIC DATASET INFORMATION")
        print("=" * 80)
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nColumn Names:")
        print(self.df.columns.tolist())
        print(f"\nData Types:")
        print(self.df.dtypes)
        print(f"\nMissing Values:")
        print(self.df.isnull().sum())
        print(f"\nBasic Statistics:")
        print(self.df.describe())
        print("=" * 80)
    
    def plot_distributions(self, save_path='results/distributions.png'):
        """Plot distributions of key features."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Distribution of Key Features', fontsize=16, fontweight='bold')
        
        features = ['Attendance', 'Avg_Assignment', 'Avg_Test', 
                   'Study_Hours_Per_Week', 'Previous_GPA', 'Final_Grade']
        
        for idx, feature in enumerate(features):
            row = idx // 3
            col = idx % 3
            axes[row, col].hist(self.df[feature], bins=30, edgecolor='black', alpha=0.7)
            axes[row, col].set_title(f'Distribution of {feature}', fontweight='bold')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Distribution plots saved to {save_path}")
    
    def plot_correlation_heatmap(self, save_path='results/correlation_heatmap.png'):
        """Plot correlation heatmap."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        plt.figure(figsize=(14, 12))
        correlation_matrix = numeric_df.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Heatmap of Numeric Features', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation heatmap saved to {save_path}")
    
    def plot_grade_distribution(self, save_path='results/grade_distribution.png'):
        """Plot distribution of final grades."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Grade letter distribution
        grade_counts = self.df['Grade_Letter'].value_counts().sort_index()
        axes[0].bar(grade_counts.index, grade_counts.values, color='steelblue', edgecolor='black')
        axes[0].set_title('Distribution of Letter Grades', fontweight='bold')
        axes[0].set_xlabel('Grade Letter')
        axes[0].set_ylabel('Number of Students')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Final grade distribution
        axes[1].hist(self.df['Final_Grade'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(self.df['Final_Grade'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {self.df["Final_Grade"].mean():.2f}')
        axes[1].axvline(self.df['Final_Grade'].median(), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {self.df["Final_Grade"].median():.2f}')
        axes[1].set_title('Distribution of Final Grades (Numeric)', fontweight='bold')
        axes[1].set_xlabel('Final Grade')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Grade distribution plots saved to {save_path}")
    
    def plot_feature_vs_target(self, save_path='results/feature_vs_target.png'):
        """Plot relationship between features and target (Final_Grade)."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Relationship Between Features and Final Grade', fontsize=16, fontweight='bold')
        
        features = ['Attendance', 'Avg_Assignment', 'Avg_Test', 
                   'Study_Hours_Per_Week', 'Previous_GPA']
        
        for idx, feature in enumerate(features):
            row = idx // 3
            col = idx % 3
            axes[row, col].scatter(self.df[feature], self.df['Final_Grade'], 
                                  alpha=0.5, s=30)
            
            # Add trend line (handle potential NaN values)
            feature_data = self.df[feature].dropna()
            target_data = self.df.loc[feature_data.index, 'Final_Grade']
            
            if len(feature_data) > 1 and feature_data.std() > 0:
                try:
                    z = np.polyfit(feature_data, target_data, 1)
                    p = np.poly1d(z)
                    x_sorted = np.sort(feature_data)
                    axes[row, col].plot(x_sorted, p(x_sorted), 
                                       "r--", alpha=0.8, linewidth=2)
                except:
                    pass  # Skip trend line if polyfit fails
            
            # Calculate correlation
            corr = self.df[feature].corr(self.df['Final_Grade'])
            axes[row, col].set_title(f'{feature} vs Final Grade\n(Correlation: {corr:.3f})', 
                                    fontweight='bold')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Final Grade')
            axes[row, col].grid(True, alpha=0.3)
        
        # Gender comparison
        gender_means = self.df.groupby('Gender')['Final_Grade'].mean()
        axes[1, 2].bar(gender_means.index, gender_means.values, color=['skyblue', 'salmon'], 
                      edgecolor='black')
        axes[1, 2].set_title('Average Final Grade by Gender', fontweight='bold')
        axes[1, 2].set_xlabel('Gender')
        axes[1, 2].set_ylabel('Average Final Grade')
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature vs target plots saved to {save_path}")
    
    def plot_boxplots(self, save_path='results/boxplots.png'):
        """Plot boxplots to identify outliers."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Boxplots for Outlier Detection', fontsize=16, fontweight='bold')
        
        features = ['Attendance', 'Avg_Assignment', 'Avg_Test', 
                   'Study_Hours_Per_Week', 'Previous_GPA', 'Final_Grade']
        
        for idx, feature in enumerate(features):
            row = idx // 3
            col = idx % 3
            axes[row, col].boxplot(self.df[feature], vert=True)
            axes[row, col].set_title(f'Boxplot of {feature}', fontweight='bold')
            axes[row, col].set_ylabel(feature)
            axes[row, col].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Boxplots saved to {save_path}")
    
    def correlation_analysis(self):
        """Perform correlation analysis with target variable."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if 'Final_Grade' in numeric_df.columns:
            correlations = numeric_df.corr()['Final_Grade'].sort_values(ascending=False)
            
            print("=" * 80)
            print("CORRELATION WITH FINAL GRADE")
            print("=" * 80)
            print(correlations)
            print("=" * 80)
            
            return correlations
        return None
    
    def key_insights(self):
        """Generate key insights from the data."""
        print("\n" + "=" * 80)
        print("KEY INSIGHTS")
        print("=" * 80)
        
        print(f"\n1. Overall Performance:")
        print(f"   - Average Final Grade: {self.df['Final_Grade'].mean():.2f}")
        print(f"   - Median Final Grade: {self.df['Final_Grade'].median():.2f}")
        print(f"   - Standard Deviation: {self.df['Final_Grade'].std():.2f}")
        
        print(f"\n2. Grade Distribution:")
        grade_dist = self.df['Grade_Letter'].value_counts().sort_index()
        for grade, count in grade_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"   - Grade {grade}: {count} students ({percentage:.1f}%)")
        
        print(f"\n3. Top Factors Affecting Performance:")
        if 'Final_Grade' in self.df.columns:
            numeric_df = self.df.select_dtypes(include=[np.number])
            correlations = numeric_df.corr()['Final_Grade'].abs().sort_values(ascending=False)
            correlations = correlations.drop('Final_Grade')
            top_factors = correlations.head(5)
            for factor, corr in top_factors.items():
                print(f"   - {factor}: {corr:.3f} correlation")
        
        print(f"\n4. Attendance Analysis:")
        print(f"   - Average Attendance: {self.df['Attendance'].mean():.2f}%")
        print(f"   - Students with >90% attendance: {(self.df['Attendance'] > 90).sum()}")
        print(f"   - Students with <70% attendance: {(self.df['Attendance'] < 70).sum()}")
        
        print(f"\n5. Study Hours Analysis:")
        print(f"   - Average Study Hours/Week: {self.df['Study_Hours_Per_Week'].mean():.2f}")
        print(f"   - Students studying >20 hrs/week: {(self.df['Study_Hours_Per_Week'] > 20).sum()}")
        
        print("=" * 80 + "\n")
    
    def run_complete_eda(self, save_results=True):
        """Run complete EDA pipeline."""
        print("Starting Exploratory Data Analysis...\n")
        
        # Basic information
        self.basic_info()
        
        # Generate all visualizations
        if save_results:
            import os
            os.makedirs('results', exist_ok=True)
            
            self.plot_distributions()
            self.plot_correlation_heatmap()
            self.plot_grade_distribution()
            self.plot_feature_vs_target()
            self.plot_boxplots()
        
        # Correlation analysis
        self.correlation_analysis()
        
        # Key insights
        self.key_insights()
        
        print("EDA completed successfully!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'data/student_performance.csv'
    
    df = pd.read_csv(data_path)
    eda = StudentPerformanceEDA(df)
    eda.run_complete_eda()


"""
Generate synthetic student performance dataset.
Creates a realistic dataset with attendance, assignment scores, test results, and final grades.
"""

import pandas as pd
import numpy as np

def generate_student_data(n_students=500, seed=42):
    """
    Generate synthetic student performance data.
    
    Parameters:
    -----------
    n_students : int
        Number of students to generate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing student performance data
    """
    np.random.seed(seed)
    
    data = {
        'Student_ID': [f'STU_{i+1:04d}' for i in range(n_students)],
        'Age': np.random.randint(18, 25, n_students),
        'Gender': np.random.choice(['Male', 'Female'], n_students),
        
        # Attendance (0-100%)
        'Attendance': np.clip(np.random.normal(85, 15, n_students), 0, 100),
        
        # Assignment scores (out of 100)
        'Assignment_1': np.clip(np.random.normal(75, 15, n_students), 0, 100),
        'Assignment_2': np.clip(np.random.normal(75, 15, n_students), 0, 100),
        'Assignment_3': np.clip(np.random.normal(75, 15, n_students), 0, 100),
        'Assignment_4': np.clip(np.random.normal(75, 15, n_students), 0, 100),
        
        # Test scores (out of 100)
        'Test_1': np.clip(np.random.normal(70, 18, n_students), 0, 100),
        'Test_2': np.clip(np.random.normal(70, 18, n_students), 0, 100),
        'Midterm': np.clip(np.random.normal(72, 16, n_students), 0, 100),
        'Final_Exam': np.clip(np.random.normal(70, 20, n_students), 0, 100),
        
        # Additional features
        'Study_Hours_Per_Week': np.clip(np.random.normal(15, 5, n_students), 0, 40),
        'Previous_GPA': np.clip(np.random.normal(3.0, 0.7, n_students), 0, 4.0),
    }
    
    df = pd.DataFrame(data)
    
    # Introduce correlations: students with higher attendance and study hours perform better
    for idx in range(n_students):
        # Adjust assignment scores based on attendance
        attendance_factor = (df.loc[idx, 'Attendance'] / 100) * 10
        df.loc[idx, 'Assignment_1'] = np.clip(df.loc[idx, 'Assignment_1'] + attendance_factor, 0, 100)
        df.loc[idx, 'Assignment_2'] = np.clip(df.loc[idx, 'Assignment_2'] + attendance_factor, 0, 100)
        df.loc[idx, 'Assignment_3'] = np.clip(df.loc[idx, 'Assignment_3'] + attendance_factor, 0, 100)
        df.loc[idx, 'Assignment_4'] = np.clip(df.loc[idx, 'Assignment_4'] + attendance_factor, 0, 100)
        
        # Adjust test scores based on study hours
        study_factor = (df.loc[idx, 'Study_Hours_Per_Week'] / 40) * 15
        df.loc[idx, 'Test_1'] = np.clip(df.loc[idx, 'Test_1'] + study_factor, 0, 100)
        df.loc[idx, 'Test_2'] = np.clip(df.loc[idx, 'Test_2'] + study_factor, 0, 100)
        df.loc[idx, 'Midterm'] = np.clip(df.loc[idx, 'Midterm'] + study_factor, 0, 100)
        df.loc[idx, 'Final_Exam'] = np.clip(df.loc[idx, 'Final_Exam'] + study_factor, 0, 100)
        
        # Adjust based on previous GPA
        gpa_factor = (df.loc[idx, 'Previous_GPA'] / 4.0) * 8
        df.loc[idx, 'Test_1'] = np.clip(df.loc[idx, 'Test_1'] + gpa_factor, 0, 100)
        df.loc[idx, 'Midterm'] = np.clip(df.loc[idx, 'Midterm'] + gpa_factor, 0, 100)
        df.loc[idx, 'Final_Exam'] = np.clip(df.loc[idx, 'Final_Exam'] + gpa_factor, 0, 100)
    
    # Calculate average assignment and test scores
    df['Avg_Assignment'] = df[['Assignment_1', 'Assignment_2', 'Assignment_3', 'Assignment_4']].mean(axis=1)
    df['Avg_Test'] = df[['Test_1', 'Test_2', 'Midterm', 'Final_Exam']].mean(axis=1)
    
    # Calculate Final Grade (weighted average)
    # Attendance: 10%, Assignments: 30%, Tests: 60%
    df['Final_Grade'] = (
        df['Attendance'] * 0.10 +
        df['Avg_Assignment'] * 0.30 +
        df['Avg_Test'] * 0.60
    )
    
    # Convert Final Grade to letter grade
    def grade_to_letter(score):
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    df['Grade_Letter'] = df['Final_Grade'].apply(grade_to_letter)
    
    # Introduce some missing values (5% random missing)
    missing_indices = np.random.choice(n_students, size=int(n_students * 0.05), replace=False)
    for idx in missing_indices:
        col = np.random.choice(['Assignment_2', 'Test_1', 'Study_Hours_Per_Week'])
        df.loc[idx, col] = np.nan
    
    # Introduce some outliers (2% of data)
    outlier_indices = np.random.choice(n_students, size=int(n_students * 0.02), replace=False)
    for idx in outlier_indices:
        df.loc[idx, 'Attendance'] = np.random.choice([0, 100])  # Extreme attendance
    
    return df

if __name__ == "__main__":
    print("Generating student performance dataset...")
    df = generate_student_data(n_students=500)
    df.to_csv('data/student_performance.csv', index=False)
    print(f"Dataset generated with {len(df)} students!")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nMissing values:")
    print(df.isnull().sum())


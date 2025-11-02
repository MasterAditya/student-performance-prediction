"""
Data Preprocessing Module
Handles missing values, outliers, and feature scaling for student performance data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    """Class to handle data preprocessing tasks."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        strategy : str
            Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with missing values handled
        """
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                if strategy == 'mean':
                    df_clean = df_clean.assign(**{col: df_clean[col].fillna(df_clean[col].mean())})
                elif strategy == 'median':
                    df_clean = df_clean.assign(**{col: df_clean[col].fillna(df_clean[col].median())})
                elif strategy == 'mode':
                    df_clean = df_clean.assign(**{col: df_clean[col].fillna(df_clean[col].mode()[0])})
        
        # For categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean = df_clean.assign(**{col: df_clean[col].fillna(df_clean[col].mode()[0])})
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        
        return df_clean
    
    def detect_outliers(self, df, columns=None, method='IQR'):
        """
        Detect outliers using IQR method.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            Columns to check for outliers (None = all numeric columns)
        method : str
            Method to detect outliers ('IQR' or 'Z-score')
        
        Returns:
        --------
        pd.DataFrame
            Boolean mask indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            if method == 'IQR':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'Z-score':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask[col] = z_scores > 3
        
        return outlier_mask
    
    def handle_outliers(self, df, columns=None, method='cap'):
        """
        Handle outliers in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            Columns to handle outliers for
        method : str
            Method to handle outliers ('cap', 'remove', 'transform')
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with outliers handled
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns
        
        outlier_mask = self.detect_outliers(df_clean, columns)
        
        for col in columns:
            outliers = outlier_mask[col]
            
            if outliers.sum() > 0:
                if method == 'cap':
                    # Cap outliers at 1.5 * IQR
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                
                elif method == 'remove':
                    df_clean = df_clean[~outliers]
                
                elif method == 'transform':
                    # Log transform for right-skewed data
                    if df_clean[col].min() > 0:
                        df_clean[col] = np.log1p(df_clean[col])
        
        return df_clean
    
    def encode_categorical(self, df, columns=None):
        """
        Encode categorical variables using label encoding.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            Columns to encode (None = all categorical columns)
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with encoded categorical variables
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object']).columns
        
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def scale_features(self, df, columns=None, fit=True):
        """
        Scale features using StandardScaler.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            Columns to scale (None = all numeric columns)
        fit : bool
            Whether to fit the scaler or use existing fit
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with scaled features
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_scaled = df.copy()
        
        if fit:
            df_scaled[columns] = self.scaler.fit_transform(df[columns])
        else:
            df_scaled[columns] = self.scaler.transform(df[columns])
        
        return df_scaled
    
    def preprocess(self, df, target_column='Final_Grade', 
                   handle_missing=True, handle_outliers=True,
                   encode_categorical=True, scale_features=True):
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_column : str
            Name of target column
        handle_missing : bool
            Whether to handle missing values
        handle_outliers : bool
            Whether to handle outliers
        encode_categorical : bool
            Whether to encode categorical variables
        scale_features : bool
            Whether to scale features
        
        Returns:
        --------
        tuple
            (X, y, feature_names) where X is features, y is target
        """
        df_processed = df.copy()
        
        # Handle missing values
        if handle_missing:
            df_processed = self.handle_missing_values(df_processed, strategy='mean')
        
        # Handle outliers
        if handle_outliers:
            df_processed = self.handle_outliers(df_processed, method='cap')
        
        # Separate features and target
        if target_column and target_column in df_processed.columns:
            y = df_processed[target_column].values
            X_df = df_processed.drop(columns=[target_column, 'Grade_Letter', 'Student_ID'], errors='ignore')
        else:
            y = None
            X_df = df_processed.drop(columns=['Grade_Letter', 'Student_ID'], errors='ignore')
        
        # Encode categorical variables
        if encode_categorical:
            X_df = self.encode_categorical(X_df)
        
        # Scale features
        if scale_features:
            X_df = self.scale_features(X_df, fit=True)
        
        X = X_df.values
        feature_names = X_df.columns.tolist()
        
        return X, y, feature_names


"""
Feature Normalizer
Handles normalization and scaling of features
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Optional, Tuple


class FeatureNormalizer:
    """Normalize and scale features for model training."""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize feature normalizer.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust', or None)
        """
        self.method = method
        self.scaler = None
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method is None or method == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def fit(self, X: pd.DataFrame) -> 'FeatureNormalizer':
        """
        Fit the normalizer to the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Self
        """
        if self.scaler is not None:
            self.scaler.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Normalized feature matrix
        """
        if self.scaler is None:
            return X
        
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Normalized feature matrix
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data back to original scale.
        
        Args:
            X: Normalized feature matrix
            
        Returns:
            Original scale feature matrix
        """
        if self.scaler is None:
            return X
        
        X_original = self.scaler.inverse_transform(X)
        return pd.DataFrame(X_original, index=X.index, columns=X.columns)


class OutlierHandler:
    """Handle outliers in the data."""
    
    def __init__(self, method: str = 'clip', threshold: float = 3.0):
        """
        Initialize outlier handler.
        
        Args:
            method: Method to handle outliers ('clip', 'remove', or 'none')
            threshold: Number of standard deviations for outlier detection
        """
        self.method = method
        self.threshold = threshold
        self.outlier_mask = None
    
    def fit(self, X: pd.DataFrame) -> 'OutlierHandler':
        """
        Fit the outlier handler.
        
        Args:
            X: Feature matrix
            
        Returns:
            Self
        """
        if self.method == 'none':
            return self
        
        # Calculate z-scores
        z_scores = np.abs((X - X.mean()) / X.std())
        self.outlier_mask = (z_scores > self.threshold).any(axis=1)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by handling outliers.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if self.method == 'none':
            return X
        
        X_transformed = X.copy()
        
        if self.method == 'clip':
            # Clip values to threshold standard deviations
            for col in X.columns:
                mean = X[col].mean()
                std = X[col].std()
                lower = mean - self.threshold * std
                upper = mean + self.threshold * std
                X_transformed[col] = X[col].clip(lower, upper)
        
        elif self.method == 'remove':
            # Remove outlier rows
            X_transformed = X[~self.outlier_mask]
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X).transform(X)

"""
Data preprocessing module for PPROM prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles all data preprocessing steps"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.imputer = None
        self.encoders = {}
        
    def load_data(self, path):
        """Load and clean data"""
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[^\w_]', '', regex=True)
        return df
    
    def split_data(self, df, target_column):
        """Split data into train and test sets"""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def preprocess(self, X_train, X_test):
        """Apply preprocessing pipeline"""
        # Handle categorical variables
        X_train_processed, X_test_processed = self._encode_categorical(X_train, X_test)
        
        # Handle missing values
        X_train_processed, X_test_processed = self._impute_missing(
            X_train_processed, X_test_processed
        )
        
        # Scale features
        X_train_processed, X_test_processed = self._scale_features(
            X_train_processed, X_test_processed
        )
        
        return X_train_processed, X_test_processed
    
    def _encode_categorical(self, X_train, X_test):
        """Encode categorical variables safely"""
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in X_train.columns:
                # Handle missing values
                X_train_encoded[col] = X_train[col].fillna('Unknown')
                X_test_encoded[col] = X_test[col].fillna('Unknown')
                
                # Create encoder
                le = LabelEncoder()
                all_values = pd.concat([X_train[col], X_test[col]]).unique()
                le.fit([str(x) for x in all_values])
                
                # Transform
                X_train_encoded[col] = le.transform(X_train_encoded[col].astype(str))
                X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
                
                self.encoders[col] = le
        
        return X_train_encoded, X_test_encoded
    
    def _impute_missing(self, X_train, X_test):
        """Impute missing values"""
        if self.config.imputation_method == "knn":
            self.imputer = KNNImputer(n_neighbors=5, weights='uniform')
        else:
            self.imputer = SimpleImputer(strategy='mean')
        
        X_train_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_imputed = pd.DataFrame(
            self.imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_imputed, X_test_imputed
    
    def _scale_features(self, X_train, X_test):
        """Scale features based on config"""
        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaling_method == "maxabs":
            self.scaler = MaxAbsScaler()
        else:
            return X_train, X_test
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled

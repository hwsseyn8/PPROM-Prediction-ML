"""
Data preprocessing module for PPROM prediction study.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import warnings
import os
from typing import Tuple, Optional, Dict, Any

from .config import config


class DataPreprocessor:
    """Handles all data preprocessing steps."""
    
    def __init__(self, config=config):
        self.config = config
        self.scaler = None
        self.imputer = None
        self.label_encoders = {}
        self.feature_columns = None
        self.outcome_columns = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and clean the dataset."""
        print(f"Loading data from {self.config.raw_data_path}")
        
        try:
            df = pd.read_csv(self.config.raw_data_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found at {self.config.raw_data_path}. "
                f"Please place your data23.csv file in data/raw/"
            )
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[^\w_]', '', regex=True)
        
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"Missing values: {df.isnull().sum().sum()} total")
        
        return df
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality and report issues."""
        quality_report = {
            'total_samples': df.shape[0],
            'total_features': df.shape[1],
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'outcome_present': self.config.primary_outcome in df.columns,
        }
        
        if quality_report['outcome_present']:
            outcome_counts = df[self.config.primary_outcome].value_counts()
            quality_report['outcome_distribution'] = {
                'class_0': outcome_counts.get(0, 0),
                'class_1': outcome_counts.get(1, 0),
                'imbalance_ratio': outcome_counts.get(0, 0) / max(outcome_counts.get(1, 1), 1)
            }
        
        print("\nData Quality Report:")
        for key, value in quality_report.items():
            if key != 'outcome_distribution':
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        if 'outcome_distribution' in quality_report:
            dist = quality_report['outcome_distribution']
            print(f"  Outcome Distribution: Class 0={dist['class_0']}, Class 1={dist['class_1']}")
            print(f"  Imbalance Ratio: {dist['imbalance_ratio']:.2f}:1")
        
        return quality_report
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with threshold-based removal and KNN imputation."""
        # Remove rows with too many missing values
        missing_threshold = len(df.columns) * self.config.missing_threshold
        rows_too_many_missing = df.isnull().sum(axis=1) > missing_threshold
        
        if rows_too_many_missing.sum() > 0:
            print(f"Removing {rows_too_many_missing.sum()} rows with >{self.config.missing_threshold*100}% missing values")
            df_clean = df[~rows_too_many_missing].copy()
        else:
            df_clean = df.copy()
        
        return df_clean
    
    def safe_label_encode(self, train_series: pd.Series, test_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Safely encode categorical variables handling unseen categories."""
        all_values = pd.concat([train_series, test_series]).unique()
        unique_values = sorted([str(x) for x in all_values])
        value_to_code = {val: idx for idx, val in enumerate(unique_values)}
        
        train_encoded = train_series.map(value_to_code)
        test_encoded = test_series.map(value_to_code)
        
        return train_encoded, test_encoded
    
    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       task_name: str = "dataset") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply preprocessing pipeline to training and test data separately."""
        print(f"Preprocessing {task_name}...")
        
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # Handle categorical variables
        categorical_columns = []
        for col in X_train_processed.columns:
            if (X_train_processed[col].dtype == 'object' or 
                X_train_processed[col].nunique() <= 10):
                categorical_columns.append(col)
        
        if categorical_columns:
            print(f"  Encoding {len(categorical_columns)} categorical columns...")
            for col in categorical_columns:
                if col in X_train_processed.columns:
                    # Fill missing values
                    X_train_processed[col] = X_train_processed[col].fillna('Unknown')
                    X_test_processed[col] = X_test_processed[col].fillna('Unknown')
                    
                    # Convert to string
                    X_train_processed[col] = X_train_processed[col].astype(str)
                    X_test_processed[col] = X_test_processed[col].astype(str)
                    
                    # Safe encoding
                    train_encoded, test_encoded = self.safe_label_encode(
                        X_train_processed[col], X_test_processed[col]
                    )
                    X_train_processed[col] = train_encoded
                    X_test_processed[col] = test_encoded
        
        # Convert to float for imputation
        X_train_processed = X_train_processed.astype(float)
        X_test_processed = X_test_processed.astype(float)
        
        # Apply KNN imputation
        print("  Applying KNN imputation...")
        self.imputer = KNNImputer(n_neighbors=5, weights='uniform')
        X_train_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_train_processed),
            columns=X_train_processed.columns,
            index=X_train_processed.index
        )
        X_test_imputed = pd.DataFrame(
            self.imputer.transform(X_test_processed),
            columns=X_test_processed.columns,
            index=X_test_processed.index
        )
        
        # Apply feature scaling
        if self.config.scaling_method != "none":
            print(f"  Applying {self.config.scaling_method} scaling...")
            if self.config.scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.config.scaling_method == "maxabs":
                self.scaler = MaxAbsScaler()
            else:
                print(f"  Unknown scaling method. Using StandardScaler.")
                self.scaler = StandardScaler()
            
            X_train_final = pd.DataFrame(
                self.scaler.fit_transform(X_train_imputed),
                columns=X_train_imputed.columns,
                index=X_train_imputed.index
            )
            X_test_final = pd.DataFrame(
                self.scaler.transform(X_test_imputed),
                columns=X_test_imputed.columns,
                index=X_test_imputed.index
            )
        else:
            print("  Skipping feature scaling...")
            X_train_final = X_train_imputed
            X_test_final = X_test_imputed
        
        print(f"  Preprocessing completed. Final shape: Train {X_train_final.shape}, Test {X_test_final.shape}")
        
        return X_train_final, X_test_final
    
    def prepare_classification_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare data for classification task."""
        print("\nPreparing classification data...")
        
        # Define outcomes and features
        outcome_columns = [self.config.primary_outcome]
        self.outcome_columns = outcome_columns
        self.feature_columns = [col for col in df.columns if col not in outcome_columns]
        
        # Split features and target
        X = df[self.feature_columns].copy()
        y = df[self.config.primary_outcome]
        
        print(f"Classification dataset: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def run_preprocessing_pipeline(self) -> Dict[str, Any]:
        """Run complete preprocessing pipeline."""
        print("=" * 60)
        print("DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Check data quality
        quality_report = self.check_data_quality(df)
        
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Prepare classification data
        X, y = self.prepare_classification_data(df_clean)
        
        # Split data (stratified)
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        print(f"\nData split:")
        print(f"  Training set: {X_train_raw.shape}")
        print(f"  Test set: {X_test_raw.shape}")
        print(f"  Training class distribution: {np.bincount(y_train_raw)}")
        print(f"  Test class distribution: {np.bincount(y_test_raw)}")
        
        # Preprocess data
        X_train_processed, X_test_processed = self.preprocess_data(
            X_train_raw, X_test_raw, "classification data"
        )
        
        # Save processed data
        self.save_processed_data(X_train_processed, X_test_processed, y_train_raw, y_test_raw)
        
        return {
            'X_train_raw': X_train_raw,
            'X_test_raw': X_test_raw,
            'X_train_processed': X_train_processed,
            'X_test_processed': X_test_processed,
            'y_train_raw': y_train_raw,
            'y_test_raw': y_test_raw,
            'feature_columns': self.feature_columns,
            'quality_report': quality_report
        }
    
    def save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series):
        """Save processed data to disk."""
        processed_dir = os.path.join(self.config.data_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save as CSV
        X_train.to_csv(os.path.join(processed_dir, "X_train.csv"))
        X_test.to_csv(os.path.join(processed_dir, "X_test.csv"))
        y_train.to_csv(os.path.join(processed_dir, "y_train.csv"))
        y_test.to_csv(os.path.join(processed_dir, "y_test.csv"))
        
        print(f"\nProcessed data saved to {processed_dir}/")
    
    def load_processed_data(self) -> Tuple:
        """Load processed data from disk."""
        processed_dir = os.path.join(self.config.data_dir, "processed")
        
        X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"), index_col=0)
        X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"), index_col=0)
        y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"), index_col=0).squeeze()
        y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"), index_col=0).squeeze()
        
        return X_train, X_test, y_train, y_test

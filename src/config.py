"""
Configuration settings for PPROM prediction study
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any
import yaml

@dataclass
class Config:
    """Main configuration class"""
    
    # Paths
    data_path: str = 'data/raw/data23.csv'
    results_dir: str = 'results/ml_results'
    plots_dir: str = 'results/ml_plots'
    models_dir: str = 'models/saved_models'
    logs_dir: str = 'results/logs'
    
    # Random seed
    random_state: int = 42
    
    # Data splitting
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Preprocessing
    scaling_method: str = "maxabs"  # "standard", "maxabs", "none"
    missing_threshold: float = 0.5
    imputation_method: str = "knn"  # "knn", "mean", "median"
    
    # Feature engineering
    feature_selection: str = "all"  # "all", "boruta", "lasso", "rfe"
    boruta_iterations: int = 50
    boruta_alpha: float = 0.2
    pca_variance: float = 0.95
    
    # Class imbalance
    imbalance_method: str = "random_both"
    smote_ratio: float = 0.35
    smote_k_neighbors: int = 5
    
    # Model training
    enable_tuning: bool = True
    tuning_method: str = "random_search"  # "random_search", "bayesian_optuna"
    tuning_iterations: int = 30
    tuning_metric: str = "f1"
    
    # Calibration
    apply_calibration: bool = True
    calibration_method: str = "isotonic"  # "isotonic", "sigmoid", "platt", "none"
    
    # Models to train
    selected_models: List[str] = [
        'Random Forest',
        'Logistic Regression',
        'LightGBM',
        'XGBoost'
    ]
    
    # Evaluation metrics
    metrics: List[str] = [
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
        'specificity', 'npv', 'balanced_accuracy', 'mcc',
        'brier_score', 'ece'
    ]
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

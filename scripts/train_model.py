#!/usr/bin/env python
"""
Script to train models for PPROM prediction study.
Can be used for incremental training or specific model training.
"""

import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def train_models():
    """Train models with command-line options."""
    parser = argparse.ArgumentParser(description='Train ML models for PPROM prediction')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific models to train')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing if data already processed')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory')
    parser.add_argument('--tuning-method', type=str, choices=['random_search', 'bayesian_optuna'],
                       default='random_search', help='Hyperparameter tuning method')
    parser.add_argument('--calibration', type=str, choices=['isotonic', 'sigmoid', 'platt', 'none'],
                       default='isotonic', help='Calibration method')
    
    args = parser.parse_args()
    
    print(f"Training models with options: {args}")
    
    # Import modules
    from src.config import Config
    from src.data_preprocessing import DataPreprocessor
    from src.feature_engineering import FeatureEngineer
    from src.model_training import ModelTrainer
    
    # Load or create configuration
    if args.config:
        config = Config.load_config(args.config)
    else:
        config = Config()
    
    # Override config with command-line arguments
    if args.models:
        config.selected_models = args.models
    
    if args.output_dir:
        config.results_dir = args.output_dir
    
    if args.tuning_method:
        config.tuning_method = args.tuning_method
    
    if args.calibration:
        config.calibration_method = args.calibration
        config.apply_calibration = (args.calibration != 'none')
    
    # Create directories
    config.create_directories()
    
    # Load data
    preprocessor = DataPreprocessor(config)
    
    if args.skip_preprocessing:
        print("Loading preprocessed data...")
        X_train, X_test, y_train, y_test = preprocessor.load_processed_data()
    else:
        print("Running preprocessing pipeline...")
        preprocessed_data = preprocessor.run_preprocessing_pipeline()
        X_train = preprocessed_data['X_train_processed']
        X_test = preprocessed_data['X_test_processed']
        y_train = preprocessed_data['y_train_raw']
        y_test = preprocessed_data['y_test_raw']
    
    # Feature engineering
    feature_engineer = FeatureEngineer(config)
    feature_data = feature_engineer.run_feature_engineering_pipeline(X_train, X_test, y_train)
    
    X_train_final = feature_data['X_train_final']
    X_test_final = feature_data['X_test_final']
    
    # Handle class imbalance
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=config.random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train)
    
    # Train models
    model_trainer = ModelTrainer(config)
    results = model_trainer.train_all_models(
        X_train_balanced, y_train_balanced,
        X_test_final, y_test
    )
    
    print(f"\nTraining completed. {len(results)} models trained.")
    print(f"Results saved to: {config.results_dir}/")


if __name__ == "__main__":
    train_models()

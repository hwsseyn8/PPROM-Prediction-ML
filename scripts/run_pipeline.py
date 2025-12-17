#!/usr/bin/env python3
"""
Main pipeline script for PPROM prediction study
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation_metrics import ModelEvaluator
from src.visualization import ResultsVisualizer
import yaml

def main(config_path=None):
    """Main execution pipeline"""
    
    # Load configuration
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = Config()
    
    # Create directories
    for dir_path in [config.results_dir, config.plots_dir, 
                     config.models_dir, config.logs_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("=== PPROM PREDICTION PIPELINE ===")
    
    # 1. Data preprocessing
    print("\n1. Loading and preprocessing data...")
    preprocessor = DataPreprocessor(config)
    df = preprocessor.load_data(config.data_path)
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df, target_column='pprom'
    )
    X_train_processed, X_test_processed = preprocessor.preprocess(X_train, X_test)
    
    # 2. Feature engineering
    print("\n2. Feature engineering...")
    feature_engineer = FeatureEngineer(config)
    X_train_fe, X_test_fe = feature_engineer.apply_feature_selection(
        X_train_processed, X_test_processed, y_train
    )
    
    # 3. Handle class imbalance
    print("\n3. Handling class imbalance...")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=config.random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_fe, y_train)
    
    # 4. Model training
    print("\n4. Training models...")
    trainer = ModelTrainer(config)
    
    for model_name in config.selected_models:
        print(f"\n  Training {model_name}...")
        results = trainer.train_model(
            model_name, X_train_balanced, y_train_balanced,
            X_test_fe, y_test
        )
        
        # Save model
        model_path = os.path.join(
            config.models_dir, f"{model_name.replace(' ', '_')}.pkl"
        )
        trainer.save_model(model_name, model_path)
        print(f"    Model saved to {model_path}")
    
    # 5. Evaluation
    print("\n5. Evaluating models...")
    evaluator = ModelEvaluator(config)
    evaluation_results = evaluator.evaluate_all(trainer.results, y_test)
    
    # Save evaluation results
    results_path = os.path.join(config.results_dir, "evaluation_results.csv")
    evaluation_results.to_csv(results_path, index=False)
    print(f"    Results saved to {results_path}")
    
    # 6. Visualization
    print("\n6. Creating visualizations...")
    visualizer = ResultsVisualizer(config)
    visualizer.create_all_plots(trainer.results, y_test)
    
    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    
    return trainer.results, evaluation_results

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='PPROM Prediction Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    main(config_path=args.config)

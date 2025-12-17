#!/usr/bin/env python
"""
Main pipeline script for PPROM prediction study.
Runs the complete ML pipeline from data loading to model evaluation.
"""

import sys
import os
import warnings
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="sklearn.*")
warnings.filterwarnings("ignore", module="imblearn.*")

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

def run_complete_pipeline():
    """Run the complete PPROM prediction pipeline."""
    print("=" * 80)
    print("PPROM PREDICTION STUDY - COMPLETE MACHINE LEARNING PIPELINE")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start_time = time.time()
    
    try:
        # Import modules
        from src.config import Config
        from src.data_preprocessing import DataPreprocessor
        from src.feature_engineering import FeatureEngineer
        from src.model_training import ModelTrainer
        from src.evaluation_metrics import Evaluator
        from src.visualization import Visualizer
        from src.ensemble_methods import EnsembleBuilder
        from src.calibration import ModelCalibrator
        
        # 1. Initialize configuration
        print("\n1. INITIALIZING CONFIGURATION")
        print("-" * 40)
        config = Config()
        config.create_directories()
        config.validate_config()
        config.save_config()
        
        # 2. Data Preprocessing
        print("\n2. DATA PREPROCESSING")
        print("-" * 40)
        preprocessor = DataPreprocessor(config)
        preprocessed_data = preprocessor.run_preprocessing_pipeline()
        
        # Extract data
        X_train = preprocessed_data['X_train_processed']
        X_test = preprocessed_data['X_test_processed']
        y_train = preprocessed_data['y_train_raw']
        y_test = preprocessed_data['y_test_raw']
        feature_names = preprocessed_data.get('feature_columns', X_train.columns.tolist())
        
        # 3. Feature Engineering
        print("\n3. FEATURE ENGINEERING")
        print("-" * 40)
        feature_engineer = FeatureEngineer(config)
        feature_data = feature_engineer.run_feature_engineering_pipeline(X_train, X_test, y_train)
        
        X_train_final = feature_data['X_train_final']
        X_test_final = feature_data['X_test_final']
        selected_features = feature_data.get('selected_features', feature_names)
        
        # 4. Handle Class Imbalance
        print("\n4. CLASS IMBALANCE HANDLING")
        print("-" * 40)
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTEENN
        
        imbalance_method = config.class_balance_method
        
        print(f"Applying {imbalance_method} for class imbalance...")
        
        if imbalance_method == "smote":
            sampler = SMOTE(random_state=config.random_state, 
                          sampling_strategy=config.smote_ratio,
                          k_neighbors=config.smote_k_neighbors)
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_final, y_train)
            
        elif imbalance_method == "random_over":
            sampler = RandomOverSampler(random_state=config.random_state,
                                       sampling_strategy=config.oversampling_ratio)
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_final, y_train)
            
        elif imbalance_method == "random_under":
            sampler = RandomUnderSampler(random_state=config.random_state,
                                        sampling_strategy=config.undersampling_ratio)
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_final, y_train)
            
        elif imbalance_method == "smote_enn":
            sampler = SMOTEENN(random_state=config.random_state)
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_final, y_train)
            
        elif imbalance_method == "none":
            X_train_balanced, y_train_balanced = X_train_final, y_train
            
        else:
            print(f"Unknown imbalance method: {imbalance_method}. Using SMOTE.")
            sampler = SMOTE(random_state=config.random_state)
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_final, y_train)
        
        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")
        
        # 5. Model Training
        print("\n5. MODEL TRAINING AND TUNING")
        print("-" * 40)
        model_trainer = ModelTrainer(config)
        model_results = model_trainer.train_all_models(
            X_train_balanced, y_train_balanced,
            X_test_final, y_test
        )
        
        # 6. Ensemble Methods
        print("\n6. ENSEMBLE METHODS")
        print("-" * 40)
        ensemble_builder = EnsembleBuilder(config)
        ensemble_results = ensemble_builder.create_all_ensembles(
            model_results, X_train_balanced, y_train_balanced,
            X_test_final, y_test
        )
        
        # Combine individual and ensemble results
        all_results = {**model_results, **ensemble_results}
        
        # 7. Evaluation and Visualization
        print("\n7. EVALUATION AND VISUALIZATION")
        print("-" * 40)
        
        # Generate comprehensive report
        evaluator = Evaluator(config)
        report = evaluator.generate_comprehensive_report(all_results)
        
        # Generate visualizations
        visualizer = Visualizer(config)
        visualizer.generate_all_visualizations(
            all_results, X_test_final, y_test, selected_features
        )
        
        # 8. Final Summary
        print("\n8. FINAL SUMMARY")
        print("-" * 40)
        
        total_time = time.time() - total_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total time: {hours}h {minutes}m {seconds:.1f}s")
        print(f"Models trained: {len(model_results)}")
        print(f"Ensembles created: {len(ensemble_results)}")
        print(f"Results saved to: {config.results_dir}/")
        print(f"Plots saved to: {config.plots_dir}/")
        print(f"Models saved to: {config.models_dir}/")
        
        # Print best models
        if model_trainer.best_models:
            print("\nBEST MODELS:")
            for metric, best in model_trainer.best_models.items():
                print(f"  {metric}: {best['name']} (score: {best['score']:.4f})")
        
        # Print best ensemble
        best_ensemble_name, best_ensemble_model = ensemble_builder.get_best_ensemble()
        if best_ensemble_name:
            best_ensemble_score = ensemble_results[best_ensemble_name]['f1']
            print(f"\nBEST ENSEMBLE: {best_ensemble_name} (F1: {best_ensemble_score:.4f})")
        
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        return {
            'success': True,
            'results': all_results,
            'config': config,
            'total_time': total_time,
            'best_models': model_trainer.best_models,
            'best_ensemble': (best_ensemble_name, best_ensemble_model)
        }
        
    except Exception as e:
        print(f"\n✗ PIPELINE FAILED WITH ERROR:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Run the pipeline
    results = run_complete_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if results.get('success', False) else 1)

#!/usr/bin/env python
"""
Script to evaluate trained models for PPROM prediction study.
Can be used for model comparison and detailed analysis.
"""

import sys
import os
import argparse
import joblib
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def evaluate_models():
    """Evaluate trained models."""
    parser = argparse.ArgumentParser(description='Evaluate ML models for PPROM prediction')
    
    parser.add_argument('--model-dir', type=str, default='models/saved_models',
                       help='Directory containing trained models')
    parser.add_argument('--test-data', type=str, default='data/processed/X_test.csv',
                       help='Path to test data')
    parser.add_argument('--test-labels', type=str, default='data/processed/y_test.csv',
                       help='Path to test labels')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate evaluation plots')
    
    args = parser.parse_args()
    
    print(f"Evaluating models from: {args.model_dir}")
    
    # Load test data
    X_test = pd.read_csv(args.test_data, index_col=0)
    y_test = pd.read_csv(args.test_labels, index_col=0).squeeze()
    
    # Load all models
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("No model files found!")
        return
    
    print(f"Found {len(model_files)} model files")
    
    # Import evaluation modules
    from src.config import Config
    from src.evaluation_metrics import Evaluator
    from src.visualization import Visualizer
    
    config = Config()
    if args.output_dir:
        config.results_dir = args.output_dir
        config.plots_dir = os.path.join(args.output_dir, 'plots')
    
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)
    
    evaluator = Evaluator(config)
    visualizer = Visualizer(config)
    
    results = {}
    
    # Evaluate each model
    for model_file in model_files:
        try:
            model_path = os.path.join(args.model_dir, model_file)
            model = joblib.load(model_path)
            model_name = model_file.replace('.pkl', '').replace('_', ' ')
            
            print(f"\nEvaluating: {model_name}")
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                if y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = y_pred_proba.ravel()
                y_pred = (y_pred_proba >= 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = y_pred.astype(float)
            
            # Calculate metrics
            metrics = evaluator.calculate_all_metrics(y_test, y_pred, y_pred_proba)
            
            # Store results
            results[model_name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                **metrics
            }
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"  Error evaluating {model_file}: {e}")
    
    # Generate comprehensive report
    if results:
        print("\nGenerating evaluation report...")
        report = evaluator.generate_comprehensive_report(results)
        
        # Generate plots if requested
        if args.generate_plots:
            print("\nGenerating evaluation plots...")
            visualizer.plot_roc_curves(results)
            visualizer.plot_performance_comparison(results)
            
            # Extract feature names for importance plots
            feature_names = X_test.columns.tolist()
            
            for model_name, result in results.items():
                if 'model' in result:
                    visualizer.plot_feature_importance(
                        result['model'], feature_names, model_name
                    )
        
        print(f"\nEvaluation completed. Results saved to: {config.results_dir}/")
    
    return results


if __name__ == "__main__":
    results = evaluate_models()

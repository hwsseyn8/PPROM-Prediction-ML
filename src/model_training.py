"""
Model training module for PPROM prediction study.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import os
import time
from typing import Dict, Any, List, Tuple
import joblib

from .config import config
from .calibration import ModelCalibrator


class ModelTrainer:
    """Handles model training, hyperparameter tuning, and calibration."""
    
    def __init__(self, config=config):
        self.config = config
        self.models = {}
        self.results = {}
        self.best_models = {}
        
    def get_model_definitions(self) -> Dict[str, Dict]:
        """Get model definitions and parameter grids."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
        
        model_definitions = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=self.config.random_state, max_iter=1000),
                'params': self.config.model_params.get('Logistic Regression', {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': ['balanced', None]
                })
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=self.config.random_state, n_jobs=-1),
                'params': self.config.model_params.get('Random Forest', {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                })
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=self.config.random_state, verbose=-1, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0, 0.1, 1]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=self.config.random_state, eval_metric='logloss', verbosity=0),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'gamma': [0, 0.1, 0.3],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0, 0.5, 1]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=self.config.random_state),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'class_weight': ['balanced', None]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=self.config.random_state),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.7, 0.8, 0.9]
                }
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(n_jobs=-1),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan'],
                    'p': [1, 2]
                }
            },
            'ANN': {
                'model': MLPClassifier(random_state=self.config.random_state, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'learning_rate_init': [0.001, 0.01]
                }
            },
            'BalancedRandomForest': {
                'model': BalancedRandomForestClassifier(random_state=self.config.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'sampling_strategy': ['auto', 0.5, 0.7]
                }
            },
            'EasyEnsemble': {
                'model': EasyEnsembleClassifier(random_state=self.config.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [10, 20, 50],
                    'sampling_strategy': ['auto', 0.5, 0.7]
                }
            }
        }
        
        return model_definitions
    
    def tune_hyperparameters(self, model, params, X_train, y_train, 
                            model_name: str) -> Dict[str, Any]:
        """Tune hyperparameters using specified method."""
        print(f"  Tuning {model_name} with {self.config.tuning_method}...")
        
        start_time = time.time()
        
        if self.config.tuning_method == "random_search":
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=self.config.tuning_iterations,
                cv=StratifiedKFold(n_splits=self.config.tuning_cv_folds, 
                                 shuffle=True, random_state=self.config.random_state),
                scoring=self.config.tuning_scoring,
                n_jobs=-1,
                random_state=self.config.random_state,
                verbose=0
            )
            
            search.fit(X_train, y_train)
            best_params = search.best_params_
            best_score = search.best_score_
            
            print(f"    Best {self.config.tuning_scoring}: {best_score:.4f}")
            print(f"    Best parameters: {best_params}")
            
        elif self.config.tuning_method == "bayesian_optuna":
            # Bayesian optimization with Optuna
            best_params, best_score = self._bayesian_optimization(
                model, params, X_train, y_train, model_name
            )
        
        else:
            raise ValueError(f"Unknown tuning method: {self.config.tuning_method}")
        
        tuning_time = time.time() - start_time
        print(f"    Tuning completed in {tuning_time:.2f} seconds")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'tuning_time': tuning_time
        }
    
    def _bayesian_optimization(self, model, params, X_train, y_train, model_name):
        """Perform Bayesian optimization using Optuna."""
        try:
            import optuna
            from optuna.samplers import TPESampler
            
            print(f"    Starting Bayesian optimization with Optuna...")
            
            def objective(trial):
                # Create parameter suggestions based on model type
                param_dict = {}
                
                if 'Logistic Regression' in model_name:
                    param_dict['C'] = trial.suggest_float('C', 0.001, 100, log=True)
                    param_dict['penalty'] = trial.suggest_categorical('penalty', ['l1', 'l2'])
                    param_dict['solver'] = trial.suggest_categorical('solver', ['liblinear', 'saga'])
                    
                elif 'Random Forest' in model_name:
                    param_dict['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
                    param_dict['max_depth'] = trial.suggest_int('max_depth', 3, 15)
                    param_dict['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
                    param_dict['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
                    
                elif 'LightGBM' in model_name or 'XGBoost' in model_name:
                    param_dict['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
                    param_dict['max_depth'] = trial.suggest_int('max_depth', 3, 12)
                    param_dict['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                    param_dict['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
                    
                else:
                    # Generic parameter suggestions
                    for param_name, param_values in params.items():
                        if isinstance(param_values, list):
                            param_dict[param_name] = trial.suggest_categorical(param_name, param_values)
                
                # Set parameters and evaluate
                model_copy = model.__class__(**model.get_params())
                model_copy.set_params(**param_dict)
                
                cv_scores = cross_val_score(
                    model_copy, X_train, y_train,
                    cv=min(5, self.config.tuning_cv_folds),
                    scoring=self.config.tuning_scoring,
                    n_jobs=1
                )
                
                return cv_scores.mean()
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config.random_state)
            )
            
            # Optimize
            study.optimize(objective, n_trials=self.config.tuning_iterations)
            
            return study.best_params, study.best_value
            
        except ImportError:
            print("    Optuna not installed. Falling back to RandomizedSearchCV.")
            # Fallback to random search
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=self.config.tuning_iterations,
                cv=self.config.tuning_cv_folds,
                scoring=self.config.tuning_scoring,
                n_jobs=-1,
                random_state=self.config.random_state,
                verbose=0
            )
            
            search.fit(X_train, y_train)
            return search.best_params, search.best_score
    
    def train_final_model(self, model, best_params, X_train, y_train):
        """Train final model with best parameters."""
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict[str, Any]:
        """Evaluate model performance."""
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = y_pred  # For models without probability
        
        # Calculate metrics
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, confusion_matrix, 
                                   average_precision_score, balanced_accuracy_score,
                                   matthews_corrcoef)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.5
        
        try:
            avg_precision = average_precision_score(y_test, y_pred_proba)
        except:
            avg_precision = 0
        
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        else:
            specificity = 0
            npv = 0
        
        # ROC curve data
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        # Precision-recall curve data
        from sklearn.metrics import precision_recall_curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        
        return {
            'model_name': model_name,
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'balanced_accuracy': balanced_acc,
            'mcc': mcc,
            'npv': npv,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve
        }
    
    def run_cross_validation(self, model, X_train, y_train, model_name: str) -> Dict[str, Any]:
        """Run cross-validation on training data."""
        print(f"  Running {self.config.training_cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=self.config.training_cv_folds, 
                           shuffle=True, random_state=self.config.random_state)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Clone and train model
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Evaluate
            if hasattr(fold_model, 'predict_proba'):
                y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)
            else:
                y_pred = fold_model.predict(X_fold_val)
                y_pred_proba = y_pred
            
            cv_scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
            cv_scores['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_fold_val, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_fold_val, y_pred, zero_division=0))
            
            try:
                cv_scores['roc_auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
            except:
                cv_scores['roc_auc'].append(0.5)
        
        # Calculate mean and std
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
            cv_results[f'{metric}_scores'] = scores
        
        print(f"    CV Results: Accuracy={cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}, "
              f"F1={cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        
        return cv_results
    
    def save_model(self, model, model_name: str):
        """Save trained model to disk."""
        model_path = os.path.join(self.config.models_dir, f"{model_name.replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        print(f"    Model saved to {model_path}")
        
        # Also save model info
        model_info = {
            'model_name': model_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': {k: v for k, v in self.config.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        }
        
        info_path = os.path.join(self.config.models_dir, f"{model_name.replace(' ', '_')}_info.json")
        import json
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train all selected models."""
        print("=" * 60)
        print("MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        model_definitions = self.get_model_definitions()
        self.results = {}
        
        # Filter selected models
        models_to_train = {
            name: config for name, config in model_definitions.items() 
            if name in self.config.selected_models
        }
        
        print(f"Training {len(models_to_train)} models:")
        for name in models_to_train.keys():
            print(f"  - {name}")
        
        # Initialize calibrator
        calibrator = ModelCalibrator(self.config)
        
        for model_name, model_config in models_to_train.items():
            print(f"\n{'='*40}")
            print(f"Training: {model_name}")
            print(f"{'='*40}")
            
            try:
                # Get model and parameters
                model = model_config['model']
                params = model_config['params']
                
                # Tune hyperparameters
                tuning_results = self.tune_hyperparameters(model, params, X_train, y_train, model_name)
                
                # Train final model with best parameters
                final_model = self.train_final_model(
                    model.__class__(**model.get_params()),
                    tuning_results['best_params'],
                    X_train, y_train
                )
                
                # Apply calibration if requested
                if self.config.apply_calibration:
                    print(f"  Applying {self.config.calibration_method} calibration...")
                    calibrated_model = calibrator.calibrate_model(final_model, X_train, y_train)
                else:
                    calibrated_model = final_model
                
                # Run cross-validation
                cv_results = self.run_cross_validation(calibrated_model, X_train, y_train, model_name)
                
                # Evaluate on test set
                eval_results = self.evaluate_model(calibrated_model, X_test, y_test, model_name)
                
                # Combine results
                model_result = {
                    **eval_results,
                    'tuning_results': tuning_results,
                    'cv_results': cv_results,
                    'calibration_method': self.config.calibration_method if self.config.apply_calibration else 'none',
                    'best_params': tuning_results['best_params']
                }
                
                self.results[model_name] = model_result
                
                # Save model
                self.save_model(calibrated_model, model_name)
                
                print(f"✓ {model_name} completed successfully")
                print(f"  Test Performance: Accuracy={eval_results['accuracy']:.4f}, "
                      f"F1={eval_results['f1']:.4f}, AUC={eval_results['roc_auc']:.4f}")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save all results
        self.save_all_results()
        
        # Find best model
        self._identify_best_model()
        
        return self.results
    
    def _identify_best_model(self):
        """Identify the best model based on F1 score."""
        if not self.results:
            return
        
        best_model_name = max(self.results.items(), 
                            key=lambda x: x[1]['f1'])[0]
        
        best_model_score = self.results[best_model_name]['f1']
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"F1 Score: {best_model_score:.4f}")
        print(f"{'='*60}")
        
        self.best_models['best_by_f1'] = {
            'name': best_model_name,
            'score': best_model_score,
            'model': self.results[best_model_name]['model']
        }
        
        # Also identify by AUC
        best_model_auc = max(self.results.items(), 
                           key=lambda x: x[1]['roc_auc'])[0]
        best_auc_score = self.results[best_model_auc]['roc_auc']
        
        self.best_models['best_by_auc'] = {
            'name': best_model_auc,
            'score': best_auc_score,
            'model': self.results[best_model_auc]['model']
        }
    
    def save_all_results(self):
        """Save all model results to disk."""
        import json
        import pandas as pd
        
        # Save detailed results as JSON
        results_dict = {}
        for model_name, result in self.results.items():
            # Convert numpy arrays to lists for JSON serialization
            serializable_result = {}
            for key, value in result.items():
                if hasattr(value, 'tolist'):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                else:
                    serializable_result[key] = value
            
            results_dict[model_name] = serializable_result
        
        results_path = os.path.join(self.config.results_dir, "all_model_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save summary as CSV
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'Specificity': f"{result['specificity']:.4f}",
                'F1_Score': f"{result['f1']:.4f}",
                'ROC_AUC': f"{result['roc_auc']:.4f}",
                'Avg_Precision': f"{result['avg_precision']:.4f}",
                'Balanced_Accuracy': f"{result['balanced_accuracy']:.4f}",
                'MCC': f"{result['mcc']:.4f}",
                'NPV': f"{result['npv']:.4f}",
                'CV_Accuracy': f"{result['cv_results']['accuracy_mean']:.4f}",
                'CV_F1': f"{result['cv_results']['f1_mean']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('F1_Score', ascending=False)
        
        summary_path = os.path.join(self.config.results_dir, "model_performance_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n✓ All results saved to {self.config.results_dir}/")
        print(f"  - Detailed results: all_model_results.json")
        print(f"  - Performance summary: model_performance_summary.csv")

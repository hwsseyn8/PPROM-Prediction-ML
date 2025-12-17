"""
Model training and hyperparameter tuning module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix)
import optuna
from optuna.samplers import TPESampler
import joblib
import warnings
warnings.filterwarnings('ignore')

from .calibration import ProperCalibratedModel

class ModelTrainer:
    """Handles model training and hyperparameter optimization"""
    
    def __init__(self, config):
        self.config = config
        self.models = self._get_model_definitions()
        self.results = {}
        
    def _get_model_definitions(self):
        """Define models and their hyperparameter spaces"""
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(
                    random_state=self.config.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                    random_state=self.config.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            # Add other model definitions...
        }
        return models
    
    def train_model(self, model_name, X_train, y_train, X_test, y_test):
        """Train a single model with hyperparameter tuning"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not defined")
        
        model_config = self.models[model_name]
        
        # Hyperparameter tuning
        if self.config.enable_tuning:
            best_params = self._tune_hyperparameters(
                model_config, X_train, y_train
            )
        else:
            best_params = {}
        
        # Train model with best parameters
        final_model = model_config['model'].set_params(**best_params)
        final_model.fit(X_train, y_train)
        
        # Apply calibration if requested
        if self.config.apply_calibration:
            calibrated_model = ProperCalibratedModel(
                base_model=final_model,
                calibration_method=self.config.calibration_method,
                cv_folds=self.config.cv_folds,
                random_state=self.config.random_state
            )
            calibrated_model.fit(X_train, y_train)
            model_to_use = calibrated_model
        else:
            model_to_use = final_model
        
        # Evaluate model
        results = self._evaluate_model(
            model_to_use, X_test, y_test, model_name
        )
        
        # Store results
        results['best_params'] = best_params
        results['model'] = model_to_use
        
        self.results[model_name] = results
        
        return results
    
    def _tune_hyperparameters(self, model_config, X_train, y_train):
        """Perform hyperparameter tuning"""
        if self.config.tuning_method == "bayesian_optuna":
            return self._bayesian_optimization(
                model_config, X_train, y_train
            )
        else:  # random_search
            return self._random_search(
                model_config, X_train, y_train
            )
    
    def _random_search(self, model_config, X_train, y_train):
        """Randomized search for hyperparameters"""
        random_search = RandomizedSearchCV(
            model_config['model'],
            model_config['params'],
            n_iter=self.config.tuning_iterations,
            cv=self.config.cv_folds,
            scoring=self.config.tuning_metric,
            n_jobs=-1,
            random_state=self.config.random_state,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        return random_search.best_params_
    
    def _bayesian_optimization(self, model_config, X_train, y_train):
        """Bayesian optimization with Optuna"""
        def objective(trial):
            # Define parameter space
            params = {}
            # Add parameter suggestions based on model type
            
            # Create model with suggested parameters
            model = model_config['model'].set_params(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config.cv_folds,
                scoring=self.config.tuning_metric,
                n_jobs=-1
            )
            return cv_scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.random_state)
        )
        study.optimize(objective, n_trials=self.config.tuning_iterations)
        
        return study.best_params
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def save_model(self, model_name, path):
        """Save trained model"""
        if model_name in self.results:
            joblib.dump(self.results[model_name]['model'], path)
        else:
            raise ValueError(f"Model {model_name} not found in results")
    
    def load_model(self, model_name, path):
        """Load trained model"""
        model = joblib.load(path)
        self.results[model_name] = {'model': model}
        return model

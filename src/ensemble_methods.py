"""
Ensemble methods module for PPROM prediction study.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
from typing import Dict, List, Tuple, Any
import os

from .config import config


class EnsembleBuilder:
    """Builds and evaluates ensemble models."""
    
    def __init__(self, config=config):
        self.config = config
        self.ensembles = {}
        self.ensemble_results = {}
    
    def create_voting_ensemble(self, models: Dict[str, object], 
                              voting: str = 'soft', weights: List[float] = None) -> VotingClassifier:
        """Create a voting ensemble from multiple models."""
        print(f"Creating {voting} voting ensemble...")
        
        # Create estimator list
        estimators = [(name, model) for name, model in models.items()]
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
        
        return voting_clf
    
    def create_stacking_ensemble(self, base_models: Dict[str, object],
                                meta_model: object = None,
                                cv_folds: int = 5) -> StackingClassifier:
        """Create a stacking ensemble."""
        print("Creating stacking ensemble...")
        
        if meta_model is None:
            meta_model = LogisticRegression(
                random_state=self.config.random_state,
                class_weight='balanced',
                max_iter=1000
            )
        
        # Create estimator list
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=cv_folds,
            passthrough=True,
            n_jobs=-1
        )
        
        return stacking_clf
    
    def create_weighted_ensemble(self, models: Dict[str, object], 
                                weights: Dict[str, float]) -> object:
        """Create a weighted ensemble based on model performance."""
        print("Creating weighted ensemble...")
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Create a custom ensemble class
        class WeightedEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
                self.classes_ = None
            
            def fit(self, X, y):
                for name, model in self.models.items():
                    model.fit(X, y)
                self.classes_ = np.unique(y)
                return self
            
            def predict_proba(self, X):
                probas = []
                for name, model in self.models.items():
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        if proba.shape[1] > 1:
                            proba = proba[:, 1]
                        else:
                            proba = proba.ravel()
                    else:
                        # For models without predict_proba
                        preds = model.predict(X)
                        proba = preds.astype(float)
                    
                    probas.append(proba * self.weights[name])
                
                weighted_proba = np.sum(probas, axis=0)
                weighted_proba = np.clip(weighted_proba, 0, 1)
                
                # Return probabilities for both classes
                return np.column_stack([1 - weighted_proba, weighted_proba])
            
            def predict(self, X, threshold=0.5):
                proba = self.predict_proba(X)[:, 1]
                return (proba >= threshold).astype(int)
        
        return WeightedEnsemble(models, normalized_weights)
    
    def select_best_models(self, results: Dict[str, Dict], 
                          metric: str = 'f1', 
                          top_k: int = 3) -> Dict[str, object]:
        """Select top-k models based on specified metric."""
        print(f"Selecting top {top_k} models by {metric}...")
        
        # Sort models by metric
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )[:top_k]
        
        selected_models = {}
        for model_name, result in sorted_models:
            if 'model' in result:
                selected_models[model_name] = result['model']
                print(f"  {model_name}: {result.get(metric, 0):.4f}")
        
        return selected_models
    
    def evaluate_ensemble(self, ensemble, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray, 
                         ensemble_name: str) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                   f1_score, roc_auc_score, confusion_matrix)
        
        print(f"Evaluating {ensemble_name}...")
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Make predictions
        if hasattr(ensemble, 'predict_proba'):
            y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            y_pred = ensemble.predict(X_test)
            y_pred_proba = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve data
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        result = {
            'model_name': ensemble_name,
            'model': ensemble,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'fpr': fpr,
            'tpr': tpr
        }
        
        print(f"  {ensemble_name} Performance:")
        print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
        
        return result
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0
        return 0
    
    def create_all_ensembles(self, results: Dict[str, Dict], 
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Create and evaluate all ensemble methods."""
        print("\n" + "=" * 60)
        print("ENSEMBLE MODEL CONSTRUCTION")
        print("=" * 60)
        
        self.ensemble_results = {}
        
        # Get base models
        base_models = {}
        for model_name, result in results.items():
            if 'model' in result:
                base_models[model_name] = result['model']
        
        if len(base_models) < 2:
            print("  Need at least 2 models to create ensembles.")
            return self.ensemble_results
        
        # Select best models for ensembles
        best_models = self.select_best_models(results, metric='f1', top_k=3)
        
        if len(best_models) >= 2:
            # 1. Soft Voting Ensemble
            try:
                soft_voting = self.create_voting_ensemble(best_models, voting='soft')
                soft_voting_result = self.evaluate_ensemble(
                    soft_voting, X_train, y_train, X_test, y_test, 'Soft_Voting'
                )
                self.ensemble_results['Soft_Voting'] = soft_voting_result
                self.ensembles['Soft_Voting'] = soft_voting
            except Exception as e:
                print(f"  Soft voting failed: {e}")
            
            # 2. Hard Voting Ensemble
            try:
                hard_voting = self.create_voting_ensemble(best_models, voting='hard')
                hard_voting_result = self.evaluate_ensemble(
                    hard_voting, X_train, y_train, X_test, y_test, 'Hard_Voting'
                )
                self.ensemble_results['Hard_Voting'] = hard_voting_result
                self.ensembles['Hard_Voting'] = hard_voting
            except Exception as e:
                print(f"  Hard voting failed: {e}")
            
            # 3. Stacking Ensemble
            try:
                stacking = self.create_stacking_ensemble(best_models)
                stacking_result = self.evaluate_ensemble(
                    stacking, X_train, y_train, X_test, y_test, 'Stacking'
                )
                self.ensemble_results['Stacking'] = stacking_result
                self.ensembles['Stacking'] = stacking
            except Exception as e:
                print(f"  Stacking failed: {e}")
            
            # 4. Weighted Ensemble (weight by F1 score)
            try:
                weights = {}
                for model_name in best_models.keys():
                    if model_name in results:
                        weights[model_name] = results[model_name].get('f1', 0.5)
                
                weighted_ensemble = self.create_weighted_ensemble(best_models, weights)
                weighted_result = self.evaluate_ensemble(
                    weighted_ensemble, X_train, y_train, X_test, y_test, 'Weighted'
                )
                self.ensemble_results['Weighted'] = weighted_result
                self.ensembles['Weighted'] = weighted_ensemble
            except Exception as e:
                print(f"  Weighted ensemble failed: {e}")
        
        # Compare ensemble vs individual models
        self._compare_ensembles_with_individuals(results)
        
        # Save ensemble results
        self._save_ensemble_results()
        
        return self.ensemble_results
    
    def _compare_ensembles_with_individuals(self, individual_results: Dict[str, Dict]):
        """Compare ensemble performance with individual models."""
        print("\nEnsemble vs Individual Model Comparison:")
        print("-" * 50)
        print(f"{'Model':<20} {'F1 Score':<10} {'ROC AUC':<10}")
        print("-" * 50)
        
        # Individual models
        for model_name, result in individual_results.items():
            f1 = result.get('f1', 0)
            auc = result.get('roc_auc', 0.5)
            print(f"{model_name:<20} {f1:<10.4f} {auc:<10.4f}")
        
        # Ensemble models
        for ensemble_name, result in self.ensemble_results.items():
            f1 = result.get('f1', 0)
            auc = result.get('roc_auc', 0.5)
            print(f"{ensemble_name:<20} {f1:<10.4f} {auc:<10.4f}")
    
    def _save_ensemble_results(self):
        """Save ensemble results to disk."""
        import json
        import pandas as pd
        
        if not self.ensemble_results:
            return
        
        # Save detailed results as JSON
        results_dict = {}
        for ensemble_name, result in self.ensemble_results.items():
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
            
            results_dict[ensemble_name] = serializable_result
        
        results_path = os.path.join(self.config.results_dir, "ensemble_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save ensemble models
        for ensemble_name, ensemble in self.ensembles.items():
            import joblib
            model_path = os.path.join(self.config.models_dir, 
                                    f"ensemble_{ensemble_name.replace(' ', '_')}.pkl")
            joblib.dump(ensemble, model_path)
        
        print(f"\n✓ Ensemble results saved to {self.config.results_dir}/")
        print(f"✓ Ensemble models saved to {self.config.models_dir}/")
    
    def get_best_ensemble(self, metric: str = 'f1') -> Tuple[str, object]:
        """Get the best performing ensemble."""
        if not self.ensemble_results:
            return None, None
        
        best_ensemble = max(
            self.ensemble_results.items(),
            key=lambda x: x[1].get(metric, 0)
        )
        
        return best_ensemble[0], best_ensemble[1]['model']

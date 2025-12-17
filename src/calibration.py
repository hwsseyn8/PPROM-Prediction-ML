"""
Model calibration module for PPROM prediction study.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import warnings
from typing import Optional
import joblib

from .config import config


class ModelCalibrator:
    """Handles model calibration methods."""
    
    def __init__(self, config=config):
        self.config = config
        self.calibrators = {}
    
    def calibrate_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                       method: Optional[str] = None) -> object:
        """Calibrate a model using specified method."""
        if method is None:
            method = self.config.calibration_method
        
        if method == 'none':
            return model
        
        print(f"    Calibrating model with {method} method...")
        
        try:
            if method in ['isotonic', 'sigmoid']:
                calibrated_model = CalibratedClassifierCV(
                    estimator=model,
                    method=method,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, 
                                      random_state=self.config.random_state),
                    n_jobs=-1
                )
                calibrated_model.fit(X_train, y_train)
                
            elif method == 'platt':
                # Platt scaling (logistic regression on model outputs)
                if hasattr(model, 'predict_proba'):
                    y_train_scores = model.predict_proba(X_train)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_train_scores = model.decision_function(X_train)
                else:
                    print("      Model doesn't support Platt scaling. Skipping calibration.")
                    return model
                
                # Fit logistic regression to scores
                platt_scaler = LogisticRegression()
                platt_scaler.fit(y_train_scores.reshape(-1, 1), y_train)
                
                # Create a wrapper class for Platt-scaled model
                calibrated_model = PlattCalibratedModel(model, platt_scaler)
                
            else:
                print(f"      Unknown calibration method: {method}. Skipping.")
                return model
            
            self.calibrators[type(model).__name__] = calibrated_model
            return calibrated_model
            
        except Exception as e:
            print(f"      Calibration failed: {e}. Using uncalibrated model.")
            return model
    
    def evaluate_calibration(self, model, X_test: np.ndarray, y_test: np.ndarray,
                           method: str = 'isotonic', n_bins: int = 10) -> dict:
        """Evaluate model calibration."""
        from sklearn.metrics import brier_score_loss
        
        calibration_results = {}
        
        # Get predicted probabilities
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            print("      Model doesn't have predict_proba method.")
            return calibration_results
        
        # Calculate Brier score
        brier = brier_score_loss(y_test, y_pred_proba)
        calibration_results['brier_score'] = brier
        
        # Calculate calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, 
                                                   n_bins=n_bins, strategy='uniform')
            
            # Calculate Expected Calibration Error (ECE)
            ece = np.mean(np.abs(prob_true - prob_pred))
            calibration_results['ece'] = ece
            calibration_results['calibration_curve'] = (prob_true, prob_pred)
            
            print(f"      Brier score: {brier:.4f}, ECE: {ece:.4f}")
            
        except Exception as e:
            print(f"      Could not calculate calibration curve: {e}")
        
        return calibration_results
    
    def plot_calibration_comparison(self, models: dict, X_test: np.ndarray, 
                                   y_test: np.ndarray, save_path: str):
        """Plot calibration comparison for multiple models."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        
        for model_name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                try:
                    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, 
                                                           n_bins=10, strategy='uniform')
                    
                    brier = brier_score_loss(y_test, y_pred_proba)
                    
                    plt.plot(prob_pred, prob_true, 's-', label=f'{model_name} (Brier={brier:.3f})')
                except:
                    continue
        
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Comparison', fontweight='bold')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_calibrator(self, model_name: str, calibrator: object):
        """Save calibrator to disk."""
        import os
        
        calibrator_path = os.path.join(self.config.models_dir, 
                                      f'{model_name}_calibrator.pkl')
        joblib.dump(calibrator, calibrator_path)
        print(f"      Calibrator saved to {calibrator_path}")


class PlattCalibratedModel:
    """Wrapper class for Platt-scaled models."""
    
    def __init__(self, base_model, platt_scaler):
        self.base_model = base_model
        self.platt_scaler = platt_scaler
    
    def predict_proba(self, X):
        """Get Platt-scaled probabilities."""
        if hasattr(self.base_model, 'predict_proba'):
            base_proba = self.base_model.predict_proba(X)
            if base_proba.shape[1] > 1:
                base_scores = base_proba[:, 1]
            else:
                base_scores = base_proba.ravel()
        elif hasattr(self.base_model, 'decision_function'):
            base_scores = self.base_model.decision_function(X)
        else:
            raise ValueError("Base model doesn't support probability estimation")
        
        # Apply Platt scaling
        scaled_proba = self.platt_scaler.predict_proba(base_scores.reshape(-1, 1))
        
        # Return probabilities for both classes
        return scaled_proba
    
    def predict(self, X, threshold=0.5):
        """Get predictions with threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def __getattr__(self, name):
        """Delegate other attributes to base model."""
        return getattr(self.base_model, name)

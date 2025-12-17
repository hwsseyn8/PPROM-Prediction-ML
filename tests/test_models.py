"""
Tests for model training module.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.model_training import ModelTrainer
from src.config import Config


class TestModelTraining:
    """Test model training functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        X_train = pd.DataFrame(X[:80], columns=[f'feature_{i}' for i in range(n_features)])
        X_test = pd.DataFrame(X[80:], columns=[f'feature_{i}' for i in range(n_features)])
        y_train = pd.Series(y[:80])
        y_test = pd.Series(y[80:])
        
        return X_train, X_test, y_train, y_test
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        config.selected_models = ['Logistic Regression', 'Random Forest']
        config.tuning_iterations = 5  # Small for testing
        config.tuning_cv_folds = 3
        config.training_cv_folds = 3
        config.random_state = 42
        config.enable_fine_tuning = False  # Disable for faster tests
        return config
    
    @pytest.fixture
    def trainer(self, config):
        """Create trainer instance."""
        return ModelTrainer(config)
    
    def test_get_model_definitions(self, trainer):
        """Test model definitions loading."""
        definitions = trainer.get_model_definitions()
        
        assert isinstance(definitions, dict)
        assert 'Logistic Regression' in definitions
        assert 'Random Forest' in definitions
        assert 'model' in definitions['Logistic Regression']
        assert 'params' in definitions['Logistic Regression']
    
    def test_train_all_models(self, trainer, sample_data):
        """Test model training pipeline."""
        X_train, X_test, y_train, y_test = sample_data
        
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        assert isinstance(results, dict)
        assert len(results) <= len(trainer.config.selected_models)
        
        for model_name, result in results.items():
            assert 'accuracy' in result
            assert 'f1' in result
            assert 'roc_auc' in result
            assert 'model' in result
            assert result['accuracy'] >= 0 and result['accuracy'] <= 1
    
    def test_evaluate_model(self, trainer, sample_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train a simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        result = trainer.evaluate_model(model, X_test, y_test, 'Test Model')
        
        assert 'accuracy' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
        assert 'roc_auc' in result
        assert 'model_name' in result
        
        # Check metrics are valid
        assert 0 <= result['accuracy'] <= 1
        assert 0 <= result['f1'] <= 1
        assert 0 <= result['roc_auc'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

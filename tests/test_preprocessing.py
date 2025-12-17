"""
Tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_preprocessing import DataPreprocessor
from src.config import Config


class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = {
            'age': [25, 30, 35, 40, 45, np.nan],
            'bmi': [22.5, 24.0, 26.5, 23.0, 25.5, 27.0],
            'outcome': [0, 1, 0, 1, 0, 1],
            'category': ['A', 'B', 'A', 'B', 'A', 'B']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        config.primary_outcome = 'outcome'
        config.missing_threshold = 0.5
        config.test_size = 0.3
        config.random_state = 42
        return config
    
    @pytest.fixture
    def preprocessor(self, config):
        """Create preprocessor instance."""
        return DataPreprocessor(config)
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling."""
        df_clean = preprocessor.handle_missing_values(sample_data)
        
        # Should remove rows with >50% missing
        assert len(df_clean) == len(sample_data)
        
        # Test with more missing values
        df_many_missing = sample_data.copy()
        df_many_missing['new_col'] = [np.nan] * 6
        df_clean2 = preprocessor.handle_missing_values(df_many_missing)
        assert len(df_clean2) == len(df_many_missing)
    
    def test_safe_label_encode(self, preprocessor):
        """Test safe label encoding."""
        train_series = pd.Series(['A', 'B', 'C'])
        test_series = pd.Series(['A', 'D', 'B'])  # D not in train
        
        train_encoded, test_encoded = preprocessor.safe_label_encode(
            train_series, test_series
        )
        
        # All unique values should be encoded
        assert len(set(train_encoded.unique())) == 4  # A, B, C, D
        assert len(set(test_encoded.unique())) == 3   # A, D, B
        
        # Same values should have same encoding
        assert train_encoded.iloc[0] == test_encoded.iloc[0]  # Both A
    
    def test_prepare_classification_data(self, preprocessor, sample_data):
        """Test data preparation for classification."""
        X, y = preprocessor.prepare_classification_data(sample_data)
        
        assert X.shape[1] == 3  # age, bmi, category
        assert len(y) == len(sample_data)
        assert preprocessor.config.primary_outcome not in X.columns
        assert preprocessor.feature_columns == ['age', 'bmi', 'category']
        assert preprocessor.outcome_columns == [preprocessor.config.primary_outcome]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

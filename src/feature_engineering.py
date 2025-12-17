"""
Feature engineering and selection module
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression, LassoCV
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineer:
    """Handles feature engineering and selection"""
    
    def __init__(self, config):
        self.config = config
        self.transformers = {}
        self.selected_features = None
        
    def apply_feature_selection(self, X_train, X_test, y_train, method=None):
        """Apply feature selection method"""
        method = method or self.config.feature_selection
        
        if method == "boruta":
            return self._boruta_selection(X_train, X_test, y_train)
        elif method == "lasso":
            return self._lasso_selection(X_train, X_test, y_train)
        elif method.startswith("rfe"):
            n_features = int(method[3:])
            return self._rfe_selection(X_train, X_test, y_train, n_features)
        else:
            return X_train, X_test
    
    def _boruta_selection(self, X_train, X_test, y_train):
        """Apply Boruta feature selection"""
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config.random_state,
            class_weight='balanced'
        )
        
        boruta_selector = BorutaPy(
            estimator=rf,
            n_estimators='auto',
            random_state=self.config.random_state,
            max_iter=self.config.boruta_iterations,
            alpha=self.config.boruta_alpha
        )
        
        boruta_selector.fit(X_train.values, y_train.values)
        selected_mask = boruta_selector.support_
        
        X_train_selected = X_train.iloc[:, selected_mask]
        X_test_selected = X_test.iloc[:, selected_mask]
        
        self.transformers['boruta'] = boruta_selector
        self.selected_features = X_train_selected.columns.tolist()
        
        return X_train_selected, X_test_selected
    
    def _lasso_selection(self, X_train, X_test, y_train):
        """Apply Lasso feature selection"""
        lasso = LassoCV(cv=5, random_state=self.config.random_state)
        lasso.fit(X_train, y_train)
        
        selector = SelectFromModel(lasso, threshold='mean')
        selector.fit(X_train, y_train)
        
        selected_mask = selector.get_support()
        X_train_selected = X_train.iloc[:, selected_mask]
        X_test_selected = X_test.iloc[:, selected_mask]
        
        self.transformers['lasso'] = selector
        self.selected_features = X_train_selected.columns.tolist()
        
        return X_train_selected, X_test_selected
    
    def _rfe_selection(self, X_train, X_test, y_train, n_features):
        """Apply RFE feature selection"""
        estimator = LogisticRegression(random_state=self.config.random_state, max_iter=10000)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        
        rfe.fit(X_train, y_train)
        selected_mask = rfe.support_
        
        X_train_selected = X_train.iloc[:, selected_mask]
        X_test_selected = X_test.iloc[:, selected_mask]
        
        self.transformers['rfe'] = rfe
        self.selected_features = X_train_selected.columns.tolist()
        
        return X_train_selected, X_test_selected
    
    def apply_pca(self, X_train, X_test):
        """Apply PCA dimensionality reduction"""
        pca = PCA(n_components=self.config.pca_variance, random_state=self.config.random_state)
        
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        self.transformers['pca'] = pca
        
        return X_train_pca, X_test_pca
    
    def calculate_vif(self, X):
        """Calculate Variance Inflation Factor"""
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])]
        return vif_data.sort_values('VIF', ascending=False)

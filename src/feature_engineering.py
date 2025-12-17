"""
Feature engineering and selection module for PPROM prediction study.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression, LassoCV
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import warnings
from typing import Tuple, Dict, Any, List

from .config import config


class FeatureEngineer:
    """Handles feature engineering and selection."""
    
    def __init__(self, config=config):
        self.config = config
        self.feature_selector = None
        self.pca = None
        self.selected_features = None
        self.feature_importance = None
        
    def apply_feature_selection(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: pd.Series, feature_selection_method: str = None) -> Tuple:
        """Apply feature selection based on specified method."""
        if feature_selection_method is None:
            feature_selection_method = self.config.feature_selection
        
        print(f"Applying feature selection: {feature_selection_method}")
        
        X_train_selected = X_train.copy()
        X_test_selected = X_test.copy()
        original_columns = X_train.columns
        
        if feature_selection_method.startswith('rfe'):
            # RFE Feature Selection
            n_features = int(feature_selection_method[3:])
            n_features = min(n_features, X_train.shape[1])
            
            print(f"  RFE with {n_features} features")
            estimator = LogisticRegression(random_state=self.config.random_state, max_iter=10000)
            rfe = RFE(estimator=estimator, n_features_to_select=n_features)
            
            X_train_selected = rfe.fit_transform(X_train_selected, y_train)
            X_test_selected = rfe.transform(X_test_selected)
            self.feature_selector = rfe
            
            # Get selected features
            if hasattr(rfe, 'support_'):
                self.selected_features = original_columns[rfe.support_]
                X_train_selected = pd.DataFrame(
                    X_train_selected,
                    columns=self.selected_features,
                    index=X_train.index
                )
                X_test_selected = pd.DataFrame(
                    X_test_selected,
                    columns=self.selected_features,
                    index=X_test.index
                )
                print(f"  ✓ RFE selected {len(self.selected_features)} features")
        
        elif feature_selection_method == 'boruta':
            # Boruta Feature Selection
            X_train_selected, X_test_selected = self._apply_boruta(
                X_train_selected, X_test_selected, y_train, original_columns
            )
        
        elif feature_selection_method == 'lasso':
            # Lasso Feature Selection
            X_train_selected, X_test_selected = self._apply_lasso(
                X_train_selected, X_test_selected, y_train, original_columns
            )
        
        elif feature_selection_method == 'all':
            print("  Using all features (no feature selection)")
            self.selected_features = original_columns
        else:
            print(f"  Unknown feature selection method. Using all features.")
            self.selected_features = original_columns
        
        return X_train_selected, X_test_selected
    
    def _apply_boruta(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series, original_columns: List) -> Tuple:
        """Apply Boruta feature selection."""
        print("  Applying Boruta feature selection...")
        
        try:
            # Use Random Forest as estimator
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1,
                max_depth=10,
                min_samples_split=10,
                class_weight='balanced'
            )
            
            boruta_selector = BorutaPy(
                estimator=rf,
                n_estimators='auto',
                verbose=2,
                random_state=self.config.random_state,
                max_iter=self.config.boruta_iterations,
                alpha=self.config.boruta_alpha,
                two_step=False,
                perc=self.config.boruta_percentile
            )
            
            # Fit Boruta
            X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
            boruta_selector.fit(X_train_values, y_train.values)
            
            # Get selected features
            selected_mask = boruta_selector.support_
            self.selected_features = original_columns[selected_mask]
            
            # Handle case where no features are selected
            if len(self.selected_features) == 0:
                print("  ⚠ No features selected by Boruta. Using correlation-based fallback.")
                self.selected_features = self._fallback_feature_selection(X_train, y_train, original_columns)
            
            # Apply selection
            X_train_selected = X_train[self.selected_features]
            X_test_selected = X_test[self.selected_features]
            
            self.feature_selector = boruta_selector
            print(f"  ✓ Boruta selected {len(self.selected_features)} features")
            
            # Save Boruta results
            self._save_boruta_results(boruta_selector, original_columns)
            
        except Exception as e:
            print(f"  ✗ Boruta failed: {e}. Using correlation-based fallback.")
            self.selected_features = self._fallback_feature_selection(X_train, y_train, original_columns)
            X_train_selected = X_train[self.selected_features]
            X_test_selected = X_test[self.selected_features]
        
        return X_train_selected, X_test_selected
    
    def _apply_lasso(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, original_columns: List) -> Tuple:
        """Apply Lasso feature selection."""
        print("  Applying Lasso feature selection...")
        
        try:
            # Use LassoCV to find optimal alpha
            lasso = LassoCV(cv=5, random_state=self.config.random_state, max_iter=10000)
            lasso.fit(X_train, y_train)
            
            # Create selector
            lasso_selector = SelectFromModel(
                LassoCV(cv=5, random_state=self.config.random_state, max_iter=10000),
                threshold='mean'
            )
            lasso_selector.fit(X_train, y_train)
            
            # Get selected features
            selected_mask = lasso_selector.get_support()
            self.selected_features = original_columns[selected_mask]
            
            # Apply selection
            X_train_selected = X_train[self.selected_features]
            X_test_selected = X_test[self.selected_features]
            
            self.feature_selector = lasso_selector
            print(f"  ✓ Lasso selected {len(self.selected_features)} features")
            print(f"  Best alpha: {lasso.alpha_:.6f}")
            
            # Save Lasso results
            self._save_lasso_results(lasso_selector, original_columns, lasso.coef_)
            
        except Exception as e:
            print(f"  ✗ Lasso failed: {e}. Using all features.")
            self.selected_features = original_columns
            X_train_selected = X_train
            X_test_selected = X_test
        
        return X_train_selected, X_test_selected
    
    def _fallback_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   original_columns: List, n_features: int = 20) -> List:
        """Fallback feature selection using correlation."""
        correlation_data = X_train.copy()
        correlation_data['target'] = y_train.values
        correlations = correlation_data.corr()['target'].abs().sort_values(ascending=False)
        
        n_features = max(5, min(n_features, len(correlations) // 2))
        top_corr_features = correlations.index[1:1 + n_features]
        selected_features = [f for f in top_corr_features if f in original_columns]
        
        print(f"  Fallback selected {len(selected_features)} features by correlation")
        return selected_features
    
    def apply_pca(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series = None) -> Tuple:
        """Apply PCA for dimensionality reduction."""
        if self.config.feature_engineering != 'pca':
            return X_train, X_test
        
        print(f"Applying PCA with {self.config.pca_variance*100}% variance...")
        
        self.pca = PCA(n_components=self.config.pca_variance, random_state=self.config.random_state)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        
        # Create column names
        pca_columns = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]
        X_train_pca = pd.DataFrame(
            X_train_pca,
            columns=pca_columns,
            index=X_train.index
        )
        X_test_pca = pd.DataFrame(
            X_test_pca,
            columns=pca_columns,
            index=X_test.index
        )
        
        print(f"  Applied PCA: reduced to {X_train_pca.shape[1]} components")
        print(f"  Explained variance: {np.sum(self.pca.explained_variance_ratio_):.3f}")
        
        # Save PCA results
        self._save_pca_results()
        
        return X_train_pca, X_test_pca
    
    def _save_boruta_results(self, boruta_selector, original_columns):
        """Save Boruta detailed results."""
        import os
        
        if hasattr(boruta_selector, 'feature_importances_') and hasattr(boruta_selector, 'ranking_'):
            boruta_results = pd.DataFrame({
                'feature': original_columns,
                'selected': boruta_selector.support_,
                'importance': boruta_selector.feature_importances_,
                'ranking': boruta_selector.ranking_
            }).sort_values('importance', ascending=False)
        else:
            boruta_results = pd.DataFrame({
                'feature': original_columns,
                'selected': False,
                'importance': 0,
                'ranking': 0
            })
        
        boruta_results.to_csv(os.path.join(self.config.results_dir, "boruta_results.csv"), index=False)
        print("  ✓ Boruta results saved")
    
    def _save_lasso_results(self, lasso_selector, original_columns, coefficients):
        """Save Lasso detailed results."""
        import os
        
        selected_mask = lasso_selector.get_support()
        
        lasso_results = pd.DataFrame({
            'feature': original_columns,
            'selected': selected_mask,
            'coefficient': coefficients,
            'absolute_coefficient': np.abs(coefficients)
        }).sort_values('absolute_coefficient', ascending=False)
        
        lasso_results.to_csv(os.path.join(self.config.results_dir, "lasso_results.csv"), index=False)
        
        # Save selected features
        selected_features = original_columns[selected_mask]
        pd.DataFrame({'selected_features': selected_features}).to_csv(
            os.path.join(self.config.results_dir, "lasso_selected_features.csv"), index=False
        )
        
        print("  ✓ Lasso results saved")
    
    def _save_pca_results(self):
        """Save PCA analysis results."""
        import os
        
        if self.pca is not None:
            pca_results = pd.DataFrame({
                'component': [f'PC{i+1}' for i in range(len(self.pca.explained_variance_ratio_))],
                'explained_variance': self.pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_)
            })
            
            pca_results.to_csv(os.path.join(self.config.results_dir, "pca_results.csv"), index=False)
            print("  ✓ PCA results saved")
    
    def analyze_feature_correlations(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Analyze feature correlations with outcome."""
        import seaborn as sns
        import matplotlib.pyplot as plt
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        print("\nAnalyzing feature correlations...")
        
        # Calculate correlation with outcome
        correlation_data = X_train.copy()
        correlation_data['target'] = y_train.values
        
        outcome_correlations = correlation_data.corr()['target'].drop('target', errors='ignore')
        outcome_correlations = outcome_correlations.sort_values(ascending=False)
        outcome_correlations = outcome_correlations.dropna()
        
        if len(outcome_correlations) > 0:
            # Plot top correlations
            plt.figure(figsize=(12, 8))
            top_positive = outcome_correlations.head(10)
            top_negative = outcome_correlations.tail(10)
            top_features = pd.concat([top_positive, top_negative])
            
            colors = ['red' if x < 0 else 'blue' for x in top_features.values]
            y_pos = np.arange(len(top_features))
            
            plt.barh(y_pos, top_features.values, color=colors, alpha=0.7)
            plt.yticks(y_pos, top_features.index)
            plt.xlabel('Correlation Coefficient')
            plt.title('Top Correlations with Outcome')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.config.plots_dir, 'feature_correlations.png'), 
                       dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            # Save correlation data
            correlation_df = pd.DataFrame({
                'feature': outcome_correlations.index,
                'correlation_with_outcome': outcome_correlations.values,
                'absolute_correlation': np.abs(outcome_correlations.values)
            }).sort_values('absolute_correlation', ascending=False)
            
            correlation_df.to_csv(os.path.join(self.config.results_dir, 'feature_correlations.csv'), index=False)
            
            print(f"  Found {len(outcome_correlations)} features with valid correlations")
            print(f"  Strongest positive: {outcome_correlations.iloc[0]:.3f}")
            print(f"  Strongest negative: {outcome_correlations.iloc[-1]:.3f}")
        
        # Calculate VIF for multicollinearity
        print("\nCalculating Variance Inflation Factor (VIF)...")
        X_vif = X_train.loc[:, X_train.std() > 0]
        
        if len(X_vif.columns) > 1:
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_vif.columns
            vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
            vif_data = vif_data.sort_values('VIF', ascending=False)
            
            vif_data.to_csv(os.path.join(self.config.results_dir, 'vif_analysis.csv'), index=False)
            
            # Plot VIF
            plt.figure(figsize=(12, 8))
            colors = ['red' if vif > 10 else 'orange' if vif > 5 else 'green' for vif in vif_data['VIF']]
            bars = plt.barh(range(len(vif_data)), vif_data['VIF'], color=colors, alpha=0.7)
            plt.yticks(range(len(vif_data)), vif_data['feature'])
            plt.xlabel('Variance Inflation Factor (VIF)')
            plt.title('Multicollinearity Analysis')
            plt.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='VIF > 5')
            plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='VIF > 10')
            plt.legend()
            plt.grid(axis='x', alpha=0.3)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.config.plots_dir, 'vif_analysis.png'), 
                       dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            print(f"  VIF analysis completed: {len(vif_data)} features")
            print(f"  Features with VIF > 10: {len(vif_data[vif_data['VIF'] > 10])}")
    
    def run_feature_engineering_pipeline(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                        y_train: pd.Series) -> Dict[str, Any]:
        """Run complete feature engineering pipeline."""
        print("=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        # Analyze correlations
        self.analyze_feature_correlations(X_train, y_train)
        
        # Apply feature selection
        X_train_selected, X_test_selected = self.apply_feature_selection(
            X_train, X_test, y_train
        )
        
        # Apply PCA if requested
        if self.config.feature_engineering == 'pca':
            X_train_final, X_test_final = self.apply_pca(X_train_selected, X_test_selected, y_train)
        else:
            X_train_final, X_test_final = X_train_selected, X_test_selected
        
        print(f"\nFinal feature matrices:")
        print(f"  Training: {X_train_final.shape}")
        print(f"  Test: {X_test_final.shape}")
        
        if self.selected_features is not None:
            print(f"  Selected features: {len(self.selected_features)}")
        
        return {
            'X_train_final': X_train_final,
            'X_test_final': X_test_final,
            'selected_features': self.selected_features,
            'feature_selector': self.feature_selector,
            'pca': self.pca
        }

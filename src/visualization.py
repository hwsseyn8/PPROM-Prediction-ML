"""
Visualization module for PPROM prediction study.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import os
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .config import config


class Visualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, config=config):
        self.config = config
        self.set_plot_style()
    
    def set_plot_style(self):
        """Set consistent plot style."""
        plt.style.use(self.config.plot_style)
        sns.set_palette("husl")
        
        # Set font sizes
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
    
    def plot_roc_curves(self, results: Dict[str, Dict], save_path: Optional[str] = None):
        """Plot ROC curves for all models."""
        plt.figure(figsize=self.config.plot_figsize)
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        # Plot each model's ROC curve
        for model_name, result in results.items():
            if 'fpr' in result and 'tpr' in result:
                fpr = result['fpr']
                tpr = result['tpr']
                roc_auc = result.get('roc_auc', 0.5)
                
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, 'roc_curves.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  ROC curves saved to {save_path}")
    
    def plot_pr_curves(self, results: Dict[str, Dict], y_test: np.ndarray,
                      save_path: Optional[str] = None):
        """Plot precision-recall curves for all models."""
        plt.figure(figsize=self.config.plot_figsize)
        
        # Calculate baseline (random classifier)
        baseline = np.sum(y_test) / len(y_test)
        plt.axhline(y=baseline, color='r', linestyle='--', 
                   alpha=0.5, label=f'Random (AP = {baseline:.3f})')
        
        # Plot each model's PR curve
        for model_name, result in results.items():
            if 'precision_curve' in result and 'recall_curve' in result:
                precision = result['precision_curve']
                recall = result['recall_curve']
                avg_precision = result.get('avg_precision', 0)
                
                plt.plot(recall, precision, lw=2,
                        label=f'{model_name} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison', fontweight='bold')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, 'pr_curves.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  PR curves saved to {save_path}")
    
    def plot_confusion_matrices(self, results: Dict[str, Dict], 
                               save_dir: Optional[str] = None):
        """Plot confusion matrices for all models."""
        if save_dir is None:
            save_dir = os.path.join(self.config.plots_dir, 'confusion_matrices')
        
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, result in results.items():
            if 'confusion_matrix' in result:
                cm = result['confusion_matrix']
                
                plt.figure(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                             display_labels=['No PPROM', 'PPROM'])
                disp.plot(cmap='Blues', values_format='d')
                plt.title(f'Confusion Matrix - {model_name}', fontweight='bold')
                
                save_path = os.path.join(save_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
                plt.tight_layout()
                plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
                plt.close()
        
        print(f"  Confusion matrices saved to {save_dir}/")
    
    def plot_calibration_curves(self, results: Dict[str, Dict], y_test: np.ndarray,
                               save_path: Optional[str] = None):
        """Plot calibration curves for all models."""
        from sklearn.calibration import calibration_curve
        
        plt.figure(figsize=self.config.plot_figsize)
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        
        # Plot each model's calibration curve
        for model_name, result in results.items():
            if 'probabilities' in result:
                y_pred_proba = result['probabilities']
                
                try:
                    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, 
                                                           n_bins=10, strategy='uniform')
                    
                    brier_score = result.get('brier_score', 1.0)
                    
                    plt.plot(prob_pred, prob_true, 's-', lw=2,
                            label=f'{model_name} (Brier = {brier_score:.3f})')
                except:
                    continue
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curves - Model Comparison', fontweight='bold')
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, 'calibration_curves.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  Calibration curves saved to {save_path}")
    
    def plot_performance_comparison(self, results: Dict[str, Dict],
                                   metrics: List[str] = None,
                                   save_path: Optional[str] = None):
        """Plot performance comparison across metrics."""
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'specificity']
        
        # Prepare data
        comparison_data = []
        model_names = []
        
        for model_name, result in results.items():
            model_names.append(model_name)
            model_scores = []
            
            for metric in metrics:
                score = result.get(metric, 0)
                model_scores.append(score)
            
            comparison_data.append(model_scores)
        
        if not comparison_data:
            print("  No data for performance comparison")
            return
        
        comparison_data = np.array(comparison_data)
        
        # Create grouped bar chart
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        plt.figure(figsize=(12, 8))
        
        for i, model_name in enumerate(model_names):
            offset = (i - len(model_names) / 2) * width + width / 2
            plt.bar(x + offset, comparison_data[i], width, label=model_name)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison', fontweight='bold')
        plt.xticks(x, [m.upper() for m in metrics], rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1.05)
        
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, 'performance_comparison.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  Performance comparison saved to {save_path}")
    
    def plot_feature_importance(self, model, feature_names: List[str],
                               model_name: str, top_n: int = 20,
                               save_path: Optional[str] = None):
        """Plot feature importance for tree-based models."""
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            print(f"  No feature importance available for {model_name}")
            return
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        
        bars = plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}', fontweight='bold')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{importance:.4f}', va='center')
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, 
                                   f'feature_importance_{model_name.replace(" ", "_")}.png')
        
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  Feature importance saved to {save_path}")
        
        return importance_df
    
    def plot_shap_summary(self, model, X: pd.DataFrame, feature_names: List[str],
                         model_name: str, save_path: Optional[str] = None):
        """Plot SHAP summary plot for model interpretability."""
        try:
            import shap
            
            print(f"  Generating SHAP summary for {model_name}...")
            
            # Create explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X)
                shap_values = explainer(X)
            else:
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values[:, :, 1], X, feature_names=feature_names, 
                            show=False, max_display=20)
            
            if save_path is None:
                save_path = os.path.join(self.config.plots_dir, 
                                       f'shap_summary_{model_name.replace(" ", "_")}.png')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            # Also save SHAP values
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_shap': np.mean(np.abs(shap_values.values[:, :, 1]), axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
            
            shap_df.to_csv(os.path.join(self.config.results_dir, 
                                      f'shap_values_{model_name.replace(" ", "_")}.csv'), 
                         index=False)
            
            print(f"  SHAP summary saved to {save_path}")
            
        except Exception as e:
            print(f"  Could not generate SHAP plot: {e}")
    
    def plot_cv_results(self, cv_results: Dict[str, Dict], save_path: Optional[str] = None):
        """Plot cross-validation results."""
        metrics = ['accuracy', 'f1', 'roc_auc']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx] if len(metrics) > 1 else axes
            
            model_names = []
            means = []
            stds = []
            
            for model_name, result in cv_results.items():
                if f'{metric}_mean' in result:
                    model_names.append(model_name)
                    means.append(result[f'{metric}_mean'])
                    stds.append(result[f'{metric}_std'])
            
            if means:
                x = np.arange(len(model_names))
                bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
                
                ax.set_xlabel('Models')
                ax.set_ylabel(metric.upper())
                ax.set_title(f'CV {metric.upper()} Scores', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, mean in zip(bars, means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.config.plots_dir, 'cv_results.png')
        
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  CV results saved to {save_path}")
    
    def create_interactive_dashboard(self, results: Dict[str, Dict], 
                                    X_test: pd.DataFrame, y_test: np.ndarray,
                                    save_path: Optional[str] = None):
        """Create interactive dashboard using Plotly."""
        try:
            print("  Creating interactive dashboard...")
            
            if save_path is None:
                save_path = os.path.join(self.config.plots_dir, 'interactive_dashboard.html')
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('ROC Curves', 'Precision-Recall Curves',
                               'Performance Comparison', 'Calibration Curves'),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                      [{'type': 'bar'}, {'type': 'scatter'}]]
            )
            
            # 1. ROC Curves
            for model_name, result in results.items():
                if 'fpr' in result and 'tpr' in result:
                    fig.add_trace(
                        go.Scatter(x=result['fpr'], y=result['tpr'],
                                  mode='lines', name=f'{model_name}',
                                  hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}'),
                        row=1, col=1
                    )
            
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                          name='Random', line=dict(dash='dash')),
                row=1, col=1
            )
            
            fig.update_xaxes(title_text='False Positive Rate', row=1, col=1)
            fig.update_yaxes(title_text='True Positive Rate', row=1, col=1)
            
            # 2. PR Curves
            baseline = np.sum(y_test) / len(y_test)
            
            for model_name, result in results.items():
                if 'precision_curve' in result and 'recall_curve' in result:
                    fig.add_trace(
                        go.Scatter(x=result['recall_curve'], y=result['precision_curve'],
                                  mode='lines', name=f'{model_name}',
                                  showlegend=False,
                                  hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}'),
                        row=1, col=2
                    )
            
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[baseline, baseline], mode='lines',
                          name='Baseline', line=dict(dash='dash'),
                          showlegend=False),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text='Recall', row=1, col=2)
            fig.update_yaxes(title_text='Precision', row=1, col=2)
            
            # 3. Performance Comparison
            metrics = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
            
            for i, model_name in enumerate(results.keys()):
                scores = [results[model_name].get(metric, 0) for metric in metrics]
                
                fig.add_trace(
                    go.Bar(name=model_name, x=metrics, y=scores,
                          hovertemplate='%{x}: %{y:.3f}'),
                    row=2, col=1
                )
            
            fig.update_xaxes(title_text='Metrics', row=2, col=1)
            fig.update_yaxes(title_text='Score', range=[0, 1], row=2, col=1)
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Model Performance Dashboard",
                hovermode='closest'
            )
            
            # Save as HTML
            fig.write_html(save_path)
            
            print(f"  Interactive dashboard saved to {save_path}")
            
        except Exception as e:
            print(f"  Could not create interactive dashboard: {e}")
    
    def generate_all_visualizations(self, results: Dict[str, Dict], 
                                   X_test: pd.DataFrame, y_test: np.ndarray,
                                   feature_names: List[str] = None):
        """Generate all visualizations."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create plots directory
        os.makedirs(self.config.plots_dir, exist_ok=True)
        
        # 1. ROC curves
        self.plot_roc_curves(results)
        
        # 2. PR curves
        self.plot_pr_curves(results, y_test)
        
        # 3. Confusion matrices
        self.plot_confusion_matrices(results)
        
        # 4. Calibration curves
        self.plot_calibration_curves(results, y_test)
        
        # 5. Performance comparison
        self.plot_performance_comparison(results)
        
        # 6. CV results (if available)
        cv_results = {}
        for model_name, result in results.items():
            if 'cv_results' in result:
                cv_results[model_name] = result['cv_results']
        
        if cv_results:
            self.plot_cv_results(cv_results)
        
        # 7. Feature importance for each model
        if feature_names is not None:
            for model_name, result in results.items():
                if 'model' in result:
                    self.plot_feature_importance(
                        result['model'], feature_names, model_name
                    )
                    
                    # Generate SHAP plots for top models
                    if model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
                        self.plot_shap_summary(
                            result['model'], X_test, feature_names, model_name
                        )
        
        # 8. Interactive dashboard
        try:
            self.create_interactive_dashboard(results, X_test, y_test)
        except:
            print("  Could not create interactive dashboard (Plotly not available)")
        
        print("\n✓ All visualizations generated successfully!")
        print(f"  Plots saved to: {self.config.plots_dir}/")

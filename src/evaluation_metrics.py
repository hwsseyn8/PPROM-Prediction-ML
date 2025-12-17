"""
Evaluation metrics and statistical analysis module for PPROM prediction study.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, classification_report,
                           average_precision_score, precision_recall_curve, roc_curve,
                           brier_score_loss, balanced_accuracy_score, matthews_corrcoef)
from sklearn.calibration import calibration_curve
import scipy.stats as stats
from typing import Dict, Any, List, Tuple
import os

from .config import config


class Evaluator:
    """Handles model evaluation and statistical analysis."""
    
    def __init__(self, config=config):
        self.config = config
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['specificity'] = self.calculate_specificity(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Advanced metrics
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.5
                metrics['average_precision'] = 0
                metrics['brier_score'] = 1.0
        
        # Calculate NPV
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            metrics['npv'] = 0
            metrics['fpr'] = 0
            metrics['fnr'] = 0
        
        return metrics
    
    def calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0
        return 0
    
    def calculate_confidence_intervals(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                      metric: str = 'roc_auc', n_bootstrap: int = 1000) -> Tuple[float, Tuple]:
        """Calculate confidence intervals using bootstrapping."""
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred_proba[indices]
            
            try:
                if metric == 'roc_auc':
                    score = roc_auc_score(y_true_boot, y_pred_boot)
                elif metric == 'average_precision':
                    score = average_precision_score(y_true_boot, y_pred_boot)
                elif metric == 'accuracy':
                    y_pred_class = (y_pred_boot >= 0.5).astype(int)
                    score = accuracy_score(y_true_boot, y_pred_class)
                else:
                    continue
                
                bootstrap_scores.append(score)
            except:
                continue
        
        if bootstrap_scores:
            mean_score = np.mean(bootstrap_scores)
            ci_lower = np.percentile(bootstrap_scores, 2.5)
            ci_upper = np.percentile(bootstrap_scores, 97.5)
            return mean_score, (ci_lower, ci_upper)
        
        return 0, (0, 0)
    
    def perform_statistical_tests(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform statistical tests to compare models."""
        print("\nPerforming statistical comparisons...")
        
        statistical_tests = {}
        
        # Extract metrics for comparison
        metrics_to_compare = ['accuracy', 'f1', 'roc_auc']
        
        for metric in metrics_to_compare:
            model_scores = {}
            for model_name, result in results.items():
                if metric in result:
                    model_scores[model_name] = result[metric]
            
            if len(model_scores) > 1:
                # Perform Friedman test (non-parametric repeated measures)
                scores_matrix = np.array(list(model_scores.values()))
                
                if scores_matrix.shape[0] > 1:
                    try:
                        friedman_stat, friedman_p = stats.friedmanchisquare(*scores_matrix)
                        
                        statistical_tests[f'friedman_{metric}'] = {
                            'statistic': friedman_stat,
                            'p_value': friedman_p,
                            'significant': friedman_p < 0.05
                        }
                        
                        print(f"  Friedman test for {metric}: χ²={friedman_stat:.3f}, p={friedman_p:.4f}")
                        
                        # Post-hoc pairwise comparisons
                        if friedman_p < 0.05:
                            pairwise_comparisons = {}
                            model_names = list(model_scores.keys())
                            
                            for i in range(len(model_names)):
                                for j in range(i + 1, len(model_names)):
                                    model1 = model_names[i]
                                    model2 = model_names[j]
                                    
                                    # Wilcoxon signed-rank test
                                    try:
                                        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                                            results[model1]['cv_results'].get(f'{metric}_scores', []),
                                            results[model2]['cv_results'].get(f'{metric}_scores', [])
                                        )
                                        
                                        pairwise_comparisons[f'{model1}_vs_{model2}'] = {
                                            'statistic': wilcoxon_stat,
                                            'p_value': wilcoxon_p,
                                            'significant': wilcoxon_p < 0.05
                                        }
                                    except:
                                        continue
                            
                            statistical_tests[f'pairwise_{metric}'] = pairwise_comparisons
                    
                    except:
                        print(f"  Could not perform Friedman test for {metric}")
        
        return statistical_tests
    
    def calculate_model_stability(self, cv_results: Dict[str, List]) -> Dict[str, float]:
        """Calculate model stability metrics from cross-validation."""
        stability_metrics = {}
        
        for metric, scores in cv_results.items():
            if 'scores' in metric:
                continue
            
            if '_scores' in metric:
                base_metric = metric.replace('_scores', '')
                scores_array = np.array(scores)
                
                stability_metrics[f'{base_metric}_mean'] = np.mean(scores_array)
                stability_metrics[f'{base_metric}_std'] = np.std(scores_array)
                stability_metrics[f'{base_metric}_cv'] = (np.std(scores_array) / np.mean(scores_array)) * 100
                stability_metrics[f'{base_metric}_range'] = np.ptp(scores_array)
        
        return stability_metrics
    
    def generate_comprehensive_report(self, results: Dict[str, Dict], 
                                     save_path: str = None) -> str:
        """Generate comprehensive evaluation report."""
        if save_path is None:
            save_path = os.path.join(self.config.results_dir, "comprehensive_report.txt")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {pd.Timestamp.now()}")
        report_lines.append(f"Total Models Evaluated: {len(results)}")
        report_lines.append("")
        
        # Summary table
        report_lines.append("MODEL PERFORMANCE SUMMARY")
        report_lines.append("-" * 80)
        
        # Create summary table
        summary_data = []
        for model_name, result in results.items():
            summary_data.append([
                model_name,
                f"{result.get('accuracy', 0):.4f}",
                f"{result.get('precision', 0):.4f}",
                f"{result.get('recall', 0):.4f}",
                f"{result.get('f1', 0):.4f}",
                f"{result.get('roc_auc', 0):.4f}",
                f"{result.get('specificity', 0):.4f}"
            ])
        
        # Sort by F1 score
        summary_data.sort(key=lambda x: float(x[4]), reverse=True)
        
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Specificity"]
        header_line = " | ".join([h.ljust(15) for h in headers])
        separator = "-" * len(header_line)
        
        report_lines.append(header_line)
        report_lines.append(separator)
        
        for row in summary_data:
            row_line = " | ".join([str(item).ljust(15) for item in row])
            report_lines.append(row_line)
        
        report_lines.append("")
        
        # Detailed results per model
        report_lines.append("DETAILED MODEL RESULTS")
        report_lines.append("=" * 80)
        
        for model_name, result in results.items():
            report_lines.append(f"\n{model_name}")
            report_lines.append("-" * 40)
            
            # Basic metrics
            report_lines.append("Test Set Performance:")
            report_lines.append(f"  Accuracy:    {result.get('accuracy', 0):.4f}")
            report_lines.append(f"  Precision:   {result.get('precision', 0):.4f}")
            report_lines.append(f"  Recall:      {result.get('recall', 0):.4f}")
            report_lines.append(f"  Specificity: {result.get('specificity', 0):.4f}")
            report_lines.append(f"  F1 Score:    {result.get('f1', 0):.4f}")
            report_lines.append(f"  ROC AUC:     {result.get('roc_auc', 0):.4f}")
            report_lines.append(f"  Avg Precision: {result.get('avg_precision', 0):.4f}")
            report_lines.append(f"  MCC:         {result.get('mcc', 0):.4f}")
            report_lines.append(f"  NPV:         {result.get('npv', 0):.4f}")
            
            # Cross-validation results
            if 'cv_results' in result:
                cv = result['cv_results']
                report_lines.append("\n  Cross-Validation (Mean ± SD):")
                report_lines.append(f"    Accuracy:  {cv.get('accuracy_mean', 0):.4f} ± {cv.get('accuracy_std', 0):.4f}")
                report_lines.append(f"    F1 Score:  {cv.get('f1_mean', 0):.4f} ± {cv.get('f1_std', 0):.4f}")
                report_lines.append(f"    ROC AUC:   {cv.get('roc_auc_mean', 0):.4f} ± {cv.get('roc_auc_std', 0):.4f}")
            
            # Calibration info
            if 'calibration_method' in result:
                report_lines.append(f"\n  Calibration: {result['calibration_method']}")
                if 'brier_score' in result:
                    report_lines.append(f"  Brier Score: {result['brier_score']:.4f}")
        
        # Statistical comparisons
        statistical_tests = self.perform_statistical_tests(results)
        
        if statistical_tests:
            report_lines.append("\nSTATISTICAL COMPARISONS")
            report_lines.append("=" * 80)
            
            for test_name, test_result in statistical_tests.items():
                if 'friedman' in test_name:
                    report_lines.append(f"\nFriedman Test ({test_name.replace('friedman_', '')}):")
                    report_lines.append(f"  Statistic: χ² = {test_result['statistic']:.3f}")
                    report_lines.append(f"  p-value: {test_result['p_value']:.4f}")
                    report_lines.append(f"  Significant: {'Yes' if test_result['significant'] else 'No'}")
        
        # Best model recommendations
        report_lines.append("\n" + "=" * 80)
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("=" * 80)
        
        # Find best by different metrics
        best_by_metric = {}
        for metric in ['f1', 'roc_auc', 'accuracy', 'specificity', 'balanced_accuracy']:
            best_model = max(results.items(), key=lambda x: x[1].get(metric, 0))[0]
            best_score = results[best_model].get(metric, 0)
            best_by_metric[metric] = (best_model, best_score)
        
        report_lines.append("\nBest Models by Metric:")
        for metric, (model, score) in best_by_metric.items():
            report_lines.append(f"  {metric.upper()}: {model} ({score:.4f})")
        
        # Overall recommendation
        best_f1_model, best_f1_score = best_by_metric['f1']
        best_auc_model, best_auc_score = best_by_metric['roc_auc']
        
        report_lines.append(f"\nOverall Recommendation:")
        report_lines.append(f"  For clinical use (balance precision/recall): {best_f1_model}")
        report_lines.append(f"  For research (discriminatory power): {best_auc_model}")
        
        # Save report
        report_text = "\n".join(report_lines)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Comprehensive report saved to {save_path}")
        
        return report_text

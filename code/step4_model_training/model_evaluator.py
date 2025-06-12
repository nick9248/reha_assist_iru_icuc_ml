"""
Model Evaluator Module for NBE Prediction Project
Handles comprehensive evaluation and comparison of trained models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime
import json

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, log_loss
)
from sklearn.calibration import calibration_curve
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive evaluation and comparison of trained models
    """

    def __init__(self, plots_path: Path, log_path: Path):
        self.plots_path = plots_path
        self.log_path = log_path
        self.logger = self._setup_logger()

        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create plots directory
        self.plots_dir = self.plots_path / 'step4_model_training'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for model evaluation operations"""
        logger = logging.getLogger('ModelEvaluator')
        logger.setLevel(logging.INFO)

        # Create log directory
        log_dir = self.log_path / 'step4'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'model_evaluator_{timestamp}.log'

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)

        return logger

    def evaluate_single_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                             model_name: str, model_type: str) -> Dict[str, Any]:
        """
        Evaluate a single model on test data

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            model_type: 'baseline' or 'enhanced'

        Returns:
            Dict containing evaluation metrics
        """
        self.logger.info(f"Evaluating {model_name} ({model_type})")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for class 1

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)

        # Log loss (cross-entropy)
        try:
            logloss = log_loss(y_test, y_pred_proba)
        except:
            logloss = None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        # ROC curve data
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)

        # Precision-Recall curve data
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)

        results = {
            'model_name': model_name,
            'model_type': model_type,
            'test_samples': len(y_test),
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'false_positive_rate': false_positive_rate,
                'false_negative_rate': false_negative_rate,
                'log_loss': logloss
            },
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp),
                'matrix': cm.tolist()
            },
            'curves': {
                'roc': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds_roc.tolist()
                },
                'precision_recall': {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist(),
                    'thresholds': thresholds_pr.tolist()
                }
            },
            'predictions': {
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist(),
                'y_true': y_test.tolist()
            }
        }

        self.logger.info(f"{model_name} ({model_type}) - AUC: {auc_roc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        return results

    def evaluate_all_models(self, baseline_models: Dict[str, Any], enhanced_models: Dict[str, Any],
                           test_datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Evaluate all trained models on test data

        Args:
            baseline_models: Trained baseline models
            enhanced_models: Trained enhanced models
            test_datasets: Test datasets

        Returns:
            Dict containing all evaluation results
        """
        self.logger.info("Starting comprehensive model evaluation")
        self.logger.info(f"Baseline models to evaluate: {len(baseline_models)}")
        self.logger.info(f"Enhanced models to evaluate: {len(enhanced_models)}")

        all_results = {
            'baseline': {},
            'enhanced': {},
            'timestamp': datetime.now().isoformat()
        }

        # Prepare test data
        X_baseline_test, y_baseline_test = self._prepare_features_target(test_datasets['baseline_test'])
        X_enhanced_test, y_enhanced_test = self._prepare_features_target(test_datasets['enhanced_test'])

        self.logger.info(f"Baseline test data shape: {X_baseline_test.shape}")
        self.logger.info(f"Enhanced test data shape: {X_enhanced_test.shape}")

        # Evaluate baseline models
        for model_key, model_data in baseline_models.items():
            try:
                self.logger.info(f"Evaluating baseline {model_key}...")
                results = self.evaluate_single_model(
                    model_data['model'], X_baseline_test, y_baseline_test,
                    model_data['model_name'], 'baseline'
                )
                all_results['baseline'][model_key] = results
                self.logger.info(f"✅ Baseline {model_key} evaluation completed")
            except Exception as e:
                self.logger.error(f"❌ Error evaluating baseline {model_key}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

        # Evaluate enhanced models
        for model_key, model_data in enhanced_models.items():
            try:
                self.logger.info(f"Evaluating enhanced {model_key}...")
                results = self.evaluate_single_model(
                    model_data['model'], X_enhanced_test, y_enhanced_test,
                    model_data['model_name'], 'enhanced'
                )
                all_results['enhanced'][model_key] = results
                self.logger.info(f"✅ Enhanced {model_key} evaluation completed")
            except Exception as e:
                self.logger.error(f"❌ Error evaluating enhanced {model_key}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

        self.logger.info(f"Model evaluation completed: {len(all_results['baseline'])} baseline + {len(all_results['enhanced'])} enhanced")
        return all_results

    def _prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable"""
        if 'nbe' not in df.columns:
            raise ValueError("Target variable 'nbe' not found in dataset")
        X = df.drop('nbe', axis=1)
        y = df['nbe']
        return X, y

    def compare_model_performances(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare performances between baseline and enhanced models

        Args:
            evaluation_results: Results from evaluate_all_models

        Returns:
            Dict containing comparison analysis
        """
        self.logger.info("Comparing model performances")

        comparison = {
            'summary': {},
            'best_models': {},
            'improvement_analysis': {},
            'recommendations': []
        }

        # Extract metrics for comparison
        baseline_metrics = {}
        enhanced_metrics = {}

        for model_key, results in evaluation_results['baseline'].items():
            baseline_metrics[model_key] = results['metrics']

        for model_key, results in evaluation_results['enhanced'].items():
            enhanced_metrics[model_key] = results['metrics']

        # Find best models
        if baseline_metrics:
            best_baseline_key = max(baseline_metrics.keys(), key=lambda k: baseline_metrics[k]['auc_roc'])
            comparison['best_models']['baseline'] = {
                'model': best_baseline_key,
                'auc_roc': baseline_metrics[best_baseline_key]['auc_roc'],
                'precision': baseline_metrics[best_baseline_key]['precision'],
                'recall': baseline_metrics[best_baseline_key]['recall']
            }

        if enhanced_metrics:
            best_enhanced_key = max(enhanced_metrics.keys(), key=lambda k: enhanced_metrics[k]['auc_roc'])
            comparison['best_models']['enhanced'] = {
                'model': best_enhanced_key,
                'auc_roc': enhanced_metrics[best_enhanced_key]['auc_roc'],
                'precision': enhanced_metrics[best_enhanced_key]['precision'],
                'recall': enhanced_metrics[best_enhanced_key]['recall']
            }

        # Calculate improvements
        for model_key in baseline_metrics.keys():
            if model_key in enhanced_metrics:
                baseline_auc = baseline_metrics[model_key]['auc_roc']
                enhanced_auc = enhanced_metrics[model_key]['auc_roc']
                improvement = enhanced_auc - baseline_auc

                comparison['improvement_analysis'][model_key] = {
                    'baseline_auc': baseline_auc,
                    'enhanced_auc': enhanced_auc,
                    'auc_improvement': improvement,
                    'improvement_percentage': (improvement / baseline_auc) * 100 if baseline_auc > 0 else 0
                }

        # Generate recommendations
        if comparison['best_models'].get('enhanced') and comparison['best_models'].get('baseline'):
            enhanced_auc = comparison['best_models']['enhanced']['auc_roc']
            baseline_auc = comparison['best_models']['baseline']['auc_roc']

            if enhanced_auc > baseline_auc + 0.02:  # Significant improvement threshold
                comparison['recommendations'].append("Use enhanced model for production - significant improvement detected")
            elif enhanced_auc > baseline_auc:
                comparison['recommendations'].append("Enhanced model shows improvement but consider complexity trade-off")
            else:
                comparison['recommendations'].append("Baseline model sufficient - enhanced features may not add value")

        # Summary statistics
        if baseline_metrics and enhanced_metrics:
            avg_baseline_auc = np.mean([m['auc_roc'] for m in baseline_metrics.values()])
            avg_enhanced_auc = np.mean([m['auc_roc'] for m in enhanced_metrics.values()])

            comparison['summary'] = {
                'average_baseline_auc': avg_baseline_auc,
                'average_enhanced_auc': avg_enhanced_auc,
                'average_improvement': avg_enhanced_auc - avg_baseline_auc,
                'models_compared': len(baseline_metrics)
            }

        return comparison

    def create_performance_comparison_plot(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Create comprehensive performance comparison visualization

        Args:
            evaluation_results: Results from evaluate_all_models

        Returns:
            str: Path to saved plot
        """
        self.logger.info("Creating performance comparison plot")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        # Prepare data for plotting
        models = []
        model_types = []
        auc_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for model_type in ['baseline', 'enhanced']:
            for model_key, results in evaluation_results[model_type].items():
                models.append(f"{results['model_name']}")
                model_types.append(model_type)
                auc_scores.append(results['metrics']['auc_roc'])
                precision_scores.append(results['metrics']['precision'])
                recall_scores.append(results['metrics']['recall'])
                f1_scores.append(results['metrics']['f1_score'])

        # Create DataFrame for easy plotting
        plot_data = pd.DataFrame({
            'Model': models,
            'Type': model_types,
            'AUC': auc_scores,
            'Precision': precision_scores,
            'Recall': recall_scores,
            'F1': f1_scores
        })

        # Plot 1: AUC Comparison
        ax1 = axes[0, 0]
        sns.barplot(data=plot_data, x='Model', y='AUC', hue='Type', ax=ax1)
        ax1.set_title('AUC-ROC Comparison', fontweight='bold')
        ax1.set_ylim(0.5, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Precision vs Recall
        ax2 = axes[0, 1]
        for model_type in ['baseline', 'enhanced']:
            type_data = plot_data[plot_data['Type'] == model_type]
            ax2.scatter(type_data['Recall'], type_data['Precision'],
                       label=model_type, s=100, alpha=0.7)
            # Add model labels
            for idx, row in type_data.iterrows():
                ax2.annotate(row['Model'], (row['Recall'], row['Precision']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision vs Recall', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: F1 Score Comparison
        ax3 = axes[1, 0]
        sns.barplot(data=plot_data, x='Model', y='F1', hue='Type', ax=ax3)
        ax3.set_title('F1 Score Comparison', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Overall Performance Radar (for best models)
        ax4 = axes[1, 1]

        # Find best baseline and enhanced models (with safety check)
        baseline_data = plot_data[plot_data['Type'] == 'baseline']
        enhanced_data = plot_data[plot_data['Type'] == 'enhanced']

        if len(baseline_data) > 0 and len(enhanced_data) > 0:
            best_baseline = baseline_data.loc[baseline_data['AUC'].idxmax()]
            best_enhanced = enhanced_data.loc[enhanced_data['AUC'].idxmax()]

            metrics = ['AUC', 'Precision', 'Recall', 'F1']
            baseline_values = [best_baseline[metric] for metric in metrics]
            enhanced_values = [best_enhanced[metric] for metric in metrics]

            x = np.arange(len(metrics))
            width = 0.35

            ax4.bar(x - width/2, baseline_values, width, label=f'Best Baseline ({best_baseline["Model"]})', alpha=0.7)
            ax4.bar(x + width/2, enhanced_values, width, label=f'Best Enhanced ({best_enhanced["Model"]})', alpha=0.7)

            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Score')
            ax4.set_title('Best Model Comparison', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
        else:
            # Handle case where we don't have both types
            ax4.text(0.5, 0.5, 'Insufficient data for comparison',
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_title('Best Model Comparison', fontweight='bold')

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.plots_dir / f'model_performance_comparison_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Performance comparison plot saved: {plot_path}")
        return str(plot_path)

    def create_roc_curves_plot(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Create ROC curves for all models

        Args:
            evaluation_results: Results from evaluate_all_models

        Returns:
            str: Path to saved plot
        """
        self.logger.info("Creating ROC curves plot")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('ROC Curves Comparison', fontsize=16, fontweight='bold')

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        # Plot baseline models
        ax1.set_title('Baseline Models', fontweight='bold')
        for i, (model_key, results) in enumerate(evaluation_results['baseline'].items()):
            fpr = results['curves']['roc']['fpr']
            tpr = results['curves']['roc']['tpr']
            auc = results['metrics']['auc_roc']

            ax1.plot(fpr, tpr, color=colors[i % len(colors)],
                    label=f"{results['model_name']} (AUC = {auc:.3f})", linewidth=2)

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.500)')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot enhanced models
        ax2.set_title('Enhanced Models', fontweight='bold')
        for i, (model_key, results) in enumerate(evaluation_results['enhanced'].items()):
            fpr = results['curves']['roc']['fpr']
            tpr = results['curves']['roc']['tpr']
            auc = results['metrics']['auc_roc']

            ax2.plot(fpr, tpr, color=colors[i % len(colors)],
                    label=f"{results['model_name']} (AUC = {auc:.3f})", linewidth=2)

        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.500)')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.plots_dir / f'roc_curves_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"ROC curves plot saved: {plot_path}")
        return str(plot_path)

    def create_confusion_matrices_plot(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Create confusion matrices for all models

        Args:
            evaluation_results: Results from evaluate_all_models

        Returns:
            str: Path to saved plot
        """
        self.logger.info("Creating confusion matrices plot")

        # Count total models
        total_models = len(evaluation_results['baseline']) + len(evaluation_results['enhanced'])

        # Calculate subplot grid
        cols = 3
        rows = (total_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

        # Flatten axes for easy indexing
        if rows == 1:
            axes = [axes] if total_models == 1 else axes
        else:
            axes = axes.flatten()

        plot_idx = 0

        # Plot baseline models
        for model_key, results in evaluation_results['baseline'].items():
            ax = axes[plot_idx]
            cm = np.array(results['confusion_matrix']['matrix'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['NBE No', 'NBE Yes'],
                       yticklabels=['NBE No', 'NBE Yes'])

            ax.set_title(f"{results['model_name']} (Baseline)\nAUC: {results['metrics']['auc_roc']:.3f}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

            plot_idx += 1

        # Plot enhanced models
        for model_key, results in evaluation_results['enhanced'].items():
            ax = axes[plot_idx]
            cm = np.array(results['confusion_matrix']['matrix'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                       xticklabels=['NBE No', 'NBE Yes'],
                       yticklabels=['NBE No', 'NBE Yes'])

            ax.set_title(f"{results['model_name']} (Enhanced)\nAUC: {results['metrics']['auc_roc']:.3f}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.plots_dir / f'confusion_matrices_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Confusion matrices plot saved: {plot_path}")
        return str(plot_path)

    def create_feature_importance_plot(self, baseline_models: Dict[str, Any],
                                      enhanced_models: Dict[str, Any]) -> str:
        """
        Create feature importance comparison plots

        Args:
            baseline_models: Trained baseline models
            enhanced_models: Trained enhanced models

        Returns:
            str: Path to saved plot
        """
        self.logger.info("Creating feature importance plot")

        # Find models with feature importance
        baseline_importance = {}
        enhanced_importance = {}

        for model_key, model_data in baseline_models.items():
            if 'feature_importance' in model_data:
                baseline_importance[model_data['model_name']] = model_data['feature_importance']

        for model_key, model_data in enhanced_models.items():
            if 'feature_importance' in model_data:
                enhanced_importance[model_data['model_name']] = model_data['feature_importance']

        if not baseline_importance and not enhanced_importance:
            self.logger.warning("No feature importance data available")
            return None

        # Create subplots
        n_plots = len(baseline_importance) + len(enhanced_importance)
        cols = 2
        rows = (n_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')

        if rows == 1:
            axes = [axes] if n_plots == 1 else axes
        else:
            axes = axes.flatten()

        plot_idx = 0

        # Plot baseline feature importance
        for model_name, importance in baseline_importance.items():
            ax = axes[plot_idx]

            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_features)

            bars = ax.barh(range(len(features)), importances, alpha=0.7, color='skyblue')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name} (Baseline Features)', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                       f'{width:.3f}', ha='left', va='center', fontsize=9)

            plot_idx += 1

        # Plot enhanced feature importance
        for model_name, importance in enhanced_importance.items():
            ax = axes[plot_idx]

            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_features)

            # Limit to top 10 features for readability
            if len(features) > 10:
                features = features[:10]
                importances = importances[:10]

            bars = ax.barh(range(len(features)), importances, alpha=0.7, color='lightgreen')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name} (Enhanced Features)', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                       f'{width:.3f}', ha='left', va='center', fontsize=9)

            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.plots_dir / f'feature_importance_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Feature importance plot saved: {plot_path}")
        return str(plot_path)

    def save_evaluation_results(self, evaluation_results: Dict[str, Any],
                               comparison_results: Dict[str, Any],
                               plot_paths: Dict[str, str]) -> str:
        """
        Save comprehensive evaluation results and metadata

        Args:
            evaluation_results: Complete evaluation results
            comparison_results: Model comparison analysis
            plot_paths: Paths to generated plots

        Returns:
            str: Path to saved results file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Compile complete results
        complete_results = {
            'timestamp': timestamp,
            'evaluation_results': evaluation_results,
            'comparison_analysis': comparison_results,
            'generated_plots': plot_paths,
            'summary': {
                'total_models_evaluated': len(evaluation_results['baseline']) + len(evaluation_results['enhanced']),
                'baseline_models': len(evaluation_results['baseline']),
                'enhanced_models': len(evaluation_results['enhanced'])
            }
        }

        # Add best model recommendations
        if comparison_results.get('best_models'):
            complete_results['recommendations'] = {
                'best_baseline_model': comparison_results['best_models'].get('baseline'),
                'best_enhanced_model': comparison_results['best_models'].get('enhanced'),
                'overall_recommendations': comparison_results.get('recommendations', [])
            }

        # Save results
        results_file = self.plots_dir.parent / f'step4_evaluation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)

        self.logger.info(f"Evaluation results saved: {results_file}")
        return str(results_file)

    def generate_comprehensive_evaluation(self, baseline_models: Dict[str, Any],
                                        enhanced_models: Dict[str, Any],
                                        test_datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate complete evaluation report with all analyses and visualizations

        Args:
            baseline_models: Trained baseline models
            enhanced_models: Trained enhanced models
            test_datasets: Test datasets

        Returns:
            Dict containing complete evaluation results and file paths
        """
        self.logger.info("Starting comprehensive model evaluation")

        try:
            # Evaluate all models
            evaluation_results = self.evaluate_all_models(baseline_models, enhanced_models, test_datasets)

            # Compare performances
            comparison_results = self.compare_model_performances(evaluation_results)

            # Generate visualizations
            plot_paths = {}
            plot_paths['performance_comparison'] = self.create_performance_comparison_plot(evaluation_results)
            plot_paths['roc_curves'] = self.create_roc_curves_plot(evaluation_results)
            plot_paths['confusion_matrices'] = self.create_confusion_matrices_plot(evaluation_results)

            feature_importance_path = self.create_feature_importance_plot(baseline_models, enhanced_models)
            if feature_importance_path:
                plot_paths['feature_importance'] = feature_importance_path

            # Save complete results
            results_file = self.save_evaluation_results(evaluation_results, comparison_results, plot_paths)

            complete_output = {
                'evaluation_results': evaluation_results,
                'comparison_results': comparison_results,
                'plot_paths': plot_paths,
                'results_file': results_file
            }

            self.logger.info("Comprehensive evaluation completed successfully")

            # Log key findings
            if comparison_results.get('best_models'):
                best_baseline = comparison_results['best_models'].get('baseline')
                best_enhanced = comparison_results['best_models'].get('enhanced')

                if best_baseline:
                    self.logger.info(f"Best baseline model: {best_baseline['model']} (AUC: {best_baseline['auc_roc']:.4f})")
                if best_enhanced:
                    self.logger.info(f"Best enhanced model: {best_enhanced['model']} (AUC: {best_enhanced['auc_roc']:.4f})")

            return complete_output

        except Exception as e:
            self.logger.error(f"Error in comprehensive evaluation: {str(e)}")
            raise
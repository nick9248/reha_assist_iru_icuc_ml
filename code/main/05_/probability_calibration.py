"""
Probability calibration analysis for the NBE prediction model.
Evaluates and improves the reliability of probability estimates.
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import joblib


def load_data(datasets_path, logger):
    """
    Load the prepared datasets

    Parameters:
    -----------
    datasets_path : str
        Path to the prepared datasets
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Dictionary containing datasets
    """
    logger.info(f"Loading datasets from {datasets_path}")
    with open(datasets_path, 'rb') as f:
        datasets = pickle.load(f)
    return datasets


def load_model(model_path, logger):
    """
    Load the refined model

    Parameters:
    -----------
    model_path : str
        Path to the model
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    object
        Trained model
    """
    logger.info(f"Loading model from {model_path}")
    try:
        # Try loading with pickle first
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            # If model is a dictionary, extract the model object
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
        return model
    except:
        # If that fails, try joblib
        logger.info("Pickle loading failed, trying joblib...")
        model = joblib.load(model_path)
        return model


def plot_calibration_curve(y_true, y_prob, output_dir, logger, model_name="Model", n_bins=10):
    """
    Plot calibration curve

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    output_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    model_name : str
        Name of the model for plot title
    n_bins : int
        Number of bins for calibration curve
    """
    logger.info(f"Plotting calibration curve for {model_name}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Calculate metrics
    brier = brier_score_loss(y_true, y_prob)

    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'{model_name} (Brier={brier:.4f})')

    # Add diagonal line for perfect calibration
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

    # Add histogram of predicted probabilities - FIXED parameter from 'normed' to 'density'
    plt.hist(y_prob, range=(0, 1), bins=n_bins, histtype='step', density=True,
             label='Predicted probability distribution')

    # Configure plot
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability in each bin')
    plt.title(f'Calibration Curve - {model_name}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    model_filename = model_name.replace(' ', '_').lower()
    plt.savefig(os.path.join(output_dir, f'calibration_curve_{model_filename}.png'), dpi=300)
    plt.close()

    logger.info(f"Calibration curve saved to {os.path.join(output_dir, f'calibration_curve_{model_filename}.png')}")

    return prob_true, prob_pred, brier

def create_reliability_diagram(y_true, y_prob, output_dir, logger, model_name="Model", n_bins=10):
    """
    Create reliability diagram

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    output_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    model_name : str
        Name of the model for plot title
    n_bins : int
        Number of bins for reliability diagram
    """
    logger.info(f"Creating reliability diagram for {model_name}")

    # Create bins and calculate statistics
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    # Initialize arrays
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Calculate statistics for each bin
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_accs[i] = np.mean(y_true[bin_mask])
            bin_confs[i] = np.mean(y_prob[bin_mask])
            bin_counts[i] = np.sum(bin_mask)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})

    # Plot reliability diagram
    ax1.plot(bin_confs, bin_accs, marker='o', linewidth=2, label=f'{model_name}')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Fraction of positives')
    ax1.set_title(f'Reliability Diagram - {model_name}')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot bin counts
    ax2.bar(range(n_bins), bin_counts, alpha=0.7, align='center')
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels([f'{e:.1f}' for e in bin_edges[:-1]])
    ax2.set_xlabel('Predicted probability bin')
    ax2.set_ylabel('Count')
    ax2.set_title('Histogram of predictions')

    # Save plot
    plt.tight_layout()
    model_filename = model_name.replace(' ', '_').lower()
    plt.savefig(os.path.join(output_dir, f'reliability_diagram_{model_filename}.png'), dpi=300)
    plt.close()

    logger.info(f"Reliability diagram saved to {os.path.join(output_dir, f'reliability_diagram_{model_filename}.png')}")

    return bin_accs, bin_confs, bin_counts


def calibrate_probabilities(model, X_train, y_train, X_val, y_val, method='isotonic', logger=None):
    """
    Calibrate model probabilities

    Parameters:
    -----------
    model : object
        Model to calibrate
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    X_val : pandas.DataFrame
        Validation features
    y_val : pandas.Series
        Validation target
    method : str
        Calibration method ('isotonic' or 'sigmoid')
    logger : logging.Logger
        Logger instance (optional)

    Returns:
    --------
    object
        Calibrated model
    dict
        Calibration metrics
    """
    if logger:
        logger.info(f"Calibrating probabilities using {method} method")

    # Get original probabilities
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate original metrics
    orig_brier = brier_score_loss(y_val, y_val_pred_proba)
    orig_log_loss_val = log_loss(y_val, y_val_pred_proba)

    if logger:
        logger.info(f"Original metrics - Brier: {orig_brier:.4f}, Log Loss: {orig_log_loss_val:.4f}")

    # Calibrate model
    calibrated_model = CalibratedClassifierCV(model, method=method, cv='prefit')
    calibrated_model.fit(X_train, y_train)

    # Get calibrated probabilities
    cal_val_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]

    # Calculate calibrated metrics
    cal_brier = brier_score_loss(y_val, cal_val_pred_proba)
    cal_log_loss_val = log_loss(y_val, cal_val_pred_proba)

    if logger:
        logger.info(f"Calibrated metrics - Brier: {cal_brier:.4f}, Log Loss: {cal_log_loss_val:.4f}")

        # Calculate improvement
        brier_improvement = (orig_brier - cal_brier) / orig_brier * 100
        logloss_improvement = (orig_log_loss_val - cal_log_loss_val) / orig_log_loss_val * 100

        logger.info(f"Brier score improvement: {brier_improvement:.2f}%")
        logger.info(f"Log loss improvement: {logloss_improvement:.2f}%")

    # Create metrics dictionary
    calibration_metrics = {
        'original': {
            'brier_score': orig_brier,
            'log_loss': orig_log_loss_val
        },
        'calibrated': {
            'brier_score': cal_brier,
            'log_loss': cal_log_loss_val
        },
        'improvement': {
            'brier_score': orig_brier - cal_brier,
            'brier_score_pct': (orig_brier - cal_brier) / orig_brier * 100,
            'log_loss': orig_log_loss_val - cal_log_loss_val,
            'log_loss_pct': (orig_log_loss_val - cal_log_loss_val) / orig_log_loss_val * 100
        }
    }

    return calibrated_model, calibration_metrics

def compare_calibration(original_model, calibrated_model, X_val, y_val, output_dir, logger):
    """
    Compare calibration of original and calibrated models

    Parameters:
    -----------
    original_model : object
        Original model
    calibrated_model : object
        Calibrated model
    X_val : pandas.DataFrame
        Validation features
    y_val : pandas.Series
        Validation target
    output_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Calibration comparison results
    """
    logger.info("Comparing calibration of original and calibrated models")

    # Get probabilities
    orig_probs = original_model.predict_proba(X_val)[:, 1]
    cal_probs = calibrated_model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    orig_brier = brier_score_loss(y_val, orig_probs)
    cal_brier = brier_score_loss(y_val, cal_probs)

    orig_log_loss_val = log_loss(y_val, orig_probs)
    cal_log_loss_val = log_loss(y_val, cal_probs)

    # Create calibration curves
    orig_fraction, orig_mean_pred, _ = plot_calibration_curve(
        y_val, orig_probs, output_dir, logger, "Original Model", n_bins=10
    )

    cal_fraction, cal_mean_pred, _ = plot_calibration_curve(
        y_val, cal_probs, output_dir, logger, "Calibrated Model", n_bins=10
    )

    # Create reliability diagrams
    orig_bin_accs, orig_bin_confs, orig_bin_counts = create_reliability_diagram(
        y_val, orig_probs, output_dir, logger, "Original Model", n_bins=10
    )

    cal_bin_accs, cal_bin_confs, cal_bin_counts = create_reliability_diagram(
        y_val, cal_probs, output_dir, logger, "Calibrated Model", n_bins=10
    )

    # Plot comparison
    plt.figure(figsize=(10, 8))
    plt.plot(orig_mean_pred, orig_fraction, marker='o', linewidth=2,
             label=f'Original Model (Brier={orig_brier:.4f})')
    plt.plot(cal_mean_pred, cal_fraction, marker='s', linewidth=2,
             label=f'Calibrated Model (Brier={cal_brier:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

    plt.xlabel('Predicted probability')
    plt.ylabel('True probability in each bin')
    plt.title('Calibration Curve Comparison')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_comparison.png'), dpi=300)
    plt.close()

    # Create comparison results
    comparison_results = {
        'original': {
            'brier_score': orig_brier,
            'log_loss': orig_log_loss_val,
            'calibration_curve': {
                'fraction_of_positives': orig_fraction,
                'mean_predicted_value': orig_mean_pred
            },
            'reliability_diagram': {
                'bin_accuracies': orig_bin_accs,
                'bin_confidences': orig_bin_confs,
                'bin_counts': orig_bin_counts
            }
        },
        'calibrated': {
            'brier_score': cal_brier,
            'log_loss': cal_log_loss_val,
            'calibration_curve': {
                'fraction_of_positives': cal_fraction,
                'mean_predicted_value': cal_mean_pred
            },
            'reliability_diagram': {
                'bin_accuracies': cal_bin_accs,
                'bin_confidences': cal_bin_confs,
                'bin_counts': cal_bin_counts
            }
        },
        'improvement': {
            'brier_score': orig_brier - cal_brier,
            'brier_score_pct': (orig_brier - cal_brier) / orig_brier * 100,
            'log_loss': orig_log_loss_val - cal_log_loss_val,
            'log_loss_pct': (orig_log_loss_val - cal_log_loss_val) / orig_log_loss_val * 100
        }
    }

    # Log improvements
    logger.info(f"Calibration improvement - Brier: {comparison_results['improvement']['brier_score_pct']:.2f}%, "
                f"Log Loss: {comparison_results['improvement']['log_loss_pct']:.2f}%")

    return comparison_results


def analyze_probability_distribution(y_prob, y_true, output_dir, logger, model_name="Model"):
    """
    Analyze distribution of predicted probabilities

    Parameters:
    -----------
    y_prob : array-like
        Predicted probabilities
    y_true : array-like
        True binary labels
    output_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    model_name : str
        Name of the model for plot title

    Returns:
    --------
    dict
        Probability distribution statistics
    """
    logger.info(f"Analyzing probability distribution for {model_name}")

    # Create dataframe
    df = pd.DataFrame({
        'probability': y_prob,
        'actual': y_true
    })

    # Calculate statistics
    stats = {
        'overall': {
            'mean': y_prob.mean(),
            'std': y_prob.std(),
            'min': y_prob.min(),
            'max': y_prob.max(),
            'median': np.median(y_prob)
        },
        'by_class': {
            0: {
                'mean': df[df['actual'] == 0]['probability'].mean(),
                'std': df[df['actual'] == 0]['probability'].std(),
                'min': df[df['actual'] == 0]['probability'].min(),
                'max': df[df['actual'] == 0]['probability'].max(),
                'median': df[df['actual'] == 0]['probability'].median()
            },
            1: {
                'mean': df[df['actual'] == 1]['probability'].mean(),
                'std': df[df['actual'] == 1]['probability'].std(),
                'min': df[df['actual'] == 1]['probability'].min(),
                'max': df[df['actual'] == 1]['probability'].max(),
                'median': df[df['actual'] == 1]['probability'].median()
            }
        }
    }

    # Log statistics
    logger.info(f"Probability statistics for {model_name}:")
    logger.info(f"  Overall - Mean: {stats['overall']['mean']:.4f}, Std: {stats['overall']['std']:.4f}")
    logger.info(f"  Class 0 - Mean: {stats['by_class'][0]['mean']:.4f}, Std: {stats['by_class'][0]['std']:.4f}")
    logger.info(f"  Class 1 - Mean: {stats['by_class'][1]['mean']:.4f}, Std: {stats['by_class'][1]['std']:.4f}")

    # Create distribution plot
    plt.figure(figsize=(12, 6))

    # Overall distribution
    plt.subplot(1, 2, 1)
    sns.histplot(y_prob, bins=20, kde=True)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Decision threshold')
    plt.xlabel('Predicted probability')
    plt.ylabel('Count')
    plt.title(f'Overall Probability Distribution - {model_name}')
    plt.legend()

    # Distribution by class
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='probability', hue='actual', bins=20, kde=True,
                 element="step", common_norm=False,
                 palette={0: 'red', 1: 'green'})
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision threshold')
    plt.xlabel('Predicted probability')
    plt.ylabel('Density')
    plt.title(f'Probability Distribution by Class - {model_name}')
    plt.legend(title='Actual', labels=['Outside NBE (0)', 'Within NBE (1)', 'Threshold'])

    plt.tight_layout()
    model_filename = model_name.replace(' ', '_').lower()
    plt.savefig(os.path.join(output_dir, f'probability_distribution_{model_filename}.png'), dpi=300)
    plt.close()

    # Create separation plot (sorted probabilities by class)
    plt.figure(figsize=(12, 6))

    # Sort by probability
    df_sorted = df.sort_values('probability')

    # Create colormap
    cmap = plt.cm.RdYlGn
    colors = [cmap(0) if c == 0 else cmap(1) for c in df_sorted['actual']]

    # Plot bars
    plt.bar(range(len(df_sorted)), df_sorted['probability'], width=1.0, color=colors,
            linewidth=0, alpha=0.8)

    # Add threshold line
    plt.axhline(y=0.5, color='black', linestyle='--', label='Decision threshold')

    plt.xlabel('Instances (sorted by predicted probability)')
    plt.ylabel('Predicted probability')
    plt.title(f'Separation Plot - {model_name}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'separation_plot_{model_filename}.png'), dpi=300)
    plt.close()

    return stats


def evaluate_threshold_metrics(y_true, y_prob, output_dir, logger, model_name="Model", thresholds=None):
    """
    Evaluate model performance at different decision thresholds

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    output_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    model_name : str
        Name of the model for plot title
    thresholds : array-like
        Decision thresholds to evaluate (if None, uses np.linspace(0, 1, 101))

    Returns:
    --------
    pandas.DataFrame
        Threshold metrics dataframe
    """
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

    logger.info(f"Evaluating threshold metrics for {model_name}")

    # Define thresholds if not provided
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    # Calculate metrics for each threshold
    results = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        # Skip if all predictions are the same class (causes division by zero)
        if np.unique(y_pred).size == 1:
            continue

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # Convert to dataframe
    df_thresholds = pd.DataFrame(results)

    # Save to CSV
    df_thresholds.to_csv(os.path.join(output_dir, f'threshold_metrics_{model_name.replace(" ", "_").lower()}.csv'),
                         index=False)

    # Create plot
    plt.figure(figsize=(12, 8))

    plt.plot(df_thresholds['threshold'], df_thresholds['accuracy'], label='Accuracy')
    plt.plot(df_thresholds['threshold'], df_thresholds['precision'], label='Precision')
    plt.plot(df_thresholds['threshold'], df_thresholds['recall'], label='Recall')
    plt.plot(df_thresholds['threshold'], df_thresholds['f1'], label='F1')

    # Add vertical line at threshold=0.5
    plt.axvline(x=0.5, color='black', linestyle='--', label='Default threshold (0.5)')

    plt.xlabel('Decision Threshold')
    plt.ylabel('Metric Value')
    plt.title(f'Performance Metrics by Decision Threshold - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    model_filename = model_name.replace(' ', '_').lower()
    plt.savefig(os.path.join(output_dir, f'threshold_metrics_{model_filename}.png'), dpi=300)
    plt.close()

    # Find optimal thresholds
    optimal_thresholds = {
        'accuracy': df_thresholds.loc[df_thresholds['accuracy'].idxmax(), 'threshold'],
        'precision': df_thresholds.loc[df_thresholds['precision'].idxmax(), 'threshold'],
        'recall': df_thresholds.loc[df_thresholds['recall'].idxmax(), 'threshold'],
        'f1': df_thresholds.loc[df_thresholds['f1'].idxmax(), 'threshold']
    }

    logger.info(f"Optimal thresholds for {model_name}:")
    for metric, threshold in optimal_thresholds.items():
        logger.info(f"  {metric.capitalize()}: {threshold:.4f}")

    return df_thresholds, optimal_thresholds


def run_probability_calibration_analysis(model, datasets, output_dir, logger, calibrate=True):
    """
    Run complete probability calibration analysis

    Parameters:
    -----------
    model : object
        Trained model
    datasets : dict
        Dictionary with datasets
    output_dir : str
        Output directory
    logger : logging.Logger
        Logger instance
    calibrate : bool
        Whether to calibrate the model

    Returns:
    --------
    dict
        Calibration analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get data
    X_train = datasets.get('X_train_scaled', datasets.get('X_train'))
    y_train = datasets['y_train']
    X_val = datasets.get('X_val_scaled', datasets.get('X_val'))
    y_val = datasets['y_val']
    X_test = datasets.get('X_test_scaled', datasets.get('X_test'))
    y_test = datasets['y_test']

    # Evaluate current model calibration
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    # Create calibration curve for original model
    plot_calibration_curve(y_val, y_val_pred_proba, output_dir, logger, "Original Model")

    # Create reliability diagram for original model
    create_reliability_diagram(y_val, y_val_pred_proba, output_dir, logger, "Original Model")

    # Analyze probability distribution
    prob_stats = analyze_probability_distribution(y_val_pred_proba, y_val, output_dir, logger, "Original Model")

    # Evaluate threshold metrics
    threshold_metrics, optimal_thresholds = evaluate_threshold_metrics(
        y_val, y_val_pred_proba, output_dir, logger, "Original Model"
    )

    results = {
        'original_model': {
            'probability_stats': prob_stats,
            'threshold_metrics': threshold_metrics,
            'optimal_thresholds': optimal_thresholds
        }
    }

    # Calibrate model if requested
    if calibrate:
        # Calibrate with isotonic regression
        isotonic_model, isotonic_metrics = calibrate_probabilities(
            model, X_train, y_train, X_val, y_val, 'isotonic', logger
        )

        # Calibrate with sigmoid method (Platt scaling)
        sigmoid_model, sigmoid_metrics = calibrate_probabilities(
            model, X_train, y_train, X_val, y_val, 'sigmoid', logger
        )

        # Compare calibration
        isotonic_comparison = compare_calibration(
            model, isotonic_model, X_val, y_val, output_dir, logger
        )

        # Choose best calibrated model based on Brier score
        if isotonic_metrics['calibrated']['brier_score'] <= sigmoid_metrics['calibrated']['brier_score']:
            logger.info("Isotonic calibration selected (lower Brier score)")
            best_calibrated_model = isotonic_model
            best_method = 'isotonic'
        else:
            logger.info("Sigmoid calibration selected (lower Brier score)")
            best_calibrated_model = sigmoid_model
            best_method = 'sigmoid'

        # Analyze best calibrated model
        y_val_cal_proba = best_calibrated_model.predict_proba(X_val)[:, 1]

        # Analyze probability distribution for calibrated model
        cal_prob_stats = analyze_probability_distribution(
            y_val_cal_proba, y_val, output_dir, logger, "Calibrated Model"
        )

        # Evaluate threshold metrics for calibrated model
        cal_threshold_metrics, cal_optimal_thresholds = evaluate_threshold_metrics(
            y_val, y_val_cal_proba, output_dir, logger, "Calibrated Model"
        )

        # Save calibrated model
        calibrated_model_path = os.path.join(output_dir, 'calibrated_model.pkl')
        with open(calibrated_model_path, 'wb') as f:
            pickle.dump(best_calibrated_model, f)

        # Also save with joblib
        joblib.dump(best_calibrated_model, os.path.join(output_dir, 'calibrated_model.joblib'))

        logger.info(f"Calibrated model saved to {calibrated_model_path}")

        # Add calibration results
        results['calibration'] = {
            'isotonic': isotonic_metrics,
            'sigmoid': sigmoid_metrics,
            'best_method': best_method,
            'comparison': isotonic_comparison,
            'calibrated_model': {
                'probability_stats': cal_prob_stats,
                'threshold_metrics': cal_threshold_metrics,
                'optimal_thresholds': cal_optimal_thresholds
            }
        }

    # Save results
    results_path = os.path.join(output_dir, 'calibration_analysis_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"Calibration analysis results saved to {results_path}")

    # Create summary report
    summary = {
        'original_model': {
            'brier_score': brier_score_loss(y_val, y_val_pred_proba),
            'log_loss': log_loss(y_val, y_val_pred_proba),
            'optimal_thresholds': optimal_thresholds
        }
    }

    if calibrate:
        summary['calibrated_model'] = {
            'method': best_method,
            'brier_score': results['calibration'][best_method]['calibrated']['brier_score'],
            'log_loss': results['calibration'][best_method]['calibrated']['log_loss'],
            'brier_improvement': results['calibration'][best_method]['improvement']['brier_score_pct'],
            'log_loss_improvement': results['calibration'][best_method]['improvement']['log_loss_pct'],
            'optimal_thresholds': cal_optimal_thresholds
        }

    # Save summary as JSON
    import json
    with open(os.path.join(output_dir, 'calibration_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    return results


if __name__ == "__main__":
    from utils.project_setup import setup_logging, create_project_structure
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Probability calibration analysis")
    parser.add_argument("--datasets_path", type=str, help="Path to prepared datasets")
    parser.add_argument("--model_path", type=str, help="Path to refined model")
    parser.add_argument("--no_calibrate", action="store_true", help="Skip model calibration")
    args = parser.parse_args()

    # Create project structure
    log_dir, plot_dir, model_dir = create_project_structure()

    # Setup logging
    logger = setup_logging(log_dir, logger_name="probability_calibration")
    logger.info("=== PROBABILITY CALIBRATION ANALYSIS ===")

    # Find paths if not provided
    if not args.datasets_path or not args.model_path:
        # Try to find the most recent files
        model_base = os.environ.get('MODEL_FOLDER', 'models')
        model_tuning_dir = os.path.join(model_base, "04_model_tuning")

        if os.path.exists(model_tuning_dir):
            subdirs = [os.path.join(model_tuning_dir, d) for d in os.listdir(model_tuning_dir)
                       if os.path.isdir(os.path.join(model_tuning_dir, d))]
            if subdirs:
                latest_subdir = max(subdirs, key=os.path.getmtime)

                # Try to find the refined model
                for model_name in ['refined_model.pkl', 'refined_model_prod.joblib', 'final_model.pkl']:
                    potential_model_path = os.path.join(latest_subdir, '04_model_tuning', model_name)
                    if os.path.exists(potential_model_path):
                        args.model_path = args.model_path or potential_model_path
                        break

                # If we can't find the model in the latest dir, look for prepared datasets
                if not args.datasets_path:
                    # Go back to step 3 for datasets
                    model_dev_dir = os.path.join(model_base, "03_model_development")
                    if os.path.exists(model_dev_dir):
                        dev_subdirs = [os.path.join(model_dev_dir, d) for d in os.listdir(model_dev_dir)
                                       if os.path.isdir(os.path.join(model_dev_dir, d))]
                        if dev_subdirs:
                            latest_dev_subdir = max(dev_subdirs, key=os.path.getmtime)
                            datasets_path = os.path.join(latest_dev_subdir, "prepared_datasets.pkl")
                            if os.path.exists(datasets_path):
                                args.datasets_path = datasets_path

    # Check if paths exist
    if not args.datasets_path or not os.path.exists(args.datasets_path):
        raise ValueError(f"Datasets not found at {args.datasets_path}")
    if not args.model_path or not os.path.exists(args.model_path):
        raise ValueError(f"Model not found at {args.model_path}")

    # Log paths
    logger.info(f"Using datasets from: {args.datasets_path}")
    logger.info(f"Using model from: {args.model_path}")

    # Load datasets and model
    datasets = load_data(args.datasets_path, logger)
    model = load_model(args.model_path, logger)

    # Create output directory
    output_dir = os.path.join(model_dir, "probability_calibration")
    os.makedirs(output_dir, exist_ok=True)

    # Run calibration analysis
    results = run_probability_calibration_analysis(
        model, datasets, output_dir, logger, not args.no_calibrate
    )

    logger.info("=== PROBABILITY CALIBRATION ANALYSIS COMPLETE ===")
    logger.info(f"Results saved to: {output_dir}")
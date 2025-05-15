"""
Model refinement module for the NBE prediction model.
Combines hyperparameter tuning and cross-validation to refine the best model.
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, brier_score_loss, log_loss,
                             confusion_matrix, classification_report)

# Import from our modules
from hyperparameter_tuning import load_data, run_hyperparameter_tuning
from cross_validation import run_patient_cross_validation


def load_final_model(model_path, logger):
    """
    Load the final model from step 3

    Parameters:
    -----------
    model_path : str
        Path to the final model
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Dictionary with model information
    """
    logger.info(f"Loading final model from {model_path}")
    with open(model_path, 'rb') as f:
        final_model_info = pickle.load(f)
    return final_model_info


def compare_models(final_model_info, optimized_model, cv_metrics, datasets, logger, output_dir):
    """
    Compare the final model from step 3 with the optimized model from step 4

    Parameters:
    -----------
    final_model_info : dict
        Dictionary with final model information
    optimized_model : object
        Optimized model
    cv_metrics : dict
        Cross-validation metrics
    datasets : dict
        Dictionary with datasets
    logger : logging.Logger
        Logger instance
    output_dir : str
        Output directory

    Returns:
    --------
    dict
        Comparison results
    """
    logger.info("Comparing final model with optimized model")

    # Extract models
    final_model = final_model_info['model']

    # Get test data
    X_test = datasets.get('X_test_scaled', datasets['X_test'])
    y_test = datasets['y_test']

    # Evaluate final model on test set
    final_y_pred = final_model.predict(X_test)
    final_y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    final_metrics = {
        'accuracy': accuracy_score(y_test, final_y_pred),
        'precision': precision_score(y_test, final_y_pred),
        'recall': recall_score(y_test, final_y_pred),
        'f1': f1_score(y_test, final_y_pred),
        'roc_auc': roc_auc_score(y_test, final_y_pred_proba),
        'brier_score': brier_score_loss(y_test, final_y_pred_proba),
        'log_loss': log_loss(y_test, final_y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, final_y_pred),
        'classification_report': classification_report(y_test, final_y_pred)
    }

    # Evaluate optimized model on test set
    opt_y_pred = optimized_model.predict(X_test)
    opt_y_pred_proba = optimized_model.predict_proba(X_test)[:, 1]

    opt_metrics = {
        'accuracy': accuracy_score(y_test, opt_y_pred),
        'precision': precision_score(y_test, opt_y_pred),
        'recall': recall_score(y_test, opt_y_pred),
        'f1': f1_score(y_test, opt_y_pred),
        'roc_auc': roc_auc_score(y_test, opt_y_pred_proba),
        # Continuing model_refinement.py

        'brier_score': brier_score_loss(y_test, opt_y_pred_proba),
        'log_loss': log_loss(y_test, opt_y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, opt_y_pred),
        'classification_report': classification_report(y_test, opt_y_pred)
    }

    # Compare metrics
    comparison = {
        'final_model': final_metrics,
        'optimized_model': opt_metrics,
        'cross_validation': cv_metrics
    }

    # Log comparison
    logger.info("Final model vs. Optimized model performance:")
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score', 'log_loss']

    for metric in metrics_to_compare:
        logger.info(f"  {metric}:")
        logger.info(f"    Final model: {final_metrics[metric]:.4f}")
        logger.info(f"    Optimized model: {opt_metrics[metric]:.4f}")
        logger.info(f"    CV mean: {cv_metrics.get(f'{metric}_mean', 'N/A')}")

    # Create comparison plot
    plt.figure(figsize=(12, 8))

    # Create dataframe for plotting
    df_compare = pd.DataFrame({
        'Metric': metrics_to_compare * 2,
        'Model': ['Final Model'] * len(metrics_to_compare) + ['Optimized Model'] * len(metrics_to_compare),
        'Value': [final_metrics[m] for m in metrics_to_compare] + [opt_metrics[m] for m in metrics_to_compare]
    })

    # Plot
    sns.barplot(x='Metric', y='Value', hue='Model', data=df_compare)
    plt.title('Final Model vs. Optimized Model Performance')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

    # Save comparison
    comparison_path = os.path.join(output_dir, 'model_comparison.pkl')
    with open(comparison_path, 'wb') as f:
        pickle.dump(comparison, f)

    logger.info(f"Model comparison saved to {comparison_path}")

    return comparison


def select_best_model(comparison, logger):
    """
    Select the best model based on comparison

    Parameters:
    -----------
    comparison : dict
        Comparison results
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    str
        Name of the best model ('final_model' or 'optimized_model')
    """
    logger.info("Selecting the best model based on metrics")

    # Get metrics
    final_metrics = comparison['final_model']
    opt_metrics = comparison['optimized_model']

    # Compare ROC AUC (primary metric)
    if opt_metrics['roc_auc'] > final_metrics['roc_auc']:
        best_model = 'optimized_model'
        improvement = (opt_metrics['roc_auc'] - final_metrics['roc_auc']) / final_metrics['roc_auc'] * 100
        logger.info(f"Optimized model selected (ROC AUC improved by {improvement:.2f}%)")
    else:
        best_model = 'final_model'
        difference = (final_metrics['roc_auc'] - opt_metrics['roc_auc']) / final_metrics['roc_auc'] * 100
        logger.info(f"Final model retained (Optimized model ROC AUC lower by {difference:.2f}%)")

    return best_model


def save_refined_model(best_model_name, final_model, optimized_model, output_dir, logger):
    """
    Save the refined model

    Parameters:
    -----------
    best_model_name : str
        Name of the best model
    final_model : object
        Final model from step 3
    optimized_model : object
        Optimized model from step 4
    output_dir : str
        Output directory
    logger : logging.Logger
        Logger instance
    """
    # Select model to save
    if best_model_name == 'optimized_model':
        model_to_save = optimized_model
        model_name = "refined_model"
    else:
        model_to_save = final_model
        model_name = "final_model_validated"

    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_to_save, f)

    # Also save as joblib for production
    import joblib
    joblib_path = os.path.join(output_dir, f"{model_name}_prod.joblib")
    joblib.dump(model_to_save, joblib_path)

    logger.info(f"Best model ({best_model_name}) saved as:")
    logger.info(f"  - {model_path}")
    logger.info(f"  - {joblib_path}")


def run_model_refinement(datasets_path, final_model_path, output_base_dir, logger, n_trials=100, timeout=600):
    """
    Run the complete model refinement pipeline

    Parameters:
    -----------
    datasets_path : str
        Path to the prepared datasets
    final_model_path : str
        Path to the final model from step 3
    output_base_dir : str
        Base output directory
    logger : logging.Logger
        Logger instance
    n_trials : int
        Number of trials for hyperparameter optimization
    timeout : int
        Timeout for hyperparameter optimization in seconds

    Returns:
    --------
    dict
        Refinement results
    """
    # Create output directories
    hp_dir = os.path.join(output_base_dir, "hyperparameter_tuning")
    cv_dir = os.path.join(output_base_dir, "cross_validation")

    # Load data and model
    datasets = load_data(datasets_path, logger)
    final_model_info = load_final_model(final_model_path, logger)

    # Step 1: Hyperparameter Optimization
    logger.info("Step 1: Hyperparameter Optimization")
    optimized_model, best_params, _ = run_hyperparameter_tuning(
        datasets_path, hp_dir, logger, n_trials=n_trials, timeout=timeout
    )

    # Step 2: Patient-level Cross-Validation
    logger.info("Step 2: Patient-level Cross-Validation")
    cv_metrics = run_patient_cross_validation(
        datasets_path, cv_dir, model_params=best_params, logger=logger, n_splits=5
    )

    # Step 3: Model Comparison
    logger.info("Step 3: Model Comparison")
    comparison = compare_models(
        final_model_info, optimized_model, cv_metrics, datasets, logger, output_base_dir
    )

    # Step 4: Select Best Model
    best_model_name = select_best_model(comparison, logger)

    # Step 5: Save Refined Model
    save_refined_model(
        best_model_name,
        final_model_info['model'],
        optimized_model,
        output_base_dir,
        logger
    )

    # Return results
    refinement_results = {
        'best_model': best_model_name,
        'comparison': comparison,
        'hyperparameters': best_params,
        'cv_metrics': cv_metrics
    }

    # Save refinement results
    results_path = os.path.join(output_base_dir, "refinement_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(refinement_results, f)

    logger.info(f"Refinement results saved to {results_path}")

    return refinement_results


if __name__ == "__main__":
    # This allows the module to be run as a script for testing
    from utils.project_setup import setup_logging, create_project_structure

    # Create project structure
    log_dir, plot_dir, model_dir = create_project_structure()

    # Setup logging
    logger = setup_logging(log_dir, logger_name="model_refinement")
    logger.info("=== MODEL REFINEMENT ===")

    # Find the most recent data and model
    model_base = os.environ.get('MODEL_FOLDER', 'models')
    model_dev_dir = os.path.join(model_base, "03_model_development")

    if os.path.exists(model_dev_dir):
        subdirs = [os.path.join(model_dev_dir, d) for d in os.listdir(model_dev_dir)
                   if os.path.isdir(os.path.join(model_dev_dir, d))]
        if subdirs:
            latest_subdir = max(subdirs, key=os.path.getmtime)
            datasets_path = os.path.join(latest_subdir, "prepared_datasets.pkl")
            final_model_path = os.path.join(latest_subdir, "final_model.pkl")

            if os.path.exists(datasets_path) and os.path.exists(final_model_path):
                # Create output directory
                output_dir = os.path.join(model_dir, "model_refinement")
                os.makedirs(output_dir, exist_ok=True)

                # Run with fewer trials and shorter timeout for demonstration
                refinement_results = run_model_refinement(
                    datasets_path, final_model_path, output_dir, logger, n_trials=20, timeout=300
                )

                logger.info("=== MODEL REFINEMENT COMPLETE ===")
            else:
                logger.error(f"Required files not found in {latest_subdir}")
        else:
            logger.error(f"No subdirectories found in {model_dev_dir}")
    else:
        logger.error(f"Model development directory not found at {model_dev_dir}")
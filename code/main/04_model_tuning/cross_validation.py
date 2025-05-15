"""
Cross-validation module for the NBE prediction model.
Implements patient-level cross-validation to get robust performance estimates.
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
                             confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, auc)


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
    Load a trained model

    Parameters:
    -----------
    model_path : str
        Path to the model
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    object
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def create_patient_folds(X, patient_ids, n_splits=5, random_state=42):
    """
    Create cross-validation folds at the patient level

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    patient_ids : pandas.Series
        Patient identifiers
    n_splits : int
        Number of folds
    random_state : int
        Random seed

    Returns:
    --------
    list
        List of tuples (train_indices, val_indices)
    """
    # Get unique patient IDs
    unique_patients = np.unique(patient_ids)

    # Shuffle patients
    np.random.seed(random_state)
    np.random.shuffle(unique_patients)

    # Split patients into folds
    patient_folds = np.array_split(unique_patients, n_splits)

    # Create folds at the observation level
    folds = []
    for i in range(n_splits):
        val_patients = patient_folds[i]
        val_indices = np.where(np.isin(patient_ids, val_patients))[0]
        train_indices = np.where(~np.isin(patient_ids, val_patients))[0]
        folds.append((train_indices, val_indices))

    return folds


def evaluate_fold(X, y, train_indices, val_indices, model_params=None):
    """
    Evaluate a single fold

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    train_indices : numpy.ndarray
        Indices for training
    val_indices : numpy.ndarray
        Indices for validation
    model_params : dict
        Model hyperparameters (if None, use default)

    Returns:
    --------
    dict
        Performance metrics
    model
        Trained model
    """
    # Split data
    X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
    y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

    # Create and train model
    if model_params is None:
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(**model_params, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    # Make predictions
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_pred_proba),
        'brier_score': brier_score_loss(y_val, y_val_pred_proba),
        'log_loss': log_loss(y_val, y_val_pred_proba),
        'confusion_matrix': confusion_matrix(y_val, y_val_pred),
        'y_val': y_val,
        'y_val_pred': y_val_pred,
        'y_val_pred_proba': y_val_pred_proba
    }

    return metrics, model


def run_cross_validation(X, y, patient_ids, n_splits=5, model_params=None, logger=None):
    """
    Run patient-level cross-validation

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    patient_ids : pandas.Series
        Patient identifiers
    n_splits : int
        Number of folds
    model_params : dict
        Model hyperparameters (if None, use default)
    logger : logging.Logger
        Logger instance (optional)

    Returns:
    --------
    list
        List of dictionaries with fold metrics
    """
    if logger:
        logger.info(f"Running {n_splits}-fold patient-level cross-validation")

    # Create folds
    folds = create_patient_folds(X, patient_ids, n_splits)

    # Evaluate each fold
    fold_results = []
    for i, (train_indices, val_indices) in enumerate(folds):
        if logger:
            logger.info(f"Evaluating fold {i + 1}/{n_splits}")

        # Evaluate fold
        metrics, model = evaluate_fold(X, y, train_indices, val_indices, model_params)

        # Add fold index
        metrics['fold'] = i

        # Add to results
        fold_results.append(metrics)

        if logger:
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")

    return fold_results


def aggregate_metrics(fold_results, logger=None):
    """
    Aggregate metrics across folds

    Parameters:
    -----------
    fold_results : list
        List of dictionaries with fold metrics
    logger : logging.Logger
        Logger instance (optional)

    Returns:
    --------
    dict
        Aggregated metrics
    """
    # Metrics to aggregate
    metrics_to_agg = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score', 'log_loss']

    # Initialize results
    agg_metrics = {}

    # Aggregate metrics
    for metric in metrics_to_agg:
        values = [fold[metric] for fold in fold_results]
        agg_metrics[f'{metric}_mean'] = np.mean(values)
        agg_metrics[f'{metric}_std'] = np.std(values)
        agg_metrics[f'{metric}_min'] = np.min(values)
        agg_metrics[f'{metric}_max'] = np.max(values)

    if logger:
        logger.info("Aggregated cross-validation metrics:")
        for metric in metrics_to_agg:
            logger.info(f"  {metric}: {agg_metrics[f'{metric}_mean']:.4f} ± {agg_metrics[f'{metric}_std']:.4f}")

    return agg_metrics


def plot_cv_metrics(fold_results, output_dir, logger=None):
    """
    Plot cross-validation metrics

    Parameters:
    -----------
    fold_results : list
        List of dictionaries with fold metrics
    output_dir : str
        Output directory
    logger : logging.Logger
        Logger instance (optional)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Metrics to plot
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score', 'log_loss']

    # Create dataframe with metrics
    df_metrics = pd.DataFrame([
        {
            'fold': fold['fold'],
            **{metric: fold[metric] for metric in metrics_to_plot}
        }
        for fold in fold_results
    ])

    # Plot metrics
    plt.figure(figsize=(12, 8))
    df_metrics[['fold'] + metrics_to_plot].set_index('fold').plot(kind='bar', figsize=(12, 8))
    plt.title('Cross-Validation Metrics by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_metrics_by_fold.png'), dpi=300)
    plt.close()

    # Box plots of metrics
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_metrics[metrics_to_plot])
    plt.title('Cross-Validation Metrics Distribution')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_metrics_distribution.png'), dpi=300)
    plt.close()

    if logger:
        logger.info(f"Cross-validation plots saved to {output_dir}")


def run_patient_cross_validation(datasets_path, output_dir, model_params=None, logger=None, n_splits=5):
    """
    Run the complete patient-level cross-validation pipeline

    Parameters:
    -----------
    datasets_path : str
        Path to the prepared datasets
    output_dir : str
        Output directory
    model_params : dict
        Model hyperparameters (if None, use default)
    logger : logging.Logger
        Logger instance
    n_splits : int
        Number of folds

    Returns:
    --------
    dict
        Aggregated metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    datasets = load_data(datasets_path, logger)

    # Get training data
    X_train = datasets.get('X_train_scaled', datasets['X_train'])
    y_train = datasets['y_train']

    # Get patient IDs (assuming they're in the index of X_train)
    # If patient IDs are stored differently, adjust this
    patient_ids = datasets.get('patient_ids_train', pd.Series([x.split('-')[0] for x in X_train.index.astype(str)]))

    # Run cross-validation
    fold_results = run_cross_validation(X_train, y_train, patient_ids, n_splits, model_params, logger)

    # Aggregate metrics
    agg_metrics = aggregate_metrics(fold_results, logger)

    # Plot metrics
    plot_cv_metrics(fold_results, output_dir, logger)

    # Save results
    results_path = os.path.join(output_dir, 'cv_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'fold_results': fold_results,
            'aggregated_metrics': agg_metrics
        }, f)

    if logger:
        logger.info(f"Cross-validation results saved to {results_path}")

    return agg_metrics


if __name__ == "__main__":
    # This allows the module to be run as a script for testing
    from utils.project_setup import setup_logging, create_project_structure

    # Create project structure
    log_dir, plot_dir, model_dir = create_project_structure()

    # Setup logging
    logger = setup_logging(log_dir, logger_name="cross_validation")
    logger.info("=== PATIENT-LEVEL CROSS-VALIDATION ===")

    # Run cross-validation
    # Find the most recent prepared datasets
    model_base = os.environ.get('MODEL_FOLDER', 'models')
    model_dev_dir = os.path.join(model_base, "03_model_development")

    if os.path.exists(model_dev_dir):
        subdirs = [os.path.join(model_dev_dir, d) for d in os.listdir(model_dev_dir)
                   if os.path.isdir(os.path.join(model_dev_dir, d))]
        if subdirs:
            latest_subdir = max(subdirs, key=os.path.getmtime)
            datasets_path = os.path.join(latest_subdir, "prepared_datasets.pkl")
            if os.path.exists(datasets_path):
                # Create output directory
                output_dir = os.path.join(model_dir, "cross_validation")

                # Run cross-validation
                agg_metrics = run_patient_cross_validation(
                    datasets_path, output_dir, model_params=None, logger=logger, n_splits=5
                )

                logger.info("=== CROSS-VALIDATION COMPLETE ===")
            else:
                logger.error(f"Prepared datasets not found at {datasets_path}")
        else:
            logger.error(f"No subdirectories found in {model_dev_dir}")
    else:
        logger.error(f"Model development directory not found at {model_dev_dir}")
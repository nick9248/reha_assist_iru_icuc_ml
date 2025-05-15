# Import necessary libraries
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, brier_score_loss,
                             confusion_matrix, classification_report)


def train_logistic_regression(X_train, y_train, logger, C=1.0, max_iter=1000, random_state=42):
    """
    Train a logistic regression model

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    logger : logging.Logger
        Logger instance
    C : float
        Inverse of regularization strength
    max_iter : int
        Maximum number of iterations
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    model : LogisticRegression
        Trained model
    train_time : float
        Training time in seconds
    """
    logger.info(f"Training Logistic Regression (C={C}, max_iter={max_iter})")

    # Initialize model
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)

    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def train_decision_tree(X_train, y_train, logger, max_depth=5, min_samples_split=2, random_state=42):
    """
    Train a decision tree model

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    logger : logging.Logger
        Logger instance
    max_depth : int
        Maximum depth of the tree
    min_samples_split : int
        Minimum samples required to split an internal node
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    model : DecisionTreeClassifier
        Trained model
    train_time : float
        Training time in seconds
    """
    logger.info(f"Training Decision Tree (max_depth={max_depth}, min_samples_split={min_samples_split})")

    # Initialize model
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)

    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def evaluate_model(model, X, y, logger, model_name="Model"):
    """
    Evaluate model performance

    Parameters:
    -----------
    model : sklearn model
        Trained model with predict and predict_proba methods
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    logger : logging.Logger
        Logger instance
    model_name : str
        Name of the model for logging

    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    logger.info(f"Evaluating {model_name}")

    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    brier = brier_score_loss(y, y_pred_proba)
    logloss = log_loss(y, y_pred_proba)
    cm = confusion_matrix(y, y_pred)

    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'brier_score': brier,
        'log_loss': logloss,
        'confusion_matrix': cm,
        'classification_report': classification_report(y, y_pred)
    }

    # Log performance
    logger.info(f"{model_name} performance:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")

    # Log confusion matrix
    logger.info(f"Confusion Matrix:")
    logger.info(cm)

    # Log classification report
    logger.info(f"Classification Report:")
    logger.info(metrics['classification_report'])

    return metrics


def train_baseline_models(datasets, logger):
    """
    Train and evaluate baseline models

    Parameters:
    -----------
    datasets : dict
        Dictionary containing train/val/test datasets
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    models : dict
        Dictionary containing trained models and their metrics
    """
    logger.info("Training baseline models")

    # Determine which data to use (scaled if available)
    X_train = datasets.get('X_train_scaled', datasets['X_train'])
    y_train = datasets['y_train']
    X_val = datasets.get('X_val_scaled', datasets['X_val'])
    y_val = datasets['y_val']

    # Check for NaN values in training data
    if y_train.isna().any():
        logger.warning(f"Found {y_train.isna().sum()} NaN values in target variable. Removing rows with NaN targets.")
        # Get indices of non-NaN values
        valid_indices = y_train[~y_train.isna()].index
        # Filter X_train and y_train
        X_train = X_train.loc[valid_indices]
        y_train = y_train.loc[valid_indices]
        logger.info(f"Rows after removing NaN targets: {len(X_train)}")

    # Check for NaN values in validation data
    if y_val.isna().any():
        logger.warning(f"Found {y_val.isna().sum()} NaN values in validation target. Removing rows with NaN targets.")
        # Get indices of non-NaN values
        valid_indices = y_val[~y_val.isna()].index
        # Filter X_val and y_val
        X_val = X_val.loc[valid_indices]
        y_val = y_val.loc[valid_indices]
        logger.info(f"Validation rows after removing NaN targets: {len(X_val)}")

    # Initialize dictionary to store models and performance
    baseline_models = {}

    # Train Logistic Regression
    lr_model, lr_train_time = train_logistic_regression(X_train, y_train, logger)
    lr_metrics = evaluate_model(lr_model, X_val, y_val, logger, "Logistic Regression")

    # Store in dictionary
    baseline_models['Logistic Regression'] = {
        'model': lr_model,
        'train_time': lr_train_time,
        **lr_metrics
    }

    # Train Decision Tree
    dt_model, dt_train_time = train_decision_tree(X_train, y_train, logger)
    dt_metrics = evaluate_model(dt_model, X_val, y_val, logger, "Decision Tree")

    # Store in dictionary
    baseline_models['Decision Tree'] = {
        'model': dt_model,
        'train_time': dt_train_time,
        **dt_metrics
    }

    # Compare models
    logger.info("Baseline Models Comparison:")
    comparison_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score']

    for metric in comparison_metrics:
        logger.info(f"  {metric.capitalize()}:")
        for model_name in baseline_models:
            logger.info(f"    {model_name}: {baseline_models[model_name][metric]:.4f}")

    return baseline_models


# Example usage (when run as a script)
if __name__ == "__main__":
    import os
    import pickle
    from utils.project_setup import create_project_structure, setup_logging

    # Setup project structure and logging
    log_dir, plot_dir, model_dir = create_project_structure()
    logger = setup_logging(log_dir, "baseline_models")

    # Load prepared datasets
    datasets_path = os.environ.get('PREPARED_DATASETS')
    if not datasets_path:
        # Try to find prepared datasets in model directory
        model_base = os.environ.get('MODEL_FOLDER', 'models')
        data_prep_dir = os.path.join(model_base, "03_model_development")

        if os.path.exists(data_prep_dir):
            subdirs = [os.path.join(data_prep_dir, d) for d in os.listdir(data_prep_dir)
                       if os.path.isdir(os.path.join(data_prep_dir, d))]
            if subdirs:
                latest_subdir = max(subdirs, key=os.path.getmtime)
                datasets_path = os.path.join(latest_subdir, "prepared_datasets.pkl")

    if not datasets_path or not os.path.exists(datasets_path):
        logger.error("Prepared datasets not found. Run data_preparation.py first.")
        raise FileNotFoundError("Prepared datasets not found")

    logger.info(f"Loading prepared datasets from: {datasets_path}")
    with open(datasets_path, "rb") as f:
        datasets = pickle.load(f)

    # Train and evaluate baseline models
    baseline_models = train_baseline_models(datasets, logger)

    # Save models and results
    with open(os.path.join(model_dir, "baseline_models.pkl"), "wb") as f:
        pickle.dump(baseline_models, f)

    logger.info(f"Baseline models saved to: {os.path.join(model_dir, 'baseline_models.pkl')}")
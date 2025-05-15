# Import necessary libraries
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, brier_score_loss,
                             confusion_matrix, classification_report)


def train_random_forest(X_train, y_train, logger, n_estimators=100, max_depth=10, min_samples_split=2, random_state=42):
    """
    Train a Random Forest model

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    logger : logging.Logger
        Logger instance
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of the trees
    min_samples_split : int
        Minimum samples required to split an internal node
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    model : RandomForestClassifier
        Trained model
    train_time : float
        Training time in seconds
    """
    logger.info(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")

    # Initialize model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )

    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def train_gradient_boosting(X_train, y_train, logger, n_estimators=100, learning_rate=0.1, max_depth=3,
                            random_state=42):
    """
    Train a Gradient Boosting model

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    logger : logging.Logger
        Logger instance
    n_estimators : int
        Number of boosting stages
    learning_rate : float
        Learning rate shrinks the contribution of each tree
    max_depth : int
        Maximum depth of the trees
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    model : GradientBoostingClassifier
        Trained model
    train_time : float
        Training time in seconds
    """
    logger.info(
        f"Training Gradient Boosting (n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth})")

    # Initialize model
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )

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


def train_advanced_models(datasets, logger, use_smote=True):
    """
    Train and evaluate advanced models

    Parameters:
    -----------
    datasets : dict
        Dictionary containing train/val/test datasets
    logger : logging.Logger
        Logger instance
    use_smote : bool
        Whether to train models with SMOTE-resampled data

    Returns:
    --------
    models : dict
        Dictionary containing trained models and their metrics
    """
    logger.info("Training advanced models")

    # Determine which data to use (scaled if available)
    X_train = datasets.get('X_train_scaled', datasets['X_train'])
    y_train = datasets['y_train']
    X_val = datasets.get('X_val_scaled', datasets['X_val'])
    y_val = datasets['y_val']

    # Initialize dictionary to store models and performance
    advanced_models = {}

    # Train Random Forest
    rf_model, rf_train_time = train_random_forest(X_train, y_train, logger)
    rf_metrics = evaluate_model(rf_model, X_val, y_val, logger, "Random Forest")

    # Store in dictionary
    advanced_models['Random Forest'] = {
        'model': rf_model,
        'train_time': rf_train_time,
        **rf_metrics
    }

    # Train Gradient Boosting
    gb_model, gb_train_time = train_gradient_boosting(X_train, y_train, logger)
    gb_metrics = evaluate_model(gb_model, X_val, y_val, logger, "Gradient Boosting")

    # Store in dictionary
    advanced_models['Gradient Boosting'] = {
        'model': gb_model,
        'train_time': gb_train_time,
        **gb_metrics
    }

    # If SMOTE data is available and requested, train models on it
    if use_smote and 'X_train_smote_scaled' in datasets:
        logger.info("Training models with SMOTE-resampled data")

        X_train_smote = datasets.get('X_train_smote_scaled', datasets.get('X_train_smote'))
        y_train_smote = datasets['y_train_smote']

        # Train Random Forest with SMOTE
        rf_smote_model, rf_smote_train_time = train_random_forest(X_train_smote, y_train_smote, logger)
        rf_smote_metrics = evaluate_model(rf_smote_model, X_val, y_val, logger, "Random Forest with SMOTE")

        # Store in dictionary
        advanced_models['Random Forest with SMOTE'] = {
            'model': rf_smote_model,
            'train_time': rf_smote_train_time,
            **rf_smote_metrics
        }

        # Train Gradient Boosting with SMOTE
        gb_smote_model, gb_smote_train_time = train_gradient_boosting(X_train_smote, y_train_smote, logger)
        gb_smote_metrics = evaluate_model(gb_smote_model, X_val, y_val, logger, "Gradient Boosting with SMOTE")

        # Store in dictionary
        advanced_models['Gradient Boosting with SMOTE'] = {
            'model': gb_smote_model,
            'train_time': gb_smote_train_time,
            **gb_smote_metrics
        }

    # Compare models
    logger.info("Advanced Models Comparison:")
    comparison_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score']

    for metric in comparison_metrics:
        logger.info(f"  {metric.capitalize()}:")
        for model_name in advanced_models:
            logger.info(f"    {model_name}: {advanced_models[model_name][metric]:.4f}")

    return advanced_models


# Example usage (when run as a script)
if __name__ == "__main__":
    import os
    import pickle
    from utils.project_setup import create_project_structure, setup_logging

    # Setup project structure and logging
    log_dir, plot_dir, model_dir = create_project_structure()
    logger = setup_logging(log_dir, "advanced_models")

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

    # Train and evaluate advanced models
    advanced_models = train_advanced_models(datasets, logger)

    # Save models and results
    with open(os.path.join(model_dir, "advanced_models.pkl"), "wb") as f:
        pickle.dump(advanced_models, f)

    logger.info(f"Advanced models saved to: {os.path.join(model_dir, 'advanced_models.pkl')}")
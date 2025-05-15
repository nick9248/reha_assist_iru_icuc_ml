# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss


def calibrate_model(model, X_train, y_train, X_val, y_val, logger, method='isotonic'):
    """
    Calibrate the probability estimates of a model

    Parameters:
    -----------
    model : sklearn estimator
        Model to calibrate
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    X_val : pandas.DataFrame
        Validation features
    y_val : pandas.Series
        Validation target
    logger : logging.Logger
        Logger instance
    method : str
        Calibration method ('isotonic' or 'sigmoid')

    Returns:
    --------
    calibrated_model : CalibratedClassifierCV
        Calibrated model
    calibration_metrics : dict
        Performance metrics before and after calibration
    """
    logger.info(f"Calibrating model using {method} method")

    # Get original probability predictions
    orig_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics before calibration
    orig_brier = brier_score_loss(y_val, orig_proba)
    orig_log_loss = log_loss(y_val, orig_proba)

    logger.info(f"Metrics before calibration:")
    logger.info(f"  Brier Score: {orig_brier:.4f}")
    logger.info(f"  Log Loss: {orig_log_loss:.4f}")

    # Create calibrated classifier
    calibrated_model = CalibratedClassifierCV(model, method=method, cv='prefit')

    # Fit on training data
    calibrated_model.fit(X_train, y_train)

    # Get calibrated probability predictions
    cal_proba = calibrated_model.predict_proba(X_val)[:, 1]

    # Calculate metrics after calibration
    cal_brier = brier_score_loss(y_val, cal_proba)
    cal_log_loss = log_loss(y_val, cal_proba)

    logger.info(f"Metrics after calibration:")
    logger.info(f"  Brier Score: {cal_brier:.4f} ({'improved' if cal_brier < orig_brier else 'worsened'})")
    logger.info(f"  Log Loss: {cal_log_loss:.4f} ({'improved' if cal_log_loss < orig_log_loss else 'worsened'})")

    # Store metrics for comparison
    calibration_metrics = {
        'before_calibration': {
            'brier_score': orig_brier,
            'log_loss': orig_log_loss
        },
        'after_calibration': {
            'brier_score': cal_brier,
            'log_loss': cal_log_loss
        },
        'improvement': {
            'brier_score': orig_brier - cal_brier,
            'log_loss': orig_log_loss - cal_log_loss
        }
    }

    return calibrated_model, calibration_metrics


def calibrate_best_model(all_models, datasets, logger, metric='roc_auc'):
    """
    Identify the best model and calibrate its probabilities

    Parameters:
    -----------
    all_models : dict
        Dictionary of all trained models and their metrics
    datasets : dict
        Dictionary containing datasets
    logger : logging.Logger
        Logger instance
    metric : str
        Metric to use for identifying the best model

    Returns:
    --------
    best_model_name : str
        Name of the best model
    calibrated_model : CalibratedClassifierCV
        Calibrated version of the best model
    calibration_metrics : dict
        Performance metrics before and after calibration
    """
    logger.info(f"Identifying best model based on {metric} and calibrating its probabilities")

    # Identify best model
    if metric == 'brier_score':
        # For Brier score, lower is better
        best_model_name = min(all_models.items(), key=lambda x: x[1][metric])[0]
    else:
        # For other metrics, higher is better
        best_model_name = max(all_models.items(), key=lambda x: x[1][metric])[0]

    best_model = all_models[best_model_name]['model']

    logger.info(f"Best model: {best_model_name} with {metric} = {all_models[best_model_name][metric]:.4f}")

    # Get datasets
    X_train = datasets.get('X_train_scaled', datasets['X_train'])
    y_train = datasets['y_train']
    X_val = datasets.get('X_val_scaled', datasets['X_val'])
    y_val = datasets['y_val']

    # Calibrate the model
    calibrated_model, calibration_metrics = calibrate_model(best_model, X_train, y_train, X_val, y_val, logger)

    return best_model_name, calibrated_model, calibration_metrics


# Example usage (when run as a script)
if __name__ == "__main__":
    import os
    import pickle
    from utils.project_setup import create_project_structure, setup_logging

    # Setup project structure and logging
    log_dir, plot_dir, model_dir = create_project_structure()
    logger = setup_logging(log_dir, "model_calibration")

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

    # Load all models
    all_models_path = os.path.join(model_dir, "all_models.pkl")

    if not os.path.exists(all_models_path):
        logger.error("All models not found. Run model_evaluation.py first.")
        raise FileNotFoundError("All models not found")

    with open(all_models_path, "rb") as f:
        all_models = pickle.load(f)

    # Calibrate best model
    best_model_name, calibrated_model, calibration_metrics = calibrate_best_model(all_models, datasets, logger)

    # Save calibrated model
    calibrated_model_info = {
        'model_name': best_model_name,
        'calibrated_model': calibrated_model,
        'calibration_metrics': calibration_metrics
    }

    with open(os.path.join(model_dir, "calibrated_model.pkl"), "wb") as f:
        pickle.dump(calibrated_model_info, f)

    logger.info(f"Calibrated model saved to: {os.path.join(model_dir, 'calibrated_model.pkl')}")
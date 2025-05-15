# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, brier_score_loss,
                             confusion_matrix, classification_report)


def select_best_model(all_models, calibrated_model_info, datasets, logger, metric='roc_auc'):
    """
    Select the best model based on specified metric and final evaluation

    Parameters:
    -----------
    all_models : dict
        Dictionary of all trained models and their metrics
    calibrated_model_info : dict
        Information about the calibrated model
    datasets : dict
        Dictionary containing datasets
    logger : logging.Logger
        Logger instance
    metric : str
        Metric to use for identifying the best model

    Returns:
    --------
    final_model_info : dict
        Information about the final selected model
    """
    logger.info(f"Selecting the best model based on {metric}")

    # Get validation metrics for all models
    validation_metrics = {}
    for name, model_info in all_models.items():
        validation_metrics[name] = model_info[metric]

    # Add calibrated model
    calibrated_model_name = f"Calibrated {calibrated_model_info['model_name']}"

    # Get data
    X_val = datasets.get('X_val_scaled', datasets['X_val'])
    y_val = datasets['y_val']

    # Get predictions from calibrated model
    calibrated_model = calibrated_model_info['calibrated_model']
    y_val_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]

    # Calculate metric for calibrated model
    if metric == 'roc_auc':
        validation_metrics[calibrated_model_name] = roc_auc_score(y_val, y_val_pred_proba)
    elif metric == 'brier_score':
        validation_metrics[calibrated_model_name] = brier_score_loss(y_val, y_val_pred_proba)
    elif metric == 'log_loss':
        validation_metrics[calibrated_model_name] = log_loss(y_val, y_val_pred_proba)

    # Find best model
    if metric in ['brier_score', 'log_loss']:
        # Lower is better
        best_model_name = min(validation_metrics.items(), key=lambda x: x[1])[0]
    else:
        # Higher is better
        best_model_name = max(validation_metrics.items(), key=lambda x: x[1])[0]

    logger.info(f"Best model based on validation {metric}: {best_model_name}")

    # Get the best model
    if best_model_name == calibrated_model_name:
        best_model = calibrated_model
        logger.info("Calibrated model selected as the best model")
    else:
        best_model = all_models[best_model_name]['model']
        logger.info(f"Original model '{best_model_name}' selected as the best model")

    # Final evaluation on test set
    X_test = datasets.get('X_test_scaled', datasets['X_test'])
    y_test = datasets['y_test']

    # Make predictions
    y_test_pred = best_model.predict(X_test)
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    brier = brier_score_loss(y_test, y_test_pred_proba)
    logloss = log_loss(y_test, y_test_pred_proba)

    # Log test performance
    logger.info("Final model performance on test set:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")

    # Calculate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred)

    logger.info("Confusion Matrix:")
    logger.info(cm)
    logger.info("Classification Report:")
    logger.info(class_report)

    # Create test set metrics dictionary
    test_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'brier_score': brier,
        'log_loss': logloss,
        'confusion_matrix': cm,
        'classification_report': class_report
    }

    # Create final model info dictionary
    final_model_info = {
        'model_name': best_model_name,
        'model': best_model,
        'validation_metric': validation_metrics[best_model_name],
        'test_metrics': test_metrics
    }

    return final_model_info


def create_final_test_plots(final_model, X_test, y_test, plot_dir, logger):
    """
    Create final evaluation plots on the test set

    Parameters:
    -----------
    final_model : object
        The final selected model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    plot_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    """
    logger.info("Creating final test set evaluation plots")

    # Get predictions
    y_test_pred = final_model.predict(X_test)
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]

    # Create confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'final_confusion_matrix.png'), dpi=300)
    plt.close()

    # Create ROC curve
    plt.figure(figsize=(10, 8))
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'final_roc_curve.png'), dpi=300)
    plt.close()

    # Create precision-recall curve
    plt.figure(figsize=(10, 8))
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'final_precision_recall_curve.png'), dpi=300)
    plt.close()

    # Create calibration plot
    plt.figure(figsize=(10, 8))
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_pred_proba, n_bins=10)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot (Test Set)')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'final_calibration_plot.png'), dpi=300)
    plt.close()

    logger.info("Final test set evaluation plots created")


# Example usage (when run as a script)
if __name__ == "__main__":
    import os
    import pickle
    import joblib
    from utils.project_setup import create_project_structure, setup_logging

    # Setup project structure and logging
    log_dir, plot_dir, model_dir = create_project_structure()
    logger = setup_logging(log_dir, "model_selection")

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

    # Load calibrated model
    calibrated_model_path = os.path.join(model_dir, "calibrated_model.pkl")

    if not os.path.exists(calibrated_model_path):
        logger.error("Calibrated model not found. Run model_calibration.py first.")
        raise FileNotFoundError("Calibrated model not found")

    with open(calibrated_model_path, "rb") as f:
        calibrated_model_info = pickle.load(f)

    # Select best model
    final_model_info = select_best_model(all_models, calibrated_model_info, datasets, logger)

    # Create final test plots
    create_final_test_plots(
        final_model_info['model'],
        datasets.get('X_test_scaled', datasets['X_test']),
        datasets['y_test'],
        plot_dir,
        logger
    )

    # Save final model
    with open(os.path.join(model_dir, "final_model.pkl"), "wb") as f:
        pickle.dump(final_model_info, f)

    # Also save the model using joblib for production deployment
    joblib.dump(final_model_info['model'], os.path.join(model_dir, "final_model_prod.joblib"))

    logger.info(f"Final model saved to: {os.path.join(model_dir, 'final_model.pkl')}")
    logger.info(f"Production model saved to: {os.path.join(model_dir, 'final_model_prod.joblib')}")
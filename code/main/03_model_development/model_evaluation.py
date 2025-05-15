# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve,
                           auc, confusion_matrix)
from sklearn.calibration import calibration_curve

def create_roc_curves(models_dict, X_val, y_val, plot_dir, logger):
    """
    Create ROC curves for comparing models

    Parameters:
    -----------
    models_dict : dict
        Dictionary of models with their metrics
    X_val : pandas.DataFrame
        Validation features
    y_val : pandas.Series
        Validation target
    plot_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    """
    logger.info("Creating ROC curves")

    plt.figure(figsize=(12, 8))

    # Plot ROC curve for each model
    for name, model_info in models_dict.items():
        model = model_info['model']
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot the curve
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    # Plot the diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # Set axis limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'roc_curves.png'), dpi=300)
    plt.close()

    logger.info(f"ROC curves saved to: {os.path.join(plot_dir, 'roc_curves.png')}")


def create_precision_recall_curves(models_dict, X_val, y_val, plot_dir, logger):
    """
    Create precision-recall curves for comparing models

    Parameters:
    -----------
    models_dict : dict
        Dictionary of models with their metrics
    X_val : pandas.DataFrame
        Validation features
    y_val : pandas.Series
        Validation target
    plot_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    """
    logger.info("Creating precision-recall curves")

    plt.figure(figsize=(12, 8))

    # Plot precision-recall curve for each model
    for name, model_info in models_dict.items():
        model = model_info['model']
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_val, y_val_pred_proba)
        pr_auc = auc(recall, precision)

        # Plot the curve
        plt.plot(recall, precision, lw=2, label=f'{name} (AUC = {pr_auc:.3f})')

    # Set axis labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'precision_recall_curves.png'), dpi=300)
    plt.close()

    logger.info(f"Precision-recall curves saved to: {os.path.join(plot_dir, 'precision_recall_curves.png')}")


def create_calibration_plots(models_dict, X_val, y_val, plot_dir, logger, n_bins=10):
    """
    Create calibration plots for comparing models

    Parameters:
    -----------
    models_dict : dict
        Dictionary of models with their metrics
    X_val : pandas.DataFrame
        Validation features
    y_val : pandas.Series
        Validation target
    plot_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    n_bins : int
        Number of bins for calibration curve
    """
    logger.info("Creating calibration plots")

    plt.figure(figsize=(12, 8))

    # Plot the perfectly calibrated line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

    # Plot calibration curve for each model
    for name, model_info in models_dict.items():
        model = model_info['model']
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_val, y_val_pred_proba, n_bins=n_bins)

        # Plot the curve
        plt.plot(mean_predicted_value, fraction_of_positives, 's-',
                 label=f'{name} (Brier: {model_info["brier_score"]:.3f})')

    # Set axis labels and title
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves')
    plt.legend(loc="best")

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'calibration_curves.png'), dpi=300)
    plt.close()

    logger.info(f"Calibration plots saved to: {os.path.join(plot_dir, 'calibration_curves.png')}")


def create_confusion_matrix_plot(model, X, y, plot_dir, logger, model_name="Model"):
    """
    Create confusion matrix visualization for a model

    Parameters:
    -----------
    model : sklearn model
        Model with predict method
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    plot_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    model_name : str
        Name of the model for plot title and filename
    """
    logger.info(f"Creating confusion matrix plot for {model_name}")

    # Make predictions
    y_pred = model.predict(X)

    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save plot
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').lower()
    plt.savefig(os.path.join(plot_dir, f'confusion_matrix_{safe_name}.png'), dpi=300)
    plt.close()

    logger.info(f"Confusion matrix plot saved to: {os.path.join(plot_dir, f'confusion_matrix_{safe_name}.png')}")


def create_feature_importance_plot(model, feature_names, plot_dir, logger, model_name="Model", top_n=20):
    """
    Create feature importance visualization

    Parameters:
    -----------
    model : sklearn model
        Model with feature_importances_ attribute or coef_ attribute
    feature_names : list
        List of feature names
    plot_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    model_name : str
        Name of the model for plot title and filename
    top_n : int
        Number of top features to show
    """
    logger.info(f"Creating feature importance plot for {model_name}")

    # Extract feature importance
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
        title = f'Top {top_n} Feature Importance - {model_name}'
        xlabel = 'Importance'

        # Create dataframe for visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Select top features
        plot_df = importance_df.head(top_n)
        y_col = 'Importance'

    elif hasattr(model, 'coef_'):
        # For linear models
        coefficients = model.coef_[0]
        title = f'Top {top_n} Feature Coefficients - {model_name}'
        xlabel = 'Coefficient'

        # Create dataframe for visualization
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        })

        # Use absolute coefficients for sorting
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

        # Select top features
        plot_df = coef_df.head(top_n)[['Feature', 'Coefficient']]
        y_col = 'Coefficient'

    else:
        logger.warning(f"Model {model_name} does not have feature_importances_ or coef_ attribute")
        return

    # Create plot
    plt.figure(figsize=(12, 10))
    plot_df.sort_values(y_col).plot(kind='barh', x='Feature', y=y_col)
    plt.title(title)
    plt.xlabel(xlabel)

    # Save plot
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').lower()
    plt.savefig(os.path.join(plot_dir, f'feature_importance_{safe_name}.png'), dpi=300)
    plt.close()

    logger.info(f"Feature importance plot saved to: {os.path.join(plot_dir, f'feature_importance_{safe_name}.png')}")


def create_model_comparison_plot(models_dict, plot_dir, logger):
    """
    Create comparison plot for model metrics

    Parameters:
    -----------
    models_dict : dict
        Dictionary of models with their metrics
    plot_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    """
    logger.info("Creating model comparison plot")

    # Select metrics for comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score']

    # Create dataframe for the plot
    metrics_df = pd.DataFrame(index=metrics)

    # Fill dataframe with metrics from each model
    for name, model_info in models_dict.items():
        model_metrics = [model_info[metric] for metric in metrics]
        metrics_df[name] = model_metrics

    # For brier score, lower is better, so invert for consistent visualization
    metrics_df.loc['brier_score'] = 1 - metrics_df.loc['brier_score']
    metrics = metrics[:-1] + ['1 - brier_score']  # Update metric names
    metrics_df.index = metrics

    # Create plot
    plt.figure(figsize=(14, 10))
    metrics_df.T.plot(kind='bar', figsize=(14, 8))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.legend(title='Metric')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'model_performance_comparison.png'), dpi=300)
    plt.close()

    logger.info(f"Model comparison plot saved to: {os.path.join(plot_dir, 'model_performance_comparison.png')}")


def evaluate_models(baseline_models, advanced_models, datasets, plot_dir, logger):
    """
    Create comprehensive evaluation visualizations for all models

    Parameters:
    -----------
    baseline_models : dict
        Dictionary of baseline models
    advanced_models : dict
        Dictionary of advanced models
    datasets : dict
        Dictionary containing datasets
    plot_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    all_models : dict
        Combined dictionary of all models
    """
    logger.info("Starting comprehensive model evaluation")

    # Combine models for comparison
    all_models = {**baseline_models, **advanced_models}

    # Get validation data
    X_val = datasets.get('X_val_scaled', datasets['X_val'])
    y_val = datasets['y_val']

    # Create evaluation plots
    create_roc_curves(all_models, X_val, y_val, plot_dir, logger)
    create_precision_recall_curves(all_models, X_val, y_val, plot_dir, logger)
    create_calibration_plots(all_models, X_val, y_val, plot_dir, logger)
    create_model_comparison_plot(all_models, plot_dir, logger)

    # Create individual model visualizations
    feature_names = X_val.columns

    for name, model_info in all_models.items():
        model = model_info['model']

        # Confusion matrix
        create_confusion_matrix_plot(model, X_val, y_val, plot_dir, logger, name)

        # Feature importance (if model supports it)
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            create_feature_importance_plot(model, feature_names, plot_dir, logger, name)

    logger.info("Comprehensive model evaluation complete")
    return all_models


# Example usage (when run as a script)
if __name__ == "__main__":
    import os
    import pickle
    from utils.project_setup import create_project_structure, setup_logging

    # Setup project structure and logging
    log_dir, plot_dir, model_dir = create_project_structure()
    logger = setup_logging(log_dir, "model_evaluation")

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

    # Load trained models
    baseline_models_path = os.path.join(model_dir, "baseline_models.pkl")
    advanced_models_path = os.path.join(model_dir, "advanced_models.pkl")

    if not os.path.exists(baseline_models_path) or not os.path.exists(advanced_models_path):
        logger.error("Trained models not found. Run baseline_models.py and advanced_models.py first.")
        raise FileNotFoundError("Trained models not found")

    with open(baseline_models_path, "rb") as f:
        baseline_models = pickle.load(f)

    with open(advanced_models_path, "rb") as f:
        advanced_models = pickle.load(f)

    # Evaluate models
    all_models = evaluate_models(baseline_models, advanced_models, datasets, plot_dir, logger)

    # Save combined models
    with open(os.path.join(model_dir, "all_models.pkl"), "wb") as f:
        pickle.dump(all_models, f)

    logger.info(f"All models saved to: {os.path.join(model_dir, 'all_models.pkl')}")
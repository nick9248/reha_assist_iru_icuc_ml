# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import logging
import pickle
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss, log_loss,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Set display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Set random seed for reproducibility
np.random.seed(42)


def create_project_structure():
    """
    Create structured project folders for logs and plots

    Returns:
    --------
    log_dir : str
        Path to the log directory for this run
    plot_dir : str
        Path to the plot directory for this run
    model_dir : str
        Path to the directory for saving models
    """
    # Get base folders from environment variables
    log_base = os.environ.get('LOG_FOLDER')
    plot_base = os.environ.get('PLOT_FOLDER')
    model_base = os.environ.get('MODEL_FOLDER', os.path.join(os.environ.get('LOG_FOLDER', './logs'), 'models'))

    # Validate environment variables
    if not log_base:
        raise ValueError("LOG_FOLDER environment variable is not set.")
    if not plot_base:
        raise ValueError("PLOT_FOLDER environment variable is not set.")

    # Create timestamp for unique folder names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create structured folders
    stage_name = "03_model_selection"
    log_dir = os.path.join(log_base, stage_name, timestamp)
    plot_dir = os.path.join(plot_base, stage_name, timestamp)
    model_dir = os.path.join(model_base, stage_name, timestamp)

    # Create directories
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    return log_dir, plot_dir, model_dir


def setup_logging(log_folder):
    """
    Setup logging configuration

    Parameters:
    -----------
    log_folder : str
        Path to the folder where logs should be saved

    Returns:
    --------
    logger : logging.Logger
        Configured logger instance
    """
    # Configure logging
    log_file = os.path.join(log_folder, f"model_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create logger
    logger = logging.getLogger('model_selection')
    logger.setLevel(logging.INFO)

    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_dataset(logger):
    """
    Load the feature-engineered dataset from the previous step

    Parameters:
    -----------
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df : pandas.DataFrame
        The loaded dataset
    """
    # Get dataset path from environment variable
    # This should point to the output from the feature engineering step
    dataset_path = os.environ.get('ENGINEERED_DATASET')
    if not dataset_path:
        # If not set, try to find the most recent engineered dataset
        feature_eng_dir = os.path.join(os.environ.get('LOG_FOLDER', './logs'), '02_feature_engineering')
        if os.path.exists(feature_eng_dir):
            subdirs = [os.path.join(feature_eng_dir, d) for d in os.listdir(feature_eng_dir)
                       if os.path.isdir(os.path.join(feature_eng_dir, d))]
            if subdirs:
                latest_dir = max(subdirs, key=os.path.getmtime)
                potential_files = [f for f in os.listdir(latest_dir) if f.endswith('.pkl')]
                if potential_files:
                    dataset_path = os.path.join(latest_dir, 'engineered_features.pkl')

    # If still not found, use the raw dataset path
    if not dataset_path or not os.path.exists(dataset_path):
        dataset_path = os.environ.get('DATASET')
        logger.warning("Engineered dataset not found, using raw dataset")

    if not dataset_path:
        raise ValueError("No dataset path found in environment variables")

    logger.info(f"Loading data from: {dataset_path}")

    # Load the dataset
    try:
        if dataset_path.endswith('.pkl') or dataset_path.endswith('.pickle'):
            df = pd.read_pickle(dataset_path)
        elif dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
            df = pd.read_excel(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    logger.info(f"Dataset loaded with shape: {df.shape}")

    return df


def prepare_data_for_modeling(df, logger):
    """
    Prepare the dataset for modeling by splitting into train/test sets

    Parameters:
    -----------
    df : pandas.DataFrame
        Feature-engineered dataframe
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Testing target
    feature_names : list
        List of feature names
    patient_groups_train : pandas.Series
        Patient identifiers for grouped cross-validation
    patient_groups_test : pandas.Series
        Patient identifiers for grouped evaluation
    """
    logger.info("Preparing data for modeling")

    # Filter out records with nbe=2 (insufficient information) for training
    df_model = df[df['nbe'] != 2].copy()
    logger.info(f"Filtered out records with nbe=2, remaining shape: {df_model.shape}")

    # Ensure nbe_binary is correctly defined
    if 'nbe_binary' not in df_model.columns:
        df_model['nbe_binary'] = df_model['nbe'].map({0: 0, 1: 1})

    # Get target variable
    y = df_model['nbe_binary']

    # Class distribution
    class_dist = y.value_counts(normalize=True) * 100
    logger.info(f"Class distribution in filtered dataset: {class_dist.to_dict()}")

    # Define features to exclude
    exclude_cols = ['nbe', 'nbe_binary', 'accident_number', 'accident_date', 'contact_date']

    # Get feature names
    feature_names = [col for col in df_model.columns if col not in exclude_cols]

    # Extract features
    X = df_model[feature_names]

    # Get patient identifiers for grouped cross-validation
    patient_groups = df_model['accident_number']

    # Split data into training and testing sets while preserving patient grouping
    # This ensures all consultations from the same patient are in the same split
    unique_patients = df_model['accident_number'].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )

    train_indices = df_model['accident_number'].isin(train_patients)
    test_indices = df_model['accident_number'].isin(test_patients)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    patient_groups_train = patient_groups[train_indices]
    patient_groups_test = patient_groups[test_indices]

    logger.info(f"Train set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Number of features: {len(feature_names)}")
    logger.info(f"Number of unique patients in train: {patient_groups_train.nunique()}")
    logger.info(f"Number of unique patients in test: {patient_groups_test.nunique()}")

    # Check for class imbalance in train set
    train_class_dist = y_train.value_counts(normalize=True) * 100
    logger.info(f"Class distribution in train set: {train_class_dist.to_dict()}")

    # Check for class imbalance in test set
    test_class_dist = y_test.value_counts(normalize=True) * 100
    logger.info(f"Class distribution in test set: {test_class_dist.to_dict()}")

    return X_train, X_test, y_train, y_test, feature_names, patient_groups_train, patient_groups_test


def train_baseline_models(X_train, y_train, patient_groups_train, feature_names, logger, plot_dir, model_dir):
    """
    Train and evaluate baseline models

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    patient_groups_train : pandas.Series
        Patient identifiers for grouped cross-validation
    feature_names : list
        List of feature names
    logger : logging.Logger
        Logger instance
    plot_dir : str
        Directory to save plots
    model_dir : str
        Directory to save models

    Returns:
    --------
    baseline_models : dict
        Dictionary of trained baseline models
    """
    logger.info("Training baseline models")

    # Define baseline models
    baseline_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42)
    }

    # Define cross-validation strategy that respects patient grouping
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Metrics to evaluate
    metrics = {
        'Accuracy': accuracy_score,
        'ROC AUC': roc_auc_score,
        'Brier Score Loss': brier_score_loss,
        'Log Loss': log_loss
    }

    # Results dictionary
    results = {}

    # Train and evaluate each model with cross-validation
    for name, model in baseline_models.items():
        logger.info(f"Training {name} model")

        # Train the model
        model.fit(X_train, y_train)

        # Predict probabilities and classes on cross-validation folds
        cv_results = {}

        # Perform cross-validation
        y_prob = np.zeros(len(y_train))
        y_pred = np.zeros(len(y_train))

        for train_idx, test_idx in cv.split(X_train, y_train, patient_groups_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

            model_fold = baseline_models[name].__class__(**baseline_models[name].get_params())
            model_fold.fit(X_fold_train, y_fold_train)

            if hasattr(model_fold, "predict_proba"):
                y_prob[test_idx] = model_fold.predict_proba(X_fold_val)[:, 1]
            else:
                y_prob[test_idx] = model_fold.predict(X_fold_val)

            y_pred[test_idx] = (y_prob[test_idx] > 0.5).astype(int)

        # Calculate metrics on cross-validation results
        for metric_name, metric_func in metrics.items():
            if metric_name == 'ROC AUC':
                score = metric_func(y_train, y_prob)
            elif metric_name in ['Brier Score Loss', 'Log Loss']:
                score = metric_func(y_train, y_prob)
            else:
                score = metric_func(y_train, y_pred)

            cv_results[metric_name] = score
            logger.info(f"  {metric_name}: {score:.4f}")

        results[name] = cv_results

        # Feature importance for interpretable models
        if name == 'Logistic Regression':
            lr_coefs = pd.Series(model.coef_[0], index=feature_names)
            lr_coefs_abs = lr_coefs.abs().sort_values(ascending=False)

            plt.figure(figsize=(12, 8))
            lr_coefs_abs.head(20).plot(kind='barh')
            plt.title(f'Top 20 Feature Importance - {name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_importance.png'))
            plt.close()

            logger.info(f"Top 5 important features for {name}:")
            for feature, importance in lr_coefs_abs.head(5).items():
                logger.info(f"  - {feature}: {importance:.4f} (coef: {lr_coefs[feature]:.4f})")

        elif name == 'Decision Tree':
            dt_importances = pd.Series(model.feature_importances_, index=feature_names)
            dt_importances = dt_importances.sort_values(ascending=False)

            plt.figure(figsize=(12, 8))
            dt_importances.head(20).plot(kind='barh')
            plt.title(f'Top 20 Feature Importance - {name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_importance.png'))
            plt.close()

            logger.info(f"Top 5 important features for {name}:")
            for feature, importance in dt_importances.head(5).items():
                logger.info(f"  - {feature}: {importance:.4f}")

        # Save the trained model
        model_filename = os.path.join(model_dir, f"{name.replace(' ', '_').lower()}.pkl")
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"  Model saved to {model_filename}")

        # Create ROC curve for the model
        plt.figure(figsize=(8, 8))
        fpr, tpr, _ = roc_curve(y_train, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {cv_results["ROC AUC"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_roc.png'))
        plt.close()

        # Create calibration curve
        plt.figure(figsize=(10, 6))
        prob_true, prob_pred = calibration_curve(y_train, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=name)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration Curve - {name}')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_calibration.png'))
        plt.close()

    # Create comparison plot for all baseline models
    plt.figure(figsize=(12, 8))
    x = list(results.keys())
    for metric in ['ROC AUC', 'Accuracy']:
        values = [results[model][metric] for model in x]
        plt.bar(x, values, alpha=0.6, label=metric)

    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Baseline Models Performance Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'baseline_models_comparison.png'))
    plt.close()

    return baseline_models


def train_advanced_models(X_train, y_train, patient_groups_train, feature_names, logger, plot_dir, model_dir):
    """
    Train and evaluate advanced models

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    patient_groups_train : pandas.Series
        Patient identifiers for grouped cross-validation
    feature_names : list
        List of feature names
    logger : logging.Logger
        Logger instance
    plot_dir : str
        Directory to save plots
    model_dir : str
        Directory to save models

    Returns:
    --------
    advanced_models : dict
        Dictionary of trained advanced models
    """
    logger.info("Training advanced models")

    # Define advanced models
    advanced_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

    # Define cross-validation strategy that respects patient grouping
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Metrics to evaluate
    metrics = {
        'Accuracy': accuracy_score,
        'ROC AUC': roc_auc_score,
        'Brier Score Loss': brier_score_loss,
        'Log Loss': log_loss
    }

    # Results dictionary
    results = {}

    # SMOTE for handling class imbalance
    smote = SMOTE(random_state=42)

    # Check if SMOTE should be applied based on class imbalance
    class_counts = y_train.value_counts()
    imbalance_ratio = class_counts.min() / class_counts.max()
    apply_smote = imbalance_ratio < 0.5

    if apply_smote:
        logger.info(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}), applying SMOTE")
    else:
        logger.info("Class balance is acceptable, not applying SMOTE")

    # Train and evaluate each model with cross-validation
    for name, model in advanced_models.items():
        logger.info(f"Training {name} model")

        # Train the model
        if apply_smote and name != 'Neural Network':  # SMOTE can be problematic with NN
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            model.fit(X_resampled, y_resampled)
            logger.info(f"  Applied SMOTE: increased samples from {len(X_train)} to {len(X_resampled)}")
        else:
            model.fit(X_train, y_train)

        # Predict probabilities and classes on cross-validation folds
        cv_results = {}

        # Perform cross-validation
        y_prob = np.zeros(len(y_train))
        y_pred = np.zeros(len(y_train))

        for train_idx, test_idx in cv.split(X_train, y_train, patient_groups_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

            model_fold = advanced_models[name].__class__(**advanced_models[name].get_params())

            if apply_smote and name != 'Neural Network':
                X_fold_resampled, y_fold_resampled = smote.fit_resample(X_fold_train, y_fold_train)
                model_fold.fit(X_fold_resampled, y_fold_resampled)
            else:
                model_fold.fit(X_fold_train, y_fold_train)

            if hasattr(model_fold, "predict_proba"):
                y_prob[test_idx] = model_fold.predict_proba(X_fold_val)[:, 1]
            else:
                y_prob[test_idx] = model_fold.predict(X_fold_val)

            y_pred[test_idx] = (y_prob[test_idx] > 0.5).astype(int)

        # Calculate metrics on cross-validation results
        for metric_name, metric_func in metrics.items():
            if metric_name == 'ROC AUC':
                score = metric_func(y_train, y_prob)
            elif metric_name in ['Brier Score Loss', 'Log Loss']:
                score = metric_func(y_train, y_prob)
            else:
                score = metric_func(y_train, y_pred)

            cv_results[metric_name] = score
            logger.info(f"  {metric_name}: {score:.4f}")

        results[name] = cv_results

        # Feature importance for tree-based models
        if name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
            if name == 'XGBoost':
                feature_imp = model.feature_importances_
            elif name == 'LightGBM':
                feature_imp = model.feature_importances_
            else:
                feature_imp = model.feature_importances_

            importances = pd.Series(feature_imp, index=feature_names)
            importances = importances.sort_values(ascending=False)

            plt.figure(figsize=(12, 8))
            importances.head(20).plot(kind='barh')
            plt.title(f'Top 20 Feature Importance - {name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_importance.png'))
            plt.close()

            logger.info(f"Top 5 important features for {name}:")
            for feature, importance in importances.head(5).items():
                logger.info(f"  - {feature}: {importance:.4f}")

        # Save the trained model
        model_filename = os.path.join(model_dir, f"{name.replace(' ', '_').lower()}.pkl")
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"  Model saved to {model_filename}")

        # Create ROC curve for the model
        plt.figure(figsize=(8, 8))
        fpr, tpr, _ = roc_curve(y_train, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {cv_results["ROC AUC"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_roc.png'))
        plt.close()

        # Create calibration curve
        plt.figure(figsize=(10, 6))
        prob_true, prob_pred = calibration_curve(y_train, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=name)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration Curve - {name}')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_calibration.png'))
        plt.close()

    # Create comparison plot for all advanced models
    plt.figure(figsize=(15, 8))
    x = list(results.keys())

    # Plot for ROC AUC
    plt.subplot(1, 2, 1)
    values = [results[model]['ROC AUC'] for model in x]
    bars = plt.bar(x, values, alpha=0.7)
    plt.xlabel('Model')
    plt.ylabel('ROC AUC Score')
    plt.title('Advanced Models - ROC AUC Comparison')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', rotation=0)

    # Plot for Brier Score Loss (lower is better)
    plt.subplot(1, 2, 2)
    values = [results[model]['Brier Score Loss'] for model in x]
    bars = plt.bar(x, values, alpha=0.7, color='orange')
    plt.xlabel('Model')
    plt.ylabel('Brier Score Loss (lower is better)')
    plt.title('Advanced Models - Calibration Comparison')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'advanced_models_comparison.png'))
    plt.close()

    return advanced_models


def calibrate_best_models(X_train, y_train, advanced_models, logger, model_dir):
    """
    Calibrate the best models for improved probability estimation

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    advanced_models : dict
        Dictionary of trained advanced models
    logger : logging.Logger
        Logger instance
    model_dir : str
        Directory to save models

    Returns:
    --------
    calibrated_models : dict
        Dictionary of calibrated models
    """
    logger.info("Calibrating best models for improved probability estimation")

    # Select the best models for calibration
    # Using tree-based models which typically need calibration
    models_to_calibrate = {
        'Random Forest': advanced_models['Random Forest'],
        'Gradient Boosting': advanced_models['Gradient Boosting'],
        'XGBoost': advanced_models['XGBoost']
    }

    calibrated_models = {}

    for name, model in models_to_calibrate.items():
        logger.info(f"Calibrating {name} model")

        # Create and train calibrated model with isotonic regression
        calibrated_clf = CalibratedClassifierCV(model, method='isotonic', cv=5)
        calibrated_clf.fit(X_train, y_train)

        calibrated_name = f"Calibrated {name}"
        calibrated_models[calibrated_name] = calibrated_clf

        # Save the calibrated model
        model_filename = os.path.join(model_dir, f"{calibrated_name.replace(' ', '_').lower()}.pkl")
        with open(model_filename, 'wb') as file:
            pickle.dump(calibrated_clf, file)
        logger.info(f"  Calibrated model saved to {model_filename}")

    return calibrated_models


def evaluate_models_on_test_set(X_test, y_test, all_models, patient_groups_test, logger, plot_dir):
    """
    Evaluate all models on the test set

    Parameters:
    -----------
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    all_models : dict
        Dictionary of all trained models
    patient_groups_test : pandas.Series
        Patient identifiers for test set
    logger : logging.Logger
        Logger instance
    plot_dir : str
        Directory to save plots

    Returns:
    --------
    test_results : dict
        Dictionary of test results for all models
    """
    logger.info("Evaluating all models on the test set")

    # Metrics to evaluate
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': lambda y, y_pred: precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score,
        'F1 Score': f1_score,
        'ROC AUC': roc_auc_score,
        'Brier Score Loss': brier_score_loss,
        'Log Loss': log_loss,
        'Average Precision': average_precision_score
    }

    # Results dictionary
    test_results = {}

    # Evaluate each model
    for name, model in all_models.items():
        logger.info(f"Evaluating {name} model on test set")

        # Make predictions
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)

        y_pred = (y_prob > 0.5).astype(int)

        # Calculate metrics
        model_results = {}
        for metric_name, metric_func in metrics.items():
            if metric_name == 'ROC AUC' or metric_name == 'Average Precision':
                score = metric_func(y_test, y_prob)
            elif metric_name in ['Brier Score Loss', 'Log Loss']:
                score = metric_func(y_test, y_prob)
            else:
                score = metric_func(y_test, y_pred)

            model_results[metric_name] = score
            logger.info(f"  {metric_name}: {score:.4f}")

        # Add confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"  Confusion Matrix:\n{cm}")

        # Add classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"  Classification Report:\n{pd.DataFrame(class_report).transpose().round(3)}")

        test_results[name] = model_results

        # Create ROC curve
        plt.figure(figsize=(8, 8))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {model_results["ROC AUC"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Test Set ROC Curve - {name}')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_test_roc.png'))
        plt.close()

        # Create precision-recall curve
        plt.figure(figsize=(8, 8))
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, label=f'{name} (AP = {model_results["Average Precision"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Test Set Precision-Recall Curve - {name}')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_test_pr.png'))
        plt.close()

        # Create calibration curve
        plt.figure(figsize=(10, 6))
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=name)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Test Set Calibration Curve - {name}')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{name.replace(" ", "_").lower()}_test_calibration.png'))
        plt.close()

    # Create comparison plot for all models on test set
    plt.figure(figsize=(15, 12))

    # Plot for ROC AUC
    plt.subplot(2, 2, 1)
    models = list(test_results.keys())
    values = [test_results[model]['ROC AUC'] for model in models]

    # Sort models by performance
    sorted_indices = np.argsort(values)[::-1]  # Descending order
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    bars = plt.barh(sorted_models[:10], sorted_values[:10], alpha=0.7)  # Show top 10 models
    plt.xlabel('ROC AUC Score')
    plt.ylabel('Model')
    plt.title('Test Set - ROC AUC Comparison')
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2., f'{width:.3f}',
                 ha='left', va='center')

    # Plot for Average Precision
    plt.subplot(2, 2, 2)
    values = [test_results[model]['Average Precision'] for model in models]

    # Sort models by performance
    sorted_indices = np.argsort(values)[::-1]  # Descending order
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    bars = plt.barh(sorted_models[:10], sorted_values[:10], alpha=0.7, color='green')  # Show top 10 models
    plt.xlabel('Average Precision Score')
    plt.ylabel('Model')
    plt.title('Test Set - Average Precision Comparison')
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2., f'{width:.3f}',
                 ha='left', va='center')

    # Plot for Brier Score Loss (lower is better)
    plt.subplot(2, 2, 3)
    values = [test_results[model]['Brier Score Loss'] for model in models]

    # Sort models by performance (ascending order for Brier score)
    sorted_indices = np.argsort(values)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    bars = plt.barh(sorted_models[:10], sorted_values[:10], alpha=0.7, color='orange')  # Show top 10 models
    plt.xlabel('Brier Score Loss (lower is better)')
    plt.ylabel('Model')
    plt.title('Test Set - Calibration Comparison')
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2., f'{width:.3f}',
                 ha='left', va='center')

    # Plot for F1 Score
    plt.subplot(2, 2, 4)
    values = [test_results[model]['F1 Score'] for model in models]

    # Sort models by performance
    sorted_indices = np.argsort(values)[::-1]  # Descending order
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    bars = plt.barh(sorted_models[:10], sorted_values[:10], alpha=0.7, color='purple')  # Show top 10 models
    plt.xlabel('F1 Score')
    plt.ylabel('Model')
    plt.title('Test Set - F1 Score Comparison')
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2., f'{width:.3f}',
                 ha='left', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'test_set_models_comparison.png'))
    plt.close()

    # Evaluate patient-level performance
    logger.info("Evaluating patient-level performance")

    # Select best model based on ROC AUC
    best_model_name = sorted_models[0]
    best_model = all_models[best_model_name]

    # Calculate patient-level predictions
    patient_level_results = {}
    unique_patients = patient_groups_test.unique()

    for patient in unique_patients:
        patient_indices = patient_groups_test == patient
        patient_X = X_test[patient_indices]
        patient_y = y_test[patient_indices]

        if hasattr(best_model, "predict_proba"):
            patient_probs = best_model.predict_proba(patient_X)[:, 1]
        else:
            patient_probs = best_model.predict(patient_X)

        # Average probabilities for the patient
        avg_prob = np.mean(patient_probs)
        patient_pred = (avg_prob > 0.5).astype(int)

        # Majority vote for ground truth
        patient_true = (np.mean(patient_y) > 0.5).astype(int)

        patient_level_results[patient] = {
            'true': patient_true,
            'pred': patient_pred,
            'prob': avg_prob
        }

    # Calculate patient-level metrics
    patient_true = [result['true'] for result in patient_level_results.values()]
    patient_pred = [result['pred'] for result in patient_level_results.values()]
    patient_prob = [result['prob'] for result in patient_level_results.values()]

    logger.info(f"Patient-level accuracy: {accuracy_score(patient_true, patient_pred):.4f}")
    logger.info(f"Patient-level ROC AUC: {roc_auc_score(patient_true, patient_prob):.4f}")
    logger.info(f"Patient-level Brier score: {brier_score_loss(patient_true, patient_prob):.4f}")

    # Create patient-level ROC curve
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(patient_true, patient_prob)
    plt.plot(fpr, tpr, label=f'Patient-level (AUC = {roc_auc_score(patient_true, patient_prob):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Patient-level ROC Curve - {best_model_name}')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'patient_level_roc.png'))
    plt.close()

    return test_results
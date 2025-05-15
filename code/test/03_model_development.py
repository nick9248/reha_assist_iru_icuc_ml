def analyze_feature_importance(best_model, X_train, feature_names, plot_dir, logger):
    """
    Analyze feature importance of the best model

    Parameters:
    -----------
    best_model : sklearn.pipeline.Pipeline
        The best performing model
    X_train : pandas.DataFrame
        Training features
    feature_names : list
        List of feature names
    plot_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    """
    logger.info("Analyzing feature importance")
    
    # Extract the model from the pipeline
    if hasattr(best_model, 'named_steps') and 'model' in best_model.named_steps:
        model = best_model.named_steps['model']
    else:
        model = best_model
    
    # Check if model has feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Log top 20 features
        logger.info("Top 20 important features:")
        for idx, row in importance_df.head(20).iterrows():
            logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 10))
        importance_df.head(20).plot(kind='barh', x='Feature', y='Importance')
        plt.title('Top 20 Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'feature_importance.png'))
        plt.close()
        
        return importance_df
    
    # For logistic regression, check if model has coef_ attribute
    elif hasattr(model, 'coef_'):
        # Get coefficients
        coefficients = model.coef_[0]
        
        # Create DataFrame for visualization
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)
        
        # Log top features by absolute coefficient
        logger.info("Top 20 features by coefficient magnitude:")
        abs_coef_df = coef_df.copy()
        abs_coef_df['AbsCoefficient'] = abs_coef_df['Coefficient'].abs()
        abs_coef_df = abs_coef_df.sort_values('AbsCoefficient', ascending=False)
        
        for idx, row in abs_coef_df.head(20).iterrows():
            logger.info(f"  {row['Feature']}: {row['Coefficient']:.4f}")
        
        # Plot coefficients
        plt.figure(figsize=(12, 10))
        abs_coef_df.head(20).plot(kind='barh', x='Feature', y='Coefficient')
        plt.title('Top 20 Feature Coefficients (by Magnitude)')
        plt.xlabel('Coefficient')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'feature_coefficients.png'))
        plt.close()
        
        return coef_df
    
    else:
        logger.warning("Model does not provide feature importance or coefficients")
        return None


def save_model(model, model_dir, metrics, logger):
    """
    Save the final model and its metrics

    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        The final model
    model_dir : str
        Directory to save the model
    metrics : dict
        Dictionary of model performance metrics
    logger : logging.Logger
        Logger instance
    """
    logger.info(f"Saving model to: {model_dir}")
    
    # Save the model
    model_path = os.path.join(model_dir, "final_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save metrics as JSON
    metrics_path = os.path.join(model_dir, "model_metrics.json")
    
    # Convert numpy values to Python types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if key in ['confusion_matrix', 'classification_report']:
            serializable_metrics[key] = str(value)
        elif isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
            serializable_metrics[key] = value.item()
        else:
            serializable_metrics[key] = value
    
    with open(metrics_path, 'w') as f:
        import json
        json.dump(serializable_metrics, f, indent=4)
    
    logger.info(f"Metrics saved to: {metrics_path}")


def main():
    """
    Main function to execute the model development pipeline
    """
    # Create project structure
    log_dir, plot_dir, model_dir = create_project_structure()
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("=== MODEL DEVELOPMENT PIPELINE ===")
    
    # Load the engineered dataset
    df = load_dataset(logger)
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_split(df, logger)
    
    # Train baseline models
    logger.info("Step 1: Training baseline models")
    baseline_models = train_baseline_models(X_train, y_train, X_val, y_val, logger)
    
    # Train advanced models
    logger.info("Step 2: Training advanced models")
    advanced_models = train_advanced_models(X_train, y_train, X_val, y_val, logger)
    
    # Combine all models for comparison
    all_models = {**baseline_models, **advanced_models}
    
    # Create evaluation plots
    create_evaluation_plots(all_models, X_val, y_val, plot_dir, logger)
    
    # Find best model based on ROC AUC
    best_model_name = max(all_models.items(), key=lambda x: x[1]['roc_auc'])[0]
    best_model = all_models[best_model_name]['model']
    
    logger.info(f"Best model: {best_model_name} with ROC AUC: {all_models[best_model_name]['roc_auc']:.4f}")
    
    # Calibrate probabilities for the best model
    logger.info("Step 3: Calibrating probabilities for best model")
    calibrated_model = calibrate_probabilities(best_model, X_train, y_train, X_val, y_val, logger)
    
    # Analyze feature importance for the best model
    logger.info("Step 4: Analyzing feature importance")
    feature_importance = analyze_feature_importance(best_model, X_train, X_train.columns, plot_dir, logger)
    
    # Evaluate final model on test set
    logger.info("Step 5: Evaluating final model on test set")
    test_metrics = evaluate_model_on_test(calibrated_model, X_test, y_test, logger)
    
    # Save the final model and metrics
    logger.info("Step 6: Saving final model")
    save_model(calibrated_model, model_dir, test_metrics, logger)
    
    logger.info("=== MODEL DEVELOPMENT COMPLETE ===")
    logger.info(f"Final model saved to: {model_dir}")
    logger.info(f"Model evaluation plots saved to: {plot_dir}")
    
    return calibrated_model, test_metrics, feature_importance


if __name__ == "__main__":
    main()# Import necessary libraries
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
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, log_loss, brier_score_loss,
                            precision_recall_curve, roc_curve, auc,
                            confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Load environment variables from .env file
load_dotenv()

# Set display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Set random seed for reproducibility
np.random.seed(42)


def create_project_structure():
    """
    Create structured project folders for logs, plots, and models

    Returns:
    --------
    log_dir : str
        Path to the log directory for this run
    plot_dir : str
        Path to the plot directory for this run
    model_dir : str
        Path to the model directory for this run
    """
    # Get base folders from environment variables
    log_base = os.environ.get('LOG_FOLDER')
    plot_base = os.environ.get('PLOT_FOLDER')
    model_base = os.environ.get('MODEL_FOLDER', 'models')  # Default to 'models' if not set

    # Validate environment variables
    if not log_base:
        raise ValueError("LOG_FOLDER environment variable is not set.")
    if not plot_base:
        raise ValueError("PLOT_FOLDER environment variable is not set.")

    # Create timestamp for unique folder names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create structured folders
    stage_name = "03_model_development"
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
    log_file = os.path.join(log_folder, f"model_development_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create logger
    logger = logging.getLogger('model_development')
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
    Load the engineered dataset from the previous step

    Parameters:
    -----------
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df : pandas.DataFrame
        The loaded dataset
    """
    # Get dataset path from environment variables
    engineered_data_path = os.environ.get('ENGINEERED_DATASET')
    
    if not engineered_data_path:
        # Try to find the most recent engineered dataset if not specified
        logger.info("ENGINEERED_DATASET environment variable not set, searching for the most recent dataset")
        log_base = os.environ.get('LOG_FOLDER')
        if not log_base:
            raise ValueError("LOG_FOLDER environment variable is not set.")
        
        feature_eng_dir = os.path.join(log_base, "02_feature_engineering")
        if not os.path.exists(feature_eng_dir):
            raise ValueError(f"Feature engineering directory not found: {feature_eng_dir}")
        
        # Find most recent subdirectory
        subdirs = [os.path.join(feature_eng_dir, d) for d in os.listdir(feature_eng_dir) 
                   if os.path.isdir(os.path.join(feature_eng_dir, d))]
        if not subdirs:
            raise ValueError(f"No subdirectories found in {feature_eng_dir}")
        
        latest_subdir = max(subdirs, key=os.path.getmtime)
        engineered_data_path = os.path.join(latest_subdir, "engineered_features.pkl")
    
    logger.info(f"Loading engineered dataset from: {engineered_data_path}")
    
    # Load the dataset
    if engineered_data_path.endswith('.pkl') or engineered_data_path.endswith('.pickle'):
        df = pd.read_pickle(engineered_data_path)
    else:
        raise ValueError(f"Unsupported file format: {engineered_data_path}")
    
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df


def prepare_data_split(df, logger):
    """
    Prepare dataset splits for model training and evaluation,
    ensuring patient-level separation

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with engineered features
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    X_train : pandas.DataFrame
        Training features
    X_val : pandas.DataFrame
        Validation features
    X_test : pandas.DataFrame
        Test features
    y_train : pandas.Series
        Training target
    y_val : pandas.Series
        Validation target
    y_test : pandas.Series
        Test target
    """
    logger.info("Preparing data splits with patient-level separation")
    
    # Check if 'nbe_binary' exists, otherwise create it
    if 'nbe_binary' not in df.columns:
        # Filter out nbe=2 (no information) or convert it if needed
        df = df[df['nbe'] != 2].copy()  # Remove 'no information' cases
        df['nbe_binary'] = df['nbe']  # Use existing nbe column (0 or 1)
    
    # Get unique patient IDs
    unique_patients = df['accident_number'].unique()
    logger.info(f"Total unique patients: {len(unique_patients)}")
    
    # Split patient IDs into train, validation, and test
    # 70% training, 15% validation, 15% test
    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=0.3, random_state=42
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.5, random_state=42
    )
    
    logger.info(f"Training patients: {len(train_patients)}")
    logger.info(f"Validation patients: {len(val_patients)}")
    logger.info(f"Test patients: {len(test_patients)}")
    
    # Create masks for each set
    train_mask = df['accident_number'].isin(train_patients)
    val_mask = df['accident_number'].isin(val_patients)
    test_mask = df['accident_number'].isin(test_patients)
    
    # Create dataframes for each set
    df_train = df[train_mask]
    df_val = df[val_mask]
    df_test = df[test_mask]
    
    logger.info(f"Training set size: {df_train.shape[0]} consultations")
    logger.info(f"Validation set size: {df_val.shape[0]} consultations")
    logger.info(f"Test set size: {df_test.shape[0]} consultations")
    
    # Check class distribution in each set
    logger.info("Class distribution in training set:")
    logger.info(df_train['nbe_binary'].value_counts(normalize=True) * 100)
    
    logger.info("Class distribution in validation set:")
    logger.info(df_val['nbe_binary'].value_counts(normalize=True) * 100)
    
    logger.info("Class distribution in test set:")
    logger.info(df_test['nbe_binary'].value_counts(normalize=True) * 100)
    
    # Select features and target
    # Exclude non-feature columns
    exclude_cols = ['accident_number', 'accident_date', 'contact_date', 
                   'nbe', 'nbe_binary', 'tel_recovery_combined']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Create feature matrices and target vectors
    X_train = df_train[feature_cols]
    y_train = df_train['nbe_binary']
    
    X_val = df_val[feature_cols]
    y_val = df_val['nbe_binary']
    
    X_test = df_test[feature_cols]
    y_test = df_test['nbe_binary']
    
    logger.info(f"Feature set size: {len(feature_cols)} features")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_baseline_models(X_train, y_train, X_val, y_val, logger):
    """
    Train baseline models and evaluate performance

    Parameters:
    -----------
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

    Returns:
    --------
    baseline_models : dict
        Dictionary of trained baseline models and their performance metrics
    """
    logger.info("Training baseline models")
    
    # Initialize dictionary to store models and performance
    baseline_models = {}
    
    # Define baseline models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        logger.info(f"Training {name}...")
        start_time = time.time()
        
        # Create a pipeline with standard scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Fit the model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_val_pred = pipeline.predict(X_val)
        y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Evaluate performance
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        roc_auc = roc_auc_score(y_val, y_val_pred_proba)
        brier = brier_score_loss(y_val, y_val_pred_proba)
        logloss = log_loss(y_val, y_val_pred_proba)
        
        # Calculate training time
        train_time = time.time() - start_time
        
        # Store results
        baseline_models[name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'brier_score': brier,
            'log_loss': logloss,
            'train_time': train_time
        }
        
        # Log performance
        logger.info(f"{name} performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        logger.info(f"  Brier Score: {brier:.4f}")
        logger.info(f"  Log Loss: {logloss:.4f}")
        logger.info(f"  Training Time: {train_time:.2f} seconds")
        
        # Classification report
        logger.info(f"Classification Report:\n{classification_report(y_val, y_val_pred)}")
    
    return baseline_models


def train_advanced_models(X_train, y_train, X_val, y_val, logger):
    """
    Train advanced models with SMOTE handling for class imbalance

    Parameters:
    -----------
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

    Returns:
    --------
    advanced_models : dict
        Dictionary of trained advanced models and their performance metrics
    """
    logger.info("Training advanced models with class imbalance handling")
    
    # Initialize dictionary to store models and performance
    advanced_models = {}
    
    # Define advanced models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    # Train and evaluate each model, with and without SMOTE
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Regular model (without SMOTE)
        start_time = time.time()
        
        # Create a pipeline with standard scaling
        standard_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Fit the model
        standard_pipeline.fit(X_train, y_train)
        
        # Predictions
        y_val_pred = standard_pipeline.predict(X_val)
        y_val_pred_proba = standard_pipeline.predict_proba(X_val)[:, 1]
        
        # Evaluate performance
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        roc_auc = roc_auc_score(y_val, y_val_pred_proba)
        brier = brier_score_loss(y_val, y_val_pred_proba)
        logloss = log_loss(y_val, y_val_pred_proba)
        
        # Calculate training time
        train_time = time.time() - start_time
        
        # Store results
        advanced_models[name] = {
            'model': standard_pipeline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'brier_score': brier,
            'log_loss': logloss,
            'train_time': train_time
        }
        
        # Log performance
        logger.info(f"{name} performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        logger.info(f"  Brier Score: {brier:.4f}")
        logger.info(f"  Log Loss: {logloss:.4f}")
        logger.info(f"  Training Time: {train_time:.2f} seconds")
        
        # Classification report
        logger.info(f"Classification Report:\n{classification_report(y_val, y_val_pred)}")
        
        # SMOTE version of the model
        logger.info(f"Training {name} with SMOTE...")
        start_time = time.time()
        
        # Create a pipeline with SMOTE
        smote_pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
        
        # Fit the model
        smote_pipeline.fit(X_train, y_train)
        
        # Predictions
        y_val_pred_smote = smote_pipeline.predict(X_val)
        y_val_pred_proba_smote = smote_pipeline.predict_proba(X_val)[:, 1]
        
        # Evaluate performance
        accuracy_smote = accuracy_score(y_val, y_val_pred_smote)
        precision_smote = precision_score(y_val, y_val_pred_smote)
        recall_smote = recall_score(y_val, y_val_pred_smote)
        f1_smote = f1_score(y_val, y_val_pred_smote)
        roc_auc_smote = roc_auc_score(y_val, y_val_pred_proba_smote)
        brier_smote = brier_score_loss(y_val, y_val_pred_proba_smote)
        logloss_smote = log_loss(y_val, y_val_pred_proba_smote)
        
        # Calculate training time
        train_time_smote = time.time() - start_time
        
        # Store results
        advanced_models[f"{name} with SMOTE"] = {
            'model': smote_pipeline,
            'accuracy': accuracy_smote,
            'precision': precision_smote,
            'recall': recall_smote,
            'f1': f1_smote,
            'roc_auc': roc_auc_smote,
            'brier_score': brier_smote,
            'log_loss': logloss_smote,
            'train_time': train_time_smote
        }
        
        # Log performance
        logger.info(f"{name} with SMOTE performance:")
        logger.info(f"  Accuracy: {accuracy_smote:.4f}")
        logger.info(f"  Precision: {precision_smote:.4f}")
        logger.info(f"  Recall: {recall_smote:.4f}")
        logger.info(f"  F1 Score: {f1_smote:.4f}")
        logger.info(f"  ROC AUC: {roc_auc_smote:.4f}")
        logger.info(f"  Brier Score: {brier_smote:.4f}")
        logger.info(f"  Log Loss: {logloss_smote:.4f}")
        logger.info(f"  Training Time: {train_time_smote:.2f} seconds")
        
        # Classification report
        logger.info(f"Classification Report (SMOTE):\n{classification_report(y_val, y_val_pred_smote)}")
    
    return advanced_models


def calibrate_probabilities(best_model, X_train, y_train, X_val, y_val, logger):
    """
    Calibrate probability estimates of the best model

    Parameters:
    -----------
    best_model : sklearn.pipeline.Pipeline
        The best performing model
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

    Returns:
    --------
    calibrated_model : sklearn.calibration.CalibratedClassifierCV
        Calibrated version of the best model
    """
    logger.info("Calibrating model probabilities")
    
    # Extract the model from the pipeline
    model_step = best_model.named_steps['model']
    
    # Create a new pipeline with scaling but without the model
    # (we'll add the calibrated model later)
    if 'scaler' in best_model.named_steps:
        scaler = best_model.named_steps['scaler']
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
    
    # Calibrate the model
    logger.info("Applying probability calibration with isotonic regression")
    calibrated_model = CalibratedClassifierCV(
        model_step, method='isotonic', cv='prefit'
    )
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Evaluate calibrated model
    y_val_pred_proba_calibrated = calibrated_model.predict_proba(X_val_scaled)[:, 1]
    y_val_pred_calibrated = (y_val_pred_proba_calibrated >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred_calibrated)
    precision = precision_score(y_val, y_val_pred_calibrated)
    recall = recall_score(y_val, y_val_pred_calibrated)
    f1 = f1_score(y_val, y_val_pred_calibrated)
    roc_auc = roc_auc_score(y_val, y_val_pred_proba_calibrated)
    brier = brier_score_loss(y_val, y_val_pred_proba_calibrated)
    logloss = log_loss(y_val, y_val_pred_proba_calibrated)
    
    # Log performance
    logger.info("Calibrated model performance:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")
    
    # Create a new pipeline with the calibrated model
    if 'scaler' in best_model.named_steps:
        calibrated_pipeline = Pipeline([
            ('scaler', scaler),
            ('model', calibrated_model)
        ])
    else:
        calibrated_pipeline = calibrated_model
    
    return calibrated_pipeline


def create_evaluation_plots(models_dict, X_val, y_val, plot_dir, logger):
    """
    Create evaluation plots for model comparison

    Parameters:
    -----------
    models_dict : dict
        Dictionary of trained models and their performance metrics
    X_val : pandas.DataFrame
        Validation features
    y_val : pandas.Series
        Validation target
    plot_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    """
    logger.info("Creating evaluation plots")
    
    # 1. ROC Curve Plot
    plt.figure(figsize=(12, 8))
    for name, model_info in models_dict.items():
        model = model_info['model']
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'roc_curves.png'))
    plt.close()
    
    # 2. Precision-Recall Curve
    plt.figure(figsize=(12, 8))
    for name, model_info in models_dict.items():
        model = model_info['model']
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_val, y_val_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, lw=2, label=f'{name} (AUC = {pr_auc:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'precision_recall_curves.png'))
    plt.close()
    
    # 3. Calibration Plot
    plt.figure(figsize=(12, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    for name, model_info in models_dict.items():
        model = model_info['model']
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_val, y_val_pred_proba, n_bins=10
        )
        
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label=f'{name} (Brier: {model_info["brier_score"]:.3f})')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'calibration_curves.png'))
    plt.close()
    
    # 4. Performance Metrics Comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score']
    metrics_df = pd.DataFrame(index=metrics)
    
    for name, model_info in models_dict.items():
        model_metrics = [model_info[metric] for metric in metrics]
        metrics_df[name] = model_metrics
    
    plt.figure(figsize=(14, 10))
    metrics_df.T.plot(kind='bar', figsize=(14, 8))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'model_performance_comparison.png'))
    plt.close()
    
    # 5. Confusion Matrix for Best Model
    # Identify best model by ROC AUC
    best_model_name = max(models_dict.items(), key=lambda x: x[1]['roc_auc'])[0]
    best_model = models_dict[best_model_name]['model']
    
    y_val_pred = best_model.predict(X_val)
    cm = confusion_matrix(y_val, y_val_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'best_model_confusion_matrix.png'))
    plt.close()


def evaluate_model_on_test(model, X_test, y_test, logger):
    """
    Evaluate the final model on the test set

    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        The final model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    test_metrics : dict
        Dictionary of test performance metrics
    """
    logger.info("Evaluating final model on test set")
    
    # Make predictions
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    brier = brier_score_loss(y_test, y_test_pred_proba)
    logloss = log_loss(y_test, y_test_pred_proba)
    
    # Create classification report
    class_report = classification_report(y_test, y_test_pred)
    
    # Log performance
    logger.info("Final model performance on test set:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")
    logger.info(f"Classification Report:\n{class_report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info("Confusion Matrix:")
    logger.info(cm)
    
    # Store metrics in a dictionary
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
    
    return test_metrics
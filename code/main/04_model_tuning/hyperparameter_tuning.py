"""
Hyperparameter tuning module for the NBE prediction model.
Implements Bayesian optimization for finding optimal hyperparameters.
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


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


def create_patient_folds(X, y, patient_ids, n_splits=5, random_state=42):
    """
    Create cross-validation folds at the patient level

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


def patient_cross_val_score(estimator, X, y, patient_ids, scoring, n_splits=5):
    """
    Calculate cross-validation score with patient-level splits

    Parameters:
    -----------
    estimator : object
        Estimator with fit and predict methods
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    patient_ids : pandas.Series
        Patient identifiers
    scoring : callable
        Scoring function
    n_splits : int
        Number of folds

    Returns:
    --------
    float
        Mean cross-validation score
    """
    folds = create_patient_folds(X, y, patient_ids, n_splits)
    scores = []

    for train_idx, val_idx in folds:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict_proba(X_val)[:, 1]
        scores.append(scoring(y_val, y_pred))

    return np.mean(scores)


def objective(trial, X, y, patient_ids):
    """
    Objective function for Optuna optimization

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    patient_ids : pandas.Series
        Patient identifiers

    Returns:
    --------
    float
        Mean cross-validation score
    """
    # Define hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    # Create model with suggested hyperparameters
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1
    )

    # Calculate patient-level cross-validation score
    scoring = roc_auc_score
    cv_score = patient_cross_val_score(rf, X, y, patient_ids, scoring, n_splits=5)

    return cv_score


def optimize_hyperparameters(X, y, patient_ids, logger, n_trials=100, timeout=600):
    """
    Optimize hyperparameters using Optuna

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    patient_ids : pandas.Series
        Patient identifiers
    logger : logging.Logger
        Logger instance
    n_trials : int
        Number of trials for optimization
    timeout : int
        Timeout in seconds

    Returns:
    --------
    dict
        Dictionary with best hyperparameters
    """
    logger.info("Starting hyperparameter optimization with Bayesian optimization")
    logger.info(f"Running {n_trials} trials with a timeout of {timeout} seconds")

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(),
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, patient_ids),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # Log results
    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"Best CV score: {best_score:.4f}")
    logger.info("Best hyperparameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")

    # Create a dataframe with all trial results for analysis
    trials_df = pd.DataFrame([
        {
            **trial.params,
            'score': trial.value,
            'trial_number': trial.number
        }
        for trial in study.trials
    ])

    return best_params, best_score, trials_df, study


def train_optimized_model(X, y, best_params, logger):
    """
    Train model with the best hyperparameters

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    best_params : dict
        Best hyperparameters
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    object
        Trained model
    """
    logger.info("Training model with optimized hyperparameters")

    # Create model with best hyperparameters
    rf = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    rf.fit(X, y)

    return rf


def save_optimization_results(trials_df, study, optimized_model, output_dir, logger):
    """
    Save optimization results

    Parameters:
    -----------
    trials_df : pandas.DataFrame
        DataFrame with trial results
    study : optuna.Study
        Optuna study
    optimized_model : object
        Trained model with best hyperparameters
    output_dir : str
        Output directory
    logger : logging.Logger
        Logger instance
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save trials dataframe
    trials_path = os.path.join(output_dir, 'hyperparameter_trials.csv')
    trials_df.to_csv(trials_path, index=False)
    logger.info(f"Trials saved to {trials_path}")

    # Save study
    study_path = os.path.join(output_dir, 'optuna_study.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    logger.info(f"Study saved to {study_path}")

    # Save optimized model
    model_path = os.path.join(output_dir, 'optimized_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(optimized_model, f)
    logger.info(f"Optimized model saved to {model_path}")


def run_hyperparameter_tuning(datasets_path, output_dir, logger, n_trials=100, timeout=600):
    """
    Run the complete hyperparameter tuning pipeline

    Parameters:
    -----------
    datasets_path : str
        Path to the prepared datasets
    output_dir : str
        Output directory
    logger : logging.Logger
        Logger instance
    n_trials : int
        Number of trials for optimization
    timeout : int
        Timeout in seconds

    Returns:
    --------
    object
        Trained model with best hyperparameters
    """
    # Load data
    datasets = load_data(datasets_path, logger)

    # Get training data
    X_train = datasets.get('X_train_scaled', datasets['X_train'])
    y_train = datasets['y_train']

    # Get patient IDs (assuming they're in the index of X_train)
    # If patient IDs are stored differently, adjust this
    patient_ids = datasets.get('patient_ids_train', pd.Series([x.split('-')[0] for x in X_train.index.astype(str)]))

    # Optimize hyperparameters
    best_params, best_score, trials_df, study = optimize_hyperparameters(
        X_train, y_train, patient_ids, logger, n_trials, timeout
    )

    # Train model with best hyperparameters
    optimized_model = train_optimized_model(X_train, y_train, best_params, logger)

    # Save results
    save_optimization_results(trials_df, study, optimized_model, output_dir, logger)

    return optimized_model, best_params, best_score


if __name__ == "__main__":
    # This allows the module to be run as a script for testing
    from utils.project_setup import setup_logging, create_project_structure

    # Create project structure
    log_dir, plot_dir, model_dir = create_project_structure()

    # Setup logging
    logger = setup_logging(log_dir, logger_name="hyperparameter_tuning")
    logger.info("=== HYPERPARAMETER TUNING ===")

    # Run hyperparameter tuning
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
                output_dir = os.path.join(model_dir, "hyperparameter_tuning")

                # Run hyperparameter tuning with fewer trials for demonstration
                optimized_model, best_params, best_score = run_hyperparameter_tuning(
                    datasets_path, output_dir, logger, n_trials=20, timeout=300
                )

                logger.info("=== HYPERPARAMETER TUNING COMPLETE ===")
            else:
                logger.error(f"Prepared datasets not found at {datasets_path}")
        else:
            logger.error(f"No subdirectories found in {model_dev_dir}")
    else:
        logger.error(f"Model development directory not found at {model_dev_dir}")
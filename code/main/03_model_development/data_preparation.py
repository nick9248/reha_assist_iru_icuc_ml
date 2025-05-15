# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import os


def prepare_data_split(df, logger, test_size=0.15, val_size=0.15, random_state=42):
    """
    Prepare dataset splits for model training and evaluation,
    ensuring patient-level separation

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with engineered features
    logger : logging.Logger
        Logger instance
    test_size : float
        Proportion of data for test set
    val_size : float
        Proportion of data for validation set
    random_state : int
        Random seed for reproducibility

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
    # First split: separate test set
    remaining_patients, test_patients = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state
    )

    # Second split: separate validation set from remaining
    val_size_adjusted = val_size / (1 - test_size)
    train_patients, val_patients = train_test_split(
        remaining_patients, test_size=val_size_adjusted, random_state=random_state
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


def apply_smote(X_train, y_train, logger, random_state=42):
    """
    Apply SMOTE for handling class imbalance

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    logger : logging.Logger
        Logger instance
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X_train_smote : pandas.DataFrame
        SMOTE-resampled training features
    y_train_smote : pandas.Series
        SMOTE-resampled training target
    """
    logger.info("Applying SMOTE to handle class imbalance")

    # Check for NaN values in the target
    if y_train.isna().any():
        logger.warning(f"Found {y_train.isna().sum()} NaN values in target variable. Removing rows with NaN targets.")
        # Get indices of non-NaN values
        valid_indices = y_train[~y_train.isna()].index
        # Filter X_train and y_train
        X_train_clean = X_train.loc[valid_indices]
        y_train_clean = y_train.loc[valid_indices]
        logger.info(f"Rows after removing NaN targets: {len(X_train_clean)}")
    else:
        X_train_clean = X_train
        y_train_clean = y_train

    # Check for NaN values in features
    if X_train_clean.isna().any().any():
        logger.warning(f"Found NaN values in features. Filling with median values.")
        # Fill NaN values with median
        X_train_clean = X_train_clean.fillna(X_train_clean.median())

    # Original class distribution
    logger.info("Original class distribution:")
    logger.info(y_train_clean.value_counts(normalize=True) * 100)

    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_clean, y_train_clean)

    # New class distribution
    logger.info("Class distribution after SMOTE:")
    logger.info(pd.Series(y_train_smote).value_counts(normalize=True) * 100)
    logger.info(f"SMOTE increased sample size from {len(X_train_clean)} to {len(X_train_smote)}")

    return X_train_smote, y_train_smote


def scale_features(X_train, X_val, X_test, logger):
    """
    Scale features using StandardScaler

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    X_val : pandas.DataFrame
        Validation features
    X_test : pandas.DataFrame
        Test features
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    X_train_scaled : pandas.DataFrame
        Scaled training features
    X_val_scaled : pandas.DataFrame
        Scaled validation features
    X_test_scaled : pandas.DataFrame
        Scaled test features
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler
    """
    logger.info("Scaling features using StandardScaler")

    # Initialize scaler
    scaler = StandardScaler()

    # Fit on training data only
    scaler.fit(X_train)

    # Transform all datasets
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    logger.info("Feature scaling complete")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_datasets(df, logger, apply_scaling=True, handle_imbalance=False):
    """
    Complete data preparation pipeline

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with engineered features
    logger : logging.Logger
        Logger instance
    apply_scaling : bool
        Whether to scale features
    handle_imbalance : bool
        Whether to apply SMOTE for class imbalance

    Returns:
    --------
    dict containing:
        X_train, X_val, X_test : pandas.DataFrame
            Feature sets for train, validation, test
        y_train, y_val, y_test : pandas.Series
            Target values for train, validation, test
        X_train_smote, y_train_smote : pandas.DataFrame, pandas.Series (optional)
            SMOTE-resampled training data (if handle_imbalance=True)
        scaler : sklearn.preprocessing.StandardScaler (optional)
            Fitted scaler (if apply_scaling=True)
    """
    logger.info("Starting complete data preparation pipeline")

    # Create binary target if needed and filter out nbe=2 cases
    if 'nbe_binary' not in df.columns:
        logger.info("Creating binary target variable")
        df = df[df['nbe'] != 2].copy()  # Remove 'no information' cases
        df['nbe_binary'] = df['nbe'].astype(float)  # Convert to float

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_split(df, logger)

    # Check for any remaining NaN values in target
    for target_name, target in [('y_train', y_train), ('y_val', y_val), ('y_test', y_test)]:
        if target.isna().any():
            logger.warning(f"Found {target.isna().sum()} NaN values in {target_name}. Removing these rows.")
            # Get non-NaN indices
            valid_indices = target[~target.isna()].index
            if target_name == 'y_train':
                X_train = X_train.loc[valid_indices]
                y_train = y_train.loc[valid_indices]
            elif target_name == 'y_val':
                X_val = X_val.loc[valid_indices]
                y_val = y_val.loc[valid_indices]
            else:  # y_test
                X_test = X_test.loc[valid_indices]
                y_test = y_test.loc[valid_indices]

    # Initialize results dictionary
    datasets = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

    # Apply SMOTE if requested
    if handle_imbalance:
        X_train_smote, y_train_smote = apply_smote(X_train, y_train, logger)
        datasets['X_train_smote'] = X_train_smote
        datasets['y_train_smote'] = y_train_smote

    # Scale features if requested
    if apply_scaling:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test, logger)
        datasets['X_train_scaled'] = X_train_scaled
        datasets['X_val_scaled'] = X_val_scaled
        datasets['X_test_scaled'] = X_test_scaled
        datasets['scaler'] = scaler

        # Also scale SMOTE data if it exists
        if handle_imbalance and 'X_train_smote' in datasets:
            datasets['X_train_smote_scaled'] = pd.DataFrame(
                scaler.transform(datasets['X_train_smote']),
                columns=datasets['X_train_smote'].columns,
                index=datasets['X_train_smote'].index
            )

    logger.info("Data preparation complete")
    return datasets


# Example usage (when run as a script)
if __name__ == "__main__":
    from utils.project_setup import create_project_structure, setup_logging
    from utils.data_loader import load_engineered_dataset

    # Setup project structure and logging
    log_dir, plot_dir, model_dir = create_project_structure()
    logger = setup_logging(log_dir, "data_preparation")

    # Load dataset
    df = load_engineered_dataset(logger)

    # Prepare datasets
    datasets = prepare_datasets(df, logger, apply_scaling=True, handle_imbalance=True)

    # Save processed datasets for later use
    import pickle

    with open(os.path.join(model_dir, "prepared_datasets.pkl"), "wb") as f:
        pickle.dump(datasets, f)

    logger.info(f"Prepared datasets saved to: {os.path.join(model_dir, 'prepared_datasets.pkl')}")
# Import necessary libraries
import os
import pandas as pd
import logging


def load_engineered_dataset(logger):
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
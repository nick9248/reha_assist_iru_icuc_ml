# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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
    """
    # Get base folders from environment variables
    log_base = os.environ.get('LOG_FOLDER')
    plot_base = os.environ.get('PLOT_FOLDER')

    # Validate environment variables
    if not log_base:
        raise ValueError("LOG_FOLDER environment variable is not set.")
    if not plot_base:
        raise ValueError("PLOT_FOLDER environment variable is not set.")

    # Create timestamp for unique folder names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create structured folders
    stage_name = "02_feature_engineering"
    log_dir = os.path.join(log_base, stage_name, timestamp)
    plot_dir = os.path.join(plot_base, stage_name, timestamp)

    # Create directories
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    return log_dir, plot_dir


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
    log_file = os.path.join(log_folder, f"feature_engineering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create logger
    logger = logging.getLogger('feature_engineering')
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
    Load the processed dataset from the previous step

    Parameters:
    -----------
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df : pandas.DataFrame
        The loaded dataset
    """
    # Get dataset path
    dataset_path = os.environ.get('DATASET')
    if not dataset_path:
        raise ValueError("DATASET environment variable is not set.")

    logger.info(f"Loading raw data from: {dataset_path}")

    # Determine file type and load accordingly
    if dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
        df = pd.read_excel(dataset_path)
    elif dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.pkl') or dataset_path.endswith('.pickle'):
        df = pd.read_pickle(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")

    # Basic preprocessing
    if 'accident_date' in df.columns and not pd.api.types.is_datetime64_dtype(df['accident_date']):
        df['accident_date'] = pd.to_datetime(df['accident_date'], errors='coerce')

    if 'contact_date' in df.columns and not pd.api.types.is_datetime64_dtype(df['contact_date']):
        df['contact_date'] = pd.to_datetime(df['contact_date'], errors='coerce')

    # Calculate days since accident if not already present
    if 'days_since_accident' not in df.columns:
        df['days_since_accident'] = (df['contact_date'] - df['accident_date']).dt.days

    # Sort data by accident_number and contact_date
    df_sorted = df.sort_values(['accident_number', 'contact_date'])

    # Add consultation sequence for each patient if not already present
    if 'consult_seq' not in df.columns:
        df_sorted['consult_seq'] = df_sorted.groupby('accident_number').cumcount() + 1

    # Calculate days between consultations if not already present
    if 'days_since_last_consult' not in df.columns:
        df_sorted['days_since_last_consult'] = df_sorted.groupby('accident_number')['contact_date'].diff().dt.days

    logger.info(f"Dataset loaded with shape: {df_sorted.shape}")
    return df_sorted


def create_time_based_features(df, logger):
    """
    Create features based on time and temporal patterns

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df : pandas.DataFrame
        Dataframe with additional time-based features
    """
    logger.info("Creating time-based features")

    # Create recovery timeline stage (early, mid, late recovery)
    recovery_bins = [0, 30, 90, 180, 365, float('inf')]
    recovery_labels = ['very_early', 'early', 'mid', 'late', 'very_late']
    df['recovery_stage'] = pd.cut(df['days_since_accident'], bins=recovery_bins, labels=recovery_labels)

    # Extract date components
    df['accident_year'] = df['accident_date'].dt.year
    df['accident_month'] = df['accident_date'].dt.month
    df['accident_day'] = df['accident_date'].dt.day
    df['accident_dayofweek'] = df['accident_date'].dt.dayofweek

    df['contact_year'] = df['contact_date'].dt.year
    df['contact_month'] = df['contact_date'].dt.month
    df['contact_day'] = df['contact_date'].dt.day
    df['contact_dayofweek'] = df['contact_date'].dt.dayofweek

    # Create time difference features between consecutive consultations (rate features)
    df['days_per_consult'] = df['days_since_accident'] / df['consult_seq']

    # Create consultation density features
    df['consult_frequency'] = df.groupby('accident_number')['accident_number'].transform('count')
    df['consult_density'] = df['consult_frequency'] / (
                df.groupby('accident_number')['days_since_accident'].transform('max') + 1)

    # Create weekend/weekday indicators
    df['is_accident_weekend'] = df['accident_dayofweek'].isin([5, 6]).astype(int)
    df['is_contact_weekend'] = df['contact_dayofweek'].isin([5, 6]).astype(int)

    logger.info(f"Created {df.shape[1] - 9} time-based features")
    return df


def create_score_features(df, logger):
    """
    Create features based on pain and function limitation scores

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df : pandas.DataFrame
        Dataframe with additional score-based features
    """
    logger.info("Creating score-based features")

    # Create combined score features
    df['total_score'] = df['p_score'] + df['fl_score']
    df['score_ratio'] = df['p_score'] / (df['fl_score'] + 0.1)  # Add 0.1 to avoid division by zero

    # Create status-based features
    df['total_status'] = df['p_status'] + df['fl_status']
    df['is_improving'] = ((df['p_status'] == 2) & (df['fl_status'] == 2)).astype(int)
    df['is_worsening'] = ((df['p_status'] == 0) | (df['fl_status'] == 0)).astype(int)

    # Create interaction features
    df['p_score_status_interaction'] = df['p_score'] * df['p_status']
    df['fl_score_status_interaction'] = df['fl_score'] * df['fl_status']

    # Create normalized scores (relative to maximum)
    df['p_score_normalized'] = df['p_score'] / 4.0
    df['fl_score_normalized'] = df['fl_score'] / 4.0

    # Create score difference from expected based on recovery stage
    p_score_by_stage = df.groupby('recovery_stage')['p_score'].transform('mean')
    fl_score_by_stage = df.groupby('recovery_stage')['fl_score'].transform('mean')

    df['p_score_vs_expected'] = df['p_score'] - p_score_by_stage
    df['fl_score_vs_expected'] = df['fl_score'] - fl_score_by_stage

    logger.info(f"Created {df.shape[1] - 25} score-based features")
    return df


def create_sequential_features(df, logger):
    """
    Create features based on consultation sequence and patient history

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df : pandas.DataFrame
        Dataframe with additional sequential features
    """
    logger.info("Creating sequential features")

    # Mark first and last consultation
    df['is_first_consult'] = (df['consult_seq'] == 1).astype(int)
    df['is_last_consult'] = df.groupby('accident_number')['consult_seq'].transform(lambda x: x == x.max()).astype(int)

    # Create features for score changes
    df['p_score_diff'] = df.groupby('accident_number')['p_score'].diff()
    df['fl_score_diff'] = df.groupby('accident_number')['fl_score'].diff()

    # Create features for cumulative changes
    df['p_score_cumsum'] = df.groupby('accident_number')['p_score'].cumsum()
    df['fl_score_cumsum'] = df.groupby('accident_number')['fl_score'].cumsum()

    # Create features for mean scores up to current consultation
    df['p_score_mean_sofar'] = df.groupby('accident_number')['p_score'].transform(
        lambda x: x.expanding().mean())
    df['fl_score_mean_sofar'] = df.groupby('accident_number')['fl_score'].transform(
        lambda x: x.expanding().mean())

    # Create features for status consistency
    df['p_status_changes'] = df.groupby('accident_number')['p_status'].transform(
        lambda x: x.diff().abs().cumsum())
    df['fl_status_changes'] = df.groupby('accident_number')['fl_status'].transform(
        lambda x: x.diff().abs().cumsum())

    # Create features for previous NBE result
    df['prev_nbe'] = df.groupby('accident_number')['nbe'].shift(1)
    df['nbe_same_as_prev'] = (df['nbe'] == df['prev_nbe']).astype(int)

    # Calculate rate of change for scores
    df['p_score_rate'] = df['p_score_diff'] / df['days_since_last_consult'].replace(0, 1)
    df['fl_score_rate'] = df['fl_score_diff'] / df['days_since_last_consult'].replace(0, 1)

    # Calculate exponentially weighted features (with higher weight to recent consultations)
    df['p_score_ewm'] = df.groupby('accident_number')['p_score'].transform(
        lambda x: x.ewm(span=3).mean())
    df['fl_score_ewm'] = df.groupby('accident_number')['fl_score'].transform(
        lambda x: x.ewm(span=3).mean())

    logger.info(f"Created {df.shape[1] - 35} sequential features")
    return df


def create_categorical_features(df, logger):
    """
    Create features based on categorical variables

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df : pandas.DataFrame
        Dataframe with additional categorical features
    """
    logger.info("Creating categorical features")

    # Create one-hot encoding for telephone category
    telephone_dummies = pd.get_dummies(df['telephone_category'], prefix='telephone_cat')
    df = pd.concat([df, telephone_dummies], axis=1)

    # Create one-hot encoding for recovery stage
    recovery_dummies = pd.get_dummies(df['recovery_stage'], prefix='recovery')
    df = pd.concat([df, recovery_dummies], axis=1)

    # Create complex case flag
    df['is_complex_case'] = (df['telephone_category'] == 4).astype(int)

    # Create case closed flag
    df['is_case_closed'] = (df['telephone_category'] == 3).astype(int)

    # Create combinations of categorical features
    df['tel_recovery_combined'] = df['telephone_category'].astype(str) + "_" + df['recovery_stage'].astype(str)

    # Create dummies for combined features
    tel_recovery_dummies = pd.get_dummies(df['tel_recovery_combined'], prefix='tel_recovery')

    # Only include the most frequent combinations to avoid too many features
    top_combinations = df['tel_recovery_combined'].value_counts().head(10).index
    filtered_columns = [col for col in tel_recovery_dummies.columns
                        if any(combo in col for combo in top_combinations)]
    tel_recovery_dummies = tel_recovery_dummies[filtered_columns]

    # Add the dummies to the dataframe
    df = pd.concat([df, tel_recovery_dummies], axis=1)

    logger.info(
        f"Created {df.shape[1] - 53 - len(telephone_dummies.columns) - len(recovery_dummies.columns)} categorical features")
    return df


def create_aggregate_features(df, logger):
    """
    Create aggregated features at the patient level

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df : pandas.DataFrame
        Dataframe with additional aggregate features
    """
    logger.info("Creating aggregate features")

    # Calculate patient-level statistics for pain score
    df['p_score_patient_mean'] = df.groupby('accident_number')['p_score'].transform('mean')
    df['p_score_patient_min'] = df.groupby('accident_number')['p_score'].transform('min')
    df['p_score_patient_max'] = df.groupby('accident_number')['p_score'].transform('max')
    df['p_score_patient_range'] = df['p_score_patient_max'] - df['p_score_patient_min']
    df['p_score_patient_std'] = df.groupby('accident_number')['p_score'].transform('std').fillna(0)

    # Calculate patient-level statistics for function limitation score
    df['fl_score_patient_mean'] = df.groupby('accident_number')['fl_score'].transform('mean')
    df['fl_score_patient_min'] = df.groupby('accident_number')['fl_score'].transform('min')
    df['fl_score_patient_max'] = df.groupby('accident_number')['fl_score'].transform('max')
    df['fl_score_patient_range'] = df['fl_score_patient_max'] - df['fl_score_patient_min']
    df['fl_score_patient_std'] = df.groupby('accident_number')['fl_score'].transform('std').fillna(0)

    # Calculate patient-level proportions for NBE outcomes
    df['patient_prop_nbe_1'] = df.groupby('accident_number')['nbe'].transform(lambda x: (x == 1).mean())
    df['patient_prop_nbe_0'] = df.groupby('accident_number')['nbe'].transform(lambda x: (x == 0).mean())

    # Calculate patient-level variation in consultation spacing
    df['consult_spacing_std'] = df.groupby('accident_number')['days_since_last_consult'].transform('std').fillna(0)
    df['consult_spacing_mean'] = df.groupby('accident_number')['days_since_last_consult'].transform('mean').fillna(0)
    df['consult_spacing_cv'] = df['consult_spacing_std'] / df['consult_spacing_mean'].replace(0, 1)

    # Calculate proportion of consultations with improvement
    df['patient_prop_improving'] = df.groupby('accident_number')['is_improving'].transform('mean')

    # Calculate proportion of consultations with worsening
    df['patient_prop_worsening'] = df.groupby('accident_number')['is_worsening'].transform('mean')

    # Count number of new features created
    start_cols = df.shape[1]
    agg_features_count = 16  # Number of features created in this function

    logger.info(f"Created {agg_features_count} aggregate features")
    return df


def handle_missing_values(df, logger):
    """
    Handle missing values in the dataframe

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with missing values
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df : pandas.DataFrame
        Dataframe with handled missing values
    """
    logger.info("Handling missing values")

    # Check for missing values
    missing_count = df.isnull().sum()
    missing_features = missing_count[missing_count > 0]

    if len(missing_features) > 0:
        logger.info(f"Found {len(missing_features)} features with missing values:")
        for feature, count in missing_features.items():
            logger.info(f"  - {feature}: {count} missing values ({count / len(df) * 100:.2f}%)")

        # Handle missing values in different feature types

        # Sequential features with missing values (typically first consultation)
        sequential_features = ['p_score_diff', 'fl_score_diff', 'days_since_last_consult',
                               'p_score_rate', 'fl_score_rate', 'prev_nbe', 'nbe_same_as_prev']

        for feature in sequential_features:
            if feature in df.columns and feature in missing_features:
                # For first consultation, set differences to 0
                df[feature] = df[feature].fillna(0)

        # Fill remaining numerical features with median
        numerical_features = df.select_dtypes(include=['number']).columns
        for feature in numerical_features:
            if feature in missing_features:
                median_value = df[feature].median()
                df[feature] = df[feature].fillna(median_value)
                logger.info(f"  - Filled {feature} with median value: {median_value}")

        # Fill categorical features with mode
        categorical_features = df.select_dtypes(include=['object', 'category']).columns
        for feature in categorical_features:
            if feature in missing_features:
                mode_value = df[feature].mode()[0]
                df[feature] = df[feature].fillna(mode_value)
                logger.info(f"  - Filled {feature} with mode value: {mode_value}")
    else:
        logger.info("No missing values found in the dataset")

    return df


def evaluate_features(df, logger, plot_dir):
    """
    Evaluate created features and their relationship with the target

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with engineered features
    logger : logging.Logger
        Logger instance
    plot_dir : str
        Directory to save plots

    Returns:
    --------
    feature_importance : pandas.Series
        Feature importance scores
    """
    logger.info("Evaluating engineered features")

    # Prepare for feature evaluation
    from sklearn.ensemble import RandomForestClassifier

    # Select only numerical features for evaluation
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()

    # Remove target and redundant columns
    exclude_cols = ['nbe', 'accident_number', 'accident_date', 'contact_date',
                    'accident_year', 'accident_month', 'accident_day', 'accident_dayofweek',
                    'contact_year', 'contact_month', 'contact_day', 'contact_dayofweek']

    features_for_evaluation = [col for col in numeric_features if col not in exclude_cols]

    # Get data ready for model
    X = df[features_for_evaluation]
    y = df['nbe']

    # Handle any remaining missing values
    X = X.fillna(X.median())

    # Initialize model for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Get feature importance
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    logger.info(f"Top 10 most important features:")
    for feature, importance in feature_importance.head(10).items():
        logger.info(f"  - {feature}: {importance:.4f}")

    # Save feature importance plot
    plt.figure(figsize=(12, 10))
    feature_importance.head(20).plot(kind='barh')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'feature_importance.png'))
    plt.close()

    # Plot correlation matrix of top features with target
    plt.figure(figsize=(14, 12))
    top_features = feature_importance.head(15).index.tolist()
    corr_data = df[top_features + ['nbe']]
    corr_matrix = corr_data.corr()

    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Top 15 Features')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'top_features_correlation.png'))
    plt.close()

    # Create histogram plots for top features by NBE class
    for feature in top_features[:5]:
        plt.figure(figsize=(10, 6))
        for nbe_value in sorted(df['nbe'].unique()):
            subset = df[df['nbe'] == nbe_value]
            sns.kdeplot(subset[feature], label=f'NBE={nbe_value}')

        plt.title(f'Distribution of {feature} by NBE Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{feature}_by_nbe.png'))
        plt.close()

    return feature_importance


def prepare_final_dataset(df, feature_importance, logger):
    """
    Prepare the final dataset for modeling

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with all features
    feature_importance : pandas.Series
        Feature importance scores
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df_final : pandas.DataFrame
        Final prepared dataset
    """
    logger.info("Preparing final dataset for modeling")

    # Select top features
    top_features = feature_importance.head(50).index.tolist()

    # Add essential columns
    essential_cols = ['accident_number', 'accident_date', 'contact_date',
                      'fl_score', 'fl_status', 'p_score', 'p_status',
                      'telephone_category', 'consult_seq', 'days_since_accident']

    final_cols = list(set(top_features + essential_cols))

    # Add target variable
    final_cols.append('nbe')

    # Create final dataset
    df_final = df[final_cols].copy()

    # Convert target to binary for probability prediction (if nbe=2, exclude from training)
    df_final['nbe_binary'] = df_final['nbe'].map({0: 0, 1: 1})

    # Number of samples for each class
    logger.info(f"Final dataset shape: {df_final.shape}")
    logger.info(f"Number of selected features: {len(final_cols)}")

    return df_final


def main():
    """
    Main function to execute the feature engineering pipeline
    """
    # Create project structure
    log_dir, plot_dir = create_project_structure()

    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("=== FEATURE ENGINEERING PIPELINE ===")

    # Load the processed dataset
    df = load_dataset(logger)

    # Original feature count
    original_feature_count = df.shape[1]
    logger.info(f"Original dataset has {original_feature_count} features")

    # Create features
    df = create_time_based_features(df, logger)
    df = create_score_features(df, logger)
    df = create_sequential_features(df, logger)
    df = create_categorical_features(df, logger)
    df = create_aggregate_features(df, logger)

    # Handle missing values
    df = handle_missing_values(df, logger)

    # Feature count after engineering
    engineered_feature_count = df.shape[1]
    logger.info(f"After feature engineering: {engineered_feature_count} features")
    logger.info(f"Created {engineered_feature_count - original_feature_count} new features")

    # Evaluate features
    feature_importance = evaluate_features(df, logger, plot_dir)

    # Prepare final dataset
    df_final = prepare_final_dataset(df, feature_importance, logger)

    # Save the final dataset
    final_data_path = os.path.join(log_dir, "engineered_features.pkl")
    df_final.to_pickle(final_data_path)
    logger.info(f"Final engineered dataset saved to: {final_data_path}")

    # Save feature importance
    feature_importance.to_csv(os.path.join(log_dir, "feature_importance.csv"))

    logger.info("=== FEATURE ENGINEERING COMPLETE ===")
    logger.info("Next step: Model selection and development")

    return df_final, feature_importance


if __name__ == "__main__":
    main()
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
import time

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
    stage_name = "01_data_exploration"
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
    # Create log folder if it doesn't exist
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = os.path.join(log_folder, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create logger
    logger = logging.getLogger('patient_analysis')
    logger.setLevel(logging.INFO)

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


def load_data(file_path, logger):
    """
    Load dataset from various file formats

    Parameters:
    -----------
    file_path : str
        Path to the file containing the dataset
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataframe
    """
    logger.info(f"Loading data from: {file_path}")

    file_extension = file_path.split('.')[-1].lower()

    if file_extension in ['xlsx', 'xls']:
        df = pd.read_excel(file_path)
    elif file_extension == 'csv':
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            logger.info("UTF-8 encoding failed, trying latin1 encoding")
            df = pd.read_csv(file_path, encoding='latin1')
    else:
        error_msg = f"Unsupported file format: {file_extension}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return df


def preprocess_data(df, logger):
    """
    Preprocess the loaded data

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataframe
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    df : pandas.DataFrame
        Preprocessed dataframe
    """
    logger.info("Preprocessing data...")

    # Convert date columns to datetime
    df['accident_date'] = pd.to_datetime(df['accident_date'], errors='coerce')
    df['contact_date'] = pd.to_datetime(df['contact_date'], errors='coerce')

    # Calculate days since accident
    df['days_since_accident'] = (df['contact_date'] - df['accident_date']).dt.days

    # Sort data by accident_number and contact_date
    df_sorted = df.sort_values(['accident_number', 'contact_date'])

    # Calculate days between consultations for each patient
    df_sorted['days_since_last_consult'] = df_sorted.groupby('accident_number')['contact_date'].diff().dt.days

    # Add consultation sequence for each patient
    df_sorted['consult_seq'] = df_sorted.groupby('accident_number').cumcount() + 1

    return df_sorted


def explore_data_basic(df, logger):
    """
    Perform basic data exploration

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe
    logger : logging.Logger
        Logger instance
    """
    logger.info("=== Dataset Overview ===")
    logger.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    logger.info("\nColumn names and data types:")
    logger.info(df.dtypes)

    # Display number of unique patients
    num_patients = df['accident_number'].nunique()
    logger.info(f"\nNumber of unique patients: {num_patients}")

    # Calculate consultations per patient
    consultations_per_patient = df.groupby('accident_number').size()
    logger.info(f"\nConsultations per patient:")
    logger.info(f"  Average: {consultations_per_patient.mean():.2f}")
    logger.info(f"  Min: {consultations_per_patient.min()}")
    logger.info(f"  Max: {consultations_per_patient.max()}")
    logger.info(f"  Distribution:\n{consultations_per_patient.value_counts().sort_index()}")

    # Check for missing values
    logger.info("\n=== Missing Values ===")
    logger.info(df.isnull().sum())

    # Summary statistics for numerical features
    logger.info("\n=== Summary Statistics ===")
    logger.info(df.describe())


def analyze_target_distribution(df, logger):
    """
    Analyze target variable distribution

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe
    logger : logging.Logger
        Logger instance
    """
    logger.info("\n=== Target Variable (nbe) Distribution ===")
    nbe_counts = df['nbe'].value_counts(normalize=True) * 100
    logger.info(nbe_counts)

    # Telephone category distribution
    logger.info("\n=== Telephone Category Distribution ===")
    telephone_mapping = {
        0: "First Contact",
        1: "Following Contact",
        2: "Not Reached",
        3: "Case Closed",
        4: "Complex Case"
    }
    tel_counts = df['telephone_category'].value_counts()
    for category, count in tel_counts.items():
        logger.info(
            f"  {category} ({telephone_mapping.get(category, 'Unknown')}): {count} ({count / len(df) * 100:.2f}%)")


def analyze_temporal_patterns(df, logger):
    """
    Analyze temporal patterns in the consultations

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe
    logger : logging.Logger
        Logger instance
    """
    logger.info("\n=== Temporal Patterns ===")
    logger.info(f"Average days since accident: {df['days_since_accident'].mean():.2f}")
    logger.info(f"Average days between consultations: {df['days_since_last_consult'].dropna().mean():.2f}")


def get_data_insights(df, logger):
    """
    Generate key insights from the data analysis

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe
    logger : logging.Logger
        Logger instance
    """
    logger.info("\n=== KEY INSIGHTS ===")

    # 1. NBE distribution insights
    nbe_dist = df['nbe'].value_counts(normalize=True) * 100
    logger.info(f"1. NBE Distribution: {nbe_dist.to_dict()}")

    # 2. Most important correlations with NBE
    corr_cols = ['p_score', 'p_status', 'fl_score', 'fl_status', 'telephone_category']
    nbe_corr = df[corr_cols + ['nbe']].corr()['nbe'].drop('nbe').sort_values(ascending=False)
    logger.info(f"2. Top correlations with NBE:\n{nbe_corr}")

    # 3. Best predictors analysis
    # Using basic decision tree to get feature importance
    from sklearn.tree import DecisionTreeClassifier
    X = pd.get_dummies(df[['p_score', 'p_status', 'fl_score', 'fl_status', 'telephone_category',
                           'days_since_accident']], drop_first=True)
    y = df['nbe']

    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X, y)

    feature_importance = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
    logger.info(f"3. Top 5 feature importance from decision tree:\n{feature_importance.head()}")

    # 4. NBE by telephone category
    nbe_by_tel = pd.crosstab(df['telephone_category'], df['nbe'], normalize='index') * 100
    logger.info(f"4. NBE distribution by telephone category:\n{nbe_by_tel}")


def create_distribution_plots(df, plot_folder):
    """
    Create plots showing distributions of key variables

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe
    plot_folder : str
        Folder to save plots
    """
    # Create plot folder if it doesn't exist
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    # 1. Create a figure with subplots for score distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Pain and function limitation scores
    sns.countplot(x='p_score', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Pain Scores')
    axes[0, 0].set_xlabel('Pain Score (0=No Pain, 4=Maximum Pain)')
    axes[0, 0].set_ylabel('Count')

    sns.countplot(x='fl_score', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Function Limitation Scores')
    axes[0, 1].set_xlabel('Function Limitation (0=No Limit, 4=Maximum Limit)')
    axes[0, 1].set_ylabel('Count')

    # Status values
    sns.countplot(x='p_status', data=df, ax=axes[0, 2])
    axes[0, 2].set_title('Distribution of Pain Status')
    axes[0, 2].set_xlabel('Pain Status (0=Worse, 1=Unchanged, 2=Better)')
    axes[0, 2].set_ylabel('Count')

    sns.countplot(x='fl_status', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Function Limitation Status')
    axes[1, 0].set_xlabel('Function Limitation Status (0=Worse, 1=Unchanged, 2=Better)')
    axes[1, 0].set_ylabel('Count')

    # Target variable
    sns.countplot(x='nbe', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of NBE')
    axes[1, 1].set_xlabel('NBE Value (0=Outside NBE, 1=Within NBE, 2=No Info)')
    axes[1, 1].set_ylabel('Count')

    # Telephone category
    sns.countplot(x='telephone_category', data=df, ax=axes[1, 2])
    axes[1, 2].set_title('Distribution of Telephone Category')
    axes[1, 2].set_xlabel('Telephone Category')
    axes[1, 2].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'data_distributions.png'))
    plt.close()


def create_correlation_plot(df, plot_folder):
    """
    Create correlation heatmap

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe
    plot_folder : str
        Folder to save plots
    """
    plt.figure(figsize=(10, 8))
    corr_cols = ['p_score', 'p_status', 'fl_score', 'fl_status', 'nbe', 'telephone_category', 'days_since_accident']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'correlation_matrix.png'))
    plt.close()


def create_temporal_plots(df, plot_folder):
    """
    Create plots related to temporal patterns

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe
    plot_folder : str
        Folder to save plots
    """
    # Days since accident by NBE
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='nbe', y='days_since_accident', data=df)
    plt.title('Days Since Accident by NBE Category')
    plt.xlabel('NBE Value (0=Outside NBE, 1=Within NBE, 2=No Info)')
    plt.ylabel('Days Since Accident')
    plt.savefig(os.path.join(plot_folder, 'days_since_accident_by_nbe.png'))
    plt.close()

    # Consultation sequence distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='consult_seq', data=df[df['consult_seq'] <= 10])
    plt.title('Distribution of Consultation Sequence')
    plt.xlabel('Consultation Sequence Number')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plot_folder, 'consultation_sequence.png'))
    plt.close()

    # NBE by consultation sequence
    plt.figure(figsize=(12, 6))
    consult_nbe = df.pivot_table(index='consult_seq', columns='nbe',
                                 values='accident_number', aggfunc='count', fill_value=0)
    consult_nbe_pct = consult_nbe.div(consult_nbe.sum(axis=1), axis=0)
    consult_nbe_pct.plot(kind='bar', stacked=True)
    plt.title('NBE Distribution by Consultation Sequence')
    plt.xlabel('Consultation Sequence')
    plt.ylabel('Percentage')
    plt.legend(title='NBE Value', labels=['Outside NBE', 'Within NBE', 'No Info'])
    plt.savefig(os.path.join(plot_folder, 'nbe_by_consult_sequence.png'))
    plt.close()


def create_feature_importance_plot(df, plot_folder):
    """
    Create feature importance plot

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe
    plot_folder : str
        Folder to save plots
    """
    from sklearn.tree import DecisionTreeClassifier

    # Prepare data for model
    X = pd.get_dummies(df[['p_score', 'p_status', 'fl_score', 'fl_status',
                           'telephone_category', 'days_since_accident']], drop_first=True)
    y = df['nbe']

    # Fit a decision tree to get feature importance
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X, y)

    # Plot feature importance
    feature_importance = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    feature_importance.head(10).plot(kind='barh')
    plt.title('Top 10 Feature Importance for NBE Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'feature_importance.png'))
    plt.close()


def create_relationship_plots(df, plot_folder):
    """
    Create plots showing relationships between features and target

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe
    plot_folder : str
        Folder to save plots
    """
    # Pain score vs NBE
    plt.figure(figsize=(10, 8))
    p_score_nbe = pd.crosstab(df['p_score'], df['nbe'], normalize='index')
    sns.heatmap(p_score_nbe, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Relationship Between Pain Score and NBE')
    plt.xlabel('NBE Value (0=Outside NBE, 1=Within NBE, 2=No Info)')
    plt.ylabel('Pain Score (0=No Pain, 4=Maximum Pain)')
    plt.savefig(os.path.join(plot_folder, 'p_score_nbe_relationship.png'))
    plt.close()

    # Function limitation score vs NBE
    plt.figure(figsize=(10, 8))
    fl_score_nbe = pd.crosstab(df['fl_score'], df['nbe'], normalize='index')
    sns.heatmap(fl_score_nbe, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Relationship Between Function Limitation Score and NBE')
    plt.xlabel('NBE Value (0=Outside NBE, 1=Within NBE, 2=No Info)')
    plt.ylabel('Function Limitation Score (0=No Limit, 4=Maximum Limit)')
    plt.savefig(os.path.join(plot_folder, 'fl_score_nbe_relationship.png'))
    plt.close()

    # Pain status vs NBE
    plt.figure(figsize=(10, 8))
    p_status_nbe = pd.crosstab(df['p_status'], df['nbe'], normalize='index')
    sns.heatmap(p_status_nbe, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Relationship Between Pain Status and NBE')
    plt.xlabel('NBE Value (0=Outside NBE, 1=Within NBE, 2=No Info)')
    plt.ylabel('Pain Status (0=Worse, 1=Unchanged, 2=Better)')
    plt.savefig(os.path.join(plot_folder, 'p_status_nbe_relationship.png'))
    plt.close()

    # Function limitation status vs NBE
    plt.figure(figsize=(10, 8))
    fl_status_nbe = pd.crosstab(df['fl_status'], df['nbe'], normalize='index')
    sns.heatmap(fl_status_nbe, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Relationship Between Function Limitation Status and NBE')
    plt.xlabel('NBE Value (0=Outside NBE, 1=Within NBE, 2=No Info)')
    plt.ylabel('Function Limitation Status (0=Worse, 1=Unchanged, 2=Better)')
    plt.savefig(os.path.join(plot_folder, 'fl_status_nbe_relationship.png'))
    plt.close()


def main():
    """
    Main function to execute the data exploration pipeline
    """
    # Create project structure
    log_dir, plot_dir = create_project_structure()

    # Get dataset path
    dataset_path = os.environ.get('DATASET')
    if not dataset_path:
        raise ValueError("DATASET environment variable is not set.")

    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("=== PATIENT CONSULTATION DATA ANALYSIS ===")
    logger.info(f"Using dataset: {dataset_path}")
    logger.info(f"Saving logs to: {log_dir}")
    logger.info(f"Saving plots to: {plot_dir}")

    # Load and preprocess data
    df_raw = load_data(dataset_path, logger)
    df = preprocess_data(df_raw, logger)

    # Exploratory data analysis
    explore_data_basic(df, logger)
    analyze_target_distribution(df, logger)
    analyze_temporal_patterns(df, logger)
    get_data_insights(df, logger)

    # Create visualizations
    create_distribution_plots(df, plot_dir)
    create_correlation_plot(df, plot_dir)
    create_temporal_plots(df, plot_dir)
    create_feature_importance_plot(df, plot_dir)
    create_relationship_plots(df, plot_dir)

    # Save processed dataframe for next steps
    processed_data_path = os.path.join(log_dir, "processed_data.pkl")
    df.to_pickle(processed_data_path)
    logger.info(f"Processed data saved to: {processed_data_path}")

    logger.info("\n=== ANALYSIS COMPLETE ===")
    logger.info("Next steps: Feature engineering and model development")

    return df


# Call the main function when script is executed
if __name__ == "__main__":
    main()
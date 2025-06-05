"""
Patient data exploration and initial analysis.
This script loads patient consultation data, performs initial preprocessing,
and conducts exploratory analysis to understand patterns in the dataset.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import os
from sklearn.tree import DecisionTreeClassifier

# Load environment variables
load_dotenv()

# Display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

np.random.seed(42)


def create_log_folder() -> Path:
    """Create a timestamped log directory and return its path."""
    stage = "01_data_exploration"
    log_base = os.environ.get("LOG_FOLDER")
    if not log_base:
        raise ValueError("LOG_FOLDER must be set in .env")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(log_base) / stage / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger("patient_analysis")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    log_file = log_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_data(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load dataset from various file formats.

    Args:
        path: Path to the dataset file
        logger: Logger instance

    Returns:
        The loaded dataset
    """
    logger.info(f"Loading dataset: {path}")
    ext = path.suffix.lower().lstrip('.')

    try:
        if ext in ["xlsx", "xls"]:
            return pd.read_excel(path)
        elif ext == "csv":
            try:
                return pd.read_csv(path)
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="latin1")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def preprocess_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Preprocess the dataset for analysis.

    Args:
        df: Raw dataframe to preprocess
        logger: Logger instance

    Returns:
        Preprocessed dataframe
    """
    # Convert dates
    df['accident_date'] = pd.to_datetime(df['accident_date'], errors='coerce')
    df['contact_date'] = pd.to_datetime(df['contact_date'], errors='coerce')
    df['days_since_accident'] = (df['contact_date'] - df['accident_date']).dt.days

    # Sort and create sequential features
    df = df.sort_values(['accident_number', 'contact_date'])
    df['days_since_last_consult'] = df.groupby('accident_number')['contact_date'].diff().dt.days
    df['days_since_last_consult'] = df['days_since_last_consult'].fillna(0)
    df['consult_seq'] = df.groupby('accident_number').cumcount() + 1

    # Check for negative days (data errors)
    negative_days = df[df['days_since_accident'] < 0]
    if not negative_days.empty:
        logger.warning(f"{len(negative_days)} records with negative days_since_accident - these will be removed:")
        logger.warning(
            negative_days[['accident_number', 'accident_date', 'contact_date', 'days_since_accident']].to_string(
                index=False))
        # Remove negative days records
        df = df[df['days_since_accident'] >= 0].copy()
        logger.info(f"Dataset shape after removing negative days: {df.shape}")

    return df


def explore_data(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Log basic dataset statistics and distributions."""
    logger.info("=== DATA OVERVIEW ===")
    logger.info(f"Shape: {df.shape}")
    logger.info("Data types:\n%s", df.dtypes)

    num_patients = df['accident_number'].nunique()
    logger.info(f"Unique patients: {num_patients}")

    consult_counts = df.groupby('accident_number').size()
    logger.info("Consults per patient – Avg: %.2f, Min: %d, Max: %d",
                consult_counts.mean(), consult_counts.min(), consult_counts.max())
    logger.info("Consult count distribution:\n%s", consult_counts.value_counts().sort_index())

    # Check for missing values
    missing = df.isnull().sum()
    logger.info("\n=== Missing Values ===\n%s", missing)
    if missing.sum() == 0:
        logger.info("No missing values found in the dataset.")

    logger.info("\n=== Summary Stats ===\n%s", df.describe())


def analyze_target(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Analyze the target variable (NBE) distributions."""
    logger.info("\n=== NBE Distribution ===\n%s", df['nbe'].value_counts(normalize=True) * 100)

    # Check class balance
    nbe_counts = df['nbe'].value_counts()
    if len(nbe_counts) > 1:
        min_class = nbe_counts.min()
        max_class = nbe_counts.max()
        ratio = min_class / max_class
        logger.info(f"Class balance ratio (min/max): {ratio:.4f}")
        if ratio < 0.2:
            logger.warning("Significant class imbalance detected (ratio < 0.2)")

    logger.info("\n=== Telephone Categories ===\n%s",
                df['telephone_category'].value_counts(normalize=True) * 100)


def analyze_temporal(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Analyze temporal patterns in the dataset."""
    logger.info("\n=== TEMPORAL PATTERNS ===")
    logger.info("Avg. days since accident: %.2f", df['days_since_accident'].mean())
    logger.info("Avg. days between consults: %.2f", df['days_since_last_consult'].mean())


def compute_feature_importance(df: pd.DataFrame, logger: logging.Logger) -> pd.Series:
    """
    Compute feature importance using a decision tree classifier.

    Args:
        df: Dataframe with features and target
        logger: Logger instance

    Returns:
        Series with feature importance scores
    """
    logger.info("\n=== DECISION TREE FEATURE IMPORTANCE ===")

    # Prepare features (one-hot encoding for categorical)
    X = pd.get_dummies(df[['p_score', 'p_status', 'fl_score', 'fl_status',
                           'telephone_category', 'days_since_accident']], drop_first=True)
    y = df['nbe']

    # Train simple decision tree
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X, y)

    # Get and sort feature importances
    importances = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    logger.info("Top 5 features:\n%s", importances.head(5))

    return importances


def correlation_analysis(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Analyze correlations between features and the target variable."""
    corr = df[['p_score', 'p_status', 'fl_score', 'fl_status',
               'telephone_category', 'nbe']].corr()['nbe'].drop('nbe')
    logger.info("\n=== CORRELATION WITH NBE ===\n%s", corr.sort_values(ascending=False))


def save_processed_data(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Save processed dataset to pickle and CSV files."""
    # Define output folder structure
    output_base = os.environ.get("OUTPUT_FOLDER")
    if not output_base:
        raise ValueError("OUTPUT_FOLDER is not set in environment variables")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / "01_data_exploration" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save output files
    pkl_path = output_dir / "processed_data.pkl"
    csv_path = output_dir / "processed_data.csv"

    df.to_pickle(pkl_path)
    df.to_csv(csv_path, index=False)

    logger.info(f"Processed data saved to:\n- {pkl_path}\n- {csv_path}")


def main() -> None:
    """Main function to run the data exploration pipeline."""
    try:
        # Create log directory and setup logging
        log_dir = create_log_folder()
        logger = setup_logging(log_dir)
        logger.info("=== PATIENT CONSULTATION DATA ANALYSIS ===")
        logger.info(f"Log directory: {log_dir}")

        # Load dataset path
        dataset_path = os.environ.get("DATASET")
        if not dataset_path:
            raise ValueError("DATASET is not set in environment variables")
        logger.info(f"Dataset path: {dataset_path}")

        # Load and preprocess data
        df_raw = load_data(Path(dataset_path), logger)
        df = preprocess_data(df_raw, logger)

        # Perform analysis steps
        explore_data(df, logger)
        analyze_target(df, logger)
        analyze_temporal(df, logger)
        correlation_analysis(df, logger)
        compute_feature_importance(df, logger)

        # Save processed data
        save_processed_data(df, logger)

        logger.info("=== ANALYSIS COMPLETE ===")

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error in main execution: {e}", exc_info=True)
        else:
            print(f"Error before logger initialization: {e}")
        raise


if __name__ == "__main__":
    main()
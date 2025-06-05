"""
Feature engineering for patient consultation data.
This script generates features for predicting NBE, evaluates feature importance,
and exports the engineered dataset for model training.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import Tuple, List, Dict, Optional, Union
from sklearn.ensemble import RandomForestClassifier

# === Config ===
load_dotenv()
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
np.random.seed(42)

# Constants
STAGE = "02_feature_engineering"
ESSENTIAL_COLS = [
    'accident_number', 'accident_date', 'contact_date',
    'fl_score', 'fl_status', 'p_score', 'p_status',
    'consult_seq', 'days_since_accident', 'nbe'
]


def setup_folders() -> Tuple[str, str, str]:
    """
    Create necessary folders for logs, plots, and outputs.

    Returns:
        Tuple of paths (log_dir, plot_dir, output_dir)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(os.environ["LOG_FOLDER"]) / STAGE / timestamp
    plot_dir = Path(os.environ["PLOT_FOLDER"]) / STAGE / timestamp
    output_dir = Path(os.environ["OUTPUT_FOLDER"]) / STAGE / timestamp

    for d in [log_dir, plot_dir, output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return str(log_dir), str(plot_dir), str(output_dir)


def setup_logger(log_dir: str) -> logging.Logger:
    """
    Set up a logger for the feature engineering process.

    Args:
        log_dir: Directory where log files will be stored

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(STAGE)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(Path(log_dir) / f"{STAGE}.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    return logger


def load_data(logger: logging.Logger) -> pd.DataFrame:
    """
    Load preprocessed dataset from previous pipeline stage.

    Args:
        logger: Logger instance

    Returns:
        Loaded dataframe
    """
    path = os.environ.get("PRE_PROCESSED_DATASET")
    if not path:
        raise ValueError("PRE_PROCESSED_DATASET not defined in .env")

    ext = Path(path).suffix.lower()
    logger.info(f"Loading preprocessed dataset: {path}")

    try:
        if ext == ".pkl":
            return pd.read_pickle(path)
        elif ext == ".csv":
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format for PRE_PROCESSED_DATASET: {ext}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def validate_columns(df: pd.DataFrame, cols: List[str], logger: logging.Logger) -> None:
    """
    Validate that required columns exist in the dataframe.

    Args:
        df: Dataframe to validate
        cols: List of required column names
        logger: Logger instance

    Raises:
        ValueError: If any required columns are missing
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")


def safe_divide(a: Union[pd.Series, np.ndarray],
                b: Union[pd.Series, np.ndarray],
                fill_value: float = 0) -> Union[pd.Series, np.ndarray]:
    """
    Safely divide two series or arrays, handling zeros in denominator.

    Args:
        a: Numerator
        b: Denominator
        fill_value: Value to use when denominator is zero

    Returns:
        Result of division with zeros handled
    """
    return np.divide(a, b, out=np.full_like(a, fill_value, dtype=float), where=b != 0)


def add_temporal_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add features related to time and recovery stages.

    Args:
        df: Input dataframe
        logger: Logger instance

    Returns:
        Dataframe with added temporal features
    """
    logger.info("Adding temporal features")
    df = df.copy()

    # Recovery stages
    df['recovery_stage'] = pd.cut(
        df['days_since_accident'],
        bins=[0, 30, 90, 180, 365, float("inf")],
        labels=['very_early', 'early', 'mid', 'late', 'very_late']
    )

    # Time-based features
    df['days_per_consult'] = safe_divide(df['days_since_accident'], df['consult_seq'])
    df['consult_frequency'] = df.groupby('accident_number')['accident_number'].transform('count')

    # Safe division for consult density
    max_days = df.groupby('accident_number')['days_since_accident'].transform('max') + 1
    df['consult_density'] = safe_divide(df['consult_frequency'], max_days)

    # Weekend indicators
    df['is_accident_weekend'] = df['accident_date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_contact_weekend'] = df['contact_date'].dt.dayofweek.isin([5, 6]).astype(int)

    return df


def add_score_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add features related to scores and their interactions.

    Args:
        df: Input dataframe
        logger: Logger instance

    Returns:
        Dataframe with added score features
    """
    logger.info("Adding score-based features")
    df = df.copy()

    # Basic score combinations
    df['total_score'] = df['p_score'] + df['fl_score']
    df['score_ratio'] = safe_divide(df['p_score'], df['fl_score'] + 0.1)  # Add small constant to avoid division by zero
    df['total_status'] = df['p_status'] + df['fl_status']

    # Status indicators
    df['is_improving'] = ((df['p_status'] == 2) & (df['fl_status'] == 2)).astype(int)
    df['is_worsening'] = ((df['p_status'] == 0) | (df['fl_status'] == 0)).astype(int)

    # Interactions
    df['p_score_status_interaction'] = df['p_score'] * df['p_status']
    df['fl_score_status_interaction'] = df['fl_score'] * df['fl_status']

    # Normalized scores
    df['p_score_normalized'] = df['p_score'] / 4.0
    df['fl_score_normalized'] = df['fl_score'] / 4.0

    # Compare to recovery stage averages (no future information used)
    df['p_score_vs_expected'] = df['p_score'] - df.groupby(['recovery_stage'], observed=False)['p_score'].transform(
        'mean')
    df['fl_score_vs_expected'] = df['fl_score'] - df.groupby(['recovery_stage'], observed=False)['fl_score'].transform(
        'mean')

    return df


def add_sequence_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add features based on sequential aspects of consultations.

    Args:
        df: Input dataframe
        logger: Logger instance

    Returns:
        Dataframe with added sequence features
    """
    logger.info("Adding sequence-based features")
    df = df.copy()

    # Indicators for first and last consultation
    df['is_first_consult'] = (df['consult_seq'] == 1).astype(int)
    df['is_last_consult'] = df.groupby('accident_number')['consult_seq'].transform(lambda x: x == x.max()).astype(int)

    # Changes and accumulated values (avoid using future information)
    df['p_score_diff'] = df.groupby('accident_number')['p_score'].diff().fillna(0)
    df['fl_score_diff'] = df.groupby('accident_number')['fl_score'].diff().fillna(0)
    df['p_score_cumsum'] = df.groupby('accident_number')['p_score'].cumsum()
    df['fl_score_cumsum'] = df.groupby('accident_number')['fl_score'].cumsum()

    # Status changes over time
    df['p_status_changes'] = df.groupby('accident_number')['p_status'].transform(
        lambda x: x.diff().abs().cumsum()).fillna(0)
    df['fl_status_changes'] = df.groupby('accident_number')['fl_status'].transform(
        lambda x: x.diff().abs().cumsum()).fillna(0)

    # Previous NBE (if available, for sequential modeling)
    df['prev_nbe'] = df.groupby('accident_number')['nbe'].shift(1).fillna(2)
    df['nbe_same_as_prev'] = (df['nbe'] == df['prev_nbe']).astype(int)

    # Calculate rates of change (safely)
    days_denom = df['days_since_last_consult'].replace(0, 1)  # Avoid division by zero
    df['p_score_rate'] = safe_divide(df['p_score_diff'], days_denom)
    df['fl_score_rate'] = safe_divide(df['fl_score_diff'], days_denom)

    # Exponentially weighted moving averages
    df['p_score_ewm'] = df.groupby('accident_number')['p_score'].transform(lambda x: x.ewm(span=3).mean())
    df['fl_score_ewm'] = df.groupby('accident_number')['fl_score'].transform(lambda x: x.ewm(span=3).mean())

    # Proper lagged aggregates (no data leakage)
    df['p_score_mean_sofar'] = df.groupby('accident_number')['p_score'].transform(
        lambda x: x.shift().expanding().mean()).fillna(0)
    df['fl_score_mean_sofar'] = df.groupby('accident_number')['fl_score'].transform(
        lambda x: x.shift().expanding().mean()).fillna(0)

    return df


def add_categorical_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Create dummy variables for categorical features.

    Args:
        df: Input dataframe
        logger: Logger instance

    Returns:
        Dataframe with added categorical features
    """
    logger.info("Adding categorical features")

    # Create dummy variables
    return pd.get_dummies(
        df,
        columns=['telephone_category', 'recovery_stage'],
        prefix=['tel', 'stage']
    )


def create_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Create all engineered features for the model.

    Args:
        df: Input dataframe with raw features
        logger: Logger instance

    Returns:
        Dataframe with all engineered features
    """
    logger.info("Starting feature engineering process")

    # Apply feature engineering by category
    df = add_temporal_features(df, logger)
    df = add_score_features(df, logger)
    df = add_sequence_features(df, logger)
    df = add_categorical_features(df, logger)

    logger.info(f"Created {len(df.columns)} total features")
    return df


def evaluate_features(df: pd.DataFrame, logger: logging.Logger, plot_dir: str) -> pd.Series:
    """
    Evaluate feature importance using RandomForest and correlation analysis.

    Args:
        df: DataFrame with all engineered features
        logger: Logger instance
        plot_dir: Directory to save plots

    Returns:
        Series with feature importances
    """
    logger.info("=== Feature Importance Evaluation ===")

    # Filter for labeled data only
    df = df[df['nbe'].isin([0, 1])].copy()

    # Select numerical features and exclude known non-features
    exclude = {'nbe', 'nbe_binary', 'accident_number', 'accident_date', 'contact_date'}
    X = df.select_dtypes(include='number')
    X = X.drop(columns=[col for col in exclude if col in X.columns], errors='ignore')
    y = df['nbe']

    # Log feature set 
    logger.info(f"Using {len(X.columns)} numerical features for importance evaluation")

    # Check for multicollinearity (add correlation analysis)
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(corr_matrix.index[row], corr_matrix.columns[col], upper_tri.iloc[row, col])
                       for row, col in zip(*np.where(upper_tri > 0.8))]

    if high_corr_pairs:
        logger.warning(f"Found {len(high_corr_pairs)} highly correlated feature pairs (r > 0.8):")
        for feat1, feat2, corr_val in high_corr_pairs:
            logger.warning(f"  {feat1} <-> {feat2}: {corr_val:.4f}")

    # Plot correlation with target
    target_corr = X.apply(lambda x: x.corr(y) if x.dtype.kind in 'bifc' else np.nan).dropna()
    target_corr = target_corr.sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    top_corr = target_corr.abs().nlargest(20)
    colors = ['red' if c < 0 else 'blue' for c in target_corr[top_corr.index]]
    top_corr.plot(kind='barh', color=colors)
    plt.title("Top 20 Feature Correlations with Target (NBE)")
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(plot_dir) / "feature_correlation_with_target.png")
    plt.close()

    # Fit RandomForest model
    logger.info("Training RandomForest for feature importance...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Get importance
    importance = pd.Series(model.feature_importances_, index=X.columns)

    # Clean up and sort
    importance = importance[importance.index.notna()]
    importance = importance[importance.index != '']
    importance = importance.sort_values(ascending=False)

    # Log top features
    logger.info("Top 10 features by RandomForest importance:")
    for col, val in importance.head(10).items():
        logger.info(f"  {col}: {val:.5f}")

    # Plot: bar chart
    plt.figure(figsize=(12, 8))
    importance.head(20).plot(kind='barh')
    plt.title("Top 20 Feature Importances (RandomForest)")
    plt.tight_layout()
    plt.savefig(Path(plot_dir) / "feature_importance.png")
    plt.close()

    # Plot: KDEs for top 5 features
    for feature in importance.head(5).index:
        plt.figure(figsize=(10, 6))
        for val in sorted(df['nbe'].unique()):
            subset = df[df['nbe'] == val]
            sns.kdeplot(subset[feature], label=f"NBE={val}", fill=True)
        plt.title(f"Distribution of {feature} by NBE Class")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(plot_dir) / f"{feature}_by_nbe.png")
        plt.close()

    # Combined ranking: combine correlation and importance
    combined_rank = pd.DataFrame({
        'rf_importance': importance,
        'target_corr': target_corr.reindex(importance.index).fillna(0).abs()
    })
    combined_rank['combined_score'] = combined_rank['rf_importance'] * 0.7 + combined_rank['target_corr'] * 0.3
    combined_rank = combined_rank.sort_values('combined_score', ascending=False)

    logger.info("Top 10 features by combined score (70% RF importance, 30% correlation):")
    for col, row in combined_rank.head(10).iterrows():
        logger.info(
            f"  {col}: {row['combined_score']:.5f} (RF: {row['rf_importance']:.5f}, Corr: {row['target_corr']:.5f})")

    return combined_rank['combined_score']


def run_feature_engineering() -> None:
    """Main function to run the feature engineering pipeline."""
    try:
        # Setup environment
        log_dir, plot_dir, output_dir = setup_folders()
        logger = setup_logger(log_dir)
        logger.info("=== FEATURE ENGINEERING START ===")

        # Load and validate data
        df = load_data(logger)
        validate_columns(df, ESSENTIAL_COLS, logger)

        # Data cleaning - remove negative days records
        orig_count = len(df)
        df = df[df['days_since_accident'] >= 0].copy()
        if len(df) < orig_count:
            logger.info(f"Removed {orig_count - len(df)} records with negative days_since_accident")

        # Create features
        df = create_features(df, logger)

        # Create binary target for binary classification
        df['nbe_binary'] = df['nbe'].map({0: 0, 1: 1, 2: np.nan})

        # Evaluate features
        importance = evaluate_features(df, logger, plot_dir)

        # Build final dataset with top features
        top_feats = importance.head(50).index.tolist()
        final_cols = list(set(top_feats + ESSENTIAL_COLS + ['nbe', 'nbe_binary']))
        df_final = df[final_cols]

        # Save outputs
        df_final.to_pickle(Path(output_dir) / "engineered_features.pkl")
        df_final.to_csv(Path(output_dir) / "engineered_features.csv", index=False)
        importance.rename("importance").to_csv(Path(output_dir) / "feature_importance.csv", header=True,
                                               index_label="feature")

        logger.info(f"Final dataset shape: {df_final.shape}")
        logger.info(f"Saved engineered dataset and feature importance to: {output_dir}")
        logger.info("=== FEATURE ENGINEERING COMPLETE ===")

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error in feature engineering: {e}", exc_info=True)
        else:
            print(f"Error before logger initialization: {e}")
        raise


if __name__ == "__main__":
    run_feature_engineering()
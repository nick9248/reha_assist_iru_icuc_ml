# Technical Documentation: Feature Engineering Module

## Overview
This technical documentation covers the implementation details of the `02_feature_engineering.py` script. The script transforms preprocessed patient consultation data into an enriched feature set optimized for predicting the NBE variable.

## Script Structure
The script is organized into modular functions that handle specific aspects of the feature engineering pipeline:

```
02_feature_engineering.py
├── setup_folders() - Creates directory structure for outputs
├── setup_logger() - Configures logging system
├── load_data() - Loads preprocessed data
├── validate_columns() - Ensures required columns exist
├── safe_divide() - Handles division by zero safely
├── add_temporal_features() - Creates time-based features
├── add_score_features() - Creates score-related features
├── add_sequence_features() - Creates sequential features
├── add_categorical_features() - Creates dummy variables
├── create_features() - Orchestrates feature creation process
├── evaluate_features() - Assesses feature importance
└── run_feature_engineering() - Main execution function
```

## Environment Configuration
The script uses environment variables from a `.env` file:
- `LOG_FOLDER`: Directory for log files
- `PLOT_FOLDER`: Directory for visualization plots
- `OUTPUT_FOLDER`: Directory for output datasets
- `PRE_PROCESSED_DATASET`: Path to the preprocessed data from the exploration stage

## Key Implementation Details

### Directory Structure
The script creates a consistent directory structure with timestamped folders:

```python
def setup_folders() -> Tuple[str, str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(os.environ["LOG_FOLDER"]) / STAGE / timestamp
    plot_dir = Path(os.environ["PLOT_FOLDER"]) / STAGE / timestamp
    output_dir = Path(os.environ["OUTPUT_FOLDER"]) / STAGE / timestamp
    
    for d in [log_dir, plot_dir, output_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    return str(log_dir), str(plot_dir), str(output_dir)
```

### Data Validation
The script validates input data before processing:

```python
def validate_columns(df: pd.DataFrame, cols: List[str], logger: logging.Logger) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")
```

### Safe Mathematical Operations
The script implements safe division to handle potential division by zero errors:

```python
def safe_divide(a: Union[pd.Series, np.ndarray], 
                b: Union[pd.Series, np.ndarray], 
                fill_value: float = 0) -> Union[pd.Series, np.ndarray]:
    return np.divide(a, b, out=np.full_like(a, fill_value, dtype=float), where=b!=0)
```

### Modular Feature Creation
The script divides feature creation into logical modules for different feature types:

```python
def create_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Starting feature engineering process")
    
    # Apply feature engineering by category
    df = add_temporal_features(df, logger)
    df = add_score_features(df, logger)
    df = add_sequence_features(df, logger)
    df = add_categorical_features(df, logger)
    
    logger.info(f"Created {len(df.columns)} total features")
    return df
```

#### Temporal Features Implementation
```python
def add_temporal_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
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
```

#### Score Features Implementation
The script creates features based on patient scores and status:

```python
# Examples of score features
df['total_score'] = df['p_score'] + df['fl_score']
df['score_ratio'] = safe_divide(df['p_score'], df['fl_score'] + 0.1)
df['is_improving'] = ((df['p_status'] == 2) & (df['fl_status'] == 2)).astype(int)
```

#### Sequential Features Implementation
The script creates features that capture the sequential aspects of consultations:

```python
# Examples of sequential features
df['p_score_diff'] = df.groupby('accident_number')['p_score'].diff().fillna(0)
df['p_score_cumsum'] = df.groupby('accident_number')['p_score'].cumsum()
df['prev_nbe'] = df.groupby('accident_number')['nbe'].shift(1).fillna(2)
```

### Feature Importance Evaluation
The script evaluates feature importance using multiple methods:

```python
def evaluate_features(df: pd.DataFrame, logger: logging.Logger, plot_dir: str) -> pd.Series:
    # RandomForest importance
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    
    # Correlation with target
    target_corr = X.apply(lambda x: x.corr(y) if x.dtype.kind in 'bifc' else np.nan).dropna()
    
    # Combined ranking
    combined_rank = pd.DataFrame({
        'rf_importance': importance,
        'target_corr': target_corr.reindex(importance.index).fillna(0).abs()
    })
    combined_rank['combined_score'] = combined_rank['rf_importance'] * 0.7 + combined_rank['target_corr'] * 0.3
    
    return combined_rank['combined_score']
```

### Multicollinearity Detection
The script detects and logs highly correlated feature pairs:

```python
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [(corr_matrix.index[row], corr_matrix.columns[col], upper_tri.iloc[row, col]) 
                  for row, col in zip(*np.where(upper_tri > 0.8))]

if high_corr_pairs:
    logger.warning(f"Found {len(high_corr_pairs)} highly correlated feature pairs (r > 0.8):")
    for feat1, feat2, corr_val in high_corr_pairs:
        logger.warning(f"  {feat1} <-> {feat2}: {corr_val:.4f}")
```

### Visualization Implementation
The script creates multiple visualizations:

1. Feature importance bar charts:
```python
plt.figure(figsize=(12, 8))
importance.head(20).plot(kind='barh')
plt.title("Top 20 Feature Importances (RandomForest)")
plt.tight_layout()
plt.savefig(Path(plot_dir) / "feature_importance.png")
```

2. Feature distribution by class:
```python
for feature in importance.head(5).index:
    plt.figure(figsize=(10, 6))
    for val in sorted(df['nbe'].unique()):
        subset = df[df['nbe'] == val]
        sns.kdeplot(subset[feature], label=f"NBE={val}", fill=True)
    plt.title(f"Distribution of {feature} by NBE Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(plot_dir) / f"{feature}_by_nbe.png")
```

3. Target correlation bar charts:
```python
plt.figure(figsize=(12, 8))
top_corr = target_corr.abs().nlargest(20)
colors = ['red' if c < 0 else 'blue' for c in target_corr[top_corr.index]]
top_corr.plot(kind='barh', color=colors)
plt.title("Top 20 Feature Correlations with Target (NBE)")
```

### Output Generation
The script creates several output files:

1. Engineered dataset with selected features (both .pkl and .csv formats)
2. Feature importance rankings (.csv)
3. Visualization plots (.png)

## Technical Dependencies
- Python 3.13
- pandas: For data manipulation
- numpy: For numerical operations
- matplotlib & seaborn: For visualization
- scikit-learn: For RandomForest feature importance
- dotenv: For environment variable loading
- pathlib: For path manipulation
- logging: For logging capabilities

## Performance Considerations
- Uses numpy's optimized functions for mathematical operations
- Consolidates groupby operations where possible
- Uses in-place operations where appropriate
- Implements proper error handling with try-except blocks
- Creates a copy of the dataframe before modifications to prevent unintended changes

## Error Handling
The script wraps the main execution in a try-except block for robust error handling:

```python
try:
    # Pipeline steps
except Exception as e:
    if 'logger' in locals():
        logger.error(f"Error in feature engineering: {e}", exc_info=True)
    else:
        print(f"Error before logger initialization: {e}")
    raise
```

## Type Annotations
The script uses Python type hints throughout to improve code readability and IDE support:

```python
def validate_columns(df: pd.DataFrame, cols: List[str], logger: logging.Logger) -> None:
    # Implementation
    
def safe_divide(a: Union[pd.Series, np.ndarray], 
                b: Union[pd.Series, np.ndarray], 
                fill_value: float = 0) -> Union[pd.Series, np.ndarray]:
    # Implementation
```

Type annotations help with:
- Code documentation
- IDE autocompletion
- Static type checking
- Better understanding of function interfaces

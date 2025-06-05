# Technical Documentation: Data Exploration Module

## Overview
This documentation covers the technical implementation details of the `01_data_exploration.py` script. The script provides a structured pipeline for loading, preprocessing, and analyzing patient consultation data to prepare it for feature engineering and model training.

## Script Structure
The script is organized into focused functions that handle specific aspects of the data exploration pipeline:

```
01_data_exploration.py
├── create_log_folder() - Creates timestamped logging directory
├── setup_logging() - Configures logging to file and console
├── load_data() - Handles loading from different file formats
├── preprocess_data() - Transforms raw data into analysis-ready format
├── explore_data() - Calculates and logs basic statistics
├── analyze_target() - Examines NBE distribution and class balance
├── analyze_temporal() - Analyzes time-based patterns
├── compute_feature_importance() - Calculates initial feature importance
├── correlation_analysis() - Examines correlations with target
├── save_processed_data() - Exports processed data
└── main() - Orchestrates the workflow
```

## Environment Setup
The script uses environment variables from a `.env` file to configure paths:
- `LOG_FOLDER`: Directory where logs are stored
- `DATASET`: Path to the raw dataset
- `OUTPUT_FOLDER`: Directory where processed data is saved

## Key Implementation Details

### Data Loading and Error Handling
```python
def load_data(path: Path, logger: logging.Logger) -> pd.DataFrame:
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
```

The function handles multiple file formats and potential encoding issues. It uses proper exception handling to catch and log errors.

### Preprocessing Implementation
The preprocessing function:
1. Converts date columns to datetime format
2. Calculates days between dates
3. Sorts data by patient and consultation date
4. Computes sequential features (days between consultations, consultation sequence)
5. Filters out records with negative days (data entry errors)

```python
# Key preprocessing code
df['days_since_accident'] = (df['contact_date'] - df['accident_date']).dt.days
df = df.sort_values(['accident_number', 'contact_date'])
df['days_since_last_consult'] = df.groupby('accident_number')['contact_date'].diff().dt.days
df['days_since_last_consult'] = df['days_since_last_consult'].fillna(0)
df['consult_seq'] = df.groupby('accident_number').cumcount() + 1

# Remove negative days (data errors)
df = df[df['days_since_accident'] >= 0].copy()
```

### Data Analysis Methods
The script implements several analysis functions:
- `explore_data()`: Provides summary statistics, checks for missing values
- `analyze_target()`: Examines class distribution in the target variable
- `analyze_temporal()`: Calculates average times between events
- `correlation_analysis()`: Computes correlations between features and target
- `compute_feature_importance()`: Uses a decision tree to estimate feature importance

### Logging Implementation
The script implements comprehensive logging using Python's `logging` module:
- Logs are written to both console and timestamped files
- Critical warnings (like negative days) are prominently logged
- Summary statistics are logged in formatted tables
- Feature importance results are logged in readable format

### Output Files
The script saves processed data in two formats:
1. Pickle format (.pkl) for efficient loading in subsequent pipeline stages
2. CSV format (.csv) for potential examination in spreadsheet software

## Exception Handling
The main function wraps the entire workflow in a try-except block to catch any uncaught exceptions:

```python
try:
    # Pipeline steps
except Exception as e:
    if 'logger' in locals():
        logger.error(f"Error in main execution: {e}", exc_info=True)
    else:
        print(f"Error before logger initialization: {e}")
    raise
```

This ensures that errors are properly logged with traceback information, even if they occur before logger initialization.

## Technical Dependencies
- Python 3.13
- pandas: For data manipulation
- numpy: For numerical operations
- scikit-learn: For feature importance calculation
- dotenv: For environment variable loading
- pathlib: For path manipulation
- logging: For logging capabilities

## Performance Considerations
- The script is designed to work efficiently with medium-sized datasets
- All pandas operations use proper indexing to avoid performance issues
- GroupBy operations are minimized and organized efficiently

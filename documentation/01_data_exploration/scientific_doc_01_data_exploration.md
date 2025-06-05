# Scientific Documentation: Data Exploration Module

## Overview
This documentation explains the scientific reasoning and analytical approaches used in the `01_data_exploration.py` script. The script implements data exploration techniques to understand patterns in patient consultation data and prepare it for predictive modeling of the NBE variable.

## Scientific Objectives
The exploration phase has several key scientific objectives:
1. Understand the structure and quality of the patient consultation dataset
2. Identify temporal patterns in patient consultations
3. Examine the distribution of the target variable (NBE)
4. Detect relationships between features and the target
5. Identify potential predictive features for model development

## Data Quality Assessment

### Negative Days Detection
The script identifies records where `days_since_accident` is negative, which is scientifically impossible (consultation cannot happen before the accident):

```python
negative_days = df[df['days_since_accident'] < 0]
if not negative_days.empty:
    logger.warning(f"{len(negative_days)} records with negative days_since_accident - these will be removed")
    # Remove negative days records
    df = df[df['days_since_accident'] >= 0].copy()
```

These negative values represent data entry errors rather than actual phenomena and are removed to maintain data integrity.

### Missing Value Analysis
The script analyzes missing values in all columns:

```python
missing = df.isnull().sum()
logger.info("\n=== Missing Values ===\n%s", missing)
```

Missing value patterns can indicate:
- Data collection issues
- Systematic biases in recording
- Potential challenges for model training

### Distributional Analysis
The script performs distributional analysis to understand:
- Shape of feature distributions
- Presence of outliers
- Potential normalization needs for modeling

## Target Variable Analysis

### Class Balance Assessment
The script examines the distribution of the target variable (NBE):

```python
nbe_counts = df['nbe'].value_counts()
if len(nbe_counts) > 1:
    min_class = nbe_counts.min()
    max_class = nbe_counts.max()
    ratio = min_class / max_class
    logger.info(f"Class balance ratio (min/max): {ratio:.4f}")
    if ratio < 0.2:
        logger.warning("Significant class imbalance detected (ratio < 0.2)")
```

Class imbalance is a critical factor that can:
- Bias model predictions toward the majority class
- Result in poor performance on minority classes
- Require specialized techniques (sampling, weighting) during model training

### Telephone Category Relationship
The script analyzes the relationship between telephone categories and NBE, which can reveal important patterns for prediction:

```python
logger.info("\n=== Telephone Categories ===\n%s", 
            df['telephone_category'].value_counts(normalize=True) * 100)
```

## Temporal Pattern Analysis
The script analyzes temporal patterns that may influence the NBE outcome:

```python
logger.info("Avg. days since accident: %.2f", df['days_since_accident'].mean())
logger.info("Avg. days between consults: %.2f", df['days_since_last_consult'].mean())
```

These temporal patterns are important because:
- Recovery typically follows temporal patterns
- Frequency of consultations may indicate severity
- Time since accident may influence recovery stage

## Correlation Analysis
The script examines linear relationships between features and the target:

```python
corr = df[['p_score', 'p_status', 'fl_score', 'fl_status', 
          'telephone_category', 'nbe']].corr()['nbe'].drop('nbe')
```

Correlation analysis:
- Identifies linear relationships between features and target
- Helps prioritize features for modeling
- May reveal redundant features that capture similar information

## Feature Importance Analysis
The script uses a decision tree model to estimate feature importance:

```python
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X, y)
importances = pd.Series(tree.feature_importances_, index=X.columns)
```

Decision trees are used because:
1. They can capture non-linear relationships
2. They provide native feature importance measures
3. They can handle mixed data types after encoding
4. They don't require feature scaling

A maximum depth of 5 is used to:
- Prevent overfitting to training data
- Focus on more general patterns
- Identify robustly important features

## Patient Consultation Sequence Analysis
The script analyzes consultation patterns by creating features like:
- `consult_seq`: The order of consultations for each patient
- `days_since_last_consult`: Time between consecutive consultations

These sequential patterns are important because:
- Recovery typically follows a progression
- Consultation frequency may indicate severity
- Changes in consultation patterns may predict outcomes

## Scientific Limitations
The exploratory analysis has several scientific limitations:
1. **Simple Modeling Approach**: Uses basic decision tree without cross-validation
2. **Limited Feature Engineering**: Minimal transformation of raw features
3. **No Outlier Handling**: Outliers are identified but not explicitly addressed
4. **Basic Missing Value Treatment**: Simplistic approach to missing values
5. **Single-Model Feature Importance**: Relies on a single model type for importance scoring

These limitations are acceptable for initial exploration but are addressed in subsequent stages of the machine learning pipeline.

## Scientific Conclusions
The exploration phase produces several key outputs:
1. Understanding of data quality and structure
2. Initial assessment of feature-target relationships
3. Identification of potentially predictive features
4. Preprocessed dataset for feature engineering
5. Documentation of data characteristics for model development

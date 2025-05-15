# NBE Prediction Model Development: Documentation

## Project Overview

This document details the model development phase (step 3) of the project aimed at predicting the Normal-Based Evaluation (NBE) status for patients based on consultation data. The NBE status indicates whether a patient's recovery is within expected norms (1), outside expected norms (0), or has insufficient information (2).

The project follows a structured machine learning pipeline approach, with previous phases covering data exploration and feature engineering. This phase encompasses model selection, training, evaluation, and final model selection for deployment.

## 1. Background and Previous Steps

### 1.1 Data Understanding (Step 1)

The initial data exploration phase provided the following key insights:

- **Dataset Structure**: 7,491 consultations for 2,379 unique patients, with an average of 3.15 consultations per patient
- **Target Distribution**: 55.88% within normal range (nbe=1), 27.95% insufficient information (nbe=2), 16.17% outside expected range (nbe=0)
- **Key Features**: 
  - Pain scores (p_score): Mean of 1.44 (scale 0-4)
  - Function limitation scores (fl_score): Mean of 1.97 (scale 0-4)
  - Pain status (p_status): 58% showing improvement
  - Function limitation status (fl_status): 57% showing improvement
- **Consultation Types**: First contacts (27.61%), follow-up consultations (51.54%), case closures (16.30%), not reached (3.54%), complex cases (1.01%)
- **Temporal Patterns**: Average of 119.21 days between accident and consultation, 34.73 days between consecutive consultations
- **Key Predictors**: Days since accident (37.21%), pain status (32.19%), function limitation score (12.95%), function limitation status (12.15%), pain score (3.98%)

### 1.2 Feature Engineering (Step 2)

The feature engineering phase enhanced the original dataset by:

- **Feature Creation**: 81 new features developed across 5 categories:
  - Time-based features (17): Recovery stages, consultation frequency/density, etc.
  - Score-based features (12): Combined scores, status indicators, normalized scores
  - Sequential features (18): Score trends, rate of change metrics, etc.
  - Categorical features (13): Encoded consultation types, recovery stages
  - Aggregate features (16): Patient-level metrics, historical NBE proportions
- **Feature Selection**: Top 50 features selected based on importance
- **Key Predictors**: Patient history (previous NBE values), temporal metrics, and improvement trends were the strongest predictors

## 2. Model Development Methodology

### 2.1 Pipeline Implementation

The model development phase was implemented using a modular pipeline consisting of:

1. **Data Preparation**
   - Split data ensuring patient-level separation (no patient appears in multiple sets)
   - Handle NaN values in target variable
   - Apply SMOTE for class imbalance
   - Scale features using StandardScaler

2. **Baseline Models**
   - Logistic Regression
   - Decision Tree

3. **Advanced Models**
   - Random Forest
   - Gradient Boosting
   - Models with SMOTE-resampled training data

4. **Model Evaluation**
   - Comprehensive metrics (accuracy, precision, recall, F1, ROC AUC, Brier score)
   - Visualization (ROC curves, PR curves, calibration plots)
   - Feature importance analysis

5. **Model Calibration**
   - Isotonic regression for probability calibration
   - Evaluation of calibration quality

6. **Final Model Selection**
   - Selection based on validation metrics
   - Final evaluation on held-out test set

### 2.2 Data Processing Details

During the data preparation phase, several key data quality issues were addressed:

- **Target NaN Values**: A significant number of NaN values were found in the target variable (1,528 in training, 301 in validation, 265 in test), likely corresponding to the "nbe=2" (no information) cases. These were appropriately removed.
- **Class Imbalance**: The binary target showed imbalance (76.8% class 1, 23.2% class 0), which was addressed using SMOTE to create a balanced training set.
- **Feature Scaling**: All features were standardized using StandardScaler for optimal model performance.

The final prepared sets contained:
- Training set: 3,742 consultations (after NaN removal)
- Validation set: 794 consultations (after NaN removal)
- Test set: 861 consultations (after NaN removal)

## 3. Model Training and Evaluation Results

### 3.1 Baseline Models Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Brier Score |
|-------|----------|-----------|--------|----------|---------|-------------|
| Logistic Regression | 95.47% | 98.05% | 96.18% | 97.11% | 98.62% | 0.0308 |
| Decision Tree | 96.47% | 97.32% | 98.25% | 97.78% | 99.29% | 0.0260 |

### 3.2 Advanced Models Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Brier Score |
|-------|----------|-----------|--------|----------|---------|-------------|
| Random Forest | 98.24% | 98.73% | 99.04% | 98.89% | 99.83% | 0.0192 |
| Gradient Boosting | 97.73% | 98.72% | 98.41% | 98.56% | 99.73% | 0.0175 |
| Random Forest with SMOTE | 96.73% | 98.86% | 96.97% | 97.91% | 99.44% | 0.0251 |
| Gradient Boosting with SMOTE | 97.23% | 99.51% | 96.97% | 98.23% | 99.65% | 0.0204 |

### 3.3 Calibration Results

For the best-performing model (Random Forest):
- **Before calibration**: Brier Score = 0.0192, Log Loss = 0.0808
- **After calibration**: Brier Score = 0.0175 (improved), Log Loss = 0.4270 (worsened)

Based on overall performance, the original (uncalibrated) Random Forest model was selected as the final model.

### 3.4 Final Model Performance (Test Set)

The Random Forest model achieved the following metrics on the held-out test set:

- **Accuracy**: 96.75%
- **Precision**: 97.26%
- **Recall**: 98.68%
- **F1 Score**: 97.97%
- **ROC AUC**: 99.53%
- **Brier Score**: 0.0263
- **Log Loss**: 0.0987

**Confusion Matrix**:
```
[[159  19]
 [  9 674]]
```

This indicates:
- True Negatives: 159 (correctly predicted outside normal range)
- False Positives: 19 (incorrectly predicted within normal range)
- False Negatives: 9 (incorrectly predicted outside normal range)
- True Positives: 674 (correctly predicted within normal range)

### 3.5 Feature Importance

The final Random Forest model identified the following top features:

1. **Patient history features**: Previous NBE values, proportion of previous consultations within normal range
2. **Temporal features**: Days since accident, consultation frequency patterns
3. **Recovery indicators**: Improvement trends in pain and function limitation
4. **Score features**: Combined and normalized pain and function limitation scores

## 4. Discussion

### 4.1 Model Performance Analysis

The model development process yielded several important insights:

1. **Excellent Overall Performance**: All models performed well, with the final Random Forest model achieving over 96.75% accuracy and 99.53% ROC AUC on the test set. This indicates strong predictive capability for determining whether a patient's recovery is within normal expected ranges.

2. **Class Imbalance Handling**: While SMOTE was applied to address class imbalance, models trained on the original imbalanced data (particularly Random Forest) actually performed better. This suggests that for this specific problem, the class imbalance did not significantly hinder model performance.

3. **Calibration Trade-offs**: The probability calibration process improved the Brier Score but worsened the Log Loss. This suggests a trade-off in probability calibration that should be considered based on the specific application requirements.

4. **Validation to Test Performance Gap**: There is a small performance drop from validation to test sets, which is normal and indicates good generalization with minimal overfitting.

### 4.2 Clinical Relevance

The model's ability to predict NBE status has significant clinical relevance:

1. **Early Identification**: The model can identify patients whose recovery is outside normal expected ranges, allowing for earlier intervention.

2. **Resource Allocation**: Healthcare providers can more efficiently allocate resources by focusing on patients at higher risk of abnormal recovery patterns.

3. **Recovery Trajectory**: The importance of temporal features and patient history highlights the significance of monitoring recovery trajectories over time.

4. **Pain and Function Relationship**: The model effectively utilizes both pain and function limitation metrics, confirming their joint importance in assessing recovery status.

### 4.3 Limitations

Despite the strong performance, several limitations should be acknowledged:

1. **Missing Information**: A substantial portion of the original data had "no information" (nbe=2) or NaN values, which were excluded from the analysis. Understanding these cases better might provide additional insights.

2. **Feature Engineering Assumptions**: The engineered features were based on domain knowledge and exploratory analysis, but alternative feature engineering approaches might yield different results.

3. **Temporal Dependencies**: The current model treats consultations somewhat independently, with patient history features. More sophisticated time-series approaches might capture temporal dependencies better.

4. **Imbalanced Classes**: While the models performed well despite class imbalance, the relatively low proportion of "outside normal range" cases (class 0) might limit the model's exposure to diverse abnormal recovery patterns.

## 5. Conclusion and Next Steps

### 5.1 Conclusion

The model development phase successfully produced a high-performing Random Forest model capable of predicting whether a patient's recovery is within normal expected ranges with high accuracy and reliability. The model effectively leverages patient history, temporal patterns, and clinical assessment scores to make these predictions.

The selected model has been saved in two formats:
- A detailed version (`final_model.pkl`) with additional information for analysis
- A production-ready version (`final_model_prod.joblib`) for deployment

### 5.2 Recommendations for Next Steps

Based on the project outline and current results, the following next steps are recommended:

1. **Model Deployment**
   - Develop a production pipeline for real-time prediction
   - Create an API endpoint for integration with clinical systems
   - Implement monitoring for model performance in production

2. **Explainability Enhancement**
   - Develop patient-level interpretations of model predictions
   - Create visual explanations for clinicians
   - Generate case studies for typical patient trajectories

3. **Validation Studies**
   - Conduct prospective validation with new patient data
   - Perform sensitivity analysis for different patient subgroups
   - Validate model performance across different injury types and severity levels

4. **User Interface Development**
   - Design a clinician-facing dashboard for prediction visualization
   - Implement risk stratification visualization
   - Create patient timeline views with predicted NBE probabilities

5. **Feedback Loop Implementation**
   - Establish mechanisms for clinicians to provide feedback on predictions
   - Develop a retraining schedule and criteria
   - Monitor concept drift and model degradation over time

By following these next steps, the project can move from successful model development to impactful clinical implementation, ultimately improving patient care by enabling more personalized and timely interventions based on predicted recovery trajectories.

## 6. Technical Appendix

### 6.1 Environment and Dependencies

- Python version: 3.13
- Key libraries: scikit-learn, pandas, numpy, matplotlib, seaborn, imbalanced-learn (SMOTE), joblib

### 6.2 File Locations

- **Logs**: `C:\Users\Nick\PycharmProjects\reha_assist_iru\logs\03_model_development\20250515_135640`
- **Plots**: `C:\Users\Nick\PycharmProjects\reha_assist_iru\plots\03_model_development\20250515_135640`
- **Models**: `C:\Users\Nick\PycharmProjects\reha_assist_iru\models\03_model_development\20250515_135640`

### 6.3 Model Artifacts

- **Prepared datasets**: `prepared_datasets.pkl`
- **Baseline models**: `baseline_models.pkl`
- **Advanced models**: `advanced_models.pkl`
- **All models**: `all_models.pkl`
- **Calibrated model**: `calibrated_model.pkl`
- **Final model**: `final_model.pkl`
- **Production model**: `final_model_prod.joblib`

### 6.4 Pipeline Implementation

The model development pipeline was implemented as a modular Python package with the following structure:

```
03_model_development/
├── utils/
│   ├── project_setup.py       # Project structure, logging setup
│   └── data_loader.py         # Load engineered dataset
├── 01_data_preparation.py     # Data splitting, handling class imbalance
├── 02_baseline_models.py      # Simple models (Logistic Regression, Decision Tree)
├── 03_advanced_models.py      # Complex models (Random Forest, Gradient Boosting)
├── 04_model_evaluation.py     # Evaluation metrics, visualization
├── 05_model_calibration.py    # Probability calibration techniques
├── 06_model_selection.py      # Compare models and select best one
└── main.py                    # Orchestrates the entire pipeline
```

This modular design allows for both end-to-end pipeline execution and individual component testing and optimization.

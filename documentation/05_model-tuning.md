# Model Training and Tuning: Documentation

## Project Overview

This document details the model training and tuning phase (step 4) of the project aimed at predicting the Normal-Based Evaluation (NBE) status for patients based on consultation data. Building upon the success of the model development phase (step 3), this phase focused on refining the best-performing Random Forest model through hyperparameter optimization and robust cross-validation.

The NBE status indicates whether a patient's recovery is within expected norms (1), outside expected norms (0), or has insufficient information (2). The goal is to build a model that accurately predicts the probability of NBE=1 versus NBE=0 for each patient consultation.

## 1. Background and Previous Steps

### 1.1 Prior Phases

The project has followed a structured machine learning pipeline approach:

1. **Data Understanding (Step 1)**: Initial exploration of 7,491 consultations for 2,379 unique patients, identifying key predictors including days since accident, pain status, and function limitation metrics.

2. **Feature Engineering (Step 2)**: Creation of 81 new features across temporal, score-based, sequential, categorical, and aggregate categories, with selection of the top 50 features.

3. **Model Development (Step 3)**: Training of baseline and advanced models, with Random Forest emerging as the best-performing model with 96.75% accuracy and 99.53% ROC AUC on the test set.

### 1.2 Previous Model Performance (Step 3)

The Random Forest model selected in Step 3 demonstrated strong performance:

- **Accuracy**: 96.75%
- **Precision**: 97.26%
- **Recall**: 98.68%
- **F1 Score**: 97.97%
- **ROC AUC**: 99.53%
- **Brier Score**: 0.0263
- **Log Loss**: 0.0987

This model used default hyperparameters (100 trees, max depth of 10) and was evaluated on a single train/validation/test split.

## 2. Methodology

### 2.1 Pipeline Implementation

The model training and tuning phase focused on three key components:

1. **Hyperparameter Optimization**
   - Bayesian optimization using Optuna
   - Exploration of key Random Forest parameters
   - Patient-level cross-validation for hyperparameter selection

2. **Robust Cross-Validation**
   - 5-fold patient-level cross-validation
   - Ensuring no patient appears in both training and validation sets
   - Comprehensive performance metrics across folds

3. **Model Refinement and Comparison**
   - Comparison between original and optimized models
   - Final selection based on performance improvements
   - Production-ready model preparation

### 2.2 Hyperparameter Optimization

The hyperparameter space was defined to explore key Random Forest parameters:

- **n_estimators**: Number of trees (50 to 500)
- **max_depth**: Maximum tree depth (5 to 30)
- **min_samples_split**: Minimum samples required to split a node (2 to 20)
- **min_samples_leaf**: Minimum samples required at a leaf node (1 to 10)
- **max_features**: Feature subset strategy ('sqrt', 'log2', None)
- **bootstrap**: Whether to use bootstrap samples (True, False)

The optimization process:
1. Used Bayesian optimization with Tree-structured Parzen Estimator (TPE)
2. Ran 100 trials with a timeout of 600 seconds
3. Used ROC AUC on patient-level cross-validation as the objective metric

### 2.3 Patient-Level Cross-Validation

To ensure robust performance estimation:

1. Created folds at the patient level (not observation level)
2. Implemented 5-fold cross-validation
3. Ensured the same patient's consultations never appeared in both training and validation sets
4. Calculated detailed metrics for each fold
5. Aggregated results with mean and standard deviation

### 2.4 Model Comparison Framework

The comparison between the original model from Step 3 and the newly optimized model:

1. Evaluated both models on the same test set
2. Compared performance across all metrics
3. Used ROC AUC as the primary selection criterion
4. Considered other metrics for a comprehensive comparison

## 3. Results

### 3.1 Hyperparameter Optimization Results

The Bayesian optimization process identified the following optimal hyperparameters:

| Parameter | Optimal Value | Default Value |
|-----------|---------------|---------------|
| n_estimators | 153 | 100 |
| max_depth | 26 | 10 |
| min_samples_split | 11 | 2 |
| min_samples_leaf | 2 | 1 |
| max_features | None | 'sqrt' |
| bootstrap | True | True |

The optimization achieved a best cross-validation ROC AUC score of 0.9972.

### 3.2 Cross-Validation Performance

The 5-fold patient-level cross-validation of the optimized model demonstrated excellent and consistent performance:

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.9770 | 0.0016 | 0.9759 | 0.9799 |
| Precision | 0.9867 | 0.0018 | - | - |
| Recall | 0.9833 | 0.0014 | - | - |
| F1 Score | 0.9850 | 0.0010 | - | - |
| ROC AUC | 0.9972 | 0.0001 | 0.9971 | 0.9974 |
| Brier Score | 0.0188 | 0.0002 | - | - |
| Log Loss | 0.0637 | 0.0014 | - | - |

The remarkably low standard deviations across all metrics indicate a highly stable model performance regardless of which patients are in the training or validation sets.

### 3.3 Model Comparison

The comparison between the original model (from Step 3) and the optimized model shows improvements across all performance metrics:

| Metric | Original Model | Optimized Model | Improvement (%) |
|--------|-----------------|-----------------|-----------------|
| Accuracy | 0.9675 | 0.9768 | +0.93% |
| Precision | 0.9726 | 0.9825 | +0.99% |
| Recall | 0.9868 | 0.9883 | +0.15% |
| F1 Score | 0.9797 | 0.9854 | +0.57% |
| ROC AUC | 0.9953 | 0.9967 | +0.14% |
| Brier Score | 0.0263 | 0.0208 | -20.91% |
| Log Loss | 0.0987 | 0.0701 | -29.08% |

The most significant improvements were in Brier Score (-20.91%) and Log Loss (-29.08%), which are critical metrics for probability calibration.

## 4. Analysis and Discussion

### 4.1 Hyperparameter Impact Analysis

The optimized hyperparameters revealed several important insights:

1. **Tree Depth**: The substantially higher max_depth (26 vs. 10) indicates that deeper trees capture more complex patterns in the data without overfitting, suggesting the relationships between features and the NBE outcome are more intricate than initially modeled.

2. **Number of Trees**: The modest increase in n_estimators (153 vs. 100) suggests that while more trees help, the ensemble quickly reaches diminishing returns, confirming the model's stability.

3. **Node Splitting Criteria**: The higher min_samples_split (11 vs. 2) means the model is more conservative in creating splits, helping to avoid overfitting to noise in the training data.

4. **Feature Subset Strategy**: Using all features (max_features=None) rather than a subset ('sqrt') indicates that considering all features at each split is beneficial, suggesting the engineered features are highly relevant and not redundant.

### 4.2 Cross-Validation Insights

The cross-validation results provide several key insights:

1. **Consistency Across Folds**: The extremely low standard deviations (e.g., ROC AUC: 0.9972 ± 0.0001) demonstrate that the model performs consistently regardless of which patients are in the training or validation sets, indicating strong generalization capabilities.

2. **Robustness to Patient Variation**: Patient-level cross-validation ensures that the model isn't memorizing individual patient patterns but is learning generalizable patterns across different patients.

3. **Performance Stability**: The narrow range between minimum and maximum performance across folds suggests the model is not sensitive to specific subsets of the data, further confirming its robustness.

### 4.3 Probability Calibration Improvements

The most significant improvements were in metrics related to probability calibration:

1. **Brier Score Improvement**: The reduction from 0.0263 to 0.0208 (20.91% improvement) indicates that the optimized model's probability estimates are better calibrated, making them more reliable for clinical decision-making.

2. **Log Loss Improvement**: The substantial reduction from 0.0987 to 0.0701 (29.08% improvement) further confirms the improved reliability of the predicted probabilities.

These improvements are particularly important for the project's goal of predicting the probability of a patient being within normal expected ranges, as they directly impact the trustworthiness of these probability estimates.

### 4.4 Clinical Implications

The refined model offers several advantages for clinical applications:

1. **More Reliable Risk Stratification**: Better calibrated probabilities allow for more precise patient risk stratification.

2. **Improved Decision Support**: Clinicians can have greater confidence in the model's predictions when deciding on intervention strategies.

3. **Balance of Sensitivity and Specificity**: The model achieves high recall (98.83%) while maintaining excellent precision (98.25%), minimizing both false negatives (missed opportunities for intervention) and false positives (unnecessary interventions).

4. **Consistent Performance**: The demonstrated stability across different patient subsets suggests the model will perform reliably across the patient population.

## 5. Implementation Details

### 5.1 Technical Implementation

The model training and tuning was implemented as a modular Python pipeline:

1. **Hyperparameter Optimization Module** (`hyperparameter_tuning.py`)
   - Objective function for patient-level cross-validation
   - Bayesian optimization with Optuna
   - Visualization of hyperparameter importance

2. **Cross-Validation Module** (`cross_validation.py`)
   - Patient-level fold creation
   - Comprehensive metric calculation
   - Visualization of cross-validation results

3. **Model Refinement Module** (`model_refinement.py`)
   - Comparison framework
   - Model selection logic
   - Production model preparation

4. **Main Orchestration Script** (`main_step4.py`)
   - Pipeline coordination
   - Command-line interface
   - Results aggregation

### 5.2 Computational Resources

The hyperparameter optimization process:
- Executed 100 trials over approximately 100 seconds
- Required minimal computational resources
- Could be further extended with more trials or larger hyperparameter space if needed

### 5.3 Artifacts and Outputs

The pipeline produced several artifacts:

1. **Optimized Model Files**:
   - `refined_model.pkl`: Complete model with metadata
   - `refined_model_prod.joblib`: Production-ready model

2. **Analysis Artifacts**:
   - Hyperparameter trials data
   - Cross-validation results
   - Model comparison metrics

3. **Visualizations**:
   - Cross-validation metrics by fold
   - Model performance comparison
   - Hyperparameter importance

## 6. Conclusion and Next Steps

### 6.1 Conclusions

The model training and tuning phase has successfully refined the already strong Random Forest model from the previous step, resulting in:

1. **Performance Improvements**: Enhancements across all metrics, with significant improvements in probability calibration (Brier Score and Log Loss).

2. **Robust Validation**: Patient-level cross-validation confirms the model's consistency and generalizability.

3. **Optimized Architecture**: The refined hyperparameters better capture the complex patterns in the patient consultation data.

4. **Clinical Utility**: The improved probability calibration enhances the model's value for clinical decision-making.

The systematic approach to hyperparameter optimization and cross-validation has yielded a model that not only performs better but also provides more reliable probability estimates, which is crucial for the project's goal of predicting NBE status.

### 6.2 Recommendations for Next Steps

Building on this successful model tuning phase, the following next steps are recommended:

1. **Model Interpretation** (Step 5)
   - Implement SHAP (SHapley Additive exPlanations) values for detailed feature contribution analysis
   - Develop patient-level interpretations of predictions
   - Create visual explanations for clinical users

2. **Further Probability Calibration**
   - Consider additional calibration techniques beyond the inherent improvements from hyperparameter tuning
   - Evaluate calibration performance on specific patient subgroups

3. **Deployment Preparation**
   - Optimize the model for inference speed if needed
   - Create deployment wrappers with pre/post-processing
   - Develop monitoring systems for model performance

4. **Validation Studies**
   - Plan prospective validation with new patient data
   - Evaluate performance across different patient demographics or injury types
   - Compare model predictions with expert clinical assessments

### 6.3 Final Assessment

The model training and tuning phase has delivered a refined Random Forest model that significantly improves upon the previous version. The optimized model not only achieves higher accuracy and discrimination ability but also provides better calibrated probability estimates, making it more valuable for clinical decision support.

The use of patient-level cross-validation provides strong confidence in the model's ability to generalize to new patients, while the detailed hyperparameter optimization has created a model architecture that better captures the complex relationships in the patient consultation data.

The refined model is ready for the next phases of interpretation, deployment preparation, and clinical validation, moving closer to the ultimate goal of providing reliable predictions of whether patients are within normal expected recovery ranges.

## 7. Technical Appendix

### 7.1 Environment and Dependencies

- **Python Version**: 3.13
- **Key Libraries**:
  - scikit-learn (model implementation)
  - optuna (hyperparameter optimization)
  - pandas, numpy (data handling)
  - matplotlib, seaborn (visualization)
  - joblib (model serialization)

### 7.2 File Locations

- **Logs**: `C:\Users\Nick\PycharmProjects\reha_assist_iru\logs\04_model_tuning\20250515_142639`
- **Plots**: `C:\Users\Nick\PycharmProjects\reha_assist_iru\plots\04_model_tuning\20250515_142639`
- **Models**: `C:\Users\Nick\PycharmProjects\reha_assist_iru\models\04_model_tuning\20250515_142639\04_model_tuning`

### 7.3 Hyperparameter Optimization Details

The Bayesian optimization with Optuna used the following settings:

- **Sampler**: Tree-structured Parzen Estimator (TPE)
- **Pruner**: Median pruner for early stopping ineffective trials
- **Direction**: Maximize (for ROC AUC)
- **Trials**: 100
- **Timeout**: 600 seconds

Full hyperparameter search space:

```python
# Hyperparameter space
n_estimators = trial.suggest_int('n_estimators', 50, 500)
max_depth = trial.suggest_int('max_depth', 5, 30)
min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
bootstrap = trial.suggest_categorical('bootstrap', [True, False])
```

### 7.4 Cross-Validation Implementation

The patient-level cross-validation was implemented by:

1. Identifying unique patient IDs
2. Randomly assigning patients to 5 folds
3. Ensuring all consultations for the same patient stay in the same fold
4. Training on 4 folds and validating on the remaining fold
5. Repeating for all 5 folds

```python
def create_patient_folds(X, patient_ids, n_splits=5, random_state=42):
    # Get unique patient IDs
    unique_patients = np.unique(patient_ids)
    
    # Shuffle patients
    np.random.seed(random_state)
    np.random.shuffle(unique_patients)
    
    # Split patients into folds
    patient_folds = np.array_split(unique_patients, n_splits)
    
    # Create folds at the observation level
    folds = []
    for i in range(n_splits):
        val_patients = patient_folds[i]
        val_indices = np.where(np.isin(patient_ids, val_patients))[0]
        train_indices = np.where(~np.isin(patient_ids, val_patients))[0]
        folds.append((train_indices, val_indices))
    
    return folds
```

### 7.5 Performance Metrics Details

All evaluation metrics were calculated using scikit-learn implementations:

- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **Brier Score**: Mean squared difference between predicted probabilities and actual outcomes
- **Log Loss**: Negative log-likelihood of the true labels given the predicted probabilities

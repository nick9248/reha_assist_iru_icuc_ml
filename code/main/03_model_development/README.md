# Model Development Pipeline

## Overview
This module implements step 3 of the machine learning project to predict NBE values from patient consultation data. It follows the feature engineering phase and focuses on model selection, training, evaluation, and final model selection.

## Pipeline Structure
The model development pipeline is organized into the following modular components:

1. **Data Preparation** (`01_data_preparation.py`)
   - Splits data with patient-level separation (train/validation/test)
   - Handles class imbalance using SMOTE
   - Scales features for model training

2. **Baseline Models** (`02_baseline_models.py`)
   - Implements simple models (Logistic Regression, Decision Trees)
   - Evaluates performance on validation set

3. **Advanced Models** (`03_advanced_models.py`)
   - Implements ensemble methods (Random Forest, Gradient Boosting)
   - Trains models with and without SMOTE 
   - Evaluates performance on validation set

4. **Model Evaluation** (`04_model_evaluation.py`)
   - Creates comprehensive visualizations (ROC curves, PR curves, etc.)
   - Compares all models on key metrics
   - Creates feature importance visualizations

5. **Model Calibration** (`05_model_calibration.py`)
   - Calibrates probability estimates for the best model
   - Improves reliability of probability predictions

6. **Model Selection** (`06_model_selection.py`)
   - Selects the final model based on performance metrics
   - Evaluates final model on held-out test set
   - Creates final model visualizations

## Utility Modules
- `utils/project_setup.py`: Project structure and logging setup
- `utils/data_loader.py`: Load engineered dataset from previous phase

## Running the Pipeline
You can run the complete pipeline using the main script:

```bash
python main.py
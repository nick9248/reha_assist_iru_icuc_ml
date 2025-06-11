# Machine Learning Pipeline for NBE Prediction - Revised Outline

## Project Overview
**Goal**: Train a binary classifier to predict NBE probability from consultation data and deploy as API for enaio DMS integration.

**Input Features**: p_score, p_status, fl_score, fl_status  
**Output**: Probabilities for nbe_yes and nbe_no

## 1. Data Pipeline Module (`data_pipeline/`)

### 1.1 Data Loading & Validation
```
data_loader.py
├── load_excel_data()
├── validate_data_schema()
└── log_data_quality_metrics()
```

### 1.2 Data Preprocessing
```
preprocessor.py
├── anonymize_patients()          # Generate unique patient IDs
├── handle_missing_values()       # Simple imputation strategies
├── encode_categorical_features() # Label encoding for status variables
└── create_train_test_split()     # Patient-level stratified split
```

### 1.3 Feature Engineering (Minimal)
```
feature_engineer.py
├── create_interaction_features() # p_score * fl_score combinations
├── derive_severity_scores()      # Combined pain/function severity
└── validate_feature_ranges()     # Ensure 0-4 ranges, 0-2 status ranges
```

## 2. Model Training Module (`models/`)

### 2.1 Model Selection Strategy
**Primary Models** (keep simple):
- Logistic Regression (baseline - interpretable)
- Random Forest (handles interactions well)
- XGBoost (production-ready, robust)

### 2.2 Training Pipeline
```
trainer.py
├── train_baseline_models()
├── hyperparameter_tuning()      # GridSearchCV with 5-fold CV
├── evaluate_model_performance()  # Focus on AUC, precision, recall
└── save_best_model()            # Pickle + metadata
```

### 2.3 Model Evaluation
```
evaluator.py
├── calculate_classification_metrics()
├── generate_probability_calibration() # Platt scaling if needed
├── create_confusion_matrix()
└── feature_importance_analysis()
```

## 3. API Development Module (`api/`)

### 3.1 FastAPI Implementation
```
main.py
├── health_check_endpoint()
├── predict_nbe_endpoint()       # POST /predict
└── model_info_endpoint()        # GET /model/info
```

### 3.2 Input/Output Schema
```
schemas.py
├── PredictionRequest
│   ├── p_score: int (0-4)
│   ├── p_status: int (0-2)
│   ├── fl_score: int (0-4)
│   └── fl_status: int (0-2)
└── PredictionResponse
    ├── nbe_yes_probability: float
    ├── nbe_no_probability: float
    └── prediction_timestamp: datetime
```

### 3.3 Model Service
```
model_service.py
├── load_trained_model()
├── validate_input_data()
├── make_prediction()
└── format_response()
```

## 4. Configuration & Deployment (`config/`)

### 4.1 Configuration Management
```
config.py
├── model_parameters
├── api_settings
├── data_paths
└── logging_config
```

### 4.2 Docker Deployment
```
Dockerfile
├── Python 3.13 base image
├── Install dependencies
├── Copy model artifacts
└── Expose API port
```

## 5. Simplified Project Structure

```
nbe_prediction/
├── data_pipeline/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   └── feature_engineer.py
├── models/
│   ├── __init__.py
│   ├── trainer.py
│   └── evaluator.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── model_service.py
├── config/
│   ├── __init__.py
│   └── config.py
├── tests/
├── notebooks/          # For EDA only
├── artifacts/          # Trained models
├── requirements.txt
├── Dockerfile
└── main_pipeline.py    # Orchestration script
```

## 6. Implementation Priority

### Phase 1: Core Functionality
1. Data loading and basic preprocessing
2. Simple model training (Logistic Regression)
3. Basic API with prediction endpoint

### Phase 2: Production Ready
1. Model comparison and selection
2. Probability calibration
3. Comprehensive API with validation

### Phase 3: Integration & Monitoring
1. enaio DMS integration testing
2. Model performance monitoring
3. Automated retraining pipeline

## Key Design Principles

**Modularity**: Each component has single responsibility  
**Simplicity**: Start with proven algorithms, avoid over-engineering  
**Scalability**: Clean interfaces, easy to extend  
**Production-Ready**: Proper logging, error handling, validation  

## Success Metrics

- Model AUC > 0.75
- API response time < 200ms
- Input validation coverage 100%
- Successful enaio integration

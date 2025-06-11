# NBE Prediction Project - Final Structure

```
nbe_prediction_project/
├── .env                           # Environment variables
├── data/
│   ├── raw/
│   │   └── icuc_ml_dataset.xlsx
│   ├── processed/
│   │   ├── step1_data_exploration_YYYYMMDD_HHMMSS.csv
│   │   ├── step2_cleaned_data_YYYYMMDD_HHMMSS.csv
│   │   └── step3_feature_engineered_YYYYMMDD_HHMMSS.csv
│   └── anonymized/
│       └── anonymized_dataset_YYYYMMDD_HHMMSS.csv
│
├── code/
│   ├── step1_data_exploration/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_explorer.py
│   │   └── data_validator.py
│   ├── step2_data_preprocessing/
│   │   ├── __init__.py
│   │   ├── data_cleaner.py
│   │   ├── anonymizer.py
│   │   └── preprocessor.py
│   ├── step3_feature_engineering/
│   │   ├── __init__.py
│   │   ├── feature_creator.py
│   │   └── feature_validator.py
│   ├── step4_model_training/
│   │   ├── __init__.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluator.py
│   │   └── hyperparameter_tuner.py
│   ├── step5_model_selection/
│   │   ├── __init__.py
│   │   ├── model_comparator.py
│   │   └── final_model_selector.py
│   ├── step6_api_development/
│   │   ├── __init__.py
│   │   ├── api_main.py
│   │   ├── api_schemas.py
│   │   ├── model_service.py
│   │   └── api_validator.py
│   └── step7_deployment/
│       ├── __init__.py
│       ├── deployment_validator.py
│       └── integration_tester.py
│
├── logs/
│   ├── step1/
│   │   ├── data_loader_20250611_143000.log
│   │   ├── data_explorer_20250611_143500.log
│   │   └── data_validator_20250611_144000.log
│   ├── step2/
│   │   ├── data_cleaner_20250611_150000.log
│   │   ├── anonymizer_20250611_150500.log
│   │   └── preprocessor_20250611_151000.log
│   ├── step3/
│   │   ├── feature_creator_20250611_153000.log
│   │   └── feature_validator_20250611_153500.log
│   ├── step4/
│   │   ├── model_trainer_20250611_160000.log
│   │   ├── model_evaluator_20250611_162000.log
│   │   └── hyperparameter_tuner_20250611_164000.log
│   ├── step5/
│   │   ├── model_comparator_20250611_170000.log
│   │   └── final_model_selector_20250611_171000.log
│   ├── step6/
│   │   ├── api_main_20250611_173000.log
│   │   ├── model_service_20250611_173500.log
│   │   └── api_validator_20250611_174000.log
│   └── step7/
│       ├── deployment_validator_20250611_180000.log
│       └── integration_tester_20250611_180500.log
│
├── plots/
│   ├── step1_data_exploration/
│   │   ├── data_distribution_20250611_143000.png
│   │   ├── correlation_matrix_20250611_143000.png
│   │   ├── missing_values_heatmap_20250611_143000.png
│   │   └── target_variable_distribution_20250611_143000.png
│   ├── step3_feature_engineering/
│   │   ├── feature_importance_20250611_153000.png
│   │   └── feature_correlations_20250611_153000.png
│   ├── step4_model_training/
│   │   ├── model_performance_comparison_20250611_160000.png
│   │   ├── confusion_matrix_20250611_160000.png
│   │   ├── roc_curves_20250611_160000.png
│   │   └── probability_calibration_20250611_160000.png
│   └── step5_model_selection/
│       ├── final_model_performance_20250611_170000.png
│       └── feature_importance_final_20250611_170000.png
│
├── models/
│   ├── artifacts/
│   │   ├── step4_logistic_regression_20250611_160000.pkl
│   │   ├── step4_random_forest_20250611_161000.pkl
│   │   ├── step4_xgboost_20250611_162000.pkl
│   │   └── step5_final_model_20250611_170000.pkl
│   └── metadata/
│       ├── model_performance_metrics_20250611_170000.json
│       └── model_training_config_20250611_170000.json
│
├── config/
│   ├── __init__.py
│   ├── logging_config.py
│   ├── model_config.py
│   └── api_config.py
│
├── utils/
│   ├── __init__.py
│   ├── logger_utils.py
│   ├── datetime_utils.py
│   └── file_utils.py
│
├── tests/
│   ├── test_step1/
│   ├── test_step2/
│   ├── test_step3/
│   ├── test_step4/
│   ├── test_step5/
│   ├── test_step6/
│   └── test_step7/
│
├── notebooks/
│   ├── step1_exploration_notebook_20250611.ipynb
│   ├── step4_model_analysis_notebook_20250611.ipynb
│   └── step5_final_results_notebook_20250611.ipynb
│
├── notebooks/
│   ├── step1_exploration_notebook_20250611.ipynb
│   ├── step4_model_analysis_notebook_20250611.ipynb
│   └── step5_final_results_notebook_20250611.ipynb
│
├── requirements.txt
├── .env.example                   # Example environment variables
├── Dockerfile
├── docker-compose.yml
├── README.md
└── main_orchestrator.py
```

## Environment Variables (.env file)

Create a `.env` file in the project root with:

```bash
# Project Paths
PROJECT_ROOT=/path/to/nbe_prediction_project
DATA_PATH=${PROJECT_ROOT}/data
LOGS_PATH=${PROJECT_ROOT}/logs
PLOTS_PATH=${PROJECT_ROOT}/plots
MODELS_PATH=${PROJECT_ROOT}/models

# Data Files
RAW_DATA_FILE=icuc_ml_dataset.xlsx

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Model Configuration
RANDOM_STATE=42
TEST_SIZE=0.2
CV_FOLDS=5

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_TITLE=NBE Prediction API
API_VERSION=1.0.0
```

## Path Management Strategy

**Using pathlib + python-dotenv:**
- Load base paths from `.env` file
- Use `pathlib.Path` for all path operations
- Create dynamic paths with timestamps using utility functions

**Example usage in code:**
```python
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT'))
DATA_PATH = PROJECT_ROOT / 'data'
LOGS_PATH = PROJECT_ROOT / 'logs'
```

### Step 1: Data Exploration & Understanding
- Load and examine the icuc_ml_dataset.xlsx
- Generate comprehensive data quality report
- Create visualizations for data distribution
- Validate data schema and ranges

### Step 2: Data Preprocessing & Anonymization  
- Clean missing values and outliers
- Anonymize patient identifiers
- Prepare data for modeling

### Step 3: Feature Engineering
- Create interaction features
- Derive severity scores
- Validate feature quality

### Step 4: Model Training & Evaluation
- Train baseline models (Logistic Regression, Random Forest, XGBoost)
- Evaluate model performance
- Generate performance visualizations

### Step 5: Model Selection & Finalization
- Compare model performances
- Select best model
- Finalize model artifacts

### Step 6: API Development
- Build FastAPI endpoints
- Create input/output schemas
- Implement model service

### Step 7: Deployment & Integration Testing
- Create Docker container
- Test enaio integration readiness
- Validate production deployment

## Naming Conventions

**Files**: `{module_name}_{timestamp}.{extension}`  
**Logs**: `{script_name}_{YYYYMMDD_HHMMSS}.log`  
**Plots**: `{plot_description}_{YYYYMMDD_HHMMSS}.png`  
**Models**: `{step_name}_{model_type}_{YYYYMMDD_HHMMSS}.pkl`

## Logging Strategy

Each step will have:
- **INFO**: Progress updates and key metrics
- **DEBUG**: Detailed execution information  
- **WARNING**: Data quality issues or performance concerns
- **ERROR**: Execution failures with stack traces
- **CRITICAL**: System failures requiring immediate attention

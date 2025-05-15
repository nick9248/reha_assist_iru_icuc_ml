# Model Interpretation and Insights (Step 5)

This module implements Step 5 of the NBE prediction project, focusing on model interpretation and probability calibration. The goal is to make the model more transparent, interpretable, and reliable for clinical decision-making.

## Overview

The module consists of two main components:

1. **Feature Importance Analysis**: Uses SHAP (SHapley Additive exPlanations) values to provide detailed insights into how each feature contributes to the model's predictions.

2. **Probability Calibration**: Evaluates and improves the reliability of the model's probability estimates, ensuring they represent true probabilities.

## Key Files

- `feature_importance_analysis.py`: Implements SHAP-based feature importance analysis
- `probability_calibration.py`: Implements probability calibration analysis and improvement
- `main_step5.py`: Orchestrates the entire model interpretation process
- `utils/project_setup.py`: Utility functions for project setup and logging
- `utils/data_loader.py`: Utility functions for loading data and models

## Usage

### Running the Complete Pipeline

```bash
python main_step5.py
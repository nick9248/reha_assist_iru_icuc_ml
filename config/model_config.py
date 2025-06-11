"""
Model Configuration for NBE Prediction Project
Centralized configuration for model training and evaluation
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class ModelConfig:
    """Configuration class for model training parameters"""

    # Random state for reproducibility
    RANDOM_STATE: int = 42

    # Cross-validation settings
    CV_FOLDS: int = 5
    CV_SHUFFLE: bool = True

    # Model evaluation metrics
    PRIMARY_METRIC: str = 'auc_roc'
    EVALUATION_METRICS: List[str] = None

    # Performance thresholds
    MIN_ACCEPTABLE_AUC: float = 0.70
    SIGNIFICANT_IMPROVEMENT_THRESHOLD: float = 0.02

    # Feature sets
    BASELINE_FEATURES: List[str] = None
    ENHANCED_FEATURES: List[str] = None

    # Model hyperparameters
    LOGISTIC_REGRESSION_PARAMS: Dict[str, Any] = None
    RANDOM_FOREST_PARAMS: Dict[str, Any] = None
    XGBOOST_PARAMS: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values after object creation"""

        if self.EVALUATION_METRICS is None:
            self.EVALUATION_METRICS = [
                'accuracy', 'precision', 'recall', 'f1_score',
                'auc_roc', 'auc_pr', 'specificity'
            ]

        if self.BASELINE_FEATURES is None:
            self.BASELINE_FEATURES = [
                'p_score', 'p_status', 'fl_score', 'fl_status'
            ]

        if self.ENHANCED_FEATURES is None:
            self.ENHANCED_FEATURES = [
                'p_score', 'p_status', 'fl_score', 'fl_status',
                'days_since_accident', 'consultation_number'
            ]

        if self.LOGISTIC_REGRESSION_PARAMS is None:
            self.LOGISTIC_REGRESSION_PARAMS = {
                'random_state': self.RANDOM_STATE,
                'max_iter': 1000,
                'solver': 'liblinear',
                'class_weight': 'balanced'  # Handle class imbalance
            }

        if self.RANDOM_FOREST_PARAMS is None:
            self.RANDOM_FOREST_PARAMS = {
                'random_state': self.RANDOM_STATE,
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced'
            }

        if self.XGBOOST_PARAMS is None:
            self.XGBOOST_PARAMS = {
                'random_state': self.RANDOM_STATE,
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'eval_metric': 'logloss',
                'scale_pos_weight': 1  # Will be calculated based on class balance
            }

    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get parameters for specific model type

        Args:
            model_type: 'logistic_regression', 'random_forest', or 'xgboost'

        Returns:
            Dict containing model parameters
        """
        param_map = {
            'logistic_regression': self.LOGISTIC_REGRESSION_PARAMS,
            'random_forest': self.RANDOM_FOREST_PARAMS,
            'xgboost': self.XGBOOST_PARAMS
        }

        return param_map.get(model_type, {})

    def update_xgboost_class_weight(self, y_train: np.ndarray):
        """
        Update XGBoost scale_pos_weight based on class distribution

        Args:
            y_train: Training target variable
        """
        negative_count = np.sum(y_train == 0)
        positive_count = np.sum(y_train == 1)

        if positive_count > 0:
            scale_pos_weight = negative_count / positive_count
            self.XGBOOST_PARAMS['scale_pos_weight'] = scale_pos_weight


# Default configuration instance
DEFAULT_CONFIG = ModelConfig()
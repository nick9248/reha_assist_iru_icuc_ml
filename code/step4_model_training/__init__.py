"""
Step 4: Model Training & Evaluation Module
Handles training and evaluation of NBE prediction models
"""

from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator

__all__ = ['ModelTrainer', 'ModelEvaluator']
__version__ = '1.0.0'
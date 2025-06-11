"""
Step 2 Data Preprocessing & Anonymization Module
NBE Prediction Project

This module handles:
1. Data cleaning and binary classification preparation
2. Patient identifier anonymization
3. Dual feature engineering (baseline + enhanced models)
4. Patient-level train/test splitting

Components:
- DataCleaner: Data quality and NBE filtering
- PatientAnonymizer: Sequential ID anonymization with mapping
- DualFeaturePreprocessor: Feature engineering for both model types
"""

from .data_cleaner import DataCleaner
from .anonymizer import PatientAnonymizer
from .preprocessor import DualFeaturePreprocessor

__version__ = "1.0.0"
__author__ = "NBE Prediction Team"

__all__ = [
    "DataCleaner",
    "PatientAnonymizer",
    "DualFeaturePreprocessor"
]
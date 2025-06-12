"""
Step 6: API Development Module for NBE Prediction Project
Provides FastAPI endpoints for NBE prediction with dual model architecture
"""

from .api_main import app
from .api_schemas import (
    BaselinePredictionRequest,
    EnhancedPredictionRequest,
    MinimalPredictionResponse,
    DetailedPredictionResponse,
    ResponseType
)
from .model_service import ModelService
from .api_validator import APIValidator

__version__ = "1.0.0"
__author__ = "NBE Prediction Team"

__all__ = [
    "app",
    "BaselinePredictionRequest",
    "EnhancedPredictionRequest",
    "MinimalPredictionResponse",
    "DetailedPredictionResponse",
    "ResponseType",
    "ModelService",
    "APIValidator"
]
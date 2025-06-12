"""
API Schemas Module for NBE Prediction Project
Defines Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


class ResponseType(str, Enum):
    """Response type enumeration"""
    minimal = "minimal"
    detailed = "detailed"


class BaselinePredictionRequest(BaseModel):
    """
    Request schema for baseline NBE prediction (4 core features)
    """
    p_score: int = Field(..., ge=0, le=4, description="Pain score (0=no pain, 4=maximum pain)")
    p_status: int = Field(..., ge=0, le=2, description="Pain status (0=worse, 1=same, 2=better)")
    fl_score: int = Field(..., ge=0, le=4, description="Function limitation score (0=no limit, 4=highest limit)")
    fl_status: int = Field(..., ge=0, le=2, description="Function limitation status (0=worse, 1=same, 2=better)")
    response_type: ResponseType = Field(ResponseType.minimal, description="Response detail level")

    class Config:
        schema_extra = {
            "example": {
                "p_score": 2,
                "p_status": 1,
                "fl_score": 3,
                "fl_status": 2,
                "response_type": "minimal"
            }
        }


class EnhancedPredictionRequest(BaseModel):
    """
    Request schema for enhanced NBE prediction (10 features with temporal context)
    """
    # Core features (required)
    p_score: int = Field(..., ge=0, le=4, description="Pain score (0=no pain, 4=maximum pain)")
    p_status: int = Field(..., ge=0, le=2, description="Pain status (0=worse, 1=same, 2=better)")
    fl_score: int = Field(..., ge=0, le=4, description="Function limitation score (0=no limit, 4=highest limit)")
    fl_status: int = Field(..., ge=0, le=2, description="Function limitation status (0=worse, 1=same, 2=better)")

    # Temporal features (optional with defaults)
    days_since_accident: Optional[int] = Field(21, ge=0, le=1000, description="Days since accident (default: 21)")
    consultation_number: Optional[int] = Field(2, ge=1, le=20, description="Consultation sequence number (default: 2)")

    # Response configuration
    response_type: ResponseType = Field(ResponseType.minimal, description="Response detail level")

    @validator('days_since_accident', pre=True, always=True)
    def set_default_days_since_accident(cls, v):
        return 21 if v is None else v

    @validator('consultation_number', pre=True, always=True)
    def set_default_consultation_number(cls, v):
        return 2 if v is None else v

    class Config:
        schema_extra = {
            "example": {
                "p_score": 2,
                "p_status": 1,
                "fl_score": 3,
                "fl_status": 2,
                "days_since_accident": 21,
                "consultation_number": 2,
                "response_type": "detailed"
            }
        }


class MinimalPredictionResponse(BaseModel):
    """
    Minimal response schema for NBE predictions
    """
    nbe_yes_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of NBE compliance")
    nbe_no_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of NBE non-compliance")

    class Config:
        schema_extra = {
            "example": {
                "nbe_yes_probability": 0.847,
                "nbe_no_probability": 0.153
            }
        }


class DetailedPredictionResponse(BaseModel):
    """
    Detailed response schema for NBE predictions with metadata
    """
    nbe_yes_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of NBE compliance")
    nbe_no_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of NBE non-compliance")
    confidence_level: Literal["low", "medium", "high"] = Field(..., description="Prediction confidence level")
    model_used: str = Field(..., description="Model algorithm used for prediction")
    model_type: Literal["baseline", "enhanced"] = Field(..., description="Feature set type used")
    prediction_timestamp: datetime = Field(..., description="When prediction was made")
    feature_engineering_applied: bool = Field(..., description="Whether interaction features were computed")
    input_validation_passed: bool = Field(True, description="Whether input validation succeeded")

    class Config:
        schema_extra = {
            "example": {
                "nbe_yes_probability": 0.847,
                "nbe_no_probability": 0.153,
                "confidence_level": "high",
                "model_used": "xgboost",
                "model_type": "enhanced",
                "prediction_timestamp": "2025-06-12T15:30:45.123456",
                "feature_engineering_applied": True,
                "input_validation_passed": True
            }
        }


class HealthCheckResponse(BaseModel):
    """
    Health check response schema
    """
    status: Literal["healthy", "unhealthy"] = Field(..., description="API health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    models_loaded: dict = Field(..., description="Status of loaded models")
    api_version: str = Field(..., description="API version")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-06-12T15:30:45.123456",
                "models_loaded": {
                    "baseline_model": "loaded",
                    "enhanced_model": "loaded"
                },
                "api_version": "1.0.0"
            }
        }


class ModelInfoResponse(BaseModel):
    """
    Model information response schema
    """
    baseline_model: dict = Field(..., description="Baseline model information")
    enhanced_model: dict = Field(..., description="Enhanced model information")
    training_timestamp: str = Field(..., description="When models were trained")
    feature_sets: dict = Field(..., description="Feature sets for each model type")

    class Config:
        schema_extra = {
            "example": {
                "baseline_model": {
                    "algorithm": "LogisticRegression",
                    "features": 4,
                    "auc_score": 0.746,
                    "training_samples": 4365
                },
                "enhanced_model": {
                    "algorithm": "XGBoost",
                    "features": 10,
                    "auc_score": 0.801,
                    "training_samples": 4365
                },
                "training_timestamp": "2025-06-12T11:27:36",
                "feature_sets": {
                    "baseline": ["p_score", "p_status", "fl_score", "fl_status"],
                    "enhanced": ["p_score", "p_status", "fl_score", "fl_status", "days_since_accident", "consultation_number", "severity_index", "p_score_fl_score_interaction", "both_improving", "high_severity"]
                }
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response schema
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Detailed error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "p_score must be between 0 and 4",
                "timestamp": "2025-06-12T15:30:45.123456",
                "request_id": "req_123456789"
            }
        }
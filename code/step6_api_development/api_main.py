"""
Main FastAPI Application for NBE Prediction API
Provides dual endpoints for baseline and enhanced NBE predictions
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import uuid
from typing import Union

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import our modules
from code.step6_api_development.api_schemas import (
    BaselinePredictionRequest, EnhancedPredictionRequest,
    MinimalPredictionResponse, DetailedPredictionResponse,
    HealthCheckResponse, ModelInfoResponse, ErrorResponse,
    ResponseType
)
from code.step6_api_development.model_service import ModelService
from code.step6_api_development.api_validator import APIValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NBE Prediction API",
    description="API for predicting Normal Business Expectation (NBE) compliance in medical consultations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for services
model_service: ModelService = None
validator: APIValidator = None


def load_environment_paths():
    """Load paths from environment or use defaults"""
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        project_root = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent.parent))
    except ImportError:
        project_root = Path(__file__).parent.parent.parent

    return {
        'project_root': project_root,
        'models_path': project_root / 'models'
    }


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global model_service, validator

    logger.info("🚀 Starting NBE Prediction API...")

    try:
        # Load paths
        paths = load_environment_paths()
        logger.info(f"Project root: {paths['project_root']}")
        logger.info(f"Models path: {paths['models_path']}")

        # Initialize model service
        model_service = ModelService(paths['models_path'])
        models_loaded = model_service.load_models()

        if not models_loaded:
            logger.error("❌ Failed to load models")
            raise RuntimeError("Model loading failed")

        # Initialize validator
        validator = APIValidator()

        logger.info("✅ API services initialized successfully")

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    request_id = str(uuid.uuid4())[:8]
    logger.error(f"Request {request_id} failed: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An internal error occurred",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
    )


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NBE Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "baseline_prediction": "/api/v1/nbe/predict/baseline",
            "enhanced_prediction": "/api/v1/nbe/predict/enhanced",
            "health_check": "/api/v1/health",
            "model_info": "/api/v1/models/info",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        if model_service is None:
            raise HTTPException(
                status_code=503,
                detail="Model service not initialized"
            )

        health_status = model_service.get_health_status()

        if health_status['status'] == 'unhealthy':
            return JSONResponse(
                status_code=503,
                content=health_status
            )

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )


@app.get("/api/v1/models/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about loaded models"""
    try:
        if model_service is None:
            raise HTTPException(
                status_code=503,
                detail="Model service not initialized"
            )

        return model_service.get_model_info()

    except Exception as e:
        logger.error(f"Model info request failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/api/v1/nbe/predict/baseline",
          response_model=Union[MinimalPredictionResponse, DetailedPredictionResponse])
async def predict_baseline(request: BaselinePredictionRequest):
    """
    Baseline NBE prediction using 4 core features
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"Request {request_id}: Baseline prediction started")

    try:
        # Convert request to dict
        request_data = request.dict()
        response_type = request_data.pop('response_type', ResponseType.minimal)

        # Validate request
        validation_result = validator.validate_request(request_data, 'baseline')

        if not validation_result['is_valid']:
            logger.warning(f"Request {request_id}: Validation failed - {validation_result['errors']}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ValidationError",
                    "message": "Input validation failed",
                    "details": validation_result['errors'],
                    "request_id": request_id
                }
            )

        # Log warnings if any
        if validation_result['warnings']:
            logger.warning(f"Request {request_id}: Validation warnings - {validation_result['warnings']}")

        # Make prediction
        prediction_result = model_service.predict_baseline(request_data)

        # Format response based on type
        if response_type == ResponseType.detailed:
            response = DetailedPredictionResponse(**prediction_result)
        else:
            response = MinimalPredictionResponse(
                nbe_yes_probability=prediction_result['nbe_yes_probability'],
                nbe_no_probability=prediction_result['nbe_no_probability']
            )

        logger.info(f"Request {request_id}: Baseline prediction completed - "
                   f"NBE Yes: {prediction_result['nbe_yes_probability']:.3f}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request {request_id}: Baseline prediction failed - {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PredictionError",
                "message": f"Baseline prediction failed: {str(e)}",
                "request_id": request_id
            }
        )


@app.post("/api/v1/nbe/predict/enhanced",
          response_model=Union[MinimalPredictionResponse, DetailedPredictionResponse])
async def predict_enhanced(request: EnhancedPredictionRequest):
    """
    Enhanced NBE prediction using 10 features including temporal context
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"Request {request_id}: Enhanced prediction started")

    try:
        # Convert request to dict
        request_data = request.dict()
        response_type = request_data.pop('response_type', ResponseType.minimal)

        # Validate request
        validation_result = validator.validate_request(request_data, 'enhanced')

        if not validation_result['is_valid']:
            logger.warning(f"Request {request_id}: Validation failed - {validation_result['errors']}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ValidationError",
                    "message": "Input validation failed",
                    "details": validation_result['errors'],
                    "request_id": request_id
                }
            )

        # Log warnings if any
        if validation_result['warnings']:
            logger.warning(f"Request {request_id}: Validation warnings - {validation_result['warnings']}")

        # Make prediction
        prediction_result = model_service.predict_enhanced(request_data)

        # Format response based on type
        if response_type == ResponseType.detailed:
            response = DetailedPredictionResponse(**prediction_result)
        else:
            response = MinimalPredictionResponse(
                nbe_yes_probability=prediction_result['nbe_yes_probability'],
                nbe_no_probability=prediction_result['nbe_no_probability']
            )

        logger.info(f"Request {request_id}: Enhanced prediction completed - "
                   f"NBE Yes: {prediction_result['nbe_yes_probability']:.3f}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request {request_id}: Enhanced prediction failed - {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PredictionError",
                "message": f"Enhanced prediction failed: {str(e)}",
                "request_id": request_id
            }
        )


@app.get("/api/v1/validation/rules")
async def get_validation_rules():
    """Get information about validation rules and feature requirements"""
    try:
        if validator is None:
            raise HTTPException(
                status_code=503,
                detail="Validator service not initialized"
            )

        return validator.get_feature_info()

    except Exception as e:
        logger.error(f"Validation rules request failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get validation rules: {str(e)}"
        )


# Development server runner
def run_dev_server():
    """Run development server"""
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    run_dev_server()
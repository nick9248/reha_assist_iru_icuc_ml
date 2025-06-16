"""
Production-Ready FastAPI Application for NBE Prediction API
Windows-compatible version without fcntl dependencies
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import uuid
from typing import Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status, Depends
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

# Try to import production security features
try:
    from code.step6_api_development.production_auth import (
        verify_api_key, verify_prediction_permission,
        SecurityHeadersMiddleware, log_requests,
        limiter, production_config, RATE_LIMITING_AVAILABLE
    )
    PRODUCTION_FEATURES_AVAILABLE = True
except ImportError as e:
    # Fallback for development
    PRODUCTION_FEATURES_AVAILABLE = False
    RATE_LIMITING_AVAILABLE = False
    limiter = None

    # Create mock production_config for development
    class MockProductionConfig:
        def __init__(self):
            self.require_auth = False
            self.enable_rate_limiting = False
            self.log_level = "INFO"
            self.cors_origins = ["*"]

        def is_production(self):
            return False

        def get_server_config(self):
            return {"host": "0.0.0.0", "port": 8000, "workers": 1, "timeout": 30, "keepalive": 2}

    production_config = MockProductionConfig()
    print(f"Production features not available: {e}")

# Configure logging
log_level = getattr(logging, getattr(production_config, 'log_level', 'INFO'))
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for services
model_service: ModelService = None
validator: APIValidator = None


def load_environment_paths():
    """Load paths from environment or use defaults"""
    try:
        from dotenv import load_dotenv
        import os

        # Load from project root
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / '.env'

        if env_file.exists():
            load_dotenv(env_file)
        else:
            load_dotenv()

        project_root = Path(os.getenv('PROJECT_ROOT', project_root))
    except ImportError:
        project_root = Path(__file__).parent.parent.parent

    return {
        'project_root': project_root,
        'models_path': project_root / 'models'
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan event handler"""
    # Startup
    global model_service, validator

    env_mode = "production" if (PRODUCTION_FEATURES_AVAILABLE and production_config.is_production()) else "development"
    logger.info(f"🚀 Starting NBE Prediction API in {env_mode} mode...")

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

        # Log security status
        if PRODUCTION_FEATURES_AVAILABLE:
            auth_status = "enabled" if production_config.require_auth else "disabled"
            rate_limit_status = "enabled" if production_config.enable_rate_limiting else "disabled"
            logger.info(f"🔐 Authentication: {auth_status}")
            logger.info(f"⚡ Rate limiting: {rate_limit_status}")
        else:
            logger.info(f"🔐 Authentication: disabled (development mode)")
            logger.info(f"⚡ Rate limiting: disabled (development mode)")

        logger.info("✅ API services initialized successfully")

        yield  # Application is running

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

    # Shutdown
    logger.info("🔄 Shutting down NBE Prediction API...")


# Initialize FastAPI app with lifespan
app_config = {
    "title": "NBE Prediction API",
    "description": "Production-ready API for predicting Normal Business Expectation (NBE) compliance in medical consultations",
    "version": "1.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "lifespan": lifespan
}

# Add OpenAPI customization for production
if PRODUCTION_FEATURES_AVAILABLE and production_config.is_production():
    app_config.update({
        "docs_url": None,  # Disable docs in production
        "redoc_url": None,  # Disable redoc in production
        "openapi_url": None  # Disable OpenAPI schema in production
    })

app = FastAPI(**app_config)

# Add production middleware if available
if PRODUCTION_FEATURES_AVAILABLE:
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # Rate limiting middleware
    if production_config.enable_rate_limiting and RATE_LIMITING_AVAILABLE and limiter:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        from slowapi.middleware import SlowAPIMiddleware

        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)

    # Request logging middleware
    app.middleware("http")(log_requests)

    # CORS with production settings
    cors_origins = production_config.cors_origins
else:
    cors_origins = ["*"]  # Development mode

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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
    endpoints = {
        "baseline_prediction": "/api/v1/nbe/predict/baseline",
        "enhanced_prediction": "/api/v1/nbe/predict/enhanced",
        "health_check": "/api/v1/health",
        "model_info": "/api/v1/models/info"
    }

    # Add docs only in development
    if not (PRODUCTION_FEATURES_AVAILABLE and production_config.is_production()):
        endpoints["docs"] = "/docs"

    return {
        "message": "NBE Prediction API",
        "version": "1.0.0",
        "environment": "production" if (PRODUCTION_FEATURES_AVAILABLE and production_config.is_production()) else "development",
        "endpoints": endpoints,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check(request: Request):
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
async def get_model_info(
    request: Request,
    key_info: dict = Depends(verify_api_key) if PRODUCTION_FEATURES_AVAILABLE else None
):
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
async def predict_baseline(
    request_data: BaselinePredictionRequest,
    request: Request,
    key_info: dict = Depends(verify_prediction_permission) if PRODUCTION_FEATURES_AVAILABLE else None
):
    """
    Baseline NBE prediction using 4 core features
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"Request {request_id}: Baseline prediction started")

    try:
        # Convert request to dict
        request_dict = request_data.model_dump()
        response_type = request_dict.pop('response_type', ResponseType.minimal)

        # Validate request
        validation_result = validator.validate_request(request_dict, 'baseline')

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
        prediction_result = model_service.predict_baseline(request_dict)

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
async def predict_enhanced(
    request_data: EnhancedPredictionRequest,
    request: Request,
    key_info: dict = Depends(verify_prediction_permission) if PRODUCTION_FEATURES_AVAILABLE else None
):
    """
    Enhanced NBE prediction using 10 features including temporal context
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"Request {request_id}: Enhanced prediction started")

    try:
        # Convert request to dict
        request_dict = request_data.model_dump()
        response_type = request_dict.pop('response_type', ResponseType.minimal)

        # Validate request
        validation_result = validator.validate_request(request_dict, 'enhanced')

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
        prediction_result = model_service.predict_enhanced(request_dict)

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
async def get_validation_rules(
    request: Request,
    key_info: dict = Depends(verify_api_key) if PRODUCTION_FEATURES_AVAILABLE else None
):
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
"""
Step 6 Orchestrator: API Development and Testing
Main execution script for NBE prediction API
"""

import sys
from pathlib import Path
import logging
import asyncio
import requests
import json
from datetime import datetime
import uvicorn

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_logging():
    """Setup logging for the orchestrator"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('Step6Orchestrator')

def find_project_root():
    """Find the project root by looking for key directories"""
    current_path = Path(__file__).parent

    # Walk up the directory tree to find project root
    for parent in [current_path] + list(current_path.parents):
        # Look for characteristic project directories
        if (parent / 'models').exists() and (parent / 'data').exists():
            return parent
        # Alternative: look for specific model files
        models_artifacts = parent / 'models' / 'artifacts'
        if models_artifacts.exists():
            step4_files = list(models_artifacts.glob('step4_*.pkl'))
            if step4_files:
                return parent

    # Fallback to current directory structure
    return current_path.parent

def load_environment_paths():
    """Load paths from environment or use defaults"""
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        project_root = Path(os.getenv('PROJECT_ROOT', find_project_root()))
    except ImportError:
        # Fallback if python-dotenv not available
        project_root = find_project_root()

    return {
        'project_root': project_root,
        'models_path': project_root / 'models'
    }

def validate_step4_models(project_root: Path, logger: logging.Logger) -> bool:
    """
    Validate that Step 4 models are available for the API

    Args:
        project_root: Path to project root
        logger: Logger instance

    Returns:
        bool: True if models are available
    """
    logger.info("Validating Step 4 model artifacts...")
    logger.info(f"Project root detected: {project_root}")

    models_dir = project_root / 'models' / 'artifacts'
    logger.info(f"Looking for models in: {models_dir}")

    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")

        # Debug: Check what directories do exist
        if (project_root / 'models').exists():
            logger.info(f"Contents of models directory:")
            for item in (project_root / 'models').iterdir():
                logger.info(f"  - {item.name}")
        else:
            logger.error(f"Models parent directory doesn't exist: {project_root / 'models'}")
            logger.info(f"Contents of project root:")
            for item in project_root.iterdir():
                logger.info(f"  - {item.name}")

        return False

    # List all files in models directory for debugging
    logger.info(f"Contents of {models_dir}:")
    for item in models_dir.iterdir():
        logger.info(f"  - {item.name}")

    # Check for required model files
    required_models = [
        'step4_logistic_regression_baseline_*.pkl',
        'step4_xgboost_enhanced_*.pkl'
    ]

    missing_models = []
    found_models = []

    for model_pattern in required_models:
        matching_files = list(models_dir.glob(model_pattern))
        if not matching_files:
            missing_models.append(model_pattern)
        else:
            found_models.extend([f.name for f in matching_files])

    logger.info(f"Found models: {found_models}")

    if missing_models:
        logger.error(f"Missing model files: {missing_models}")
        logger.error("Please run Step 4 (Model Training) first")
        return False

    logger.info("✅ All required model artifacts found")
    return True

def test_api_endpoints(logger: logging.Logger, base_url: str = "http://localhost:8000"):
    """
    Test API endpoints to verify functionality

    Args:
        logger: Logger instance
        base_url: Base URL for API testing
    """
    logger.info("Testing API endpoints...")

    # Test data
    baseline_test_data = {
        "p_score": 2,
        "p_status": 1,
        "fl_score": 3,
        "fl_status": 2,
        "response_type": "detailed"
    }

    enhanced_test_data = {
        "p_score": 2,
        "p_status": 1,
        "fl_score": 3,
        "fl_status": 2,
        "days_since_accident": 21,
        "consultation_number": 2,
        "response_type": "detailed"
    }

    try:
        # Test health check
        response = requests.get(f"{base_url}/api/v1/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Health check passed")
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False

        # Test baseline prediction
        response = requests.post(
            f"{base_url}/api/v1/nbe/predict/baseline",
            json=baseline_test_data,
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Baseline prediction: NBE Yes = {result.get('nbe_yes_probability', 'N/A'):.3f}")
        else:
            logger.error(f"❌ Baseline prediction failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False

        # Test enhanced prediction
        response = requests.post(
            f"{base_url}/api/v1/nbe/predict/enhanced",
            json=enhanced_test_data,
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Enhanced prediction: NBE Yes = {result.get('nbe_yes_probability', 'N/A'):.3f}")
        else:
            logger.error(f"❌ Enhanced prediction failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False

        # Test model info
        response = requests.get(f"{base_url}/api/v1/models/info", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Model info endpoint working")
        else:
            logger.error(f"❌ Model info failed: {response.status_code}")

        logger.info("🎉 All API tests passed!")
        return True

    except requests.exceptions.ConnectionError:
        logger.error("❌ Could not connect to API - make sure it's running")
        return False
    except Exception as e:
        logger.error(f"❌ API testing failed: {str(e)}")
        return False

def print_api_info(logger: logging.Logger):
    """Print API usage information"""
    logger.info("=" * 60)
    logger.info("NBE PREDICTION API - READY FOR USE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("🌐 API Endpoints:")
    logger.info("   • Health Check:      GET  http://localhost:8000/api/v1/health")
    logger.info("   • Model Info:        GET  http://localhost:8000/api/v1/models/info")
    logger.info("   • Baseline Predict:  POST http://localhost:8000/api/v1/nbe/predict/baseline")
    logger.info("   • Enhanced Predict:  POST http://localhost:8000/api/v1/nbe/predict/enhanced")
    logger.info("")
    logger.info("📚 Documentation:")
    logger.info("   • Interactive Docs:  http://localhost:8000/docs")
    logger.info("   • ReDoc:            http://localhost:8000/redoc")
    logger.info("")
    logger.info("🧪 Example Usage:")
    logger.info("""
   curl -X POST "http://localhost:8000/api/v1/nbe/predict/enhanced" \\
        -H "Content-Type: application/json" \\
        -d '{
            "p_score": 2,
            "p_status": 1,
            "fl_score": 3,
            "fl_status": 2,
            "days_since_accident": 21,
            "consultation_number": 2,
            "response_type": "detailed"
        }'
    """)
    logger.info("=" * 60)

def run_api_server():
    """Run the FastAPI server"""
    # Import the FastAPI app
    from code.step6_api_development.api_main import app

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )

def main():
    """Main execution function"""
    print("🚀 Starting Step 6: API Development")
    print("=" * 60)

    logger = setup_logging()
    logger.info("Step 6: API Development started")

    try:
        # Load paths with improved detection
        paths = load_environment_paths()
        project_root = paths['project_root']
        logger.info(f"Project root: {project_root}")

        if not validate_step4_models(project_root, logger):
            print("❌ Prerequisites not met. Please run Step 4 first.")
            return False

        # Print usage information
        print_api_info(logger)

        # Ask user what to do
        print("\nChoose an option:")
        print("1. Start API server")
        print("2. Test existing API server")
        print("3. Exit")

        while True:
            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == "1":
                logger.info("Starting API server...")
                run_api_server()
                break
            elif choice == "2":
                logger.info("Testing API server...")
                success = test_api_endpoints(logger)
                if success:
                    print("✅ API tests completed successfully!")
                else:
                    print("❌ API tests failed!")
                break
            elif choice == "3":
                logger.info("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

        return True

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return True
    except Exception as e:
        logger.error(f"Error in Step 6 execution: {str(e)}")
        print(f"\n❌ {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Health Check Script for NBE Prediction API Docker Container
Used by Docker HEALTHCHECK and monitoring systems
"""

import sys
import requests
import json
import time
from pathlib import Path

def check_api_health():
    """Check if the API is responding correctly"""
    try:
        # Check basic health endpoint
        response = requests.get(
            "http://localhost:8000/api/v1/health",
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"Health check failed: HTTP {response.status_code}")
            return False
        
        health_data = response.json()
        
        # Check if models are loaded
        if health_data.get('status') != 'healthy':
            print(f"API unhealthy: {health_data.get('status')}")
            return False
        
        models_status = health_data.get('models_loaded', {})
        
        # Verify both models are loaded
        required_models = ['baseline_model', 'enhanced_model']
        for model in required_models:
            if models_status.get(model) != 'loaded':
                print(f"Model not loaded: {model}")
                return False
        
        print("✅ Health check passed")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection refused - API not running")
        return False
    except requests.exceptions.Timeout:
        print("❌ Health check timeout")
        return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def check_model_files():
    """Check if required model files exist"""
    models_path = Path("/app/models/artifacts")
    
    if not models_path.exists():
        print("❌ Models directory not found")
        return False
    
    # Check for required model files
    required_patterns = [
        "*logistic_regression_baseline*.pkl",
        "*xgboost_enhanced*.pkl"
    ]
    
    for pattern in required_patterns:
        model_files = list(models_path.glob(pattern))
        if not model_files:
            print(f"❌ No model files found matching: {pattern}")
            return False
    
    print("✅ Model files check passed")
    return True

def main():
    """Main health check function"""
    print(f"🔍 Starting health check at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    checks = [
        ("API Health", check_api_health),
        ("Model Files", check_model_files),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {str(e)}")
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("✅ All health checks passed")
        sys.exit(0)
    else:
        print("❌ Some health checks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
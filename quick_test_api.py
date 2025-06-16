"""
Quick API Test Script for NBE Prediction API
Run this in PyCharm to validate your API
"""

import requests
import json
import time


def test_api_quickly():
    """Quick test of all API endpoints"""
    base_url = "http://localhost:8000"

    print("🧪 Quick API Test Suite")
    print("=" * 50)

    # Test 1: Health Check
    print("1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/api/v1/health")
        if response.status_code == 200:
            print(f"   ✅ Health check passed: {response.json()['status']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")

    # Test 2: Model Info
    print("\n2️⃣ Testing Model Info...")
    try:
        response = requests.get(f"{base_url}/api/v1/models/info")
        if response.status_code == 200:
            info = response.json()
            print(f"   ✅ Models loaded:")
            print(
                f"      Baseline: {info['baseline_model']['algorithm']} (AUC: {info['baseline_model']['auc_score']:.3f})")
            print(
                f"      Enhanced: {info['enhanced_model']['algorithm']} (AUC: {info['enhanced_model']['auc_score']:.3f})")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Model info error: {e}")

    # Test 3: Baseline Prediction
    print("\n3️⃣ Testing Baseline Prediction...")
    baseline_payload = {
        "p_score": 2,
        "p_status": 1,
        "fl_score": 3,
        "fl_status": 2,
        "response_type": "detailed"
    }

    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/api/v1/nbe/predict/baseline", json=baseline_payload)
        response_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Baseline prediction successful ({response_time:.1f}ms):")
            print(f"      NBE Yes: {result['nbe_yes_probability']:.3f}")
            print(f"      NBE No: {result['nbe_no_probability']:.3f}")
            print(f"      Confidence: {result['confidence_level']}")
            print(f"      Model: {result['model_used']}")
        else:
            print(f"   ❌ Baseline prediction failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Baseline prediction error: {e}")

    # Test 4: Enhanced Prediction
    print("\n4️⃣ Testing Enhanced Prediction...")
    enhanced_payload = {
        "p_score": 2,
        "p_status": 1,
        "fl_score": 3,
        "fl_status": 2,
        "days_since_accident": 21,
        "consultation_number": 2,
        "response_type": "detailed"
    }

    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/api/v1/nbe/predict/enhanced", json=enhanced_payload)
        response_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Enhanced prediction successful ({response_time:.1f}ms):")
            print(f"      NBE Yes: {result['nbe_yes_probability']:.3f}")
            print(f"      NBE No: {result['nbe_no_probability']:.3f}")
            print(f"      Confidence: {result['confidence_level']}")
            print(f"      Model: {result['model_used']}")
        else:
            print(f"   ❌ Enhanced prediction failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Enhanced prediction error: {e}")

    # Test 5: Error Handling
    print("\n5️⃣ Testing Error Handling...")
    invalid_payload = {
        "p_score": 5,  # Invalid: should be 0-4
        "p_status": 1,
        "fl_score": 3,
        "fl_status": 2
    }

    try:
        response = requests.post(f"{base_url}/api/v1/nbe/predict/baseline", json=invalid_payload)
        if response.status_code == 422:
            print(f"   ✅ Error handling works: correctly rejected invalid input")
        else:
            print(f"   ⚠️ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")

    # Test 6: Documentation
    print("\n6️⃣ Testing Documentation...")
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print(f"   ✅ Documentation accessible at {base_url}/docs")
        else:
            print(f"   ❌ Documentation not accessible: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Documentation test error: {e}")

    print("\n" + "=" * 50)
    print("🎉 Quick test completed!")
    print(f"🌐 Visit {base_url}/docs for interactive API testing")
    print("=" * 50)


if __name__ == "__main__":
    test_api_quickly()
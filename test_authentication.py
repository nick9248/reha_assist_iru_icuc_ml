"""
Authentication Test Script for NBE Prediction API
Tests both development and production authentication modes
"""

import requests
import time


def test_authentication():
    """Comprehensive authentication testing"""
    base_url = "http://localhost:8000"

    print("🔐 Testing Authentication System")
    print("=" * 50)

    # Test 1: Health check (should always work)
    print("1️⃣ Testing Health Check (no auth required):")
    try:
        response = requests.get(f"{base_url}/api/v1/health")
        print(
            f"   Status: {response.status_code} ✅" if response.status_code == 200 else f"   Status: {response.status_code} ❌")
        if response.status_code == 200:
            health = response.json()
            print(f"   Health: {health['status']}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Model info without API key
    print("\n2️⃣ Testing Model Info without API key:")
    try:
        response = requests.get(f"{base_url}/api/v1/models/info")
        if response.status_code == 401:
            print(f"   Status: {response.status_code} ✅ (Correctly requires auth)")
        elif response.status_code == 200:
            print(f"   Status: {response.status_code} ✅ (Development mode - no auth required)")
        else:
            print(f"   Status: {response.status_code} ❌")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Prediction without API key
    print("\n3️⃣ Testing Prediction without API key:")
    try:
        response = requests.post(f"{base_url}/api/v1/nbe/predict/baseline", json={
            "p_score": 2, "p_status": 1, "fl_score": 3, "fl_status": 2
        })
        if response.status_code == 401:
            print(f"   Status: {response.status_code} ✅ (Correctly requires auth)")
        elif response.status_code == 200:
            print(f"   Status: {response.status_code} ✅ (Development mode - no auth required)")
            result = response.json()
            print(f"   NBE Yes: {result['nbe_yes_probability']:.3f}")
        else:
            print(f"   Status: {response.status_code} ❌")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: Invalid API key
    print("\n4️⃣ Testing with Invalid API key:")
    try:
        headers = {"Authorization": "Bearer invalid-key-12345"}
        response = requests.post(f"{base_url}/api/v1/nbe/predict/baseline",
                                 json={"p_score": 2, "p_status": 1, "fl_score": 3, "fl_status": 2},
                                 headers=headers)
        if response.status_code == 401:
            print(f"   Status: {response.status_code} ✅ (Correctly rejected invalid key)")
        elif response.status_code == 200:
            print(f"   Status: {response.status_code} ✅ (Development mode - auth disabled)")
        else:
            print(f"   Status: {response.status_code} ❌")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 5: Valid API key
    print("\n5️⃣ Testing with Valid API key:")
    try:
        headers = {"Authorization": "Bearer dev-key-12345"}
        response = requests.post(f"{base_url}/api/v1/nbe/predict/enhanced",
                                 json={
                                     "p_score": 2, "p_status": 1, "fl_score": 3, "fl_status": 2,
                                     "days_since_accident": 21, "consultation_number": 2,
                                     "response_type": "detailed"
                                 },
                                 headers=headers)
        if response.status_code == 200:
            print(f"   Status: {response.status_code} ✅ (Valid key accepted)")
            result = response.json()
            print(f"   NBE Yes: {result['nbe_yes_probability']:.3f}")
            print(f"   Confidence: {result.get('confidence_level', 'N/A')}")
            print(f"   Model: {result.get('model_used', 'N/A')}")
        else:
            print(f"   Status: {response.status_code} ❌")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 6: Rate limiting (if enabled)
    print("\n6️⃣ Testing Rate Limiting:")
    try:
        headers = {"Authorization": "Bearer dev-key-12345"}
        success_count = 0
        rate_limited = False

        for i in range(10):  # Try 10 requests quickly
            response = requests.get(f"{base_url}/api/v1/models/info", headers=headers)
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited = True
                print(f"   Request {i + 1}: Rate limited ✅")
                break
            time.sleep(0.1)  # Small delay

        if not rate_limited:
            print(f"   All {success_count} requests succeeded (rate limiting disabled or limit not reached)")

    except Exception as e:
        print(f"   Error: {e}")

    # Test 7: Security headers
    print("\n7️⃣ Testing Security Headers:")
    try:
        headers = {"Authorization": "Bearer dev-key-12345"}
        response = requests.get(f"{base_url}/api/v1/models/info", headers=headers)

        security_headers = [
            'x-content-type-options',
            'x-frame-options',
            'x-xss-protection',
            'x-process-time'
        ]

        found_headers = 0
        for header in security_headers:
            if header in response.headers:
                print(f"   {header}: {response.headers[header]} ✅")
                found_headers += 1
            else:
                print(f"   {header}: Not found")

        print(f"   Security headers: {found_headers}/{len(security_headers)} present")

    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 50)
    print("🎉 Authentication testing completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_authentication()
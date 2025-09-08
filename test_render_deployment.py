"""
Test Render deployment of the API
"""
import requests
import json

# Your Render URL
RENDER_URL = "https://weather-prediction-api-lbhz.onrender.com"

def test_deployment():
    print("=== Testing Render Deployment ===\n")
    print(f"URL: {RENDER_URL}\n")
    
    # 1. Test root endpoint
    print("1. Testing project info endpoint")
    try:
        response = requests.get(f"{RENDER_URL}/", timeout=30)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Project: {data.get('project')}")
            print(f"   GitHub: {data.get('github_repository')}")
    except requests.exceptions.Timeout:
        print("   Note: First request may take ~30s to wake up the service")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Test health check
    print("\n2. Testing health check")
    try:
        response = requests.get(f"{RENDER_URL}/health/", timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Test rain prediction
    print("\n3. Testing rain prediction")
    test_date = "2025-09-07"
    try:
        response = requests.get(f"{RENDER_URL}/predict/rain/", params={"date": test_date}, timeout=10)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. Test precipitation prediction
    print("\n4. Testing precipitation prediction")
    try:
        response = requests.get(f"{RENDER_URL}/predict/precipitation/fall", params={"date": test_date}, timeout=10)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n=== Deployment Test Complete ===")
    print(f"\nVisit API Documentation: {RENDER_URL}/docs")

if __name__ == "__main__":
    print("Testing deployed API...")
    test_deployment()
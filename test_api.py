import requests
import json
import time
import subprocess
import sys
import os
from threading import Thread

def test_api_endpoints():
    """Test the FastAPI endpoints."""
    base_url = "http://localhost:8000"
    
    # Wait for server to start
    print("Waiting for API server to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                print("✓ API server is running!")
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        print("❌ API server failed to start within 30 seconds")
        return False
    
    print("\n" + "=" * 50)
    print("TESTING API ENDPOINTS")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        print("✓ Health check passed!")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        print("✓ Root endpoint passed!")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False
    
    # Test 3: Main prediction endpoint
    print("\n3. Testing main prediction endpoint...")
    test_requests = [
        {
            "text": "the quick brown",
            "top_k": 3
        },
        {
            "text": "machine learning is",
            "top_k": 2
        },
        {
            "text": "neural networks can"
            # top_k defaults to 3
        }
    ]
    
    for i, test_data in enumerate(test_requests):
        try:
            print(f"\n  Test 3.{i+1}: {test_data}")
            response = requests.post(
                f"{base_url}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Input: {result['input_text']}")
                print(f"  Predictions: {len(result['predictions'])}")
                for j, pred in enumerate(result['predictions']):
                    print(f"    {j+1}. '{pred['word']}' (prob: {pred['probability']:.4f})")
                print("  ✓ Prediction successful!")
            else:
                print(f"  ❌ Prediction failed: {response.text}")
                return False
        except Exception as e:
            print(f"  ❌ Prediction test failed: {e}")
            return False
    
    # Test 4: Single word prediction
    print("\n4. Testing single word prediction...")
    try:
        test_data = {"text": "the quick brown"}
        response = requests.post(
            f"{base_url}/predict/single",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
            print("✓ Single word prediction passed!")
        else:
            print(f"❌ Single word prediction failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Single word prediction failed: {e}")
        return False
    
    # Test 5: Text generation
    print("\n5. Testing text generation...")
    try:
        test_data = {
            "text": "machine learning is",
            "max_words": 10,
            "temperature": 1.0
        }
        response = requests.post(
            f"{base_url}/generate",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Input: {result['input_text']}")
            print(f"Generated: {result['generated_text']}")
            print("✓ Text generation passed!")
        else:
            print(f"❌ Text generation failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False
    
    # Test 6: Error handling
    print("\n6. Testing error handling...")
    try:
        # Empty text should return 400
        response = requests.post(
            f"{base_url}/predict",
            json={"text": "", "top_k": 3},
            headers={"Content-Type": "application/json"}
        )
        print(f"Empty text status: {response.status_code}")
        assert response.status_code == 400
        
        # Invalid top_k should return 400
        response = requests.post(
            f"{base_url}/predict",
            json={"text": "test", "top_k": 15},
            headers={"Content-Type": "application/json"}
        )
        print(f"Invalid top_k status: {response.status_code}")
        assert response.status_code == 400
        
        print("✓ Error handling passed!")
    except Exception as e:
        print(f"❌ Error handling failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("ALL API TESTS PASSED! ✓")
    print("=" * 50)
    return True

def main():
    """Main test function."""
    print("FastAPI Next Word Prediction API Test")
    print("=" * 50)
    
    # Check if model file exists
    if not os.path.exists('next_word_lstm.pth'):
        print("❌ Model file 'next_word_lstm.pth' not found.")
        print("Please run 'python3 train_model.py' first to train a model.")
        return
    
    # Start the API server in a separate process
    print("Starting API server...")
    try:
        # Start server process
        server_process = subprocess.Popen([
            sys.executable, "api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Run tests
        success = test_api_endpoints()
        
        # Stop server
        print("\nStopping API server...")
        server_process.terminate()
        server_process.wait(timeout=10)
        
        if success:
            print("\n✅ ALL TESTS PASSED!")
            print("\nTo start the API server manually, run:")
            print("  python3 api.py")
            print("\nAPI Documentation: http://localhost:8000/docs")
        else:
            print("\n❌ Some tests failed.")
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        try:
            server_process.terminate()
        except:
            pass

if __name__ == "__main__":
    main()

import requests
import json
import time

def test_local_api():
    """Test the locally running API"""
    # URL for local testing - Flask default port is 5000
    # base_url = "http://127.0.0.1:5000"
    base_url = "https://prakruthinischikaran-api.onrender.com"
    
    # Test the home endpoint
    print("\n===== Testing Home Endpoint =====")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Home endpoint test passed")
        else:
            print("❌ Home endpoint test failed")
    except Exception as e:
        print(f"❌ Home endpoint test failed with error: {str(e)}")
    
    # Test the prediction endpoint
    print("\n===== Testing Prediction Endpoint =====")
    
    # Create test payload with ALL required features (using exact names from app.py)
    payload = {
        'Skin color': 1,
        'Skin texture': 1,
        'Details about hair type': 1,
        'Texture of hair ': 1,
        'My hairs are': 1,
        'Hair falling tendency\t': 1,
        'Eyes\t': 1,
        'My eyes  eye balls are': 1,
        'Physique\t': 1,
        'Veins\t': 1,
        'Usual body temperature \t': 1,
        'Teeth \t': 1,
        'Is your lips\t': 2,
        'Hands(Length)': 2,
        'Body hair\t': 2,
        'Joints\t': 2,
        'Chest\t': 2,
        'Tongue\t': 2,
        'Body frame (Length)\t': 2,
        'Body frame (Breadth)\t': 2,
        'Bodyweight changes\t': 2,
        'Style of speaking\t': 2,
        'Voice \t': 2,
        'Walk \t': 2,
        'Appetite \t ': 2,
        'Intensity of hunger\t': 2,
        'Favorite climate\t': 2,
        'Stamina\t': 2,
        'Tastes, I like most': 2,
        'Eagerness about drinking water': 2,
        'Frequency of liquid intake\t': 2,
        'Sleep\t': 2,
        'Quantity of sweat\t': 2,
        'Emotion\t': 2,
        'Intellect\t': 2,
        'Grasping\t': 2,
        'Concentration\t': 2,
        'Memory\t': 2,
        'Temperament\t': 2,
        'Patience\t': 2,
        'Satisfaction/content\t': 2,
        'While making decisions\t': 2,
        'Activities\t': 2,
        'Nature of dreams\t': 2,
        'Friendships & relations\t': 2,
        'Faith in God\t': 2,
        'Self-control\t': 2,
        'Gratefulness\t': 2,
        'Likings\t': 2,
        'Liking- type of food': 2,
        'Politeness/humbleness':2,
        'Biting nails ': 1,
        'Fond of praise ': 1,
        'Brave': 1,
        'Egoistic': 1
    }
    
    try:
        # Measure response time
        start_time = time.time()
        response = requests.post(f"{base_url}/predict", json=payload)
        end_time = time.time()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {(end_time - start_time) * 1000:.2f} ms")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result.get('prediction', 'Not found')}")
            print("Probabilities:")
            for dosha, prob in result.get('probabilities', {}).items():
                print(f"  {dosha}: {prob:.4f}")
            print("✅ Prediction endpoint test passed")
        else:
            print(f"❌ Prediction endpoint test failed: {response.text}")
    except Exception as e:
        print(f"❌ Prediction endpoint test failed with error: {str(e)}")
    
    # Test with missing features
    print("\n===== Testing Error Handling (Missing Features) =====")
    
    # Create a payload with missing features
    incomplete_payload = {
        'Skin color': 1,
        'Skin texture': 1
        # Missing all other features
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=incomplete_payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 400:
            print("✅ Error handling test passed (correctly detected missing features)")
        else:
            print("❌ Error handling test failed (should return 400 for missing features)")
    except Exception as e:
        print(f"❌ Error handling test failed with error: {str(e)}")
    
    # Test with non-numeric values
    print("\n===== Testing Error Handling (Non-numeric Values) =====")
    
    # Create a payload with string values (but valid numbers)
    string_payload = {key: "1" for key in payload}  # Convert all values to strings
    
    try:
        response = requests.post(f"{base_url}/predict", json=string_payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ String conversion test passed (API successfully converted strings to integers)")
        else:
            print(f"❌ String conversion test failed: {response.text}")
    except Exception as e:
        print(f"❌ String conversion test failed with error: {str(e)}")
        
    print("\n===== All API Tests Completed =====")

if __name__ == "__main__":
    test_local_api()
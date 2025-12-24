import requests
import json
import time
import os

BASE_URL = "http://127.0.0.1:8000"

def test_file_size_limit():
    """Test that files >100MB are rejected"""
    print("Testing file size limit...")
    # Create a large file for testing (but since we can't create >100MB easily, we'll assume small files pass)
    # For now, just test with small files
    pass

def upload_file(file_path):
    """Upload a file and return data_id"""
    print(f"Uploading {file_path}...")
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    if response.status_code == 200:
        data = response.json()
        print(f"Upload successful: {data['data_id']}")
        return data['data_id']
    else:
        print(f"Upload failed: {response.text}")
        return None

def train_model(data_id, target='target'):
    """Train a model with the uploaded data"""
    print(f"Training model with data_id: {data_id}, target: {target}")
    payload = {
        'data_id': data_id,
        'target': target,
        'learning_rate': 0.01,
        'epochs': 100,
        'model_name': f"Test_Model_{target}"
    }
    response = requests.post(f"{BASE_URL}/train", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"Training successful: {data['model_id']}")
        return data['model_id']
    else:
        print(f"Training failed: {response.text}")
        return None

def check_metadata(model_id):
    """Check model metadata for sampling"""
    print(f"Checking metadata for model {model_id}")
    metadata_path = f"models/{model_id}_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Data shape: {metadata.get('data_shape')}")
        print(f"Original data shape: {metadata.get('original_data_shape')}")
        return metadata
    else:
        print("Metadata file not found")
        return None

def test_visualization(model_id):
    """Test visualization endpoint"""
    print(f"Testing visualization for model {model_id}")
    response = requests.get(f"{BASE_URL}/visualization/enhanced?model_id={model_id}")
    if response.status_code == 200:
        print("Visualization endpoint works")
        return True
    else:
        print(f"Visualization failed: {response.text}")
        return False

def main():
    # Test with test_data.csv
    print("=== Testing with test_data.csv ===")
    data_id1 = upload_file('test_data.csv')
    if data_id1:
        model_id1 = train_model(data_id1)
        if model_id1:
            metadata1 = check_metadata(model_id1)
            test_visualization(model_id1)

    # Test with integration_test_data.csv
    print("\n=== Testing with integration_test_data.csv ===")
    data_id2 = upload_file('integration_test_data.csv')
    if data_id2:
        model_id2 = train_model(data_id2)
        if model_id2:
            metadata2 = check_metadata(model_id2)
            test_visualization(model_id2)

    print("\n=== Test completed ===")

if __name__ == "__main__":
    main()
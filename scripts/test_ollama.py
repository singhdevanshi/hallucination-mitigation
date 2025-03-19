import os
import json
import requests

# Set up basic configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def test_ollama():
    """Simple test to check Ollama connectivity"""
    print("Testing Ollama API...")
    
    # Simple payload
    payload = {
        "model": "mistral:7b",
        "prompt": "Hello, world!",
        "stream": False
    }
    
    try:
        # Make request
        response = requests.post(OLLAMA_API_URL, json=payload)
        print(f"Status code: {response.status_code}")
        
        # Check response
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data.get('response', '')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

if __name__ == "__main__":
    test_ollama()
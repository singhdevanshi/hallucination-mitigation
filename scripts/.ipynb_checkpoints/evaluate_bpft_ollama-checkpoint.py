import sys
import logging
import requests
import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def ollama_generate(prompt, model="mistral:7b"):
    """Generate response from Ollama using the specified model."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    
    try:
        logging.info(f"Sending request to Ollama API for model {model}...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json().get("response", "")
            return result
        else:
            logging.error(f"Error response text: {response.text}")
            raise Exception(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        logging.error(f"Exception in ollama_generate: {str(e)}")
        raise

def load_model(model_name):
    """Check if the model is available in Ollama"""
    try:
        # Test if model is available
        test_prompt = "Hello, can you hear me?"
        ollama_generate(test_prompt, model=model_name)
        logging.info(f"Model {model_name} is available")
        return model_name
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}")
        sys.exit(1)

def load_evaluation_dataset(file_path=None):
    """Load evaluation dataset from file or use a small sample"""
    if file_path and os.path.exists(file_path):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                return df.to_dict(orient='records')
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
    
    # Use a small sample dataset if file not found
    logging.info("Using sample dataset")
    return [
        {'input': 'What is the capital of France?', 'label': 'Paris is the capital of France.'},
        {'input': 'Who wrote Hamlet?', 'label': 'William Shakespeare wrote Hamlet.'},
        {'input': 'What is the tallest mountain in the world?', 'label': 'Mount Everest is the tallest mountain in the world.'}
    ]

def preprocess_predictions(predictions, labels, tokenizer):
    """Preprocess predictions and labels for evaluation"""
    processed_predictions = []
    processed_labels = []
    
    for pred, label in zip(predictions, labels):
        # Tokenize and truncate to first 50 tokens for comparison
        pred_tokens = tokenizer.encode(pred, truncation=True, max_length=50)
        label_tokens = tokenizer.encode(label, truncation=True, max_length=50)
        
        # Convert back to text for simple comparison
        proc_pred = tokenizer.decode(pred_tokens)
        proc_label = tokenizer.decode(label_tokens)
        
        processed_predictions.append(proc_pred)
        processed_labels.append(proc_label)
    
    return processed_predictions, processed_labels

def compute_metrics(predictions, true_labels):
    """Compute simple metrics for text similarity"""
    # For simplicity, check if key information is present
    binary_predictions = []
    binary_labels = []
    
    for pred, label in zip(predictions, true_labels):
        # Check if prediction contains key information from label
        contains_key_info = any(key_phrase in pred.lower() for key_phrase in label.lower().split())
        binary_predictions.append(contains_key_info)
        binary_labels.append(True)  # Assume label is always correct
    
    accuracy = sum(binary_predictions) / len(binary_predictions)
    return accuracy

def evaluate_model(model, dataset):
    """Evaluate model on dataset"""
    logging.info(f"Evaluating model {model} on {len(dataset)} examples")
    predictions = []
    true_labels = []

    for data in dataset:
        try:
            prompt = data['input']
            label = data['label']
            
            # Generate response using Ollama
            prediction = ollama_generate(prompt, model=model)
            
            predictions.append(prediction)
            true_labels.append(label)
            
            logging.info(f"Q: {prompt}")
            logging.info(f"A (model): {prediction[:100]}...")
            logging.info(f"A (reference): {label[:100]}...")
            logging.info("-" * 40)
            
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")

    # Use BERT tokenizer for preprocessing
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    processed_predictions, processed_labels = preprocess_predictions(predictions, true_labels, tokenizer)
    
    accuracy = compute_metrics(processed_predictions, processed_labels)
    
    return predictions, true_labels, accuracy

def main():
    # Set up models to evaluate
    base_model = "mistral:7b"
    bpft_model = "mistral-bpft:latest"
    
    # Load evaluation dataset
    dataset_path = '/workspace/data/evaluation/eval_samples.json'
    dataset = load_evaluation_dataset(dataset_path)
    
    # Evaluate base model
    logging.info("Evaluating base model...")
    base_model_loaded = load_model(base_model)
    base_predictions, base_labels, base_accuracy = evaluate_model(base_model_loaded, dataset)
    
    # Evaluate BPFT model if available
    try:
        logging.info("Evaluating BPFT model...")
        bpft_model_loaded = load_model(bpft_model)
        bpft_predictions, bpft_labels, bpft_accuracy = evaluate_model(bpft_model_loaded, dataset)
        
        # Compare results
        logging.info("\nEvaluation Results:")
        logging.info(f"Base Model Accuracy: {base_accuracy:.4f}")
        logging.info(f"BPFT Model Accuracy: {bpft_accuracy:.4f}")
        logging.info(f"Improvement: {(bpft_accuracy - base_accuracy) * 100:.2f}%")
        
        # Save results
        os.makedirs('/workspace/results', exist_ok=True)
        with open('/workspace/results/evaluation_comparison.json', 'w') as f:
            json.dump({
                'base_model': {
                    'model': base_model,
                    'accuracy': base_accuracy,
                },
                'bpft_model': {
                    'model': bpft_model,
                    'accuracy': bpft_accuracy,
                },
                'examples': [
                    {
                        'prompt': dataset[i]['input'],
                        'reference': dataset[i]['label'],
                        'base_response': base_predictions[i],
                        'bpft_response': bpft_predictions[i]
                    }
                    for i in range(min(5, len(dataset)))  # Save first 5 examples
                ]
            }, f, indent=2)
        
    except Exception as e:
        logging.error(f"Failed to evaluate BPFT model: {e}")
        logging.info("\nEvaluation Results:")
        logging.info(f"Base Model Accuracy: {base_accuracy:.4f}")
        logging.info("BPFT model not available for comparison")

if __name__ == "__main__":
    main()
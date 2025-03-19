import os
import json
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

# Create directories for storing training data
os.makedirs('/workspace/data/bpft', exist_ok=True)

# Download necessary resources
nltk.download('punkt')

OLLAMA_API_URL = "http://localhost:11434/api/generate"
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_response(prompt, model="mistral:7b"):
    """Generate response using Ollama"""
    payload = {"model": model, "prompt": prompt, "stream": False}
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""

def compute_contradiction_score(correct_embeddings, incorrect_embeddings):
    """Compute contradiction score between correct and incorrect answers"""
    similarities = []
    for correct_emb in correct_embeddings:
        for incorrect_emb in incorrect_embeddings:
            sim = np.dot(correct_emb, incorrect_emb) / (np.linalg.norm(correct_emb) * np.linalg.norm(incorrect_emb))
            similarities.append(sim)
    contradiction_score = 1.0 - (sum(similarities) / len(similarities)) if similarities else 1.0
    return contradiction_score

def load_truthful_qa():
    """Load TruthfulQA dataset"""
    print("Loading TruthfulQA dataset...")
    truthful_qa = load_dataset("truthful_qa", "multiple_choice")
    
    dataset = []
    
    # Use validation set
    for item in tqdm(truthful_qa['validation']):
        question = item['question']
        
        # Get correct answer
        if 'mc1_targets' in item and 'mc1_choices' in item:
            correct_idx = item['mc1_targets'].index(1) if 1 in item['mc1_targets'] else 0
            correct_answer = item['mc1_choices'][correct_idx]
            
            # Find incorrect answer
            incorrect_indices = [i for i, target in enumerate(item['mc1_targets']) if target == 0]
            if incorrect_indices:
                incorrect_idx = incorrect_indices[0]  # Take the first incorrect answer
                incorrect_answer = item['mc1_choices'][incorrect_idx]
                
                dataset.append({
                    'question': question,
                    'correct_answer': correct_answer,
                    'incorrect_answer': incorrect_answer
                })
    
    print(f"Loaded {len(dataset)} samples from TruthfulQA")
    return dataset

def get_model_generated_answers(questions, model="mistral:7b"):
    """Get model-generated answers for questions"""
    print(f"Getting model-generated answers using {model}...")
    answers = []
    
    for question in tqdm(questions):
        prompt = f"Question: {question}\nAnswer: "
        response = generate_response(prompt, model=model)
        answers.append(response)
    
    return answers

def generate_contradictory_samples(dataset, num_samples=100):
    """Generate training samples with correct and incorrect answers"""
    print("Generating training samples...")
    samples = []

    for i, item in tqdm(enumerate(dataset), total=min(num_samples, len(dataset))):
        if i >= num_samples:
            break

        question = item['question']
        correct_answer = item['correct_answer']
        incorrect_answer = item['incorrect_answer']

        prompt = f"Question: {question}\nAnswer: "
        
        # Skip if answers are too short
        if len(correct_answer) < 5 or len(incorrect_answer) < 5:
            continue
            
        correct_sentences = sent_tokenize(correct_answer)
        incorrect_sentences = sent_tokenize(incorrect_answer)
        
        # Skip if not enough sentences
        if len(correct_sentences) < 1 or len(incorrect_sentences) < 1:
            continue

        # Calculate embeddings
        try:
            correct_embeddings = sentence_model.encode(correct_sentences)
            incorrect_embeddings = sentence_model.encode(incorrect_sentences)
            
            contradiction_score = compute_contradiction_score(correct_embeddings, incorrect_embeddings)
            
            samples.append({
                "input": prompt,
                "response": correct_answer,
                "factuality_score": 1.0,
                "contradiction_score": 0.0
            })
            samples.append({
                "input": prompt,
                "response": incorrect_answer,
                "factuality_score": 0.0,
                "contradiction_score": float(contradiction_score)
            })
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Save samples as JSON
    with open("/workspace/data/bpft/bpft_dataset.json", "w") as f:
        json.dump(samples, f, indent=2)
        
    # Save samples as CSV
    df = pd.DataFrame(samples)
    df.to_csv("/workspace/data/bpft/bpft_training_data.csv", index=False)
    
    print(f"Generated {len(samples)} training samples")
    return samples

if __name__ == "__main__":
    # Load TruthfulQA dataset
    dataset = load_truthful_qa()
    
    # Generate BPFT dataset
    samples = generate_contradictory_samples(dataset, num_samples=100)
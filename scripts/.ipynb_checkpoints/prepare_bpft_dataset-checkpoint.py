import os
import json
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import numpy as np

# Create directories for storing training data
os.makedirs('/workspace/data/bpft', exist_ok=True)

# Download necessary resources
nltk.download('punkt')

OLLAMA_API_URL = "http://localhost:11434/api/generate"
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_response(prompt):
    """Generate response using Ollama"""
    payload = {"model": "mistral:7b", "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def compute_contradiction_score(correct_embeddings, incorrect_embeddings):
    """Compute contradiction score between correct and incorrect answers"""
    similarities = []
    for correct_emb in correct_embeddings:
        for incorrect_emb in incorrect_embeddings:
            sim = np.dot(correct_emb, incorrect_emb) / (np.linalg.norm(correct_emb) * np.linalg.norm(incorrect_emb))
            similarities.append(sim)
    contradiction_score = 1.0 - (sum(similarities) / len(similarities)) if similarities else 1.0
    return contradiction_score

def generate_contradictory_samples(dataset, num_samples=1000):
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
        correct_sentences = sent_tokenize(correct_answer)
        incorrect_sentences = sent_tokenize(incorrect_answer)

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
            "contradiction_score": contradiction_score
        })

    # Save samples as JSON
    with open("/workspace/data/bpft/bpft_dataset.json", "w") as f:
        json.dump(samples, f, indent=4)

if __name__ == "__main__":
    # Load dataset (replace with actual dataset)
    dataset = [
        {"question": "What is the capital of France?", "correct_answer": "Paris.", "incorrect_answer": "London."},
        {"question": "Who wrote Hamlet?", "correct_answer": "William Shakespeare.", "incorrect_answer": "Charles Dickens."}
    ]
    generate_contradictory_samples(dataset, num_samples=1000)
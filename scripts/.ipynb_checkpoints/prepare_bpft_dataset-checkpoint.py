import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import numpy as np

# Create directories for storing training data
os.makedirs('/workspace/data/bpft', exist_ok=True)

# Download necessary resources
nltk.download('punkt')

def init_models():
    """Initialize models and tokenizers"""
    print("Loading models...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    # Load sentence embedding model for semantic similarity
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return tokenizer, sentence_model

def load_factual_datasets():
    """Load factual QA datasets for training"""
    print("Loading datasets...")
    
    # Load TruthfulQA dataset (multiple choice variant)
    truthful_qa = load_dataset("truthful_qa", "multiple_choice")
    
    return truthful_qa

def compute_contradiction_score(correct_embeddings, incorrect_embeddings):
    """Compute contradiction score between correct and incorrect answers"""
    similarities = []
    
    for correct_emb in correct_embeddings:
        for incorrect_emb in incorrect_embeddings:
            sim = torch.nn.functional.cosine_similarity(
                torch.tensor(correct_emb).unsqueeze(0),
                torch.tensor(incorrect_emb).unsqueeze(0),
                dim=1
            )
            similarities.append(sim.item())
    
    # Lower similarity indicates higher contradiction
    contradiction_score = 1.0 - (sum(similarities) / len(similarities)) if similarities else 1.0
    return contradiction_score

def generate_contradictory_samples(tokenizer, sentence_model, dataset, num_samples=1000):
    """Generate training samples with correct and incorrect answers"""
    print("Generating training samples...")
    
    samples = []
    
    for i, item in tqdm(enumerate(dataset['validation']), total=min(num_samples, len(dataset['validation']))):
        if i >= num_samples:
            break
            
        question = item['question']
        
        # Get correct and incorrect answers
        correct_idx = item['mc1_targets'].index(1) if 1 in item['mc1_targets'] else 0
        correct_answer = item['mc1_choices'][correct_idx]
        
        incorrect_idx = [i for i, x in enumerate(item['mc1_targets']) if x == 0]
        if incorrect_idx:
            incorrect_answer = item['mc1_choices'][incorrect_idx[0]]
        else:
            continue
        
        # Tokenize the question for prompt creation
        prompt = f"Question: {question}\nAnswer: "
        
        # Tokenize and split answers into sentences
        correct_sentences = sent_tokenize(correct_answer)
        incorrect_sentences = sent_tokenize(incorrect_answer)
        
        # Get embeddings for correct and incorrect answers
        correct_embeddings = sentence_model.encode(correct_sentences)
        incorrect_embeddings = sentence_model.encode(incorrect_sentences)
        
        # Compute contradiction score
        contradiction_score = compute_contradiction_score(correct_embeddings, incorrect_embeddings)
        
        # Add correct sample
        samples.append({
            "input": prompt,
            "response": correct_answer,
            "factuality_score": 1.0,
            "contradiction_score": 0.0,  # Correct answer → low contradiction
            "is_correct": True
        })
        
        # Add incorrect sample
        samples.append({
            "input": prompt,
            "response": incorrect_answer,
            "factuality_score": 0.0,
            "contradiction_score": contradiction_score,
            "is_correct": False
        })
    
    # Save to CSV and JSON for training
    df = pd.DataFrame(samples)
    df.to_csv('/workspace/data/bpft/bpft_training_data.csv', index=False)
    
    with open('/workspace/data/bpft/bpft_training_data.json', 'w') as f:
        json.dump(samples, f, indent=4)
    
    print(f"Dataset created with {len(samples)} samples")
    return samples

def main():
    """Main entry point to generate BPFT training data"""
    tokenizer, sentence_model = init_models()
    truthful_qa = load_factual_datasets()
    generate_contradictory_samples(tokenizer, sentence_model, truthful_qa)

if __name__ == "__main__":
    main()

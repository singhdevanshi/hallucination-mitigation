import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import numpy as np

os.makedirs('/workspace/data/bpft', exist_ok=True)

nltk.download('punkt')

def init_models():
    """Initialize models and tokenizers"""
    print("Loading models...")
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return tokenizer, sentence_model

def load_factual_datasets():
    """Load factual QA datasets for training"""
    print("Loading datasets...")
    
    truthful_qa = load_dataset("truthful_qa", "multiple_choice")
    
    
    return truthful_qa

def generate_contradictory_samples(tokenizer, sentence_model, dataset, num_samples=1000):
    print("Generating training samples...")
    
    samples = []
    
    for i, item in tqdm(enumerate(dataset['validation']), total=min(num_samples, len(dataset['validation']))):
        if i >= num_samples:
            break
            
        question = item['question']
        
        correct_idx = item['mc1_targets'].index(1) if 1 in item['mc1_targets'] else 0
        correct_answer = item['mc1_choices'][correct_idx]
        
        incorrect_idx = [i for i, x in enumerate(item['mc1_targets']) if x == 0]
        if incorrect_idx:
            incorrect_answer = item['mc1_choices'][incorrect_idx[0]]
        else:
            continue
        
        prompt = f"Question: {question}\nAnswer: "
        
        correct_sentences = sent_tokenize(correct_answer)
        incorrect_sentences = sent_tokenize(incorrect_answer)
        
        factuality_score = 1.0  
        
        contradiction_score = 0.0
        if len(correct_sentences) > 1:
            correct_embeddings = sentence_model.encode(correct_sentences)
            similarities = []
            for i in range(len(correct_embeddings)):
                for j in range(i+1, len(correct_embeddings)):
                    sim = torch.nn.functional.cosine_similarity(
                        torch.tensor(correct_embeddings[i]).unsqueeze(0),
                        torch.tensor(correct_embeddings[j]).unsqueeze(0)
                    )
                    similarities.append(sim.item())
            
            if similarities:
                # Lower similarity indicates higher contradiction
                contradiction_score = 1.0 - (sum(similarities) / len(similarities))
        
        samples.append({
            "input": prompt,
            "response": correct_answer,
            "factuality_score": factuality_score,
            "contradiction_score": 0.0,  # Lower is better
            "is_correct": True
        })
        
        samples.append({
            "input": prompt,
            "response": incorrect_answer,
            "factuality_score": 0.0, 
            "contradiction_score": 1.0,  
            "is_correct": False
        })
    
    df = pd.DataFrame(samples)
    df.to_csv('/workspace/data/bpft/bpft_training_data.csv', index=False)
    
    with open('/workspace/data/bpft/bpft_training_data.json', 'w') as f:
        json.dump(samples, f)
    
    print(f"Dataset created with {len(samples)} samples")
    return samples

def main():
    tokenizer, sentence_model = init_models()
    truthful_qa = load_factual_datasets()
    generate_contradictory_samples(tokenizer, sentence_model, truthful_qa)

if __name__ == "__main__":
    main()
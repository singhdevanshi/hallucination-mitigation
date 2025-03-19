import os
import json
import torch
import numpy as np
import requests
from datasets import load_dataset
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer
import nltk

nltk.download('punkt')

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Create directory
os.makedirs('/workspace/data/isc', exist_ok=True)

def ollama_generate(prompt, model="mistral"):
    """Generate response from Ollama using the specified model."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return json.loads(response.text)["response"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

class EmbeddingExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize Sentence Transformer for extracting embeddings"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Sentence Transformer model {model_name} for embeddings...")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
    
    def extract_embeddings(self, text):
        """Extract embeddings from text"""
        return self.model.encode(text, convert_to_tensor=True).cpu().numpy()

def prepare_dataset():
    """Prepare dataset for hidden state extraction"""
    print("Loading dataset...")

    # Load TruthfulQA for factual vs non-factual statements
    truthful_qa = load_dataset("truthful_qa", "multiple_choice")

    extraction_samples = []

    # Process validation set
    for i, item in enumerate(truthful_qa['validation']):
        if i >= 200:  # Limit to 200 samples
            break

        question = item['question']

        # Get correct (factual) answer
        correct_idx = item['mc1_targets'].index(1) if 1 in item['mc1_targets'] else 0
        correct_answer = item['mc1_choices'][correct_idx]

        # Get incorrect (non-factual) answer
        incorrect_idx = [i for i, x in enumerate(item['mc1_targets']) if x == 0]
        if incorrect_idx:
            incorrect_answer = item['mc1_choices'][incorrect_idx[0]]
        else:
            continue

        # Create input prompts
        factual_prompt = f"Question: {question}\nAnswer: {correct_answer}"
        non_factual_prompt = f"Question: {question}\nAnswer: {incorrect_answer}"

        extraction_samples.append({
            "factual": factual_prompt,
            "non_factual": non_factual_prompt
        })

    # Save extraction samples
    with open('/workspace/data/isc/extraction_samples.json', 'w') as f:
        json.dump(extraction_samples, f)

    return extraction_samples

def extract_and_save_embeddings(samples):
    """Extract embeddings and save them"""
    print("Extracting embeddings...")

    extractor = EmbeddingExtractor()
    factual_embeddings = []
    non_factual_embeddings = []

    for sample in tqdm(samples):
        # Process factual sample
        factual_response = ollama_generate(sample["factual"], model="mistral")
        factual_embedding = extractor.extract_embeddings(factual_response)
        factual_embeddings.append({
            "prompt": sample["factual"],
            "response": factual_response,
            "embedding": factual_embedding
        })

        # Process non-factual sample
        non_factual_response = ollama_generate(sample["non_factual"], model="mistral")
        non_factual_embedding = extractor.extract_embeddings(non_factual_response)
        non_factual_embeddings.append({
            "prompt": sample["non_factual"],
            "response": non_factual_response,
            "embedding": non_factual_embedding
        })

    # Save embeddings (using pickle for efficiency with large arrays)
    with open('/workspace/data/isc/factual_embeddings.pkl', 'wb') as f:
        pickle.dump(factual_embeddings, f)

    with open('/workspace/data/isc/non_factual_embeddings.pkl', 'wb') as f:
        pickle.dump(non_factual_embeddings, f)

    print("Embeddings extracted and saved.")

    # Save a small sample for quick testing
    with open('/workspace/data/isc/factual_embeddings_sample.pkl', 'wb') as f:
        pickle.dump(factual_embeddings[:10], f)

    with open('/workspace/data/isc/non_factual_embeddings_sample.pkl', 'wb') as f:
        pickle.dump(non_factual_embeddings[:10], f)

    print("Sample embeddings saved for quick testing.")

def main():
    samples = prepare_dataset()
    extract_and_save_embeddings(samples)

if __name__ == "__main__":
    main()

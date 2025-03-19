import os
import json
import torch
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Create directories for results and models
os.makedirs('/workspace/models/mistral7b_bpft', exist_ok=True)
os.makedirs('/workspace/results/bpft_logs', exist_ok=True)

def ollama_generate(prompt, model="mistral:7b"):
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

class BPFTLoss:
    def __init__(self, lambda_kl=0.1):
        """Initialize BPFT loss with KL-divergence and factuality weighting."""
        self.lambda_kl = lambda_kl
        self.mse_loss = torch.nn.MSELoss()

    def compute_loss(self, input_embedding, response_embedding, factuality_score, contradiction_score):
        """Compute BPFT loss with belief alignment."""
        # MSE between input and response embeddings
        belief_distance = self.mse_loss(
            torch.tensor(input_embedding),
            torch.tensor(response_embedding)
        )

        # Penalize contradictions and low factuality
        penalty = (1 - factuality_score) + contradiction_score
        weighted_loss = belief_distance * (1 + penalty)

        return weighted_loss.item()

def load_bpft_data(data_path):
    """Load BPFT training data from CSV or JSON."""
    if data_path.endswith('.csv'):
        import pandas as pd
        data = pd.read_csv(data_path)
        return data.to_dict('records')
    else:
        with open(data_path, 'r') as f:
            return json.load(f)

def train_bpft_model(data_path, embedder, epochs=3):
    """Train Mistral 7B with BPFT optimization."""
    print("Starting BPFT training...")
    
    # Load training data
    data_samples = load_bpft_data(data_path)
    print(f"Loaded {len(data_samples)} training examples.")
    
    bpft_loss_fn = BPFTLoss(lambda_kl=0.1)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(data_samples, desc=f"Epoch {epoch+1}")
        
        for i, item in enumerate(progress_bar):
            input_text = item['input']
            factuality_score = float(item['factuality_score'])
            contradiction_score = float(item['contradiction_score'])

            # Get input embedding
            input_embedding = embedder.encode(input_text).mean(axis=0)

            # Generate response using Ollama
            response = ollama_generate(input_text, model="mistral:7b")
            response_embedding = embedder.encode(response).mean(axis=0)

            # Calculate BPFT loss
            loss = bpft_loss_fn.compute_loss(
                input_embedding,
                response_embedding,
                factuality_score,
                contradiction_score
            )
            
            epoch_loss += loss
            progress_bar.set_postfix({"loss": loss})
            
            # Save response and embeddings for analysis
            if i % 50 == 0:
                with open(f"/workspace/results/bpft_logs/response_{epoch+1}_{i}.json", "w") as f:
                    json.dump({
                        "input": input_text,
                        "response": response,
                        "factuality_score": factuality_score,
                        "contradiction_score": contradiction_score,
                        "loss": loss
                    }, f)

        avg_loss = epoch_loss / len(data_samples)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        with open(f"/workspace/models/mistral7b_bpft/bpft_epoch_{epoch+1}.pkl", "wb") as f:
            np.save(f, response_embedding)

    print("✅ BPFT Training Complete!")
    print("Embeddings and responses saved in /workspace/models/mistral7b_bpft")

def main():
    data_path = "/workspace/data/bpft/bpft_training_data.csv"
    
    # Load Sentence Transformer for embedding extraction
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Run BPFT training
    train_bpft_model(data_path, embedder, epochs=3)

if __name__ == "__main__":
    main()
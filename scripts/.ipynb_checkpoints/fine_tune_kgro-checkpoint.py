import os
import json
import torch
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from retrieve_external_knowledge import retrieve_knowledge

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Create directory for saving fine-tuned embeddings
os.makedirs('/workspace/models/mistral7b_kgro_finetuned', exist_ok=True)

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

class KGROLoss:
    def __init__(self, alpha=0.7):
        """Initialize KGRO Loss with dynamic weighting."""
        self.alpha = alpha
        self.cos_sim = torch.nn.CosineSimilarity(dim=0)

    def compute_loss(self, knowledge_embedding, response_embedding):
        """Compute KGRO loss using cosine similarity."""
        knowledge_consistency = self.cos_sim(
            torch.tensor(knowledge_embedding),
            torch.tensor(response_embedding)
        ).item()

        # Maximize consistency, minimize cosine distance
        kgro_loss = 1 - knowledge_consistency
        weighted_loss = self.alpha * kgro_loss
        return weighted_loss

def fine_tune_with_kgro(train_texts, knowledge_texts, embedder, epochs=3):
    """Fine-tune Mistral 7B with KGRO optimization."""
    print("Starting KGRO fine-tuning...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for i, text in enumerate(train_texts):
            # Retrieve relevant knowledge
            retrieved_docs = retrieve_knowledge(text)
            knowledge_text = " ".join(retrieved_docs)
            
            # Get embeddings
            knowledge_embedding = embedder.encode(knowledge_text).mean(axis=0)
            
            # Generate model response using Ollama
            response = ollama_generate(text, model="mistral")
            response_embedding = embedder.encode(response).mean(axis=0)
            
            # Calculate KGRO loss
            kgro_loss = KGROLoss()
            loss = kgro_loss.compute_loss(knowledge_embedding, response_embedding)
            
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_texts)
        print(f"Epoch [{epoch + 1}/{epochs}], Average KGRO Loss: {avg_loss:.4f}")

    # Save optimized embeddings for later use
    with open("/workspace/models/mistral7b_kgro_finetuned/knowledge_embeddings.pkl", "wb") as f:
        np.save(f, knowledge_embedding)

    with open("/workspace/models/mistral7b_kgro_finetuned/response_embeddings.pkl", "wb") as f:
        np.save(f, response_embedding)

    print("✅ KGRO Fine-Tuning Complete!")
    print("Embeddings saved to /workspace/models/mistral7b_kgro_finetuned")

def main():
    train_texts = [
        "The Eiffel Tower was constructed in 1889.",
        "Isaac Newton discovered gravity in 1687.",
    ]

    # Load Sentence Transformer for embedding extraction
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Dummy knowledge texts (used for initial retrieval simulation)
    knowledge_texts = [
        "The Eiffel Tower is located in Paris and was built in 1889.",
        "Isaac Newton formulated the laws of motion and universal gravitation."
    ]
    
    fine_tune_with_kgro(train_texts, knowledge_texts, embedder)

if __name__ == "__main__":
    main()

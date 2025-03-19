import os
import torch
import requests
import json
from retrieve_external_knowledge import retrieve_knowledge
from sentence_transformers import SentenceTransformer

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Load Sentence-BERT for embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

def ollama_generate(prompt, model="mistral:7b", max_length=100, temperature=0.7):
    """Generate response using Ollama API with knowledge-enhanced prompt."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "max_tokens": max_length}
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return json.loads(response.text)["response"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def optimize_response(query, max_length=100, top_k=3, model="mistral:7b"):
    """Optimize response by integrating retrieved knowledge using KGRO."""
    try:
        # Retrieve relevant external knowledge
        retrieved_docs = retrieve_knowledge(query, top_k)
        
        if not retrieved_docs:
            print("No relevant knowledge retrieved. Proceeding without additional context.")
            knowledge_context = ""
        else:
            knowledge_context = " ".join(retrieved_docs)
            print(f"Retrieved {len(retrieved_docs)} knowledge sources for context.")
        
        # Construct knowledge-enhanced prompt
        prompt = f"{query}\n[KNOWLEDGE]: {knowledge_context}"
        
        # Generate response using Ollama
        response = ollama_generate(prompt, model=model, max_length=max_length, temperature=0.7)
        return response
    
    except Exception as e:
        print(f"Error during response optimization: {e}")
        return "I'm sorry, I couldn't generate a response at the moment."

if __name__ == "__main__":
    sample_query = "Who invented the telephone?"
    optimized_response = optimize_response(sample_query, max_length=100, top_k=3, model="mistral:7b")
    print(f"Optimized Response: {optimized_response}")
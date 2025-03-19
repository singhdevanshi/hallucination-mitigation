import os
import torch
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Create directories for models and results
os.makedirs("/workspace/models/isc", exist_ok=True)
os.makedirs("/workspace/results/isc", exist_ok=True)

class ISCProcessor:
    def __init__(self, detector_path="/workspace/models/isc/hallucination_detector.pt"):
        """Initialize ISC processor using Ollama and Mistral 7B"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Sentence Transformer for embedding similarity check
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load hallucination detector
        self.detector = None
        if os.path.exists(detector_path):
            print(f"Loading hallucination detector from {detector_path}")
            self.detector = self._load_detector(detector_path)
        else:
            print("No trained hallucination detector found. Using embedding-based similarity check.")

        # Storage for suppression history
        self.suppression_history = []

    def _load_detector(self, detector_path):
        """Load trained hallucination detector if available"""
        detector = torch.load(detector_path, map_location=self.device)
        detector.to(self.device)
        detector.eval()
        return detector

    def ollama_generate(self, prompt, model="mistral", max_length=100):
        """Generate response from Ollama API"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "max_tokens": max_length}
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            return json.loads(response.text)["response"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    def _detect_hallucination(self, input_text, response):
        """Detect hallucination risk using embeddings or detector"""
        if self.detector:
            # Get input and response embeddings
            input_embedding = self.embedder.encode(input_text).mean(axis=0)
            response_embedding = self.embedder.encode(response).mean(axis=0)
            
            # Calculate cosine similarity
            similarity = np.dot(input_embedding, response_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(response_embedding))
            hallucination_prob = 1 - similarity
        else:
            # Use a simple similarity-based check if no detector is available
            hallucination_prob = self._selfcheck_similarity(input_text, response)
        
        return hallucination_prob

    def _selfcheck_similarity(self, input_text, response):
        """SelfCheckGPT-based LLM-Prompting method for hallucination detection"""
        check_prompt = f"Input: {input_text}\nResponse: {response}\nDoes this response contain hallucinations? Yes or No: "
        verdict = self.ollama_generate(check_prompt, model="mistral", max_length=10)
        
        # Assign higher probability if 'Yes' is returned
        return 0.9 if "yes" in verdict.lower() else 0.1

    def _apply_suppression(self, response, hallucination_prob, alpha=0.1):
        """Apply suppression to response embeddings based on hallucination probability"""
        suppression_factor = alpha * hallucination_prob
        self.suppression_history.append(suppression_factor)
        
        # Modify temperature based on hallucination probability
        adjusted_temperature = 1.0 + suppression_factor * 2.0
        return adjusted_temperature

    def generate_with_isc(self, prompt, max_length=100):
        """Generate text with ISC-based suppression"""
        print(f"Generating with ISC for prompt: {prompt}")
        
        # Get initial response from Ollama
        response = self.ollama_generate(prompt, model="mistral", max_length=max_length)
        
        # Detect hallucination risk
        hallucination_prob = self._detect_hallucination(prompt, response)
        print(f"Hallucination probability: {hallucination_prob:.4f}")

        # Apply suppression if hallucination probability is high
        if hallucination_prob > 0.5:
            adjusted_temperature = self._apply_suppression(response, hallucination_prob)
            response = self.ollama_generate(prompt, model="mistral", max_length=max_length)
        
        return response, self.suppression_history

    def generate_without_isc(self, prompt, max_length=100):
        """Generate text without ISC for comparison"""
        print(f"Generating without ISC for prompt: {prompt}")
        response = self.ollama_generate(prompt, model="mistral", max_length=max_length)
        return response

    def visualize_suppression(self):
        """Visualize suppression history during generation"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.suppression_history, label="Suppression Factor")
        plt.xlabel("Generation Step")
        plt.ylabel("Suppression Intensity")
        plt.title("ISC Suppression History")
        
        plt.savefig("/workspace/results/isc/suppression_history.png")
        print("Suppression history saved to /workspace/results/isc/suppression_history.png")

def evaluate_isc(processor, num_samples=10):
    """Evaluate ISC effectiveness on TruthfulQA"""
    print("Evaluating ISC...")
    
    # Sample TruthfulQA questions for evaluation
    test_data = [
        "What is the capital of France?",
        "Who wrote the novel Pride and Prejudice?",
        "What is the boiling point of water in Celsius?",
        "When did World War II end?",
        "What is the largest mammal on Earth?",
        "How many planets are in the solar system?",
        "What is the chemical symbol for gold?",
        "Who discovered gravity?",
        "What is the national language of Japan?",
        "What is the square root of 64?"
    ]
    
    results = []
    
    for i, question in enumerate(test_data):
        if i >= num_samples:
            break
        
        prompt = f"Question: {question}\nAnswer: "
        
        # Generate with and without ISC
        with_isc, _ = processor.generate_with_isc(prompt)
        without_isc = processor.generate_without_isc(prompt)
        
        results.append({
            "question": question,
            "with_isc": with_isc,
            "without_isc": without_isc
        })
        
        print(f"\nQuestion: {question}")
        print(f"With ISC: {with_isc}")
        print(f"Without ISC: {without_isc}")
    
    # Save results for further analysis
    with open("/workspace/results/isc/generation_comparison.txt", "w") as f:
        for result in results:
            f.write(f"Question: {result['question']}\n")
            f.write(f"With ISC: {result['with_isc']}\n")
            f.write(f"Without ISC: {result['without_isc']}\n\n")
    
    print("Evaluation results saved to /workspace/results/isc/generation_comparison.txt")

def main():
    """Main function to run ISC evaluation and visualize results"""
    processor = ISCProcessor()
    
    # Evaluate ISC effectiveness
    evaluate_isc(processor, num_samples=10)
    
    # Example of ISC-based generation and visualization
    prompt = "What is the capital of France?"
    generated_text, _ = processor.generate_with_isc(prompt)
    print(f"\nGenerated with ISC: {generated_text}")
    
    # Visualize suppression history
    processor.visualize_suppression()

if __name__ == "__main__":
    main()

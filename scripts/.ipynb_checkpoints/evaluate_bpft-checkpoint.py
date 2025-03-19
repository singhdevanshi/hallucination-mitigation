import os
import json
import torch
import pandas as pd
import numpy as np
import requests
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import matplotlib.pyplot as plt

nltk.download('punkt')

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

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

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    """Load Sentence Transformer for consistency evaluation"""
    print("Loading sentence transformer model for consistency evaluation...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_model.to(device)
    return sentence_model

def prepare_evaluation_dataset():
    """Prepare dataset for evaluation"""
    print("Preparing evaluation dataset...")

    # Load TruthfulQA for evaluation
    truthful_qa = load_dataset("truthful_qa", "multiple_choice")

    # Prepare evaluation samples
    eval_samples = []

    # Use validation set
    for item in truthful_qa['validation']:
        question = item['question']

        # Get correct (truthful) answer
        correct_idx = item['mc1_targets'].index(1) if 1 in item['mc1_targets'] else 0
        correct_answer = item['mc1_choices'][correct_idx]

        eval_samples.append({
            "question": question,
            "reference_answer": correct_answer
        })

    # Save evaluation dataset
    os.makedirs('/workspace/data/evaluation', exist_ok=True)
    with open('/workspace/data/evaluation/eval_samples.json', 'w') as f:
        json.dump(eval_samples, f)

    return eval_samples

def calculate_consistency_score(sentences, sentence_model):
    """Calculate internal consistency score for a response"""
    if len(sentences) <= 1:
        return 1.0  # Single sentence is consistent with itself

    # Calculate embeddings for all sentences
    embeddings = sentence_model.encode(sentences)

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = torch.nn.functional.cosine_similarity(
                torch.tensor(embeddings[i]).unsqueeze(0),
                torch.tensor(embeddings[j]).unsqueeze(0)
            )
            similarities.append(sim.item())

    # Return average similarity as consistency score
    return sum(similarities) / len(similarities) if similarities else 1.0

def generate_responses(question, model="mistral", num_samples=5):
    """Generate multiple responses using Ollama to calculate consistency"""
    responses = []
    prompt = f"Question: {question}\nAnswer: "

    for _ in range(num_samples):
        try:
            response = ollama_generate(prompt, model=model)
            response = response.replace(prompt, "").strip()
            responses.append(response)
        except Exception as e:
            print(f"Error generating response: {e}")
            responses.append("")

    return responses

def evaluate_model(eval_samples, sentence_model, model_name="base", model="mistral"):
    """Evaluate model's hallucination tendency using Ollama"""
    print(f"Evaluating {model_name} model using Ollama...")

    results = []

    for sample in tqdm(eval_samples[:50]):  # Limit to 50 samples for faster evaluation
        question = sample["question"]
        reference_answer = sample["reference_answer"]

        # Generate multiple responses
        responses = generate_responses(question, model=model)

        # Calculate metrics for each response
        sample_results = []
        for response in responses:
            # Split into sentences
            sentences = sent_tokenize(response)

            # Calculate consistency
            consistency_score = calculate_consistency_score(sentences, sentence_model)

            # Store results
            sample_results.append({
                "response": response,
                "num_sentences": len(sentences),
                "consistency_score": consistency_score
            })

        # Calculate average consistency
        avg_consistency = sum(r["consistency_score"] for r in sample_results) / len(sample_results)

        # Store sample results
        results.append({
            "question": question,
            "reference_answer": reference_answer,
            "responses": sample_results,
            "avg_consistency": avg_consistency
        })

    # Save evaluation results
    os.makedirs('/workspace/results', exist_ok=True)
    with open(f'/workspace/results/{model_name}_evaluation_results.json', 'w') as f:
        json.dump(results, f)

    # Calculate overall metrics
    overall_consistency = sum(r["avg_consistency"] for r in results) / len(results)

    print(f"Overall consistency score for {model_name}: {overall_consistency:.4f}")

    return results, overall_consistency

def plot_comparison(base_results, bpft_results=None):
    """Plot comparison of consistency scores"""

    # Extract consistency scores
    base_scores = [r["avg_consistency"] for r in base_results]

    if bpft_results:
        bpft_scores = [r["avg_consistency"] for r in bpft_results]

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot histograms
        plt.hist(base_scores, alpha=0.5, bins=10, label='Base Model')
        plt.hist(bpft_scores, alpha=0.5, bins=10, label='BPFT Model')

        plt.xlabel('Consistency Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Consistency Scores')
        plt.legend()

        # Save figure
        plt.savefig('/workspace/results/consistency_comparison.png')
        print("Comparison plot saved to /workspace/results/consistency_comparison.png")
    else:
        # Plot only base model
        plt.figure(figsize=(10, 6))
        plt.hist(base_scores, bins=10)
        plt.xlabel('Consistency Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Base Model Consistency Scores')
        plt.savefig('/workspace/results/base_consistency.png')
        print("Base model plot saved to /workspace/results/base_consistency.png")

def main():
    # Load models
    sentence_model = load_models()

    # Prepare evaluation dataset
    eval_samples = prepare_evaluation_dataset()

    # Evaluate base model using Mistral 7B via Ollama
    base_results, base_consistency = evaluate_model(
        eval_samples, sentence_model, "base", model="mistral"
    )

    # Evaluate BPFT fine-tuned model if available
    if os.path.exists('/workspace/models/mistral-7b-bpft-final'):
        print("Evaluating BPFT fine-tuned model with Ollama...")
        bpft_results, bpft_consistency = evaluate_model(
            eval_samples, sentence_model, "bpft", model="mistral"
        )

        # Plot comparison
        plot_comparison(base_results, bpft_results)

        print("\nEvaluation Summary:")
        print(f"Base Model Consistency: {base_consistency:.4f}")
        print(f"BPFT Model Consistency: {bpft_consistency:.4f}")
        print(f"Improvement: {(bpft_consistency - base_consistency) * 100:.2f}%")
    else:
        # Plot only base model results
        plot_comparison(base_results)

        print("\nEvaluation Summary:")
        print(f"Base Model Consistency: {base_consistency:.4f}")
        print("BPFT model not available for comparison")

if __name__ == "__main__":
    main()

# run_kgro_pipeline.py
import os
from fine_tune_kgro import fine_tune_with_kgro
from optimize_response_kgro import optimize_response

# Paths and dataset preparation
data_path = "/workspace/data/kgro_dataset.txt"
knowledge_path = "/workspace/data/knowledge_corpus.txt"

# Load training data
with open(data_path, "r") as f:
    train_texts = f.readlines()

# Fine-tune the model with KGRO
fine_tune_with_kgro(train_texts, knowledge_path, model="mistral:7b")  # Updated model name

# Optimize a sample query
test_query = "When was the first airplane flown?"
optimized_response = optimize_response(test_query, model="mistral:7b")  # Updated model name
print(f"KGRO Pipeline Complete. Optimized Response: {optimized_response}")
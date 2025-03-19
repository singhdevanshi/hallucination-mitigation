# run_isc_pipeline.py
import os
import numpy as np
from extract_hidden_states import extract_hidden_states
from train_isc_detector import train_model, ISCHallucinationDetector
from implement_isc import monitor_hallucination

# Define file paths
hidden_states_path = "/workspace/data/isc_hidden_states.npy"
labels_path = "/workspace/data/isc_labels.npy"

# Sample data for demonstration
sample_texts = [
    "The Great Wall of China can be seen from space.",
    "Water boils at 100 degrees Celsius under normal atmospheric pressure."
]
labels = [1, 0]  # 1: hallucinated, 0: factual

# Extract hidden states and save for training
hidden_states_data = []
for text in sample_texts:
    extracted_states = extract_hidden_states(text)
    flattened_states = np.mean(extracted_states["layer_16"], axis=1)
    hidden_states_data.append(flattened_states)

np.save(hidden_states_path, np.array(hidden_states_data))
np.save(labels_path, np.array(labels))

# Train ISC detector
input_dim = hidden_states_data[0].shape[1]
model = ISCHallucinationDetector(input_dim)
train_model(model, hidden_states_data, labels, hidden_states_data, labels)

# Monitor a new sample
test_text = "The moon is made of cheese."
hallucinated = monitor_hallucination(test_text)
print(f"ISC Pipeline Completed. Hallucination Detected: {hallucinated}")

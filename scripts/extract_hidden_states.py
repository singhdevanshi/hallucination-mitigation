import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pickle

# Create directory
os.makedirs('/workspace/data/isc', exist_ok=True)

class HiddenStateExtractor:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        """Initialize model to extract hidden states"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with hooks to extract hidden states
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,  # Optimize memory usage
            device_map="auto",
            torch_dtype=torch.float16,
            output_hidden_states=True  # Important: output all hidden states
        )
        
        # Target layers to extract (middle layers for Mistral 7B)
        # For Mistral, useful layers are often in the middle (e.g., 10-20 of 32 layers)
        self.target_layers = list(range(10, 21))
        print(f"Will extract hidden states from layers: {self.target_layers}")
        
        # Hook for capturing hidden states
        self.hidden_states = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture hidden states"""
        for name, module in self.model.named_modules():
            # Look for decoder layers in transformer architecture
            if "layers" in name and any(f".{layer}." in name for layer in self.target_layers):
                layer_num = int(name.split(".")[-2])  # Extract layer number
                if layer_num in self.target_layers:
                    # Register hook for this layer
                    module.register_forward_hook(self._get_hook(layer_num))
                    print(f"Registered hook for layer {layer_num}")
    
    def _get_hook(self, layer_num):
        """Create a hook function for a specific layer"""
        def hook(module, input, output):
            # Store hidden states for this layer
            self.hidden_states[layer_num] = output.detach().cpu()
        return hook
    
    def extract_hidden_states(self, inputs, factual=True):
        """Extract hidden states for inputs"""
        # Reset hidden states
        self.hidden_states = {}
        
        # Process inputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Collect hidden states for each layer
        layer_states = {}
        for layer_num in self.target_layers:
            if layer_num in self.hidden_states:
                layer_states[layer_num] = self.hidden_states[layer_num].numpy()
        
        return {
            "hidden_states": layer_states,
            "factual": factual
        }

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

def extract_and_save_states(samples):
    """Extract hidden states and save them"""
    print("Extracting hidden states...")
    
    extractor = HiddenStateExtractor()
    tokenizer = extractor.tokenizer
    
    factual_states = []
    non_factual_states = []
    
    for sample in tqdm(samples):
        # Process factual sample
        factual_inputs = tokenizer(sample["factual"], return_tensors="pt").to(extractor.device)
        factual_data = extractor.extract_hidden_states(factual_inputs, factual=True)
        factual_states.append(factual_data)
        
        # Process non-factual sample
        non_factual_inputs = tokenizer(sample["non_factual"], return_tensors="pt").to(extractor.device)
        non_factual_data = extractor.extract_hidden_states(non_factual_inputs, factual=False)
        non_factual_states.append(non_factual_data)
    
    # Save states (using pickle for efficiency with large arrays)
    with open('/workspace/data/isc/factual_states.pkl', 'wb') as f:
        pickle.dump(factual_states, f)
    
    with open('/workspace/data/isc/non_factual_states.pkl', 'wb') as f:
        pickle.dump(non_factual_states, f)
    
    print("Hidden states extracted and saved.")
    
    # Save a small sample for quick testing
    with open('/workspace/data/isc/factual_states_sample.pkl', 'wb') as f:
        pickle.dump(factual_states[:10], f)
    
    with open('/workspace/data/isc/non_factual_states_sample.pkl', 'wb') as f:
        pickle.dump(non_factual_states[:10], f)
    
    print("Sample states saved for quick testing.")

def main():
    samples = prepare_dataset()
    extract_and_save_states(samples)

if __name__ == "__main__":
    main()
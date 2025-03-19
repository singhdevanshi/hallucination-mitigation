import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# Create directory
os.makedirs('/workspace/models/isc', exist_ok=True)

class HallucinationDetector(nn.Module):
    def __init__(self, input_dim=10240, hidden_dim=512):
        """Neural network for hallucination detection"""
        super(HallucinationDetector, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).squeeze()

class ISCProcessor:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", detector_path="/workspace/models/isc/hallucination_detector.pt"):
        """Initialize ISC processor"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with output_hidden_states=True
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            output_hidden_states=True
        )
        
        # Target layers to monitor
        self.target_layers = list(range(10, 21))
        print(f"Will monitor hidden states from layers: {self.target_layers}")
        
        # Load hallucination detector if available
        self.detector = None
        if os.path.exists(detector_path):
            print(f"Loading hallucination detector from {detector_path}")
            # First, create the detector with a placeholder input dimension
            self.detector = HallucinationDetector(input_dim=10240)
            
            # Load the state dict
            detector_state = torch.load(detector_path, map_location=self.device)
            self.detector.load_state_dict(detector_state)
            self.detector.to(self.device)
            self.detector.eval()
            print("Hallucination detector loaded successfully")
        else:
            print(f"Detector not found at {detector_path}. Will run without detector.")
        
        # Register hooks for hidden states
        self.hidden_states = {}
        self._register_hooks()
        
        # Storage for suppression history
        self.suppression_history = []
    
    def _register_hooks(self):
        """Register forward hooks to capture hidden states"""
        for name, module in self.model.named_modules():
            # Look for decoder layers
            if "layers" in name and any(f".{layer}." in name for layer in self.target_layers):
                layer_num = int(name.split(".")[-2])
                if layer_num in self.target_layers:
                    # Register hook for this layer
                    module.register_forward_hook(self._get_hook(layer_num))
    
    def _get_hook(self, layer_num):
        """Create a hook function for a specific layer"""
        def hook(module, input, output):
            # Store hidden states for this layer
            self.hidden_states[layer_num] = output.detach()
        return hook
    
    def _extract_features(self, hidden_states):
        """Extract feature vector from hidden states"""
        # Concatenate statistics from all layers
        features = []
        
        for layer_num, layer_state in hidden_states.items():
            # Get hidden state tensor
            hidden = layer_state.cpu().numpy()
            
            # Calculate statistics across the sequence dimension
            mean_hidden = np.mean(hidden, axis=1)  # Mean across sequence
            std_hidden = np.std(hidden, axis=1)    # Standard deviation
            max_hidden = np.max(hidden, axis=1)    # Max values
            
            # Flatten statistics
            flat_mean = mean_hidden.flatten()
            flat_std = std_hidden.flatten()
            flat_max = max_hidden.flatten()
            
            # Concatenate statistics
            layer_features = np.concatenate([flat_mean, flat_std, flat_max])
            
            # Downsample if needed (for memory efficiency)
            if len(layer_features) > 1024:
                indices = np.linspace(0, len(layer_features) - 1, 1024, dtype=int)
                layer_features = layer_features[indices]
            
            features.append(layer_features)
        
        # Concatenate all layer features
        all_features = np.concatenate(features)
        
        # Ensure consistent size
        if len(all_features) > 10240:  # Limit feature size
            indices = np.linspace(0, len(all_features) - 1, 10240, dtype=int)
            all_features = all_features[indices]
        
        return all_features
    
    def _detect_hallucination(self):
        """Detect hallucination risk based on hidden states"""
        if not self.detector:
            print("No detector available. Skipping hallucination detection.")
            return 0.0
        
        # Extract features from hidden states
        features = self._extract_features(self.hidden_states)
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Get hallucination probability
        with torch.no_grad():
            hallucination_prob = self.detector(features_tensor).item()
        
        return hallucination_prob
    
    def _apply_suppression(self, logits, hallucination_prob, alpha=0.1):
        """Apply suppression to logits based on hallucination probability"""
        # Higher hallucination probability = more suppression
        suppression_factor = alpha * hallucination_prob
        
        # Store suppression history
        self.suppression_history.append(suppression_factor)
        
        # Apply temperature scaling (increase temperature to decrease confidence)
        temperature = 1.0 + suppression_factor * 2.0
        
        # Apply temperature scaling to logits
        scaled_logits = logits / temperature
        
        return scaled_logits
    
    def generate_with_isc(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """Generate text with ISC-based hallucination suppression"""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Reset suppression history
        self.suppression_history = []
        
        # Generate with ISC
        generated_ids = []
        input_ids = inputs.input_ids.clone()
        
        for _ in tqdm(range(max_length), desc="Generating with ISC"):
            # Get model output
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            
            # Get logits for next token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Detect hallucination risk
            hallucination_prob = self._detect_hallucination()
            
            # Apply suppression if hallucination detected
            if hallucination_prob > 0.5:  # Threshold for suppression
                next_token_logits = self._apply_suppression(next_token_logits, hallucination_prob)
            
            # Apply temperature and top-p sampling
            next_token_logits = next_token_logits / temperature
            
            # Remove low probability tokens
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample next token
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add token to generated sequence
            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text, self.suppression_history
    
    def generate_without_isc(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """Generate text without ISC for comparison"""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate without ISC
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=len(inputs.input_ids[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text.replace(prompt, "")
        
        return generated_text
    
    def visualize_suppression(self):
        """Visualize suppression history"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.suppression_history)
        plt.xlabel('Generation Step')
        plt.ylabel('Suppression Factor')
        plt.title('Hallucination Suppression During Generation')
        
        # Save the plot
        os.makedirs('/workspace/results/isc', exist_ok=True)
        plt.savefig('/workspace/results/isc/suppression_history.png')
        print("Suppression history saved to /workspace/results/isc/suppression_history.png")

def evaluate_isc(processor, num_samples=10):
    """Evaluate ISC on a small test set"""
    print("Evaluating ISC...")
    
    # Load TruthfulQA for evaluation
    truthful_qa = load_dataset("truthful_qa", "multiple_choice")
    
    results = []
    
    for i, item in enumerate(truthful_qa['validation']):
        if i >= num_samples:
            break
        
        question = item['question']
        
        # Generate with and without ISC
        prompt = f"Question: {question}\nAnswer: "
        
        with_isc, _ = processor.generate_with_isc(prompt)
        without_isc = processor.generate_without_isc(prompt)
        
        # Store results
        results.append({
            "question": question,
            "with_isc": with_isc,
            "without_isc": without_isc
        })
        
        print(f"\nQuestion: {question}")
        print(f"With ISC: {with_isc}")
        print(f"Without ISC: {without_isc}")
    
    # Save results
    os.makedirs('/workspace/results/isc', exist_ok=True)
    with open('/workspace/results/isc/generation_comparison.txt', 'w') as f:
        for result in results:
            f.write(f"Question: {result['question']}\n")
            f.write(f"With ISC: {result['with_isc']}\n")
            f.write(f"Without ISC: {result['without_isc']}\n\n")
    
    print("Evaluation results saved to /workspace/results/isc/generation_comparison.txt")

def main():
    # Initialize ISC processor
    processor = ISCProcessor()
    
    # Evaluate ISC
    evaluate_isc(processor)
    
    # Example of generating with ISC and visualizing suppression
    prompt = "What is the capital of France?"
    generated_text, _ = processor.generate_with_isc(prompt)
    print(f"\nGenerated with ISC: {generated_text}")
    
    # Visualize suppression history
    processor.visualize_suppression()

if __name__ == "__main__":
    main()
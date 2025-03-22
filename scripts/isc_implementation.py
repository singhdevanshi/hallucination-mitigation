# ISC: Internal State Calibration Implementation
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from torch import nn

# Setup constants
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./llama-8b-isc"
LEARNING_RATE = 3e-5
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 2
SUPPRESSION_FACTOR = 0.2  # Alpha value for hallucination suppression

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
tokenizer.pad_token = tokenizer.eos_token

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.environ.get("HF_TOKEN")
)

# Define ISC model wrapper that modifies hidden states
class ISCModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        
        # Create a hallucination detector for middle layers
        hidden_size = self.model.config.hidden_size
        self.hallucination_detector = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Define which layers to monitor (middle transformer layers)
        # For Mixtral, we'll focus on the middle layers
        num_layers = len(self.model.model.layers)
        self.monitored_layers = list(range(num_layers // 3, 2 * num_layers // 3))
        print(f"Monitoring layers: {self.monitored_layers}")
    
    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=True):
        # Get the original model outputs with hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        if self.training:
            # During training, we don't modify the outputs, just compute the hallucination scores
            # for the loss function to use
            return outputs
        
        # During inference, modify the hidden states to suppress hallucinations
        hidden_states = outputs.hidden_states
        
        # Only apply ISC during generation, not for the entire sequence
        if labels is None:
            modified_hidden_states = []
            for layer_idx, layer_hidden_state in enumerate(hidden_states):
                if layer_idx in self.monitored_layers:
                    # Apply hallucination detector
                    hallucination_scores = self.hallucination_detector(layer_hidden_state)
                    
                    # Create suppression mask based on hallucination scores
                    suppression = SUPPRESSION_FACTOR * hallucination_scores
                    
                    # Apply suppression to hidden states
                    modified_state = layer_hidden_state - suppression * layer_hidden_state
                    modified_hidden_states.append(modified_state)
                else:
                    modified_hidden_states.append(layer_hidden_state)
            
            # Replace hidden states in outputs
            outputs.hidden_states = tuple(modified_hidden_states)
        
        return outputs

# Load dataset with factual and hallucinated examples
# For ISC, we need examples of factual statements and hallucinated statements
# You should replace this with your specific dataset
def load_hallucination_dataset():
    # This is a placeholder - you should use your actual dataset
    # For example, you might use a combination of:
    # - Factual QA datasets (like TruthfulQA)
    # - Known hallucination examples from your baseline evaluations
    
    # For demonstration, let's use TruthfulQA again
    truthful_qa = load_dataset("truthful_qa", "multiple_choice")
    
    factual_examples = []
    hallucinated_examples = []
    
    for example in truthful_qa['validation']:
        question = example['question']
        
        # Get a factual answer (correct according to dataset)
        correct_idx = example['mc1_targets']['labels'].index(1)
        factual_answer = example['mc1_targets']['choices'][correct_idx]
        
        # Get a hallucinated answer (incorrect according to dataset)
        incorrect_indices = [j for j, label in enumerate(example['mc1_targets']['labels']) if label == 0]
        if incorrect_indices:
            incorrect_idx = incorrect_indices[0]
            hallucinated_answer = example['mc1_targets']['choices'][incorrect_idx]
            
            factual_examples.append({
                "text": f"Question: {question}\nAnswer: {factual_answer}",
                "is_hallucination": 0
            })
            
            hallucinated_examples.append({
                "text": f"Question: {question}\nAnswer: {hallucinated_answer}",
                "is_hallucination": 1
            })
    
    # Combine and shuffle
    all_examples = factual_examples + hallucinated_examples
    np.random.shuffle(all_examples)
    
    return all_examples

# Prepare dataset
hallucination_dataset = load_hallucination_dataset()
print(f"Loaded {len(hallucination_dataset)} examples for ISC training")

# Create ISC dataset class
class ISCDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example["text"]
        is_hallucination = example["is_hallucination"]
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels for language modeling
        labels = encoding["input_ids"].clone()
        
        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": labels[0],
            "is_hallucination": torch.tensor(is_hallucination, dtype=torch.float)
        }

# Create ISC dataset
isc_dataset = ISCDataset(hallucination_dataset, tokenizer)

# Wrap the base model with ISC wrapper
isc_model = ISCModel(base_model)

# Define custom trainer for ISC
class ISCTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        is_hallucination = inputs["is_hallucination"]
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        # Standard language modeling loss
        lm_loss = outputs.loss
        
        # Extract hidden states from the monitored layers
        hidden_states = outputs.hidden_states
        
        # Calculate hallucination detection loss
        hallucination_loss = 0
        for layer_idx in model.module.monitored_layers:
            layer_hidden_state = hidden_states[layer_idx]
            
            # Use the last token's hidden state for hallucination detection
            last_token_idx = (inputs["attention_mask"].sum(dim=1) - 1).to(torch.long)
            batch_indices = torch.arange(layer_hidden_state.size(0), device=layer_hidden_state.device)
            last_hidden_states = layer_hidden_state[batch_indices, last_token_idx]
            
            # Get hallucination predictions
            hallucination_preds = model.module.hallucination_detector(last_hidden_states).squeeze(-1)
            
            # Binary cross-entropy loss for hallucination detection
            bce_loss = nn.BCELoss()
            hallucination_loss += bce_loss(hallucination_preds, is_hallucination)
        
        # Average hallucination loss across monitored layers
        hallucination_loss /= len(model.module.monitored_layers)
        
        # Combine losses
        total_loss = lm_loss + hallucination_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

# Configure training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    report_to="tensorboard",
)

# Create ISC trainer
trainer = ISCTrainer(
    model=isc_model,
    args=training_args,
    train_dataset=isc_dataset,
    tokenizer=tokenizer,
)

# Train model
print("Starting ISC training...")
trainer.train()

# Save the model
isc_model.model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
torch.save(isc_model.hallucination_detector.state_dict(), f"{OUTPUT_DIR}/final/hallucination_detector.pt")
print(f"Model saved to {OUTPUT_DIR}/final")

# Define a helper function for inference with ISC
def generate_with_isc(prompt, max_new_tokens=100):
    # Load model with ISC
    loaded_model = AutoModelForCausalLM.from_pretrained(
        f"{OUTPUT_DIR}/final",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    loaded_model_isc = ISCModel(loaded_model)
    
    # Load hallucination detector weights
    detector_path = f"{OUTPUT_DIR}/final/hallucination_detector.pt"
    if os.path.exists(detector_path):
        loaded_model_isc.hallucination_detector.load_state_dict(
            torch.load(detector_path, map_location=loaded_model.device)
        )
    
    # Generate text with ISC
    inputs = tokenizer(prompt, return_tensors="pt").to(loaded_model.device)
    outputs = loaded_model_isc.model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model with a sample prompt
test_prompt = "What is the capital of France?"
print(f"Prompt: {test_prompt}")
print(f"Response: {generate_with_isc(test_prompt)}")
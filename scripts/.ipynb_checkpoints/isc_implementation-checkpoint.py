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
import gc

# Setup constants
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./llama-8b-isc"
LEARNING_RATE = 3e-5
BATCH_SIZE = 1  # Reduced batch size
GRADIENT_ACCUMULATION_STEPS = 16  # Increased gradient accumulation
NUM_EPOCHS = 2
SUPPRESSION_FACTOR = 0.2  # Alpha value for hallucination suppression
MAX_LENGTH = 256  # Added max sequence length to reduce memory usage

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
tokenizer.pad_token = tokenizer.eos_token

# Load the base model with 8-bit quantization to reduce memory
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    load_in_8bit=True,  # Use 8-bit quantization instead of bfloat16
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
            nn.Linear(hidden_size, 64),  # Reduced hidden size
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Define which layers to monitor (just a few middle layers to save memory)
        num_layers = len(self.model.model.layers)
        # Monitor fewer layers - just 3 in the middle
        self.monitored_layers = [num_layers // 2 - 1, num_layers // 2, num_layers // 2 + 1]
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
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model."""
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the base model."""
        self.model.gradient_checkpointing_disable()
    
    def is_gradient_checkpointing(self):
        """Check if gradient checkpointing is enabled."""
        return self.model.is_gradient_checkpointing()

# Load dataset with factual and hallucinated examples
def load_hallucination_dataset():
    # For demonstration, let's use TruthfulQA
    truthful_qa = load_dataset("truthful_qa", "multiple_choice")
    
    factual_examples = []
    hallucinated_examples = []
    
    # Limit examples to save memory
    max_examples = 500  # Limiting the dataset size
    
    for i, example in enumerate(truthful_qa['validation']):
        if i >= max_examples:
            break
            
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
    def __init__(self, examples, tokenizer, max_length=MAX_LENGTH):
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
        
        # Ensure is_hallucination is a float tensor
        is_hallucination = torch.tensor(float(is_hallucination), dtype=torch.float32)
        
        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": labels[0],
            "is_hallucination": is_hallucination
        }

# Create ISC dataset
isc_dataset = ISCDataset(hallucination_dataset, tokenizer)

# Create custom data collator for ISC
class ISCDataCollator:
    def __call__(self, features):
        batch = {}
        
        # Process input_ids, attention_mask, and labels
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])
        
        # Process is_hallucination - ensure it's properly stacked
        if "is_hallucination" in features[0]:
            batch["is_hallucination"] = torch.stack([f["is_hallucination"] for f in features])
        else:
            # If missing, create a default tensor of zeros
            batch["is_hallucination"] = torch.zeros(len(features), dtype=torch.float32)
        
        return batch

# Wrap the base model with ISC wrapper
isc_model = ISCModel(base_model)

# Define custom trainer for ISC
class ISCTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Create a copy of inputs and pop the is_hallucination field
        # before passing the rest to the model
        model_inputs = {k: v for k, v in inputs.items() if k != "is_hallucination"}
        is_hallucination = inputs.get("is_hallucination")
        
        if is_hallucination is None:
            # If missing, create a default tensor of zeros
            is_hallucination = torch.zeros(inputs["input_ids"].size(0), dtype=torch.float32, device=inputs["input_ids"].device)
        
        # Forward pass
        outputs = model(
            **model_inputs,
            output_hidden_states=True
        )
        
        # Standard language modeling loss
        lm_loss = outputs.loss
        
        # Extract hidden states from the monitored layers
        hidden_states = outputs.hidden_states
        
        # Get the actual model (it may be wrapped in DistributedDataParallel)
        if hasattr(model, "module"):
            actual_model = model.module
        else:
            actual_model = model
        
        # Calculate hallucination detection loss
        hallucination_loss = 0
        for layer_idx in actual_model.monitored_layers:
            layer_hidden_state = hidden_states[layer_idx]
            
            # Use the last token's hidden state for hallucination detection
            last_token_idx = (model_inputs["attention_mask"].sum(dim=1) - 1).to(torch.long)
            batch_indices = torch.arange(layer_hidden_state.size(0), device=layer_hidden_state.device)
            last_hidden_states = layer_hidden_state[batch_indices, last_token_idx]
            
            # Get hallucination predictions
            hallucination_preds = actual_model.hallucination_detector(last_hidden_states).squeeze(-1)
            
            # Binary cross-entropy loss for hallucination detection
            bce_loss = nn.BCELoss()
            hallucination_loss += bce_loss(hallucination_preds, is_hallucination)
        
        # Average hallucination loss across monitored layers
        hallucination_loss /= len(actual_model.monitored_layers)
        
        # Combine losses
        total_loss = lm_loss + hallucination_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step on a batch of inputs.
        Override to add explicit memory cleanup
        """
        loss = super().training_step(model, inputs)
        
        # Manual garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return loss

# Configure training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    optim="adamw_8bit",  # Use 8-bit Adam optimizer
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    report_to="tensorboard",
    # Memory optimizations
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    # Don't use FP16/BF16 with 8-bit quantization
    fp16=False,
    bf16=False,
)

# Create ISC trainer
trainer = ISCTrainer(
    model=isc_model,
    args=training_args,
    train_dataset=isc_dataset,
    data_collator=ISCDataCollator(),
)

# Train model
print("Starting ISC training...")
trainer.train()

# Save the model
print("Saving model...")
base_model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
torch.save(isc_model.hallucination_detector.state_dict(), f"{OUTPUT_DIR}/final/hallucination_detector.pt")
print(f"Model saved to {OUTPUT_DIR}/final")

# Define a helper function for inference with ISC
def generate_with_isc(prompt, max_new_tokens=100):
    # Load model with 8-bit quantization
    loaded_model = AutoModelForCausalLM.from_pretrained(
        f"{OUTPUT_DIR}/final",
        load_in_8bit=True,
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
if __name__ == "__main__":
    test_prompt = "What is the capital of France?"
    print(f"Prompt: {test_prompt}")
    print(f"Response: {generate_with_isc(test_prompt)}")
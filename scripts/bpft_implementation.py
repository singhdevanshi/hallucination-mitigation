# BPFT: Belief Propagation Fine-Tuning Implementation
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.nn import KLDivLoss
from torch.nn.functional import log_softmax, softmax

# Setup constants
MODEL_ID = "mistralai/Mixtral-8x7B-v0.1"
OUTPUT_DIR = "./mixtral-bpft"
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LAMBDA = 0.1  # Hyperparameter for belief alignment strength

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
tokenizer.pad_token = tokenizer.eos_token

# Initialize model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.environ.get("HF_TOKEN")
)

# Load factual dataset (example: TruthfulQA)
# You should replace this with your specific dataset
dataset = load_dataset("truthful_qa", "multiple_choice")
print(f"Loaded dataset with {len(dataset['validation'])} examples")

# Convert the dataset to the format required for BPFT
def preprocess_dataset(examples):
    # For each example, we need:
    # 1. Input prompt (question)
    # 2. Correct response
    # 3. Incorrect response (contradictory)
    
    prompts = []
    correct_responses = []
    incorrect_responses = []
    
    for i, example in enumerate(examples):
        question = example['question']
        correct_idx = example['mc1_targets']['labels'].index(1)
        correct_answer = example['mc1_targets']['choices'][correct_idx]
        
        # Find an incorrect answer
        incorrect_indices = [j for j, label in enumerate(example['mc1_targets']['labels']) if label == 0]
        if incorrect_indices:
            incorrect_idx = incorrect_indices[0]
            incorrect_answer = example['mc1_targets']['choices'][incorrect_idx]
        else:
            # Skip examples without incorrect answers
            continue
        
        prompts.append(f"Question: {question}\nAnswer:")
        correct_responses.append(correct_answer)
        incorrect_responses.append(incorrect_answer)
    
    return {
        "prompts": prompts,
        "correct_responses": correct_responses,
        "incorrect_responses": incorrect_responses
    }

# Apply preprocessing
train_data = preprocess_dataset(dataset['validation'])

# Create a custom dataset class for BPFT
class BPFTDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, correct_responses, incorrect_responses, tokenizer, max_length=512):
        self.prompts = prompts
        self.correct_responses = correct_responses
        self.incorrect_responses = incorrect_responses
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        correct = self.correct_responses[idx]
        incorrect = self.incorrect_responses[idx]
        
        # Tokenize prompt with correct response
        correct_input = f"{prompt} {correct}"
        correct_encoding = self.tokenizer(
            correct_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize prompt with incorrect response
        incorrect_input = f"{prompt} {incorrect}"
        incorrect_encoding = self.tokenizer(
            incorrect_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels for correct response (used for standard CE loss)
        labels = correct_encoding["input_ids"].clone()
        # Set prompt tokens to -100 so they don't contribute to loss
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        labels[0, :prompt_tokens] = -100
        
        return {
            "input_ids": correct_encoding["input_ids"][0],
            "attention_mask": correct_encoding["attention_mask"][0],
            "labels": labels[0],
            "incorrect_input_ids": incorrect_encoding["input_ids"][0],
            "incorrect_attention_mask": incorrect_encoding["attention_mask"][0],
        }

# Create BPFT dataset
bpft_dataset = BPFTDataset(
    train_data["prompts"],
    train_data["correct_responses"],
    train_data["incorrect_responses"],
    tokenizer
)

# Define custom loss function for BPFT
class BPFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        incorrect_input_ids = inputs["incorrect_input_ids"]
        incorrect_attention_mask = inputs["incorrect_attention_mask"]
        
        # Forward pass for correct response
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        correct_loss = outputs.loss
        
        # Forward pass for incorrect response (without labels)
        with torch.no_grad():
            baseline_outputs = model(
                input_ids=incorrect_input_ids,
                attention_mask=incorrect_attention_mask,
                output_hidden_states=True
            )
        
        # Calculate KL divergence loss to enforce belief alignment
        kl_loss = 0
        # Focus on output logits where we have valid labels (not -100)
        valid_pos = (labels != -100).nonzero(as_tuple=True)[0]
        
        if len(valid_pos) > 0:
            # Get logits from both outputs at valid positions
            correct_logits = outputs.logits[valid_pos]
            baseline_logits = baseline_outputs.logits[valid_pos]
            
            # Calculate KL divergence
            kl_div = KLDivLoss(reduction="batchmean")
            kl_loss = kl_div(
                log_softmax(correct_logits, dim=-1),
                softmax(baseline_logits, dim=-1)
            )
        
        # Combine losses
        total_loss = correct_loss + LAMBDA * kl_loss
        
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

# Create trainer
trainer = BPFTTrainer(
    model=model,
    args=training_args,
    train_dataset=bpft_dataset,
    tokenizer=tokenizer,
)

# Train model
print("Starting BPFT training...")
trainer.train()

# Save the model
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"Model saved to {OUTPUT_DIR}/final")

# Inference example
def generate_with_bpft(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model with a sample prompt
test_prompt = "What is the capital of France?"
print(f"Prompt: {test_prompt}")
print(f"Response: {generate_with_bpft(test_prompt)}")
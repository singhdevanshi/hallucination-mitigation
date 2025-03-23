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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from torch.nn import KLDivLoss
from torch.nn.functional import log_softmax, softmax
from dataclasses import dataclass
from typing import Dict, List, Union
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Setup constants
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./llama-8b-bpft"
LEARNING_RATE = 2e-5
BATCH_SIZE = 1  # Reduced to 1
GRADIENT_ACCUMULATION_STEPS = 16  # Increased to compensate for smaller batch size
NUM_EPOCHS = 3
LAMBDA = 0.1  # Hyperparameter for belief alignment strength
MAX_LENGTH = 128  # Further reduced to save memory

# Set PyTorch memory allocator configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
tokenizer.pad_token = tokenizer.eos_token

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# Initialize model with 8-bit quantization and CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    token=os.environ.get("HF_TOKEN"),
    offload_folder="offload",  # Enable CPU offloading
    offload_state_dict=True,  # Offload state dict to CPU
    torch_dtype=torch.float16
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA with reduced rank
lora_config = LoraConfig(
    r=8,  # Reduced rank from 16 to 8
    lora_alpha=16,  # Reduced alpha from 32 to 16
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target modules for LoRA
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Get PEFT model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable parameters

model.gradient_checkpointing_enable()  # Enable gradient checkpointing

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

# Add custom data collator
@dataclass
class BPFTDataCollator:
    tokenizer: AutoTokenizer
    max_length: int = 512
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate correct and incorrect inputs
        correct_inputs = []
        incorrect_inputs = []
        
        for feature in features:
            # Handle correct inputs
            correct_inputs.append({
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"]
            })
            
            # Handle incorrect inputs
            if "incorrect_input_ids" in feature and "incorrect_attention_mask" in feature:
                incorrect_inputs.append({
                    "input_ids": feature["incorrect_input_ids"],
                    "attention_mask": feature["incorrect_attention_mask"]
                })
            else:
                # If incorrect inputs are missing, use the correct inputs as a fallback
                incorrect_inputs.append({
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"]
                })
        
        # Collate correct inputs
        correct_batch = self.tokenizer.pad(
            correct_inputs,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Collate incorrect inputs
        incorrect_batch = self.tokenizer.pad(
            incorrect_inputs,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": correct_batch["input_ids"],
            "attention_mask": correct_batch["attention_mask"],
            "labels": correct_batch["input_ids"].clone(),
            "incorrect_input_ids": incorrect_batch["input_ids"],
            "incorrect_attention_mask": incorrect_batch["attention_mask"]
        }

# Define custom loss function for BPFT
class BPFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        incorrect_input_ids = inputs["incorrect_input_ids"]
        incorrect_attention_mask = inputs["incorrect_attention_mask"]
        
        # Forward pass for correct response with memory optimization
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False  # Disable KV cache to save memory
        )
        correct_loss = outputs.loss
        
        # Forward pass for incorrect response with memory optimization
        with torch.no_grad():
            baseline_outputs = model(
                input_ids=incorrect_input_ids,
                attention_mask=incorrect_attention_mask,
                output_hidden_states=True,
                use_cache=False  # Disable KV cache to save memory
            )
        
        # Calculate KL divergence loss to enforce belief alignment
        kl_loss = 0
        # Focus on output logits where we have valid labels (not -100)
        valid_pos = (labels != -100).nonzero(as_tuple=True)[0]
        
        if len(valid_pos) > 0:
            # Process logits in smaller chunks to save memory
            chunk_size = 2  # Further reduced chunk size
            num_chunks = (len(valid_pos) + chunk_size - 1) // chunk_size
            
            kl_div = KLDivLoss(reduction="batchmean")
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(valid_pos))
                chunk_pos = valid_pos[start_idx:end_idx]
                
                # Get logits for current chunk
                correct_chunk = outputs.logits[chunk_pos]
                baseline_chunk = baseline_outputs.logits[chunk_pos]
                
                # Calculate softmax and log_softmax in smaller steps to save memory
                with torch.no_grad():
                    # Process baseline chunk in smaller sub-chunks
                    baseline_probs = []
                    sub_chunk_size = 1  # Process one token at a time
                    for j in range(0, baseline_chunk.size(0), sub_chunk_size):
                        sub_chunk = baseline_chunk[j:j+sub_chunk_size]
                        sub_probs = softmax(sub_chunk, dim=-1)
                        baseline_probs.append(sub_probs)
                        del sub_chunk
                        torch.cuda.empty_cache()
                    baseline_probs = torch.cat(baseline_probs, dim=0)
                
                # Process correct chunk in smaller sub-chunks
                correct_log_probs = []
                for j in range(0, correct_chunk.size(0), sub_chunk_size):
                    sub_chunk = correct_chunk[j:j+sub_chunk_size]
                    sub_log_probs = log_softmax(sub_chunk, dim=-1)
                    correct_log_probs.append(sub_log_probs)
                    del sub_chunk
                    torch.cuda.empty_cache()
                correct_log_probs = torch.cat(correct_log_probs, dim=0)
                
                # Calculate KL divergence for chunk
                chunk_kl_loss = kl_div(correct_log_probs, baseline_probs)
                
                # Accumulate loss
                kl_loss += chunk_kl_loss
                
                # Clear chunk tensors
                del correct_chunk
                del baseline_chunk
                del baseline_probs
                del correct_log_probs
                torch.cuda.empty_cache()
            
            # Average KL loss across chunks
            kl_loss /= num_chunks
        
        # Combine losses
        total_loss = correct_loss + LAMBDA * kl_loss
        
        # Clear unnecessary tensors to free memory
        del outputs.logits
        del baseline_outputs.logits
        del outputs.hidden_states
        del baseline_outputs.hidden_states
        torch.cuda.empty_cache()
        
        return (total_loss, outputs) if return_outputs else total_loss

# Configure training arguments with memory optimizations
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
    gradient_checkpointing=True,  # Enable gradient checkpointing
    optim="adamw_torch_fused",  # Use fused optimizer for better memory efficiency
    max_grad_norm=0.5,  # Reduce gradient norm to save memory
    warmup_ratio=0.1,  # Add warmup to stabilize training
    lr_scheduler_type="cosine",  # Use cosine learning rate schedule
    dataloader_pin_memory=False,  # Disable pin memory to save RAM
    dataloader_num_workers=0,  # Disable multiprocessing to save memory
    deepspeed="ds_config.json",  # Enable DeepSpeed for memory optimization
    remove_unused_columns=True,  # Remove unused columns to save memory
    group_by_length=True,  # Group similar length sequences together
    length_column_name="length",  # Column name for sequence length
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Use non-reentrant checkpointing
)

# Create trainer
trainer = BPFTTrainer(
    model=model,
    args=training_args,
    train_dataset=bpft_dataset,
    data_collator=BPFTDataCollator(tokenizer=tokenizer),
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
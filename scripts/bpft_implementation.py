import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from torch.nn import KLDivLoss
from torch.nn.functional import log_softmax, softmax
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Constants
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./llama-8b-bpft"
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
NUM_EPOCHS = 3
LAMBDA = 0.1
MAX_LENGTH = 512

# Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization with double quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Model initialization with quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    token=os.environ.get("HF_TOKEN"),
    offload_folder="offload",
    offload_state_dict=True,
    torch_dtype=torch.float16
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration with corrected parameters
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
dataset = load_dataset("truthful_qa", "multiple_choice")
print(f"Loaded dataset with {len(dataset['validation'])} examples")

# Improved dataset preprocessing
def preprocess_dataset(examples):
    processed_data = {
        "prompts": [],
        "correct_responses": [],
        "incorrect_responses": []
    }
    
    print(f"Processing {len(examples)} examples...")
    
    for i, example in enumerate(examples):
        question = example['question']
        
        # Find correct answer (label = 1)
        correct_indices = [i for i, label in enumerate(example['mc1_targets']['labels']) if label == 1]
        if not correct_indices:
            print(f"Example {i}: No correct answer found")
            continue
        correct_idx = correct_indices[0]
        correct_answer = example['mc1_targets']['choices'][correct_idx]
        
        # Find incorrect answer (label = 0)
        incorrect_indices = [i for i, label in enumerate(example['mc1_targets']['labels']) if label == 0]
        if not incorrect_indices:
            print(f"Example {i}: No incorrect answer found")
            continue
        incorrect_idx = incorrect_indices[0]
        incorrect_answer = example['mc1_targets']['choices'][incorrect_idx]
        
        # Add to processed data
        processed_data["prompts"].append(f"Question: {question}\nAnswer:")
        processed_data["correct_responses"].append(correct_answer)
        processed_data["incorrect_responses"].append(incorrect_answer)
        
        # Debug print for first few examples
        if i < 3:
            print(f"\nExample {i}:")
            print(f"Question: {question}")
            print(f"Correct answer: {correct_answer}")
            print(f"Incorrect answer: {incorrect_answer}")
    
    print(f"\nProcessed {len(processed_data['prompts'])} valid examples")
    print(f"Sample prompt: {processed_data['prompts'][0]}")
    print(f"Sample correct response: {processed_data['correct_responses'][0]}")
    print(f"Sample incorrect response: {processed_data['incorrect_responses'][0]}")
    return processed_data

# Preprocess the dataset
print("Starting dataset preprocessing...")
train_data = preprocess_dataset(dataset['validation'])

# Fixed BPFT dataset class
class BPFTDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, correct_responses, incorrect_responses, tokenizer, max_length=512):
        self.prompts = prompts
        self.correct_responses = correct_responses
        self.incorrect_responses = incorrect_responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Verify data integrity
        assert len(self.prompts) == len(self.correct_responses) == len(self.incorrect_responses), \
            f"Data mismatch: prompts({len(self.prompts)}), correct({len(self.correct_responses)}), incorrect({len(self.incorrect_responses)})"
        
        print(f"Dataset initialized with {len(self.prompts)} examples")
        
        # Debug first item
        if len(self.prompts) > 0:
            first_item = self[0]
            print("\nFirst item debug info:")
            print(f"Keys: {list(first_item.keys())}")
            for key, value in first_item.items():
                print(f"{key}: shape={value.shape}, type={type(value)}")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        if idx >= len(self.prompts):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.prompts)} items")
            
        prompt = self.prompts[idx]
        correct = self.correct_responses[idx]
        incorrect = self.incorrect_responses[idx]
        
        # Create formatted inputs
        correct_input = f"{prompt} {correct}"
        incorrect_input = f"{prompt} {incorrect}"
        
        # Tokenize with consistent settings
        correct_encoding = self.tokenizer(
            correct_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        incorrect_encoding = self.tokenizer(
            incorrect_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels for the correct sequence
        labels = correct_encoding["input_ids"].clone()
        
        # Find prompt length to mask prompt tokens in labels
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        labels[0, :prompt_tokens] = -100  # Set prompt tokens to -100 to ignore in loss
        
        # Create and return dictionary with all required fields
        # Use standard field names for regular inputs and special names for incorrect inputs
        item = {
            "input_ids": correct_encoding["input_ids"][0],
            "attention_mask": correct_encoding["attention_mask"][0],
            "labels": labels[0],
            "incorrect_ids": incorrect_encoding["input_ids"][0],
            "incorrect_mask": incorrect_encoding["attention_mask"][0],
        }
        
        # Debug print for this item (only occasionally to avoid flooding logs)
        if idx % 200 == 0:
            print(f"\nItem {idx} debug info:")
            print(f"Keys: {list(item.keys())}")
            for key, value in item.items():
                print(f"{key}: shape={value.shape}, type={type(value)}")
        
        return item

# Create the dataset
print("Creating BPFT dataset...")
bpft_dataset = BPFTDataset(
    train_data["prompts"],
    train_data["correct_responses"],
    train_data["incorrect_responses"],
    tokenizer,
    max_length=MAX_LENGTH
)

# Simplified data collator
class BPFTDataCollator:
    def __call__(self, features):
        if not features:
            raise ValueError("No features provided to data collator")
        
        # Debug output
        first_feat = features[0]
        print("\nData collator debug info:")
        print(f"Number of features: {len(features)}")
        print(f"First feature keys: {list(first_feat.keys())}")
        
        # Create batch by stacking tensors
        batch = {}
        for key in first_feat.keys():
            if all(key in f for f in features):
                try:
                    batch[key] = torch.stack([f[key] for f in features])
                except Exception as e:
                    print(f"Error stacking {key}: {e}")
        
        print(f"Batch keys: {list(batch.keys())}")
        return batch

# BPFT custom loss trainer
class BPFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        incorrect_input_ids = inputs["incorrect_ids"]  # Changed field name
        incorrect_attention_mask = inputs["incorrect_mask"]  # Changed field name
        
        # Compute loss for correct outputs
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        correct_loss = outputs.loss
        
        # Compute KL divergence loss between correct and incorrect outputs
        with torch.no_grad():
            correct_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits.detach()
        
        incorrect_outputs = model(
            input_ids=incorrect_input_ids,
            attention_mask=incorrect_attention_mask
        )
        incorrect_logits = incorrect_outputs.logits
        
        # Apply KL divergence to make incorrect less likely
        kl_loss = 0
        valid_batch_size = 0
        
        for i in range(input_ids.size(0)):
            # Find positions where labels are not -100 (i.e., non-padding tokens to predict)
            pred_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(pred_positions) == 0:
                continue
                
            # Get logits for these positions
            correct_pred_logits = correct_logits[i, pred_positions]
            incorrect_pred_logits = incorrect_logits[i, pred_positions]
            
            # Compute KL divergence
            kl_criterion = KLDivLoss(reduction="batchmean")
            kl = kl_criterion(
                log_softmax(incorrect_pred_logits, dim=-1),
                softmax(correct_pred_logits, dim=-1)
            )
            kl_loss += kl
            valid_batch_size += 1
        
        if valid_batch_size > 0:
            kl_loss = kl_loss / valid_batch_size
        else:
            kl_loss = torch.tensor(0.0, device=correct_loss.device)
        
        # Combine losses
        total_loss = correct_loss + LAMBDA * kl_loss
        
        # Free up memory
        torch.cuda.empty_cache()
        
        return (total_loss, outputs) if return_outputs else total_loss

# DeepSpeed configuration
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "contiguous_gradients": True,
        "overlap_comm": True
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "train_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
}

with open("ds_config.json", "w") as f:
    import json
    json.dump(deepspeed_config, f, indent=4)

# Update training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    report_to="tensorboard",
    gradient_checkpointing=True,
    deepspeed="ds_config.json",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    # Disable dataloader shuffling to help debugging
    dataloader_drop_last=False,
    dataloader_num_workers=0  # Avoid multiprocessing issues
)

# Create the trainer
print("Creating trainer...")
trainer = BPFTTrainer(
    model=model,
    args=training_args,
    train_dataset=bpft_dataset,
    eval_dataset=bpft_dataset,
    data_collator=BPFTDataCollator()
)

# Train
print("Starting training...")
trainer.train()

# Save the final model
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("Training complete. Model saved to", f"{OUTPUT_DIR}/final")

# Test function
def test_model(model_path, tokenizer_path, test_question):
    # Load the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with same quantization settings
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Generate response
    prompt = f"Question: {test_question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# Optional: Test the model with a sample question from the dataset
if len(train_data["prompts"]) > 0:
    sample_question = dataset['validation'][0]['question']
    print("\nTesting model with a sample question:")
    print("Question:", sample_question)
    print("Answer:", test_model(f"{OUTPUT_DIR}/final", f"{OUTPUT_DIR}/final", sample_question))
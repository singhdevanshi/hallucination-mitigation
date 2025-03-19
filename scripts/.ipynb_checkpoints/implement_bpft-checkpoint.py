import os
import math
import json
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DefaultDataCollator
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import bitsandbytes as bnb
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class BPFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        else:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(self.data, list):
            item = self.data[idx]
        else:
            item = self.data.iloc[idx]
        
        full_text = f"{item['input']}{item['response']}"
        
        encodings = self.tokenizer(full_text, 
                                  truncation=True,
                                  max_length=self.max_length,
                                  padding="max_length",
                                  return_tensors="pt")
        
        factuality_score = float(item['factuality_score'])
        contradiction_score = float(item['contradiction_score'])
        
        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "factuality_score": torch.tensor(factuality_score, dtype=torch.float),
            "contradiction_score": torch.tensor(contradiction_score, dtype=torch.float),
        }

class BPFTLoss(torch.nn.Module):
    def __init__(self, lambda_kl=0.1):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')
        self.lambda_kl = lambda_kl
        
    def forward(self, logits, labels, factuality_scores, contradiction_scores):
        # Standard cross-entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce_loss = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), 
                              shift_labels.view(-1))
        
        ce_loss = ce_loss.view(shift_labels.size())
        
        # Weight the loss by factuality scores (higher factuality = lower loss)
        batch_size = factuality_scores.size(0)
        factuality_weights = factuality_scores.view(batch_size, 1)
        weighted_ce_loss = ce_loss * (2 - factuality_weights)  # Higher factuality = lower weight
        
        # Calculate mean CE loss
        masked_ce_loss = weighted_ce_loss.sum() / (weighted_ce_loss.size(0) * weighted_ce_loss.size(1))
        
        # Create belief-aligned distribution by modifying logits
        # This represents PBPFT in the formula
        belief_penalty = contradiction_scores.view(-1, 1, 1) * 0.1
        modified_logits = shift_logits - belief_penalty * (1 - factuality_scores.view(-1, 1, 1))
        
        # Calculate KL divergence between original and modified distributions
        log_softmax_original = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        softmax_modified = torch.nn.functional.softmax(modified_logits, dim=-1)
        
        kl_loss = self.kl_div(log_softmax_original, softmax_modified)
        
        # Combine losses
        total_loss = masked_ce_loss + self.lambda_kl * kl_loss
        
        return total_loss

def prepare_model_for_training():
    """Prepare Mistral model for BPFT training with quantization"""
    print("Loading Mistral model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization settings for RTX 4090 (24GB VRAM)
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config={
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Set up LoRA configuration
    peft_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Create PEFT model
    model = get_peft_model(model, peft_config)
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_bpft_model():
    """Train Mistral model with BPFT approach"""
    print("Starting BPFT training...")
    
    # Prepare the model
    model, tokenizer = prepare_model_for_training()
    
    # Load the dataset
    train_dataset = BPFTDataset('/workspace/data/bpft/bpft_training_data.csv', tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="/workspace/models/mistral-7b-bpft",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="/workspace/results/bpft_logs",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="tensorboard"
    )
    
    # Create a custom training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size, 
        shuffle=True
    )
    
    # Training loop
    num_epochs = training_args.num_train_epochs
    loss_fn = BPFTLoss(lambda_kl=0.1)
    
    for epoch in range(int(num_epochs)):
        print(f"Starting epoch {epoch+1}/{int(num_epochs)}")
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            
            # Calculate BPFT loss
            loss = loss_fn(
                outputs.logits, 
                batch["input_ids"], 
                batch["factuality_score"], 
                batch["contradiction_score"]
            )
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Log every 10 steps
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # End of epoch
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{int(num_epochs)}, Average Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        model.save_pretrained(f"/workspace/models/mistral-7b-bpft-epoch-{epoch+1}")
    
    # Save final model
    model.save_pretrained("/workspace/models/mistral-7b-bpft-final")
    print("BPFT training completed and model saved.")

if __name__ == "__main__":
    train_bpft_model()
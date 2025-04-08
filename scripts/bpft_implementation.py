import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Any

class BPFTTrainer:
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
        output_dir: str = "./llama-8b-bpft",
        learning_rate: float = 5e-5,
        belief_alignment_weight: float = 0.1
    ):
        """
        Initialize BPFT Fine-Tuning Setup
        
        Args:
            model_name (str): Base model to fine-tune
            output_dir (str): Directory to save fine-tuned model
            learning_rate (float): Learning rate for training
            belief_alignment_weight (float): Hyperparameter λ for belief consistency
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Training configurations
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.belief_alignment_weight = belief_alignment_weight
    
    def prepare_dataset(self, dataset_name: str = "wikipedia", config_name: str = "20220301.en", split: str = "train"):
        """
        Prepare dataset for BPFT fine-tuning
        
        Args:
            dataset_name (str): Hugging Face dataset name
            config_name (str): Specific language configuration
            split (str): Dataset split to use
        
        Returns:
            Processed dataset
        """
        # Load dataset (English Wikipedia)
        dataset = load_dataset(dataset_name, config_name, split=split)
        
        # Preprocess dataset
        def preprocess_function(examples):
            # Extract text and create context-rich samples
            texts = [
                f"Context: {text[:500]}\nSummary: {text[500:1000]}" 
                for text in examples['text']
                if text and len(text) > 1000
            ]
            
            return self.tokenizer(
                texts, 
                truncation=True, 
                max_length=512, 
                padding="max_length"
            )
        
        processed_dataset = dataset.map(
            preprocess_function, 
            batched=True, 
            remove_columns=dataset.column_names
        )
        
        return processed_dataset
    
    def custom_loss_fn(self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False):
        """
        Custom loss function for BPFT
        
        Args:
            model (nn.Module): The model
            inputs (Dict[str, Any]): Input dictionary
            return_outputs (bool): Whether to return model outputs
        
        Returns:
            Tuple or Tensor of loss
        """
        # Forward pass
        outputs = model(**inputs)
        
        # Standard cross-entropy loss
        loss = outputs.loss
        
        # Compute belief consistency loss (KL divergence)
        logits = outputs.logits
        baseline_probs = torch.softmax(logits, dim=-1)
        
        # KL divergence consistency penalty
        consistency_loss = torch.nn.functional.kl_div(
            baseline_probs.log(), 
            baseline_probs, 
            reduction='batchmean'
        )
        
        # Combine losses
        total_loss = loss + self.belief_alignment_weight * consistency_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def train(self, num_train_epochs: int = 3, batch_size: int = 4):
        """
        Execute BPFT Fine-Tuning
        
        Args:
            num_train_epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        # Prepare dataset
        train_dataset = self.prepare_dataset()
        val_dataset = self.prepare_dataset(split="validation")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_loss=self.custom_loss_fn
        )
        
        # Start training
        trainer.train()
        
        # Save final model and tokenizer
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"BPFT Fine-Tuned model saved to {self.output_dir}")

def main():
    # Initialize and run BPFT Training
    bpft_trainer = BPFTTrainer()
    bpft_trainer.train()

if __name__ == "__main__":
    main()
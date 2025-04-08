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

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./llama-8b-isc"
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 3
SUPPRESSION_FACTOR = 0.2
MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=os.environ.get("HF_TOKEN")
)

class ISCModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        
        hidden_size = self.model.config.hidden_size
        self.hallucination_detector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        num_layers = len(self.model.model.layers)
        num_monitor = min(6, num_layers)
        self.monitored_layers = [
            int(idx * num_layers / (num_monitor + 1)) for idx in range(1, num_monitor + 1)
        ]
        print(f"Monitoring layers: {self.monitored_layers}")
    
    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=True):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        if self.training:
            return outputs
        
        hidden_states = outputs.hidden_states
        
        if labels is None:
            modified_hidden_states = []
            for layer_idx, layer_hidden_state in enumerate(hidden_states):
                if layer_idx in self.monitored_layers:
                    hallucination_scores = self.hallucination_detector(layer_hidden_state)
                    suppression = SUPPRESSION_FACTOR * hallucination_scores
                    modified_state = layer_hidden_state - suppression * layer_hidden_state
                    modified_hidden_states.append(modified_state)
                else:
                    modified_hidden_states.append(layer_hidden_state)
            
            outputs.hidden_states = tuple(modified_hidden_states)
        
        return outputs
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()
    
    def is_gradient_checkpointing(self):
        return self.model.is_gradient_checkpointing()

def load_hallucination_dataset():
    datasets = []
    
    print("Loading TruthfulQA dataset...")
    truthful_qa = load_dataset("truthful_qa", "multiple_choice")
    
    max_truthful_examples = 2000
    truthful_examples = []
    
    for i, example in enumerate(truthful_qa['validation']):
        if i >= max_truthful_examples:
            break
            
        question = example['question']
        
        correct_idx = example['mc1_targets']['labels'].index(1)
        factual_answer = example['mc1_targets']['choices'][correct_idx]
        
        incorrect_indices = [j for j, label in enumerate(example['mc1_targets']['labels']) if label == 0]
        if incorrect_indices:
            incorrect_idx = incorrect_indices[0]
            hallucinated_answer = example['mc1_targets']['choices'][incorrect_idx]
            
            truthful_examples.append({
                "text": f"Question: {question}\nAnswer: {factual_answer}",
                "is_hallucination": 0
            })
            
            truthful_examples.append({
                "text": f"Question: {question}\nAnswer: {hallucinated_answer}",
                "is_hallucination": 1
            })
    
    datasets.extend(truthful_examples)
    
    try:
        print("Loading FEVER dataset...")
        fever = load_dataset("fever", "v1.0")
        
        max_fever_examples = 1500
        fever_examples = []
        
        for i, example in enumerate(fever['train']):
            if i >= max_fever_examples:
                break
                
            claim = example['claim']
            label = example['label']
            
            if label in [0, 1]:
                fever_examples.append({
                    "text": f"Claim: {claim}\nThis claim is {'true' if label == 0 else 'false'}.",
                    "is_hallucination": 1 if label == 1 else 0
                })
        
        datasets.extend(fever_examples)
        del fever
        gc.collect()
    except Exception as e:
        print(f"Error loading FEVER dataset: {e}")
    
    print("Creating synthetic examples...")
    synthetic_examples = [
        {"text": "Question: What is the capital of France?\nAnswer: The capital of France is Paris.", "is_hallucination": 0},
        {"text": "Question: What is 2+2?\nAnswer: 2+2 equals 4.", "is_hallucination": 0},
        {"text": "Question: Who wrote Hamlet?\nAnswer: William Shakespeare wrote Hamlet.", "is_hallucination": 0},
        {"text": "Question: What is the chemical symbol for water?\nAnswer: The chemical formula for water is H2O.", "is_hallucination": 0},
        {"text": "Question: What is the largest planet in our solar system?\nAnswer: Jupiter is the largest planet in our solar system.", "is_hallucination": 0},
        
        {"text": "Question: What is the capital of France?\nAnswer: The capital of France is Lyon.", "is_hallucination": 1},
        {"text": "Question: What is 2+2?\nAnswer: 2+2 equals 5.", "is_hallucination": 1},
        {"text": "Question: Who wrote Hamlet?\nAnswer: Charles Dickens wrote Hamlet.", "is_hallucination": 1},
        {"text": "Question: What is the chemical symbol for water?\nAnswer: The chemical formula for water is CO2.", "is_hallucination": 1},
        {"text": "Question: What is the largest planet in our solar system?\nAnswer: Venus is the largest planet in our solar system.", "is_hallucination": 1},
    ]
    
    datasets.extend(synthetic_examples)
    
    np.random.shuffle(datasets)
    
    print(f"Total dataset size: {len(datasets)} examples")
    
    del truthful_qa
    gc.collect()
    
    return datasets

hallucination_dataset = load_hallucination_dataset()
print(f"Loaded {len(hallucination_dataset)} examples for ISC training")

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
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = encoding["input_ids"].clone()
        is_hallucination = torch.tensor(float(is_hallucination), dtype=torch.float32)
        
        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": labels[0],
            "is_hallucination": is_hallucination
        }

# Fix here: Create proper train/val split using indices
train_size = int(0.9 * len(hallucination_dataset))
val_size = len(hallucination_dataset) - train_size

# Create a list of indices and split it
indices = list(range(len(hallucination_dataset)))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_isc_dataset = ISCDataset([hallucination_dataset[i] for i in train_indices], tokenizer)
val_isc_dataset = ISCDataset([hallucination_dataset[i] for i in val_indices], tokenizer)

class ISCDataCollator:
    def __call__(self, features):
        batch = {}
        
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])
        
        if "is_hallucination" in features[0]:
            batch["is_hallucination"] = torch.stack([f["is_hallucination"] for f in features])
        else:
            batch["is_hallucination"] = torch.zeros(len(features), dtype=torch.float32)
        
        return batch

isc_model = ISCModel(base_model)

class ISCTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = {k: v for k, v in inputs.items() if k != "is_hallucination"}
        is_hallucination = inputs.get("is_hallucination")
        
        if is_hallucination is None:
            is_hallucination = torch.zeros(model_inputs["input_ids"].size(0), dtype=torch.float32, device=model_inputs["input_ids"].device)
        else:
            is_hallucination = is_hallucination.to(model_inputs["input_ids"].device)
        
        outputs = model(
            **model_inputs,
            output_hidden_states=True
        )
        
        lm_loss = outputs.loss
        
        hidden_states = outputs.hidden_states
        
        if hasattr(model, "module"):
            actual_model = model.module
        else:
            actual_model = model
        
        hallucination_loss = 0
        for layer_idx in actual_model.monitored_layers:
            layer_hidden_state = hidden_states[layer_idx]
            
            last_token_idx = (model_inputs["attention_mask"].sum(dim=1) - 1).to(torch.long)
            batch_indices = torch.arange(layer_hidden_state.size(0), device=layer_hidden_state.device)
            last_hidden_states = layer_hidden_state[batch_indices, last_token_idx]
            
            hallucination_preds = actual_model.hallucination_detector(last_hidden_states).squeeze(-1)
            
            bce_loss = nn.BCELoss()
            layer_loss = bce_loss(hallucination_preds, is_hallucination)
            hallucination_loss += layer_loss
            
            if self.state.global_step % 10 == 0:
                self.log({f"hallucination_loss_layer_{layer_idx}": layer_loss.item()})
        
        hallucination_loss /= len(actual_model.monitored_layers)
        
        hallucination_weight = min(0.5, 0.1 + (0.4 * self.state.epoch / NUM_EPOCHS))
        total_loss = (1 - hallucination_weight) * lm_loss + hallucination_weight * hallucination_loss
        
        if self.state.global_step % 10 == 0:
            self.log({
                "lm_loss": lm_loss.item(),
                "hallucination_loss": hallucination_loss.item(),
                "hallucination_weight": hallucination_weight,
                "total_loss": total_loss.item()
            })
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def evaluation_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None):
        outputs = super().evaluation_step(model, inputs, prediction_loss_only, ignore_keys)
        
        if prediction_loss_only:
            return outputs
        
        model_inputs = {k: v for k, v in inputs.items() if k != "is_hallucination"}
        is_hallucination = inputs.get("is_hallucination")
        
        if is_hallucination is not None:
            with torch.no_grad():
                full_outputs = model(**model_inputs, output_hidden_states=True)
                hidden_states = full_outputs.hidden_states
                
                if hasattr(model, "module"):
                    actual_model = model.module
                else:
                    actual_model = model
                
                middle_layer_idx = actual_model.monitored_layers[len(actual_model.monitored_layers) // 2]
                middle_layer_hidden = hidden_states[middle_layer_idx]
                
                last_token_idx = (model_inputs["attention_mask"].sum(dim=1) - 1).to(torch.long)
                batch_indices = torch.arange(middle_layer_hidden.size(0), device=middle_layer_hidden.device)
                last_hidden_states = middle_layer_hidden[batch_indices, last_token_idx]
                
                hallucination_preds = actual_model.hallucination_detector(last_hidden_states).squeeze(-1)
                
                pred_labels = (hallucination_preds > 0.5).float()
                accuracy = (pred_labels == is_hallucination).float().mean().item()
                
                self.log({
                    "eval_hallucination_accuracy": accuracy,
                })
        
        return outputs

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    eval_strategy="epoch",
    report_to="tensorboard",
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    bf16=True,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

isc_model = isc_model.to_empty(device=device)
for module in isc_model.hallucination_detector.modules():
    if hasattr(module, 'weight') and module.weight.is_meta:
        module.weight = nn.Parameter(torch.randn_like(module.weight, device=device))
    if hasattr(module, 'bias') and module.bias is not None and module.bias.is_meta:
        module.bias = nn.Parameter(torch.zeros_like(module.bias, device=device))

trainer = ISCTrainer(
    model=isc_model,
    args=training_args,
    train_dataset=train_isc_dataset,
    eval_dataset=val_isc_dataset,
    data_collator=ISCDataCollator(),
)

print("Starting ISC training...")
trainer.train()

print("Saving model...")
base_model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
torch.save(isc_model.hallucination_detector.state_dict(), f"{OUTPUT_DIR}/final/hallucination_detector.pt")
print(f"Model saved to {OUTPUT_DIR}/final")

def evaluate_hallucination_detection(model, example_text, is_hallucination=None):
    inputs = tokenizer(example_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states
        hallucination_scores = []
        
        for layer_idx in model.monitored_layers:
            layer_hidden = hidden_states[layer_idx]
            last_hidden = layer_hidden[:, -1, :]
            hall_score = model.hallucination_detector(last_hidden).item()
            hallucination_scores.append(hall_score)
        
        avg_score = sum(hallucination_scores) / len(hallucination_scores)
        
        result = {
            "text": example_text,
            "hallucination_score": avg_score,
            "layer_scores": hallucination_scores,
            "prediction": "Hallucination" if avg_score > 0.5 else "Factual"
        }
        
        if is_hallucination is not None:
            expected = "Hallucination" if is_hallucination else "Factual"
            result["expected"] = expected
            result["correct"] = (expected == result["prediction"])
        
        return result

def generate_with_isc(prompt, max_new_tokens=200, temperature=0.7, do_sample=True):
    try:
        loaded_model = AutoModelForCausalLM.from_pretrained(
            f"{OUTPUT_DIR}/final",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        loaded_model_isc = ISCModel(loaded_model)
        
        detector_path = f"{OUTPUT_DIR}/final/hallucination_detector.pt"
        if os.path.exists(detector_path):
            model_device = next(loaded_model.parameters()).device
            detector_state_dict = torch.load(detector_path, map_location=model_device)
            loaded_model_isc = loaded_model_isc.to_empty(device=model_device)
            for module in loaded_model_isc.hallucination_detector.modules():
                if hasattr(module, 'weight') and module.weight.is_meta:
                    module.weight = nn.Parameter(torch.randn_like(module.weight, device=model_device))
                if hasattr(module, 'bias') and module.bias is not None and module.bias.is_meta:
                    module.bias = nn.Parameter(torch.zeros_like(module.bias, device=model_device))
            loaded_model_isc.hallucination_detector.load_state_dict(detector_state_dict)
        
        loaded_model_isc.eval()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(loaded_model.device)
        outputs = loaded_model_isc.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        print(f"Error during generation: {e}")
        try:
            loaded_model = AutoModelForCausalLM.from_pretrained(
                f"{OUTPUT_DIR}/final",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(loaded_model.device)
            outputs = loaded_model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as fallback_e:
            return f"Error generating with the model: {fallback_e}"

if __name__ == "__main__":
    test_prompts = [
        "What is the capital of France?",
        "How many legs does a spider have?",
        "Who was the first person to walk on the moon?",
        "How many rings are on the Olympic flag?",
        "What color is a banana?",
        
        "What is the capital of the fictional country of Wakanda?",
        "How many fingers does Mickey Mouse have on each hand?",
        "What is the population of Mars?",
        "Who was the 51st president of the United States?",
        "Can you explain quantum chromodynamics in simple terms?"
    ]
    
    print("\n===== TESTING THE MODEL ON VARIOUS PROMPTS =====\n")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = generate_with_isc(prompt)
        print(f"Response: {response}\n")
        print("-" * 80)
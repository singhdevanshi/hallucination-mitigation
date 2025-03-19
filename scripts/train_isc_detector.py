import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Create directory
os.makedirs('/workspace/models/isc', exist_ok=True)

class HiddenStateDataset(Dataset):
    def __init__(self, factual_states, non_factual_states):
        """Dataset for training hallucination detector"""
        self.samples = []
        self.labels = []
        
        # Process factual states
        for state_data in factual_states:
            # Extract features from middle layers
            features = self._extract_features(state_data["hidden_states"])
            self.samples.append(features)
            self.labels.append(1)  # 1 = factual
        
        # Process non-factual states
        for state_data in non_factual_states:
            # Extract features from middle layers
            features = self._extract_features(state_data["hidden_states"])
            self.samples.append(features)
            self.labels.append(0)  # 0 = non-factual (hallucination)
    
    def _extract_features(self, hidden_states):
        """Extract feature vector from hidden states"""
        # Concatenate statistics from all layers
        features = []
        
        for layer_num, layer_state in hidden_states.items():
            # Get hidden state tensor
            hidden = layer_state
            
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.samples[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

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

def load_hidden_states():
    """Load hidden states from pickled files"""
    print("Loading hidden states...")
    
    try:
        with open('/workspace/data/isc/factual_states.pkl', 'rb') as f:
            factual_states = pickle.load(f)
        
        with open('/workspace/data/isc/non_factual_states.pkl', 'rb') as f:
            non_factual_states = pickle.load(f)
        
        print(f"Loaded {len(factual_states)} factual and {len(non_factual_states)} non-factual samples")
        return factual_states, non_factual_states
    except FileNotFoundError:
        print("Hidden states not found. Trying to load sample files...")
        
        try:
            with open('/workspace/data/isc/factual_states_sample.pkl', 'rb') as f:
                factual_states = pickle.load(f)
            
            with open('/workspace/data/isc/non_factual_states_sample.pkl', 'rb') as f:
                non_factual_states = pickle.load(f)
            
            print(f"Loaded {len(factual_states)} factual and {len(non_factual_states)} non-factual sample states")
            return factual_states, non_factual_states
        except FileNotFoundError:
            print("Sample hidden states not found either. Please run extract_hidden_states.py first.")
            return None, None

def train_detector(factual_states, non_factual_states):
    """Train hallucination detector model"""
    print("Preparing dataset...")
    
    # Create dataset
    dataset = HiddenStateDataset(factual_states, non_factual_states)
    
    # Split into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(train_indices)
    )
    
    val_loader = DataLoader(
        dataset, batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(val_indices)
    )
    
    # Initialize model
    input_dim = dataset[0]["features"].shape[0]
    print(f"Feature dimension: {input_dim}")
    
    model = HallucinationDetector(input_dim=input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            features = batch["features"].to(device)
            labels = batch["label"].float().to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                features = batch["features"].to(device)
                labels = batch["label"].float().to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Store predictions and targets for metrics
                preds = (outputs > 0.5).float().cpu().numpy()
                targets = labels.cpu().numpy()
                
                val_preds.extend(preds)
                val_targets.extend(targets)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate metrics
        accuracy = accuracy_score(val_targets, val_preds)
        precision = precision_score(val_targets, val_preds, zero_division=0)
        recall = recall_score(val_targets, val_preds, zero_division=0)
        f1 = f1_score(val_targets, val_preds, zero_division=0)
        
        val_accuracies.append(accuracy)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Save model
    torch.save(model.state_dict(), '/workspace/models/isc/hallucination_detector.pt')
    print("Model saved to /workspace/models/isc/hallucination_detector.pt")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    os.makedirs('/workspace/results/isc', exist_ok=True)
    plt.savefig('/workspace/results/isc/training_curves.png')
    print("Training curves saved to /workspace/results/isc/training_curves.png")
    
    return model

def main():
    factual_states, non_factual_states = load_hidden_states()
    
    if factual_states is not None and non_factual_states is not None:
        train_detector(factual_states, non_factual_states)
    else:
        print("Could not load hidden states. Exiting.")

if __name__ == "__main__":
    main()
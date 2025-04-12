import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

class SecurityDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SecurityClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=3):
        super(SecurityClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class SecurityMLTrainer:
    def __init__(self, results_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = results_dir or Path(__file__).parent / "ml_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = []
    
    def prepare_data(self, data_path):
        """Prepare data for training"""
        # Load and preprocess data
        df = pd.read_csv(data_path)
        
        # Extract features (example features - adjust based on your data)
        features = df[['cpu_usage', 'memory_usage', 'inference_latency']].values
        
        # Generate sample labels (0: Normal, 1: Suspicious, 2: Attack)
        # In real implementation, these would come from your actual data
        labels = np.where(
            features[:, 0] > np.percentile(features[:, 0], 90),
            2,  # Attack
            np.where(
                features[:, 0] > np.percentile(features[:, 0], 70),
                1,  # Suspicious
                0   # Normal
            )
        )
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = SecurityDataset(X_train, y_train)
        test_dataset = SecurityDataset(X_test, y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32)
        
        return X_train.shape[1]  # Return input size
    
    def train(self, num_epochs=50, progress_callback=None):
        """Train the model with live progress updates"""
        input_size = self.prepare_data(next(self.results_dir.parent.glob("results/*/*.csv")))
        
        # Initialize model
        self.model = SecurityClassifier(input_size).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        best_val_acc = 0
        no_improve_count = 0
        
        if progress_callback:
            progress_callback({
                'status': 'init',
                'message': f'Initialized model on {self.device}',
                'total_epochs': num_epochs,
                'current_epoch': 0
            })
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            if progress_callback:
                progress_callback({
                    'status': 'epoch_start',
                    'message': f'Starting epoch {epoch + 1}/{num_epochs}',
                    'current_epoch': epoch + 1
                })
            
            # Training batches
            for batch_idx, (features, labels) in enumerate(self.train_loader):
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Batch progress
                if progress_callback and batch_idx % 5 == 0:
                    progress_callback({
                        'status': 'batch_progress',
                        'message': f'Batch {batch_idx + 1}/{len(self.train_loader)}',
                        'batch_loss': loss.item(),
                        'batch_acc': 100. * predicted.eq(labels).sum().item() / labels.size(0)
                    })
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            if progress_callback:
                progress_callback({
                    'status': 'validation_start',
                    'message': 'Starting validation'
                })
            
            with torch.no_grad():
                for features, labels in self.test_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Calculate epoch metrics
            train_loss = train_loss / len(self.train_loader)
            train_acc = 100. * correct / total
            val_loss = val_loss / len(self.test_loader)
            val_acc = 100. * val_correct / val_total
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Save metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            self.training_history.append(metrics)
            
            if progress_callback:
                progress_callback({
                    'status': 'epoch_end',
                    'message': f'Epoch {epoch + 1} completed',
                    'metrics': metrics,
                    'best_val_acc': best_val_acc
                })
            
            # Save model and metrics
            self.save_training_results()
            
            # Early stopping
            if no_improve_count >= 5:
                if progress_callback:
                    progress_callback({
                        'status': 'early_stop',
                        'message': 'Early stopping: No improvement for 5 epochs'
                    })
                break
        
        if progress_callback:
            progress_callback({
                'status': 'complete',
                'message': 'Training completed',
                'final_metrics': self.training_history[-1],
                'best_val_acc': best_val_acc
            })
    
    def predict(self, features):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(self.scaler.transform(features)).to(self.device)
            outputs = self.model(features)
            _, predicted = outputs.max(1)
            return predicted.cpu().numpy()
    
    def save_training_results(self):
        """Save model and training history"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.results_dir / timestamp
        save_dir.mkdir(exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), save_dir / "model.pth")
        
        # Save training history
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save scaler
        torch.save(self.scaler, save_dir / "scaler.pth")

def train_security_model():
    """Function to train the security model"""
    trainer = SecurityMLTrainer()
    trainer.train()
    return trainer.results_dir 

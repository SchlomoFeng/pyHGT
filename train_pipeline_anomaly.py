"""
Training Script for Pipeline Anomaly Detection using Adapted pyHGT

This script trains a model to detect anomalies in gas pipeline networks.
"""

import os
import sys
import json
import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional

# Add pyHGT to path
sys.path.append('/home/runner/work/pyHGT/pyHGT')

# Import our modules
from pipeline_data import PipelineDataProcessor
from pipeline_hgt import PipelineAnomalyDetector, PipelineGraph

# Try to import machine learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using simple algorithms")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class SimpleNeuralAnomalyDetector:
    """Simple neural network for anomaly detection when PyTorch is not available"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights with small random values
        self.w1 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [random.gauss(0, 0.1) for _ in range(hidden_dim)]
        
        self.w2 = [[random.gauss(0, 0.1)] for _ in range(hidden_dim)]
        self.b2 = [random.gauss(0, 0.1)]
        
        self.learning_rate = 0.01
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))  # Clip to prevent overflow
    
    def forward(self, inputs: List[float]) -> float:
        """Forward pass"""
        # Hidden layer
        hidden = []
        for h in range(self.hidden_dim):
            activation = self.b1[h]
            for i in range(self.input_dim):
                activation += inputs[i] * self.w1[i][h]
            hidden.append(self.sigmoid(activation))
        
        # Output layer
        output = self.b2[0]
        for h in range(self.hidden_dim):
            output += hidden[h] * self.w2[h][0]
        
        return self.sigmoid(output)
    
    def train_step(self, inputs: List[float], target: float):
        """Single training step with backpropagation"""
        # Forward pass
        hidden = []
        for h in range(self.hidden_dim):
            activation = self.b1[h]
            for i in range(self.input_dim):
                activation += inputs[i] * self.w1[i][h]
            hidden.append(self.sigmoid(activation))
        
        output = self.b2[0]
        for h in range(self.hidden_dim):
            output += hidden[h] * self.w2[h][0]
        prediction = self.sigmoid(output)
        
        # Calculate loss and gradients
        error = prediction - target
        
        # Output layer gradients
        output_grad = error * prediction * (1 - prediction)
        
        # Update output weights
        for h in range(self.hidden_dim):
            self.w2[h][0] -= self.learning_rate * output_grad * hidden[h]
        self.b2[0] -= self.learning_rate * output_grad
        
        # Hidden layer gradients
        for h in range(self.hidden_dim):
            hidden_grad = output_grad * self.w2[h][0] * hidden[h] * (1 - hidden[h])
            
            # Update hidden weights
            for i in range(self.input_dim):
                self.w1[i][h] -= self.learning_rate * hidden_grad * inputs[i]
            self.b1[h] -= self.learning_rate * hidden_grad
        
        return error * error  # Return squared error


class PipelineAnomalyTrainer:
    """Training pipeline for anomaly detection"""
    
    def __init__(self, data_dir: str = "/home/runner/work/pyHGT/pyHGT"):
        self.data_dir = data_dir
        self.blueprint_path = os.path.join(data_dir, "blueprint/0708YTS4.json")
        self.sensor_path = os.path.join(data_dir, "StreamData/0708YTS4.csv")
        
        self.processor = None
        self.detector = None
        self.pipeline_graph = None
        self.model = None
        
    def load_and_process_data(self):
        """Load and process pipeline data"""
        print("Loading and processing pipeline data...")
        
        self.processor = PipelineDataProcessor(self.blueprint_path, self.sensor_path)
        node_features, edge_features, sensor_data, temporal_features = self.processor.process_all_data()
        
        # Initialize detector
        self.detector = PipelineAnomalyDetector()
        self.pipeline_graph = self.detector.build_graph_from_pipeline_data(
            node_features, edge_features, sensor_data, temporal_features
        )
        
        return node_features, edge_features, sensor_data, temporal_features
    
    def create_training_data(self, anomaly_rate: float = 0.1) -> Tuple[List, List]:
        """Create training data with features and labels"""
        if not self.pipeline_graph:
            raise ValueError("Must load data first")
        
        features = []
        labels = []
        
        # Create synthetic anomalies for training
        edge_ids = list(self.pipeline_graph.edges.keys())
        num_anomalies = int(len(edge_ids) * anomaly_rate)
        anomaly_edges = set(random.sample(edge_ids, num_anomalies))
        
        print(f"Creating training data with {num_anomalies} anomalies out of {len(edge_ids)} edges")
        
        # Extract features for each edge
        for edge_id, edge_data in self.pipeline_graph.edges.items():
            edge_features = self.extract_edge_training_features(edge_id)
            features.append(edge_features)
            labels.append(1.0 if edge_id in anomaly_edges else 0.0)
        
        print(f"Created {len(features)} training samples with {len(edge_features)} features each")
        return features, labels
    
    def extract_edge_training_features(self, edge_id: str) -> List[float]:
        """Extract comprehensive features for an edge for training"""
        edge = self.pipeline_graph.edges[edge_id]
        features = []
        
        # Basic edge features
        features.extend(self.pipeline_graph.get_edge_features(edge_id))
        
        # Source node features
        source_features = self.pipeline_graph.get_node_features(edge['source'])
        features.extend(source_features)
        
        # Target node features
        target_features = self.pipeline_graph.get_node_features(edge['target'])
        features.extend(target_features)
        
        # Connectivity features
        source_neighbors = len(self.pipeline_graph.get_neighbors(edge['source']))
        target_neighbors = len(self.pipeline_graph.get_neighbors(edge['target']))
        features.extend([float(source_neighbors), float(target_neighbors)])
        
        # Graph structural features
        total_nodes = len(self.pipeline_graph.nodes)
        total_edges = len(self.pipeline_graph.edges)
        features.extend([float(total_nodes), float(total_edges)])
        
        return features
    
    def train_model(self, features: List[List[float]], labels: List[float], 
                   epochs: int = 100, validation_split: float = 0.2):
        """Train the anomaly detection model"""
        if not features:
            raise ValueError("No training data provided")
        
        # Split data
        n_samples = len(features)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Shuffle data
        combined = list(zip(features, labels))
        random.shuffle(combined)
        features, labels = zip(*combined)
        
        train_features = features[:n_train]
        train_labels = labels[:n_train]
        val_features = features[n_train:]
        val_labels = labels[n_train:]
        
        print(f"Training on {n_train} samples, validating on {n_val} samples")
        
        # Initialize model
        input_dim = len(features[0])
        print(f"Input dimension: {input_dim}")
        
        if TORCH_AVAILABLE:
            self.model = self.create_pytorch_model(input_dim)
        else:
            self.model = SimpleNeuralAnomalyDetector(input_dim)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            for feat, label in zip(train_features, train_labels):
                if TORCH_AVAILABLE:
                    loss = self.train_pytorch_step(feat, label)
                else:
                    loss = self.model.train_step(feat, label)
                train_loss += loss
            
            train_loss /= len(train_features)
            
            # Validation
            val_loss = 0.0
            val_predictions = []
            for feat, label in zip(val_features, val_labels):
                if TORCH_AVAILABLE:
                    pred = self.predict_pytorch(feat)
                else:
                    pred = self.model.forward(feat)
                val_predictions.append(pred)
                val_loss += (pred - label) ** 2
            
            val_loss /= len(val_features)
            
            # Calculate accuracy
            val_acc = self.calculate_accuracy(val_predictions, val_labels)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss {train_loss:.4f}, "
                      f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.3f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return self.model
    
    def create_pytorch_model(self, input_dim: int):
        """Create PyTorch model if available"""
        class AnomalyNet(nn.Module):
            def __init__(self, input_dim, hidden_dim=64):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = AnomalyNet(input_dim)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        return model
    
    def train_pytorch_step(self, features: List[float], label: float) -> float:
        """Training step for PyTorch model"""
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor([label], dtype=torch.float32).unsqueeze(0)
        
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_pytorch(self, features: List[float]) -> float:
        """Prediction with PyTorch model"""
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(x)
        return output.item()
    
    def calculate_accuracy(self, predictions: List[float], labels: List[float], 
                          threshold: float = 0.5) -> float:
        """Calculate binary classification accuracy"""
        correct = 0
        for pred, label in zip(predictions, labels):
            pred_binary = 1.0 if pred > threshold else 0.0
            if pred_binary == label:
                correct += 1
        return correct / len(predictions)
    
    def evaluate_model(self, features: List[List[float]], labels: List[float]):
        """Evaluate trained model"""
        if not self.model:
            raise ValueError("Model not trained yet")
        
        predictions = []
        for feat in features:
            if TORCH_AVAILABLE:
                pred = self.predict_pytorch(feat)
            else:
                pred = self.model.forward(feat)
            predictions.append(pred)
        
        # Calculate metrics
        accuracy = self.calculate_accuracy(predictions, labels)
        
        # Calculate precision and recall
        tp = fp = tn = fn = 0
        for pred, label in zip(predictions, labels):
            pred_binary = 1.0 if pred > 0.5 else 0.0
            if pred_binary == 1.0 and label == 1.0:
                tp += 1
            elif pred_binary == 1.0 and label == 0.0:
                fp += 1
            elif pred_binary == 0.0 and label == 0.0:
                tn += 1
            else:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\nModel Evaluation Results:")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions
        }
    
    def detect_real_anomalies(self) -> Dict[str, float]:
        """Use trained model to detect real anomalies in the pipeline"""
        if not self.model or not self.pipeline_graph:
            raise ValueError("Model and data must be loaded first")
        
        anomaly_scores = {}
        
        for edge_id in self.pipeline_graph.edges:
            features = self.extract_edge_training_features(edge_id)
            
            if TORCH_AVAILABLE:
                score = self.predict_pytorch(features)
            else:
                score = self.model.forward(features)
            
            anomaly_scores[edge_id] = score
        
        return anomaly_scores


def main():
    """Main training and evaluation pipeline"""
    print("=== Pipeline Anomaly Detection Training ===")
    
    # Initialize trainer
    trainer = PipelineAnomalyTrainer()
    
    # Load and process data
    trainer.load_and_process_data()
    
    # Create training data
    features, labels = trainer.create_training_data(anomaly_rate=0.15)
    
    # Train model
    model = trainer.train_model(features, labels, epochs=50)
    
    # Evaluate model
    results = trainer.evaluate_model(features, labels)
    
    # Detect anomalies in real data
    print("\n=== Real Anomaly Detection ===")
    anomaly_scores = trainer.detect_real_anomalies()
    
    # Show top anomalous edges
    top_anomalies = sorted(anomaly_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Most Anomalous Edges:")
    for i, (edge_id, score) in enumerate(top_anomalies, 1):
        edge = trainer.pipeline_graph.edges[edge_id]
        print(f"{i:2d}. Edge {edge_id[:8]}... (score: {score:.3f}) "
              f"Length: {edge.get('length', 0):.1f}")
    
    print(f"\nTraining completed successfully!")
    print(f"Model can now be used for real-time anomaly detection.")
    
    return trainer, results, anomaly_scores


if __name__ == "__main__":
    trainer, results, anomalies = main()
"""
Pipeline Graph Neural Network for Anomaly Detection

This module adapts pyHGT for gas pipeline network anomaly detection.
It creates a heterogeneous graph representation suitable for detecting leakages.
"""

import os
import sys
import json
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Add pyHGT to path
sys.path.append('/home/runner/work/pyHGT/pyHGT')

# Import pyHGT modules
try:
    from pyHGT.data import Graph
    from pyHGT.conv import HGTLayer
    PYHGT_AVAILABLE = True
except ImportError:
    print("Warning: pyHGT modules not available, creating minimal implementation")
    PYHGT_AVAILABLE = False

# Import our pipeline data processor
from pipeline_data import PipelineDataProcessor

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using minimal implementation")


class PipelineGraph:
    """Graph representation for pipeline network"""
    
    def __init__(self):
        self.nodes = {}  # node_id -> node_data
        self.edges = {}  # edge_id -> edge_data  
        self.node_types = ['Stream', 'VavlePro', 'Mixer', 'Tee']
        self.edge_types = ['pipe']
        self.adjacency = defaultdict(lambda: defaultdict(list))  # type -> type -> edges
        
    def add_node(self, node_id: str, node_data: Dict):
        """Add a node to the graph"""
        self.nodes[node_id] = node_data
        
    def add_edge(self, edge_id: str, source_id: str, target_id: str, edge_data: Dict):
        """Add an edge to the graph"""
        self.edges[edge_id] = {
            'source': source_id,
            'target': target_id,
            **edge_data
        }
        
        # Add to adjacency structure
        source_type = self.nodes[source_id]['type']
        target_type = self.nodes[target_id]['type']
        self.adjacency[source_type][target_type].append(edge_id)
        
    def get_node_features(self, node_id: str) -> List[float]:
        """Extract numerical features for a node"""
        node = self.nodes[node_id]
        features = []
        
        # Position features
        features.extend([node.get('position_x', 0.0), node.get('position_y', 0.0)])
        
        # Sensor features
        features.append(float(node.get('has_sensor', False)))
        features.append(float(node.get('sensor_count', 0)))
        
        # Node type one-hot encoding
        type_encoding = [0.0] * len(self.node_types)
        if node['type'] in self.node_types:
            type_encoding[self.node_types.index(node['type'])] = 1.0
        features.extend(type_encoding)
        
        return features
        
    def get_edge_features(self, edge_id: str) -> List[float]:
        """Extract numerical features for an edge"""
        edge = self.edges[edge_id]
        features = []
        
        # Length feature
        features.append(edge.get('length', 1.0))
        
        # Edge type (all pipes for now)
        features.append(1.0)  # pipe indicator
        
        return features
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighboring nodes"""
        neighbors = []
        for edge_id, edge in self.edges.items():
            if edge['source'] == node_id:
                neighbors.append(edge['target'])
            elif edge['target'] == node_id:
                neighbors.append(edge['source'])
        return neighbors


class PipelineAnomalyDetector:
    """Anomaly detection model for pipeline networks"""
    
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pipeline_graph = None
        self.temporal_features = None
        
    def build_graph_from_pipeline_data(self, node_features: Dict, edge_features: Dict, 
                                     sensor_data: Dict, temporal_features: Dict):
        """Build graph representation from processed pipeline data"""
        graph = PipelineGraph()
        
        # Add nodes
        for node_id, node_data in node_features.items():
            graph.add_node(node_id, node_data)
        
        # Add edges
        for edge_id, edge_data in edge_features.items():
            source_id = edge_data['source']
            target_id = edge_data['target']
            
            # Only add edge if both nodes exist
            if source_id in node_features and target_id in node_features:
                graph.add_edge(edge_id, source_id, target_id, edge_data)
        
        self.pipeline_graph = graph
        self.temporal_features = temporal_features
        
        print(f"Built graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
    
    def create_anomaly_labels(self, injection_rate: float = 0.05) -> Dict[str, int]:
        """Create synthetic anomaly labels for training"""
        if not self.pipeline_graph:
            return {}
            
        labels = {}
        num_anomalies = int(len(self.pipeline_graph.edges) * injection_rate)
        
        # Random anomaly injection for demonstration
        edge_ids = list(self.pipeline_graph.edges.keys())
        anomaly_edges = random.sample(edge_ids, num_anomalies)
        
        for edge_id in edge_ids:
            labels[edge_id] = 1 if edge_id in anomaly_edges else 0
            
        print(f"Created {num_anomalies} synthetic anomalies out of {len(edge_ids)} edges")
        return labels
    
    def detect_anomalies_simple(self) -> Dict[str, float]:
        """Simple anomaly detection based on graph structure"""
        if not self.pipeline_graph:
            return {}
            
        anomaly_scores = {}
        
        # Calculate anomaly scores for edges based on:
        # 1. Length deviation from average
        # 2. Node connectivity patterns
        # 3. Sensor availability
        
        edge_lengths = [e.get('length', 1.0) for e in self.pipeline_graph.edges.values()]
        avg_length = sum(edge_lengths) / len(edge_lengths) if edge_lengths else 1.0
        
        for edge_id, edge in self.pipeline_graph.edges.items():
            score = 0.0
            
            # Length anomaly score
            length_diff = abs(edge.get('length', 1.0) - avg_length) / avg_length
            score += length_diff * 0.3
            
            # Connectivity anomaly score
            source_neighbors = len(self.pipeline_graph.get_neighbors(edge['source']))
            target_neighbors = len(self.pipeline_graph.get_neighbors(edge['target']))
            
            # Edges connecting poorly connected nodes are more suspicious
            connectivity_score = 1.0 / (1.0 + source_neighbors + target_neighbors)
            score += connectivity_score * 0.4
            
            # Sensor coverage score
            source_node = self.pipeline_graph.nodes[edge['source']]
            target_node = self.pipeline_graph.nodes[edge['target']]
            
            sensor_coverage = (source_node.get('has_sensor', False) + 
                             target_node.get('has_sensor', False)) / 2.0
            
            # Edges with no sensor coverage are more suspicious
            score += (1.0 - sensor_coverage) * 0.3
            
            anomaly_scores[edge_id] = score
            
        return anomaly_scores
    
    def detect_anomalies_temporal(self) -> Dict[str, float]:
        """Temporal anomaly detection using sensor data patterns"""
        if not self.temporal_features:
            return {}
            
        anomaly_scores = {}
        
        # Analyze temporal patterns for each sensor
        for sensor_name, features in self.temporal_features.items():
            if not features:
                continue
                
            # Calculate anomaly indicators
            std_values = [f['std'] for f in features]
            trend_values = [f['trend'] for f in features]
            range_values = [f['range'] for f in features]
            
            # High variability indicates potential anomalies
            avg_std = sum(std_values) / len(std_values) if std_values else 0
            avg_trend = abs(sum(trend_values) / len(trend_values)) if trend_values else 0
            avg_range = sum(range_values) / len(range_values) if range_values else 0
            
            # Combined anomaly score
            temporal_anomaly = avg_std * 0.4 + avg_trend * 0.3 + avg_range * 0.3
            anomaly_scores[sensor_name] = temporal_anomaly
            
        return anomaly_scores


class PipelineGraphHGT:
    """Integration layer between pipeline data and pyHGT"""
    
    def __init__(self):
        self.graph = None
        self.node_dict = {}
        self.edge_list = {}
        
    def convert_to_hgt_format(self, pipeline_graph: PipelineGraph):
        """Convert pipeline graph to pyHGT Graph format"""
        if not PYHGT_AVAILABLE:
            print("pyHGT not available, skipping conversion")
            return None
            
        # Create pyHGT Graph instance
        graph = Graph()
        
        # Add nodes to graph
        for node_id, node_data in pipeline_graph.nodes.items():
            node = {
                'id': node_id,
                'type': node_data['type']
            }
            
            # Add features
            features = pipeline_graph.get_node_features(node_id)
            for i, feature in enumerate(features):
                node[f'feature_{i}'] = feature
                
            graph.add_node(node)
        
        # Add edges to graph
        for edge_id, edge_data in pipeline_graph.edges.items():
            source_node = {
                'id': edge_data['source'],
                'type': pipeline_graph.nodes[edge_data['source']]['type']
            }
            target_node = {
                'id': edge_data['target'], 
                'type': pipeline_graph.nodes[edge_data['target']]['type']
            }
            
            graph.add_edge(source_node, target_node, time=0, relation_type='pipe')
        
        self.graph = graph
        return graph


def create_pipeline_anomaly_detection_system():
    """Create complete pipeline anomaly detection system"""
    print("Creating Pipeline Anomaly Detection System...")
    
    # Initialize data processor
    blueprint_path = "/home/runner/work/pyHGT/pyHGT/blueprint/0708YTS4.json"
    sensor_path = "/home/runner/work/pyHGT/pyHGT/StreamData/0708YTS4.csv"
    
    processor = PipelineDataProcessor(blueprint_path, sensor_path)
    node_features, edge_features, sensor_data, temporal_features = processor.process_all_data()
    
    # Initialize anomaly detector
    detector = PipelineAnomalyDetector()
    pipeline_graph = detector.build_graph_from_pipeline_data(
        node_features, edge_features, sensor_data, temporal_features
    )
    
    # Create synthetic anomaly labels for demonstration
    anomaly_labels = detector.create_anomaly_labels()
    
    # Detect anomalies using simple methods
    structural_anomalies = detector.detect_anomalies_simple()
    temporal_anomalies = detector.detect_anomalies_temporal()
    
    # Print results
    print("\n=== ANOMALY DETECTION RESULTS ===")
    print(f"Structural anomalies detected: {len(structural_anomalies)}")
    print(f"Temporal anomalies detected: {len(temporal_anomalies)}")
    
    # Show top anomalous edges
    if structural_anomalies:
        top_structural = sorted(structural_anomalies.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        print("\nTop structural anomalies:")
        for edge_id, score in top_structural:
            edge = pipeline_graph.edges[edge_id]
            print(f"  Edge {edge_id[:8]}... (score: {score:.3f}) "
                  f"from {edge['source'][:8]}... to {edge['target'][:8]}...")
    
    # Show top temporal anomalies
    if temporal_anomalies:
        top_temporal = sorted(temporal_anomalies.items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        print("\nTop temporal anomalies:")
        for sensor, score in top_temporal:
            print(f"  Sensor {sensor}: score {score:.3f}")
    
    return detector, pipeline_graph, structural_anomalies, temporal_anomalies


if __name__ == "__main__":
    # Test the complete system
    print("Testing Pipeline Anomaly Detection System...")
    detector, graph, structural, temporal = create_pipeline_anomaly_detection_system()
    
    print(f"\nSystem ready with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print("Anomaly detection capabilities:")
    print("  ✓ Structural anomaly detection")
    print("  ✓ Temporal anomaly detection") 
    print("  ✓ Graph-based analysis")
    print("  ✓ Synthetic anomaly injection")
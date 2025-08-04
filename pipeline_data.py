"""
Pipeline Data Processing Module for Gas Network Anomaly Detection

This module processes industrial gas pipeline network data for use with pyHGT.
It handles:
- Pipeline topology from JSON blueprint
- Sensor data from CSV files
- Conversion to pyHGT Graph format
"""

import json
import csv
import os
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Try to import packages, fallback to basic functionality if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available, using basic CSV processing")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available, using Python math")


class PipelineDataProcessor:
    """Process pipeline topology and sensor data for anomaly detection"""
    
    def __init__(self, blueprint_path: str, sensor_data_path: str):
        self.blueprint_path = blueprint_path
        self.sensor_data_path = sensor_data_path
        self.node_data = {}
        self.edge_data = {}
        self.sensor_data = {}
        self.node_types = ['Stream', 'VavlePro', 'Mixer', 'Tee']
        self.edge_types = ['pipe']
        
    def load_blueprint(self) -> Tuple[List[Dict], List[Dict]]:
        """Load pipeline topology from JSON blueprint"""
        with open(self.blueprint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = data.get('nodelist', [])
        edges = data.get('linklist', [])
        
        print(f"Loaded {len(nodes)} nodes and {len(edges)} edges from blueprint")
        return nodes, edges
    
    def load_sensor_data(self) -> Dict[str, List[Dict]]:
        """Load sensor data from CSV file"""
        sensor_data = {}
        
        if not os.path.exists(self.sensor_data_path):
            print(f"Warning: Sensor data file {self.sensor_data_path} not found")
            return sensor_data
        
        if PANDAS_AVAILABLE:
            # Use pandas for efficient data loading
            df = pd.read_csv(self.sensor_data_path)
            
            # Group by sensor (column names except timestamp)
            sensor_columns = [col for col in df.columns if col != 'timestamp']
            
            for col in sensor_columns:
                sensor_data[col] = []
                for idx, row in df.iterrows():
                    if pd.notna(row[col]):  # Skip NaN values
                        sensor_data[col].append({
                            'timestamp': row['timestamp'],
                            'value': row[col]
                        })
        else:
            # Basic CSV processing without pandas
            with open(self.sensor_data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames
                sensor_columns = [col for col in header if col != 'timestamp']
                
                # Initialize sensor data structure
                for col in sensor_columns:
                    sensor_data[col] = []
                
                # Read data row by row
                for row in reader:
                    for col in sensor_columns:
                        if row[col] and row[col].strip():  # Skip empty values
                            try:
                                value = float(row[col])
                                sensor_data[col].append({
                                    'timestamp': row['timestamp'],
                                    'value': value
                                })
                            except ValueError:
                                continue  # Skip invalid numeric values
        
        print(f"Loaded sensor data for {len(sensor_data)} sensors")
        for sensor, data in sensor_data.items():
            print(f"  {sensor}: {len(data)} readings")
            
        return sensor_data
    
    def extract_node_features(self, nodes: List[Dict]) -> Dict[str, Dict]:
        """Extract features from node data"""
        node_features = {}
        
        for node in nodes:
            node_id = node['id']
            node_type = 'Unknown'
            position = [0.0, 0.0]
            
            # Parse node parameters
            try:
                if isinstance(node.get('parameter', ''), str):
                    params = json.loads(node['parameter'])
                else:
                    params = node.get('parameter', {})
                
                # Extract node type
                node_type = params.get('type', 'Unknown')
                
                # Extract position
                if 'styles' in params and 'position' in params['styles']:
                    pos = params['styles']['position']
                    position = [float(pos.get('x', 0)), float(pos.get('y', 0))]
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Error parsing node {node_id} parameters: {e}")
            
            node_features[node_id] = {
                'id': node_id,
                'type': node_type,
                'position_x': position[0],
                'position_y': position[1],
                'has_sensor': False,
                'sensor_count': 0,
                'sensor_types': []
            }
        
        return node_features
    
    def extract_edge_features(self, edges: List[Dict]) -> Dict[str, Dict]:
        """Extract features from edge data"""
        edge_features = {}
        
        for edge in edges:
            edge_id = edge['id']
            source_id = edge.get('sourceid')
            target_id = edge.get('targetid')
            length = 1.0
            
            # Parse edge parameters
            try:
                if isinstance(edge.get('parameter', ''), str):
                    params = json.loads(edge['parameter'])
                else:
                    params = edge.get('parameter', {})
                
                # Extract pipe length
                if 'parameter' in params and 'Length' in params['parameter']:
                    length = float(params['parameter']['Length'])
                elif 'Length' in params:
                    length = float(params['Length'])
                
                if length <= 0:
                    length = 1.0
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Error parsing edge {edge_id} parameters: {e}")
            
            edge_features[edge_id] = {
                'id': edge_id,
                'source': source_id,
                'target': target_id,
                'length': length,
                'type': 'pipe'
            }
        
        return edge_features
    
    def map_sensors_to_nodes(self, node_features: Dict, sensor_data: Dict) -> Dict:
        """Map sensor data to nodes based on naming convention"""
        updated_nodes = node_features.copy()
        
        # Extract sensor mapping from sensor names
        # Format: YT.{node_id}{sensor_type}.PV
        for sensor_name, data in sensor_data.items():
            if not data:  # Skip sensors with no data
                continue
                
            # Parse sensor name to extract node and type
            parts = sensor_name.split('.')
            if len(parts) >= 2:
                # Extract potential node ID and sensor type from middle part
                middle_part = parts[1] if len(parts) > 1 else parts[0]
                
                # Try to match with existing nodes
                for node_id, node_info in updated_nodes.items():
                    # Check if sensor name contains node ID
                    if str(node_id) in middle_part or middle_part in str(node_id):
                        node_info['has_sensor'] = True
                        node_info['sensor_count'] += 1
                        
                        # Determine sensor type
                        sensor_type = 'unknown'
                        if 'PI' in middle_part or 'PIC' in middle_part:
                            sensor_type = 'pressure'
                        elif 'FI' in middle_part:
                            sensor_type = 'flow'
                        elif 'TI' in middle_part or 'TIC' in middle_part:
                            sensor_type = 'temperature'
                        
                        if sensor_type not in node_info['sensor_types']:
                            node_info['sensor_types'].append(sensor_type)
                        
                        # Add sensor data reference
                        if 'sensors' not in node_info:
                            node_info['sensors'] = {}
                        node_info['sensors'][sensor_name] = {
                            'type': sensor_type,
                            'data_count': len(data)
                        }
                        break
        
        # Statistics
        nodes_with_sensors = sum(1 for n in updated_nodes.values() if n['has_sensor'])
        print(f"Mapped sensors to {nodes_with_sensors} nodes out of {len(updated_nodes)} total nodes")
        
        return updated_nodes
    
    def create_temporal_features(self, sensor_data: Dict, window_size: int = 10) -> Dict:
        """Create temporal features from sensor data for anomaly detection"""
        temporal_features = {}
        
        for sensor_name, data in sensor_data.items():
            if len(data) < window_size:
                continue
                
            features = []
            values = [d['value'] for d in data]
            
            # Calculate sliding window statistics
            for i in range(len(values) - window_size + 1):
                window = values[i:i + window_size]
                
                # Basic statistics
                if NUMPY_AVAILABLE:
                    mean_val = np.mean(window)
                    std_val = np.std(window)
                    min_val = np.min(window)
                    max_val = np.max(window)
                else:
                    mean_val = sum(window) / len(window)
                    variance = sum((x - mean_val) ** 2 for x in window) / len(window)
                    std_val = math.sqrt(variance)
                    min_val = min(window)
                    max_val = max(window)
                
                # Trend features
                trend = (window[-1] - window[0]) / window_size
                
                features.append({
                    'timestamp': data[i + window_size - 1]['timestamp'],
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'range': max_val - min_val,
                    'trend': trend
                })
            
            temporal_features[sensor_name] = features
        
        return temporal_features
    
    def process_all_data(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """Process all pipeline data and return structured format"""
        print("Processing pipeline data...")
        
        # Load raw data
        nodes, edges = self.load_blueprint()
        sensor_data = self.load_sensor_data()
        
        # Extract features
        node_features = self.extract_node_features(nodes)
        edge_features = self.extract_edge_features(edges)
        
        # Map sensors to nodes
        node_features = self.map_sensors_to_nodes(node_features, sensor_data)
        
        # Create temporal features
        temporal_features = self.create_temporal_features(sensor_data)
        
        print("Data processing complete!")
        print(f"Nodes: {len(node_features)}")
        print(f"Edges: {len(edge_features)}")
        print(f"Sensors: {len(sensor_data)}")
        print(f"Temporal features: {len(temporal_features)}")
        
        # Print node type distribution
        type_counts = defaultdict(int)
        for node in node_features.values():
            type_counts[node['type']] += 1
        
        print("\nNode type distribution:")
        for node_type, count in type_counts.items():
            print(f"  {node_type}: {count}")
        
        return node_features, edge_features, sensor_data, temporal_features


def test_pipeline_data_processor():
    """Test the pipeline data processor with sample data"""
    blueprint_path = "/home/runner/work/pyHGT/pyHGT/blueprint/0708YTS4.json"
    sensor_path = "/home/runner/work/pyHGT/pyHGT/StreamData/0708YTS4.csv"
    
    processor = PipelineDataProcessor(blueprint_path, sensor_path)
    node_features, edge_features, sensor_data, temporal_features = processor.process_all_data()
    
    return processor, node_features, edge_features, sensor_data, temporal_features


if __name__ == "__main__":
    # Test the data processor
    print("Testing Pipeline Data Processor...")
    processor, nodes, edges, sensors, temporal = test_pipeline_data_processor()
    
    # Print some sample data
    print("\nSample node with sensor:")
    for node_id, node_info in nodes.items():
        if node_info['has_sensor']:
            print(f"Node {node_id}: {node_info}")
            break
    
    print("\nSample edge:")
    for edge_id, edge_info in list(edges.items())[:1]:
        print(f"Edge {edge_id}: {edge_info}")
"""
Pipeline Anomaly Detection Prediction and Visualization

This module provides real-time anomaly detection and visualization capabilities
for gas pipeline networks.
"""

import os
import sys
import json
import math
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Add pyHGT to path
sys.path.append('/home/runner/work/pyHGT/pyHGT')

# Import our modules
from pipeline_data import PipelineDataProcessor
from pipeline_hgt import PipelineAnomalyDetector
from train_pipeline_anomaly import PipelineAnomalyTrainer

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, using text-based visualization")


class PipelineAnomalyPredictor:
    """Real-time anomaly prediction for pipeline networks"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.trainer = None
        self.model = None
        self.pipeline_graph = None
        self.anomaly_threshold = 0.5
        
    def load_trained_model(self, trainer: PipelineAnomalyTrainer):
        """Load a pre-trained model"""
        self.trainer = trainer
        self.model = trainer.model
        self.pipeline_graph = trainer.pipeline_graph
        print("Model loaded successfully")
    
    def predict_edge_anomaly(self, edge_id: str) -> float:
        """Predict anomaly score for a specific edge"""
        if not self.model or not self.trainer:
            raise ValueError("Model not loaded")
        
        features = self.trainer.extract_edge_training_features(edge_id)
        
        # Use appropriate prediction method
        try:
            import torch
            score = self.trainer.predict_pytorch(features)
        except ImportError:
            score = self.model.forward(features)
        
        return score
    
    def predict_all_anomalies(self) -> Dict[str, float]:
        """Predict anomaly scores for all edges"""
        if not self.pipeline_graph:
            raise ValueError("Pipeline graph not loaded")
        
        anomaly_scores = {}
        
        for edge_id in self.pipeline_graph.edges:
            score = self.predict_edge_anomaly(edge_id)
            anomaly_scores[edge_id] = score
        
        return anomaly_scores
    
    def get_anomalous_edges(self, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """Get edges that are classified as anomalous"""
        if threshold is None:
            threshold = self.anomaly_threshold
        
        scores = self.predict_all_anomalies()
        anomalous = [(edge_id, score) for edge_id, score in scores.items() 
                    if score > threshold]
        
        # Sort by score descending
        anomalous.sort(key=lambda x: x[1], reverse=True)
        
        return anomalous
    
    def get_edge_details(self, edge_id: str) -> Dict[str, Any]:
        """Get detailed information about an edge"""
        if not self.pipeline_graph:
            return {}
        
        edge = self.pipeline_graph.edges.get(edge_id, {})
        if not edge:
            return {}
        
        source_node = self.pipeline_graph.nodes.get(edge['source'], {})
        target_node = self.pipeline_graph.nodes.get(edge['target'], {})
        
        return {
            'edge_id': edge_id,
            'length': edge.get('length', 0),
            'source': {
                'id': edge['source'],
                'type': source_node.get('type', 'Unknown'),
                'position': [source_node.get('position_x', 0), 
                           source_node.get('position_y', 0)],
                'has_sensor': source_node.get('has_sensor', False),
                'sensor_count': source_node.get('sensor_count', 0)
            },
            'target': {
                'id': edge['target'],
                'type': target_node.get('type', 'Unknown'),
                'position': [target_node.get('position_x', 0), 
                           target_node.get('position_y', 0)],
                'has_sensor': target_node.get('has_sensor', False),
                'sensor_count': target_node.get('sensor_count', 0)
            }
        }
    
    def generate_anomaly_report(self, top_n: int = 20) -> Dict[str, Any]:
        """Generate comprehensive anomaly detection report"""
        print("Generating anomaly detection report...")
        
        anomalous_edges = self.get_anomalous_edges()
        all_scores = self.predict_all_anomalies()
        
        # Statistics
        total_edges = len(all_scores)
        num_anomalous = len(anomalous_edges)
        anomaly_rate = num_anomalous / total_edges if total_edges > 0 else 0
        
        # Score statistics
        scores = list(all_scores.values())
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        
        # Top anomalies with details
        top_anomalies = []
        for edge_id, score in anomalous_edges[:top_n]:
            details = self.get_edge_details(edge_id)
            details['anomaly_score'] = score
            top_anomalies.append(details)
        
        # Node type analysis
        node_type_anomalies = defaultdict(int)
        for edge_id, _ in anomalous_edges:
            edge = self.pipeline_graph.edges[edge_id]
            source_type = self.pipeline_graph.nodes[edge['source']]['type']
            target_type = self.pipeline_graph.nodes[edge['target']]['type']
            
            node_type_anomalies[f"{source_type}-{target_type}"] += 1
        
        report = {
            'summary': {
                'total_edges': total_edges,
                'anomalous_edges': num_anomalous,
                'anomaly_rate': anomaly_rate,
                'detection_threshold': self.anomaly_threshold
            },
            'score_statistics': {
                'average': avg_score,
                'maximum': max_score,
                'minimum': min_score
            },
            'top_anomalies': top_anomalies,
            'node_type_patterns': dict(node_type_anomalies),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return report


class PipelineAnomalyVisualizer:
    """Visualization tools for pipeline anomaly detection"""
    
    def __init__(self, predictor: PipelineAnomalyPredictor):
        self.predictor = predictor
        self.pipeline_graph = predictor.pipeline_graph
    
    def print_text_visualization(self, report: Dict[str, Any]):
        """Create text-based visualization of anomaly detection results"""
        print("\n" + "="*80)
        print("           PIPELINE ANOMALY DETECTION REPORT")
        print("="*80)
        
        # Summary
        summary = report['summary']
        print(f"\n📊 DETECTION SUMMARY")
        print(f"   Total Edges Analyzed: {summary['total_edges']}")
        print(f"   Anomalous Edges Found: {summary['anomalous_edges']}")
        print(f"   Anomaly Rate: {summary['anomaly_rate']:.1%}")
        print(f"   Detection Threshold: {summary['detection_threshold']:.3f}")
        
        # Score statistics
        stats = report['score_statistics']
        print(f"\n📈 SCORE STATISTICS")
        print(f"   Average Score: {stats['average']:.3f}")
        print(f"   Maximum Score: {stats['maximum']:.3f}")
        print(f"   Minimum Score: {stats['minimum']:.3f}")
        
        # Top anomalies
        print(f"\n🚨 TOP ANOMALOUS EDGES")
        print("   Rank | Score | Length | Source Type -> Target Type | Has Sensors")
        print("   " + "-"*70)
        
        for i, anomaly in enumerate(report['top_anomalies'][:15], 1):
            source = anomaly['source']
            target = anomaly['target']
            sensors = "✓" if (source['has_sensor'] or target['has_sensor']) else "✗"
            
            print(f"   {i:2d}   | {anomaly['anomaly_score']:.3f} | {anomaly['length']:6.1f} | "
                  f"{source['type']:8s} -> {target['type']:8s} | {sensors:^11s}")
        
        # Node type patterns
        if report['node_type_patterns']:
            print(f"\n🔗 ANOMALY PATTERNS BY NODE TYPES")
            for pattern, count in sorted(report['node_type_patterns'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:
                print(f"   {pattern:20s}: {count:2d} anomalies")
        
        print(f"\n📅 Report generated: {report['timestamp']}")
        print("="*80)
    
    def create_matplotlib_visualization(self, report: Dict[str, Any], 
                                      save_path: str = "anomaly_visualization.png"):
        """Create matplotlib visualization if available"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping graphical visualization")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Pipeline Anomaly Detection Report', fontsize=16, fontweight='bold')
        
        # 1. Anomaly score distribution
        scores = [anomaly['anomaly_score'] for anomaly in report['top_anomalies']]
        if scores:
            ax1.hist(scores, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax1.axvline(self.predictor.anomaly_threshold, color='green', 
                       linestyle='--', label=f'Threshold ({self.predictor.anomaly_threshold})')
            ax1.set_xlabel('Anomaly Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Anomaly Score Distribution')
            ax1.legend()
        
        # 2. Edge length vs anomaly score
        lengths = [anomaly['length'] for anomaly in report['top_anomalies']]
        if lengths and scores:
            ax2.scatter(lengths, scores, alpha=0.6, color='red')
            ax2.set_xlabel('Edge Length')
            ax2.set_ylabel('Anomaly Score')
            ax2.set_title('Edge Length vs Anomaly Score')
        
        # 3. Node type patterns
        patterns = report['node_type_patterns']
        if patterns:
            pattern_names = list(patterns.keys())[:10]
            pattern_counts = [patterns[name] for name in pattern_names]
            
            bars = ax3.bar(range(len(pattern_names)), pattern_counts, color='orange', alpha=0.7)
            ax3.set_xlabel('Node Type Connections')
            ax3.set_ylabel('Number of Anomalies')
            ax3.set_title('Anomalies by Node Type Patterns')
            ax3.set_xticks(range(len(pattern_names)))
            ax3.set_xticklabels(pattern_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, pattern_counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        # 4. Summary statistics
        ax4.axis('off')
        summary_text = f"""
        DETECTION SUMMARY
        
        Total Edges: {report['summary']['total_edges']}
        Anomalous Edges: {report['summary']['anomalous_edges']}
        Anomaly Rate: {report['summary']['anomaly_rate']:.1%}
        
        SCORE STATISTICS
        
        Average: {report['score_statistics']['average']:.3f}
        Maximum: {report['score_statistics']['maximum']:.3f}
        Minimum: {report['score_statistics']['minimum']:.3f}
        
        Generated: {report['timestamp']}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        
        return fig


def run_anomaly_detection_demo():
    """Run complete anomaly detection demonstration"""
    print("=== PIPELINE ANOMALY DETECTION DEMONSTRATION ===")
    
    # Train model
    print("\n1. Training anomaly detection model...")
    trainer = PipelineAnomalyTrainer()
    trainer.load_and_process_data()
    features, labels = trainer.create_training_data(anomaly_rate=0.12)
    trainer.train_model(features, labels, epochs=30)
    
    # Initialize predictor
    print("\n2. Initializing predictor...")
    predictor = PipelineAnomalyPredictor()
    predictor.load_trained_model(trainer)
    
    # Generate anomaly report
    print("\n3. Generating anomaly detection report...")
    report = predictor.generate_anomaly_report(top_n=25)
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    visualizer = PipelineAnomalyVisualizer(predictor)
    
    # Text visualization (always available)
    visualizer.print_text_visualization(report)
    
    # Graphical visualization (if matplotlib available)
    if MATPLOTLIB_AVAILABLE:
        visualizer.create_matplotlib_visualization(report)
    
    # Additional analysis
    print("\n5. Additional Analysis...")
    
    # Identify critical paths (edges with high anomaly scores and high connectivity)
    anomalous_edges = predictor.get_anomalous_edges(threshold=0.3)
    
    if anomalous_edges:
        print(f"\n🔍 CRITICAL ANALYSIS")
        print(f"   Found {len(anomalous_edges)} edges with scores > 0.3")
        
        # Check for clusters of anomalies
        node_anomaly_count = defaultdict(int)
        for edge_id, score in anomalous_edges:
            edge = predictor.pipeline_graph.edges[edge_id]
            node_anomaly_count[edge['source']] += 1
            node_anomaly_count[edge['target']] += 1
        
        # Find nodes with multiple anomalous connections
        critical_nodes = [(node_id, count) for node_id, count in node_anomaly_count.items() 
                         if count >= 2]
        
        if critical_nodes:
            critical_nodes.sort(key=lambda x: x[1], reverse=True)
            print(f"\n   Critical nodes (involved in multiple anomalies):")
            for node_id, count in critical_nodes[:10]:
                node = predictor.pipeline_graph.nodes[node_id]
                print(f"     Node {node_id[:8]}... ({node['type']}): {count} anomalous connections")
    
    print(f"\n✅ Anomaly detection demonstration completed!")
    print(f"   The system is ready for real-time pipeline monitoring.")
    
    return predictor, report, visualizer


if __name__ == "__main__":
    # Run the complete demonstration
    predictor, report, visualizer = run_anomaly_detection_demo()
    
    print(f"\n💡 USAGE EXAMPLES:")
    print(f"   # Predict single edge:")
    print(f"   # score = predictor.predict_edge_anomaly('edge_id')")
    print(f"   # ")
    print(f"   # Get all anomalous edges:")
    print(f"   # anomalies = predictor.get_anomalous_edges(threshold=0.4)")
    print(f"   # ")
    print(f"   # Generate new report:")
    print(f"   # report = predictor.generate_anomaly_report(top_n=15)")
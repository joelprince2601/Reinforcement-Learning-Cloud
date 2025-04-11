from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

@dataclass
class DatasetConfig:
    """Configuration for the cloud security dataset"""
    # Dataset characteristics
    name: str = "CloudSec-RL-2024"  # Custom dataset name
    total_samples: int = 100000      # Total number of environment states
    feature_count: int = 14          # Total features across all metrics
    
    # Feature groups
    network_features: List[str] = [
        "packet_rate",
        "bytes_sent",
        "bytes_received",
        "connections"
    ]
    
    user_features: List[str] = [
        "login_attempts",
        "active_sessions",
        "failed_logins",
        "suspicious_activities"
    ]
    
    system_features: List[str] = [
        "cpu_utilization",
        "memory_usage",
        "disk_io",
        "network_io"
    ]
    
    anomaly_features: List[str] = [
        "network_anomaly",
        "user_anomaly"
    ]
    
    # Attack distribution (class balance)
    attack_distribution: Dict[str, float] = {
        "normal": 0.70,          # 70% normal traffic
        "ddos": 0.10,           # 10% DDoS attacks
        "brute_force": 0.05,    # 5% brute force attacks
        "data_exfil": 0.05,     # 5% data exfiltration
        "crypto_mining": 0.05,  # 5% crypto mining
        "port_scan": 0.05       # 5% port scanning
    }

class EvaluationMetrics:
    """Class to calculate and store evaluation metrics"""
    def __init__(self):
        self.episode_rewards = []
        self.true_labels = []
        self.predicted_labels = []
        self.false_positive_rates = []
        self.detection_latency = []
        self.mitigation_success = []
    
    def update_metrics(
        self,
        true_label: str,
        predicted_label: str,
        reward: float,
        detection_time: float,
        mitigation_success: bool
    ):
        """Update metrics with new data point"""
        self.true_labels.append(true_label)
        self.predicted_labels.append(predicted_label)
        self.episode_rewards.append(reward)
        self.detection_latency.append(detection_time)
        self.mitigation_success.append(int(mitigation_success))
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        metrics = {
            # Classification metrics
            "accuracy": accuracy_score(self.true_labels, self.predicted_labels),
            "precision": precision_score(self.true_labels, self.predicted_labels, average='weighted'),
            "recall": recall_score(self.true_labels, self.predicted_labels, average='weighted'),
            "f1_score": f1_score(self.true_labels, self.predicted_labels, average='weighted'),
            
            # RL metrics
            "average_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            
            # Security-specific metrics
            "false_positive_rate": self._calculate_fpr(),
            "average_detection_latency": np.mean(self.detection_latency),
            "mitigation_success_rate": np.mean(self.mitigation_success),
        }
        return metrics
    
    def _calculate_fpr(self) -> float:
        """Calculate False Positive Rate"""
        tn, fp, fn, tp = confusion_matrix(
            [1 if label != "normal" else 0 for label in self.true_labels],
            [1 if label != "normal" else 0 for label in self.predicted_labels]
        ).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def print_metrics(self):
        """Print all metrics in a formatted way"""
        metrics = self.calculate_metrics()
        print("\n=== Evaluation Metrics ===")
        print(f"Classification Metrics:")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1-Score:   {metrics['f1_score']:.4f}")
        print(f"\nRL Performance:")
        print(f"  Avg Reward: {metrics['average_reward']:.4f} ± {metrics['std_reward']:.4f}")
        print(f"\nSecurity KPIs:")
        print(f"  False Positive Rate:      {metrics['false_positive_rate']:.4f}")
        print(f"  Avg Detection Latency:    {metrics['average_detection_latency']:.2f}s")
        print(f"  Mitigation Success Rate:  {metrics['mitigation_success_rate']:.4f}") 
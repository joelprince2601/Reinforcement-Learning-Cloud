import psutil
import time
from dataclasses import dataclass
from typing import Dict, List, Any
import random
from collections import deque
import numpy as np

@dataclass
class NetworkMetrics:
    packet_rate: float
    bytes_sent: int
    bytes_received: int
    connections: int

@dataclass
class UserActivity:
    login_attempts: int
    active_sessions: int
    failed_logins: int
    suspicious_activities: int

@dataclass
class SystemResources:
    cpu_utilization: float
    memory_usage: float
    disk_io: float
    network_io: float

class CloudEnvironmentSimulator:
    def __init__(self):
        self.network_history = deque(maxlen=100)  # Store last 100 network states
        self.user_history = deque(maxlen=100)     # Store last 100 user activities
        self.resource_history = deque(maxlen=100)  # Store last 100 resource states
        
        # Initialize baseline metrics
        self.baseline_network_traffic = 1000  # packets/sec
        self.baseline_user_activity = 10      # normal login attempts/min
        self.baseline_cpu_usage = 30.0        # percentage
        
    def collect_environment_state(self) -> Dict[str, Any]:
        """
        Collect and return the current state of the cloud environment.
        This function simulates collecting real metrics from a cloud environment.
        
        Returns:
            Dict containing the current state features including:
            - Network metrics
            - User activity
            - System resources
            - Historical patterns
            - Anomaly indicators
        """
        # Simulate network metrics collection
        network_metrics = self._collect_network_metrics()
        
        # Simulate user activity monitoring
        user_activity = self._collect_user_activity()
        
        # Simulate system resource monitoring
        system_resources = self._collect_system_resources()
        
        # Calculate historical patterns and anomaly scores
        anomaly_scores = self._calculate_anomaly_scores(
            network_metrics, user_activity, system_resources
        )
        
        # Update historical data
        self.network_history.append(network_metrics)
        self.user_history.append(user_activity)
        self.resource_history.append(system_resources)
        
        return {
            "timestamp": time.time(),
            "network_metrics": {
                "packet_rate": network_metrics.packet_rate,
                "bytes_sent": network_metrics.bytes_sent,
                "bytes_received": network_metrics.bytes_received,
                "connections": network_metrics.connections
            },
            "user_activity": {
                "login_attempts": user_activity.login_attempts,
                "active_sessions": user_activity.active_sessions,
                "failed_logins": user_activity.failed_logins,
                "suspicious_activities": user_activity.suspicious_activities
            },
            "system_resources": {
                "cpu_utilization": system_resources.cpu_utilization,
                "memory_usage": system_resources.memory_usage,
                "disk_io": system_resources.disk_io,
                "network_io": system_resources.network_io
            },
            "anomaly_scores": anomaly_scores
        }

    def _collect_network_metrics(self) -> NetworkMetrics:
        """Simulate collecting network metrics"""
        # In a real environment, this would collect actual network metrics
        # Here we simulate with some random variations around baseline
        return NetworkMetrics(
            packet_rate=self.baseline_network_traffic * (1 + random.uniform(-0.2, 0.2)),
            bytes_sent=int(random.uniform(1000, 10000)),
            bytes_received=int(random.uniform(1000, 10000)),
            connections=int(random.uniform(10, 100))
        )

    def _collect_user_activity(self) -> UserActivity:
        """Simulate collecting user activity metrics"""
        return UserActivity(
            login_attempts=int(self.baseline_user_activity * (1 + random.uniform(-0.3, 0.3))),
            active_sessions=int(random.uniform(5, 20)),
            failed_logins=int(random.uniform(0, 5)),
            suspicious_activities=int(random.uniform(0, 3))
        )

    def _collect_system_resources(self) -> SystemResources:
        """Collect actual system resources using psutil"""
        return SystemResources(
            cpu_utilization=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_io=psutil.disk_usage('/').percent,
            network_io=random.uniform(0, 100)  # Simulated network I/O
        )

    def _calculate_anomaly_scores(
        self,
        network_metrics: NetworkMetrics,
        user_activity: UserActivity,
        system_resources: SystemResources
    ) -> Dict[str, float]:
        """
        Calculate anomaly scores based on current metrics and historical patterns
        Returns a dictionary of anomaly scores for different aspects of the system
        """
        scores = {
            "network_anomaly": self._calculate_network_anomaly(network_metrics),
            "user_anomaly": self._calculate_user_anomaly(user_activity),
            "resource_anomaly": self._calculate_resource_anomaly(system_resources)
        }
        return scores

    def _calculate_network_anomaly(self, current_metrics: NetworkMetrics) -> float:
        """Calculate network-based anomaly score"""
        if not self.network_history:
            return 0.0
        
        # Compare current packet rate with historical average
        historical_rates = [n.packet_rate for n in self.network_history]
        mean_rate = np.mean(historical_rates)
        std_rate = np.std(historical_rates) if len(historical_rates) > 1 else 1.0
        
        z_score = abs(current_metrics.packet_rate - mean_rate) / std_rate
        return min(1.0, z_score / 3.0)  # Normalize to [0,1]

    def _calculate_user_anomaly(self, current_activity: UserActivity) -> float:
        """Calculate user activity-based anomaly score"""
        # Simple anomaly score based on failed logins and suspicious activities
        return min(1.0, (current_activity.failed_logins * 0.3 + 
                        current_activity.suspicious_activities * 0.7) / 10.0)

    def _calculate_resource_anomaly(self, current_resources: SystemResources) -> float:
        """Calculate resource usage-based anomaly score"""
        # Simple threshold-based anomaly detection
        cpu_score = current_resources.cpu_utilization / 100.0
        memory_score = current_resources.memory_usage / 100.0
        return max(cpu_score, memory_score) 
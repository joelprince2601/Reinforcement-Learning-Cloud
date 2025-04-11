from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class DeploymentConfig:
    """Configuration for real-world deployment"""
    
    # Distributed setup
    num_agents: int = int(os.getenv("NUM_AGENTS", "3"))
    sync_interval: int = 100  # Steps between model synchronization
    max_concurrent_actions: int = 5
    
    # Real-time constraints
    max_inference_latency_ms: float = 100.0  # Maximum allowed inference time
    max_memory_usage_mb: float = 2048.0      # Maximum memory usage per agent
    max_cpu_usage_percent: float = 50.0      # Maximum CPU usage per agent
    
    # Model retraining
    performance_threshold: float = 0.8        # Threshold for triggering retraining
    min_samples_before_retrain: int = 10000   # Minimum samples needed before retraining
    max_retrain_frequency_hours: int = 24     # Minimum hours between retraining
    
    # Resource scaling
    auto_scaling: bool = True
    min_agents: int = 2
    max_agents: int = 10
    scale_up_threshold: float = 0.8   # CPU/Memory threshold for scaling up
    scale_down_threshold: float = 0.3  # CPU/Memory threshold for scaling down
    
    # Monitoring and logging
    metrics_collection_interval: int = 60     # Seconds between metrics collection
    log_level: str = "INFO"
    enable_performance_logging: bool = True
    enable_action_logging: bool = True
    
    # Multi-tenant settings
    max_tenants_per_agent: int = 5
    tenant_isolation_level: str = "strict"  # strict/moderate/flexible
    
    # Fallback and safety
    enable_fallback_actions: bool = True
    fallback_action_timeout_ms: float = 50.0
    max_consecutive_timeouts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "distributed": {
                "num_agents": self.num_agents,
                "sync_interval": self.sync_interval,
                "max_concurrent_actions": self.max_concurrent_actions
            },
            "real_time": {
                "max_inference_latency_ms": self.max_inference_latency_ms,
                "max_memory_usage_mb": self.max_memory_usage_mb,
                "max_cpu_usage_percent": self.max_cpu_usage_percent
            },
            "retraining": {
                "performance_threshold": self.performance_threshold,
                "min_samples_before_retrain": self.min_samples_before_retrain,
                "max_retrain_frequency_hours": self.max_retrain_frequency_hours
            },
            "scaling": {
                "auto_scaling": self.auto_scaling,
                "min_agents": self.min_agents,
                "max_agents": self.max_agents,
                "scale_thresholds": {
                    "up": self.scale_up_threshold,
                    "down": self.scale_down_threshold
                }
            },
            "monitoring": {
                "metrics_interval": self.metrics_collection_interval,
                "log_level": self.log_level,
                "performance_logging": self.enable_performance_logging,
                "action_logging": self.enable_action_logging
            },
            "multi_tenant": {
                "max_tenants_per_agent": self.max_tenants_per_agent,
                "isolation_level": self.tenant_isolation_level
            },
            "safety": {
                "enable_fallback": self.enable_fallback_actions,
                "fallback_timeout_ms": self.fallback_action_timeout_ms,
                "max_consecutive_timeouts": self.max_consecutive_timeouts
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DeploymentConfig':
        """Create config from dictionary"""
        instance = cls()
        
        if "distributed" in config_dict:
            instance.num_agents = config_dict["distributed"].get("num_agents", instance.num_agents)
            instance.sync_interval = config_dict["distributed"].get("sync_interval", instance.sync_interval)
            instance.max_concurrent_actions = config_dict["distributed"].get("max_concurrent_actions", instance.max_concurrent_actions)
        
        if "real_time" in config_dict:
            instance.max_inference_latency_ms = config_dict["real_time"].get("max_inference_latency_ms", instance.max_inference_latency_ms)
            instance.max_memory_usage_mb = config_dict["real_time"].get("max_memory_usage_mb", instance.max_memory_usage_mb)
            instance.max_cpu_usage_percent = config_dict["real_time"].get("max_cpu_usage_percent", instance.max_cpu_usage_percent)
        
        # Add other config sections as needed
        return instance 
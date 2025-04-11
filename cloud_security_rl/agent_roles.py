from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any

class AgentRole(Enum):
    """Different roles for specialized agents"""
    NETWORK_MONITOR = "network_monitor"
    USER_ACTIVITY = "user_activity"
    RESOURCE_MONITOR = "resource_monitor"
    ATTACK_DETECTOR = "attack_detector"
    RESPONSE_COORDINATOR = "response_coordinator"

@dataclass
class AgentSpecialization:
    """Specialization configuration for different agent roles"""
    
    # Features this agent focuses on
    primary_features: List[str]
    
    # Action weights (how much importance to give different action types)
    action_weights: Dict[str, float]
    
    # Monitoring region (e.g., specific VPC, subnet, or service)
    monitoring_region: str
    
    # Alert thresholds specific to this role
    alert_thresholds: Dict[str, float]

class AgentRoleManager:
    """Manages different agent roles and their specializations"""
    
    def __init__(self):
        self.role_configs = {
            AgentRole.NETWORK_MONITOR: AgentSpecialization(
                primary_features=[
                    "packet_rate",
                    "bytes_sent",
                    "bytes_received",
                    "connections"
                ],
                action_weights={
                    "BLOCK_IP": 1.0,
                    "RESET_CONNECTIONS": 0.8,
                    "SCALE_RESOURCES": 0.6,
                    "ENABLE_CAPTCHA": 0.2
                },
                monitoring_region="network",
                alert_thresholds={
                    "packet_rate_threshold": 10000,
                    "connection_spike_threshold": 100,
                    "bandwidth_threshold_mbps": 1000
                }
            ),
            
            AgentRole.USER_ACTIVITY: AgentSpecialization(
                primary_features=[
                    "login_attempts",
                    "active_sessions",
                    "failed_logins",
                    "suspicious_activities"
                ],
                action_weights={
                    "ENABLE_CAPTCHA": 1.0,
                    "BLOCK_IP": 0.8,
                    "RESET_CONNECTIONS": 0.6,
                    "SCALE_RESOURCES": 0.2
                },
                monitoring_region="auth",
                alert_thresholds={
                    "failed_login_threshold": 5,
                    "suspicious_activity_threshold": 3,
                    "session_anomaly_threshold": 0.8
                }
            ),
            
            AgentRole.RESOURCE_MONITOR: AgentSpecialization(
                primary_features=[
                    "cpu_utilization",
                    "memory_usage",
                    "disk_io",
                    "network_io"
                ],
                action_weights={
                    "SCALE_RESOURCES": 1.0,
                    "RESET_CONNECTIONS": 0.6,
                    "BLOCK_IP": 0.4,
                    "ENABLE_CAPTCHA": 0.2
                },
                monitoring_region="compute",
                alert_thresholds={
                    "cpu_threshold": 80.0,
                    "memory_threshold": 85.0,
                    "io_threshold": 90.0
                }
            ),
            
            AgentRole.ATTACK_DETECTOR: AgentSpecialization(
                primary_features=[
                    "network_anomaly",
                    "user_anomaly",
                    "resource_anomaly",
                    "attack_signatures"
                ],
                action_weights={
                    "BLOCK_IP": 1.0,
                    "RESET_CONNECTIONS": 0.9,
                    "ENABLE_CAPTCHA": 0.7,
                    "SCALE_RESOURCES": 0.5
                },
                monitoring_region="security",
                alert_thresholds={
                    "anomaly_threshold": 0.7,
                    "attack_confidence_threshold": 0.8,
                    "false_positive_threshold": 0.1
                }
            ),
            
            AgentRole.RESPONSE_COORDINATOR: AgentSpecialization(
                primary_features=[
                    "attack_type",
                    "attack_severity",
                    "system_state",
                    "mitigation_status"
                ],
                action_weights={
                    "COORDINATE_RESPONSE": 1.0,
                    "ESCALATE_ALERT": 0.9,
                    "UPDATE_POLICY": 0.8,
                    "ROLLBACK_CHANGES": 0.7
                },
                monitoring_region="global",
                alert_thresholds={
                    "severity_threshold": 0.8,
                    "escalation_threshold": 0.9,
                    "coordination_threshold": 0.7
                }
            )
        }
    
    def get_role_config(self, role: AgentRole) -> AgentSpecialization:
        """Get configuration for a specific role"""
        return self.role_configs[role]
    
    def assign_role(self, agent_id: int) -> AgentRole:
        """Assign a role to an agent based on its ID"""
        roles = list(AgentRole)
        return roles[agent_id % len(roles)]
    
    def get_agent_config(self, agent_id: int) -> Dict[str, Any]:
        """Get complete configuration for an agent"""
        role = self.assign_role(agent_id)
        spec = self.get_role_config(role)
        
        return {
            "agent_id": agent_id,
            "role": role.value,
            "specialization": {
                "features": spec.primary_features,
                "weights": spec.action_weights,
                "region": spec.monitoring_region,
                "thresholds": spec.alert_thresholds
            }
        } 
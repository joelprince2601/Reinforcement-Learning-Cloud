from dataclasses import dataclass
from enum import Enum
import random
import numpy as np
from typing import Dict, Any, List, Optional

class AttackType(Enum):
    DDOS = "ddos"
    BRUTE_FORCE = "brute_force"
    DATA_EXFILTRATION = "data_exfiltration"
    CRYPTO_MINING = "crypto_mining"
    PORT_SCAN = "port_scan"

@dataclass
class AttackImpact:
    network_impact: Dict[str, float]
    system_impact: Dict[str, float]
    user_impact: Dict[str, float]

class AttackSimulator:
    def __init__(self):
        self.active_attacks: Dict[AttackType, float] = {}  # Attack type -> intensity
        self._configure_attack_patterns()
    
    def _configure_attack_patterns(self):
        """Configure the impact patterns for different types of attacks"""
        self.attack_patterns = {
            AttackType.DDOS: AttackImpact(
                network_impact={
                    "packet_rate": 5.0,  # 5x normal traffic
                    "connections": 10.0,  # 10x normal connections
                },
                system_impact={
                    "cpu_utilization": 2.0,  # 2x normal CPU
                    "memory_usage": 1.5,    # 1.5x normal memory
                },
                user_impact={
                    "active_sessions": 0.5,  # Reduced legitimate sessions
                }
            ),
            AttackType.BRUTE_FORCE: AttackImpact(
                network_impact={
                    "packet_rate": 1.2,
                },
                system_impact={
                    "cpu_utilization": 1.1,
                },
                user_impact={
                    "failed_logins": 20.0,
                    "suspicious_activities": 5.0,
                }
            ),
            AttackType.DATA_EXFILTRATION: AttackImpact(
                network_impact={
                    "bytes_sent": 3.0,
                    "connections": 1.5,
                },
                system_impact={
                    "disk_io": 2.0,
                },
                user_impact={
                    "suspicious_activities": 2.0,
                }
            ),
            AttackType.CRYPTO_MINING: AttackImpact(
                network_impact={
                    "bytes_sent": 1.2,
                    "bytes_received": 1.2,
                },
                system_impact={
                    "cpu_utilization": 3.0,
                    "memory_usage": 1.5,
                },
                user_impact={}
            ),
            AttackType.PORT_SCAN: AttackImpact(
                network_impact={
                    "packet_rate": 2.0,
                    "connections": 3.0,
                },
                system_impact={},
                user_impact={
                    "suspicious_activities": 1.5,
                }
            )
        }
    
    def start_attack(self, attack_type: AttackType, intensity: float = 1.0):
        """Start a new attack with given intensity (0.0 to 1.0)"""
        self.active_attacks[attack_type] = max(0.0, min(1.0, intensity))
    
    def stop_attack(self, attack_type: AttackType):
        """Stop an ongoing attack"""
        self.active_attacks.pop(attack_type, None)
    
    def stop_all_attacks(self):
        """Stop all ongoing attacks"""
        self.active_attacks.clear()
    
    def apply_attack_effects(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Apply effects of active attacks to the given metrics"""
        if not self.active_attacks:
            return metrics
        
        modified_metrics = metrics.copy()
        
        for attack_type, intensity in self.active_attacks.items():
            pattern = self.attack_patterns[attack_type]
            
            # Apply network impacts
            for metric, multiplier in pattern.network_impact.items():
                if metric in modified_metrics["network_metrics"]:
                    modified_metrics["network_metrics"][metric] *= (1 + (multiplier - 1) * intensity)
            
            # Apply system impacts
            for metric, multiplier in pattern.system_impact.items():
                if metric in modified_metrics["system_resources"]:
                    modified_metrics["system_resources"][metric] *= (1 + (multiplier - 1) * intensity)
            
            # Apply user impacts
            for metric, multiplier in pattern.user_impact.items():
                if metric in modified_metrics["user_activity"]:
                    modified_metrics["user_activity"][metric] *= (1 + (multiplier - 1) * intensity)
        
        return modified_metrics

    def get_random_attack(self) -> tuple[AttackType, float]:
        """Generate a random attack with random intensity"""
        attack_type = random.choice(list(AttackType))
        intensity = random.uniform(0.3, 1.0)
        return attack_type, intensity 
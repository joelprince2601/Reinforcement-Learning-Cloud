from dataclasses import dataclass
from enum import Enum
import random
import numpy as np
from typing import Dict, Any, List, Optional
import time

class AdvancedAttackType(Enum):
    APT = "advanced_persistent_threat"
    ZERO_DAY = "zero_day_exploit"
    RANSOMWARE = "ransomware"
    POLYMORPHIC_MALWARE = "polymorphic_malware"
    SUPPLY_CHAIN = "supply_chain_attack"
    INSIDER_THREAT = "insider_threat"

@dataclass
class AttackPhase:
    name: str
    duration: int  # in steps
    network_pattern: Dict[str, float]
    system_pattern: Dict[str, float]
    user_pattern: Dict[str, float]

class AdvancedAttackSimulator:
    def __init__(self):
        self.active_attacks: Dict[AdvancedAttackType, Dict[str, Any]] = {}
        self._configure_attack_patterns()
        
    def _configure_attack_patterns(self):
        """Configure sophisticated multi-phase attack patterns"""
        self.attack_patterns = {
            AdvancedAttackType.APT: {
                "phases": [
                    AttackPhase(
                        name="reconnaissance",
                        duration=10,
                        network_pattern={
                            "packet_rate": 1.2,
                            "connections": 1.5
                        },
                        system_pattern={
                            "cpu_utilization": 1.1
                        },
                        user_pattern={
                            "suspicious_activities": 1.2
                        }
                    ),
                    AttackPhase(
                        name="infiltration",
                        duration=15,
                        network_pattern={
                            "bytes_received": 2.0,
                            "connections": 2.0
                        },
                        system_pattern={
                            "memory_usage": 1.5,
                            "disk_io": 1.3
                        },
                        user_pattern={
                            "failed_logins": 3.0
                        }
                    ),
                    AttackPhase(
                        name="exfiltration",
                        duration=20,
                        network_pattern={
                            "bytes_sent": 5.0,
                            "connections": 3.0
                        },
                        system_pattern={
                            "disk_io": 2.0
                        },
                        user_pattern={
                            "suspicious_activities": 4.0
                        }
                    )
                ],
                "stealth_factor": 0.8  # Reduces visibility of attack patterns
            },
            
            AdvancedAttackType.RANSOMWARE: {
                "phases": [
                    AttackPhase(
                        name="encryption_preparation",
                        duration=5,
                        network_pattern={
                            "bytes_received": 3.0
                        },
                        system_pattern={
                            "cpu_utilization": 2.0,
                            "memory_usage": 1.5
                        },
                        user_pattern={}
                    ),
                    AttackPhase(
                        name="file_encryption",
                        duration=30,
                        network_pattern={},
                        system_pattern={
                            "cpu_utilization": 4.0,
                            "disk_io": 8.0
                        },
                        user_pattern={
                            "suspicious_activities": 10.0
                        }
                    )
                ],
                "stealth_factor": 0.3  # Very visible attack
            },
            
            AdvancedAttackType.POLYMORPHIC_MALWARE: {
                "phases": [
                    AttackPhase(
                        name="initial_infection",
                        duration=10,
                        network_pattern={
                            "bytes_received": 1.5
                        },
                        system_pattern={
                            "memory_usage": 1.2
                        },
                        user_pattern={}
                    ),
                    AttackPhase(
                        name="mutation",
                        duration=5,
                        network_pattern={},
                        system_pattern={
                            "cpu_utilization": 2.0,
                            "memory_usage": 1.8
                        },
                        user_pattern={}
                    ),
                    AttackPhase(
                        name="attack",
                        duration=15,
                        network_pattern={
                            "packet_rate": 3.0,
                            "connections": 2.0
                        },
                        system_pattern={
                            "cpu_utilization": 3.0
                        },
                        user_pattern={
                            "suspicious_activities": 2.0
                        }
                    )
                ],
                "stealth_factor": 0.6,
                "mutation_interval": 10  # Steps between mutations
            }
        }
    
    def start_attack(self, attack_type: AdvancedAttackType):
        """Start a new advanced attack"""
        if attack_type not in self.attack_patterns:
            return
        
        self.active_attacks[attack_type] = {
            "current_phase": 0,
            "phase_step": 0,
            "start_time": time.time(),
            "mutations": 0  # For polymorphic attacks
        }
    
    def stop_attack(self, attack_type: AdvancedAttackType):
        """Stop an ongoing attack"""
        self.active_attacks.pop(attack_type, None)
    
    def apply_attack_effects(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Apply sophisticated attack effects to metrics"""
        if not self.active_attacks:
            return metrics
        
        modified_metrics = metrics.copy()
        
        for attack_type, attack_state in self.active_attacks.items():
            pattern = self.attack_patterns[attack_type]
            current_phase = pattern["phases"][attack_state["current_phase"]]
            stealth_factor = pattern["stealth_factor"]
            
            # Apply phase-specific effects
            self._apply_phase_effects(
                modified_metrics,
                current_phase,
                stealth_factor
            )
            
            # Update attack state
            attack_state["phase_step"] += 1
            if attack_state["phase_step"] >= current_phase.duration:
                attack_state["current_phase"] += 1
                attack_state["phase_step"] = 0
                
                # Check if attack is complete
                if attack_state["current_phase"] >= len(pattern["phases"]):
                    self.stop_attack(attack_type)
            
            # Handle polymorphic mutations
            if attack_type == AdvancedAttackType.POLYMORPHIC_MALWARE:
                if attack_state["phase_step"] % pattern["mutation_interval"] == 0:
                    self._mutate_attack_pattern(attack_type)
        
        return modified_metrics
    
    def _apply_phase_effects(
        self,
        metrics: Dict[str, Any],
        phase: AttackPhase,
        stealth_factor: float
    ):
        """Apply phase-specific effects with stealth consideration"""
        # Network impacts
        for metric, multiplier in phase.network_pattern.items():
            if metric in metrics["network_metrics"]:
                impact = (multiplier - 1) * stealth_factor
                metrics["network_metrics"][metric] *= (1 + impact)
        
        # System impacts
        for metric, multiplier in phase.system_pattern.items():
            if metric in metrics["system_resources"]:
                impact = (multiplier - 1) * stealth_factor
                metrics["system_resources"][metric] *= (1 + impact)
        
        # User impacts
        for metric, multiplier in phase.user_pattern.items():
            if metric in metrics["user_activity"]:
                impact = (multiplier - 1) * stealth_factor
                metrics["user_activity"][metric] *= (1 + impact)
    
    def _mutate_attack_pattern(self, attack_type: AdvancedAttackType):
        """Modify attack patterns for polymorphic malware"""
        if attack_type != AdvancedAttackType.POLYMORPHIC_MALWARE:
            return
        
        pattern = self.attack_patterns[attack_type]
        attack_state = self.active_attacks[attack_type]
        
        # Randomly modify the current phase patterns
        current_phase = pattern["phases"][attack_state["current_phase"]]
        
        for pattern_dict in [current_phase.network_pattern,
                           current_phase.system_pattern,
                           current_phase.user_pattern]:
            for metric in pattern_dict:
                # Random mutation of the impact
                pattern_dict[metric] *= random.uniform(0.8, 1.2)
        
        attack_state["mutations"] += 1
    
    def get_attack_status(self) -> Dict[str, Any]:
        """Get current status of all active attacks"""
        status = {}
        for attack_type, attack_state in self.active_attacks.items():
            pattern = self.attack_patterns[attack_type]
            current_phase = pattern["phases"][attack_state["current_phase"]]
            
            status[attack_type.value] = {
                "phase": current_phase.name,
                "progress": attack_state["phase_step"] / current_phase.duration,
                "duration": time.time() - attack_state["start_time"],
                "mutations": attack_state.get("mutations", 0)
            }
        
        return status 
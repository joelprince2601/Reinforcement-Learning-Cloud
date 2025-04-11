import numpy as np
from typing import Dict, List, Tuple, Any
from enum import Enum
import random
from collections import deque

class SecurityAction(Enum):
    BLOCK_IP = "block_ip"
    INCREASE_MONITORING = "increase_monitoring"
    SCALE_RESOURCES = "scale_resources"
    RESET_CONNECTIONS = "reset_connections"
    ENABLE_CAPTCHA = "enable_captcha"
    NO_ACTION = "no_action"

class RLAgent:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table as a dictionary for sparse state representation
        self.q_table: Dict[str, Dict[SecurityAction, float]] = {}
        
        # Experience replay buffer
        self.memory = deque(maxlen=2000)
        
        # Action space
        self.actions = list(SecurityAction)
        
        # Metrics for tracking performance
        self.episode_rewards = []
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state dict to a string key for Q-table"""
        # Extract relevant features and discretize them
        network_anomaly = self._discretize_score(state["anomaly_scores"]["network_anomaly"])
        user_anomaly = self._discretize_score(state["anomaly_scores"]["user_anomaly"])
        resource_anomaly = self._discretize_score(state["anomaly_scores"]["resource_anomaly"])
        
        return f"n{network_anomaly}u{user_anomaly}r{resource_anomaly}"
    
    def _discretize_score(self, score: float, bins: int = 5) -> int:
        """Discretize a continuous score into bins"""
        return min(bins - 1, int(score * bins))
    
    def get_action(self, state: Dict[str, Any]) -> SecurityAction:
        """Select action using epsilon-greedy policy"""
        state_key = self._state_to_key(state)
        
        # Initialize state in Q-table if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state_key].items(), key=lambda x: x[1])[0]
    
    def update(
        self,
        state: Dict[str, Any],
        action: SecurityAction,
        reward: float,
        next_state: Dict[str, Any]
    ):
        """Update Q-values using Q-learning"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Initialize states in Q-table if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {action: 0.0 for action in self.actions}
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        next_max_q = max(self.q_table[next_state_key].values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def calculate_reward(
        self,
        state: Dict[str, Any],
        action: SecurityAction,
        next_state: Dict[str, Any]
    ) -> float:
        """Calculate reward based on state transition and action taken"""
        # Base reward
        reward = 0.0
        
        # Get anomaly scores
        current_anomalies = state["anomaly_scores"]
        next_anomalies = next_state["anomaly_scores"]
        
        # Calculate average anomaly reduction
        current_avg = np.mean(list(current_anomalies.values()))
        next_avg = np.mean(list(next_anomalies.values()))
        
        # Reward for reducing anomalies
        anomaly_reduction = current_avg - next_avg
        reward += anomaly_reduction * 10  # Scale factor for anomaly reduction
        
        # Penalty for taking actions (cost of action)
        action_costs = {
            SecurityAction.NO_ACTION: 0,
            SecurityAction.BLOCK_IP: -0.1,
            SecurityAction.INCREASE_MONITORING: -0.05,
            SecurityAction.SCALE_RESOURCES: -0.2,
            SecurityAction.RESET_CONNECTIONS: -0.15,
            SecurityAction.ENABLE_CAPTCHA: -0.1
        }
        
        reward += action_costs[action]
        
        # Additional rewards/penalties based on specific conditions
        if next_avg < 0.1:  # Very low anomaly state
            reward += 1.0
        elif next_avg > 0.8:  # Very high anomaly state
            reward -= 1.0
        
        return reward
    
    def save_experience(
        self,
        state: Dict[str, Any],
        action: SecurityAction,
        reward: float,
        next_state: Dict[str, Any]
    ):
        """Save experience to replay buffer"""
        self.memory.append((state, action, reward, next_state))
    
    def train_from_replay(self, batch_size: int = 32):
        """Train the agent using experience replay"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state in batch:
            self.update(state, action, reward, next_state)
    
    def get_policy(self, state: Dict[str, Any]) -> Tuple[SecurityAction, float]:
        """Get the best action and its Q-value for a given state"""
        state_key = self._state_to_key(state)
        if state_key not in self.q_table:
            return SecurityAction.NO_ACTION, 0.0
        
        action = max(self.q_table[state_key].items(), key=lambda x: x[1])[0]
        value = self.q_table[state_key][action]
        return action, value 
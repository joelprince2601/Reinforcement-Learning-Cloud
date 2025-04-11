from environment import CloudEnvironmentSimulator
from attack_simulator import AttackSimulator, AttackType
from rl_agent import RLAgent, SecurityAction
from dataset_config import DatasetConfig, EvaluationMetrics
import numpy as np
from typing import Dict, Any
import time
import random
from collections import deque
import json
import matplotlib.pyplot as plt
from pathlib import Path

class SecurityEnvironment:
    def __init__(self):
        self.cloud_env = CloudEnvironmentSimulator()
        self.attack_simulator = AttackSimulator()
        self.current_state = None
        self.episode_steps = 0
        self.max_steps = 100
        self.attack_probability = 0.1
        self.current_attack = None
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment for a new episode"""
        self.episode_steps = 0
        self.attack_simulator.stop_all_attacks()
        self.current_attack = None
        self.current_state = self.cloud_env.collect_environment_state()
        return self.current_state
    
    def step(self, action: SecurityAction) -> tuple[Dict[str, Any], float, bool]:
        """Execute one step in the environment"""
        self.episode_steps += 1
        
        # Randomly start new attacks
        if random.random() < self.attack_probability and not self.current_attack:
            attack_type, intensity = self.attack_simulator.get_random_attack()
            self.attack_simulator.start_attack(attack_type, intensity)
            self.current_attack = attack_type
        
        # Collect new state
        raw_state = self.cloud_env.collect_environment_state()
        
        # Apply attack effects
        new_state = self.attack_simulator.apply_attack_effects(raw_state)
        
        # Apply defense action effects
        new_state = self._apply_defense_effects(new_state, action)
        
        # Store new state
        self.current_state = new_state
        
        # Calculate reward
        reward = self._calculate_defense_reward(action)
        
        # Check if episode is done
        done = self.episode_steps >= self.max_steps
        
        return new_state, reward, done
    
    def _apply_defense_effects(
        self,
        state: Dict[str, Any],
        action: SecurityAction
    ) -> Dict[str, Any]:
        """Apply the effects of defense actions to the state"""
        modified_state = state.copy()
        
        if action == SecurityAction.BLOCK_IP and self.current_attack in [AttackType.DDOS, AttackType.BRUTE_FORCE]:
            self.attack_simulator.stop_attack(self.current_attack)
            self.current_attack = None
            
        elif action == SecurityAction.SCALE_RESOURCES and self.current_attack == AttackType.DDOS:
            # Reduce the impact of DDoS
            modified_state["network_metrics"]["packet_rate"] *= 0.7
            modified_state["system_resources"]["cpu_utilization"] *= 0.8
            
        elif action == SecurityAction.RESET_CONNECTIONS and self.current_attack in [AttackType.DATA_EXFILTRATION, AttackType.PORT_SCAN]:
            self.attack_simulator.stop_attack(self.current_attack)
            self.current_attack = None
            
        elif action == SecurityAction.ENABLE_CAPTCHA and self.current_attack == AttackType.BRUTE_FORCE:
            self.attack_simulator.stop_attack(self.current_attack)
            self.current_attack = None
        
        return modified_state
    
    def _calculate_defense_reward(self, action: SecurityAction) -> float:
        """Calculate reward for defense action"""
        reward = 0.0
        
        # Reward for successful defense
        if self.current_attack is None:
            reward += 1.0
            
        # Penalty for unnecessary actions when no attack
        elif action != SecurityAction.NO_ACTION and self.current_attack is None:
            reward -= 0.5
            
        # Reward for appropriate action for attack type
        if self.current_attack:
            appropriate_actions = {
                AttackType.DDOS: [SecurityAction.BLOCK_IP, SecurityAction.SCALE_RESOURCES],
                AttackType.BRUTE_FORCE: [SecurityAction.BLOCK_IP, SecurityAction.ENABLE_CAPTCHA],
                AttackType.DATA_EXFILTRATION: [SecurityAction.RESET_CONNECTIONS],
                AttackType.CRYPTO_MINING: [SecurityAction.BLOCK_IP],
                AttackType.PORT_SCAN: [SecurityAction.RESET_CONNECTIONS, SecurityAction.BLOCK_IP]
            }
            
            if action in appropriate_actions.get(self.current_attack, []):
                reward += 0.5
        
        return reward

def train_agent(
    num_episodes: int = 1000,
    max_steps: int = 100,
    eval_interval: int = 50
) -> tuple[RLAgent, list[float], EvaluationMetrics]:
    """Train the RL agent"""
    env = SecurityEnvironment()
    agent = RLAgent()
    metrics = EvaluationMetrics()
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_start_time = time.time()
        
        for step in range(max_steps):
            # Select and take action
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            # Update metrics
            metrics.update_metrics(
                true_label=env.current_attack.name if env.current_attack else "normal",
                predicted_label=action.name,
                reward=reward,
                detection_time=time.time() - episode_start_time,
                mitigation_success=env.current_attack is None
            )
            
            # Store experience and train
            agent.save_experience(state, action, reward, next_state)
            agent.train_from_replay()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Evaluation
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(agent, env, 5)
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            metrics.print_metrics()
            print(f"Current Epsilon: {agent.epsilon:.4f}\n")
    
    return agent, metrics

def evaluate_agent(
    agent: RLAgent,
    env: SecurityEnvironment,
    num_episodes: int = 5
) -> float:
    """Evaluate the agent's performance"""
    eval_metrics = EvaluationMetrics()
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_start_time = time.time()
        total_reward = 0
        
        for _ in range(env.max_steps):
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            # Update evaluation metrics
            eval_metrics.update_metrics(
                true_label=env.current_attack.name if env.current_attack else "normal",
                predicted_label=action.name,
                reward=reward,
                detection_time=time.time() - episode_start_time,
                mitigation_success=env.current_attack is None
            )
            
            total_reward += reward
            state = next_state
            
            if done:
                break
    
    return np.mean(eval_metrics.episode_rewards)

def plot_training_results(metrics: EvaluationMetrics, save_dir: str = "results"):
    """Plot and save training metrics"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(metrics.episode_rewards)
    plt.title("Training Progress - Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.grid(True)
    plt.savefig(f"{save_dir}/rewards.png")
    plt.close()
    
    # Plot detection metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(metrics.detection_latency)
    plt.title("Detection Latency")
    plt.xlabel("Episode")
    plt.ylabel("Time (s)")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics.mitigation_success)
    plt.title("Mitigation Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/detection_metrics.png")
    plt.close()
    
    # Save numerical metrics
    final_metrics = metrics.calculate_metrics()
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

if __name__ == "__main__":
    # Train the agent
    trained_agent, metrics = train_agent(
        num_episodes=1000,
        max_steps=100,
        eval_interval=50
    )
    
    # Plot and save results
    plot_training_results(metrics) 
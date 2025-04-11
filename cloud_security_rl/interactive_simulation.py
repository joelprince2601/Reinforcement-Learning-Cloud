import threading
import time
from typing import Dict, Any
from visualization import SecurityDashboard
from advanced_attacks import AdvancedAttackSimulator, AdvancedAttackType
from environment import CloudEnvironmentSimulator
from rl_agent import RLAgent, SecurityAction
import random
import argparse
from pathlib import Path
import json
import signal
import sys

class InteractiveSimulation:
    def __init__(
        self,
        dashboard_port: int = 8050,
        update_interval: float = 1.0,
        load_trained_agent: bool = False
    ):
        # Initialize components
        self.cloud_env = CloudEnvironmentSimulator()
        self.attack_simulator = AdvancedAttackSimulator()
        self.agent = RLAgent()
        self.dashboard = SecurityDashboard()
        
        # Load trained agent if requested
        if load_trained_agent and Path("results/q_table.json").exists():
            with open("results/q_table.json", "r") as f:
                q_table_data = json.load(f)
                # Convert string keys back to SecurityAction enum
                self.agent.q_table = {
                    state: {SecurityAction(action): value 
                           for action, value in actions.items()}
                    for state, actions in q_table_data.items()
                }
        
        self.update_interval = update_interval
        self.running = False
        self.current_state = None
        self.current_attack = None
        self.episode_rewards = []
        
        # Statistics
        self.stats = {
            "attacks_detected": 0,
            "successful_defenses": 0,
            "false_positives": 0,
            "total_reward": 0
        }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nReceived shutdown signal. Cleaning up...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the interactive simulation"""
        self.running = True
        
        # Start the dashboard in a separate thread
        dashboard_thread = threading.Thread(
            target=self._run_dashboard
        )
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Give the dashboard time to start
        time.sleep(2)
        
        # Start the main simulation loop
        self.simulation_loop()
    
    def _run_dashboard(self):
        """Run the dashboard in a separate thread"""
        try:
            print("Starting dashboard... Open http://localhost:8050 to view")
            self.dashboard.run_server(debug=False, port=8050)
        except Exception as e:
            print(f"Error starting dashboard: {e}")
            self.running = False
    
    def simulation_loop(self):
        """Main simulation loop"""
        print("Starting simulation...")
        print("Press Ctrl+C to stop")
        
        while self.running:
            try:
                # Get current state
                raw_state = self.cloud_env.collect_environment_state()
                
                # Apply attack effects if any
                state = self.attack_simulator.apply_attack_effects(raw_state)
                self.current_state = state
                
                # Get RL agent action
                action = self.agent.get_action(state)
                
                # Apply defense action
                self._apply_defense(action)
                
                # Randomly start new attacks
                self._manage_attacks()
                
                # Update dashboard
                self.dashboard.update_metrics(state)
                self.dashboard.update_attack_status(self.current_attack)
                self.dashboard.update_defense_action(action)
                
                # Update statistics
                self._update_stats(action)
                
                # Sleep for update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in simulation loop: {e}")
                continue
    
    def _manage_attacks(self):
        """Manage ongoing and new attacks"""
        # Randomly start new advanced attacks
        if not self.current_attack and random.random() < 0.05:  # 5% chance per step
            attack_type = random.choice(list(AdvancedAttackType))
            self.attack_simulator.start_attack(attack_type)
            self.current_attack = attack_type
            print(f"\nNew attack started: {attack_type.value}")
        
        # Get attack status
        if self.current_attack:
            status = self.attack_simulator.get_attack_status()
            if not status:  # Attack has completed
                self.current_attack = None
    
    def _apply_defense(self, action: SecurityAction):
        """Apply defense action and calculate reward"""
        reward = 0
        
        # Check if defense was successful
        if self.current_attack:
            appropriate_actions = {
                AdvancedAttackType.APT: [
                    SecurityAction.INCREASE_MONITORING,
                    SecurityAction.RESET_CONNECTIONS
                ],
                AdvancedAttackType.RANSOMWARE: [
                    SecurityAction.BLOCK_IP,
                    SecurityAction.RESET_CONNECTIONS
                ],
                AdvancedAttackType.POLYMORPHIC_MALWARE: [
                    SecurityAction.BLOCK_IP,
                    SecurityAction.INCREASE_MONITORING
                ]
            }
            
            if action in appropriate_actions.get(self.current_attack, []):
                reward = 1.0
                self.stats["successful_defenses"] += 1
                self.attack_simulator.stop_attack(self.current_attack)
                self.current_attack = None
                print(f"\nSuccessful defense! Action: {action.value}")
            else:
                reward = -0.5
        elif action != SecurityAction.NO_ACTION:
            # Penalty for unnecessary action
            reward = -0.2
            self.stats["false_positives"] += 1
        
        self.stats["total_reward"] += reward
        self.episode_rewards.append(reward)
    
    def _update_stats(self, action: SecurityAction):
        """Update simulation statistics"""
        if self.current_attack:
            self.stats["attacks_detected"] += 1
    
    def stop(self):
        """Stop the simulation"""
        self.running = False
        print("\nFinal Statistics:")
        print(f"Attacks Detected: {self.stats['attacks_detected']}")
        print(f"Successful Defenses: {self.stats['successful_defenses']}")
        print(f"False Positives: {self.stats['false_positives']}")
        print(f"Total Reward: {self.stats['total_reward']:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Interactive Cloud Security Simulation")
    parser.add_argument("--port", type=int, default=8050,
                       help="Dashboard port (default: 8050)")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Update interval in seconds (default: 1.0)")
    parser.add_argument("--load-agent", action="store_true",
                       help="Load trained agent from results/q_table.json")
    
    args = parser.parse_args()
    
    simulation = InteractiveSimulation(
        dashboard_port=args.port,
        update_interval=args.interval,
        load_trained_agent=args.load_agent
    )
    
    try:
        simulation.start()
    except KeyboardInterrupt:
        simulation.stop()

if __name__ == "__main__":
    main() 
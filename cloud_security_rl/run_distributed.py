import asyncio
import argparse
from pathlib import Path
import yaml
import logging
from distributed_agent import DistributedAgentManager
from deployment_config import DeploymentConfig
from results_manager import ResultsManager
from rl_agent import RLAgent
from environment import CloudEnvironmentSimulator
from agent_roles import AgentRoleManager
from typing import Dict, Any

async def run_agent(
    agent_manager: DistributedAgentManager,
    agent_id: str,
    config: DeploymentConfig,
    results_manager: ResultsManager,
    role_config: Dict[str, Any],
    num_steps: int = 1000
):
    """Run a single agent's monitoring loop"""
    env = CloudEnvironmentSimulator()
    state = env.collect_environment_state()
    
    # Log agent role and specialization
    logging.info(f"Agent {agent_id} starting with role: {role_config['role']}")
    logging.info(f"Monitoring region: {role_config['specialization']['region']}")
    logging.info(f"Primary features: {role_config['specialization']['features']}")
    
    for step in range(num_steps):
        # Get action with timeout constraint
        action, metrics = await agent_manager.get_action(
            agent_id=agent_id,
            state=state,
            deadline_ms=config.max_inference_latency_ms
        )
        
        if metrics:
            # Save metrics with role information
            results_manager.save_metrics(agent_id, {
                "step": step,
                "role": role_config['role'],
                "region": role_config['specialization']['region'],
                "inference_latency": metrics.inference_latency,
                "memory_usage": metrics.memory_usage,
                "cpu_usage": metrics.cpu_usage,
                "timestamp": metrics.action_timestamp
            })
            
            # Log performance metrics if enabled
            if config.enable_performance_logging:
                logging.info(
                    f"Agent {agent_id} ({role_config['role']}) - Step {step}: "
                    f"Latency={metrics.inference_latency*1000:.2f}ms, "
                    f"Memory={metrics.memory_usage:.1f}MB, "
                    f"CPU={metrics.cpu_usage:.1f}%"
                )
        
        # Get next state
        state = env.collect_environment_state()
        
        # Check if retraining is needed
        if agent_manager.should_retrain(config.performance_threshold):
            logging.warning(
                f"Agent {agent_id} ({role_config['role']}) performance below threshold, "
                f"retraining recommended"
            )

async def run_distributed_system(config: DeploymentConfig, results_manager: ResultsManager):
    """Run the distributed security monitoring system"""
    # Save configuration
    results_manager.save_config(config.to_dict())
    
    # Initialize role manager
    role_manager = AgentRoleManager()
    
    # Initialize agent manager
    agent_manager = DistributedAgentManager(
        num_agents=config.num_agents,
        sync_interval=config.sync_interval,
        max_concurrent_actions=config.max_concurrent_actions
    )
    
    # Create and add agents with roles
    for i in range(config.num_agents):
        agent_id = f"agent_{i}"
        role_config = role_manager.get_agent_config(i)
        
        # Initialize agent with role-specific configuration
        agent = RLAgent()  # You would pass role_config here in a full implementation
        agent_manager.add_agent(agent_id, agent)
        
        # Save role configuration
        results_manager.save_config({
            "agent_roles": {agent_id: role_config}
        })
    
    try:
        # Create tasks for each agent
        tasks = []
        for i in range(config.num_agents):
            agent_id = f"agent_{i}"
            role_config = role_manager.get_agent_config(i)
            task = run_agent(
                agent_manager,
                agent_id,
                config,
                results_manager,
                role_config
            )
            tasks.append(task)
        
        # Run all agents concurrently
        await asyncio.gather(*tasks)
        
    except KeyboardInterrupt:
        logging.info("Shutting down distributed system...")
    finally:
        # Generate plots and summary
        results_manager.plot_metrics()
        results_manager.print_summary()
        agent_manager.cleanup()

def load_config(config_path: str = None) -> DeploymentConfig:
    """Load configuration from file or use defaults"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            return DeploymentConfig.from_dict(config_dict)
    return DeploymentConfig()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run distributed cloud security monitoring")
    parser.add_argument("--config", type=str, help="Path to configuration YAML file")
    parser.add_argument("--num-agents", type=int, help="Number of agents to run")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store results")
    args = parser.parse_args()
    
    # Initialize results manager
    results_manager = ResultsManager(args.results_dir)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.num_agents:
        config.num_agents = args.num_agents
    
    # Print configuration
    logging.info("Starting distributed system with configuration:")
    for key, value in config.to_dict().items():
        logging.info(f"{key}: {value}")
    
    # Run the system
    asyncio.run(run_distributed_system(config, results_manager))

if __name__ == "__main__":
    main() 
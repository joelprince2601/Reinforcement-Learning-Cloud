from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import numpy as np
from rl_agent import RLAgent, SecurityAction
from environment import CloudEnvironmentSimulator
import torch.distributed as dist
import torch
from concurrent.futures import ThreadPoolExecutor
import logging
from collections import deque
import asyncio
from functools import partial

@dataclass
class AgentMetrics:
    """Metrics for monitoring agent performance and overhead"""
    inference_latency: float
    memory_usage: float
    cpu_usage: float
    action_timestamp: float

class DistributedAgentManager:
    """Manages multiple RL agents in a distributed cloud environment"""
    def __init__(
        self,
        num_agents: int = 3,
        sync_interval: int = 100,
        max_concurrent_actions: int = 5,
        performance_history_size: int = 1000
    ):
        self.agents: Dict[str, RLAgent] = {}
        self.agent_metrics: Dict[str, deque] = {}
        self.num_agents = num_agents
        self.sync_interval = sync_interval
        self.steps_since_sync = 0
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_actions)
        self.performance_history = deque(maxlen=performance_history_size)
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize distributed setup
        self._setup_distributed()
    
    def _setup_distributed(self):
        """Initialize distributed training backend"""
        try:
            dist.init_process_group(backend='gloo')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logging.info(f"Initialized distributed agent {self.rank}/{self.world_size}")
        except Exception as e:
            logging.warning(f"Failed to initialize distributed backend: {e}")
            self.world_size = 1
            self.rank = 0
    
    def add_agent(self, agent_id: str, agent: RLAgent):
        """Add a new agent to the distributed system"""
        self.agents[agent_id] = agent
        self.agent_metrics[agent_id] = deque(maxlen=1000)
        logging.info(f"Added agent {agent_id} to distributed system")
    
    async def get_action(
        self,
        agent_id: str,
        state: Dict,
        deadline_ms: float = 100.0
    ) -> tuple[SecurityAction, AgentMetrics]:
        """Get action from an agent with real-time constraints"""
        start_time = time.time()
        
        try:
            # Run get_action in thread pool
            action = await self.loop.run_in_executor(
                self.executor,
                self.agents[agent_id].get_action,
                state
            )
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > deadline_ms/1000.0:
                logging.warning(f"Action selection exceeded deadline: {elapsed*1000:.2f}ms")
                return SecurityAction.NO_ACTION, None
            
            # Record metrics
            metrics = AgentMetrics(
                inference_latency=elapsed,
                memory_usage=self._get_memory_usage(),
                cpu_usage=self._get_cpu_usage(),
                action_timestamp=time.time()
            )
            
            self.agent_metrics[agent_id].append(metrics)
            
            # Check if we need model sync
            self.steps_since_sync += 1
            if self.steps_since_sync >= self.sync_interval:
                await self.loop.run_in_executor(
                    self.executor,
                    self._sync_models
                )
            
            return action, metrics
            
        except Exception as e:
            logging.error(f"Error getting action for agent {agent_id}: {e}")
            return SecurityAction.NO_ACTION, None
    
    def _sync_models(self):
        """Synchronize models across distributed agents"""
        if self.world_size > 1:
            try:
                # Average model parameters across agents
                for agent in self.agents.values():
                    for param in agent.model.parameters():
                        dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
                logging.info("Successfully synchronized agent models")
            except Exception as e:
                logging.error(f"Model sync failed: {e}")
        self.steps_since_sync = 0
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get real-time performance statistics"""
        stats = {}
        for agent_id, metrics in self.agent_metrics.items():
            if metrics:
                recent_metrics = list(metrics)[-100:]  # Last 100 actions
                stats[agent_id] = {
                    "avg_latency": np.mean([m.inference_latency for m in recent_metrics]),
                    "95th_latency": np.percentile([m.inference_latency for m in recent_metrics], 95),
                    "avg_memory": np.mean([m.memory_usage for m in recent_metrics]),
                    "avg_cpu": np.mean([m.cpu_usage for m in recent_metrics])
                }
        return stats
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except:
            return 0.0
    
    def should_retrain(self, performance_threshold: float = 0.8) -> bool:
        """Determine if model retraining is needed based on performance"""
        if len(self.performance_history) < 100:
            return False
            
        recent_performance = list(self.performance_history)[-100:]
        avg_performance = np.mean(recent_performance)
        
        return avg_performance < performance_threshold
    
    def cleanup(self):
        """Cleanup distributed resources"""
        self.executor.shutdown()
        if self.world_size > 1:
            dist.destroy_process_group() 
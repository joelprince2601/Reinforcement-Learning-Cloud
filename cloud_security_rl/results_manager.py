import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, Any

class ResultsManager:
    """Manages saving and visualizing results from the distributed system"""
    
    def __init__(self, base_dir: str = "results"):
        """Initialize results manager with base directory"""
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to file
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging to both file and console"""
        log_file = self.run_dir / "run.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)
        
    def save_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        """Save agent metrics to CSV"""
        metrics_file = self.run_dir / f"{agent_id}_metrics.csv"
        
        # Convert metrics to DataFrame
        df = pd.DataFrame([metrics])
        
        # Append or create new file
        if metrics_file.exists():
            df.to_csv(metrics_file, mode='a', header=False, index=False)
        else:
            df.to_csv(metrics_file, index=False)
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to JSON"""
        config_file = self.run_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def plot_metrics(self):
        """Generate plots for all collected metrics"""
        # Find all metric files
        metric_files = list(self.run_dir.glob("*_metrics.csv"))
        
        if not metric_files:
            logging.warning("No metric files found to plot")
            return
        
        # Create plots directory
        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot metrics for each agent
        for metric_file in metric_files:
            agent_id = metric_file.stem.replace("_metrics", "")
            df = pd.read_csv(metric_file)
            
            # Latency plot
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, y='inference_latency', x=df.index)
            plt.title(f"{agent_id} - Inference Latency Over Time")
            plt.ylabel("Latency (s)")
            plt.xlabel("Step")
            plt.savefig(plots_dir / f"{agent_id}_latency.png")
            plt.close()
            
            # Resource usage plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            sns.lineplot(data=df, y='memory_usage', x=df.index, ax=ax1)
            ax1.set_title(f"{agent_id} - Memory Usage Over Time")
            ax1.set_ylabel("Memory Usage (MB)")
            
            sns.lineplot(data=df, y='cpu_usage', x=df.index, ax=ax2)
            ax2.set_title(f"{agent_id} - CPU Usage Over Time")
            ax2.set_ylabel("CPU Usage (%)")
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{agent_id}_resources.png")
            plt.close()
    
    def generate_summary(self):
        """Generate summary statistics for the run"""
        summary = {
            "run_timestamp": self.run_dir.name,
            "agents": {}
        }
        
        # Process each agent's metrics
        for metric_file in self.run_dir.glob("*_metrics.csv"):
            agent_id = metric_file.stem.replace("_metrics", "")
            df = pd.read_csv(metric_file)
            
            summary["agents"][agent_id] = {
                "avg_latency": df['inference_latency'].mean(),
                "95th_latency": df['inference_latency'].quantile(0.95),
                "avg_memory": df['memory_usage'].mean(),
                "avg_cpu": df['cpu_usage'].mean(),
                "total_steps": len(df)
            }
        
        # Save summary
        summary_file = self.run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def get_latest_results(self):
        """Get path to latest results directory"""
        if not self.base_dir.exists():
            return None
        
        runs = list(self.base_dir.glob("*"))
        if not runs:
            return None
            
        return max(runs, key=lambda x: x.stat().st_mtime)
    
    def print_summary(self):
        """Print summary of results to console"""
        summary = self.generate_summary()
        
        print("\n=== Run Summary ===")
        print(f"Run ID: {summary['run_timestamp']}")
        print("\nAgent Performance:")
        
        for agent_id, metrics in summary["agents"].items():
            print(f"\n{agent_id}:")
            print(f"  Average Latency: {metrics['avg_latency']*1000:.2f}ms")
            print(f"  95th Percentile Latency: {metrics['95th_latency']*1000:.2f}ms")
            print(f"  Average Memory Usage: {metrics['avg_memory']:.1f}MB")
            print(f"  Average CPU Usage: {metrics['avg_cpu']:.1f}%")
            print(f"  Total Steps: {metrics['total_steps']}")
        
        print(f"\nResults saved in: {self.run_dir}")
        print("Plots can be found in the 'plots' subdirectory") 
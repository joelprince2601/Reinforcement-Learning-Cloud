import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import os

def generate_sample_data():
    """Generate sample monitoring data for the dashboard"""
    # Create results directory
    script_dir = Path(__file__).parent
    base_dir = script_dir / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data for each agent
    agent_roles = ["network_monitor", "user_activity", "resource_monitor"]
    num_samples = 1000
    
    for i, role in enumerate(agent_roles):
        # Create timestamps
        timestamps = [
            (datetime.now() - timedelta(minutes=num_samples-j))
            for j in range(num_samples)
        ]
        
        # Generate metrics
        data = {
            "timestamp": timestamps,
            "role": [role] * num_samples,
            "inference_latency": np.random.normal(0.05, 0.01, num_samples),
            "memory_usage": np.random.normal(500, 50, num_samples),
            "cpu_usage": np.random.normal(40, 10, num_samples),
            "success_rate": np.random.uniform(0.9, 1.0, num_samples)
        }
        
        # Add some anomalies
        anomaly_indices = np.random.choice(num_samples, size=5)
        data["inference_latency"][anomaly_indices] *= 3
        data["cpu_usage"][anomaly_indices] *= 2
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(run_dir / f"agent_{i}_metrics.csv", index=False)
    
    # Generate config.json
    config = {
        "run_timestamp": timestamp,
        "num_agents": len(agent_roles),
        "agent_roles": agent_roles,
        "metrics_interval": 60
    }
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create plots directory
    (run_dir / "plots").mkdir(exist_ok=True)
    
    return run_dir

if __name__ == "__main__":
    run_dir = generate_sample_data()
    print(f"Generated sample data in: {run_dir}") 

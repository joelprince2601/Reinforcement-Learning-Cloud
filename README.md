# Cloud Security RL Simulator

This project implements a Reinforcement Learning (RL) based cloud security simulation environment. It provides a framework for developing and testing RL agents that can learn to detect and respond to security threats in cloud environments.

## Features

- Real-time monitoring of system metrics
- Simulated network traffic patterns
- User activity monitoring
- System resource utilization tracking
- Anomaly detection based on historical patterns
- Extensible framework for adding new security features

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main component is the `CloudEnvironmentSimulator` class in `cloud_security_rl/environment.py`. This class provides methods to:

- Collect environment state data
- Monitor network metrics
- Track user activity
- Monitor system resources
- Calculate anomaly scores

Example usage:

```python
from cloud_security_rl.environment import CloudEnvironmentSimulator

# Create simulator instance
env = CloudEnvironmentSimulator()

# Collect current state
state = env.collect_environment_state()

# Access different metrics
network_metrics = state['network_metrics']
user_activity = state['user_activity']
system_resources = state['system_resources']
anomaly_scores = state['anomaly_scores']
```

## Extending the Environment

To add new security features:

1. Add new metrics to the appropriate dataclass (NetworkMetrics, UserActivity, or SystemResources)
2. Implement collection logic in the corresponding _collect_* method
3. Update the anomaly detection logic in _calculate_anomaly_scores

## Free Cloud Setup Options

For testing this environment, you can use free tier services from:

1. AWS Free Tier
   - Includes 750 hours of EC2 t2.micro instances
   - Basic monitoring and CloudWatch metrics
   - Limited S3 storage

2. Google Cloud Platform Free Tier
   - Includes f1-micro instances
   - Basic monitoring services
   - Limited storage

3. Azure Free Account
   - Includes B1S virtual machines
   - Basic monitoring capabilities
   - Limited storage

Remember to set up proper security groups and follow cloud security best practices when deploying.

# Cloud Security RL Dashboard

A real-time dashboard for monitoring distributed cloud security reinforcement learning agents.

## Features

- Real-time monitoring of agent performance
- System status indicators
- Latency metrics visualization
- Resource usage tracking
- Alerts and actions display
- Auto-refresh capabilities
- Role-based filtering

## Deployment on Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select this repository and the main branch
6. Set the main file path as: `cloud_security_rl/streamlit_dashboard.py`
7. Click "Deploy"

## Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run cloud_security_rl/streamlit_dashboard.py
```

## Configuration

The dashboard can be configured through:
- `.streamlit/config.toml` for Streamlit settings
- `cloud_security_rl/config.yaml` for application settings

## Data Storage

The dashboard expects monitoring data in the `results` directory with the following structure:
```
results/
  YYYYMMDD_HHMMSS/
    agent_0_metrics.csv
    agent_1_metrics.csv
    agent_2_metrics.csv
    config.json
    summary.json
    plots/
``` 
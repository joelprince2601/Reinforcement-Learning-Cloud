import streamlit as st

def show_documentation():
    st.title("📚 Cloud Security RL Dashboard Documentation")
    
    st.markdown("""
    ## Overview
    This dashboard provides real-time monitoring and visualization of a Reinforcement Learning-based cloud security system. 
    The system uses multiple specialized agents to detect and respond to potential security threats.
    
    ## Dashboard Sections
    
    ### 1. Agent Performance Metrics
    - **Latency**: Shows the response time of each agent in milliseconds
    - **Resource Usage**: Displays CPU and memory utilization
    - **Actions**: Lists the actions taken by each agent and their success rates
    
    ### 2. System Status
    - **System Health**: Overall system status indicator
    - **Active Agents**: Number of operational agents
    - **Alert Level**: Current security alert level
    - **Active Threats**: Real-time display of detected security threats
    
    ### 3. Agent Roles
    
    #### Network Monitor
    - Monitors network traffic patterns
    - Detects unusual network activity
    - Responds to potential DDoS attacks
    
    #### User Activity Monitor
    - Tracks user login attempts
    - Monitors session activities
    - Identifies suspicious user behavior
    
    #### Resource Monitor
    - Tracks CPU and memory usage
    - Monitors system resources
    - Detects resource exhaustion attacks
    
    #### Attack Detector
    - Analyzes patterns for potential attacks
    - Correlates security events
    - Identifies attack signatures
    
    #### Response Coordinator
    - Coordinates response actions
    - Manages threat mitigation
    - Orchestrates system-wide responses
    
    ## Dashboard Controls
    
    ### Refresh Settings
    - **Refresh Interval**: Control how often the dashboard updates
    - **Auto Refresh**: Toggle automatic updates
    - **Manual Refresh**: Force an immediate update
    
    ### Filter Settings
    - **Agent Roles**: Select which agent types to display
    - **Time Range**: Choose the time window for displayed data
    
    ## Data Visualization
    The dashboard uses various visualization types:
    - Line charts for temporal metrics
    - Status indicators for system health
    - Alert boxes for active threats
    - Data tables for detailed metrics
    
    ## Interpreting the Data
    - **Green** indicators show normal operation
    - **Yellow** indicators suggest attention needed
    - **Red** indicators require immediate attention
    - Spikes in latency graphs may indicate attacks
    - Resource usage patterns help identify abnormal behavior
    """)

if __name__ == "__main__":
    show_documentation() 
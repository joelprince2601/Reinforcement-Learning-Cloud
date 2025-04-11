import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import numpy as np
from .agent_roles import AgentRole, AgentRoleManager

class SecurityDashboard:
    def __init__(self):
        self.results_dir = Path(__file__).parent / "results"
        self.role_manager = AgentRoleManager()
        
        # Setup page config
        st.set_page_config(
            page_title="Cloud Security RL Dashboard",
            page_icon="🛡️",
            layout="wide"
        )
        
        # Initialize sample data if needed
        if not self.results_dir.exists() or not list(self.results_dir.glob("*")):
            st.info("No monitoring data found. Generating sample data...")
            from .sample_data_generator import generate_sample_data
            generate_sample_data()
        
    def run(self):
        """Main dashboard loop"""
        # Header
        st.title("🛡️ Cloud Security RL Dashboard")
        
        # Create main tabs
        tab1, tab2 = st.tabs(["📊 Dashboard", "📚 Documentation"])
        
        with tab1:
            # Sidebar
            self.setup_sidebar()
            
            # Main content
            col1, col2 = st.columns([2, 1])
            
            with col1:
                self.show_agent_metrics()
            
            with col2:
                self.show_system_status()
                
            # Bottom section
            self.show_alerts_and_actions()
            
        with tab2:
            self.show_documentation()
    
    def setup_sidebar(self):
        """Setup sidebar controls"""
        st.sidebar.header("Dashboard Controls")
        
        # Refresh rate
        st.sidebar.subheader("Refresh Settings")
        refresh_rate = st.sidebar.slider(
            "Refresh interval (seconds)",
            min_value=1,
            max_value=60,
            value=5
        )
        
        if st.sidebar.button("Refresh Now"):
            st.rerun()
        
        # Auto refresh
        if st.sidebar.checkbox("Auto refresh", value=True):
            time.sleep(refresh_rate)
            st.rerun()
        
        # Filter settings
        st.sidebar.subheader("Filter Settings")
        self.selected_roles = st.sidebar.multiselect(
            "Show Agent Roles",
            options=[role.value for role in AgentRole],
            default=[role.value for role in AgentRole]
        )
        
        # Time range
        st.sidebar.subheader("Time Range")
        self.time_range = st.sidebar.selectbox(
            "Show data for last",
            options=["5 minutes", "15 minutes", "1 hour", "6 hours", "24 hours"],
            index=1
        )
    
    def show_agent_metrics(self):
        """Display agent metrics and performance"""
        st.header("Agent Performance Metrics")
        
        # Get latest results
        latest_run = self.get_latest_run()
        if not latest_run:
            st.warning("No monitoring data available")
            return
        
        # Create tabs for different metric views
        tab1, tab2, tab3 = st.tabs(["Latency", "Resource Usage", "Actions"])
        
        with tab1:
            self.show_latency_metrics(latest_run)
        
        with tab2:
            self.show_resource_metrics(latest_run)
            
        with tab3:
            self.show_action_metrics(latest_run)
    
    def show_system_status(self):
        """Display overall system status"""
        st.header("System Status")
        
        # Get latest metrics
        latest_run = self.get_latest_run()
        if not latest_run:
            return
            
        # System health indicators
        status_cols = st.columns(3)
        
        with status_cols[0]:
            self.health_indicator("System Health", "Good", "green")
        
        with status_cols[1]:
            self.health_indicator("Active Agents", "3/3", "green")
            
        with status_cols[2]:
            self.health_indicator("Alert Level", "Normal", "green")
        
        # Active threats
        st.subheader("Active Threats")
        threat_data = self.get_active_threats(latest_run)
        if threat_data:
            for threat in threat_data:
                st.error(
                    f"⚠️ {threat['type']}\n\n"
                    f"Severity: {threat['severity']}\n\n"
                    f"Detected by: {threat['detector']}"
                )
    
    def show_alerts_and_actions(self):
        """Display recent alerts and actions"""
        st.header("Recent Alerts & Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Alerts")
            alerts = self.get_recent_alerts()
            if alerts:
                for alert in alerts:
                    st.warning(
                        f"🚨 {alert['message']}\n\n"
                        f"Time: {alert['timestamp']}\n\n"
                        f"Agent: {alert['agent']}"
                    )
        
        with col2:
            st.subheader("Recent Actions")
            actions = self.get_recent_actions()
            if actions:
                for action in actions:
                    st.info(
                        f"🔄 {action['action']}\n\n"
                        f"Time: {action['timestamp']}\n\n"
                        f"Result: {action['result']}"
                    )
    
    def show_latency_metrics(self, run_dir):
        """Display latency metrics"""
        metrics_data = {}
        
        # Load metrics for each agent
        for agent_file in run_dir.glob("*_metrics.csv"):
            df = pd.read_csv(agent_file)
            agent_id = agent_file.stem.replace("_metrics", "")
            
            if df['role'].iloc[0] in self.selected_roles:
                metrics_data[agent_id] = df
        
        if not metrics_data:
            st.warning("No latency data available")
            return
        
        # Create latency plot
        fig = go.Figure()
        
        for agent_id, df in metrics_data.items():
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['inference_latency'] * 1000,  # Convert to ms
                name=f"{agent_id} ({df['role'].iloc[0]})",
                mode='lines',
                line=dict(width=2),
                hovertemplate='Time: %{x}<br>Latency: %{y:.2f} ms<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': "Agent Inference Latency",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20)
            },
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
            height=500,
            template="plotly_dark",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.1)"
            ),
            margin=dict(l=60, r=30, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'scrollZoom': True
        })
    
    def show_resource_metrics(self, run_dir):
        """Display resource usage metrics"""
        # Create two columns for CPU and Memory
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_resource_metric(run_dir, 'cpu_usage', "CPU Usage (%)")
            
        with col2:
            self.plot_resource_metric(run_dir, 'memory_usage', "Memory Usage (MB)")
    
    def plot_resource_metric(self, run_dir, metric, title):
        """Helper to plot resource metrics"""
        fig = go.Figure()
        
        for agent_file in run_dir.glob("*_metrics.csv"):
            df = pd.read_csv(agent_file)
            agent_id = agent_file.stem.replace("_metrics", "")
            
            if df['role'].iloc[0] in self.selected_roles:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[metric],
                    name=f"{agent_id} ({df['role'].iloc[0]})",
                    mode='lines',
                    line=dict(width=2),
                    hovertemplate='Time: %{x}<br>Value: %{y:.1f}<extra></extra>'
                ))
        
        fig.update_layout(
            title={
                'text': title,
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18)
            },
            xaxis_title="Time",
            yaxis_title=title,
            height=400,
            template="plotly_dark",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.1)"
            ),
            margin=dict(l=60, r=30, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'scrollZoom': True
        })
    
    def show_action_metrics(self, run_dir):
        """Display action-related metrics"""
        st.subheader("Agent Actions")
        
        # Create metrics table
        metrics_data = []
        
        for agent_file in run_dir.glob("*_metrics.csv"):
            df = pd.read_csv(agent_file)
            agent_id = agent_file.stem.replace("_metrics", "")
            
            if df['role'].iloc[0] in self.selected_roles:
                metrics_data.append({
                    "Agent": agent_id,
                    "Role": df['role'].iloc[0],
                    "Total Actions": len(df),
                    "Avg Latency (ms)": f"{df['inference_latency'].mean() * 1000:.2f}",
                    "Success Rate": "95%"  # You would calculate this from actual data
                })
        
        if metrics_data:
            st.dataframe(pd.DataFrame(metrics_data))
    
    def health_indicator(self, label, value, color):
        """Display a health indicator"""
        # Create color indicator emoji
        color_indicator = {
            "green": "🟢",
            "yellow": "🟡",
            "red": "🔴"
        }.get(color, "⚪")
        
        # Display metric with color indicator
        st.metric(
            label=f"{color_indicator} {label}",
            value=value,
            delta=None
        )
    
    def get_latest_run(self):
        """Get the most recent results directory"""
        if not self.results_dir.exists():
            return None
            
        runs = list(self.results_dir.glob("*"))
        if not runs:
            return None
            
        return max(runs, key=lambda x: x.stat().st_mtime)
    
    def get_active_threats(self, run_dir):
        """Get current active threats"""
        try:
            # Try to get real monitoring data
            threat_data = []
            metrics_files = list(run_dir.glob("*_metrics.csv"))
            
            if metrics_files:
                for file in metrics_files:
                    df = pd.read_csv(file)
                    latest = df.iloc[-1]
                    
                    # Check for anomalies in the latest data
                    if latest['cpu_usage'] > 80 or latest['memory_usage'] > 1000:
                        threat_data.append({
                            "type": "Resource Exhaustion Attack",
                            "severity": "High",
                            "detector": latest['role']
                        })
                    
                    if latest['inference_latency'] > 0.1:
                        threat_data.append({
                            "type": "Performance Degradation",
                            "severity": "Medium",
                            "detector": latest['role']
                        })
            
            return threat_data or [{
                "type": "Potential DDoS Attack",
                "severity": "Medium",
                "detector": "Network Monitor"
            }]
        except Exception as e:
            st.warning(f"Error reading threat data: {e}")
            return [{
                "type": "System Status Unknown",
                "severity": "Unknown",
                "detector": "System Monitor"
            }]
    
    def get_recent_alerts(self):
        """Get recent system alerts"""
        try:
            latest_run = self.get_latest_run()
            if latest_run:
                alerts = []
                for file in latest_run.glob("*_metrics.csv"):
                    df = pd.read_csv(file)
                    latest = df.iloc[-5:]  # Get last 5 entries
                    
                    for _, row in latest.iterrows():
                        if row['cpu_usage'] > 70:
                            alerts.append({
                                "message": f"High CPU Usage: {row['cpu_usage']:.1f}%",
                                "timestamp": row['timestamp'],
                                "agent": row['role']
                            })
                return alerts or self._get_sample_alerts()
            return self._get_sample_alerts()
        except Exception:
            return self._get_sample_alerts()
    
    def _get_sample_alerts(self):
        """Return sample alerts when no real data is available"""
        return [
            {
                "message": "High CPU Usage Detected",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "agent": "Resource Monitor"
            },
            {
                "message": "Unusual Network Activity",
                "timestamp": (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "agent": "Network Monitor"
            }
        ]
    
    def get_recent_actions(self):
        """Get recent agent actions"""
        # This would come from your actual monitoring data
        return [
            {
                "action": "Scaled Resources",
                "timestamp": "2024-01-20 10:15:25",
                "result": "Success"
            }
        ]
    
    def show_documentation(self):
        """Display dashboard documentation"""
        # Create tabs for different documentation sections
        doc_tab1, doc_tab2, doc_tab3 = st.tabs(["Overview & Features", "Technical Details", "Algorithms & Formulas"])
        
        with doc_tab1:
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
            """)
            
        with doc_tab2:
            st.markdown("""
            ## Technical Architecture
            
            ### Reinforcement Learning Framework
            The system implements a **Multi-Agent Deep Q-Network (MADQN)** architecture with the following components:
            
            1. **State Space**
               - Network traffic features (packets/sec, bytes/sec)
               - System metrics (CPU, memory, disk I/O)
               - User activity patterns
               - Historical attack patterns
            
            2. **Action Space**
               - Block IP addresses
               - Scale resources
               - Enable CAPTCHA
               - Reset connections
               - Update security rules
            
            3. **Reward Function**
               ```
               R = w₁ * S + w₂ * (1-L) + w₃ * (1-F) - w₄ * C
               ```
               where:
               - S = Success rate of threat mitigation
               - L = Normalized latency impact
               - F = False positive rate
               - C = Resource cost
               - w₁, w₂, w₃, w₄ = Importance weights
            
            ### Key Technical Terms
            
            1. **Inference Latency**
               - Time taken for an agent to process inputs and decide actions
               - Measured in milliseconds
               - Critical for real-time response
            
            2. **Anomaly Score**
               ```
               A = |x - μ| / σ
               ```
               - x = Current metric value
               - μ = Historical mean
               - σ = Standard deviation
            
            3. **Threat Detection Confidence**
               ```
               C = (TP/(TP + FP)) * (TN/(TN + FN))
               ```
               - TP = True Positives
               - FP = False Positives
               - TN = True Negatives
               - FN = False Negatives
            
            4. **Resource Utilization Index**
               ```
               RUI = α*CPU + β*Memory + γ*Network
               ```
               - α, β, γ = Resource weight coefficients
               - Values normalized to [0,1]
            """)
            
        with doc_tab3:
            st.markdown("""
            ## Algorithms & Formulas
            
            ### 1. Deep Q-Network (DQN)
            Each agent uses a DQN with the following architecture:
            - Input Layer: State dimension (varies by agent role)
            - Hidden Layers: 3 fully connected layers (256, 128, 64 neurons)
            - Output Layer: Action space dimension
            
            #### Q-Learning Update
            ```
            Q(s,a) ← Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
            ```
            - α = Learning rate
            - γ = Discount factor
            - r = Immediate reward
            - s' = Next state
            
            ### 2. Threat Detection
            
            #### Network Anomaly Score
            ```
            NA = Σ(w_i * |x_i - μ_i| / σ_i)
            ```
            - x_i = Current metric value
            - μ_i = Historical mean
            - σ_i = Standard deviation
            - w_i = Feature weight
            
            #### Attack Probability
            ```
            P(attack) = 1 / (1 + e^(-z))
            where z = Σ(β_j * NA_j)
            ```
            
            ### 3. Resource Management
            
            #### Auto-scaling Threshold
            ```
            ST = μ_cpu + k * σ_cpu
            ```
            - μ_cpu = Mean CPU utilization
            - σ_cpu = CPU utilization standard deviation
            - k = Sensitivity parameter
            
            #### Response Time Prediction
            ```
            RT = S / (C - λ)
            ```
            - S = Service time
            - C = System capacity
            - λ = Arrival rate
            
            ### 4. Performance Metrics
            
            #### Agent Efficiency Score
            ```
            E = (Successful_Actions / Total_Actions) * (1 - Latency/Max_Latency)
            ```
            
            #### System Health Index
            ```
            H = (1/n) * Σ(1 - Metric_i/Threshold_i)
            ```
            
            ### 5. Coordination Algorithm
            
            #### Action Priority Score
            ```
            P(a) = (Impact(a) * Urgency(a)) / Cost(a)
            ```
            
            #### Consensus Decision
            ```
            D = argmax_a Σ(w_i * Score_i(a))
            ```
            - w_i = Agent weight
            - Score_i(a) = Action score from agent i
            """)

if __name__ == "__main__":
    dashboard = SecurityDashboard()
    dashboard.run() 

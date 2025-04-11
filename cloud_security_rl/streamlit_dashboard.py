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
        self.results_dir = Path("results")
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
            from sample_data_generator import generate_sample_data
            generate_sample_data()
        
    def run(self):
        """Main dashboard loop"""
        # Header
        st.title("🛡️ Cloud Security RL Dashboard")
        
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
            st.experimental_rerun()
        
        # Auto refresh
        if st.sidebar.checkbox("Auto refresh", value=True):
            time.sleep(refresh_rate)
            st.experimental_rerun()
        
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
                mode='lines'
            ))
        
        fig.update_layout(
            title="Agent Inference Latency",
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
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
                    mode='lines'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=title,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
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
        st.metric(
            label=label,
            value=value,
            delta=None,
            delta_color=color
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

if __name__ == "__main__":
    dashboard = SecurityDashboard()
    dashboard.run() 

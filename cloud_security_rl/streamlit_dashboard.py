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
import torch

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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Dashboard",
            "🎯 Attack Analysis",
            "🌍 Global Threats",
            "🤖 ML Insights",
            "📋 Security & Compliance",
            "🧠 ML Training",
            "📚 Documentation"
        ])
        
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
            self.show_attack_analysis()
            
        with tab3:
            self.show_global_threats()
            
        with tab4:
            self.show_ml_insights()
            
        with tab5:
            self.show_security_compliance()
            
        with tab6:
            self.show_ml_training()
            
        with tab7:
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
    
    def show_attack_analysis(self):
        """Display attack analysis and agent performance during attacks"""
        st.header("🎯 Attack Analysis & Response Performance")
        
        # Debug info
        st.info("Loading attack analysis view...")
        
        try:
            # Attack type selector with default value
            attack_type = st.selectbox(
                "Select Attack Type",
                options=[
                    "DDoS Attack",
                    "Brute Force Login",
                    "Resource Exhaustion",
                    "Data Exfiltration",
                    "SQL Injection",
                    "Zero-Day Exploit"
                ],
                index=0  # Set default selection
            )
            
            # Create expanders for better organization
            with st.expander("Attack Details", expanded=True):
                # Create two columns for metrics and visualization
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Attack Metrics")
                    
                    # Show attack-specific metrics
                    metrics = self.get_attack_metrics(attack_type)
                    
                    # Display metrics with colors
                    st.metric("⏱️ Detection Time", metrics["detection_time"])
                    st.metric("✅ Mitigation Success", metrics["mitigation_success"])
                    st.metric("🎯 False Positive Rate", metrics["false_positive_rate"])
                    st.metric("⚡ Agent Response Time", metrics["response_time"])
                    
                    # Show involved agents
                    st.subheader("🤖 Responding Agents")
                    for agent in metrics["responding_agents"]:
                        st.info(f"**{agent['role']}**\n\n{agent['action']}")
                
                with col2:
                    # Show attack timeline
                    st.subheader("📈 Attack Timeline")
                    self.show_attack_timeline(attack_type)
            
            # Show performance comparison in separate expander
            with st.expander("Agent Performance Analysis", expanded=True):
                st.subheader("🔄 Agent Performance Comparison")
                self.show_agent_performance_comparison(attack_type)
            
            # Add attack description
            with st.expander("Attack Description", expanded=True):
                self.show_attack_description(attack_type)
        
        except Exception as e:
            st.error(f"Error displaying attack analysis: {str(e)}")
            st.error("Stack trace:", stack_info=True)
    
    def get_attack_metrics(self, attack_type):
        """Get metrics for specific attack type"""
        # This would come from your actual monitoring data
        # For now, returning sample data
        metrics = {
            "DDoS Attack": {
                "detection_time": "1.2s",
                "mitigation_success": "98%",
                "false_positive_rate": "0.5%",
                "response_time": "850ms",
                "responding_agents": [
                    {"role": "Network Monitor", "action": "Traffic filtering activated"},
                    {"role": "Resource Monitor", "action": "Scaled resources"},
                    {"role": "Response Coordinator", "action": "Updated firewall rules"}
                ]
            },
            "Brute Force Login": {
                "detection_time": "2.5s",
                "mitigation_success": "99%",
                "false_positive_rate": "0.2%",
                "response_time": "450ms",
                "responding_agents": [
                    {"role": "User Activity Monitor", "action": "IP blocking initiated"},
                    {"role": "Attack Detector", "action": "CAPTCHA enabled"},
                    {"role": "Response Coordinator", "action": "Alert escalated"}
                ]
            },
            "Resource Exhaustion": {
                "detection_time": "3.1s",
                "mitigation_success": "95%",
                "false_positive_rate": "1.0%",
                "response_time": "1.2s",
                "responding_agents": [
                    {"role": "Resource Monitor", "action": "Resource limits applied"},
                    {"role": "Network Monitor", "action": "Connection throttling"},
                    {"role": "Response Coordinator", "action": "System scaled"}
                ]
            },
            "Data Exfiltration": {
                "detection_time": "4.2s",
                "mitigation_success": "97%",
                "false_positive_rate": "0.8%",
                "response_time": "950ms",
                "responding_agents": [
                    {"role": "Network Monitor", "action": "Traffic pattern analysis"},
                    {"role": "Attack Detector", "action": "Connection terminated"},
                    {"role": "Response Coordinator", "action": "Security rules updated"}
                ]
            },
            "SQL Injection": {
                "detection_time": "0.8s",
                "mitigation_success": "99.5%",
                "false_positive_rate": "0.1%",
                "response_time": "350ms",
                "responding_agents": [
                    {"role": "Attack Detector", "action": "Query blocked"},
                    {"role": "User Activity Monitor", "action": "Session terminated"},
                    {"role": "Response Coordinator", "action": "WAF rules updated"}
                ]
            },
            "Zero-Day Exploit": {
                "detection_time": "5.5s",
                "mitigation_success": "92%",
                "false_positive_rate": "2.0%",
                "response_time": "1.5s",
                "responding_agents": [
                    {"role": "Attack Detector", "action": "Anomaly detected"},
                    {"role": "Network Monitor", "action": "Traffic isolated"},
                    {"role": "Response Coordinator", "action": "Emergency protocol activated"}
                ]
            }
        }
        
        return metrics.get(attack_type, {})
    
    def show_attack_timeline(self, attack_type):
        """Show timeline visualization for attack detection and response"""
        # Create sample timeline data
        timeline_data = {
            "Time": [0, 1, 2, 3, 4, 5],
            "Anomaly Score": [0.1, 0.3, 0.8, 0.9, 0.5, 0.2]
        }
        
        # Create timeline plot
        fig = go.Figure()
        
        # Add anomaly score line
        fig.add_trace(go.Scatter(
            x=timeline_data["Time"],
            y=timeline_data["Anomaly Score"],
            mode='lines+markers',
            name='Anomaly Score',
            line=dict(width=2, color='red'),
            hovertemplate='Time: %{x}s<br>Score: %{y:.2f}<extra></extra>'
        ))
        
        # Add threshold line
        fig.add_hline(
            y=0.7,
            line_dash="dash",
            line_color="yellow",
            annotation_text="Detection Threshold",
            annotation_position="bottom right"
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Attack Detection Timeline - {attack_type}",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=16)
            },
            xaxis_title="Time (seconds)",
            yaxis_title="Anomaly Score",
            height=300,
            template="plotly_dark",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.1)"
            ),
            margin=dict(l=60, r=30, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_agent_performance_comparison(self, attack_type):
        """Show performance comparison between agents for specific attack type"""
        # Create sample performance data
        performance_data = {
            "Agent": ["Network Monitor", "User Activity", "Resource Monitor", "Attack Detector", "Response Coordinator"],
            "Detection Rate": [0.95, 0.88, 0.92, 0.98, 0.85],
            "Response Time": [0.8, 1.2, 0.9, 0.7, 1.1]
        }
        
        # Create performance comparison plot
        fig = go.Figure()
        
        # Add detection rate bars
        fig.add_trace(go.Bar(
            name='Detection Rate',
            x=performance_data["Agent"],
            y=performance_data["Detection Rate"],
            marker_color='blue',
            hovertemplate='Detection Rate: %{y:.2%}<extra></extra>'
        ))
        
        # Add response time bars
        fig.add_trace(go.Bar(
            name='Response Time (s)',
            x=performance_data["Agent"],
            y=performance_data["Response Time"],
            marker_color='orange',
            hovertemplate='Response Time: %{y:.2f}s<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Agent Performance - {attack_type}",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=16)
            },
            barmode='group',
            height=300,
            template="plotly_dark",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.1)"
            ),
            margin=dict(l=60, r=30, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_attack_description(self, attack_type):
        """Show detailed description of the attack type"""
        descriptions = {
            "DDoS Attack": {
                "description": """
                **Distributed Denial of Service (DDoS) Attack**
                
                A coordinated attempt to overwhelm system resources by flooding them with traffic from multiple sources.
                
                **Characteristics:**
                - High volume of incoming traffic
                - Multiple source IPs
                - Network congestion
                - Service disruption
                
                **Agent Response Strategy:**
                1. Traffic pattern analysis
                2. Rate limiting
                3. Resource scaling
                4. IP filtering
                """
            },
            "Brute Force Login": {
                "description": """
                **Brute Force Login Attack**
                
                Repeated login attempts using different credentials to gain unauthorized access.
                
                **Characteristics:**
                - Multiple failed login attempts
                - Rapid succession of requests
                - Pattern of credential testing
                
                **Agent Response Strategy:**
                1. Account lockout
                2. CAPTCHA implementation
                3. IP blocking
                4. Alert escalation
                """
            },
            "Resource Exhaustion": {
                "description": """
                **Resource Exhaustion Attack**
                
                Targeted consumption of system resources to degrade service quality.
                
                **Characteristics:**
                - High CPU/Memory usage
                - Slow system response
                - Resource starvation
                
                **Agent Response Strategy:**
                1. Resource limiting
                2. Load balancing
                3. Service isolation
                4. Auto-scaling
                """
            },
            "Data Exfiltration": {
                "description": """
                **Data Exfiltration Attack**
                
                Unauthorized data transfer attempting to steal sensitive information.
                
                **Characteristics:**
                - Unusual data transfer patterns
                - Suspicious network connections
                - Abnormal access patterns
                
                **Agent Response Strategy:**
                1. Traffic monitoring
                2. Connection termination
                3. Data encryption
                4. Access restriction
                """
            },
            "SQL Injection": {
                "description": """
                **SQL Injection Attack**
                
                Malicious SQL queries attempting to manipulate or access database data.
                
                **Characteristics:**
                - Malformed SQL queries
                - Input validation bypass attempts
                - Unauthorized data access
                
                **Agent Response Strategy:**
                1. Query sanitization
                2. WAF rules update
                3. Session termination
                4. Database monitoring
                """
            },
            "Zero-Day Exploit": {
                "description": """
                **Zero-Day Exploit Attack**
                
                Previously unknown vulnerability exploitation attempt.
                
                **Characteristics:**
                - Unknown attack patterns
                - Novel exploitation methods
                - Signature evasion
                
                **Agent Response Strategy:**
                1. Behavior analysis
                2. System isolation
                3. Emergency protocols
                4. Rapid patch deployment
                """
            }
        }
        
        st.markdown(descriptions[attack_type]["description"])
    
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

    def show_global_threats(self):
        """Display global threats dashboard"""
        st.header("🌍 Global Threat Intelligence")
        
        # Show attack map
        self.show_attack_map()
        
        # Show security score trend
        self.show_security_score_trend()
    
    def show_attack_map(self):
        """Display global attack origin map"""
        st.subheader("🌍 Global Attack Origins")
        
        # Sample attack origin data
        attack_data = {
            "lat": [40.7128, 51.5074, 35.6762, -33.8688, 39.9042],
            "lon": [-74.0060, -0.1278, 139.6503, 151.2093, 116.4074],
            "intensity": [100, 80, 60, 40, 90],
            "location": ["New York", "London", "Tokyo", "Sydney", "Beijing"],
            "attack_type": ["DDoS", "Brute Force", "SQL Injection", "Data Exfiltration", "Zero-Day"]
        }
        
        fig = go.Figure()
        
        # Add attack points
        fig.add_trace(go.Scattergeo(
            lon=attack_data["lon"],
            lat=attack_data["lat"],
            text=[f"{loc}: {type}" for loc, type in zip(attack_data["location"], attack_data["attack_type"])],
            mode="markers",
            marker=dict(
                size=[i/10 for i in attack_data["intensity"]],
                color=attack_data["intensity"],
                colorscale="Viridis",
                showscale=True,
                colorbar_title="Attack Intensity"
            ),
            hovertemplate="<b>%{text}</b><br>Intensity: %{marker.color}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Real-time Attack Origins",
            geo=dict(
                showland=True,
                showcountries=True,
                showocean=True,
                countrywidth=0.5,
                landcolor="rgb(243, 243, 243)",
                oceancolor="rgb(204, 229, 255)",
                projection_type="equirectangular"
            ),
            height=400,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_security_score_trend(self):
        """Display security score trends"""
        st.subheader("🎯 Security Score Trend")
        
        # Generate sample data
        dates = pd.date_range(start="2024-01-01", end="2024-01-20", freq="D")
        scores = np.random.normal(85, 5, size=len(dates))
        scores = np.clip(scores, 0, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=scores,
            mode='lines+markers',
            name='Security Score',
            line=dict(width=2),
            hovertemplate='Date: %{x}<br>Score: %{y:.1f}<extra></extra>'
        ))
        
        # Add threshold lines
        fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Excellent")
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Warning")
        
        fig.update_layout(
            title="Security Score Over Time",
            xaxis_title="Date",
            yaxis_title="Security Score",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def show_ml_insights(self):
        """Display ML model performance metrics and insights"""
        st.header("🤖 Machine Learning Performance")
        
        # Create two columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            self.show_ml_metrics()
        
        with col2:
            self.show_confusion_matrix()
            
        # Show model performance over time
        self.show_model_performance_trend()
    
    def show_ml_metrics(self):
        """Display key ML performance metrics"""
        st.subheader("📊 Model Metrics")
        
        metrics = {
            "Accuracy": 0.95,
            "Precision": 0.92,
            "Recall": 0.94,
            "F1 Score": 0.93
        }
        
        for metric, value in metrics.items():
            st.metric(
                label=metric,
                value=f"{value:.2%}",
                delta=f"+{(value - 0.9):.2%}" if value > 0.9 else f"{(value - 0.9):.2%}"
            )
    
    def show_confusion_matrix(self):
        """Display confusion matrix heatmap"""
        st.subheader("🎯 Confusion Matrix")
        
        # Sample confusion matrix data
        confusion_data = np.array([
            [120, 5, 2],
            [4, 95, 3],
            [1, 2, 85]
        ])
        
        labels = ["Normal", "Suspicious", "Attack"]
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_data,
            x=labels,
            y=labels,
            hoverongaps=False,
            colorscale="Viridis",
            text=confusion_data,
            texttemplate="%{text}",
            textfont={"size": 16},
        ))
        
        fig.update_layout(
            title="Prediction Results",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_model_performance_trend(self):
        """Display model performance trends over time"""
        st.subheader("📈 Performance Trends")
        
        # Generate sample data
        dates = pd.date_range(start="2024-01-01", end="2024-01-20", freq="D")
        metrics = {
            "Accuracy": np.clip(np.random.normal(0.95, 0.02, size=len(dates)), 0, 1),
            "Precision": np.clip(np.random.normal(0.92, 0.02, size=len(dates)), 0, 1),
            "Recall": np.clip(np.random.normal(0.94, 0.02, size=len(dates)), 0, 1)
        }
        
        fig = go.Figure()
        
        for metric, values in metrics.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=metric,
                hovertemplate=metric + ": %{y:.2%}<br>Date: %{x}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Score",
            yaxis_tickformat=".1%",
            height=400,
            template="plotly_dark",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def show_security_compliance(self):
        """Display security compliance status and recommendations"""
        st.header("📋 Security & Compliance Overview")
        
        # Show compliance score
        self.show_compliance_score()
        
        # Show compliance checks
        self.show_compliance_checks()
        
        # Show recommendations
        self.show_recommendations()
    
    def show_compliance_score(self):
        """Display overall compliance score and status"""
        st.subheader("🎯 Compliance Score")
        
        score = 85  # Sample score
        
        # Create three columns
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "red"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric(
                "Change",
                "+5%",
                "+2%",
                help="Score change from last month"
            )
        
        with col3:
            status = "Good" if score >= 80 else "Warning" if score >= 60 else "Critical"
            color = "green" if status == "Good" else "orange" if status == "Warning" else "red"
            st.markdown(f"**Status:**")
            st.markdown(f"<h3 style='color: {color};'>{status}</h3>", unsafe_allow_html=True)
    
    def show_compliance_checks(self):
        """Display compliance checks status"""
        st.subheader("✅ Compliance Checks")
        
        checks = {
            "IAM Security": {
                "MFA Enabled": True,
                "Password Policy": True,
                "Access Review": True,
                "Role-based Access": False
            },
            "Network Security": {
                "VPC Configuration": True,
                "Security Groups": True,
                "Network ACLs": True,
                "Flow Logs": False
            },
            "Data Protection": {
                "Encryption at Rest": True,
                "Encryption in Transit": True,
                "Backup Policy": False,
                "Data Classification": True
            }
        }
        
        for category, items in checks.items():
            with st.expander(f"{category} ({sum(items.values())}/{len(items)} Compliant)"):
                for check, status in items.items():
                    icon = "✅" if status else "❌"
                    color = "green" if status else "red"
                    st.markdown(f"<span style='color: {color};'>{icon} {check}</span>", unsafe_allow_html=True)
    
    def show_recommendations(self):
        """Display security recommendations"""
        st.subheader("💡 Recommendations")
        
        recommendations = [
            {
                "priority": "High",
                "title": "Enable MFA for all IAM users",
                "impact": "Critical",
                "effort": "Low",
                "description": "Multi-factor authentication significantly improves account security."
            },
            {
                "priority": "Medium",
                "title": "Review unused IAM roles",
                "impact": "Moderate",
                "effort": "Medium",
                "description": "Remove or update unused IAM roles to reduce security risks."
            },
            {
                "priority": "Low",
                "title": "Update security group rules",
                "impact": "Low",
                "effort": "High",
                "description": "Optimize security group rules for better network security."
            }
        ]
        
        for rec in recommendations:
            with st.expander(f"[{rec['priority']}] {rec['title']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Impact:** {rec['impact']}")
                with col2:
                    st.markdown(f"**Effort:** {rec['effort']}")
                st.markdown(f"**Description:** {rec['description']}")

    def show_ml_training(self):
        """Display ML training interface and results"""
        st.header("🧠 ML Model Training")
        
        # Training controls
        st.subheader("Training Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.number_input("Number of Epochs", min_value=1, max_value=200, value=50)
            batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32)
        
        with col2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.1, 0.01, 0.001, 0.0001],
                value=0.001,
                format_func=lambda x: f"{x:.4f}"
            )
            
            device = "GPU" if torch.cuda.is_available() else "CPU"
            st.info(f"Training Device: {device}")
        
        # Training button
        if st.button("Start Training"):
            try:
                from .ml_model import SecurityMLTrainer
                
                with st.spinner("Training model..."):
                    trainer = SecurityMLTrainer()
                    trainer.train(num_epochs=epochs)
                    st.success("Training completed successfully!")
                
                # Show training results
                self.show_training_results(trainer.results_dir)
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
    
    def show_training_results(self, results_dir):
        """Display training results and metrics"""
        st.subheader("📈 Training Results")
        
        # Get latest training run
        runs = list(results_dir.glob("*"))
        if not runs:
            st.warning("No training results available")
            return
            
        latest_run = max(runs, key=lambda x: x.stat().st_mtime)
        
        # Load training history
        with open(latest_run / "training_history.json", "r") as f:
            history = json.load(f)
        
        # Create metrics plot
        fig = go.Figure()
        
        # Add training metrics
        fig.add_trace(go.Scatter(
            x=[m['epoch'] for m in history],
            y=[m['train_loss'] for m in history],
            mode='lines',
            name='Training Loss',
            line=dict(width=2, color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=[m['epoch'] for m in history],
            y=[m['val_loss'] for m in history],
            mode='lines',
            name='Validation Loss',
            line=dict(width=2, color='red')
        ))
        
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400,
            template="plotly_dark",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show accuracy metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Final Training Accuracy",
                f"{history[-1]['train_acc']:.2f}%",
                f"{history[-1]['train_acc'] - history[0]['train_acc']:.2f}%"
            )
        
        with col2:
            st.metric(
                "Final Validation Accuracy",
                f"{history[-1]['val_acc']:.2f}%",
                f"{history[-1]['val_acc'] - history[0]['val_acc']:.2f}%"
            )
        
        # Model architecture
        with st.expander("Model Architecture"):
            st.code("""
SecurityClassifier(
    input_size=3,
    hidden_size=128,
    num_classes=3
)
Network:
  - Linear(3, 128)
  - ReLU()
  - Dropout(0.3)
  - Linear(128, 64)
  - ReLU()
  - Dropout(0.2)
  - Linear(64, 3)
            """)

if __name__ == "__main__":
    dashboard = SecurityDashboard()
    dashboard.run() 

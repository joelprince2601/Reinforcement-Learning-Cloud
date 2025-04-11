import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from typing import Dict, Any, List
import numpy as np
from collections import deque
import threading
import time
from dataclasses import dataclass
from attack_simulator import AttackType
from rl_agent import SecurityAction

@dataclass
class MetricHistory:
    max_len: int = 100
    def __post_init__(self):
        self.timestamps = deque(maxlen=self.max_len)
        self.values = deque(maxlen=self.max_len)
    
    def add(self, value: float):
        self.timestamps.append(time.time())
        self.values.append(value)

class SecurityDashboard:
    def __init__(self):
        # Initialize Dash with external stylesheets
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
            ]
        )
        
        # Enable the dashboard to be embedded in other web pages
        self.app.config.suppress_callback_exceptions = True
        
        self.metrics_history = {
            "network": {
                "packet_rate": MetricHistory(),
                "connections": MetricHistory(),
                "bytes_sent": MetricHistory(),
                "bytes_received": MetricHistory()
            },
            "system": {
                "cpu_utilization": MetricHistory(),
                "memory_usage": MetricHistory(),
                "disk_io": MetricHistory()
            },
            "security": {
                "network_anomaly": MetricHistory(),
                "user_anomaly": MetricHistory(),
                "resource_anomaly": MetricHistory()
            }
        }
        self.current_attack = None
        self.current_defense = None
        self.setup_layout()
    
    def setup_layout(self):
        self.app.layout = html.Div([
            # Navigation bar
            html.Nav([
                html.Div([
                    html.H2("Cloud Security Monitoring Dashboard",
                           className="navbar-brand text-white")
                ], className="container")
            ], className="navbar navbar-dark bg-dark mb-4"),
            
            # Main content
            html.Div([
                # Alert Panel for Active Threats
                html.Div([
                    html.H3("Active Threats", className="text-danger"),
                    html.Div(id='active-threats-panel', className="alert alert-warning")
                ], className="card p-3 mb-4"),
                
                # Network Metrics
                html.Div([
                    html.H3("Network Metrics", className="card-header"),
                    html.Div([
                        dcc.Graph(id='network-metrics-graph')
                    ], className="card-body")
                ], className="card mb-4"),
                
                # System Resources
                html.Div([
                    html.H3("System Resources", className="card-header"),
                    html.Div([
                        dcc.Graph(id='system-resources-graph')
                    ], className="card-body")
                ], className="card mb-4"),
                
                # Anomaly Scores
                html.Div([
                    html.H3("Security Anomaly Scores", className="card-header"),
                    html.Div([
                        dcc.Graph(id='anomaly-scores-graph')
                    ], className="card-body")
                ], className="card mb-4"),
                
                # Defense Actions Panel
                html.Div([
                    html.H3("Defense Actions", className="card-header"),
                    html.Div(id='defense-actions-panel', className="card-body")
                ], className="card mb-4"),
                
                # Update interval
                dcc.Interval(
                    id='interval-component',
                    interval=1000,  # Update every second
                    n_intervals=0
                )
            ], className="container")
        ])
        
        self.setup_callbacks()
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('network-metrics-graph', 'figure'),
             Output('system-resources-graph', 'figure'),
             Output('anomaly-scores-graph', 'figure'),
             Output('active-threats-panel', 'children'),
             Output('defense-actions-panel', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            # Network metrics figure
            network_fig = go.Figure()
            for metric, history in self.metrics_history["network"].items():
                if history.values:
                    network_fig.add_trace(go.Scatter(
                        x=list(history.timestamps),
                        y=list(history.values),
                        name=metric,
                        mode='lines'
                    ))
            network_fig.update_layout(
                title='Network Metrics Over Time',
                xaxis_title='Time',
                yaxis_title='Value',
                height=400,
                template='plotly_dark'
            )
            
            # System resources figure
            system_fig = go.Figure()
            for metric, history in self.metrics_history["system"].items():
                if history.values:
                    system_fig.add_trace(go.Scatter(
                        x=list(history.timestamps),
                        y=list(history.values),
                        name=metric,
                        mode='lines'
                    ))
            system_fig.update_layout(
                title='System Resources Over Time',
                xaxis_title='Time',
                yaxis_title='Value',
                height=400,
                template='plotly_dark'
            )
            
            # Anomaly scores figure
            security_fig = go.Figure()
            for metric, history in self.metrics_history["security"].items():
                if history.values:
                    security_fig.add_trace(go.Scatter(
                        x=list(history.timestamps),
                        y=list(history.values),
                        name=metric,
                        mode='lines'
                    ))
            security_fig.update_layout(
                title='Security Anomaly Scores',
                xaxis_title='Time',
                yaxis_title='Score',
                height=400,
                template='plotly_dark'
            )
            
            # Active threats panel
            threat_style = {
                'color': 'red' if self.current_attack else 'green',
                'fontSize': '1.2em',
                'fontWeight': 'bold'
            }
            threats_panel = html.Div([
                html.H4(
                    f"Current Attack: {self.current_attack.value if self.current_attack else 'None'}",
                    style=threat_style
                ),
                html.P(f"Status: {'⚠️ Under Attack' if self.current_attack else '✅ Secure'}")
            ])
            
            # Defense actions panel
            defense_panel = html.Div([
                html.H4(f"Last Defense Action: {self.current_defense.value if self.current_defense else 'None'}"),
                html.P(f"Timestamp: {time.strftime('%H:%M:%S')}"),
                html.Div([
                    html.Span("Status: ", className="font-weight-bold"),
                    html.Span(
                        "🛡️ Active" if self.current_defense else "Monitoring",
                        className=f"badge badge-{'success' if self.current_defense else 'info'}"
                    )
                ])
            ])
            
            return network_fig, system_fig, security_fig, threats_panel, defense_panel
    
    def update_metrics(self, state: Dict[str, Any]):
        """Update metrics with new state data"""
        # Update network metrics
        for metric, value in state["network_metrics"].items():
            if metric in self.metrics_history["network"]:
                self.metrics_history["network"][metric].add(value)
        
        # Update system metrics
        for metric, value in state["system_resources"].items():
            if metric in self.metrics_history["system"]:
                self.metrics_history["system"][metric].add(value)
        
        # Update security metrics
        for metric, value in state["anomaly_scores"].items():
            if metric in self.metrics_history["security"]:
                self.metrics_history["security"][metric].add(value)
    
    def update_attack_status(self, attack_type: AttackType = None):
        """Update current attack status"""
        self.current_attack = attack_type
    
    def update_defense_action(self, action: SecurityAction):
        """Update current defense action"""
        self.current_defense = action
    
    def run_server(self, debug: bool = False, port: int = 8050):
        """Run the dashboard server"""
        self.app.run(
            debug=debug,
            host='0.0.0.0',  # Allow external connections
            port=port
        ) 
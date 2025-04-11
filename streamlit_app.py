import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import the dashboard
from cloud_security_rl.streamlit_dashboard import SecurityDashboard

if __name__ == "__main__":
    dashboard = SecurityDashboard()
    dashboard.run() 

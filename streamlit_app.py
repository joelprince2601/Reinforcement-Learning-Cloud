import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import and run the dashboard
from cloud_security_rl.streamlit_dashboard import SecurityDashboard

if __name__ == "__main__":
    dashboard = SecurityDashboard()
    dashboard.run() 
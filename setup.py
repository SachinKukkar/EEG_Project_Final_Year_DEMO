"""Setup script for EEG Processing Project."""
import os
import sys
from utils import setup_logging

def create_directories():
    """Create necessary directories."""
    dirs = ['assets', 'logs', 'exports']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'pandas', 'numpy', 'sklearn', 
        'joblib', 'PyQt5', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("All dependencies are installed.")
    return True

def main():
    """Main setup function."""
    print("Setting up EEG Processing Project...")
    
    # Setup logging
    setup_logging()
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("Setup complete! You can now run the application.")

if __name__ == "__main__":
    main()
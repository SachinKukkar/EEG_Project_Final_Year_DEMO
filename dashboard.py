"""Performance dashboard for EEG authentication system."""
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import os
from datetime import datetime

# Optional imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
except ImportError:
    plt = None
    sns = None
    np = None

class PerformanceDashboard(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 EEG System Performance Dashboard")
        self.setGeometry(100, 100, 900, 700)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # Title with modern styling
        title = QtWidgets.QLabel("📊 EEG Authentication Performance Dashboard")
        title.setStyleSheet("""
            font-size: 18px; 
            color: #ffffff;
            background-color: #3498db;
            padding: 15px;
            border-radius: 6px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Set main window styling
        self.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.show_stats_btn = QtWidgets.QPushButton("📈 Show Statistics")
        self.show_users_btn = QtWidgets.QPushButton("👥 User Analytics")
        self.export_btn = QtWidgets.QPushButton("📄 Export Report")
        
        # Style buttons
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 14px;
                padding: 10px 16px;
                border-radius: 5px;
                border: none;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #5dade2;
            }
        """
        
        self.show_stats_btn.setStyleSheet(button_style)
        self.show_users_btn.setStyleSheet(button_style.replace("#3498db", "#27ae60").replace("#5dade2", "#2ecc71"))
        self.export_btn.setStyleSheet(button_style.replace("#3498db", "#f39c12").replace("#5dade2", "#f1c40f"))
        
        btn_layout.addWidget(self.show_stats_btn)
        btn_layout.addWidget(self.show_users_btn)
        btn_layout.addWidget(self.export_btn)
        
        layout.addLayout(btn_layout)
        
        # Results area with modern styling
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #34495e;
                color: #ffffff;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #2c3e50;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.results_text)
        
        # Connect buttons
        self.show_stats_btn.clicked.connect(self.show_system_stats)
        self.show_users_btn.clicked.connect(self.show_user_analytics)
        self.export_btn.clicked.connect(self.export_report)
        
        self.setLayout(layout)
        
    def show_system_stats(self):
        """Display system statistics."""
        from backend import get_registered_users
        import backend
        
        stats = []
        stats.append("=" * 50)
        stats.append("🔍 EEG AUTHENTICATION SYSTEM STATISTICS")
        stats.append("=" * 50)
        stats.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        stats.append("")
        
        # User statistics
        users = get_registered_users()
        stats.append(f"👥 Total Registered Users: {len(users)}")
        
        if users:
            stats.append("📋 User Details:")
            for user in users:
                info = backend.get_user_info(user)
                if info:
                    stats.append(f"  • {user}: Subject ID {info['subject_id']}, {info['data_segments']} segments")
        
        # Model information
        model_exists = os.path.exists('assets/model.pth')
        stats.append(f"🤖 Model Status: {'✅ Trained' if model_exists else '❌ Not Trained'}")
        
        if model_exists:
            model_size = os.path.getsize('assets/model.pth') / (1024*1024)
            stats.append(f"📦 Model Size: {model_size:.2f} MB")
        
        # Data statistics
        data_dir = 'data/Filtered_Data'
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            stats.append(f"📊 Available Data Files: {len(csv_files)}")
            
            subjects = set()
            for file in csv_files:
                if file.startswith('s') and '_' in file:
                    subject = file.split('_')[0]
                    subjects.add(subject)
            stats.append(f"🧠 Available Subjects: {len(subjects)}")
        
        self.results_text.setText('\n'.join(stats))
        
    def show_user_analytics(self):
        """Show detailed user analytics."""
        from backend import get_registered_users, get_user_info
        
        analytics = []
        analytics.append("=" * 50)
        analytics.append("👥 USER ANALYTICS REPORT")
        analytics.append("=" * 50)
        
        users = get_registered_users()
        if not users:
            analytics.append("❌ No users registered yet.")
        else:
            for user in users:
                info = get_user_info(user)
                if info:
                    analytics.append(f"\n🔍 User: {user}")
                    analytics.append(f"  📋 Subject ID: {info['subject_id']}")
                    analytics.append(f"  📊 Data Segments: {info['data_segments']}")
                    analytics.append(f"  💾 Data File: {'✅ Exists' if info['data_exists'] else '❌ Missing'}")
                    
                    if info['data_exists'] and isinstance(info['data_segments'], int):
                        # Calculate data quality metrics
                        total_samples = info['data_segments'] * 256  # 256 samples per segment
                        duration_seconds = total_samples / 256  # Assuming 256 Hz sampling rate
                        analytics.append(f"  ⏱️  Total Duration: {duration_seconds:.1f} seconds")
                        analytics.append(f"  🔢 Total Samples: {total_samples:,}")
        
        self.results_text.setText('\n'.join(analytics))
        
    def export_report(self):
        """Export performance report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"eeg_report_{timestamp}.txt"
        
        # Combine all statistics
        report = []
        report.append("EEG BIOMETRIC AUTHENTICATION SYSTEM - PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add current display content
        report.append(self.results_text.toPlainText())
        
        try:
            with open(filename, 'w') as f:
                f.write('\n'.join(report))
            
            QtWidgets.QMessageBox.information(
                self, "✅ Export Successful", 
                f"Report exported to: {filename}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "❌ Export Failed", 
                f"Failed to export report: {str(e)}"
            )

def show_dashboard():
    """Show the performance dashboard."""
    import sys
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    dashboard = PerformanceDashboard()
    dashboard.show()
    return dashboard

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    dashboard = PerformanceDashboard()
    dashboard.show()
    sys.exit(app.exec_())
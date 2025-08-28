"""Settings panel for EEG system configuration."""
from PyQt5 import QtWidgets
import json
import os

class SettingsPanel(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ System Settings")
        self.setFixedSize(400, 300)
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Authentication Settings
        auth_group = QtWidgets.QGroupBox("Authentication Settings")
        auth_layout = QtWidgets.QFormLayout()
        
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.90)
        
        self.majority_spin = QtWidgets.QDoubleSpinBox()
        self.majority_spin.setRange(0.1, 1.0)
        self.majority_spin.setSingleStep(0.05)
        self.majority_spin.setValue(0.5)
        
        auth_layout.addRow("Confidence Threshold:", self.threshold_spin)
        auth_layout.addRow("Majority Vote Threshold:", self.majority_spin)
        auth_group.setLayout(auth_layout)
        
        # Model Settings
        model_group = QtWidgets.QGroupBox("Model Settings")
        model_layout = QtWidgets.QFormLayout()
        
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(10, 200)
        self.epochs_spin.setValue(50)
        
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(8, 128)
        self.batch_spin.setValue(32)
        
        model_layout.addRow("Training Epochs:", self.epochs_spin)
        model_layout.addRow("Batch Size:", self.batch_spin)
        model_group.setLayout(model_layout)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("💾 Save")
        self.cancel_btn = QtWidgets.QPushButton("❌ Cancel")
        self.reset_btn = QtWidgets.QPushButton("🔄 Reset")
        
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.reset_btn)
        
        layout.addWidget(auth_group)
        layout.addWidget(model_group)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
        # Connect buttons
        self.save_btn.clicked.connect(self.save_settings)
        self.cancel_btn.clicked.connect(self.reject)
        self.reset_btn.clicked.connect(self.reset_settings)
        
    def load_settings(self):
        """Load settings from file."""
        settings_file = "settings.json"
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    self.threshold_spin.setValue(settings.get('auth_threshold', 0.90))
                    self.majority_spin.setValue(settings.get('majority_threshold', 0.5))
                    self.epochs_spin.setValue(settings.get('epochs', 50))
                    self.batch_spin.setValue(settings.get('batch_size', 32))
            except:
                pass
                
    def save_settings(self):
        """Save settings to file."""
        settings = {
            'auth_threshold': self.threshold_spin.value(),
            'majority_threshold': self.majority_spin.value(),
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value()
        }
        
        with open("settings.json", 'w') as f:
            json.dump(settings, f, indent=2)
            
        QtWidgets.QMessageBox.information(self, "✅ Success", "Settings saved successfully!")
        self.accept()
        
    def reset_settings(self):
        """Reset to default settings."""
        self.threshold_spin.setValue(0.90)
        self.majority_spin.setValue(0.5)
        self.epochs_spin.setValue(50)
        self.batch_spin.setValue(32)
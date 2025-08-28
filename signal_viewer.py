"""EEG Signal Visualization Tool."""
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt5agg as backend
from matplotlib.figure import Figure
from PyQt5 import QtWidgets
import pandas as pd
import numpy as np
from config import CHANNELS

class SignalViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📈 EEG Signal Viewer")
        self.setGeometry(100, 100, 1000, 700)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Controls
        controls = QtWidgets.QHBoxLayout()
        
        self.load_btn = QtWidgets.QPushButton("📂 Load EEG File")
        self.load_btn.clicked.connect(self.load_file)
        
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(['All Channels'] + CHANNELS)
        self.channel_combo.currentTextChanged.connect(self.update_plot)
        
        controls.addWidget(self.load_btn)
        controls.addWidget(QtWidgets.QLabel("Channel:"))
        controls.addWidget(self.channel_combo)
        controls.addStretch()
        
        layout.addLayout(controls)
        
        # Matplotlib canvas
        self.figure = Figure(figsize=(12, 8))
        self.canvas = backend.FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        # Info panel
        self.info_label = QtWidgets.QLabel("Load an EEG file to view signals")
        self.info_label.setStyleSheet("padding: 10px; background-color: #34495e; color: white; border-radius: 5px;")
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
        self.data = None
        
    def load_file(self):
        """Load EEG CSV file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select EEG File', '', 'CSV Files (*.csv)'
        )
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path, usecols=CHANNELS)
                self.info_label.setText(f"📊 Loaded: {file_path.split('/')[-1]} | Shape: {self.data.shape} | Duration: {len(self.data)/256:.1f}s")
                self.update_plot()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def update_plot(self):
        """Update the signal plot."""
        if self.data is None:
            return
            
        self.figure.clear()
        
        selected_channel = self.channel_combo.currentText()
        
        if selected_channel == 'All Channels':
            # Plot all channels
            for i, channel in enumerate(CHANNELS):
                ax = self.figure.add_subplot(len(CHANNELS), 1, i+1)
                time = np.arange(len(self.data)) / 256  # Assuming 256 Hz
                ax.plot(time, self.data[channel], color=f'C{i}', linewidth=0.8)
                ax.set_ylabel(f'{channel} (μV)')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.set_title('EEG Signals - All Channels')
                if i == len(CHANNELS) - 1:
                    ax.set_xlabel('Time (seconds)')
        else:
            # Plot single channel
            ax = self.figure.add_subplot(1, 1, 1)
            time = np.arange(len(self.data)) / 256
            ax.plot(time, self.data[selected_channel], color='blue', linewidth=1)
            ax.set_title(f'EEG Signal - {selected_channel} Channel')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude (μV)')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = self.data[selected_channel].mean()
            std_val = self.data[selected_channel].std()
            ax.text(0.02, 0.98, f'Mean: {mean_val:.2f} μV\nStd: {std_val:.2f} μV', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.figure.tight_layout()
        self.canvas.draw()

def show_signal_viewer():
    """Show the signal viewer."""
    import sys
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    viewer = SignalViewer()
    viewer.show()
    return viewer

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    viewer = SignalViewer()
    viewer.show()
    sys.exit(app.exec_())
"""EEG frequency band analysis tool."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PyQt5 import QtWidgets
import matplotlib.backends.backend_qt5agg as backend
import pandas as pd
from config import CHANNELS

class FrequencyAnalyzer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌊 EEG Frequency Analysis")
        self.setGeometry(100, 100, 1000, 700)
        self.setup_ui()
        self.data = None
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Controls
        controls = QtWidgets.QHBoxLayout()
        
        self.load_btn = QtWidgets.QPushButton("📂 Load EEG File")
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(CHANNELS)
        self.analyze_btn = QtWidgets.QPushButton("🔍 Analyze Bands")
        
        controls.addWidget(self.load_btn)
        controls.addWidget(QtWidgets.QLabel("Channel:"))
        controls.addWidget(self.channel_combo)
        controls.addWidget(self.analyze_btn)
        controls.addStretch()
        
        layout.addLayout(controls)
        
        # Matplotlib canvas
        self.figure = plt.Figure(figsize=(12, 10))
        self.canvas = backend.FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        # Results
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setMaximumHeight(120)
        layout.addWidget(self.results_text)
        
        self.setLayout(layout)
        
        # Connect buttons
        self.load_btn.clicked.connect(self.load_file)
        self.analyze_btn.clicked.connect(self.analyze_frequency_bands)
        
    def load_file(self):
        """Load EEG file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select EEG File', '', 'CSV Files (*.csv)'
        )
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path, usecols=CHANNELS)
                self.results_text.setText(f"Loaded: {file_path.split('/')[-1]} | Shape: {self.data.shape}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")
                
    def analyze_frequency_bands(self):
        """Analyze EEG frequency bands."""
        if self.data is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load an EEG file first.")
            return
            
        channel = self.channel_combo.currentText()
        signal_data = self.data[channel].values
        
        # Sampling frequency
        fs = 256  # Hz
        
        # Compute power spectral density
        freqs, psd = signal.welch(signal_data, fs, nperseg=1024)
        
        # Define frequency bands
        bands = {
            'Delta (0.5-4 Hz)': (0.5, 4),
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-13 Hz)': (8, 13),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-50 Hz)': (30, 50)
        }
        
        # Calculate band powers
        band_powers = {}
        for band_name, (low, high) in bands.items():
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            band_powers[band_name] = np.trapz(psd[idx], freqs[idx])
            
        # Plot results
        self.figure.clear()
        
        # PSD plot
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax1.semilogy(freqs, psd)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power Spectral Density')
        ax1.set_title(f'PSD - {channel} Channel')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 50)
        
        # Band powers bar plot
        ax2 = self.figure.add_subplot(2, 2, 2)
        band_names = list(band_powers.keys())
        powers = list(band_powers.values())
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        
        bars = ax2.bar(range(len(band_names)), powers, color=colors, alpha=0.7)
        ax2.set_xlabel('Frequency Bands')
        ax2.set_ylabel('Power')
        ax2.set_title('Band Power Distribution')
        ax2.set_xticks(range(len(band_names)))
        ax2.set_xticklabels([name.split(' ')[0] for name in band_names], rotation=45)
        
        # Relative power pie chart
        ax3 = self.figure.add_subplot(2, 2, 3)
        total_power = sum(powers)
        relative_powers = [p/total_power*100 for p in powers]
        
        ax3.pie(relative_powers, labels=[name.split(' ')[0] for name in band_names], 
                colors=colors, autopct='%1.1f%%')
        ax3.set_title('Relative Band Power')
        
        # Spectrogram
        ax4 = self.figure.add_subplot(2, 2, 4)
        f, t, Sxx = signal.spectrogram(signal_data, fs, nperseg=256)
        im = ax4.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_xlabel('Time (s)')
        ax4.set_title('Spectrogram')
        ax4.set_ylim(0, 50)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Display results
        results = [f"=== FREQUENCY ANALYSIS - {channel} CHANNEL ==="]
        for band_name, power in band_powers.items():
            percentage = (power / total_power) * 100
            results.append(f"{band_name}: {power:.2e} ({percentage:.1f}%)")
            
        self.results_text.setText('\n'.join(results))

def show_frequency_analyzer():
    """Show frequency analyzer."""
    import sys
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    analyzer = FrequencyAnalyzer()
    analyzer.show()
    return analyzer
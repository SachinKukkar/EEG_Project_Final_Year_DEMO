import sys
import os
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
import backend

# ===================================================================
# MODERN DARK THEME STYLESHEET
# ===================================================================
DARK_THEME_STYLESHEET = """
    /* Main Window */
    QMainWindow {
        background-color: #2c3e50; /* Dark blue-grey background */
    }

    /* All Labels */
    QLabel {
        color: #ecf0f1; /* Light grey text */
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Group Boxes (for registration/authentication sections) */
    QGroupBox {
        color: #ecf0f1;
        font: bold 14px;
        border: 1px solid #34495e; /* Slightly lighter border */
        border-radius: 6px;
        margin-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 5px;
    }

    /* All Buttons */
    QPushButton {
        color: #ffffff;
        background-color: #3498db; /* Bright blue */
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        font: bold 12px;
    }
    QPushButton:hover {
        background-color: #5dade2; /* Lighter blue on hover */
    }
    QPushButton:pressed {
        background-color: #217dbb; /* Darker blue when pressed */
    }

    /* Text Input Fields */
    QLineEdit {
        background-color: #34495e; /* Darker input field */
        color: #ecf0f1;
        border: 1px solid #2c3e50;
        border-radius: 5px;
        padding: 8px;
    }
    QLineEdit:focus {
        border: 1px solid #3498db; /* Blue border when selected */
    }
    
    /* SpinBox for Subject ID */
    QSpinBox {
        background-color: #34495e;
        color: #ecf0f1;
        border: 1px solid #2c3e50;
        border-radius: 5px;
        padding: 8px;
    }
    QSpinBox:focus {
        border: 1px solid #3498db;
    }

    /* The main Status Label */
    #StatusLabel {
        font: bold 14px;
        color: #f1c40f; /* Yellow for status */
    }
"""

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('main_window.ui', self)
        
        self.selected_file_path = ""

        # --- Set Icons for Buttons ---
        self.setup_icons()

        # --- Connect Button Clicks to Functions ---
        self.RegisterButton.clicked.connect(self.register_clicked)
        self.TrainButton.clicked.connect(self.train_clicked)
        self.BrowseButton.clicked.connect(self.browse_clicked)
        self.AuthenticateButton.clicked.connect(self.authenticate_clicked)
        
        self.show()

    def setup_icons(self):
        """Sets icons for the main buttons."""
        # Note: You need to download these icons and place them in the 'assets' folder.
        try:
            self.RegisterButton.setIcon(QIcon('assets/user-plus.svg'))
            self.TrainButton.setIcon(QIcon('assets/cpu.svg'))
            self.BrowseButton.setIcon(QIcon('assets/folder.svg'))
            self.AuthenticateButton.setIcon(QIcon('assets/log-in.svg'))
            
            # Optional: Adjust icon size for better appearance
            self.RegisterButton.setIconSize(QSize(20, 20))
            self.AuthenticateButton.setIconSize(QSize(20, 20))
        except Exception as e:
            print(f"Could not load icons. Please check the 'assets' folder. Error: {e}")

    # --- Backend Connection Functions ---
    def register_clicked(self):
        username = self.UsernameLineEdit.text()
        subject_id = self.SubjectIDSpinBox.value()
        if not username or subject_id == 0:
            self.update_status("Please provide a username and valid Subject ID.", "red")
            return
        
        self.update_status(f"Registering {username}...", "#f1c40f") # Yellow
        success = backend.register_user(username, subject_id)
        if success:
            self.update_status("Registration successful.", "green")
        else:
            self.update_status("Registration failed. Check terminal for errors.", "red")

    def train_clicked(self):
        self.update_status("Training model... This may take a while.", "#f1c40f")
        QtWidgets.QApplication.processEvents() # Keep GUI responsive
        success = backend.train_model()
        if success:
            self.update_status("Training complete.", "green")
        else:
            self.update_status("Training failed. Check terminal for errors.", "red")

    def browse_clicked(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select EEG File', '', 'CSV Files (*.csv)')
        if file_path:
            self.selected_file_path = file_path
            self.update_status(f"Selected: {os.path.basename(file_path)}", "#3498db") # Blue

    def authenticate_clicked(self):
        username = self.UsernameLineEdit.text()
        if not username or not self.selected_file_path:
            self.update_status("Provide username and select a file.", "red")
            return

        self.update_status(f"Authenticating {username}...", "#f1c40f")
        QtWidgets.QApplication.processEvents()
        is_auth = backend.authenticate(username, self.selected_file_path)
        
        if is_auth:
            self.update_status("ACCESS GRANTED", "green")
        else:
            self.update_status("ACCESS DENIED", "red")

    def update_status(self, message, color):
        """Helper function to update the status label text and color."""
        self.StatusLabel.setText(message)
        self.StatusLabel.setStyleSheet(f"color: {color}; font: bold 14px;")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # --- Apply the stylesheet to the entire app ---
    app.setStyleSheet(DARK_THEME_STYLESHEET)
    
    window = MainWindow()
    sys.exit(app.exec_())
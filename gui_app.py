import sys
import os
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, Qt
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

        # --- Set Subject ID Range ---
        self.SubjectIDSpinBox.setRange(1, 20)  # Based on your data (s01 to s20)
        self.SubjectIDSpinBox.setValue(1)
        
        # --- Connect Button Clicks to Functions ---
        self.RegisterButton.clicked.connect(self.register_clicked)
        self.TrainButton.clicked.connect(self.train_clicked)
        self.BrowseButton.clicked.connect(self.browse_clicked)
        self.AuthenticateButton.clicked.connect(self.authenticate_clicked)
        
        # Connect de-register button (should exist in UI now)
        if hasattr(self, 'DeregisterButton'):
            self.DeregisterButton.clicked.connect(self.deregister_clicked)
        
        # Setup context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        self.show()

    def setup_icons(self):
        """Sets icons for the main buttons."""
        # Icons disabled to prevent file not found errors
        # You can add icon files to assets/ folder and uncomment below if needed
        pass
        # try:
        #     self.RegisterButton.setIcon(QIcon('assets/user-plus.svg'))
        #     self.TrainButton.setIcon(QIcon('assets/cpu.svg'))
        #     self.BrowseButton.setIcon(QIcon('assets/folder.svg'))
        #     self.AuthenticateButton.setIcon(QIcon('assets/log-in.svg'))
        # except Exception as e:
        #     print(f"Could not load icons: {e}")

    # --- Backend Connection Functions ---
    def register_clicked(self):
        username = self.UsernameLineEdit.text().strip()
        subject_id = self.SubjectIDSpinBox.value()
        
        if not username:
            self.update_status("Please provide a username.", "red")
            return
        if subject_id == 0:
            self.update_status("Please select a valid Subject ID (1-20).", "red")
            return
        
        self.update_status(f"Registering {username}...", "#f1c40f") # Yellow
        QtWidgets.QApplication.processEvents()
        
        result = backend.register_user(username, subject_id)
        if isinstance(result, tuple):
            success, message = result
            if success:
                self.update_status(f"✅ {message}", "green")
                self.UsernameLineEdit.clear()
                self.SubjectIDSpinBox.setValue(0)
            else:
                self.update_status(f"❌ {message}", "red")
        else:
            # Handle old return format (backward compatibility)
            if result:
                self.update_status("Registration successful.", "green")
            else:
                self.update_status("Registration failed. Check terminal for errors.", "red")
    
    def deregister_clicked(self):
        username = self.UsernameLineEdit.text().strip()
        
        if not username:
            self.update_status("Please provide a username to de-register.", "red")
            return
        
        # Get user info for confirmation dialog
        user_info = backend.get_user_info(username)
        if not user_info:
            self.update_status(f"❌ User '{username}' is not registered.", "red")
            return
        
        # Create custom confirmation dialog
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("⚠️ Confirm De-registration")
        msg_box.setIcon(QtWidgets.QMessageBox.Warning)
        
        # Enhanced message with user details
        message = f"""<h3>De-register User: <span style='color: #e74c3c;'>{username}</span></h3>
        
<b>User Details:</b>
• Subject ID: <b>{user_info['subject_id']}</b>
• Data Segments: <b>{user_info.get('data_segments', 'Unknown')}</b>
• Data File: <b>{'✅ Exists' if user_info['data_exists'] else '❌ Missing'}</b>

<p style='color: #e74c3c; font-weight: bold;'>⚠️ WARNING: This action cannot be undone!</p>
<p>All EEG training data for this user will be <b>permanently deleted</b>.</p>
<p>You will need to re-register and re-train the model if you want to use this user again.</p>

<p style='color: #2c3e50;'>Are you sure you want to proceed?</p>"""
        
        msg_box.setText(message)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        msg_box.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        
        # Customize button text
        yes_button = msg_box.button(QtWidgets.QMessageBox.Yes)
        yes_button.setText("🗑️ Delete User")
        yes_button.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; font-weight: bold; padding: 8px 16px; }")
        
        cancel_button = msg_box.button(QtWidgets.QMessageBox.Cancel)
        cancel_button.setText("❌ Cancel")
        cancel_button.setStyleSheet("QPushButton { background-color: #95a5a6; color: white; padding: 8px 16px; }")
        
        # Set dialog size and styling
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #ecf0f1;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QMessageBox QLabel {
                color: #2c3e50;
                font-size: 12px;
                padding: 10px;
            }
        """)
        
        reply = msg_box.exec_()
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.update_status(f"🗑️ De-registering {username}...", "#f39c12")
            QtWidgets.QApplication.processEvents()
            
            result = backend.deregister_user(username)
            if isinstance(result, tuple):
                success, message = result
                if success:
                    self.update_status(f"✅ {message}", "green")
                    self.UsernameLineEdit.clear()
                    self.SubjectIDSpinBox.setValue(1)
                    
                    # Show enhanced success notification
                    success_msg = QtWidgets.QMessageBox(self)
                    success_msg.setWindowTitle("✅ De-registration Complete")
                    success_msg.setIcon(QtWidgets.QMessageBox.Information)
                    
                    success_text = f"""<div style='text-align: center;'>
<h2 style='color: #27ae60; margin-bottom: 15px;'>✅ Success!</h2>
<h3 style='color: #2c3e50;'>User '<span style='color: #e74c3c; font-weight: bold;'>{username}</span>' has been de-registered</h3>

<div style='background-color: #d5f4e6; padding: 15px; border-radius: 8px; margin: 10px 0;'>
<p style='color: #27ae60; font-weight: bold; margin: 5px 0;'>✓ User removed from registry</p>
<p style='color: #27ae60; font-weight: bold; margin: 5px 0;'>✓ EEG data file deleted</p>
<p style='color: #27ae60; font-weight: bold; margin: 5px 0;'>✓ Subject ID {user_info['subject_id']} is now available</p>
</div>

<p style='color: #7f8c8d; font-style: italic; margin-top: 15px;'>The model will need to be retrained if you have other users.</p>
</div>"""
                    
                    success_msg.setText(success_text)
                    success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    
                    # Style the OK button
                    ok_button = success_msg.button(QtWidgets.QMessageBox.Ok)
                    ok_button.setText("✓ Got it!")
                    ok_button.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; padding: 10px 20px; border-radius: 5px; }")
                    
                    # Style the dialog
                    success_msg.setStyleSheet("""
                        QMessageBox {
                            background-color: #f8f9fa;
                            font-family: 'Segoe UI', Arial, sans-serif;
                            min-width: 400px;
                        }
                        QMessageBox QLabel {
                            color: #2c3e50;
                            font-size: 13px;
                            padding: 20px;
                        }
                    """)
                    
                    success_msg.exec_()
                else:
                    self.update_status(f"❌ {message}", "red")
            else:
                self.update_status("De-registration failed. Check terminal for errors.", "red")
    
    def show_registered_users(self):
        """Show list of registered users."""
        users = backend.get_registered_users()
        if not users:
            QtWidgets.QMessageBox.information(self, 'Registered Users', 'No users registered yet.')
        else:
            user_list = '\n'.join([f"• {user}" for user in users])
            QtWidgets.QMessageBox.information(self, 'Registered Users', f'Registered Users:\n\n{user_list}')

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
        username = self.UsernameLineEdit.text().strip()
        if not username:
            self.update_status("Please provide a username.", "red")
            return
        if not self.selected_file_path:
            self.update_status("Please select an EEG file for authentication.", "red")
            return

        self.update_status(f"Authenticating {username}...", "#f1c40f")
        QtWidgets.QApplication.processEvents()
        
        result = backend.authenticate(username, self.selected_file_path)
        if isinstance(result, tuple):
            is_auth, reason = result
            if is_auth:
                self.update_status(f"✅ {reason}", "green")
            else:
                self.update_status(f"❌ {reason}", "red")
        else:
            # Handle old return format (backward compatibility)
            if result:
                self.update_status("✅ ACCESS GRANTED", "green")
            else:
                self.update_status("❌ ACCESS DENIED", "red")

    def update_status(self, message, color):
        """Helper function to update the status label text and color."""
        self.StatusLabel.setText(message)
        self.StatusLabel.setStyleSheet(f"color: {color}; font: bold 12px; padding: 5px;")
        # Enable word wrap for long messages
        self.StatusLabel.setWordWrap(True)
    
    def show_context_menu(self, position):
        """Show context menu with additional options."""
        context_menu = QtWidgets.QMenu(self)
        
        show_users_action = context_menu.addAction("Show Registered Users")
        show_users_action.triggered.connect(self.show_registered_users)
        
        context_menu.exec_(self.mapToGlobal(position))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # --- Apply the stylesheet to the entire app ---
    app.setStyleSheet(DARK_THEME_STYLESHEET)
    
    window = MainWindow()
    sys.exit(app.exec_())
import sys
import os
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, Qt
import backend
from themes import LIGHT_THEME, DARK_THEME
import traceback

# Using themes from themes.py file

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('main_window.ui', self)
        
        self.selected_file_path = ""
        self.is_dark_theme = True  # Start with dark theme

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
        
        # Add top toolbar with hamburger menu and theme toggle
        self.create_toolbar()
        
        # Add keyboard shortcuts
        self.setup_shortcuts()
        
        # Set window properties
        self.setMinimumSize(600, 480)
        self.setWindowTitle("🧠 EEG Biometric Authentication System")
        
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
                self.SubjectIDSpinBox.setValue(1)
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
        yes_button.setStyleSheet("""
            QPushButton { 
                background-color: #e74c3c; 
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                padding: 10px 16px; 
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ec7063;
            }
        """)
        
        cancel_button = msg_box.button(QtWidgets.QMessageBox.Cancel)
        cancel_button.setText("❌ Cancel")
        cancel_button.setStyleSheet("""
            QPushButton { 
                background-color: #95a5a6; 
                color: white; 
                font-size: 14px;
                font-weight: 500;
                padding: 10px 16px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #a9b7bc;
            }
        """)
        
        # Set dialog size and styling
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #ecf0f1;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QMessageBox QLabel {
                color: #2c3e50;
                font-size: 14px;
                padding: 15px;
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
                    ok_button.setStyleSheet("""
                        QPushButton { 
                            background-color: #27ae60; 
                            color: white; 
                            font-weight: bold; 
                            font-size: 14px;
                            padding: 10px 16px; 
                            border-radius: 6px;
                            border: none;
                        }
                        QPushButton:hover {
                            background-color: #2ecc71;
                        }
                    """)
                    
                    # Style the dialog
                    success_msg.setStyleSheet("""
                        QMessageBox {
                            background-color: #f8f9fa;
                            font-family: 'Segoe UI', Arial, sans-serif;
                            min-width: 400px;
                        }
                        QMessageBox QLabel {
                            color: #2c3e50;
                            font-size: 14px;
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
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("👥 Registered Users")
        
        if not users:
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("<h3>No users registered yet</h3><p>Register some users to get started!</p>")
        else:
            msg.setIcon(QtWidgets.QMessageBox.Information)
            user_list = '<br>'.join([f"• <b>{user}</b>" for user in users])
            msg.setText(f"<h3>👥 Registered Users ({len(users)})</h3><br>{user_list}")
        
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Arial, sans-serif;
                min-width: 350px;
            }
            QMessageBox QLabel {
                color: #2c3e50;
                font-size: 14px;
                padding: 15px;
            }
        """)
        msg.exec_()

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
        subject_id = self.SubjectIDSpinBox.value()
        
        if not username:
            self.update_status("Please provide a username.", "red")
            return
        if subject_id == 0:
            self.update_status("Please select a valid Subject ID (1-20).", "red")
            return
        if not self.selected_file_path:
            self.update_status("Please select an EEG file for authentication.", "red")
            return

        self.update_status(f"Authenticating {username}...", "#f1c40f")
        QtWidgets.QApplication.processEvents()
        
        result = backend.authenticate_with_subject_id(username, subject_id, self.selected_file_path)
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
        self.StatusLabel.setStyleSheet(f"""
            color: {color}; 
            font-size: 14px; 
            font-weight: 500; 
            padding: 12px; 
            background-color: rgba(44, 62, 80, 0.7);
            border: 1px solid #34495e;
            border-radius: 6px;
            margin: 8px;
        """)
        # Enable word wrap for long messages
        self.StatusLabel.setWordWrap(True)
        
        # Set window properties (moved to __init__)
        pass
    
    def create_toolbar(self):
        """Create top toolbar with hamburger menu and theme toggle."""
        # Create toolbar widget
        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        
        # Hamburger menu button
        self.menu_btn = QtWidgets.QPushButton("☰")
        self.menu_btn.setFixedSize(40, 30)
        self.menu_btn.clicked.connect(self.show_hamburger_menu)
        
        # Theme toggle button
        self.theme_btn = QtWidgets.QPushButton("🌙")
        self.theme_btn.setFixedSize(40, 30)
        self.theme_btn.clicked.connect(self.toggle_theme)
        
        # Dashboard button (compact)
        self.dashboard_btn = QtWidgets.QPushButton("📊")
        self.dashboard_btn.setFixedSize(40, 30)
        self.dashboard_btn.clicked.connect(self.show_dashboard)
        self.dashboard_btn.setToolTip("Performance Dashboard")
        
        toolbar_layout.addWidget(self.menu_btn)
        toolbar_layout.addWidget(self.dashboard_btn)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.theme_btn)
        
        # Add toolbar to main layout
        if hasattr(self, 'centralwidget'):
            main_layout = self.centralwidget.layout()
            if main_layout:
                main_layout.insertWidget(0, toolbar)
    
    def show_hamburger_menu(self):
        """Show hamburger menu with all tools."""
        menu = QtWidgets.QMenu(self)
        
        # User Management
        show_users_action = menu.addAction("👥 Show Users")
        show_users_action.triggered.connect(self.show_registered_users)
        
        menu.addSeparator()
        
        # Analysis Tools
        signal_viewer_action = menu.addAction("📈 Signal Viewer")
        signal_viewer_action.triggered.connect(self.show_signal_viewer)
        
        freq_analyzer_action = menu.addAction("🌊 Frequency Analyzer")
        freq_analyzer_action.triggered.connect(self.show_frequency_analyzer)
        
        performance_action = menu.addAction("📉 Performance Analysis")
        performance_action.triggered.connect(self.show_performance_analyzer)
        
        model_compare_action = menu.addAction("🔬 Model Comparison")
        model_compare_action.triggered.connect(self.show_model_comparison)
        
        menu.addSeparator()
        
        # Settings
        settings_action = menu.addAction("⚙️ Settings")
        settings_action.triggered.connect(self.show_settings)
        
        # Show menu at button position
        menu.exec_(self.menu_btn.mapToGlobal(self.menu_btn.rect().bottomLeft()))
    
    def show_dashboard(self):
        """Show performance dashboard."""
        try:
            import dashboard
            print("Dashboard module imported successfully")
            self.dashboard_window = dashboard.show_dashboard()
            print("Dashboard window created")
            self.update_status("Dashboard opened", "green")
        except ImportError as e:
            print(f"Dashboard import error: {e}")
            self.update_status(f"❌ Dashboard module missing", "red")
        except Exception as e:
            print(f"Dashboard error: {e}")
            print(traceback.format_exc())
            self.update_status(f"❌ Dashboard error: {str(e)}", "red")
    
    def show_signal_viewer(self):
        """Show EEG signal viewer."""
        try:
            import signal_viewer
            self.signal_viewer_window = signal_viewer.show_signal_viewer()
        except Exception as e:
            print(f"Signal viewer error: {e}")
            print(traceback.format_exc())
            self.update_status(f"❌ Signal viewer unavailable", "red")
    
    def show_frequency_analyzer(self):
        """Show frequency analyzer."""
        try:
            import frequency_analyzer
            self.freq_analyzer_window = frequency_analyzer.show_frequency_analyzer()
        except Exception as e:
            print(f"Frequency analyzer error: {e}")
            self.update_status(f"❌ Frequency analyzer unavailable", "red")
    
    def show_performance_analyzer(self):
        """Show performance analyzer."""
        try:
            import performance_analyzer
            self.performance_window = performance_analyzer.show_performance_analyzer()
        except Exception as e:
            print(f"Performance analyzer error: {e}")
            self.update_status(f"❌ Performance analyzer unavailable", "red")
    
    def show_model_comparison(self):
        """Show model comparison tool."""
        try:
            import model_comparison
            self.model_comparison_window = model_comparison.show_model_comparison()
        except Exception as e:
            print(f"Model comparison error: {e}")
            self.update_status(f"❌ Model comparison unavailable", "red")
    
    def show_settings(self):
        """Show settings panel."""
        try:
            import settings
            settings_dialog = settings.SettingsPanel(self)
            settings_dialog.exec_()
        except Exception as e:
            print(f"Settings error: {e}")
            self.update_status(f"❌ Settings unavailable", "red")
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        
        # Ctrl+R for Register
        register_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        register_shortcut.activated.connect(self.register_clicked)
        
        # Ctrl+T for Train
        train_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        train_shortcut.activated.connect(self.train_clicked)
        
        # Ctrl+A for Authenticate
        auth_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        auth_shortcut.activated.connect(self.authenticate_clicked)
        
        # Ctrl+D for Dashboard
        dashboard_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        dashboard_shortcut.activated.connect(self.show_dashboard)
        
        # Ctrl+S for Settings
        settings_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        settings_shortcut.activated.connect(self.show_settings)
        
        # F1 for Help
        help_shortcut = QShortcut(QKeySequence("F1"), self)
        help_shortcut.activated.connect(self.show_help)
        
        # Escape to close any open dialogs
        escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        escape_shortcut.activated.connect(self.close_dialogs)
    
    def show_help(self):
        """Show help dialog."""
        help_text = """
🔑 KEYBOARD SHORTCUTS:

Ctrl+R - Register User
Ctrl+T - Train Model
Ctrl+A - Authenticate
Ctrl+D - Dashboard
Ctrl+S - Settings
F1 - Show Help

🖱️ RIGHT-CLICK MENU:
- Show Registered Users
- EEG Signal Viewer
- Frequency Analyzer
- Performance Analysis
- Model Comparison
- Settings
        """
        
        QtWidgets.QMessageBox.information(self, "📖 Help", help_text)
    
    def close_dialogs(self):
        """Close any open dialog windows."""
        for widget in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(widget, QtWidgets.QDialog) and widget.isVisible():
                widget.close()
    
    def toggle_theme(self):
        """Toggle between light and dark theme."""
        self.is_dark_theme = not self.is_dark_theme
        
        if self.is_dark_theme:
            QtWidgets.QApplication.instance().setStyleSheet(DARK_THEME)
            self.theme_btn.setText("🌙")
            self.update_status("Dark Theme", "#3498db")
        else:
            QtWidgets.QApplication.instance().setStyleSheet(LIGHT_THEME)
            self.theme_btn.setText("☀️")
            self.update_status("Light Theme", "#007bff")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # --- Apply the stylesheet to the entire app ---
    app.setStyleSheet(DARK_THEME)
    
    window = MainWindow()
    sys.exit(app.exec_())
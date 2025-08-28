"""Theme management for EEG system."""

LIGHT_THEME = """
    QMainWindow {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    QLabel {
        color: #2c3e50;
        font-size: 14px;
    }
    
    QGroupBox {
        color: #2c3e50;
        font-size: 15px;
        font-weight: 600;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 15px;
        background-color: #ffffff;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 5px 10px;
        background-color: #007bff;
        border-radius: 4px;
        color: white;
    }
    
    QPushButton {
        color: #ffffff;
        background-color: #007bff;
        border: none;
        padding: 10px 16px;
        border-radius: 5px;
        font-size: 14px;
        font-weight: 500;
        min-height: 16px;
    }
    QPushButton:hover {
        background-color: #0056b3;
    }
    QPushButton:pressed {
        background-color: #004085;
    }
    
    QPushButton[text="Register User"] {
        background-color: #28a745;
    }
    QPushButton[text="Register User"]:hover {
        background-color: #1e7e34;
    }
    
    QPushButton[text="De-register User"] {
        background-color: #dc3545;
    }
    QPushButton[text="De-register User"]:hover {
        background-color: #c82333;
    }
    
    QLineEdit {
        background-color: #ffffff;
        color: #2c3e50;
        border: 1px solid #ced4da;
        border-radius: 5px;
        padding: 8px 12px;
        font-size: 14px;
    }
    QLineEdit:focus {
        border: 1px solid #007bff;
        background-color: #f8f9fa;
    }
    
    QSpinBox {
        background-color: #ffffff;
        color: #2c3e50;
        border: 1px solid #ced4da;
        border-radius: 5px;
        padding: 8px 12px;
        font-size: 14px;
        min-width: 80px;
    }
    QSpinBox:focus {
        border: 1px solid #007bff;
        background-color: #f8f9fa;
    }
    
    #StatusLabel {
        font-size: 14px;
        font-weight: 500;
        color: #856404;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 6px;
        padding: 12px;
        margin: 8px;
    }
"""

DARK_THEME = """
    QMainWindow {
        background-color: #2c3e50;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    QLabel {
        color: #ecf0f1;
        font-size: 14px;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    QGroupBox {
        color: #ecf0f1;
        font-size: 15px;
        font-weight: 600;
        border: 1px solid #34495e;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 15px;
        background-color: #34495e;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 5px 10px;
        background-color: #3498db;
        border-radius: 4px;
        color: white;
    }

    QPushButton {
        color: #ffffff;
        background-color: #3498db;
        border: none;
        padding: 10px 16px;
        border-radius: 5px;
        font-size: 14px;
        font-weight: 500;
        min-height: 16px;
    }
    QPushButton:hover {
        background-color: #5dade2;
    }
    QPushButton:pressed {
        background-color: #2980b9;
    }

    QPushButton[text="Register User"] {
        background-color: #27ae60;
    }
    QPushButton[text="Register User"]:hover {
        background-color: #2ecc71;
    }

    QPushButton[text="De-register User"] {
        background-color: #e74c3c;
    }
    QPushButton[text="De-register User"]:hover {
        background-color: #ec7063;
    }

    QLineEdit {
        background-color: #34495e;
        color: #ecf0f1;
        border: 1px solid #2c3e50;
        border-radius: 5px;
        padding: 8px 12px;
        font-size: 14px;
    }
    QLineEdit:focus {
        border: 1px solid #3498db;
        background-color: #3d566e;
    }
    
    QSpinBox {
        background-color: #34495e;
        color: #ecf0f1;
        border: 1px solid #2c3e50;
        border-radius: 5px;
        padding: 8px 12px;
        font-size: 14px;
        min-width: 80px;
    }
    QSpinBox:focus {
        border: 1px solid #3498db;
        background-color: #3d566e;
    }

    #StatusLabel {
        font-size: 14px;
        font-weight: 500;
        color: #f1c40f;
        background-color: rgba(44, 62, 80, 0.7);
        border: 1px solid #34495e;
        border-radius: 6px;
        padding: 12px;
        margin: 8px;
    }
"""
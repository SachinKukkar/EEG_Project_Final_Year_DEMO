"""Utility functions for EEG processing project."""
import logging
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f'eeg_system_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_file_path(file_path):
    """Validate if file exists and is a CSV file."""
    if not os.path.exists(file_path):
        return False, "File does not exist"
    if not file_path.lower().endswith('.csv'):
        return False, "File must be a CSV file"
    return True, "Valid file"

def get_model_info():
    """Get information about the current model."""
    from config import MODEL_PATH, ENCODER_PATH, SCALER_PATH
    
    info = {
        'model_exists': os.path.exists(MODEL_PATH),
        'encoder_exists': os.path.exists(ENCODER_PATH),
        'scaler_exists': os.path.exists(SCALER_PATH)
    }
    
    if info['model_exists']:
        info['model_size'] = os.path.getsize(MODEL_PATH)
        info['model_modified'] = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
    
    return info
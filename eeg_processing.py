import os
import pandas as pd
import numpy as np
from config import CHANNELS, WINDOW_SIZE, STEP_SIZE

# Simple logging setup
try:
    import logging
    logger = logging.getLogger(__name__)
except:
    # Fallback if logging fails
    class DummyLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    logger = DummyLogger()

def get_subject_files(data_dir, subject_id):
    """Finds all resting-state EEG files for a specific subject."""
    try:
        if not os.path.exists(data_dir):
            logger.error(f"Data directory does not exist: {data_dir}")
            return []
            
        subject_files = []
        subject_str = f"s{subject_id:02d}"
        
        for filename in os.listdir(data_dir):
            # We only use resting-state data (ex01 and ex02) for stable biometrics
            if filename.startswith(subject_str) and ('_ex01_' in filename or '_ex02_' in filename):
                file_path = os.path.join(data_dir, filename)
                if os.path.getsize(file_path) > 0:  # Check file is not empty
                    subject_files.append(file_path)
                    
        logger.info(f"Found {len(subject_files)} files for subject {subject_id}")
        return subject_files
        
    except Exception as e:
        logger.error(f"Error finding subject files: {e}")
        return []

def load_and_segment_csv(file_path):
    """Loads a single filtered CSV file and segments it into windows."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return np.array([])
            
        # Use pandas to read the CSV. We specify the exact columns to use.
        df = pd.read_csv(file_path, usecols=CHANNELS)
        
        # Check if all required channels are present
        missing_channels = set(CHANNELS) - set(df.columns)
        if missing_channels:
            logger.error(f"Missing channels in {file_path}: {missing_channels}")
            return np.array([])
        
        # Ensure the column order is consistent
        df = df[CHANNELS]
        
        # Check for NaN values
        if df.isnull().any().any():
            logger.warning(f"NaN values found in {file_path}, filling with zeros")
            df = df.fillna(0)
        
        eeg_data = df.to_numpy()
        segments = []
        
        for i in range(0, len(eeg_data) - WINDOW_SIZE + 1, STEP_SIZE):
            segment = eeg_data[i : i + WINDOW_SIZE]
            if len(segment) == WINDOW_SIZE:
                segments.append(segment)
                
        logger.info(f"Extracted {len(segments)} segments from {os.path.basename(file_path)}")
        return np.array(segments)
        
    except Exception as e:
        logger.error(f"Error loading and segmenting {file_path}: {e}")
        return np.array([])

def validate_eeg_data(data):
    """Validate EEG data quality."""
    if len(data) == 0:
        return False, "No data segments found"
        
    # Check for reasonable signal amplitude (basic sanity check)
    mean_amplitude = np.mean(np.abs(data))
    if mean_amplitude < 1e-6:
        return False, "Signal amplitude too low"
    if mean_amplitude > 1000:
        return False, "Signal amplitude too high"
        
    return True, "Data validation passed"
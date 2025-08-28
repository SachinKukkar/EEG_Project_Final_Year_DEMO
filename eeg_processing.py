import os
import pandas as pd
import numpy as np

# --- CONFIGURATION BASED ON YOUR CSV FILE ---
# IMPORTANT: This channel order matches your s17_ex01_s03.csv file
CHANNELS = ['P4', 'Cz', 'F8', 'T7'] 
WINDOW_SIZE = 256  # The size of each data chunk for the CNN
STEP_SIZE = 128    # Overlap windows for more data samples

def get_subject_files(data_dir, subject_id):
    """Finds all resting-state EEG files for a specific subject."""
    subject_files = []
    subject_str = f"s{subject_id:02d}"
    
    for filename in os.listdir(data_dir):
        # We only use resting-state data (ex01 and ex02) for stable biometrics
        if filename.startswith(subject_str) and ('_ex01_' in filename or '_ex02_' in filename):
            subject_files.append(os.path.join(data_dir, filename))
            
    return subject_files

def load_and_segment_csv(file_path):
    """Loads a single filtered CSV file and segments it into windows."""
    # Use pandas to read the CSV. We specify the exact columns to use.
    df = pd.read_csv(file_path, usecols=CHANNELS)
    
    # Ensure the column order is consistent
    df = df[CHANNELS]
    
    eeg_data = df.to_numpy()
    segments = []
    for i in range(0, len(eeg_data) - WINDOW_SIZE, STEP_SIZE):
        segment = eeg_data[i : i + WINDOW_SIZE]
        if len(segment) == WINDOW_SIZE:
            segments.append(segment)
            
    return np.array(segments)
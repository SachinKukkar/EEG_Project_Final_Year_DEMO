"""Configuration settings for EEG processing project."""
import os

# Data Configuration
CHANNELS = ['P4', 'Cz', 'F8', 'T7']
WINDOW_SIZE = 256
STEP_SIZE = 128
SAMPLING_RATE = 256  # Hz

# Model Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 5
DROPOUT_RATE = 0.5

# Authentication Configuration
AUTH_THRESHOLD = 0.90
MAJORITY_VOTE_THRESHOLD = 0.5

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Filtered_Data')
MODEL_PATH = os.path.join(ASSETS_DIR, 'model.pth')
ENCODER_PATH = os.path.join(ASSETS_DIR, 'label_encoder.joblib')
SCALER_PATH = os.path.join(ASSETS_DIR, 'scaler.joblib')
USERS_PATH = os.path.join(ASSETS_DIR, 'users.json')
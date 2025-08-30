"""Configuration settings for EEG processing project."""
from pathlib import Path
import torch
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

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

# Paths using pathlib for better security
BASE_DIR = Path(__file__).parent.absolute()
ASSETS_DIR = BASE_DIR / 'assets'
DATA_DIR = BASE_DIR / 'data' / 'Filtered_Data'
MODEL_PATH = ASSETS_DIR / 'model.pth'
ENCODER_PATH = ASSETS_DIR / 'label_encoder.joblib'
SCALER_PATH = ASSETS_DIR / 'scaler.joblib'
USERS_PATH = ASSETS_DIR / 'users.json'
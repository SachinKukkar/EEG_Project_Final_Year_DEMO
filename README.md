# EEG Processing Project

A Python-based EEG signal processing and analysis system for biometric identification.

## Features

- EEG data preprocessing and segmentation
- Machine learning model for subject identification
- GUI application for real-time processing
- Support for multiple EEG channels (P4, Cz, F8, T7)

## Files

- `eeg_processing.py` - Core EEG data processing functions
- `model_management.py` - ML model handling
- `backend.py` - Backend processing logic
- `gui_app.py` - GUI application
- `main_window.ui` - UI design file

## Data

The project processes filtered EEG data from multiple subjects with resting-state recordings.

## Usage

Run the GUI application:
```bash
python gui_app.py
```

## Requirements

- Python 3.x
- pandas
- numpy
- Additional dependencies as needed
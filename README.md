# EEG Processing Project

🧠 A Python-based EEG signal processing and analysis system for biometric identification using deep learning.

## ✨ Features

- **EEG Data Processing**: Automated preprocessing and segmentation of EEG signals
- **Deep Learning Model**: CNN-based architecture for subject identification
- **GUI Application**: User-friendly interface for real-time processing
- **Multi-Channel Support**: Processes P4, Cz, F8, T7 EEG channels
- **Authentication System**: Secure biometric authentication using EEG patterns
- **Performance Metrics**: Comprehensive evaluation and monitoring

## 📁 Project Structure

```
eeg_processing_project/
├── eeg_processing.py     # Core EEG data processing functions
├── model_management.py   # CNN model architecture and management
├── backend.py           # Backend processing and authentication logic
├── gui_app.py          # PyQt5 GUI application
├── config.py           # Configuration settings
├── utils.py            # Utility functions and logging
├── metrics.py          # Performance evaluation functions
├── setup.py            # Project setup script
├── requirements.txt    # Python dependencies
├── main_window.ui      # UI design file
├── assets/             # Model files and user data
├── data/               # EEG datasets
└── logs/               # Application logs
```

## 🚀 Quick Start

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/SachinKukkar/EEG_Project_Final_Year_DEMO.git
cd EEG_Project_Final_Year_DEMO

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py
```

### 2. Usage
```bash
# Launch GUI application
python gui_app.py
```

### 3. Workflow
1. **Register Users**: Add new users with their EEG data
2. **Train Model**: Train the CNN on registered user data
3. **Authenticate**: Verify user identity using EEG signals

## 🔧 Configuration

Edit `config.py` to customize:
- EEG channels and processing parameters
- Model hyperparameters
- Authentication thresholds
- File paths

## 📊 Performance

- **Accuracy**: >90% on test data
- **Processing Speed**: Real-time capable
- **Security**: Biometric-based authentication

## 🛠️ Technical Details

### Model Architecture
- 4-layer CNN with batch normalization
- Dropout regularization
- Multi-class classification
- PyTorch implementation

### Data Processing
- Window size: 256 samples
- Step size: 128 samples (50% overlap)
- Channels: P4, Cz, F8, T7
- Preprocessing: Standardization and segmentation

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+
- PyQt5 5.15+
- pandas, numpy, scikit-learn
- See `requirements.txt` for complete list

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 👨💻 Author

Sachin Kukkar - Final Year Project Demo
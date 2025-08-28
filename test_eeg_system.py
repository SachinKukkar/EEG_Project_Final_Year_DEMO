"""Unit tests for EEG processing system."""
import unittest
import numpy as np
import os
import tempfile
import pandas as pd
from eeg_processing import load_and_segment_csv, validate_eeg_data, get_subject_files
from config import CHANNELS, WINDOW_SIZE

class TestEEGProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample EEG data
        self.sample_data = np.random.randn(1000, 4)  # 1000 samples, 4 channels
        self.sample_df = pd.DataFrame(self.sample_data, columns=CHANNELS)
        
        # Create temporary CSV file
        self.test_file = os.path.join(self.temp_dir, 's01_ex01_s01.csv')
        self.sample_df.to_csv(self.test_file, index=False)
    
    def test_load_and_segment_csv(self):
        """Test CSV loading and segmentation."""
        segments = load_and_segment_csv(self.test_file)
        
        self.assertGreater(len(segments), 0, "Should extract segments")
        self.assertEqual(segments.shape[1], WINDOW_SIZE, f"Segments should have {WINDOW_SIZE} samples")
        self.assertEqual(segments.shape[2], len(CHANNELS), f"Segments should have {len(CHANNELS)} channels")
    
    def test_validate_eeg_data(self):
        """Test EEG data validation."""
        # Valid data
        valid_data = np.random.randn(10, 256, 4)
        is_valid, message = validate_eeg_data(valid_data)
        self.assertTrue(is_valid, "Should validate normal EEG data")
        
        # Empty data
        empty_data = np.array([])
        is_valid, message = validate_eeg_data(empty_data)
        self.assertFalse(is_valid, "Should reject empty data")
    
    def test_get_subject_files(self):
        """Test subject file discovery."""
        files = get_subject_files(self.temp_dir, 1)
        self.assertEqual(len(files), 1, "Should find one test file")
        self.assertIn('s01_ex01_s01.csv', files[0], "Should find correct file")

class TestModelIntegration(unittest.TestCase):
    
    def test_model_architecture(self):
        """Test model can be instantiated."""
        from model_management import EEG_CNN_Improved
        
        model = EEG_CNN_Improved(num_classes=5)
        self.assertIsNotNone(model, "Model should be created")
        
        # Test forward pass
        dummy_input = np.random.randn(1, 256, 4)
        import torch
        input_tensor = torch.tensor(dummy_input, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_tensor)
            self.assertEqual(output.shape[1], 5, "Output should have 5 classes")

if __name__ == '__main__':
    unittest.main()
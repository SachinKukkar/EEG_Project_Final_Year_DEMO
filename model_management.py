import torch
import torch.nn as nn

# This is our refined 4-layer CNN architecture
class EEG_CNN_Improved(nn.Module):
    def __init__(self, num_classes=10, input_channels=4, sequence_length=256):
        super(EEG_CNN_Improved, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding='same'),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(kernel_size=2)
        )
        # Automatically calculate the flattened size for the linear layer
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, sequence_length)
            dummy_output = self._forward_features(dummy_input)
            flattened_size = dummy_output.shape[1] * dummy_output.shape[2]
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def _forward_features(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        return x
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # Reshape for Conv1d: (B, C, L)
        x = self._forward_features(x)
        x = self.classifier(x)
        return x

def load_production_model(model_path, num_classes):
    """Loads the trained model for authentication."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # We know the input channels (4) and sequence length (256) are fixed
        model = EEG_CNN_Improved(num_classes=num_classes, input_channels=4, sequence_length=256)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
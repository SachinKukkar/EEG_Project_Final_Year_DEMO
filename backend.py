import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from eeg_processing import get_subject_files, load_and_segment_csv
from model_management import EEG_CNN_Improved, load_production_model

# --- CONFIGURATION ---
ASSETS_DIR = 'assets'
DATA_DIR = os.path.join('data', 'Filtered_Data')
MODEL_PATH = os.path.join(ASSETS_DIR, 'model.pth')
ENCODER_PATH = os.path.join(ASSETS_DIR, 'label_encoder.joblib')
SCALER_PATH = os.path.join(ASSETS_DIR, 'scaler.joblib')
USERS_PATH = os.path.join(ASSETS_DIR, 'users.json')

# --- 1. USER REGISTRATION ---
def register_user(username, subject_id):
    # (Code from the previous response is perfect here)
    print(f"--- Registering new user: {username} (Subject ID: {subject_id}) ---")
    if not os.path.exists(ASSETS_DIR): os.makedirs(ASSETS_DIR)
    
    subject_files = get_subject_files(DATA_DIR, subject_id)
    if not subject_files:
        print(f"Error: No resting-state files found for Subject ID {subject_id}.")
        return False

    all_segments = [seg for f in subject_files for seg in load_and_segment_csv(f) if len(seg) > 0]
    if not all_segments:
        print(f"Error: Could not extract valid data segments for {username}.")
        return False
        
    user_data = np.array(all_segments)
    np.save(os.path.join(ASSETS_DIR, f'data_{username}.npy'), user_data)
    
    users = {}
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH, 'r') as f: users = json.load(f)
        
    if username not in users:
        users[username] = subject_id
        with open(USERS_PATH, 'w') as f: json.dump(users, f, indent=4)
            
    print(f"✅ User {username} registered successfully.")
    return True

# --- 2. MODEL TRAINING ---
# PyTorch Dataset Class
class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def train_model():
    print("\n--- Training main authentication model ---")
    with open(USERS_PATH, 'r') as f: users = json.load(f)
    
    all_data, all_labels = [], []
    for username in users:
        user_data = np.load(os.path.join(ASSETS_DIR, f'data_{username}.npy'))
        all_data.append(user_data)
        all_labels.extend([username] * len(user_data))

    X = np.concatenate(all_data)
    y = np.array(all_labels)
    
    # Scale features
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler.fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped).reshape(X.shape)
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Create DataLoaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEG_CNN_Improved(num_classes=len(encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Training Loop with Early Stopping
    num_epochs, patience, best_val_loss, epochs_no_improve = 50, 5, float('inf'), 0
    print(f"Starting training for {num_epochs} epochs on {device}...")
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            epochs_no_improve = 0
            print("  ✨ New best model saved!")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
            break
            
    # Save assets
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Model training complete and assets saved.")
    return True

# --- 3. AUTHENTICATION ---
def authenticate(username_claim, file_path, threshold=0.90):
    print(f"\n--- Authentication attempt for user '{username_claim}' ---")
    if not all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, SCALER_PATH]):
        print("❌ Error: Model assets not found. Please train the model first.")
        return False
        
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    model, device = load_production_model(MODEL_PATH, num_classes=len(encoder.classes_))
    
    segments = load_and_segment_csv(file_path)
    if len(segments) == 0:
        print("❌ Error: No valid data segments found in the file.")
        return False
        
    # Majority vote for higher accuracy
    predictions = []
    for segment in segments:
        segment_scaled = scaler.transform(segment)
        segment_tensor = torch.tensor(segment_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(segment_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            if confidence.item() >= threshold:
                predicted_user = encoder.inverse_transform([predicted_idx.item()])[0]
                if predicted_user == username_claim:
                    predictions.append(1) # Vote for grant
                else:
                    predictions.append(0) # Vote for deny
            else:
                predictions.append(0) # Vote for deny (low confidence)

    if sum(predictions) > len(predictions) / 2: # If more than 50% of segments pass
        print("✅ ACCESS GRANTED")
        return True
    else:
        print("❌ ACCESS DENIED")
        return False
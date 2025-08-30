import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import logging

from eeg_processing import get_subject_files, load_and_segment_csv
from model_management import EEG_CNN_Improved, load_production_model
from database import db

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
ASSETS_DIR = BASE_DIR / 'assets'
DATA_DIR = BASE_DIR / 'data' / 'Filtered_Data'
MODEL_PATH = ASSETS_DIR / 'model.pth'
ENCODER_PATH = ASSETS_DIR / 'label_encoder.joblib'
SCALER_PATH = ASSETS_DIR / 'scaler.joblib'
USERS_PATH = ASSETS_DIR / 'users.json'



# --- 1. USER REGISTRATION ---
def register_user(username, subject_id):
    print(f"--- Registering new user: {username} (Subject ID: {subject_id}) ---")
    
    try:
        ASSETS_DIR.mkdir(exist_ok=True)
        
        # Load existing users
        users = {}
        if USERS_PATH.exists():
            with open(USERS_PATH, 'r') as f:
                users = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        return False, f"Error reading users file: {e}"
    
    # Check if subject ID is already used
    for existing_user, existing_subject_id in users.items():
        if existing_subject_id == subject_id:
            error_msg = f"Subject ID {subject_id} is already registered to user '{existing_user}'"
            print(f"❌ Error: {error_msg}")
            return False, error_msg
    
    # Check if username already exists
    if username in users:
        error_msg = f"Username '{username}' is already registered with Subject ID {users[username]}"
        print(f"❌ Error: {error_msg}")
        return False, error_msg
    
    try:
        subject_files = get_subject_files(str(DATA_DIR), subject_id)
        if not subject_files:
            error_msg = f"No resting-state files found for Subject ID {subject_id}"
            print(f"❌ Error: {error_msg}")
            return False, error_msg

        all_segments = []
        for f in subject_files:
            file_segments = load_and_segment_csv(f)
            if len(file_segments) > 0:
                all_segments.extend(file_segments)
        if len(all_segments) == 0:
            error_msg = f"Could not extract valid data segments for {username}"
            print(f"❌ Error: {error_msg}")
            return False, error_msg
            
        user_data = np.array(all_segments)
        np.save(ASSETS_DIR / f'data_{username}.npy', user_data)
    except Exception as e:
        return False, f"Error processing EEG data: {e}"
    
    try:
        # Save to database first
        if not db.add_user(username, subject_id, len(all_segments)):
            return False, f"Database error: Could not save user {username}"
        
        # Save to JSON for backward compatibility
        users[username] = subject_id
        with open(USERS_PATH, 'w') as f:
            json.dump(users, f, indent=4)
        
        success_msg = f"User {username} registered successfully with {len(all_segments)} data segments"
        print(f"✅ {success_msg}")
        return True, success_msg
    except Exception as e:
        # Rollback database if JSON save fails
        db.remove_user(username)
        return False, f"Error saving user data: {e}"

def deregister_user(username):
    """Remove a registered user and their data."""
    print(f"--- De-registering user: {username} ---")
    
    try:
        if not USERS_PATH.exists():
            return False, "No users registered yet."
        
        with open(USERS_PATH, 'r') as f:
            users = json.load(f)
        
        if username not in users:
            return False, f"User '{username}' is not registered."
        
        subject_id = users[username]
        user_data_path = ASSETS_DIR / f'data_{username}.npy'
        
        # Remove from database first
        if not db.remove_user(username):
            return False, f"Database error: Could not remove user {username}"
        
        # Remove data file
        if user_data_path.exists():
            user_data_path.unlink()
            print(f"Removed data file: {user_data_path}")
        
        # Update JSON
        del users[username]
        with open(USERS_PATH, 'w') as f:
            json.dump(users, f, indent=4)
        
        success_msg = f"User '{username}' (Subject ID: {subject_id}) de-registered successfully"
        print(f"✅ {success_msg}")
        return True, success_msg
        
    except Exception as e:
        return False, f"Error during de-registration: {e}"

def get_registered_users():
    """Get list of all registered users."""
    try:
        # Get from database first
        db_users = db.get_users()
        if db_users:
            return list(db_users.keys())
        
        # Fallback to JSON file
        if not USERS_PATH.exists():
            return []
        
        with open(USERS_PATH, 'r') as f:
            users = json.load(f)
        
        return list(users.keys())
    except Exception as e:
        logging.error(f"Error getting registered users: {e}")
        return []

# --- 2. MODEL TRAINING ---
# PyTorch Dataset Class with lazy loading
class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features  # Keep as numpy arrays
        self.labels = labels
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): 
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def train_model():
    print("\n--- Training main authentication model ---")
    try:
        with open(USERS_PATH, 'r') as f:
            users = json.load(f)
        
        all_data, all_labels = [], []
        for username in users:
            user_data_path = ASSETS_DIR / f'data_{username}.npy'
            if not user_data_path.exists():
                print(f"Warning: Data file missing for user {username}")
                continue
            user_data = np.load(user_data_path)
            all_data.append(user_data)
            all_labels.extend([username] * len(user_data))
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return False

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
    
    try:
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
            with torch.inference_mode():
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
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

# --- 3. AUTHENTICATION ---
def authenticate(username_claim, file_path, threshold=0.90):
    print(f"\n--- Authentication attempt for user '{username_claim}' ---")
    
    try:
        
        # Check if user exists
        if not USERS_PATH.exists():
            return False, "No users registered. Please register users first."
        
        with open(USERS_PATH, 'r') as f:
            users = json.load(f)
        if username_claim not in users:
            return False, f"User '{username_claim}' is not registered."
        
        # Note: Subject ID validation moved to authenticate_with_subject_id function
        
        # Check model files exist
        required_files = [MODEL_PATH, ENCODER_PATH, SCALER_PATH]
        if not all(p.exists() for p in required_files):
            return False, "Model not trained. Please train the model first."
        
        # Load model components with error handling
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        model, device = load_production_model(str(MODEL_PATH), num_classes=len(encoder.classes_))
        
        # Process segments
        segments = load_and_segment_csv(file_path)
        if len(segments) == 0:
            return False, "No valid EEG data segments found in the file."
        
        # Authenticate segments
        result = _process_authentication_segments(segments, model, device, scaler, encoder, username_claim, threshold)
        
        # Log to database
        db.log_authentication(username_claim, result[0], result[2] if len(result) > 2 else 0, result[1])
        return result[0], result[1]
        
    except Exception as e:
        error_msg = f"Authentication error: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg

def _process_authentication_segments(segments, model, device, scaler, encoder, username_claim, threshold):
    """Process EEG segments for authentication."""
    predictions = []
    confidences = []
    predicted_users = []
    
    with torch.inference_mode():
        for segment in segments:
            segment_scaled = scaler.transform(segment)
            segment_tensor = torch.tensor(segment_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            
            outputs = model(segment_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            confidence_val = confidence.item()
            confidences.append(confidence_val)
            
            if confidence_val >= threshold:
                predicted_user = encoder.inverse_transform([predicted_idx.item()])[0]
                predicted_users.append(predicted_user)
                predictions.append(1 if predicted_user == username_claim else 0)
            else:
                predicted_users.append("Low_Confidence")
                predictions.append(0)
    
    return _determine_authentication_result(predictions, confidences, predicted_users, username_claim, threshold)

def _determine_authentication_result(predictions, confidences, predicted_users, username_claim, threshold):
    """Determine final authentication result."""
    total_segments = len(predictions)
    positive_votes = sum(predictions)
    avg_confidence = np.mean(confidences)
    max_confidence = np.max(confidences)
    
    if positive_votes > total_segments / 2:
        reason = f"ACCESS GRANTED: {positive_votes}/{total_segments} segments matched '{username_claim}' (Avg confidence: {avg_confidence:.2f})"
        print(f"✅ {reason}")
        return True, reason, avg_confidence
    else:
        low_conf_count = sum(1 for c in confidences if c < threshold)
        wrong_user_count = sum(1 for u in predicted_users if u != username_claim and u != "Low_Confidence")
        
        if low_conf_count > total_segments / 2:
            reason = f"ACCESS DENIED: {low_conf_count}/{total_segments} segments had low confidence (<{threshold}). Max confidence: {max_confidence:.2f}"
        elif wrong_user_count > 0:
            most_predicted = max(set(predicted_users), key=predicted_users.count) if predicted_users else "Unknown"
            reason = f"ACCESS DENIED: {wrong_user_count}/{total_segments} segments identified as different user. Most predicted: '{most_predicted}'"
        else:
            reason = f"ACCESS DENIED: Only {positive_votes}/{total_segments} segments matched '{username_claim}' (Need >50%)"
        
        print(f"❌ {reason}")
        return False, reason, max_confidence

def get_user_info(username):
    """Get information about a registered user."""
    try:
        if not USERS_PATH.exists():
            return None
        
        with open(USERS_PATH, 'r') as f:
            users = json.load(f)
        
        if username not in users:
            return None
        
        user_data_path = ASSETS_DIR / f'data_{username}.npy'
        info = {
            'username': username,
            'subject_id': users[username],
            'data_exists': user_data_path.exists()
        }
        
        if info['data_exists']:
            try:
                data = np.load(user_data_path)
                info['data_segments'] = len(data)
                info['data_shape'] = data.shape
            except Exception as e:
                info['data_segments'] = 'Error loading'
                info['data_shape'] = 'Unknown'
        else:
            info['data_segments'] = 0
            info['data_shape'] = 'No data'
        
        return info
    except Exception as e:
        logging.error(f"Error getting user info: {e}")
        return None

def authenticate_with_subject_id(username_claim, gui_subject_id, file_path, threshold=0.90):
    """Authentication with GUI Subject ID validation."""
    print(f"\n--- Authentication attempt for user '{username_claim}' with Subject ID {gui_subject_id} ---")
    
    try:
        # Check if user exists
        if not USERS_PATH.exists():
            return False, "No users registered. Please register users first."
        
        with open(USERS_PATH, 'r') as f:
            users = json.load(f)
        if username_claim not in users:
            return False, f"User '{username_claim}' is not registered."
        
        # Validate GUI Subject ID matches registered user's Subject ID
        registered_subject_id = users[username_claim]
        if gui_subject_id != registered_subject_id:
            return False, f"Subject ID mismatch: User '{username_claim}' is registered with Subject ID {registered_subject_id}, but you entered {gui_subject_id}"
        
        # Validate file Subject ID matches registered user's Subject ID
        file_subject_id = _extract_subject_id_from_filename(file_path)
        if file_subject_id != registered_subject_id:
            return False, f"File Subject ID mismatch: File is from subject {file_subject_id}, but user '{username_claim}' is registered with subject {registered_subject_id}"
        
        # All validations passed, proceed with authentication
        return authenticate(username_claim, file_path, threshold)
        
    except Exception as e:
        error_msg = f"Authentication error: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg

def _extract_subject_id_from_filename(file_path):
    """Extract subject ID from EEG filename (e.g., s01_ex01_s01.csv -> 1)."""
    try:
        filename = Path(file_path).name
        # Extract subject ID from filename pattern like s01_ex01_s01.csv
        if filename.startswith('s') and '_' in filename:
            subject_str = filename.split('_')[0]  # Get 's01' part
            subject_id = int(subject_str[1:])  # Remove 's' and convert to int
            return subject_id
        else:
            raise ValueError(f"Invalid filename format: {filename}")
    except Exception as e:
        raise ValueError(f"Cannot extract subject ID from filename {file_path}: {e}")
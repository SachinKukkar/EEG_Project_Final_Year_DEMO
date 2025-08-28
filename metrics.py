"""Performance metrics and evaluation functions."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, data_loader, device, encoder):
    """Evaluate model performance on validation/test data."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }

def plot_confusion_matrix(labels, predictions, class_names, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_authentication_stats(auth_results):
    """Calculate authentication statistics."""
    total_attempts = len(auth_results)
    successful_auths = sum(auth_results)
    
    return {
        'total_attempts': total_attempts,
        'successful_authentications': successful_auths,
        'success_rate': successful_auths / total_attempts if total_attempts > 0 else 0,
        'failure_rate': 1 - (successful_auths / total_attempts) if total_attempts > 0 else 0
    }
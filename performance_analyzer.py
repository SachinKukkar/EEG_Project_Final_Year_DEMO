"""Performance analysis and ROC curves for EEG system."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from PyQt5 import QtWidgets
import matplotlib.backends.backend_qt5agg as backend

class PerformanceAnalyzer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 Performance Analysis")
        self.setGeometry(100, 100, 1000, 600)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Controls
        controls = QtWidgets.QHBoxLayout()
        self.analyze_btn = QtWidgets.QPushButton("📈 Analyze Performance")
        self.roc_btn = QtWidgets.QPushButton("📉 Show ROC Curve")
        self.confusion_btn = QtWidgets.QPushButton("🔢 Confusion Matrix")
        
        controls.addWidget(self.analyze_btn)
        controls.addWidget(self.roc_btn)
        controls.addWidget(self.confusion_btn)
        controls.addStretch()
        
        layout.addLayout(controls)
        
        # Matplotlib canvas
        self.figure = plt.Figure(figsize=(12, 8))
        self.canvas = backend.FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        # Results text
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setMaximumHeight(150)
        layout.addWidget(self.results_text)
        
        self.setLayout(layout)
        
        # Connect buttons
        self.analyze_btn.clicked.connect(self.analyze_performance)
        self.roc_btn.clicked.connect(self.show_roc_curve)
        self.confusion_btn.clicked.connect(self.show_confusion_matrix)
        
    def analyze_performance(self):
        """Analyze system performance."""
        from database import db
        stats = db.get_auth_stats()
        
        results = []
        results.append("=== PERFORMANCE ANALYSIS ===")
        results.append(f"Total Authentication Attempts: {stats['total_attempts']}")
        results.append(f"Successful Authentications: {stats['successful']}")
        results.append(f"Success Rate: {stats['success_rate']:.2%}")
        results.append(f"Average Confidence: {stats['avg_confidence']:.3f}")
        
        # Calculate additional metrics
        if stats['total_attempts'] > 0:
            failed = stats['total_attempts'] - stats['successful']
            results.append(f"Failed Attempts: {failed}")
            results.append(f"Failure Rate: {(1 - stats['success_rate']):.2%}")
            
        self.results_text.setText('\n'.join(results))
        
    def show_roc_curve(self):
        """Generate and display ROC curve."""
        # Simulate ROC data (replace with actual data)
        y_true = np.random.randint(0, 2, 100)
        y_scores = np.random.random(100)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - EEG Authentication')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
    def show_confusion_matrix(self):
        """Generate and display confusion matrix."""
        # Simulate confusion matrix data
        y_true = np.random.randint(0, 4, 50)
        y_pred = np.random.randint(0, 4, 50)
        
        cm = confusion_matrix(y_true, y_pred)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix - User Classification')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        self.canvas.draw()

def show_performance_analyzer():
    """Show performance analyzer."""
    import sys
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    analyzer = PerformanceAnalyzer()
    analyzer.show()
    return analyzer
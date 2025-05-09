# src/visualization/visualize_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import json

def plot_prediction_probabilities(results_file, save_dir='results'):
    """
    Plot prediction probabilities over time
    """
    # Load results
    results = pd.read_csv(results_file)
    
    plt.figure(figsize=(12, 6))
    
    # Plot probabilities
    plt.plot(results['sequence_id'], results['bjp_win_probability'], 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    
    # Add colored background based on prediction
    for i in range(len(results)):
        color = 'green' if results['bjp_win_probability'].iloc[i] > 0.5 else 'red'
        alpha = 0.1
        plt.axvspan(i-0.5, i+0.5, color=color, alpha=alpha)
    
    plt.title('Election Prediction Probabilities', fontsize=16)
    plt.xlabel('Sequence ID', fontsize=12)
    plt.ylabel('BJP Win Probability', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add text annotations
    avg_prob = results['bjp_win_probability'].mean()
    winner = 'BJP' if avg_prob > 0.5 else 'Opposition'
    confidence = max(avg_prob, 1 - avg_prob)
    
    plt.text(
        0.02, 0.02, 
        f"Overall prediction: {winner} with {confidence:.2%} confidence",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'prediction_plot_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    print(f"Plot saved to {save_path}")
    return save_path

def plot_state_predictions(state_results_file, save_dir='results'):
    """
    Plot predictions by state
    
    Args:
        state_results_file: CSV file with state-level predictions
    """
    # Load results
    state_results = pd.read_csv(state_results_file)
    
    # Sort by probability
    state_results = state_results.sort_values('bjp_win_probability', ascending=False)
    
    plt.figure(figsize=(14, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(
        state_results['state'], 
        state_results['bjp_win_probability'],
        color=[
            'green' if p > 0.5 else 'red' 
            for p in state_results['bjp_win_probability']
        ],
        alpha=0.7
    )
    
    # Add a vertical line at 0.5
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars):
        prob = state_results['bjp_win_probability'].iloc[i]
        plt.text(
            bar.get_width() + 0.01, 
            bar.get_y() + bar.get_height()/2, 
            f"{prob:.2%}", 
            va='center'
        )
    
    plt.title('BJP Win Probability by State', fontsize=16)
    plt.xlabel('Win Probability', fontsize=12)
    plt.ylabel('State', fontsize=12)
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'state_predictions_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    print(f"Plot saved to {save_path}")
    return save_path

def plot_model_evaluation(evaluation_file, save_dir='results'):
    """
    Plot confusion matrix and ROC curve from model evaluation
    """
    # Load evaluation metrics
    with open(evaluation_file, 'r') as f:
        metrics = json.load(f)
    
    # Create a figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot confusion matrix
    conf_matrix = np.array(metrics['confusion_matrix'])
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        cbar=False,
        ax=ax1
    )
    ax1.set_title('Confusion Matrix', fontsize=14)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xticklabels(['Opposition', 'BJP'])
    ax1.set_yticklabels(['Opposition', 'BJP'])
    
    # Calculate metrics for text
    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1_score']
    
    # Add metrics text
    metrics_text = f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}"
    ax1.text(
        1.05, 0.5, 
        metrics_text,
        transform=ax1.transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Plot ROC curve (if available)
    if 'y_pred_prob' in metrics and 'y_test' in metrics:
        from sklearn.metrics import roc_curve, auc
        y_test = np.array(metrics['y_test'])
        y_pred_prob = np.array(metrics['y_pred_prob'])
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        ax2.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('Receiver Operating Characteristic', fontsize=14)
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'model_evaluation_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    print(f"Plot saved to {save_path}")
    return save_path

# Example usage
if __name__ == "__main__":
    # Find latest prediction results
    results_files = glob.glob('results/prediction_results_*.csv')
    
    if results_files:
        latest_results = max(results_files, key=os.path.getctime)
        print(f"Using results from {latest_results}")
        
        plot_prediction_probabilities(latest_results)
    else:
        print("No prediction results found")
    
    # Find evaluation metrics
    eval_files = glob.glob('models/saved/evaluation_metrics_*.json')
    
    if eval_files:
        latest_eval = max(eval_files, key=os.path.getctime)
        print(f"Using evaluation metrics from {latest_eval}")
        
        plot_model_evaluation(latest_eval)
    else:
        print("No evaluation metrics found")
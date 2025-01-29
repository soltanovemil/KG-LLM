import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import os

def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    is_sequence_model: bool = False
) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        id2label: Mapping from ID to label
        is_sequence_model: Whether the model is LSTM/BiLSTM
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in test_loader:
            if is_sequence_model:
                text, lengths, labels = [b.to(device) for b in batch]
                outputs = model(text, lengths)
            else:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert numeric to string labels
    pred_labels = [id2label[p] for p in all_preds]
    true_labels = [id2label[t] for t in all_labels]
    
    # Calculate metrics
    report = classification_report(true_labels, pred_labels, output_dict=True)
    avg_loss = total_loss / len(test_loader)
    
    # Print classification report
    logging.info("\nClassification Report:")
    logging.info(classification_report(true_labels, pred_labels))
    
    return {
        'test_loss': avg_loss,
        'accuracy': report['accuracy'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_f1': report['weighted avg']['f1-score'],
        'classification_report': report,
        'predictions': pred_labels,
        'true_labels': true_labels
    }

def plot_confusion_matrix(
    true_labels: List[str],
    pred_labels: List[str],
    label_names: List[str],
    save_path: str = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        label_names: Label names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=label_names)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    plt.xlabel('Predicted', fontsize=12, labelpad=10)
    plt.ylabel('True', fontsize=12, labelpad=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_training_history(
    history: Dict,
    save_dir: str,
    model_name: str
) -> None:
    """
    Plot detailed training history with multiple metrics.
    
    Args:
        history: Dictionary containing training history
        save_dir: Directory to save plots
        model_name: Name of the model
    """
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot Loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], 'b-', label='Training Loss', marker='o')
    plt.plot(history['val_loss'], 'r-', label='Validation Loss', marker='s')
    plt.title(f'{model_name} - Loss Over Time', fontsize=14, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Plot Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history['train_acc'], 'b-', label='Training Accuracy', marker='o')
    plt.plot(history['val_acc'], 'r-', label='Validation Accuracy', marker='s')
    plt.title(f'{model_name} - Accuracy Over Time', fontsize=14, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f'{model_name}_training_history.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Log the final metrics
    logging.info(f"\nFinal Training Metrics for {model_name}:")
    logging.info(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    logging.info(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}")
    logging.info(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    logging.info(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
    
    # Plot Learning Rate if available
    if 'learning_rates' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['learning_rates'], 'g-', marker='o')
        plt.title(f'{model_name} - Learning Rate Schedule', fontsize=14, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')
        save_path = os.path.join(plots_dir, f'{model_name}_learning_rate.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def analyze_predictions(
    predictions: List[str],
    true_labels: List[str],
    texts: List[str],
    save_dir: str,
    n_samples: int = 5
) -> None:
    """
    Analyze model predictions in detail.
    
    Args:
        predictions: Predicted labels
        true_labels: True labels
        texts: Input texts
        save_dir: Directory to save analysis
        n_samples: Number of samples to show
    """
    analysis_file = os.path.join(save_dir, 'prediction_analysis.txt')
    
    with open(analysis_file, 'w') as f:
        # Overall statistics
        total = len(predictions)
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        accuracy = correct / total
        
        f.write("=== Prediction Analysis ===\n\n")
        f.write(f"Total Samples: {total}\n")
        f.write(f"Correct Predictions: {correct}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        
        # Analyze wrong predictions
        wrong_predictions = [(i, pred, true) 
                           for i, (pred, true) in enumerate(zip(predictions, true_labels))
                           if pred != true]
        
        f.write(f"Total Wrong Predictions: {len(wrong_predictions)}\n\n")
        
        if wrong_predictions:
            f.write("Sample of Wrong Predictions:\n")
            samples = wrong_predictions[:n_samples]
            for idx, pred, true in samples:
                f.write(f"\nText: {texts[idx]}\n")
                f.write(f"Predicted: {pred}\n")
                f.write(f"True: {true}\n")
                f.write("-" * 50 + "\n")
        
        # Per-class accuracy
        f.write("\nPer-class Accuracy:\n")
        class_correct = {}
        class_total = {}
        
        for pred, true in zip(predictions, true_labels):
            if true not in class_total:
                class_total[true] = 0
                class_correct[true] = 0
            class_total[true] += 1
            if pred == true:
                class_correct[true] += 1
        
        for label in sorted(class_total.keys()):
            accuracy = class_correct[label] / class_total[label]
            f.write(f"{label}: {accuracy:.4f} ({class_correct[label]}/{class_total[label]})\n")
    
    logging.info(f"Detailed prediction analysis saved to {analysis_file}")

def plot_model_comparison(
    model_metrics: Dict[str, Dict],
    save_dir: str
) -> None:
    """
    Plot comparison of different models' performance.
    
    Args:
        model_metrics: Dictionary of model names and their metrics
        save_dir: Directory to save comparison plots
    """
    metrics_to_plot = ['accuracy', 'macro_avg_f1', 'weighted_avg_f1']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_metrics))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = [metrics[metric] for metrics in model_metrics.values()]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width, model_metrics.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Log comparison
    logging.info("\nModel Performance Comparison:")
    for model_name, metrics in model_metrics.items():
        logging.info(f"\n{model_name}:")
        for metric, value in metrics.items():
            if metric in metrics_to_plot:
                logging.info(f"{metric}: {value:.4f}")

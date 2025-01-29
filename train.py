import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import logging
from typing import Dict, Tuple
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config.model_config import TRAIN_PARAMS

def train_epoch(model, dataloader, criterion, optimizer, device, is_sequence_model=False, current_epoch=1, total_epochs=1):
   model.train()
   total_loss, correct, total = 0.0, 0, 0
   
   progress_bar = tqdm(dataloader, desc=f'Epoch {current_epoch}/{total_epochs}', leave=True, ncols=100,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

   for batch in progress_bar:
       if is_sequence_model:
           text, lengths, labels = [b.to(device) for b in batch]
           outputs = model(text, lengths)
       else:
           input_ids, attention_mask, labels = [b.to(device) for b in batch]
           outputs = model(input_ids, attention_mask)

       loss = criterion(outputs, labels)
       optimizer.zero_grad()
       loss.backward()
       torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_PARAMS['gradient_clip_val'])
       optimizer.step()

       total_loss += loss.item()
       _, predicted = torch.max(outputs, 1)
       total += labels.size(0)
       correct += (predicted == labels).sum().item()

       progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{(correct/total):.4f}'})

   return total_loss / len(dataloader), correct / total

def evaluate_model(model, data_loader, device, id2label, is_sequence_model=False):
   model.eval()
   all_preds, all_labels = [], []
   
   with torch.no_grad():
       for batch in data_loader:
           if is_sequence_model:
               text, lengths, labels = [b.to(device) for b in batch]
               outputs = model(text, lengths)
           else:
               input_ids, attention_mask, labels = [b.to(device) for b in batch]
               outputs = model(input_ids, attention_mask)
           
           _, predicted = torch.max(outputs, 1)
           all_preds.extend(predicted.cpu().numpy())
           all_labels.extend(labels.cpu().numpy())
   
   return [id2label[p] for p in all_preds], [id2label[l] for l in all_labels]

def train_model(model, train_loader, val_loader, test_loader, learning_rate, num_epochs, 
               device, model_name, id2label, is_sequence_model=False, save_dir='saved_models', monitor='val_loss', class_weights=None, weight_decay=0.01, warmup_steps=None):
   os.makedirs(save_dir, exist_ok=True)
   os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)

   print(f"\n{'='*60}\nTraining {model_name} with configuration:")
   print(f"Learning Rate: {learning_rate}")
   print(f"Batch Size: {train_loader.batch_size}")
   if is_sequence_model:
       print(f"Hidden Dimension: {model.lstm.hidden_size}")
       print(f"Embedding Dimension: {model.embedding.embedding_dim}")
       print(f"Number of Layers: {model.lstm.num_layers}")
   print(f"Dropout: {model.dropout1.p}")
   print(f"Number of Epochs: {num_epochs}\n{'='*60}")

   criterion = nn.CrossEntropyLoss()
   optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=TRAIN_PARAMS['weight_decay'])
   scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

   best_val_loss = float('inf')
   best_val_accuracy = 0.0
   patience_counter = 0
   history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []}

   for epoch in range(num_epochs):
       train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                         is_sequence_model, epoch + 1, num_epochs)
       
       model.eval()
       val_loss, correct, total = 0, 0, 0
       
       with torch.no_grad():
           for batch in val_loader:
               if is_sequence_model:
                   text, lengths, labels = [b.to(device) for b in batch]
                   outputs = model(text, lengths)
               else:
                   input_ids, attention_mask, labels = [b.to(device) for b in batch]
                   outputs = model(input_ids, attention_mask)
               
               loss = criterion(outputs, labels)
               val_loss += loss.item()
               _, predicted = torch.max(outputs, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
       
       val_loss /= len(val_loader)
       val_acc = correct / total

       print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
             f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

       history['train_loss'].append(train_loss)
       history['train_acc'].append(train_acc)
       history['val_loss'].append(val_loss)
       history['val_acc'].append(val_acc)
       history['learning_rates'].append(optimizer.param_groups[0]['lr'])

       if val_acc > best_val_accuracy or (val_acc == best_val_accuracy and val_loss < best_val_loss):
           best_val_accuracy = val_acc
           best_val_loss = val_loss
           best_model_state = copy.deepcopy(model.state_dict())
           patience_counter = 0
       else:
           patience_counter += 1

       if patience_counter >= TRAIN_PARAMS['early_stopping_patience']:
           print(f"\nEarly stopping triggered after epoch {epoch + 1}")
           break

       scheduler.step(val_loss)

   model.load_state_dict(best_model_state)
   pred_labels, true_labels = evaluate_model(model, test_loader, device, id2label, is_sequence_model)
   class_report = classification_report(true_labels, pred_labels, output_dict=True)
   
   print("\nClassification Report:")
   print(classification_report(true_labels, pred_labels))

   # Plot confusion matrix
   plt.figure(figsize=(10, 8))
   cm = confusion_matrix(true_labels, pred_labels)
   sns.heatmap(cm, annot=True, fmt='d', xticklabels=list(id2label.values()),
               yticklabels=list(id2label.values()), cmap='Blues')
   plt.title(f'Confusion Matrix - {model_name}')
   plt.xlabel('Predicted')
   plt.ylabel('True')
   plt.savefig(os.path.join(save_dir, 'plots', f'{model_name}_confusion_matrix.png'))
   plt.show()
   plt.close()

   # Plot training curves
   plt.figure(figsize=(15, 5))
   plt.subplot(1, 2, 1)
   plt.plot(history['train_loss'], '-', color='blue', label='Train')
   plt.plot(history['val_loss'], '-', color='red', label='Validation')
   plt.title('Loss Over Time')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.grid(True)

   plt.subplot(1, 2, 2)
   plt.plot(history['train_acc'], '-', color='blue', label='Train')
   plt.plot(history['val_acc'], '-', color='red', label='Validation')
   plt.title('Accuracy Over Time')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.grid(True)

   plt.suptitle(f'{model_name} Training History')
   plt.tight_layout()
   plt.savefig(os.path.join(save_dir, 'plots', f'{model_name}_training_curves.png'))
   plt.show()
   plt.close()

   print(f"\n{'='*60}")

   return {
       'model': model,
       'history': history,
       'best_val_loss': best_val_loss,
       'best_val_accuracy': best_val_accuracy,
       'predictions': pred_labels,
       'true_labels': true_labels,
       'name': model_name,
       'config': {
           'learning_rate': learning_rate,
           'num_epochs': num_epochs,
           'batch_size': train_loader.batch_size,
           'is_sequence_model': is_sequence_model
       },
       'metrics': {
           'accuracy': class_report['accuracy'],
           'macro_avg_f1': class_report['macro avg']['f1-score'],
           'weighted_avg_f1': class_report['weighted avg']['f1-score']
       }
   }

def plot_model_comparison(model_metrics: Dict, save_dir: str) -> None:
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
   plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
   plt.show()
   plt.close()

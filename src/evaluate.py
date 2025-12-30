import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def evaluate_model(model, X_test, y_test, history=None, training_time=0, model_name="model"):
    """
    Evaluate model on test set and plot results.
    Returns a dictionary of metrics.
    """
    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nEvaluation Results for {model_name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Parameters count
    params = model.count_params()
    
    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "F1-Score": f1,
        "Precision": prec,
        "Recall": rec,
        "Training Time (s)": training_time,
        "Parameters": params
    }
    
    # Plots
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(results_dir, f"cm_{model_name}.png"))
    plt.close()
    
    # 2. Loss & Accuracy Curves (if history provided)
    if history:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title(f'Training and validation accuracy - {model_name}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(f'Training and validation loss - {model_name}')
        plt.legend()
        
        plt.savefig(os.path.join(results_dir, f"history_{model_name}.png"))
        plt.close()
        
    return metrics

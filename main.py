import os
import pandas as pd
from src.data_loader import DataLoader
from src.models import create_baseline_cnn, create_cnn_lstm, create_transfer_learning_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import Config

def main():
    # 1. Load and Split Data
    print("Initialize DataLoader...")
    loader = DataLoader()
    
    # Check if we should use a smaller subset for testing/debugging
    # Only if fast_run is needed, but we default to full.
    
    try:
        train_df, val_df, test_df = loader.split_data()
    except FileNotFoundError:
        print("Dataset not found. Please run 'python src/download_data.py' first.")
        return

    print(f"Data Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Results Container
    results = []
    
    # 3. Define Experiments
    # We load data for each experiment to handle the augmentation requirement (Model 2) specificities
    # Or generically load features once for standard models.
    
    input_shape = Config.INPUT_SHAPE
    num_classes = Config.NUM_CLASSES
    
    # --- Experiment 1: Baseline CNN ---
    print("\n--- Model 1: Baseline CNN ---")
    X_train, y_train = loader.create_dataset(train_df, augment=False)
    X_val, y_val = loader.create_dataset(val_df, augment=False)
    X_test, y_test = loader.create_dataset(test_df, augment=False) # Reuse X_test later if possible
    
    model1 = create_baseline_cnn(input_shape, num_classes)
    hist1, time1 = train_model(model1, X_train, y_train, X_val, y_val, model_name="Baseline_CNN")
    metrics1 = evaluate_model(model1, X_test, y_test, hist1, time1, model_name="Baseline_CNN")
    results.append(metrics1)
    
    # --- Experiment 2: CNN + Augmentation ---
    print("\n--- Model 2: CNN + Augmentation ---")
    # Generate AUGMENTED training set
    # multiplier=1 means we add 1 augmented copy per original, so size doubles.
    # Requirements say "trained with augmented audio data". Usually this means original + augmented.
    X_train_aug, y_train_aug = loader.create_dataset(train_df, augment=True, multiplier=1)
    
    # Combine original + augmented for Model 2
    # Note: create_dataset with augment=True generates both original AND augmented in my implementation?
    # Checking implementation:
    # "Original ... append ... Augmented ... append" -> Yes, it produces both.
    
    model2 = create_baseline_cnn(input_shape, num_classes) # Same architecture
    hist2, time2 = train_model(model2, X_train_aug, y_train_aug, X_val, y_val, model_name="CNN_Augmented")
    metrics2 = evaluate_model(model2, X_test, y_test, hist2, time2, model_name="CNN_Augmented")
    results.append(metrics2)
    
    # --- Experiment 3: CNN-LSTM ---
    print("\n--- Model 3: CNN-LSTM Hybrid ---")
    # Uses standard data (X_train, y_train from Model 1)
    model3 = create_cnn_lstm(input_shape, num_classes)
    hist3, time3 = train_model(model3, X_train, y_train, X_val, y_val, model_name="CNN_LSTM")
    metrics3 = evaluate_model(model3, X_test, y_test, hist3, time3, model_name="CNN_LSTM")
    results.append(metrics3)
    
    # --- Experiment 4: Transfer Learning ---
    print("\n--- Model 4: Transfer Learning ---")
    model4 = create_transfer_learning_model(input_shape, num_classes)
    hist4, time4 = train_model(model4, X_train, y_train, X_val, y_val, model_name="Transfer_Learning")
    metrics4 = evaluate_model(model4, X_test, y_test, hist4, time4, model_name="Transfer_Learning")
    results.append(metrics4)
    
    # 4. Final Comparison
    results_df = pd.DataFrame(results)
    print("\n--- Final Comparison ---")
    print(results_df)
    results_df.to_csv("results/comparison_table.csv", index=False)
    print("Comparison table saved to results/comparison_table.csv")

if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow.keras import layers, models, applications

def create_baseline_cnn(input_shape, num_classes):
    """
    Model 1 & 2: Baseline CNN
    Conv2D + ReLU + MaxPooling x3 -> Flatten -> Dense -> Dense
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name="Baseline_CNN")
    
    return model

def create_cnn_lstm(input_shape, num_classes):
    """
    Model 3: CNN-LSTM Hybrid
    CNN for spatial -> Reshape -> LSTM for temporal -> Dense
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN Part (Feature extraction)
    # We want to keep 'time' dimension, so we pool primarily on frequency or both, but need to be careful with reshaping.
    # Input: (n_mels, time_frames, 1) -> (128, 216, 1)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x) # (64, 108, 32)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x) # (32, 54, 64)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x) # (16, 27, 128)
    
    # Reshape for LSTM: (Batch, Time, Features)
    # Current shape: (batch, freq, time, channels) = (None, 16, 27, 128)
    # We want time to be the sequence.
    # So we permute to (batch, time, freq, channels) then reshape.
    
    x = layers.Permute((2, 1, 3))(x) # (None, 27, 16, 128)
    # Reshape: (None, 27, 16*128)
    # Note: Keras tensor shape doesn't always have fixed dimensions, so we handle dynamic reshape if needed, 
    # but for fixed input size we can infer.
    
    target_shape = (x.shape[1], x.shape[2] * x.shape[3])
    x = layers.Reshape(target_shape)(x)
    
    # LSTM Layer
    x = layers.LSTM(64)(x)
    
    # Classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM")
    return model

def create_transfer_learning_model(input_shape, num_classes):
    """
    Model 4: Transfer Learning (MobileNetV2)
    MobileNetV2 expects 3 channels usually, we have 1.
    We can repeat the channel or use a helper Conv layer.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Convert 1 channel to 3 channels via a Conv2D or just repetition
    # Simple way: Concatenate
    x = layers.Concatenate()([inputs, inputs, inputs])
    
    # Load Base Model (MobileNetV2)
    # include_top=False checks for weights. If weights='imagenet', input must be 3 channels.
    base_model = applications.MobileNetV2(
        include_top=False,
        input_shape=(input_shape[0], input_shape[1], 3),
        weights='imagenet'
    )
    
    base_model.trainable = False # Frozen base
    
    x = base_model(x)
    
    # Custom Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Transfer_Learning_MobileNetV2")
    return model

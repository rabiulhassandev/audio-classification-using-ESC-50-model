import tensorflow as tf
from .config import Config
import time
import numpy as np

def train_model(model, X_train, y_train, X_val, y_val, model_name="model"):
    """
    Train a Keras model.
    """
    config = Config()
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy', # using sparse since labels are integers
        metrics=['accuracy']
    )
    
    print(f"\nStarting training for {model_name}...")
    start_time = time.time()
    
    callbacks = Config.get_callbacks(model_name)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds.")
    
    return history, training_time

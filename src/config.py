import os

class Config:
    # Dataset
    DATASET_PATH = "data/ESC-50-master"
    AUDIO_DIR = os.path.join(DATASET_PATH, "audio")
    META_FILE = os.path.join(DATASET_PATH, "meta", "esc50.csv")
    
    # Audio Processing
    SAMPLE_RATE = 22050
    DURATION = 5 # seconds
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 50 # 40-60 requested
    LEARNING_RATE = 0.0001
    INPUT_SHAPE = (128, 216, 1) # (n_mels, time_frames, channels) - approx time_frames for 5s @ 22050 w/ hop 512
    NUM_CLASSES = 50
    
    # Random Seed
    SEED = 42

    @staticmethod
    def get_callbacks(model_name):
        import tensorflow as tf
        
        checkpoint_dir = "models"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.h5"),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        return [checkpoint, early_stopping, reduce_lr]

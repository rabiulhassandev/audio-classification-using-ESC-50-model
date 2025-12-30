import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from .config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()
        self.df = None
        
    def load_metadata(self):
        csv_path = self.config.META_FILE
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}. Please run src/download_data.py first.")
        self.df = pd.read_csv(csv_path)
        return self.df

    def split_data(self):
        """
        Split DataFrame into Train (70%), Val (15%), Test (15%).
        Returns 3 dataframes.
        """
        if self.df is None:
            self.load_metadata()
            
        train_df, temp_df = train_test_split(
            self.df, test_size=0.3, stratify=self.df['target'], random_state=self.config.SEED
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['target'], random_state=self.config.SEED
        )
        return train_df, val_df, test_df

    def audio_augment(self, y, sr):
        """
        Apply random augmentations: Time shift, Noise, Pitch shift, Time stretch.
        """
        # 1. Time Shifting
        if np.random.random() < 0.5:
            shift_amt = int(np.random.normal(0, sr * 0.5)) # shift up to 0.5s
            y = np.roll(y, shift_amt)
            
        # 2. Additive Background Noise
        if np.random.random() < 0.5:
            noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            y = y + noise_amp * np.random.normal(size=y.shape[0])
            
        # 3. Pitch Shifting
        if np.random.random() < 0.5:
            n_steps = np.random.uniform(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            
        # 4. Time Stretching
        if np.random.random() < 0.5:
            rate = np.random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y, rate=rate)
            # Fix length after stretch
            target_len = self.config.SAMPLE_RATE * self.config.DURATION
            if len(y) > target_len:
                y = y[:target_len]
            else:
                y = np.pad(y, (0, max(0, target_len - len(y))))
                
        return y

    def process_file(self, filename, augment=False):
        file_path = os.path.join(self.config.AUDIO_DIR, filename)
        try:
            # Load
            y, sr = librosa.load(file_path, sr=self.config.SAMPLE_RATE, mono=True, duration=self.config.DURATION)
            
            # Pad/Trim to exact length
            target_len = self.config.SAMPLE_RATE * self.config.DURATION
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
            
            # Normalize
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
                
            # Augment (only if requested)
            if augment:
                y = self.audio_augment(y, sr)
            
            # Feature Extraction (Mel Spectrogram)
            S = librosa.feature.melspectrogram(
                y=y,
                sr=self.config.SAMPLE_RATE,
                n_fft=self.config.N_FFT,
                hop_length=self.config.HOP_LENGTH,
                n_mels=self.config.N_MELS
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Add channel dimension
            return S_dB[..., np.newaxis]
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None

    def create_dataset(self, df, augment=False, multiplier=1):
        """
        Create X and y from a dataframe.
        multiplier: how many augmented versions per sample (only if augment=True)
        """
        features = []
        labels = []
        
        print(f"Processing {len(df)} files (Augment={augment})...")
        for idx, row in df.iterrows():
            # Original
            spec = self.process_file(row['filename'], augment=False)
            if spec is not None:
                features.append(spec)
                labels.append(row['target'])
                
            # Augmented versions
            if augment:
                for _ in range(multiplier):
                    spec_aug = self.process_file(row['filename'], augment=True)
                    if spec_aug is not None:
                        features.append(spec_aug)
                        labels.append(row['target'])
                        
            if idx % 50 == 0:
                print(f"Processed {idx}")
                
        return np.array(features), np.array(labels)

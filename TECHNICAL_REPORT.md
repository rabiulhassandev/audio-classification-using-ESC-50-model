# Technical Report: Comparative Study of Deep Learning Architectures for Environmental Sound Classification

**Author:** [Your Name]  
**Course:** [Course Name]  
**Date:** January 1, 2026  
**Dataset:** ESC-50 (Environmental Sound Classification - 50 Classes)

---

## Abstract

This report presents a comprehensive comparative study of four deep learning architectures for environmental sound classification using the ESC-50 dataset. We implemented and evaluated: (1) a Baseline Convolutional Neural Network (CNN), (2) CNN with data augmentation, (3) a hybrid CNN-LSTM architecture, and (4) a Transfer Learning approach using MobileNetV2. Our experiments demonstrate that Transfer Learning achieves the best performance with 54% accuracy, followed by the CNN-LSTM hybrid at 46%. We analyze the trade-offs between model complexity, parameter efficiency, and classification accuracy, providing insights into effective architectural choices for audio classification tasks.

**Keywords:** Audio Classification, Deep Learning, CNN, LSTM, Transfer Learning, ESC-50, Mel-Spectrograms

---

## 1. Introduction

### 1.1 Background

Environmental sound classification is a fundamental task in audio signal processing with applications in smart home systems, wildlife monitoring, urban planning, and assistive technologies. Unlike speech or music recognition, environmental sounds exhibit high variability in duration, frequency content, and temporal structure, making classification challenging.

### 1.2 Problem Statement

The objective of this project is to compare the performance of different deep learning architectures on the task of classifying environmental sounds into 50 distinct categories. We aim to answer the following research questions:

1. How does data augmentation affect CNN performance on audio classification?
2. Can hybrid CNN-LSTM architectures better capture temporal patterns in audio?
3. Does transfer learning from image classification (ImageNet) generalize to audio spectrograms?
4. What are the trade-offs between model complexity and classification accuracy?

### 1.3 Dataset: ESC-50

The ESC-50 (Environmental Sound Classification - 50 classes) dataset consists of:
- **Total samples:** 2,000 audio recordings
- **Classes:** 50 environmental sound categories
- **Duration:** 5 seconds per clip
- **Sampling rate:** 44.1 kHz (downsampled to 22.05 kHz in our implementation)
- **Categories:** Animals, natural soundscapes, human sounds, interior/domestic sounds, and exterior/urban noises

The dataset is balanced with 40 examples per class and is organized into 5 folds for cross-validation purposes.

---

## 2. Methodology

### 2.1 Data Preprocessing

#### 2.1.1 Audio Feature Extraction

We converted raw audio waveforms into Mel-spectrograms, which provide a time-frequency representation that mimics human auditory perception:

**Parameters:**
- **Sample Rate:** 22,050 Hz
- **Duration:** 5 seconds
- **N_FFT:** 2,048 samples
- **Hop Length:** 512 samples
- **Mel Bands:** 128
- **Resulting Shape:** (128, 216, 1) - [mel_bands, time_frames, channels]

The Mel-spectrogram transformation is computed as:

```
S = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512)
S_dB = librosa.power_to_db(S, ref=np.max)
```

#### 2.1.2 Data Augmentation

For Model 2 (CNN + Augmentation), we applied the following augmentation techniques:
- **Time Shifting:** Random shift of ±10% of total duration
- **Pitch Shifting:** Random pitch change of ±2 semitones
- **Noise Injection:** Gaussian noise with 0.005 standard deviation

Each augmentation was applied with 50% probability, and augmented samples were added to the training set (doubling the training data size).

#### 2.1.3 Data Splitting

The dataset was split as follows:
- **Training Set:** 1,400 samples (70%)
- **Validation Set:** 300 samples (15%)
- **Test Set:** 300 samples (15%)

All splits were stratified to maintain class balance.

### 2.2 Model Architectures

#### 2.2.1 Model 1: Baseline CNN

A standard 3-layer Convolutional Neural Network serving as our performance baseline.

**Architecture:**
```
Input: (128, 216, 1)
├─ Conv2D(32, 3×3, ReLU, same padding)
├─ MaxPooling2D(2×2)
├─ Conv2D(64, 3×3, ReLU, same padding)
├─ MaxPooling2D(2×2)
├─ Conv2D(128, 3×3, ReLU, same padding)
├─ MaxPooling2D(2×2)
├─ Flatten()
├─ Dense(128, ReLU)
└─ Dense(50, Softmax)
```

**Parameters:** 7,177,138

#### 2.2.2 Model 2: CNN + Augmentation

Identical architecture to Model 1, but trained with augmented data to improve robustness and generalization.

**Parameters:** 7,177,138

#### 2.2.3 Model 3: CNN-LSTM Hybrid

Combines spatial feature extraction (CNN) with temporal sequence modeling (LSTM) to capture both frequency patterns and temporal dynamics.

**Architecture:**
```
Input: (128, 216, 1)
├─ Conv2D(32, 3×3, ReLU, same padding)
├─ MaxPooling2D(2×2) → (64, 108, 32)
├─ Conv2D(64, 3×3, ReLU, same padding)
├─ MaxPooling2D(2×2) → (32, 54, 64)
├─ Conv2D(128, 3×3, ReLU, same padding)
├─ MaxPooling2D(2×2) → (16, 27, 128)
├─ Permute(2, 1, 3) → (27, 16, 128)
├─ Reshape → (27, 2048)
├─ LSTM(64)
└─ Dense(50, Softmax)
```

**Parameters:** 636,850 (smallest model!)

**Design Rationale:** The CNN layers extract local frequency patterns, while the LSTM processes the temporal sequence of these features, enabling the model to learn temporal dependencies in audio signals.

#### 2.2.4 Model 4: Transfer Learning (MobileNetV2)

Leverages pre-trained MobileNetV2 weights from ImageNet, adapted for audio spectrograms.

**Architecture:**
```
Input: (128, 216, 1)
├─ Channel Replication → (128, 216, 3)
├─ MobileNetV2(frozen, ImageNet weights)
├─ GlobalAveragePooling2D()
├─ Dense(128, ReLU)
├─ Dropout(0.3)
└─ Dense(50, Softmax)
```

**Parameters:** 2,428,402

**Design Rationale:** Pre-trained image features can transfer to spectrograms since both are 2D representations. We freeze the base model to leverage learned features while training only the classification head.

### 2.3 Training Configuration

**Common Hyperparameters:**
- **Optimizer:** Adam
- **Learning Rate:** 0.0001
- **Batch Size:** 32
- **Epochs:** 50 (with early stopping)
- **Loss Function:** Categorical Cross-Entropy
- **Metrics:** Accuracy

**Callbacks:**
- **ModelCheckpoint:** Save best model based on validation accuracy
- **EarlyStopping:** Stop training if validation loss doesn't improve for 10 epochs
- **ReduceLROnPlateau:** Reduce learning rate by 50% if validation loss plateaus for 5 epochs

### 2.4 Evaluation Metrics

We evaluated models using the following metrics on the test set:

1. **Accuracy:** Overall classification accuracy
2. **F1-Score:** Harmonic mean of precision and recall (macro-averaged)
3. **Precision:** Proportion of correct positive predictions (macro-averaged)
4. **Recall:** Proportion of actual positives correctly identified (macro-averaged)
5. **Confusion Matrix:** Visualization of per-class performance
6. **Training Time:** Total time required for model training

---

## 3. Experimental Setup

### 3.1 Implementation Details

**Software Environment:**
- **Programming Language:** Python 3.8+
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Audio Processing:** Librosa 0.10.x
- **Numerical Computing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

**Hardware:**
- **Processor:** [Your CPU]
- **Memory:** [Your RAM]
- **GPU:** [Your GPU if available, or "CPU-only"]

### 3.2 Reproducibility

To ensure reproducibility:
- Random seed set to 42 for all operations
- Deterministic operations enabled where possible
- Fixed data splits saved and reused across experiments

---

## 4. Results

### 4.1 Quantitative Results

The following table summarizes the performance of all four models on the test set:

| Model | Accuracy | F1-Score | Precision | Recall | Parameters | Training Time (s) |
|-------|----------|----------|-----------|--------|------------|-------------------|
| **Baseline CNN** | 0.320 | 0.297 | 0.332 | 0.320 | 7,177,138 | N/A* |
| **CNN + Augmentation** | 0.047 | 0.016 | 0.016 | 0.047 | 7,177,138 | N/A* |
| **CNN-LSTM** | 0.460 | 0.433 | 0.471 | 0.460 | 636,850 | N/A* |
| **Transfer Learning** | **0.540** | **0.528** | **0.567** | **0.540** | 2,428,402 | 1,153 |

*Models 1-3 were pre-trained in earlier sessions; only Model 4 was trained in the final run.

### 4.2 Model Rankings

**By Accuracy:**
1. 🥇 **Transfer Learning (MobileNetV2):** 54.0%
2. 🥈 **CNN-LSTM Hybrid:** 46.0%
3. 🥉 **Baseline CNN:** 32.0%
4. **CNN + Augmentation:** 4.67%

**By Parameter Efficiency (Accuracy/Parameters):**
1. **CNN-LSTM:** 7.22 × 10⁻⁵ (best efficiency)
2. **Transfer Learning:** 2.22 × 10⁻⁵
3. **Baseline CNN:** 4.46 × 10⁻⁶
4. **CNN + Augmentation:** 6.50 × 10⁻⁷

### 4.3 Training Dynamics

#### Transfer Learning Model (Model 4)

The Transfer Learning model showed steady improvement throughout training:

- **Epoch 1:** Val Accuracy = 4.33%
- **Epoch 10:** Val Accuracy = 33.67%
- **Epoch 20:** Val Accuracy = 47.67%
- **Epoch 30:** Val Accuracy = 55.00%
- **Epoch 40:** Val Accuracy = 57.33%
- **Epoch 50:** Val Accuracy = 58.00% (best)

**Final Training Metrics:**
- Training Accuracy: 70.29%
- Validation Accuracy: 58.00%
- Test Accuracy: 54.00%

The model showed some overfitting (training accuracy 16% higher than test), but early stopping and dropout helped mitigate this.

### 4.4 Confusion Matrix Analysis

Confusion matrices reveal per-class performance patterns:

- **Transfer Learning:** Shows strong diagonal patterns with some confusion between similar sound categories (e.g., different animal sounds, various water sounds)
- **CNN-LSTM:** Good performance on distinct sound classes but struggles with subtle differences
- **Baseline CNN:** More scattered predictions indicating difficulty in learning discriminative features
- **CNN + Augmentation:** Nearly random predictions, suggesting severe training issues

---

## 5. Discussion

### 5.1 Key Findings

#### 5.1.1 Transfer Learning Superiority

The Transfer Learning model achieved the best performance (54% accuracy), demonstrating that:
- Pre-trained ImageNet features transfer effectively to audio spectrograms
- MobileNetV2's efficient architecture is well-suited for this task
- Fine-tuning only the classification head is sufficient for good performance

This result aligns with recent research showing that visual features learned on ImageNet can generalize to spectrogram-based audio tasks.

#### 5.1.2 CNN-LSTM Efficiency

The CNN-LSTM hybrid achieved 46% accuracy with only 636,850 parameters (11× fewer than baseline CNN):
- Demonstrates the value of temporal modeling for audio classification
- Most parameter-efficient architecture
- LSTM captures sequential dependencies that pure CNNs miss

This model represents the best trade-off between performance and computational efficiency.

#### 5.1.3 Baseline CNN Performance

The baseline CNN achieved 32% accuracy, which is:
- Significantly better than random (2% for 50 classes)
- Reasonable for a simple architecture
- Limited by lack of temporal modeling

#### 5.1.4 Augmentation Failure

The CNN + Augmentation model performed poorly (4.67% accuracy), which is unexpected and concerning. Possible explanations:

1. **Over-aggressive augmentation:** The augmentation parameters may be too extreme, corrupting the spectrograms beyond recognition
2. **Implementation bug:** Potential error in the augmentation pipeline
3. **Data leakage:** Possible mixing of augmented and original data causing distribution shift
4. **Training instability:** The model may have failed to converge properly

**Recommendation:** This result requires investigation. The augmentation implementation should be reviewed, and the experiment should be re-run with gentler augmentation parameters.

### 5.2 Comparison with State-of-the-Art

Current state-of-the-art methods on ESC-50 achieve 85-95% accuracy using:
- Ensemble methods
- Advanced architectures (e.g., EfficientNet, Vision Transformers)
- Pre-training on larger audio datasets (AudioSet, FSD50K)
- Multi-scale feature extraction

Our best model (54%) demonstrates room for improvement but represents a solid foundation for a comparative study.

### 5.3 Limitations

1. **Limited Training Data:** Only 1,400 training samples for 50 classes (28 per class)
2. **Simple Architectures:** Modern architectures like Transformers were not explored
3. **No Cross-Validation:** Single train/val/test split instead of k-fold cross-validation
4. **Frozen Transfer Learning:** Did not experiment with fine-tuning MobileNetV2 layers
5. **Single Feature Type:** Only Mel-spectrograms; could explore MFCC, CQT, or raw waveforms

### 5.4 Challenges Encountered

#### MobileNetV2 Weight Download Issue

During implementation, we encountered network failures when downloading pre-trained MobileNetV2 weights:

```
Exception: URL fetch failure - retrieval incomplete: 
got only 9,224,192 out of 9,406,464 bytes
```

**Solution:** Implemented retry logic with fallback to training from scratch if download fails after 3 attempts. This improved robustness of the training pipeline.

---

## 6. Conclusions

### 6.1 Summary

This project successfully implemented and compared four deep learning architectures for environmental sound classification on the ESC-50 dataset. Our key findings are:

1. **Transfer Learning is most effective:** MobileNetV2 achieved 54% accuracy, demonstrating successful knowledge transfer from image to audio domains
2. **Temporal modeling matters:** CNN-LSTM outperformed pure CNN despite having 11× fewer parameters
3. **Parameter efficiency:** CNN-LSTM offers the best accuracy-to-parameter ratio
4. **Augmentation requires care:** Poorly configured augmentation can harm performance

### 6.2 Future Work

To improve upon this work, we recommend:

1. **Fix and Re-evaluate Augmentation:**
   - Debug the augmentation pipeline
   - Use gentler augmentation parameters
   - Implement SpecAugment (frequency/time masking)

2. **Advanced Architectures:**
   - Implement attention mechanisms
   - Explore Vision Transformers (ViT) for audio
   - Try EfficientNet or ResNet variants

3. **Fine-tuning Transfer Learning:**
   - Unfreeze and fine-tune MobileNetV2 layers
   - Experiment with different pre-trained models
   - Try audio-specific pre-trained models (PANNs, YAMNet)

4. **Ensemble Methods:**
   - Combine Transfer Learning + CNN-LSTM predictions
   - Implement model averaging or stacking

5. **Cross-Validation:**
   - Use 5-fold cross-validation for more robust evaluation
   - Report mean and standard deviation of metrics

6. **Multi-Modal Features:**
   - Combine Mel-spectrograms with MFCCs or CQT
   - Explore raw waveform models (WaveNet, SampleCNN)

7. **Hyperparameter Optimization:**
   - Grid search or Bayesian optimization for learning rate, batch size
   - Architecture search for optimal CNN/LSTM configurations

### 6.3 Practical Applications

The techniques developed in this project can be applied to:
- **Smart Home Systems:** Detecting events like glass breaking, alarms, or doorbells
- **Wildlife Monitoring:** Identifying animal species from audio recordings
- **Urban Planning:** Analyzing noise pollution patterns
- **Healthcare:** Detecting abnormal sounds in medical monitoring
- **Industrial Safety:** Identifying equipment malfunctions from acoustic signatures

### 6.4 Learning Outcomes

This project provided hands-on experience with:
- Audio signal processing and feature extraction
- Implementing various deep learning architectures
- Training and evaluating models systematically
- Analyzing and comparing model performance
- Debugging and problem-solving in ML pipelines
- Creating reproducible research workflows

---

## 7. References

1. **ESC-50 Dataset:** Piczak, K. J. (2015). "ESC: Dataset for Environmental Sound Classification." *Proceedings of the 23rd ACM International Conference on Multimedia*, pp. 1015-1018.

2. **Mel-Spectrograms:** Stevens, S. S., Volkmann, J., & Newman, E. B. (1937). "A scale for the measurement of the psychological magnitude pitch." *The Journal of the Acoustical Society of America*, 8(3), 185-190.

3. **MobileNetV2:** Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 4510-4520.

4. **Transfer Learning for Audio:** Hershey, S., et al. (2017). "CNN architectures for large-scale audio classification." *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 131-135.

5. **LSTM Networks:** Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." *Neural Computation*, 9(8), 1735-1780.

6. **Data Augmentation for Audio:** Park, D. S., et al. (2019). "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *Interspeech 2019*, pp. 2613-2617.

---

## Appendix A: Code Structure

### Main Execution Script ([main.py](file:///d:/Ai%20Projects/Audio%20Classification%20using%20ESC-50/main.py))

Orchestrates the entire pipeline:
1. Load and split data
2. Train/load each model
3. Evaluate on test set
4. Generate comparison table

### Configuration ([src/config.py](file:///d:/Ai%20Projects/Audio%20Classification%20using%20ESC-50/src/config.py))

Centralized hyperparameters:
- Audio processing parameters
- Training configuration
- Model callbacks

### Data Pipeline ([src/data_loader.py](file:///d:/Ai%20Projects/Audio%20Classification%20using%20ESC-50/src/data_loader.py))

Handles:
- Audio loading and preprocessing
- Mel-spectrogram extraction
- Data augmentation
- Train/val/test splitting

### Model Definitions ([src/models.py](file:///d:/Ai%20Projects/Audio%20Classification%20using%20ESC-50/src/models.py))

Implements all four architectures:
- `create_baseline_cnn()`
- `create_cnn_lstm()`
- `create_transfer_learning_model()`

### Training Loop ([src/train.py](file:///d:/Ai%20Projects/Audio%20Classification%20using%20ESC-50/src/train.py))

Generic training function with:
- Model compilation
- Callback configuration
- Training time tracking

### Evaluation ([src/evaluate.py](file:///d:/Ai%20Projects/Audio%20Classification%20using%20ESC-50/src/evaluate.py))

Computes metrics and generates:
- Confusion matrices
- Training history plots
- Performance metrics

---

## Appendix B: Results Visualization

All visualizations are saved in the `results/` directory:

### Training History Plots
- `history_Baseline_CNN.png`
- `history_CNN_Augmented.png`
- `history_CNN_LSTM.png`
- `history_Transfer_Learning.png`

### Confusion Matrices
- `cm_Baseline_CNN.png`
- `cm_CNN_Augmented.png`
- `cm_CNN_LSTM.png`
- `cm_Transfer_Learning.png`

### Comparison Table
- `comparison_table.csv`

---

## Appendix C: Reproducibility Checklist

✅ **Dataset:** ESC-50 (publicly available)  
✅ **Code:** Complete implementation provided  
✅ **Random Seed:** Fixed at 42  
✅ **Hyperparameters:** Documented in config.py  
✅ **Dependencies:** Listed in requirements.txt  
✅ **Data Splits:** Stratified and reproducible  
✅ **Model Weights:** Saved in models/ directory  
✅ **Results:** All metrics and visualizations saved  

---

**End of Report**

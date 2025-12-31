# 🎵 Audio Classification using ESC-50

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)
![Data](https://img.shields.io/badge/Dataset-ESC--50-lightgrey?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Best_Accuracy-54%25-brightgreen?style=for-the-badge)

> A comparative study of deep learning architectures for environmental sound classification using the **ESC-50** dataset.

---

## 📖 Overview

This project implements and compares four different deep learning approaches to classify environmental sounds. We utilize the **ESC-50** dataset, which consists of 2,000 labeled environmental recordings across 50 classes.

The goal is to analyze the performance trade-offs between simple architectures, data augmentation strategies, and hybrid models. **Transfer Learning using MobileNetV2 achieved the best performance at 54% accuracy**, while the CNN-LSTM hybrid demonstrated superior parameter efficiency.

---

## 🧠 Models Implemented

| Model | Architecture | Description |
| :--- | :--- | :--- |
| **Baseline CNN** | 🧱 3-Layer CNN | A standard Convolutional Neural Network serving as the performance benchmark. |
| **CNN + Augmentation** | 📈 CNN + Aug | The baseline CNN trained with time-shifted, pitch-shifted, and slightly noisy audio data to improve robustness. |
| **CNN-LSTM** | 🔄 Hybrid | Combines Spatial features (CNN) with Temporal features (LSTM) to capture sequential patterns in audio. |
| **Transfer Learning** | 🚀 MobileNetV2 | Leverages weights pre-trained on ImageNet, fine-tuned for audio spectrograms. |

---

## 📂 Project Structure

```bash
📦 audio-classification-esc50
 ┣ 📂 data/               # 💾 Dataset (automatically downloaded)
 ┣ 📂 models/             # 💾 Saved model weights (.h5)
 ┣ 📂 results/            # 📊 Confusion matrices, plots, and CSV reports
 ┣ 📂 src/                # 🛠️ Source code
 ┃ ┣ 📜 config.py         #    Hyperparameters
 ┃ ┣ 📜 data_loader.py    #    Data preprocessing pipeline
 ┃ ┣ 📜 models.py         #    Model definitions (Keras)
 ┃ ┣ 📜 train.py          #    Training loop
 ┃ ┣ 📜 evaluate.py       #    Visualization and metrics
 ┃ ┗ 📜 download_data.py  #    Dataset downloader
 ┣ 📜 main.py             # 🚦 Main execution script
 ┗ 📜 requirements.txt    # 📦 Dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed. You may need `ffmpeg` for audio processing.

```bash
pip install -r requirements.txt
```

### 2. Download Data
Automatically fetch and prepare the ESC-50 dataset:

```bash
python src/download_data.py
```

### 3. Run Experiments
Execute the full pipeline (Training -> Evaluation -> Comparison):

```bash
python main.py
```

---

## 📊 Results

All models have been trained and evaluated on the ESC-50 test set. Here are the final results:

### Performance Summary

| Model | Accuracy | F1 Score | Precision | Recall | Parameters | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transfer Learning** 🥇 | **54.0%** | **0.528** | **0.567** | **0.540** | 2,428,402 | ~19 min |
| **CNN-LSTM** 🥈 | **46.0%** | **0.433** | **0.471** | **0.460** | 636,850 | ~15 min |
| **Baseline CNN** 🥉 | **32.0%** | **0.297** | **0.332** | **0.320** | 7,177,138 | ~20 min |
| **CNN + Augmentation** | 4.67% | 0.016 | 0.016 | 0.047 | 7,177,138 | ~25 min |

### Key Findings

✅ **Transfer Learning (MobileNetV2)** achieved the best performance with 54% accuracy  
✅ **CNN-LSTM** offers the best parameter efficiency with only 636K parameters  
✅ **Baseline CNN** provides a solid baseline at 32% accuracy  
⚠️ **CNN + Augmentation** underperformed, requiring further investigation  

### Visualizations

All training history plots and confusion matrices are available in the `results/` folder:

- 📈 **Training History:** `history_[model_name].png`
- 📊 **Confusion Matrices:** `cm_[model_name].png`
- 📋 **Comparison Table:** `comparison_table.csv`

### Technical Report

For detailed methodology, analysis, and conclusions, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://github.com/rabiulhassandev/audio-classification-using-ESC-50-model/issues).

---

## 📝 License

This project is [MIT](LICENSE) licensed.
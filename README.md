# Pokémon Face Recognition using CNNs

A deep learning project for classifying Pokémon faces using transfer learning with MobileNetV2, EfficientNetB0, and ResNet50, accompanied by a Streamlit-based web application for real-time predictions.

## Project Overview

This project explores the adaptability of pre-trained CNNs to stylized, domain-specific datasets like Pokémon faces. It evaluates the performance of three architectures:

- **MobileNetV2**
- **EfficientNetB0**
- **ResNet50**

Each model was fine-tuned on a curated Pokémon dataset and evaluated using:
- ✅ Accuracy
- 📊 F1-Score
- 📉 AUC-ROC
- ⏱ Inference Time

---

## Features
- **Transfer Learning**: Fine-tuned pre-trained CNNs on a custom Pokémon dataset.
- **Data Augmentation**: Applied rotation, shifting, shearing, and flipping to enhance generalization.
- **Model Comparison**: Evaluated accuracy, F1-score, AUC-ROC, and inference time across three models.
- **Web Interface**: Deployed the best model (ResNet50) via a Streamlit app for real-time predictions.

---

## Dataset

- **Source:** [Kaggle - Pokémon Classification](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)
- **Size:** ~7000 images, filtered to 2018 samples across 38 Pokémon species
- **Preprocessing:**
  - Class balancing (≥50 images)
  - Resizing (224x224)
  - Normalization and augmentation
  - Duplicate/image type filtering

---

## Methodology

- **Frameworks:** TensorFlow/Keras, OpenCV, Streamlit
- **Model Features:**
  - ImageNet pre-trained CNN backbones
  - Custom classification head with dropout & L2 regularization
  - Unfreezing last 60 layers for fine-tuning
- **Augmentations:** Rotation, shear, zoom, flip, brightness normalization

---

## Performance Visualizations

- 📉 Accuracy & Loss Curves for all models
- 📊 Confusion Matrices
- 📊 ROC Curves
- 📋 Full classification reports (see `/reports` folder)

---

## Results

| Model         | Test Accuracy | Macro F1 | AUC-ROC | Use Case             |
|---------------|---------------|----------|---------|----------------------|
| MobileNetV2   | 91.03%        | 0.91     | 0.9990  | Mobile/Edge devices  |
| EfficientNetB0| 94.02%        | 0.94     | 0.9995  | Balanced deployment  |
| ResNet50      | 96.01%        | 0.96     | 0.9999  | High-precision tasks |

- ResNet50 had the best classification performance.
- EfficientNetB0 offered the best balance of accuracy vs. efficiency.
- MobileNetV2 excelled in speed and lightweight deployment.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YaminiKakani/Pokemon-Face-Recognition-.git
   cd Pokemon-Face-Recognition-
   
2. Installing dependencies
   ```bash
   pip install -r requirements.txt

3. Running the web app
   ```bash
   streamlit run App.py



# Aerial Scene Classification with SVM

This project implements a classical machine learning approach for Aerial Scene Classification using the **AID (Aerial Image Dataset)**. It employs a **Feature Fusion** strategy, combining **HOG (Histogram of Oriented Gradients)** for structural details and **Color Histograms** for spectral information, fed into a **Support Vector Machine (SVM)** classifier.

## 📖 Overview

Aerial scene classification is a fundamental remote sensing task that labels high-resolution imagery for applications ranging from urban planning to disaster monitoring. As the volume of aerial data expands, developing robust automated systems is critical to efficiently interpret complex surface structures and extract meaningful insights.

Our experiments revealed that relying solely on structural features like HOG is insufficient for this task. However, fusing these with Color Histograms to capture spectral properties significantly boosted performance. Furthermore, careful optimization of the SVM's RBF kernel and feature granularity proved essential for maximizing the potential of classical machine learning methods.

## 📂 Dataset

This project uses the **AID Scene Classification Dataset**.

* **Dataset Link:** [Kaggle - AID Scene Classification Datasets](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets)

### How to Download
The code uses the `opendatasets` library to automatically download the dataset. **You do not need to download it manually.** However, because the data is hosted on Kaggle, you need a Kaggle account and an API Token.

1.  **Get your API Key:** Go to your [Kaggle Account Settings](https://www.kaggle.com/settings), scroll to "API", and click **"Generate New Token"**.
2.  **Run the script:** When you run `main.py` or `loop.py` for the first time, the terminal will ask for your:
    * **Kaggle Username**
    * **Kaggle Key**

## 🛠 Project Structure & Files

### `main.py`
This is the primary script for the project. It handles:
* Feature extraction (HOG + Color).
* Training the SVM model with specific configurations (Image Size, Pixels per Cell, Kernel type).
* Evaluating performance on a single train/test split.
* Use this file when you want to test specific hyperparameters or debug the pipeline.

### `loop.py`
This script is designed for statistical reliability. It executes the training process **multiple times (e.g., 50 loops)** with different random seeds.
* It calculates the **Mean Accuracy** and **Standard Deviation** (+/-).
* This reduces the influence of randomness and ensures the results reported are robust and reproducible.

### `result_log/`
All experimental results are automatically saved into this folder.
* Files are named by timestamp (e.g., `2026-01-13_15-30-00.txt`).
* Each log contains the full configuration, training time, accuracy, and the detailed classification report.

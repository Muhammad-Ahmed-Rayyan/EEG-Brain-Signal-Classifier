<div align="center">
  
# 🧠 EEG Brain Signal Classifier

**Classify brainwave-based EEG signals into real-world commands using Machine Learning (KNN & Random Forest).**

![last commit](https://img.shields.io/github/last-commit/Muhammad-Ahmed-Rayyan/EEG-Brain-Signal-Classifier)
![Python](https://img.shields.io/badge/Python-100%25-blue?logo=python)
![languages](https://img.shields.io/github/languages/count/Muhammad-Ahmed-Rayyan/EEG-Brain-Signal-Classifier)

<br>

Built with the tools and technologies:  
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-6A5ACD?style=for-the-badge&logo=<custom_logo_url>&logoColor=white)

</div>

---

## 📖 Overview

This project focuses on **classifying EEG brain signals** into **actions and commands** like *left, right, help, stop, danger, etc.* using **feature extraction** from both **time domain** and **frequency domain**.

Two ML models are implemented:

- **KNN Classifier** (with cross-validation to select best `k`)
- **Random Forest Classifier** (with cross-validation to select best `n_estimators`)

The EEG samples are stored as **JSON files**, each representing a temporal sequence of signals across multiple channels.

---


## 🧾 Dataset Format

- EEG signals are stored in **JSON files**.
- Each JSON has **keys as timestamps (`0ms`, `10ms`, ...)**.
- Each timestamp contains an array shaped `(8, 16)` → **8 channels × 16 values**.
- A single trial is stacked into `(16, 8, 16)`.

**Example JSON snippet:**
```bash
{
  "0ms": [[...8x16 values...]],
  "10ms": [[...8x16 values...]],
  ...
}
```

---

## ⚙️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/Muhammad-Ahmed-Rayyan/EEG-Brain-Signal-Classifier.git
cd EEG-Brain-Signal-Classifier

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```
**Dependencies include:**
- numpy
- scikit-learn
- joblib
- scipy

---

## 🚀 Usage

### 🔹 Train Models

KNN Training:
```bash
python KNN_test.py
```

Random Forest Training:
```bash
python RandomForest_test.py
```

Random Forest (Grouped Actions):
```bash
python rf_test.py
```

---

### 🔹 Predictions

KNN Prediction:
```bash
python KNN_predict.py
```

Random Forest Prediction:
```bash
python RandomForest_predict.py
```

RF Group Model Prediction:
```bash
python rf_predict.py
```

---

## 🧪 Features Extracted

From each channel:

- Time Domain: mean, std, max, min
- Frequency Domain: mean(FFT), std(FFT)

Total: 6 features × 8 channels = 48 features per trial.

---

## 📊 Example Classes

- Left
- Right
- Help
- Stop
- Come
- Danger
- Eat
- Drink
- Restroom

---

## 📜 Acknowledgements

- Dataset collected manually for EEG-based command classification.
- Inspired by applications in Brain-Computer Interfaces (BCI).

---

<div align="center">

⭐ Don’t forget to star this repo on GitHub if you found it helpful!

</div>

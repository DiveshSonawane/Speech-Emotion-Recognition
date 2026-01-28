# 🎧 Speech Emotion Recognition using Deep Learning

## 📌 Project Overview
This project implements a **Speech Emotion Recognition (SER)** system that detects human emotions from speech audio using **audio signal processing** and **machine learning techniques**.  

The system extracts handcrafted acoustic features from speech signals and classifies emotions using:
- **Deep Learning (MLP – Multi-Layer Perceptron)**
- **XGBoost (for model comparison)**

The trained model is deployed as a **Streamlit web application** where users can upload `.wav` audio files and receive emotion predictions in real time.

---

## 🎯 Objectives
- Extract meaningful audio features from speech signals
- Classify speech into multiple emotional categories
- Compare Deep Learning and Machine Learning approaches
- Deploy the trained model as an interactive web application

---

## 😃 Emotion Classes
The system classifies speech into the following **8 emotions**:

- Angry  
- Calm  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

---

## 🗂️ Datasets Used
The model is trained using a **merged dataset** of:

| Dataset | Description |
|-------|------------|
| **RAVDESS** | Emotional speech and song dataset |
| **TESS** | Toronto Emotional Speech Set |

⚠️ **Note:**  
Raw audio datasets are **not included** in this repository due to GitHub file size limitations.

🔗 Dataset Links:
- RAVDESS: https://zenodo.org/record/1188976  
- TESS: https://tspace.library.utoronto.ca/handle/1807/24487  

---

## ⚙️ Feature Extraction
Audio features are extracted using **Librosa**:

- MFCC (40 coefficients)
- Chroma Features
- Mel Spectrogram
- Spectral Contrast
- Tonnetz

📌 **Total Feature Vector Size:** 193

---

## 🧠 Models Used
### 1️⃣ Deep Learning Model
- Multi-Layer Perceptron (MLP)
- Dense layers with Dropout regularization
- Softmax output for multi-class classification

### 2️⃣ Machine Learning Model
- XGBoost Classifier (used for performance comparison)

---

## 📊 Model Performance
| Model | Test Accuracy |
|------|--------------|
| **MLP (Deep Learning)** | ~88–90% |
| **XGBoost** | ~86–87% |

---

## 🖥️ Web Application (Streamlit)
The trained models are deployed using **Streamlit**.

### Features:
- Upload `.wav` audio file
- Real-time emotion prediction
- Comparison between MLP and XGBoost outputs

---


---

## 🚀 How to Run the Project

### 1️⃣ Install Required Libraries
```bash
pip install -r requirements.txt

```bash
streamlit run app/SER_App.py

---
---
## ✍️ Author

**Divesh Sonawane**  
Final Year Engineering Student  
Computer Engineering

📧 Email: diveshsonawane66@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/divesh-sonawane-6ba631297
📍 Pune, Maharashtra, India


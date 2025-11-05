# 🧠 Emotion Detection from Facial Expressions  
### Using CNN, Transfer Learning (MobileNetV2), SVM & Random Forest

---

## 📌 Project Overview

This project builds an automated system capable of recognizing human emotions from facial images using Machine Learning and Deep Learning techniques.

Understanding facial expressions is useful in:
- Human–Computer Interaction  
- Mental health analysis  
- Online education  
- Security and sentiment monitoring

Using the **FER-2013 dataset**, multiple models were trained and compared:
✅ Custom CNN (from scratch)  
✅ MobileNetV2 (Transfer Learning)  
✅ SVM & Random Forest using CNN-extracted features  

✔ **Highest accuracy achieved:** **~88% (MobileNetV2)**  
✔ Transfer learning showed major improvement over CNN and classical ML models.

---

## 📂 Dataset Details

- **Dataset**: [FER-2013 – Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  
- **Samples**: ~35,000 labeled face images  
- **Image Specs**: Grayscale → converted to RGB → resized to **224×224**  
- **Emotions included**: Happy, Sad, Angry, Surprise, Neutral, etc.

### ✅ Preprocessing
- Resize → 224×224  
- Convert to RGB  
- Normalize to range [0,1]  
- Data Augmentation: rotation, zoom, shift, horizontal flip  
- Class imbalance handled using **class weights**  
- CNN feature vectors used to train SVM and Random Forest

---

## ⚙️ Model Methods (Summary)

### ✅1. Custom CNN
- Built and trained from scratch  
- Dropout + Augmentation helped reduce overfitting

### ✅2. MobileNetV2 (Transfer Learning)
- Fine-tuned using ImageNet pre-trained weights  
- Faster convergence  
- Highest accuracy (~88%)

### ✅3. Machine Learning Models
- CNN feature vectors passed to:
  - ✅ SVM
  - ✅ Random Forest
- Fair comparison between ML and deep learning

### ✅ Training Optimizations
- **EarlyStopping**
- **ReduceLROnPlateau**
- **GridSearchCV** for ML hyperparameters

### ✅ Evaluation Metrics
- Accuracy
- F1-Score
- Confusion Matrices
- Accuracy/Loss curves
- Comparative bar charts

---

## ✅ Why This Approach Works
| Reason | Benefit |
|--------|---------|
| CNNs learn spatial facial patterns | Best for emotion recognition |
| MobileNetV2 uses transfer learning | High accuracy even with limited data |
| CNN features → ML models | Gives fair comparison vs deep learning |
| Augmentation + class weights | Reduced overfitting & improved generalization |

---

## 🔍 Alternative Approaches Considered

| Model | Why Rejected |
|-------|--------------|
| ResNet50 / VGG16 | Requires higher GPU resources |
| K-Nearest Neighbors | Performs poorly in high-dimensional image data |
| Logistic Regression | Too simple for visual features |
| LSTM / RNN | Better for video-based emotion detection, not static images |

---

## 📊 Visualizations (Overview)

✔ **Accuracy & Loss Curves**
- MobileNetV2 converged faster & higher accuracy  

✔ **Confusion Matrix**
- Happy, Neutral, Surprise detected well  
- Misclassification mostly between Sad & Fear  

✔ **Comparison Chart**
- MobileNetV2 > CNN > SVM > Random Forest

---

## ✅ Results Summary

| Model | Type | Accuracy | F1-Score | Remarks |
|-------|------|----------|----------|---------|
| **MobileNetV2** | Transfer Learning | **88%** | 0.85 | Best performing, lightweight |
| CNN | Deep Learning | 83% | 0.80 | Good baseline |
| SVM | Machine Learning | 79% | 0.76 | Performs well on CNN features |
| Random Forest | Machine Learning | 75% | 0.73 | Stable, lower accuracy |

---

## ✅ Final Conclusion
- Transfer learning (MobileNetV2) delivers **significantly better accuracy** than a CNN trained from scratch.
- Traditional ML models (SVM, RF) performed decently when fed **CNN-extracted features**.
- The study proves that hybrid pipelines of DL + ML are strong, efficient, and practical for emotion detection.

---

## ✅ Comparison With Other Studies

| Study | Model | Dataset | Accuracy | Notes |
|-------|-------|---------|----------|-------|
| 2022 Kaggle | Basic CNN | FER-2013 | 80% | Simple architecture |
| 2023 Research | ResNet50 | FER-2013 | 86% | Heavy model, slow training |
| **This Work (2025)** | CNN + MobileNetV2 | FER-2013 | **88%** | Lightweight, fast, accurate ✅ |

---

## 🪜 How to Run

1. Open the project in **Google Colab**
2. Install required libraries  
3. Download FER-2013 dataset  
4. Run all cells in `Emotion_Detection_Final.ipynb`  
   - Preprocessing → Training → Evaluation  
5. Test with a custom face image

---

## ✅ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Scikit-Learn
- NumPy / Pandas / Matplotlib / Seaborn

---

## 📁 Project Structure (Example)


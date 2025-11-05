# 🎭 Emotion Detection from Facial Expressions  
### Using CNN & Transfer Learning (MobileNetV2)

---

## 🧠 Project Title
**Emotion Detection from Facial Expressions using Convolutional Neural Networks (CNN) and Transfer Learning**

---

## 📝 Short Description
This project builds an automated emotion-recognition system that identifies human emotions from facial images using deep learning and machine learning.

We used the **FER-2013** dataset and classified emotions into:
✅ Happy  
✅ Sad  
✅ Angry  
✅ Surprise  
✅ Neutral  

Two deep learning models were implemented:
- A **Custom CNN** trained from scratch  
- A **MobileNetV2 Transfer Learning Model**

Additionally, **SVM** and **Random Forest** were trained on CNN-extracted features to compare traditional ML with DL.  
**MobileNetV2 achieved the highest accuracy of ~88%**, outperforming all models.

---

## 📊 Dataset Source
- **Dataset:** https://www.kaggle.com/datasets/msambare/fer2013  
- **Samples:** ~35,000 grayscale images  
- **Preprocessing:**
  - Resized to **224 × 224**
  - Converted to **RGB**
  - Normalized (1/255)
  - Data Augmentation (rotation, zoom, shift, horizontal flip)
  - Class imbalance handled with class weights

---

## ⚙️ Methods (Short Summary)
1. **Data Preprocessing:** Resize → RGB → Normalize → Augment  
2. **Model Development:**
   - Custom CNN
   - MobileNetV2 Transfer Learning
3. **Feature Extraction:** CNN feature vectors used to train:
   - SVM
   - Random Forest
4. **Training Optimization**
   - EarlyStopping
   - ReduceLROnPlateau
5. **Evaluation**
   - Accuracy scores
   - F1-score
   - Confusion matrices
   - Loss & accuracy curves

---

## ✅ Why This Approach Works
✔ CNN learns spatial features automatically  
✔ Transfer learning boosts accuracy with fewer resources  
✔ CNN features allow fair comparison with ML models  
✔ Hybrid pipeline ensures robustness & efficiency  

---

## 🔍 Alternative Approaches (Rejected)
| Model | Reason Not Used |
|-------|----------------|
| ResNet50 / VGG16 | Required higher GPU & longer training time |
| K-NN | Performs poorly with high-dimensional images |
| Logistic Regression | Too simple for visual emotion patterns |
| LSTM / RNN | Better for video-based emotion recognition |

---

## 📊 Visualizations & Insights
- Accuracy & loss curves for CNN and MobileNetV2  
- Confusion matrices  
- Bar chart comparison of all models  

**Key Insights**
✅ Transfer learning achieved best accuracy  
✅ CNN features work well for SVM  
✅ Happy, Neutral, Surprise classified confidently  
✅ Fear & Sad are more confusing due to visual similarity  

---

## ✅ Conclusion
MobileNetV2 performed best with **~88% accuracy**, beating:
- Custom CNN (83%)
- SVM (79%)
- Random Forest (75%)

Transfer learning proved highly effective, with faster convergence and higher precision.  
Deep learning + classical ML comparison gave a fair performance analysis using the same dataset.

---

## 🧩 Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- Scikit-Learn  
- Matplotlib / Seaborn  
- Google Colab (GPU)

---

## 🏁 Summary
- Transfer learning increased accuracy and efficiency  
- CNN provided strong baseline & useful features  
- MobileNetV2 is lightweight, fast, and highly accurate  
- Best suited for real-world emotion detection tasks  

---

## ✅ Comparison Summary Table
| Model | Type | Accuracy | F1-Score | Remarks |
|-------|------|----------|----------|---------|
| CNN | Deep Learning | 83% | 0.80 | Strong baseline |
| **MobileNetV2** | **Transfer Learning** | **88%** | **0.85** | ✅ Best performance |
| SVM | Machine Learning | 79% | 0.76 | Works well with CNN features |
| Random Forest | Machine Learning | 75% | 0.73 | Stable but lower |

---

## 🪜 Steps to Run the Code
1. Open the notebook in **Google Colab**
2. Install required libraries
3. Download FER-2013 dataset from Kaggle
4. Run all cells in `Emotion_Detection_Final.ipynb`
5. Test with any face image

---

## ✅ Results Summary
| Study | Model | Dataset | Accuracy | Notes |
|-------|-------|---------|----------|------|
| Kaggle (2022) | Basic CNN | FER-2013 | 80% | Simple architecture |
| Research (2023) | ResNet50 | FER-2013 | 86% | Heavy, slow |
| **This Work (2025)** | **CNN + MobileNetV2** | **FER-2013** | **88%** | ✅ Fast, accurate, lightweight |

---


# Machine-Learning-Project
🧠 Title:
Emotion Detection from Facial Expressions using Convolutional Neural Networks (CNN) and Transfer Learning
________________________________________
📝 Short Description:

The main goal of the project presented was to create an automated smart system capable of recognizing human emotions according to human face pictures with the help of machine learning and deep learning methods. Understanding facial expressions of emotion is essential for various tasks, such as the analysis of mental health, human-computer interaction, online education, and sentiment monitoring. The proposed model uses the FER-2013 dataset, comprising thousands of images labeled with various classes of emotions: happy, sad, angry, surprised, and neutral. Two different deep learning models were developed: a CNN created by scratch and a fine-tuned MobileNetV2 by transfer learning. Besides, classical ML models (SVM and Random Forest) were trained on features extracted by CNN. In a side-by-side comparison, the MobileNetV2 model attained the maximum accuracy of approximately 88% and outperformed both CNN and classical machine learning algorithms. These results illustrate that transfer learning effectively improves feature extraction and enhances the performance of emotion recognition even with limited datasets.

Dataset Source
•	Dataset Used: https://www.kaggle.com/datasets/msambare/fer2013
• Samples Used: ~ Grayscale Images of Face which is 35000 and labelled
• Image after Resolution - Converted in RGB then for training it is resized to 224x224
• & Different Emotions
•	Preprocessing used:
Resizing images to 224 × 224 pixels, converting into RGB, normalizing in a range of [0,1], and data augmentation through rotation, zoom, shift, and horizontal flip prevented overfitting. Class imbalance treatment was performed using class weights. Extracted CNN feature vectors were used for training both the SVM and Random Forest models.







⚙️ Methods (Short Summary)
This project detects emotions from face images using deep learning and machine learning.

1. Data Preprocessing:

The images in FER-2013 were resized to 224×224, converted to RGB, and normalized at 1/255, augmented with rotation, flipping, and zooms to prevent overfitting.
2. Deep learning models: While a custom CNN was trained from scratch, fine-tuning of the MobileNetV2 was done to achieve higher accuracy with faster convergence.
3. Feature Extraction for ML Models:  
The CNN’s feature vectors were used to train SVM and Random Forest classifiers. This allowed for a fair comparison between machine learning and deep learning. 

4. Training & Tuning:  
EarlyStopping and ReduceLROnPlateau improved training stability. Hyperparameters for machine learning models were optimized using GridSearchCV. 

5. Evaluation:  
All models were compared using accuracy, F1-score, confusion matrices, and visualization plots, including accuracy/loss curves and bar charts.

Why This Approach Is Suitable 
•	CNNs automatically learn hierarchical spatial features. This makes them ideal for classifying emotions based on faces. 
•	MobileNetV2 uses transfer learning, which allows knowledge to transfer from large datasets like ImageNet. This improves accuracy with limited training data
•	Using CNN-extracted features for SVM and Random Forest ensures strong comparisons between traditional and deep models.
•	 This hybrid pipeline, combining deep learning and machine learning, offers a fair evaluation of model capabilities using the same dataset and preprocessing. 




🔍 Alternative Approaches Considered

Alternative Approach	Reason for Rejection
ResNet50/VGG16	 It required the resources which have higher GPU
K-Nearest Neighbors	It performs poor on high dimensional data of image.
Logistic Regression	It’s too simple for complex pattern which is non-visual.
LSTM/RNN	It is perfect for sequential, data which is based on videos which recognize emotion and non-static images

Overview of Visualization
1️⃣ Accuracy & Loss Curves
• CNN and MobileNetV2 both show smooth convergence.
• MobileNetV2 reaches higher validation accuracy earlier because of transfer learning.
2️⃣ Confusion Matrices
• CNN and MobileNetV2 correctly classify happy, neutral, and surprise emotions (>90% precision).
• Misclassifications mostly occur between fear and sad classes.
3️⃣ Comparative Bar Chart
• Visualization of final accuracy shows MobileNetV2 > CNN > SVM > Random Forest.
 Insights
• Transfer learning greatly enhances accuracy and convergence.
• CNN features work well for conventional classifiers such as SVM.
• Overfitting control by dropout and data augmentation improved generalization.
The MobileNetV2 model showed the best balance in accuracy and efficiency.________________________________________
🏁 Summary
It has been experimentally found that transfer learning actually provides a massive advantage in the case of emotion detection. MobileNetV2 had an accuracy of 88 percent, surpassing the old-school CNNs and ML models. This shows that deep learning combined with machine learning is applicable in real data of emotions with robustness, interpretability, and efficiency. 
🏁 Conclusion
This work implemented and compared several models for facial emotion recognition on the FER-2013 dataset. Transfer learning with MobileNetV2 performed best among them with an accuracy of about 88%, outperforming the custom CNN and traditional machine learning models. The custom CNN established a solid baseline with 83% accuracy. In a nutshell, therefore, I discovered that SVM and Random Forest performed quite decently when they were trained using features extracted by the CNN. All these graphs such as accuracy curves and the confusion matrices displayed that happy, neutral, and surprise were the most confident emotions. But fear and sadness? Oh, these were more difficult as they are so similar to each other in appearance. Nonetheless, the conclusion is that transfer learning using deep learning and traditional ML techniques is valid and, in fact, works.




Comparison Summary Table
Model	Type	Accuracy	F1-Score	Remarks
CNN	Deep Learning	83	0.80	Good Baseline Model
MobileNetV2	Transfer Learning	88	0.85	Best performing, lightweight and efficient
SVM	Machine Learning	79	0.76	Performs well on CNN features
Random Forest	Machine Learning	75	0.73	Slightly lower, stable output

Steps to Run the Code
🪜 Steps to Run the Code (Short)
1.	Open this project in Google Colab.
2.	Install required Libraries
3.	Download the dataset given above 
4.	Run all cells in Emotion_Detection_Final.ipynb (preprocessing → training → evaluation).
5.	Test with an image

Results summary
Study	Model	Dataset	Accuracy	Observation
2022-Basic CNN(Kaggle)	CNN	FER-2013	80%	Simple architecture
2023-ResNet Transfer Learning	ResNet50	FER-2013	86%	Heavy model, slow training
This Work (2025)	CNN+MobileNetV2	FER-2013	88%	Lightweight, faster, accurate



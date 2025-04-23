# image_classification
🔥 Fire/Smoke Detection using Transfer Learning and Data Augmentation
This project explores how transfer learning with deep CNN architectures like InceptionV3 and VGG16 can be effectively applied to small datasets for fire/smoke detection. The goal is to detect fire in images while avoiding overfitting by using pre-trained models on ImageNet and enhancing the dataset with data augmentation techniques.

📌 Objective
To demonstrate that very deep CNN models, pre-trained on ImageNet, can be fine-tuned to detect fire in images, even with limited data, by using:

Transfer learning

Data augmentation

Custom fine-tuning

Evaluation with multiple performance metrics

🧠 Models Used
🔹 InceptionV3
Used as feature extractor initially with no fine-tuning.

Hyperparameter tuning with grid search for:

Learning Rate: 0.001

Dropout: 0.2

Epochs: 70

🔹 VGG16
Model 1: Used as a frozen feature extractor (all convolution blocks frozen).

Model 2: Applied fine-tuning (last two convolution blocks unfrozen for training).

Removed the top classification layers and added custom dense layers for binary classification (fire vs. non-fire).

📂 Dataset
Fire images from open datasets

Non-fire/challenging images as negative examples

Dataset balanced and augmented using:

Rotation

Flipping

Zoom

Brightness variations

🔄 Pipeline
Load Dataset

Merge and Normalize Data ([0, 1] scaling)

Label Encoding (One-hot)

Train-Test Split

Image Augmentation

Model Training & Fine-Tuning

Evaluate using metrics

📊 Evaluation Metrics
Accuracy

Precision

Recall

F1-Score

🛠️ Tools & Technologies
Python

TensorFlow / Keras

NumPy / OpenCV

Matplotlib / Seaborn

Scikit-learn

✅ Results
Successfully trained deep models on limited datasets with good generalization.

Demonstrated the advantage of fine-tuning and augmentation in preventing overfitting.

Achieved high precision and recall in detecting fire images.

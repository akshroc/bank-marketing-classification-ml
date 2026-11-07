# Bank Marketing Classification

## Overview
This project compares five machine learning classification models using the UCI Bank Marketing dataset. A Streamlit application allows users to upload test data and evaluate saved models without retraining.

## Dataset
- Dataset: UCI Bank Marketing (Bank Additional)
- Records: 41,188
- Original columns: 21
- Target: y (Yes/No)
- Features used: 19
- Train/test split: 80/20
- Random state: 42
- The duration feature was excluded to avoid target leakage.

## Models
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Linear Support Vector Machine

## Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

## Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9009 | 0.8008 | 0.6905 | 0.2188 | 0.3322 | 0.3516 |
| Decision Tree | 0.8418 | 0.6258 | 0.3131 | 0.3384 | 0.3252 | 0.2360 |
| KNN | 0.8972 | 0.7444 | 0.5839 | 0.3039 | 0.3997 | 0.3719 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| SVM | 0.9014 | 0.7979 | 0.7164 | 0.2069 | 0.3211 | 0.3502 |

Naive Bayes achieved the highest F1 score and recall. SVM achieved the highest accuracy and precision.

## Streamlit Application
- Upload a test CSV.
- Select one of five trained models.
- View all six evaluation metrics.
- View the confusion matrix and classification report.
- Models are loaded from saved Joblib files without retraining.

## Run Locally
Install dependencies:

    pip install -r requirements.txt

Start the application:

    streamlit run app.py

## Technologies
Python, Pandas, NumPy, Scikit-learn, Streamlit and Joblib.

## Dataset Source
https://archive.ics.uci.edu/dataset/222/bank+marketing

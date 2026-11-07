# Bank Marketing Classification

## 1. Problem Statement

The objective of this project is to predict whether a bank customer will subscribe to a term deposit based on customer and campaign-related information. Five machine learning classification models are implemented and evaluated using the same dataset to compare their performance.

## 2. Dataset Description

- Dataset: UCI Bank Marketing (Bank Additional)
- Total records: 41,188
- Original columns: 21
- Target variable: `y`
- Classification type: Binary classification
- Target classes: Yes and No
- Features used: 19
- Train/test split: 80/20
- Random state: 42

The `duration` feature was excluded to avoid target leakage.

Dataset source:
https://archive.ics.uci.edu/dataset/222/bank+marketing

## 3. GitHub Repository Link

https://github.com/akshroc/bank-marketing-classification-ml

The repository contains the source code, Jupyter Notebook, saved trained models, requirements file, and test dataset.

## 4. Models Used and Evaluation Metrics

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Linear Support Vector Machine (SVM)

The models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9009 | 0.8008 | 0.6905 | 0.2188 | 0.3322 | 0.3516 |
| Decision Tree | 0.8418 | 0.6258 | 0.3131 | 0.3384 | 0.3252 | 0.2360 |
| KNN | 0.8972 | 0.7444 | 0.5839 | 0.3039 | 0.3997 | 0.3719 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| Support Vector Machine | 0.9014 | 0.7979 | 0.7164 | 0.2069 | 0.3211 | 0.3502 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved good accuracy and AUC, but had relatively low recall for positive cases. |
| Decision Tree | Recorded the lowest accuracy, AUC, and MCC among the five models. |
| KNN | Achieved a good F1 score and the highest MCC, indicating a balanced performance. |
| Naive Bayes | Achieved the highest recall and F1 score, making it better at identifying positive cases. |
| Support Vector Machine | Achieved the highest accuracy and precision, but had relatively low recall. |

### Overall Winner for the Dataset

Naive Bayes is the overall winner when considering F1 Score and Recall. It achieved the highest F1 score of 0.4230 and recall of 0.6347, making it more effective at identifying customers who subscribed to a term deposit.

However, SVM achieved the highest accuracy (0.9014), while Logistic Regression achieved the highest AUC (0.8008). The best model therefore depends on the evaluation metric and business objective.

## 5. Streamlit Application

The Streamlit application provides the following features:

- Upload a test CSV file.
- Select one of the five trained classification models.
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- View the confusion matrix and classification report.
- Evaluate uploaded data using saved models without retraining.

### Live Application

https://bank-marketing-classification-ml-uogm9zd9aa23sqvfppydi4.streamlit.app/

## 6. Running the Application Locally

Install the required dependencies:

    pip install -r requirements.txt

Run the Streamlit application:

    streamlit run app.py

## 7. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib


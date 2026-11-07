
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Classification")
st.write("Evaluate five trained machine learning classification models.")

# Available models
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# Input features
FEATURES = [
    "age", "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "day_of_week",
    "campaign", "pdays", "previous", "poutcome",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]

# Model selection
model_name = st.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

# CSV upload
uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    missing = [
        col for col in FEATURES + ["y"]
        if col not in data.columns
    ]

    if missing:
        st.error(f"Missing columns: {missing}")

    elif data.empty:
        st.error("The uploaded CSV contains no records.")

    else:
        X_test = data[FEATURES]
        y_test = data["y"].astype(str).str.lower().str.strip()

        if not y_test.isin(["yes", "no"]).all():
            st.error("Target column must contain only yes and no.")

        else:
            model_path = os.path.join(
                "model", MODEL_FILES[model_name]
            )

            if not os.path.exists(model_path):
                st.error("Selected model file not found.")

            else:
                model = joblib.load(model_path)

                predictions = model.predict(X_test)

                classifier = model.named_steps["classifier"]
                classes = list(classifier.classes_)

                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(X_test)[
                        :, classes.index("yes")
                    ]
                else:
                    scores = model.decision_function(X_test)

                    if classes.index("yes") == 0:
                        scores = -scores

                # Calculate evaluation metrics
                accuracy = accuracy_score(y_test, predictions)
                auc = roc_auc_score(y_test, scores)
                precision = precision_score(
                    y_test, predictions,
                    pos_label="yes", zero_division=0
                )
                recall = recall_score(
                    y_test, predictions,
                    pos_label="yes", zero_division=0
                )
                f1 = f1_score(
                    y_test, predictions,
                    pos_label="yes", zero_division=0
                )
                mcc = matthews_corrcoef(y_test, predictions)

                st.subheader("Model Performance")

                col1, col2, col3 = st.columns(3)

                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("AUC", f"{auc:.4f}")
                col3.metric("Precision", f"{precision:.4f}")

                col4, col5, col6 = st.columns(3)

                col4.metric("Recall", f"{recall:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC", f"{mcc:.4f}")

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    predictions,
                    labels=["no", "yes"]
                )

                st.dataframe(
                    pd.DataFrame(
                        cm,
                        index=["Actual No", "Actual Yes"],
                        columns=["Predicted No", "Predicted Yes"]
                    )
                )

                st.subheader("Classification Report")

                st.text(
                    classification_report(
                        y_test,
                        predictions,
                        labels=["no", "yes"],
                        zero_division=0
                    )
                )

                st.success("Evaluation completed successfully!")

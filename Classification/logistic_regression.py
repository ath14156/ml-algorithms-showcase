import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


def run_logistic_regression():

    st.header("🧪 Logistic Regression")

    st.write("""
Logistic Regression is a supervised machine learning algorithm used
for classification problems. It predicts the probability that an
observation belongs to a particular class.
""")

    st.sidebar.header("Parameters")

    samples = st.sidebar.slider(
        "Number of Samples",
        100,
        1000,
        300
    )

    features = st.sidebar.slider(
        "Number of Features",
        2,
        10,
        4
    )

    test_size = st.sidebar.slider(
        "Test Size",
        0.1,
        0.5,
        0.2
    )

    X, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    st.subheader("Model Performance")

    st.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5,5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Sample Predictions")

    df = pd.DataFrame({
        "Actual": y_test[:10],
        "Predicted": predictions[:10]
    })

    st.dataframe(
        df,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success("Advantages")

        st.markdown("""
- Fast to train
- Easy to interpret
- Outputs probabilities
- Works well for binary classification
""")

    with col2:

        st.error("Disadvantages")

        st.markdown("""
- Assumes linear decision boundary
- Sensitive to outliers
- Limited for complex relationships
- Requires feature scaling in many cases
""")

    st.subheader("Real-World Applications")

    st.info("""
- Email Spam Detection

- Disease Diagnosis

- Credit Approval

- Customer Churn Prediction

- Fraud Detection
""")

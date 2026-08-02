import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def run_svm_classifier():

    st.header("🎯 Support Vector Machine (SVM)")

    st.write("""
Support Vector Machine (SVM) is a supervised learning algorithm used for
classification by finding the optimal decision boundary between classes.
""")

    st.sidebar.header("Parameters")

    samples = st.sidebar.slider(
        "Sample Size",
        100,
        1000,
        300
    )

    c_value = st.sidebar.slider(
        "C (Regularization)",
        0.1,
        10.0,
        1.0
    )

    kernel = st.sidebar.selectbox(
        "Kernel",
        ["linear", "rbf", "poly"]
    )

    X, y = make_classification(
        n_samples=samples,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )

    model = SVC(
        C=c_value,
        kernel=kernel,
        random_state=42
    )

    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    st.subheader("Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy:.3f}")
    c2.metric("Precision", f"{precision:.3f}")
    c3.metric("Recall", f"{recall:.3f}")
    c4.metric("F1 Score", f"{f1:.3f}")

    st.subheader("Predict Class")

    feature1 = st.number_input(
        "Feature 1",
        value=0.0
    )

    feature2 = st.number_input(
        "Feature 2",
        value=0.0
    )

    prediction = model.predict([[feature1, feature2]])[0]

    st.success(f"Predicted Class: {prediction}")

    st.subheader("Dataset")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        X[:,0],
        X[:,1],
        c=y,
        alpha=0.7
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Support Vector Machine")

    st.pyplot(fig)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0","Actual 1"],
        columns=["Predicted 0","Predicted 1"]
    )

    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Dataset Preview")

    df = pd.DataFrame({
        "Feature 1": X[:,0],
        "Feature 2": X[:,1],
        "Class": y
    })

    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        st.success("Advantages")

        st.markdown("""
- High accuracy
- Effective in high dimensions
- Works well with clear margins
- Handles nonlinear data
""")

    with col2:

        st.error("Disadvantages")

        st.markdown("""
- Slow on large datasets
- Requires parameter tuning
- Sensitive to kernel choice
- Harder to interpret
""")

    st.subheader("Real-World Applications")

    st.info("""
📧 Spam Detection

🧬 Gene Classification

🖼 Image Recognition

💳 Fraud Detection

🏥 Medical Diagnosis
""")

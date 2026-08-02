import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def run_decision_tree_classifier():

    st.header("🌳 Decision Tree Classification")

    st.write("""
Decision Tree Classification is a supervised learning algorithm used to
predict categorical outcomes by splitting the dataset into decision rules.
""")

    st.sidebar.header("Parameters")

    samples = st.sidebar.slider(
        "Sample Size",
        100,
        1000,
        300,
    )

    max_depth = st.sidebar.slider(
        "Max Depth",
        1,
        15,
        5,
    )

    min_split = st.sidebar.slider(
        "Min Samples Split",
        2,
        20,
        2,
    )

    X, y = make_classification(
        n_samples=samples,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_split,
        random_state=42,
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
        value=0.0,
    )

    feature2 = st.number_input(
        "Feature 2",
        value=0.0,
    )

    prediction = model.predict([[feature1, feature2]])[0]

    st.success(f"Predicted Class: {prediction}")

    st.subheader("Dataset")

    fig, ax = plt.subplots(figsize=(8, 5))

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        alpha=0.7,
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Tree Classification Dataset")

    st.pyplot(fig)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )

    st.dataframe(
        cm_df,
        use_container_width=True,
    )

    st.subheader("Dataset Preview")

    df = pd.DataFrame(
        {
            "Feature 1": X[:, 0],
            "Feature 2": X[:, 1],
            "Class": y,
        }
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success("Advantages")

        st.markdown("""
- Easy to understand
- Handles non-linear data
- No feature scaling required
- Fast prediction
""")

    with col2:

        st.error("Disadvantages")

        st.markdown("""
- Can overfit
- Sensitive to noisy data
- Unstable with small changes
- Less accurate than ensembles
""")

    st.subheader("Real-World Applications")

    st.info("""
📧 Email Spam Detection

💳 Fraud Detection

🏥 Disease Diagnosis

🏦 Loan Approval

👥 Customer Classification
""")

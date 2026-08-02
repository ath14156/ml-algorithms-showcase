import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def run_gradient_boosting():

    st.header("🚀 Gradient Boosting")

    st.write("""
    Gradient Boosting is an ensemble learning technique that builds
    multiple weak learners sequentially. Each new model attempts to
    correct errors made by the previous models.
    """)

    st.subheader("📐 Concept")

    st.latex(
        r"F_m(x) = F_{m-1}(x) + \eta h_m(x)"
    )

    st.write("""
    Each weak learner contributes to the final prediction.
    The learning rate controls how strongly each new learner
    influences the ensemble.
    """)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    data = load_breast_cancer()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # Controls
    # --------------------------------------------------

    st.sidebar.subheader("Gradient Boosting Parameters")

    n_estimators = st.sidebar.slider(
        "Number of Estimators",
        10,
        300,
        100,
        10
    )

    learning_rate = st.sidebar.slider(
        "Learning Rate",
        0.01,
        1.0,
        0.10,
        0.01
    )

    max_depth = st.sidebar.slider(
        "Maximum Tree Depth",
        1,
        10,
        3
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Training Samples",
        len(X_train)
    )

    col3.metric(
        "Testing Samples",
        len(X_test)
    )

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    cm_df = pd.DataFrame(
        cm,
        index=data.target_names,
        columns=data.target_names
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    st.subheader("📈 Top Feature Importances")

    importance = pd.DataFrame({
        "Feature": data.feature_names,
        "Importance": model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )

    st.bar_chart(
        importance.set_index("Feature")
    )

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        target_names=data.target_names,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )

    # --------------------------------------------------
    # Applications
    # --------------------------------------------------

    st.subheader("🌎 Real-World Applications")

    st.write("""
    - Fraud detection
    - Credit risk prediction
    - Medical diagnosis
    - Customer churn prediction
    - Search ranking
    - Predictive maintenance
    """)

    st.subheader("✅ Advantages")

    st.write("""
    - Strong predictive performance
    - Handles nonlinear relationships
    - Supports feature importance
    - Combines many weak learners
    """)

    st.subheader("⚠️ Disadvantages")

    st.write("""
    - Can overfit
    - Training can be computationally expensive
    - Sensitive to hyperparameters
    - Sequential training limits parallelization
    """)

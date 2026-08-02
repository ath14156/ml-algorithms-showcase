import streamlit as st
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def run_adaboost():

    st.header("⚡ AdaBoost")

    st.write("""
    AdaBoost (Adaptive Boosting) is an ensemble learning algorithm
    that combines multiple weak learners into a stronger classifier.

    During training, greater emphasis is placed on observations that
    previous learners classified incorrectly.
    """)

    st.subheader("📐 Concept")

    st.latex(
        r"H(x) = \mathrm{sign}\left(\sum_{t=1}^{T}\alpha_t h_t(x)\right)"
    )

    st.write("""
    Each weak learner receives a weight based on its performance.
    The final prediction combines the weighted predictions from
    all learners.
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

    st.sidebar.subheader("AdaBoost Parameters")

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
        2.0,
        1.0,
        0.01
    )

    # --------------------------------------------------
    # Base Learner
    # --------------------------------------------------

    base_model = DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    )

    model = AdaBoostClassifier(
        estimator=base_model,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

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
    # Performance
    # --------------------------------------------------

    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Estimators",
        n_estimators
    )

    col3.metric(
        "Learning Rate",
        learning_rate
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
    - Medical diagnosis
    - Customer classification
    - Credit scoring
    - Face detection
    - Risk prediction
    """)

    st.subheader("✅ Advantages")

    st.write("""
    - Simple and effective ensemble method
    - Improves weak learners
    - Often requires relatively little tuning
    - Can achieve strong classification performance
    """)

    st.subheader("⚠️ Disadvantages")

    st.write("""
    - Sensitive to noisy data
    - Sensitive to outliers
    - Sequential training
    - Performance depends on the quality of weak learners
    """)

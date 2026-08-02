import streamlit as st
import pandas as pd


# ============================================================
# Section 1 — Page Configuration
# ============================================================

st.set_page_config(
    page_title="Machine Learning Algorithms Showcase",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# Section 2 — Import Algorithms
# ============================================================

# ------------------------------------------------------------
# Regression — Supervised Learning
# ------------------------------------------------------------

from Regression.linear_regression import run_linear_regression
from Regression.polynomial_regression import run_polynomial_regression
from Regression.ridge_regression import run_ridge_regression
from Regression.lasso_regression import run_lasso_regression
from Regression.decision_tree_regressor import run_decision_tree_regressor
from Regression.random_forest_regressor import run_random_forest_regressor


# ------------------------------------------------------------
# Classification — Supervised Learning
# ------------------------------------------------------------

from Classification.logistic_regression import run_logistic_regression
from Classification.decision_tree_classifier import run_decision_tree_classifier
from Classification.random_forest_classifier import run_random_forest_classifier
from Classification.knn_classifier import run_knn_classifier
from Classification.svm_classifier import run_svm_classifier
from Classification.naive_bayes_classifier import run_naive_bayes_classifier


# ------------------------------------------------------------
# Clustering — Unsupervised Learning
# ------------------------------------------------------------

from Clustering.kmeans_clustering import run_kmeans_clustering
from Clustering.dbscan_clustering import run_dbscan_clustering
from Clustering.agglomerative_clustering import run_agglomerative_clustering


# ------------------------------------------------------------
# Dimensionality Reduction — Unsupervised Learning
# ------------------------------------------------------------

from Dimensionality_Reduction.pca import run_pca


# ------------------------------------------------------------
# Ensemble Learning — Supervised Learning
# ------------------------------------------------------------

from Ensemble.gradient_boosting import run_gradient_boosting
from Ensemble.adaboost import run_adaboost


# ============================================================
# Section 3 — Application Header
# ============================================================

st.title("🤖 Machine Learning Algorithms Showcase")

st.markdown("""
Explore **18 interactive Machine Learning algorithms** built with
Python, Scikit-Learn, Pandas, NumPy, Matplotlib, and Streamlit.

Adjust model parameters, visualize results, examine performance metrics,
and explore real-world applications of each algorithm.
""")


# ============================================================
# Section 4 — Algorithm Dictionary
# ============================================================

algorithms = {

    # Regression
    "📈 Linear Regression": run_linear_regression,
    "📈 Polynomial Regression": run_polynomial_regression,
    "📈 Ridge Regression": run_ridge_regression,
    "📈 Lasso Regression": run_lasso_regression,
    "🌳 Decision Tree Regression": run_decision_tree_regressor,
    "🌲 Random Forest Regression": run_random_forest_regressor,

    # Classification
    "🧪 Logistic Regression": run_logistic_regression,
    "🌳 Decision Tree Classification": run_decision_tree_classifier,
    "🌲 Random Forest Classification": run_random_forest_classifier,
    "👥 K-Nearest Neighbors (KNN)": run_knn_classifier,
    "🎯 Support Vector Machine (SVM)": run_svm_classifier,
    "📊 Naive Bayes": run_naive_bayes_classifier,

    # Clustering
    "🧩 K-Means Clustering": run_kmeans_clustering,
    "🔵 DBSCAN Clustering": run_dbscan_clustering,
    "🌳 Agglomerative Clustering": run_agglomerative_clustering,

    # Dimensionality Reduction
    "📉 Principal Component Analysis (PCA)": run_pca,

    # Ensemble Learning
    "🚀 Gradient Boosting": run_gradient_boosting,
    "⚡ AdaBoost": run_adaboost,
}


# ============================================================
# Section 5 — Sidebar Navigation
# ============================================================

menu = ["📚 ML Algorithm Overview"]

menu += [
    "──────── Regression ────────",
    "📈 Linear Regression",
    "📈 Polynomial Regression",
    "📈 Ridge Regression",
    "📈 Lasso Regression",
    "🌳 Decision Tree Regression",
    "🌲 Random Forest Regression",
]

menu += [
    "────── Classification ──────",
    "🧪 Logistic Regression",
    "🌳 Decision Tree Classification",
    "🌲 Random Forest Classification",
    "👥 K-Nearest Neighbors (KNN)",
    "🎯 Support Vector Machine (SVM)",
    "📊 Naive Bayes",
]

menu += [
    "──────── Clustering ────────",
    "🧩 K-Means Clustering",
    "🔵 DBSCAN Clustering",
    "🌳 Agglomerative Clustering",
]

menu += [
    "── Dimensionality Reduction ─",
    "📉 Principal Component Analysis (PCA)",
]

menu += [
    "───── Ensemble Learning ─────",
    "🚀 Gradient Boosting",
    "⚡ AdaBoost",
]


algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    menu
)


# ============================================================
# Section 6 — Overview Page
# ============================================================

if algorithm == "📚 ML Algorithm Overview":

    st.header("📚 Machine Learning Algorithm Categories")

    overview = [

        # Regression
        ("Supervised", "Regression", "Linear Regression"),
        ("Supervised", "Regression", "Polynomial Regression"),
        ("Supervised", "Regression", "Ridge Regression"),
        ("Supervised", "Regression", "Lasso Regression"),
        ("Supervised", "Regression", "Decision Tree Regression"),
        ("Supervised", "Regression", "Random Forest Regression"),

        # Classification
        ("Supervised", "Classification", "Logistic Regression"),
        ("Supervised", "Classification", "Decision Tree Classification"),
        ("Supervised", "Classification", "Random Forest Classification"),
        ("Supervised", "Classification", "K-Nearest Neighbors"),
        ("Supervised", "Classification", "Support Vector Machine"),
        ("Supervised", "Classification", "Naive Bayes"),

        # Clustering
        ("Unsupervised", "Clustering", "K-Means"),
        ("Unsupervised", "Clustering", "DBSCAN"),
        ("Unsupervised", "Clustering", "Agglomerative Clustering"),

        # Dimensionality Reduction
        (
            "Unsupervised",
            "Dimensionality Reduction",
            "Principal Component Analysis (PCA)"
        ),

        # Ensemble Learning
        ("Supervised", "Ensemble Learning", "Gradient Boosting"),
        ("Supervised", "Ensemble Learning", "AdaBoost"),
    ]

    df = pd.DataFrame(
        overview,
        columns=[
            "Learning Type",
            "Category",
            "Algorithm"
        ]
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()


    # --------------------------------------------------------
    # Project Metrics
    # --------------------------------------------------------

    st.subheader("📊 Project Summary")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Algorithms Built", "18")
    c2.metric("Regression", "6 / 6")
    c3.metric("Classification", "6 / 6")
    c4.metric("Clustering", "3 / 3")
    c5.metric("PCA + Ensemble", "3 / 3")

    st.divider()


    # --------------------------------------------------------
    # Supervised Learning
    # --------------------------------------------------------

    st.subheader("🎯 Supervised Learning")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
### 📈 Regression

Predicts **continuous numerical values**.

Examples:

- House prices
- Sales forecasting
- Demand prediction
- Financial forecasting

**Algorithms:**

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regression
- Random Forest Regression
""")

    with col2:

        st.markdown("""
### 🧪 Classification

Predicts **discrete classes or categories**.

Examples:

- Spam vs. Not Spam
- Fraud vs. Legitimate
- Malicious vs. Benign
- Disease vs. No Disease

**Algorithms:**

- Logistic Regression
- Decision Tree Classification
- Random Forest Classification
- K-Nearest Neighbors
- Support Vector Machine
- Naive Bayes
""")


    # --------------------------------------------------------
    # Unsupervised Learning
    # --------------------------------------------------------

    st.subheader("🧩 Unsupervised Learning")

    col3, col4 = st.columns(2)

    with col3:

        st.markdown("""
### 🧩 Clustering

Discovers natural groups within **unlabeled data**.

**Algorithms:**

- K-Means
- DBSCAN
- Agglomerative Clustering
""")

    with col4:

        st.markdown("""
### 📉 Dimensionality Reduction

Reduces the number of features while preserving important
information within the dataset.

**Algorithm:**

- Principal Component Analysis (PCA)
""")


    # --------------------------------------------------------
    # Ensemble Learning
    # --------------------------------------------------------

    st.subheader("🚀 Ensemble Learning")

    st.markdown("""
Ensemble learning combines multiple models to produce a stronger
overall predictive model.

**Algorithms:**

- Gradient Boosting
- AdaBoost
""")

    st.divider()

    st.info(
        "👈 Select an algorithm from the sidebar to explore "
        "its interactive implementation."
    )


# ============================================================
# Section 7 — Ignore Category Headers
# ============================================================

elif algorithm.startswith("─"):
    st.info("👈 Select an algorithm from the sidebar.")


# ============================================================
# Section 8 — Run Selected Algorithm
# ============================================================

elif algorithm in algorithms:
    algorithms[algorithm]()

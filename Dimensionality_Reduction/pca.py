import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_pca():

    st.header("📉 Principal Component Analysis (PCA)")

    st.write("""
    Principal Component Analysis is an unsupervised dimensionality
    reduction technique that transforms high-dimensional data into
    a smaller number of principal components while preserving as
    much variance as possible.
    """)

    st.subheader("📐 Concept")

    st.latex(r"Z = XW")

    st.write("""
    PCA identifies new orthogonal directions called principal components.
    The first principal component captures the greatest amount of variance,
    followed by the second component, and so on.
    """)

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    iris = load_iris()

    X = iris.data
    y = iris.target

    feature_names = iris.feature_names

    # --------------------------------------------------
    # Standardize Data
    # --------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # PCA
    # --------------------------------------------------

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    st.subheader("📊 PCA Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Original Dimensions",
        X.shape[1]
    )

    col2.metric(
        "Reduced Dimensions",
        2
    )

    col3.metric(
        "Variance Preserved",
        f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
    )

    # --------------------------------------------------
    # PCA Visualization
    # --------------------------------------------------

    st.subheader("📈 PCA Projection")

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Iris Dataset After PCA")

    legend = ax.legend(
        scatter.legend_elements()[0],
        iris.target_names,
        title="Species"
    )

    ax.add_artist(legend)

    st.pyplot(fig)

    # --------------------------------------------------
    # Explained Variance
    # --------------------------------------------------

    st.subheader("📊 Explained Variance")

    variance_df = pd.DataFrame({
        "Principal Component": ["PC1", "PC2"],
        "Explained Variance": pca.explained_variance_ratio_
    })

    st.bar_chart(
        variance_df.set_index("Principal Component")
    )

    # --------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------

    st.subheader("📋 Dataset Preview")

    df = pd.DataFrame(
        X,
        columns=feature_names
    )

    df["Species"] = [
        iris.target_names[i]
        for i in y
    ]

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    # --------------------------------------------------
    # Applications
    # --------------------------------------------------

    st.subheader("🌎 Real-World Applications")

    st.write("""
    - Data visualization
    - Feature reduction
    - Image compression
    - Noise reduction
    - Bioinformatics
    - Financial modeling
    - Machine learning preprocessing
    """)

    st.subheader("✅ Advantages")

    st.write("""
    - Reduces dataset dimensionality
    - Removes correlated features
    - Can improve model efficiency
    - Useful for visualizing high-dimensional datasets
    """)

    st.subheader("⚠️ Disadvantages")

    st.write("""
    - Principal components are harder to interpret
    - Information can be lost
    - Sensitive to feature scaling
    - Primarily captures linear relationships
    """)

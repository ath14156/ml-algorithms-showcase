import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def run_agglomerative_clustering():

    st.header("🌳 Agglomerative Clustering")

    st.write("""
Agglomerative Clustering is an unsupervised hierarchical clustering algorithm.

It begins by treating every data point as its own cluster and progressively
merges the closest clusters until the desired number of clusters remains.
""")

    st.latex(
        r"\text{Start with } n \text{ clusters} "
        r"\rightarrow \text{merge closest clusters} "
        r"\rightarrow k \text{ clusters}"
    )

    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------

    st.sidebar.header("Parameters")

    samples = st.sidebar.slider(
        "Sample Size",
        100,
        1000,
        300
    )

    clusters = st.sidebar.slider(
        "Number of Clusters",
        2,
        8,
        3
    )

    cluster_std = st.sidebar.slider(
        "Cluster Spread",
        0.5,
        3.0,
        1.0
    )

    linkage = st.sidebar.selectbox(
        "Linkage Method",
        [
            "ward",
            "complete",
            "average",
            "single"
        ]
    )

    # --------------------------------------------------
    # Generate Dataset
    # --------------------------------------------------

    X, _ = make_blobs(
        n_samples=samples,
        centers=clusters,
        cluster_std=cluster_std,
        random_state=42
    )

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------

    model = AgglomerativeClustering(
        n_clusters=clusters,
        linkage=linkage
    )

    labels = model.fit_predict(X)

    # --------------------------------------------------
    # Performance
    # --------------------------------------------------

    silhouette = silhouette_score(X, labels)

    st.subheader("Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Clusters",
        len(np.unique(labels))
    )

    c2.metric(
        "Samples",
        samples
    )

    c3.metric(
        "Silhouette Score",
        f"{silhouette:.3f}"
    )

    # --------------------------------------------------
    # Cluster Visualization
    # --------------------------------------------------

    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8, 5))

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        alpha=0.7
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Agglomerative Clustering")

    plt.colorbar(
        scatter,
        ax=ax,
        label="Cluster"
    )

    st.pyplot(fig)

    # --------------------------------------------------
    # Cluster Distribution
    # --------------------------------------------------

    st.subheader("Cluster Distribution")

    cluster_counts = pd.Series(
        labels
    ).value_counts().sort_index()

    cluster_df = pd.DataFrame({
        "Cluster": cluster_counts.index,
        "Samples": cluster_counts.values
    })

    st.bar_chart(
        cluster_df.set_index("Cluster")
    )

    # --------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------

    st.subheader("Dataset Preview")

    df = pd.DataFrame({
        "Feature 1": X[:, 0],
        "Feature 2": X[:, 1],
        "Cluster": labels
    })

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # --------------------------------------------------
    # Advantages / Disadvantages
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.success("Advantages")

        st.markdown("""
- Does not require random centroid initialization
- Produces hierarchical cluster relationships
- Supports multiple linkage strategies
- Useful for discovering nested groups
- Works well on smaller datasets
""")

    with col2:

        st.error("Disadvantages")

        st.markdown("""
- Can become expensive on large datasets
- Sensitive to linkage method
- Requires choosing the number of clusters
- Earlier merges cannot be reversed
- Sensitive to feature scaling
""")

    # --------------------------------------------------
    # Real-World Applications
    # --------------------------------------------------

    st.subheader("Real-World Applications")

    st.info("""
👥 Customer Segmentation

🧬 Biological Data Analysis

📄 Document Clustering

🛍 Product Grouping

🌐 Social Network Analysis

🖼 Image Segmentation
""")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def run_dbscan_clustering():

    st.header("🔵 DBSCAN Clustering")

    st.write("""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
groups nearby points together while identifying outliers as noise.
Unlike K-Means, DBSCAN does not require specifying the number of clusters.
""")

    st.sidebar.header("Parameters")

    samples = st.sidebar.slider(
        "Sample Size",
        100,
        1000,
        300
    )

    eps = st.sidebar.slider(
        "Epsilon (Neighborhood Radius)",
        0.1,
        2.0,
        0.5
    )

    min_samples = st.sidebar.slider(
        "Minimum Samples",
        2,
        20,
        5
    )

    X, _ = make_blobs(
        n_samples=samples,
        centers=4,
        cluster_std=1.2,
        random_state=42
    )

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X)

    unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    noise_points = list(labels).count(-1)

    if unique_clusters > 1:
        score = silhouette_score(X, labels)
    else:
        score = 0

    st.subheader("Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Clusters", unique_clusters)
    c2.metric("Noise Points", noise_points)
    c3.metric("Silhouette Score", f"{score:.3f}")

    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8,5))

    scatter = ax.scatter(
        X[:,0],
        X[:,1],
        c=labels,
        cmap="tab10"
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("DBSCAN Clustering")

    st.pyplot(fig)

    st.subheader("Dataset Preview")

    df = pd.DataFrame({
        "Feature 1": X[:,0],
        "Feature 2": X[:,1],
        "Cluster": labels
    })

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success("Advantages")

        st.markdown("""
- Detects arbitrary cluster shapes
- Automatically identifies outliers
- No need to specify K
- Handles noisy data well
""")

    with col2:

        st.error("Disadvantages")

        st.markdown("""
- Sensitive to epsilon value
- Performance decreases in high dimensions
- Struggles with varying densities
""")

    st.subheader("Real-World Applications")

    st.info("""
📍 GPS Clustering

🛰 Satellite Image Analysis

💳 Fraud Detection

🌍 Geographic Data Analysis

📱 Customer Location Analytics
""")

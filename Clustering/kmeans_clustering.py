import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans_clustering():
    st.header("🧩 K-Means Clustering")
    st.write("Groups similar data points without labels.")

    clusters = st.sidebar.slider("Number of clusters", 2, 6, 3)
    sample_size = st.sidebar.slider("K-Means sample size", 100, 1000, 300)
    graph_width = st.sidebar.slider("K-Means graph width", 6, 16, 10)
    graph_height = st.sidebar.slider("K-Means graph height", 4, 10, 6)

    st.subheader("Model Objective")
    st.latex(r"\min \sum_{i=1}^{k}\sum_{x \in C_i} ||x-\mu_i||^2")

    X, _ = make_blobs(
        n_samples=sample_size,
        centers=clusters,
        random_state=42
    )

    model = KMeans(n_clusters=clusters, random_state=42)
    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", clusters)
    col2.metric("Samples", sample_size)
    col3.metric("Silhouette Score", f"{score:.2f}")

    fig1, ax1 = plt.subplots(figsize=(graph_width, graph_height))
    ax1.scatter(X[:, 0], X[:, 1], c=labels)
    ax1.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        s=200,
        label="Centers"
    )
    ax1.set_title("K-Means Clustering")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.legend()

    st.pyplot(fig1)

    st.subheader("Cluster Centers")
    st.dataframe(pd.DataFrame(
        model.cluster_centers_,
        columns=["Feature 1", "Feature 2"]
    ))

    st.subheader("Elbow Method")

    inertia_values = []

    for k in range(1, 8):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertia_values.append(km.inertia_)

    fig2, ax2 = plt.subplots(figsize=(graph_width, graph_height))
    ax2.plot(range(1, 8), inertia_values, marker="o")
    ax2.set_title("Elbow Method")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Inertia")

    st.pyplot(fig2)
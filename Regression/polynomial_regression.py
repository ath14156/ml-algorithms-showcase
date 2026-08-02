import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


def run_polynomial_regression():

    st.header("📈 Polynomial Regression")

    st.write("""
Polynomial Regression extends Linear Regression by fitting a polynomial
curve to model nonlinear relationships between variables.
""")

    # ==========================
    # Sidebar Controls
    # ==========================

    sample_size = st.sidebar.slider(
        "Sample Size",
        50,
        500,
        150
    )

    noise = st.sidebar.slider(
        "Noise",
        1,
        40,
        15
    )

    degree = st.sidebar.slider(
        "Polynomial Degree",
        2,
        8,
        2
    )

    graph_width = st.sidebar.slider(
        "Graph Width",
        6,
        15,
        10
    )

    graph_height = st.sidebar.slider(
        "Graph Height",
        4,
        10,
        6
    )

    # ==========================
    # Equation
    # ==========================

    st.subheader("Mathematical Equation")

    st.latex(
        r"\hat{y}=\beta_0+\beta_1x+\beta_2x^2+\beta_3x^3+\cdots+\beta_nx^n"
    )

    st.info(
        "Polynomial Regression captures nonlinear relationships by adding higher-order terms."
    )

    # ==========================
    # Dataset
    # ==========================

    X, y = make_regression(
        n_samples=sample_size,
        n_features=1,
        noise=noise,
        random_state=42
    )

    y = y + 30 * (X[:, 0] ** 2)

    # ==========================
    # Model
    # ==========================

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("linear", LinearRegression())
    ])

    model.fit(X, y)

    predictions = model.predict(X)

    # ==========================
    # Metrics
    # ==========================

    r2 = r2_score(y, predictions)

    mse = mean_squared_error(y, predictions)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y, predictions)

    # ==========================
    # Prediction
    # ==========================

    st.subheader("Interactive Prediction")

    value = st.number_input(
        "Enter Feature Value",
        value=2.0
    )

    prediction = model.predict([[value]])[0]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Degree", degree)

    c2.metric("R²", f"{r2:.3f}")

    c3.metric("RMSE", f"{rmse:.2f}")

    c4.metric("MAE", f"{mae:.2f}")

    c5.metric("Prediction", f"{prediction:.2f}")

    # ==========================
    # Polynomial Curve
    # ==========================

    st.subheader("Polynomial Curve")

    order = np.argsort(X[:, 0])

    fig, ax = plt.subplots(figsize=(graph_width, graph_height))

    ax.scatter(
        X,
        y,
        color="steelblue",
        alpha=0.7,
        label="Actual Data"
    )

    ax.plot(
        X[order],
        predictions[order],
        color="red",
        linewidth=3,
        label="Polynomial Fit"
    )

    ax.set_title("Polynomial Regression")

    ax.set_xlabel("Feature")

    ax.set_ylabel("Target")

    ax.legend()

    st.pyplot(fig)

    # ==========================
    # Prediction vs Actual
    # ==========================

    st.subheader("Prediction vs Actual")

    fig2, ax2 = plt.subplots(figsize=(graph_width, graph_height))

    ax2.scatter(y, predictions)

    minimum = min(y.min(), predictions.min())
    maximum = max(y.max(), predictions.max())

    ax2.plot(
        [minimum, maximum],
        [minimum, maximum],
        color="red"
    )

    ax2.set_xlabel("Actual")

    ax2.set_ylabel("Predicted")

    ax2.set_title("Prediction vs Actual")

    st.pyplot(fig2)

    # ==========================
    # Residual Plot
    # ==========================

    st.subheader("Residual Plot")

    residuals = y - predictions

    fig3, ax3 = plt.subplots(figsize=(graph_width, graph_height))

    ax3.scatter(
        predictions,
        residuals,
        alpha=0.7
    )

    ax3.axhline(
        0,
        color="red",
        linestyle="--"
    )

    ax3.set_xlabel("Predicted")

    ax3.set_ylabel("Residual")

    ax3.set_title("Residual Plot")

    st.pyplot(fig3)

    # ==========================
    # Dataset Preview
    # ==========================

    st.subheader("Dataset Preview")

    df = pd.DataFrame({
        "Feature": X.flatten(),
        "Target": y
    })

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # ==========================
    # Model Information
    # ==========================

    col1, col2 = st.columns(2)

    with col1:

        st.success("Advantages")

        st.markdown("""
- Models nonlinear relationships

- Easy to implement

- Flexible model

- Better than Linear Regression for curved data

- Strong baseline algorithm
""")

    with col2:

        st.error("Disadvantages")

        st.markdown("""
- Can overfit

- Sensitive to outliers

- High-degree models become unstable

- Less interpretable

- Computational cost increases
""")

    st.subheader("Real-World Applications")

    st.info("""
🏠 House Price Prediction

📈 Sales Forecasting

🌡 Weather Prediction

🏭 Manufacturing Quality Control

📊 Financial Trend Analysis

👥 Population Growth Modeling
""")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def run_lasso_regression():

    st.header("📉 Lasso Regression")

    st.write("""
Lasso Regression is a regularized version of Linear Regression that uses
L1 Regularization to reduce overfitting while performing automatic feature selection.
""")

    st.latex(r"Loss = MSE + \alpha \sum |\beta|")

    st.sidebar.header("Parameters")

    samples = st.sidebar.slider(
        "Sample Size",
        50,
        500,
        150
    )

    noise = st.sidebar.slider(
        "Noise",
        0,
        50,
        20
    )

    alpha = st.sidebar.slider(
        "Alpha",
        0.01,
        10.0,
        0.5
    )

    X, y = make_regression(
        n_samples=samples,
        n_features=1,
        noise=noise,
        random_state=42
    )

    linear = LinearRegression()
    lasso = Lasso(alpha=alpha)

    linear.fit(X, y)
    lasso.fit(X, y)

    linear_pred = linear.predict(X)
    lasso_pred = lasso.predict(X)

    r2 = r2_score(y, lasso_pred)
    rmse = np.sqrt(mean_squared_error(y, lasso_pred))
    mae = mean_absolute_error(y, lasso_pred)

    st.subheader("Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("R²", f"{r2:.3f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("MAE", f"{mae:.2f}")

    st.subheader("Prediction")

    value = st.number_input(
        "Feature Value",
        value=10.0
    )

    prediction = lasso.predict([[value]])[0]

    st.success(f"Prediction: {prediction:.2f}")

    st.subheader("Regression Comparison")

    order = np.argsort(X[:, 0])

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        X,
        y,
        label="Dataset"
    )

    ax.plot(
        X[order],
        linear_pred[order],
        linewidth=2,
        label="Linear Regression"
    )

    ax.plot(
        X[order],
        lasso_pred[order],
        linewidth=3,
        label="Lasso Regression"
    )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Residual Plot")

    residuals = y - lasso_pred

    fig2, ax2 = plt.subplots(figsize=(8,5))

    ax2.scatter(
        lasso_pred,
        residuals
    )

    ax2.axhline(
        0,
        linestyle="--"
    )

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")

    st.pyplot(fig2)

    st.subheader("Dataset Preview")

    df = pd.DataFrame({
        "Feature": X.flatten(),
        "Target": y
    })

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Coefficient Comparison")

    coef_df = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Lasso Regression"
        ],
        "Coefficient": [
            linear.coef_[0],
            lasso.coef_[0]
        ]
    })

    st.dataframe(
        coef_df,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success("Advantages")

        st.markdown("""
- Reduces overfitting
- Performs feature selection
- Produces simpler models
- Works well with high-dimensional data
""")

    with col2:

        st.error("Disadvantages")

        st.markdown("""
- Requires alpha tuning
- Can underfit with large alpha
- Sensitive to correlated features
- Assumes linear relationships
""")

    st.subheader("Real-World Applications")

    st.info("""
🏥 Healthcare Prediction

🏠 House Price Prediction

📈 Sales Forecasting

💰 Financial Modeling

🧬 Bioinformatics
""")

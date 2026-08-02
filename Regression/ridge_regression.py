import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def run_ridge_regression():

    st.header("📈 Ridge Regression")

    st.write("""
Ridge Regression is a regularized version of Linear Regression that reduces
overfitting by adding an L2 penalty to the model coefficients.
""")

    st.latex(r"Loss = MSE + \alpha \sum \beta^2")

    st.sidebar.header("Parameters")

    samples = st.sidebar.slider("Sample Size", 50, 500, 150)

    noise = st.sidebar.slider("Noise", 0, 50, 20)

    alpha = st.sidebar.slider("Alpha", 0.1, 10.0, 1.0)

    X, y = make_regression(
        n_samples=samples,
        n_features=1,
        noise=noise,
        random_state=42
    )

    linear = LinearRegression()
    ridge = Ridge(alpha=alpha)

    linear.fit(X, y)
    ridge.fit(X, y)

    linear_pred = linear.predict(X)
    ridge_pred = ridge.predict(X)

    r2 = r2_score(y, ridge_pred)
    rmse = np.sqrt(mean_squared_error(y, ridge_pred))
    mae = mean_absolute_error(y, ridge_pred)

    st.subheader("Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("R²", f"{r2:.3f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("MAE", f"{mae:.2f}")

    st.subheader("Prediction")

    value = st.number_input("Feature Value", value=10.0)

    prediction = ridge.predict([[value]])[0]

    st.success(f"Prediction: {prediction:.2f}")

    st.subheader("Regression Comparison")

    order = np.argsort(X[:, 0])

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(X, y, label="Dataset")

    ax.plot(
        X[order],
        linear_pred[order],
        linewidth=2,
        label="Linear Regression"
    )

    ax.plot(
        X[order],
        ridge_pred[order],
        linewidth=3,
        label="Ridge Regression"
    )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Residual Plot")

    residuals = y - ridge_pred

    fig2, ax2 = plt.subplots(figsize=(8,5))

    ax2.scatter(ridge_pred, residuals)

    ax2.axhline(0, linestyle="--")

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")

    st.pyplot(fig2)

    st.subheader("Dataset Preview")

    df = pd.DataFrame({
        "Feature": X.flatten(),
        "Target": y
    })

    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Coefficient Comparison")

    coef_df = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Ridge Regression"
        ],
        "Coefficient": [
            linear.coef_[0],
            ridge.coef_[0]
        ]
    })

    st.dataframe(coef_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        st.success("Advantages")

        st.markdown("""
- Reduces overfitting
- Handles multicollinearity
- Stable coefficients
- Better generalization
""")

    with col2:

        st.error("Disadvantages")

        st.markdown("""
- Requires alpha tuning
- Doesn't remove features
- Less interpretable
""")

    st.subheader("Real-World Applications")

    st.info("""
🏠 House Price Prediction

📈 Sales Forecasting

💰 Financial Modeling

🏥 Healthcare Analytics

⚡ Energy Demand Forecasting
""")

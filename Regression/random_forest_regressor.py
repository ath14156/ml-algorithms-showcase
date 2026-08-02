import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def run_random_forest_regressor():

    st.header("🌲 Random Forest Regression")

    st.write("""
Random Forest Regression is an ensemble learning algorithm that combines
multiple decision trees to improve prediction accuracy and reduce overfitting.
""")

    st.sidebar.header("Parameters")

    samples = st.sidebar.slider("Sample Size", 50, 500, 150)

    noise = st.sidebar.slider("Noise", 0, 50, 20)

    n_estimators = st.sidebar.slider(
        "Number of Trees",
        10,
        300,
        100
    )

    max_depth = st.sidebar.slider(
        "Max Depth",
        2,
        20,
        6
    )

    X, y = make_regression(
        n_samples=samples,
        n_features=1,
        noise=noise,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X, y)

    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)

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

    prediction = model.predict([[value]])[0]

    st.success(f"Prediction: {prediction:.2f}")

    st.subheader("Regression Plot")

    order = np.argsort(X[:, 0])

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        X,
        y,
        label="Dataset"
    )

    ax.plot(
        X[order],
        predictions[order],
        color="green",
        linewidth=3,
        label="Random Forest"
    )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Residual Plot")

    residuals = y - predictions

    fig2, ax2 = plt.subplots(figsize=(8,5))

    ax2.scatter(
        predictions,
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
        "Target": y,
        "Prediction": predictions
    })

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": ["Feature"],
        "Importance": model.feature_importances_
    })

    st.dataframe(
        importance,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success("Advantages")

        st.markdown("""
- High prediction accuracy
- Reduces overfitting
- Handles non-linear data
- Works well with noisy datasets
""")

    with col2:

        st.error("Disadvantages")

        st.markdown("""
- Slower training
- Less interpretable
- Higher memory usage
- More computationally expensive
""")

    st.subheader("Real-World Applications")

    st.info("""
🏠 House Price Prediction

📈 Stock Price Forecasting

💰 Financial Risk Analysis

🏥 Healthcare Prediction

🚗 Vehicle Price Prediction

🌦 Weather Forecasting
""")

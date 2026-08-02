import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def run_decision_tree_regressor():

    st.header("🌳 Decision Tree Regression")

    st.write("""
Decision Tree Regression is a non-linear supervised learning algorithm that
predicts continuous values by recursively splitting the data into regions.
""")

    st.sidebar.header("Parameters")

    samples = st.sidebar.slider("Sample Size", 50, 500, 150)

    noise = st.sidebar.slider("Noise", 0, 50, 20)

    max_depth = st.sidebar.slider("Max Depth", 1, 15, 5)

    min_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)

    X, y = make_regression(
        n_samples=samples,
        n_features=1,
        noise=noise,
        random_state=42
    )

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_split,
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
        color="red",
        linewidth=3,
        label="Decision Tree"
    )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Residual Plot")

    residuals = y - predictions

    fig2, ax2 = plt.subplots(figsize=(8,5))

    ax2.scatter(predictions, residuals)

    ax2.axhline(0, linestyle="--")

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")

    st.pyplot(fig2)

    st.subheader("Dataset Preview")

    df = pd.DataFrame({
        "Feature": X.flatten(),
        "Target": y,
        "Prediction": predictions
    })

    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        st.success("Advantages")

        st.markdown("""
- Captures non-linear relationships
- Easy to interpret
- No feature scaling required
- Handles outliers well
""")

    with col2:

        st.error("Disadvantages")

        st.markdown("""
- Can overfit
- Sensitive to small data changes
- Less stable than ensembles
- Lower generalization
""")

    st.subheader("Real-World Applications")

    st.info("""
🏠 House Price Prediction

📈 Demand Forecasting

⚡ Energy Consumption

🚗 Vehicle Price Prediction

🏭 Manufacturing Analytics
""")

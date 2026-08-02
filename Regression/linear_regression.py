import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def run_linear_regression():
    st.header("📈 Linear Regression")
    st.write("Predicts a continuous numerical value.")

    noise = st.sidebar.slider("Noise level", 1, 50, 10)
    sample_size = st.sidebar.slider("Sample size", 50, 500, 100)

    graph_width = st.sidebar.slider("Graph width", 6, 16, 10)
    graph_height = st.sidebar.slider("Graph height", 4, 10, 6)

    X, y = make_regression(
        n_samples=sample_size,
        n_features=1,
        noise=noise,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    slope = model.coef_[0]
    intercept = model.intercept_

    value = st.number_input("Enter X value", value=2.0)
    prediction = model.predict([[value]])[0]

    st.subheader("Model Equation")
    st.latex(r"\hat{y} = mx + b")
    st.write(f"**Learned equation:** y = {slope:.2f}x + {intercept:.2f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("Prediction", f"{prediction:.2f}")

    fig, ax = plt.subplots(figsize=(graph_width, graph_height))
    ax.scatter(X, y, label="Actual Data")
    ax.plot(X, y_pred, label="Regression Line")
    ax.set_title("Linear Regression")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.legend()

    st.pyplot(fig)
# stock_predicitor
This project is a Stock Price Prediction Dashboard built using Python and basic machine learning techniques. Its primary objective is to analyze historical stock data and predict future prices through an interactive web interface.

following is the source code 

```py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Streamlit title
st.title("Stock Price Prediction Dashboard")

# User input
stock = st.text_input("Enter Stock Symbol (e.g. AAPL)", "AAPL")

# Load data
data = yf.download(stock, start="2020-01-01", end="2024-01-01")

st.subheader("Raw Data")
st.write(data.tail())

# Use only closing price
data = data[['Close']]

# Create feature (days)
data['Days'] = np.arange(len(data))

# Split data
X = data[['Days']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
data['Predicted'] = model.predict(X)

# Plot
st.subheader("Stock Price vs Prediction")

fig, ax = plt.subplots()
ax.plot(data['Close'], label="Actual Price")
ax.plot(data['Predicted'], label="Predicted Price")
ax.legend()

st.pyplot(fig)

# Future prediction
future_days = st.slider("Days to Predict in Future", 1, 30)

future = np.array(range(len(data), len(data) + future_days)).reshape(-1, 1)
future_pred = model.predict(future)

st.subheader("Future Predictions")
st.write(future_pred)```

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

app = Flask(__name__)

# Fetch stock data
def fetch_stock_data(ticker):
    data = yf.download(ticker, period='10y')  # Limit to 10 years of data for quicker response
    print(f"Fetched {len(data)} data points for {ticker}")
    return data

# LSTM Model
def lstm_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = 60
    training_data_len = int(np.ceil(len(scaled_data) * .8))

    training_data = scaled_data[0:training_data_len, :]
    X_train, y_train = create_sequences(training_data, seq_length)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10)  # Reduce epochs for quicker response

    test_data = scaled_data[training_data_len - seq_length:, :]
    X_test, y_test = create_sequences(test_data, seq_length)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions[-7:]  # Return the last 7 days of predictions

# ARIMA Model
def arima_model(data):
    model = ARIMA(data['Close'], order=(5, 1, 0))
    arima_model = model.fit()
    predictions = arima_model.forecast(steps=7)
    return predictions

# Linear Regression Model
def linear_regression_model(data):
    data['Lag1'] = data['Close'].shift(1)
    data.dropna(inplace=True)

    X = data[['Lag1']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict for the next 7 days
    future_dates = pd.date_range(start=data.index[-1], periods=8)[1:]
    future_lag = data['Close'].values[-1].reshape(-1, 1)
    predictions = []
    for _ in range(7):
        pred = model.predict(future_lag)[0]
        predictions.append(pred)
        future_lag = np.array([pred]).reshape(-1, 1)

    return predictions

# Time Series Decomposition
def seasonal_decomposition(data):
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('B')  # Set the frequency to business days
    data['Close'] = data['Close'].fillna(method='ffill')  # Forward fill missing values
    decomposition = seasonal_decompose(data['Close'], model='multiplicative')
    return decomposition

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    data = fetch_stock_data(ticker)
    
    if len(data) < 60:  # Ensure at least 60 data points for LSTM's sequence length
        return render_template('index.html', error="Not enough data for prediction. Please choose another ticker or a longer date range.")

    lstm_preds, arima_preds, lr_preds, decomposition_plot = None, None, None, None

    try:
        lstm_preds = lstm_model(data)
        arima_preds = arima_model(data)
        lr_preds = linear_regression_model(data)
        decomposition = seasonal_decomposition(data)

        # Calculate average predictions
        average_preds = (np.array(lstm_preds).flatten() + np.array(arima_preds).flatten() + np.array(lr_preds).flatten()) / 3

        # Create formatted prediction data for easier display
        predictions = {
            'average': [{"Day": i+1, "Prediction": round(pred, 2)} for i, pred in enumerate(average_preds)],
        }

        # Plot decomposition
        plt.rcParams.update({'figure.figsize': (10, 10)})
        decomposition.plot()
        plt.savefig('static/decomposition.png')

    except Exception as e:
        return render_template('index.html', error=f"Error during prediction: {str(e)}")

    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)

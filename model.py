import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

MODEL_PATH = os.path.join(os.getcwd(), "saved_model", "gru_model.h5")
SCALER_PATH = os.path.join(os.getcwd(), "saved_model", "scaler.npz")
CHART_PATH = os.path.join(os.getcwd(), "static", "charts")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(CHART_PATH, exist_ok=True)

TIME_STEPS = 30  # number of past days used for prediction

# -------------------------------------------------
# Load and preprocess CSV data
# -------------------------------------------------
def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    if "Date" in df.columns:
        df = df.sort_values("Date")

    if "Target_Close_T+1" not in df.columns:
        raise ValueError("Target_Close_T+1 column not found in file")

    feature_cols = [c for c in df.columns if c not in ["Date", "Target_Close_T+1"]]
    X = df[feature_cols].values.astype(float)
    y = df["Target_Close_T+1"].values.astype(float).reshape(-1, 1)
    return df, feature_cols, X, y

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# -------------------------------------------------
# Build GRU model
# -------------------------------------------------
def build_gru(input_shape):
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------------------------------
# Train model locally
# -------------------------------------------------
def train_model_if_needed():
    train_file = os.path.join(os.getcwd(), "uploads", "train_preprocessed.csv")
    if not os.path.exists(train_file):
        return {"status": "no training file found"}

    df, feature_cols, X, y = load_and_prepare(train_file)

    X = pd.DataFrame(X, columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(method="ffill").fillna(method="bfill").values
    y = pd.DataFrame(y).replace([np.inf, -np.inf], np.nan)
    y = y.fillna(method="ffill").fillna(method="bfill").values

    scalerX = MinMaxScaler()
    scalary = MinMaxScaler()
    Xs = scalerX.fit_transform(X)
    ys = scalary.fit_transform(y)

    X_seq, y_seq = create_sequences(Xs, ys)
    if len(X_seq) == 0:
        raise ValueError(f"Not enough rows to create sequences (need > {TIME_STEPS}).")

    split = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    model = build_gru((X_train.shape[1], X_train.shape[2]))
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, callbacks=[es], verbose=1)

    model.save(MODEL_PATH)
    np.savez(SCALER_PATH,
             X_min=scalerX.data_min_, X_max=scalerX.data_max_,
             y_min=scalary.data_min_, y_max=scalary.data_max_)

    print(f"✅ Model saved to: {MODEL_PATH}")
    return {"status": "trained"}

# -------------------------------------------------
# Generate candlestick chart
# -------------------------------------------------
def generate_candlestick_chart(df, ticker):
    df_chart = df.copy()
    df_chart["Date"] = pd.to_datetime(df_chart["Date"])
    df_chart.set_index("Date", inplace=True)
    df_chart = df_chart.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close"
    })

    chart_file = os.path.join(CHART_PATH, f"{ticker}_candlestick.png")
    mpf.plot(df_chart[-60:], type="candle", style="charles", volume=True,
             title=f"{ticker} - Recent Candlestick", savefig=chart_file)
    return f"/static/charts/{ticker}_candlestick.png"

# -------------------------------------------------
# Predict from CSV file
# -------------------------------------------------
def predict_from_file(filepath, predict_date=None):
    df, feature_cols, X, y = load_and_prepare(filepath)
    if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model or scaler not found. Train the model first.")

    # Load scaler
    scaler = np.load(SCALER_PATH)
    X_min, X_max = scaler["X_min"], scaler["X_max"]
    y_min, y_max = scaler["y_min"], scaler["y_max"]

    # Scale features
    Xs = (X - X_min) / (X_max - X_min + 1e-9)
    if len(Xs) < TIME_STEPS:
        raise ValueError(f"Need at least {TIME_STEPS} rows for prediction.")
    last_seq = Xs[-TIME_STEPS:].reshape(1, TIME_STEPS, Xs.shape[1])

    # Load model and predict
    model = load_model(MODEL_PATH, compile=False)
    next_scaled = model.predict(last_seq)
    next_value = next_scaled * (y_max - y_min + 1e-9) + y_min
    next_value = float(next_value.flatten()[0])

    last_close = df["Close"].iloc[-1]
    signal = "BUY" if next_value > last_close * 1.01 else "SELL" if next_value < last_close * 0.99 else "HOLD"
    chart_url = generate_candlestick_chart(df, os.path.splitext(os.path.basename(filepath))[0])

    return {
        "predicted_next_close": next_value,
        "last_date": df["Date"].iloc[-1],
        "signal": signal,
        "chart_url": chart_url
    }

# -------------------------------------------------
# ✅ Predict LIVE stock data from Yahoo Finance
# -------------------------------------------------
def predict_live(ticker, days=180):
    """
    Fetches recent live data for the given ticker from Yahoo Finance,
    and predicts the next closing price using the trained GRU model.

    Parameters:
        ticker (str): Stock symbol, e.g., 'AAPL'
        days (int): Number of past days to use for context (default=60)
    """
    end = datetime.now()
    start = end - timedelta(days=days * 2)  # fetch extra data for safety

    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        raise ValueError(f"No live data found for ticker {ticker}.")

    df.reset_index(inplace=True)
    df.rename(columns={"Date": "Date"}, inplace=True)
    df["Target_Close_T+1"] = df["Close"].shift(-1)
    df.dropna(inplace=True)

    tmp_path = os.path.join("uploads", f"{ticker}_live.csv")
    os.makedirs("uploads", exist_ok=True)
    df.to_csv(tmp_path, index=False)

    return predict_from_file(tmp_path)

# ----------------------------------------------------
# CLEAN REAL-TIME STOCK APP (TRADINGVIEW CHART + PREDICTION)
# ----------------------------------------------------
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import math
from pathlib import Path
from datetime import datetime, date

# BASE PATHS
APP_BASE = Path(__file__).resolve().parent
DATA_DIR = APP_BASE / "data"
MODEL_DIR = APP_BASE / "models"

# Stock tickers
TICKERS = [
    "AAPL","MSFT","TSLA","AMZN","GOOGL","META","NVDA","NFLX","AMD","INTC",
    "IBM","ORCL","PYPL","SHOP","JPM","BAC","WFC","C","GS","MS","V","MA",
    "KO","PEP","MCD","NKE","SBUX","DIS","T","VZ","QCOM","AVGO","CSCO",
    "TSM","BABA","PDD","XOM","CVX","BP","WMT","COST","HD","LOW","UNH",
    "PFE","MRK","JNJ","ABBV","CRM","ADBE"
]

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret123"

# ----------------- MODEL FUNCTIONS -----------------
def load_model_npz(ticker):
    model_file = MODEL_DIR / f"{ticker}_model.npz"
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    return np.load(model_file, allow_pickle=True)


def predict_from_model(model_npz, row, ticker):
    coef = model_npz["coef"]
    means = model_npz["means"]
    stds = np.where(model_npz["stds"] == 0, 1e-8, model_npz["stds"])
    features = list(model_npz["features"].astype(str))

    # feature safety check
    missing = [f for f in features if f not in row]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = np.array([row[f] for f in features], dtype=float)
    X_scaled = (X - means) / stds
    X_bias = np.hstack([1.0, X_scaled])

    prediction = float(np.dot(X_bias+0.23, coef))

    # ✅ OPTIONAL ticker-specific bias (safe)
    if ticker == "AMZN": prediction -= 0.0504 # +2% daily log-return bias (example)
    if ticker == "MSFT": prediction += 0.23 # +2% daily log-return bias (example)
    if ticker == "TSLA": prediction +=0.0177
    return prediction


def generate_signals(df):
    signals = []
    for _, row in df.iterrows():
        signal = "BUY" if row["EMA_10"] > row["EMA_20"] else "SELL"
        signals.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "signal": signal
        })
    return signals[-5:]


# ----------------- ROUTE -----------------
@app.route("/", methods=["GET", "POST"])
def home():
    predicted_price = None
    signals = None
    selected_ticker = None
    future_date = None
    error_message = None
    today = date.today()

    if request.method == "POST":
        selected_ticker = request.form["ticker"].upper()
        future_date = request.form["future_date"]
        requested_date = datetime.strptime(future_date, "%Y-%m-%d").date()

        # ❌ BLOCK PAST DATE
        if requested_date < today:
            error_message = "❌ Past dates are not allowed."
        else:
            file_path = DATA_DIR / f"{selected_ticker}.csv"
            if not file_path.exists():
                error_message = "❌ Data file not found."
            else:
                df = pd.read_csv(file_path, parse_dates=["Date"]).sort_values("Date")

                # Feature engineering
                df["SMA_10"] = df["Close"].rolling(10).mean()
                df["SMA_20"] = df["Close"].rolling(20).mean()
                df["EMA_10"] = df["Close"].ewm(span=10).mean()
                df["EMA_20"] = df["Close"].ewm(span=20).mean()
                df = df.bfill().ffill()

                last_row = df.iloc[-1].to_dict()
                last_date = df["Date"].iloc[-1].date()
                last_close = float(df["Close"].iloc[-1])

                if requested_date <= last_date:
                    predicted_price = last_close
                else:
                    model = load_model_npz(selected_ticker)
                    days_ahead = (requested_date - last_date).days
                    log_return_daily = predict_from_model(
                        model, last_row, selected_ticker
                    )
                    predicted_price = last_close * math.exp(
                        log_return_daily * days_ahead
                    )

                signals = generate_signals(df)

    return render_template(
        "index.html",
        tickers=TICKERS,
        selected_ticker=selected_ticker,
        future_date=future_date,
        predicted_price=predicted_price,
        signals=signals,
        error_message=error_message,
        today=today.strftime("%Y-%m-%d")
    )


# ----------------- RUN -----------------
if __name__ == "__main__":
    app.run(debug=True)

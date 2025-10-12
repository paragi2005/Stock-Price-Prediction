from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from datetime import datetime, timedelta
from model import predict_live  # 🔹 new live prediction function

app = Flask(__name__)
app.secret_key = "replace-with-secure-key"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form.get("ticker", "").upper().strip()
        predict_date_str = request.form.get("predict_date", "").strip()
        model_type = request.form.get("model_type", "GRU")

        if model_type != "GRU":
            flash("Only GRU model is supported in this demo.")
            return redirect(url_for("index"))

        if not ticker:
            flash("Please enter a stock ticker (e.g., AAPL, TSLA, INFY.NS).")
            return redirect(url_for("index"))

        # Handle prediction date (optional)
        try:
            if predict_date_str:
                predict_date = pd.to_datetime(predict_date_str)
            else:
                predict_date = datetime.now()
        except Exception:
            flash("Invalid date format. Use YYYY-MM-DD.")
            return redirect(url_for("index"))

        try:
            # 🔹 Fetch live data & predict
            result = predict_live(ticker, days=60)  # fetch last 60 days of data
            predicted_close = result["predicted_next_close"]
            last_date = result["last_date"]
            signal = result.get("signal", "HOLD")
            chart_url = result.get("chart_url", None)
        except Exception as ex:
            flash(f"Prediction failed: {str(ex)}")
            return redirect(url_for("index"))

        return render_template(
            "index.html",
            ticker=ticker,
            predicted_close=predicted_close,
            last_date=last_date,
            prediction_date=predict_date.strftime("%Y-%m-%d"),
            signal=signal,
            chart_url=chart_url,
            show_result=True,
        )

    return render_template("index.html", show_result=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

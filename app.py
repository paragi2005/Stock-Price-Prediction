from flask import Flask, render_template, request, redirect, url_for, flash
import os
import tempfile
import pandas as pd
from model import predict_from_file

app = Flask(__name__)
app.secret_key = "replace-with-secure-key"

# Preloaded dataset paths
DATA_FILES = {
    "AAPL": "AAPL_UPDATED.csv"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form.get("ticker", "").upper().strip()
        predict_date_str = request.form.get("predict_date", "").strip()
        model_type = request.form.get("model_type", "GRU")

        if model_type != "GRU":
            flash("Only GRU model is supported in this demo.")
            return redirect(url_for("index"))

        if ticker not in DATA_FILES:
            flash(f"No data available for ticker {ticker}. Try AAPL.")
            return redirect(url_for("index"))

        # Load data
        filepath = os.path.join(os.getcwd(), DATA_FILES[ticker])
        df = pd.read_csv(filepath)

        if "Date" not in df.columns:
            flash("CSV must contain a 'Date' column.")
            return redirect(url_for("index"))

        # Convert date
        df["Date"] = pd.to_datetime(df["Date"])
        try:
            predict_date = pd.to_datetime(predict_date_str)
        except Exception:
            flash("Invalid prediction date format. Use YYYY-MM-DD.")
            return redirect(url_for("index"))

        # Filter last 30 days before prediction date
        df_filtered = df[df["Date"] <= predict_date].sort_values(by="Date", ascending=True).tail(30)

        if len(df_filtered) < 30:
            flash("Not enough historical data (need at least 30 days before prediction date).")
            return redirect(url_for("index"))

        # Save temp file
        tmpf = os.path.join(tempfile.gettempdir(), f"{ticker}_tmp.csv")
        df_filtered.to_csv(tmpf, index=False)

        try:
            result = predict_from_file(tmpf, predict_date=predict_date)
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
            prediction_date=predict_date_str,
            signal=signal,
            chart_url=chart_url,
            show_result=True,
        )

    return render_template("index.html", show_result=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

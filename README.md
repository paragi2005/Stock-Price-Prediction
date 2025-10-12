Simplified Flask app using provided GRU model.

To train the model locally (recommended):
1. Create a virtualenv and install requirements:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

2. Copy your preprocessed CSV (AAPL_Processed.csv is included) to uploads/train_preprocessed.csv
   (or place your own preprocessed file there and ensure it contains 'Target_Close_T+1' column).

3. Run training:
   python train.py

After training, run the app:
   python app.py
Open http://localhost:5000

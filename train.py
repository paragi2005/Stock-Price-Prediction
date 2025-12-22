
import os
import shutil
import sys
from model import train_model_if_needed

train_file = os.path.join(os.getcwd(), "uploads", "train_preprocessed.csv")
default_file = os.path.join(os.getcwd(), "AAPL_Processed.csv")

if not os.path.exists(train_file):
    if os.path.exists(default_file):
        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        shutil.copyfile(default_file, train_file)
        print("📄 Copied AAPL_Processed.csv → uploads/train_preprocessed.csv for training.")
    else:
        print("❌ No training file found.")
        print("Please place a preprocessed dataset at: uploads/train_preprocessed.csv")
        sys.exit(1)

print("🚀 Starting GRU model training (requires TensorFlow)...")
result = train_model_if_needed()

print("\n✅ Training complete! Summary:")
print(result)
print("\nSaved model and scaler are now available in 'saved_model/'.")

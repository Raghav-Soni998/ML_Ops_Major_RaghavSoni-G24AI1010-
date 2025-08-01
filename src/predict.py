"""
Author: Raghav Soni
Load trained model and predict on test samples.
"""
from utils import load_model, load_data
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    model = load_model("src/model.joblib")
    _, X_test, _, y_test = load_data()
    preds = model.predict(X_test[:5])
    print("Sample Predictions:", preds)
    print("Actual Values:", y_test[:5])
    mae = mean_absolute_error(y_test[:5], preds)
    print(f"[METRICS] MAE on 5-sample test set: {mae:.4f}")

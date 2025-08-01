"""
Author: Raghav Soni
Train the Linear Regression model and save it.
"""
from utils import train_linear_regression, save_model

if __name__ == "__main__":
    print("[INFO] Starting model training... (Developed by Raghav Soni)")
    model, r2, mse, _ = train_linear_regression()
    print(f"[METRICS] R2 Score: {r2:.4f}")
    print(f"[METRICS] MSE: {mse:.4f}")
    save_model(model, "src/model.joblib")
    print("[INFO] Training complete. Model saved to src/model.joblib")

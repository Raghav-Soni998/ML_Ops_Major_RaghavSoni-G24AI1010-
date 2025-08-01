"""
Author: Raghav Soni
Train the Linear Regression model and save it.
"""
from utils import train_linear_regression, save_model

if __name__ == "__main__":
    model, r2, mse, _ = train_linear_regression()
    print(f"R2 Score: {r2}")
    print(f"MSE: {mse}")
    save_model(model, "src/model.joblib")

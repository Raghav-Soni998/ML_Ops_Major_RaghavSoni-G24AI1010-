"""
Author: Raghav Soni
Utility functions for dataset loading, model training, and saving/loading.
"""
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, random_state=42):
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def describe_dataset(X_train, y_train):
    """Custom helper to print dataset details (unique to Raghav)."""
    print(f"[INFO] Dataset: {X_train.shape[0]} training samples, {X_train.shape[1]} features")
    print(f"[INFO] Example target values: {y_train[:5]}")


def train_linear_regression():
    X_train, X_test, y_train, y_test = load_data()
    describe_dataset(X_train, y_train)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return model, r2, mse, (X_test, y_test)


def save_model(model, path="src/model.joblib"):
    joblib.dump(model, path)


def load_model(path="src/model.joblib"):
    return joblib.load(path)

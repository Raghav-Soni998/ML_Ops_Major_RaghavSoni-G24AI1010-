"""
Author: Raghav Soni
Unit tests for dataset and training pipeline.
"""
from src import utils
from sklearn.linear_model import LinearRegression

def test_data_loading():
    X_train, X_test, y_train, y_test = utils.load_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

def test_model_training():
    model, r2, mse, _ = utils.train_linear_regression()
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")
    assert r2 > 0.5

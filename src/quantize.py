"""
Author: Raghav Soni
Manual quantization of model coefficients and intercept.
"""
import joblib
import numpy as np
from utils import load_model

def quantize_params(params, scale=100):
    return np.clip((params * scale).astype(np.uint8), 0, 255)

def dequantize_params(q_params, scale=100):
    return q_params.astype(np.float32) / scale

if __name__ == "__main__":
    model = load_model("src/model.joblib")

    unquant = {"coef": model.coef_, "intercept": model.intercept_}
    joblib.dump(unquant, "src/unquant_params.joblib")

    q_coef = quantize_params(model.coef_)
    q_intercept = quantize_params(np.array([model.intercept_]))

    joblib.dump({"coef": q_coef, "intercept": q_intercept}, "src/quant_params.joblib")

    dq_coef = dequantize_params(q_coef)
    dq_intercept = dequantize_params(q_intercept)[0]
    print("Dequantized sample:", dq_coef[:5], dq_intercept)

# MLOps Linear Regression Pipeline - By Raghav Soni

This repository implements an MLOps pipeline using:
- Dataset: California Housing (scikit-learn)
- Model: Linear Regression
- Quantization, Docker, and CI/CD workflows

## Steps to Run

1. `python src/train.py`
2. `pytest`
3. `python src/quantize.py`
4. `python src/predict.py`

Docker:
- `docker build -t mlops-raghav .`
- `docker run --rm mlops-raghav`

## Comparison Table

| Stage            | Output Files              |
|------------------|---------------------------|
| Trained Model    | src/model.joblib          |
| Raw Parameters   | src/unquant_params.joblib |
| Quantized Params | src/quant_params.joblib   |

## Unique Features
- Custom dataset summary printed before training.
- Quantization process logs a `quantization_info.txt`.
- Includes MAE metric in predictions for demonstration.


Developed by **Raghav Soni (G24AI1010)**

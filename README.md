# MLOps Linear Regression Pipeline - By Raghav Soni(G24AI1010)

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

In addition to the standard MLOps pipeline requirements, this implementation includes a few enhancements for clarity and reproducibility:

1. **Dataset summary before training**  
   The training script provides a brief summary of the dataset (shape and sample target values) before model training begins. This improves transparency and helps verify data characteristics at runtime.
   
   <img width="832" height="102" alt="image" src="https://github.com/user-attachments/assets/fea42b62-b8db-41da-8a05-5a7035ddc241" />


3. **Quantization settings log**  
   During quantization, the process records the scale factor in a log file (`src/quantization_info.txt`). This ensures that quantization parameters are documented for future reference and reproducibility.
   
   <img width="844" height="62" alt="image" src="https://github.com/user-attachments/assets/828d8fd8-2745-4aca-a34e-77ccacbd0289" />


5. **Additional metric during prediction**  
   The prediction script reports **Mean Absolute Error (MAE)** for the first five test samples, alongside predictions and actual values. This additional metric complements the R² and MSE scores from training.
   
   <img width="840" height="59" alt="image" src="https://github.com/user-attachments/assets/fbe4814d-a256-4a34-85f7-62bf6884c97a" />

These additions provide better insight into the pipeline without altering its core functionality.


Developed by **Raghav Soni (G24AI1010)**

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

from sklearn import set_config
set_config(transform_output="pandas")

def model_fn(model_dir):
    """Load preprocessing pipeline and XGBoost model."""
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))
    logger.info("Preprocessor and model loaded successfully")
    return {"preprocessor": preprocessor, "model": model}

def input_fn(request_body, request_content_type):
    """Efficient JSON parsing and array conversion."""
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    input_data = json.loads(request_body)
    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])  # Single record
    elif isinstance(input_data, list):
        return pd.DataFrame(input_data)    # Batch of records
    else:
        raise ValueError("Input must be a JSON object or array")

def predict_fn(input_data, models):
    """Transform input and predict with XGBoost."""
    try:
        transformed = models["preprocessor"].transform(input_data)
        dmatrix = xgb.DMatrix(transformed)
        return models["model"].predict(dmatrix)
    except Exception as e:
        logger.exception("Prediction failed")
        raise

def output_fn(predictions, accept):
    """Return compact JSON with class & probability."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    
    results = [
        {
            "class": int(p > 0.5),
            "probability": round(p * 100 if p > 0.5 else (1 - p) * 100, 2)
        }
        for p in predictions
    ]
    return json.dumps(results), "application/json"

import joblib
import os
import json
import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from custom_transformers import FrequencyEncoder, FeatureEngineeringTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def model_fn(model_dir):
    """Load the model from the model directory."""
    model_path = os.path.join(model_dir, "model.joblib")
    logger.info(f"Loading preprocessor from {model_path}")
    model = joblib.load(model_path)  # Using joblib since pickle is used in preprocess.py
    return model

def input_fn(request_body, request_content_type):
    """Parse input data from the request."""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        # Assuming input is a list of dictionaries or a single dictionary
        if isinstance(input_data, dict):
            data = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            data = pd.DataFrame(input_data)
        else:
            raise ValueError("Input data must be a JSON object or array")
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Apply the preprocessing pipeline to the input data."""
    logger.info("Transforming input data")
    transformed_data = model.transform(input_data)
    return transformed_data

def output_fn(prediction, accept):
    """Format the prediction output."""
    if accept == "application/json":
        prediction = pd.DataFrame(prediction)
        return prediction.to_json(orient="records")
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
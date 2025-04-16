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
    logger.info(f"Transforming input data type {type(prediction)}")
    if accept == "application/json":
        if isinstance(prediction, np.ndarray):
            prediction_1 = pd.DataFrame(prediction)
            logger.info(f"Attempting to convert it into Dataframe, result datatype is {type(prediction_1)}")
        csv_data = prediction_1.to_csv(index=False)
        logger.info(f"Attempting to convert prediction_1 data into CSV, final file type is {type(csv_data)}")
        return csv_data, 'text/csv'
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
import joblib
import os
import json
import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from custom_transformers import FrequencyEncoder, FeatureEngineeringTransformer
import xgboost as xgb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def model_fn(model_dir):
    """Load both the preprocessing pipeline and the XGBoost model."""
    # Load preprocessing pipeline
    preprocessor_path = os.path.join(model_dir, "model.joblib")
    logger.info(f"Loading preprocessor from {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    
    # Load XGBoost model
    model_path = os.path.join(model_dir, "xgboost-model")
    logger.info(f"Loading XGBoost model from {model_path}")
    logger.info(f"Model path: {model_path}, exists: {os.path.exists(model_path)}, size: {os.path.getsize(model_path)} bytes")
    model = xgb.Booster()
    model.load_model(model_path)
    
    return {"preprocessor": preprocessor, "model": model}

def input_fn(request_body, request_content_type):
    """Parse input data from the request."""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        if isinstance(input_data, dict):
            data = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            data = pd.DataFrame(input_data)
        else:
            raise ValueError("Input data must be a JSON object or array")
        
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {list(data.columns)}")
        logger.info(f"Full input data:\n{data.to_string(index=False)}")
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, models):
    """Apply preprocessing and generate predictions."""
    preprocessor = models["preprocessor"]
    model = models["model"]
    
    logger.info("Starting transformation and prediction")
    try:
        # Preprocess the input data
        transformed_data = preprocessor.transform(input_data)
        
        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(transformed_data)
        
        # Generate predictions
        predictions = model.predict(dmatrix)
        
        logger.info(f"Predictions: {predictions}")
        return predictions
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

def output_fn(prediction, accept):
    """Format the prediction output."""
    logger.info(f"Prediction data type: {type(prediction)}")
    if accept == "application/json":
        # Convert predictions to a DataFrame
        prediction_df = pd.DataFrame(prediction, columns=["prediction"])
        
        # Convert to CSV
        csv_data = prediction_df.to_csv(index=False, header=False)
        logger.info(f"Output CSV:\n{csv_data}")
        return csv_data, "text/csv"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
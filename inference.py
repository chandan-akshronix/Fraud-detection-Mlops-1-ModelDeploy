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
        
        # Log full input data
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {list(data.columns)}")
        logger.info(f"Full input data:\n{data.to_string(index=False)}")  # Log all rows and columns
        return data

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Apply the preprocessing pipeline to the input data and include numerical features list."""
    logger.info("Starting transformation of input data")
    try:
        transformed_data = model.transform(input_data)
        
        # Log full transformed data
        logger.info(f"Transformed data shape: {transformed_data.shape}")
        if isinstance(transformed_data, pd.DataFrame):
            logger.info(f"Transformed data columns: {list(transformed_data.columns)}")
            logger.info(f"Full transformed data:\n{transformed_data.to_string(index=False)}")
        elif isinstance(transformed_data, np.ndarray):
            transformed_df = pd.DataFrame(transformed_data)
            logger.info(f"Transformed data as DataFrame:\n{transformed_df.to_string(index=False)}")
        else:
            logger.info(f"Transformed data (raw): {transformed_data}")
        
        if transformed_data.size == 0:
            logger.warning("Transformed data is empty!")
        
        # Create a list of numerical features from transformed data
        if isinstance(transformed_data, pd.DataFrame):
            features_list = transformed_data.values.tolist()
        else:  # Assuming numpy array
            features_list = transformed_data.tolist()
        
        # Return both prediction and features
        return {"prediction": transformed_data, "features": features_list}
    except Exception as e:
        logger.error(f"Error during transformation: {str(e)}")
        raise

def output_fn(prediction, accept):
    """Format the prediction output."""
    logger.info(f"Prediction data type: {type(prediction)}")
    if accept == "application/json":
        # Extract prediction and features from the dictionary
        prediction_data = prediction["prediction"]
        features_list = prediction["features"]
        
        # Convert prediction to DataFrame if necessary
        if isinstance(prediction_data, np.ndarray):
            prediction_df = pd.DataFrame(prediction_data)
            logger.info(f"Converted to DataFrame, type: {type(prediction_df)}")
        else:
            prediction_df = prediction_data
        
        # Check if data is empty
        if prediction_df.empty:
            logger.warning("Prediction DataFrame is empty before CSV conversion")
        
        # Convert prediction to CSV
        csv_data = prediction_df.to_csv(index=False, header=False)
        logger.info(f"Output CSV type: {type(csv_data)}")
        logger.info(f"Full CSV output:\n{csv_data}")
        
        # Return prediction and features as JSON
        return json.dumps({"prediction": csv_data, "features": features_list}), "application/json"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
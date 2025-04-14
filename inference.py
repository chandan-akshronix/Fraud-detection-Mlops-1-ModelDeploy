import joblib
import os
import json
import numpy as np
import logging

# Set up logging to debug issues
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the model from the model directory."""
    model_path = os.path.join(model_dir, "model.joblib")
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data from the request."""
    logger.info(f"Received request_body: {request_body}")
    if request_content_type == "application/json":
        try:
            input_data = json.loads(request_body)
            # Handle different input formats
            if isinstance(input_data, list):
                # Convert list to 2D array if necessary
                data = np.array(input_data)
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                return data
            elif isinstance(input_data, dict):
                # Assume dictionary values are features
                data = np.array([list(input_data.values())])
                return data
            else:
                raise ValueError("Input must be a list or dictionary")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise ValueError(f"Invalid JSON input: {e}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Apply preprocessing transformation."""
    logger.info(f"Input data shape: {input_data.shape}")
    transformed_data = model.transform(input_data)
    logger.info(f"Transformed data shape: {transformed_data.shape}")
    return transformed_data

def output_fn(prediction, accept):
    """Format the output."""
    if accept == "application/json":
        if isinstance(prediction, np.ndarray):
            return json.dumps(prediction.tolist())
        else:
            # Handle non-array outputs (e.g., sparse matrices)
            return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
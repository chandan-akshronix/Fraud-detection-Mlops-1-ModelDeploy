import joblib
import os

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data."""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return np.array(input_data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Apply preprocessing transformation."""
    transformed_data = model.transform(input_data)
    return transformed_data

def output_fn(prediction, accept):
    """Format the output."""
    if accept == "application/json":
        return json.dumps(prediction.tolist())
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
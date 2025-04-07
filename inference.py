import joblib
import os
import pandas as pd
from io import StringIO

def model_fn(model_dir):
    """Load the preprocessor from the model directory.

    Args:
        model_dir (str): Directory where the model artifacts are stored.

    Returns:
        object: The loaded preprocessor object.
    """
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    preprocessor = joblib.load(preprocessor_path)
    return preprocessor


def transform_fn(preprocessor, input_data, content_type, accept):
    """Apply the preprocessor to the input data and return the transformed data.

    Args:
        preprocessor: The loaded preprocessor object.
        input_data (str): The input data as a string (CSV format).
        content_type (str): The MIME type of the input data (e.g., 'text/csv').
        accept (str): The desired MIME type of the output (ignored here).

    Returns:
        str: The transformed data in CSV format.
    """
    if content_type == "text/csv":
        # Read the input CSV data
        df = pd.read_csv(StringIO(input_data))
        # Apply the preprocessor
        transformed_data = preprocessor.transform(df)
        # Convert back to CSV without index
        output = pd.DataFrame(transformed_data).to_csv(index=False)
        return output
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
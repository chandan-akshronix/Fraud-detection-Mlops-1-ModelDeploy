import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
import xgboost as xgb

import boto3
from botocore.exceptions import ClientError
from datetime import datetime

# Initialize DynamoDB
dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
table = dynamodb.Table("StorageTable_Prod")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

from sklearn import set_config
set_config(transform_output="pandas")

def log_batch_to_dynamodb(transaction_ids, results):
    try:
        with table.batch_writer() as batch:
            for tid, result in zip(transaction_ids, results):
                item = {
                    "TransactionID": str(tid),
                    "Record": {k: str(v) for k, v in result.items()},
                    "Timestamp": datetime.utcnow().isoformat(),
                    "ModelVersion": "v1.0.0"
                }
                batch.put_item(Item=item)
    except ClientError as e:
        logger.error(f"Batch write failed: {e}")


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
        df = pd.DataFrame([input_data])  # Single record
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)    # Batch of records
    else:
        raise ValueError("Input must be a JSON object or array")

    # Ensure transaction_ids is a pandas Series aligned with df
    if "transaction_id" in df.columns:
        transaction_ids = df["transaction_id"]
    else:
        transaction_ids = pd.Series([None] * len(df))

    # Drop transaction_id before passing to model
    features_df = df.drop(columns=["transaction_id"], errors="ignore")

    return features_df, transaction_ids, df

def predict_fn(input_data, models):
    """Transform input and predict with XGBoost, always returning iterable predictions."""
    try:
        features_df, transaction_ids, original_inputs = input_data
        transformed = models["preprocessor"].transform(features_df)
        dmatrix = xgb.DMatrix(transformed)
        preds = models["model"].predict(dmatrix)

        # Normalize preds to 1D array
        if np.isscalar(preds) or isinstance(preds, np.generic):
            preds = np.array([preds])
        else:
            preds = np.asarray(preds)

        # Ensure transaction_ids is a sequence of same length
        # If it's a pandas Series, leave as is; if scalar or length mismatch, wrap
        if not hasattr(transaction_ids, "__len__") or len(transaction_ids) != len(preds):
            # e.g., single value
            transaction_ids = pd.Series([transaction_ids]) if not hasattr(transaction_ids, "__len__") else pd.Series(transaction_ids).reset_index(drop=True)
        # Ensure original_inputs is a DataFrame with matching rows
        if not hasattr(original_inputs, "iterrows") or len(original_inputs) != len(preds):
            # wrap single-row into DataFrame
            original_inputs = pd.DataFrame([original_inputs]) if not hasattr(original_inputs, "iterrows") else original_inputs.reset_index(drop=True)

        return preds, transaction_ids, original_inputs

    except Exception as e:
        logger.exception("Prediction failed")
        raise


def output_fn(predictions_with_ids, accept):
    """Return JSON with class, probability, original features, and transaction_id, and log to DynamoDB."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")

    predictions, transaction_ids, original_inputs = predictions_with_ids  # unpack all 3

    results = []

    for p, tid, (_, row) in zip(predictions, transaction_ids, original_inputs.iterrows()):
        input_features = row.to_dict()
        result = {
            "transaction_id": tid,
            "class": int(p > 0.5),
            "probability": round(p * 100, 2),
            "input_features": input_features
        }
        results.append(result)
    
    # Log once after collecting all results
    if len(results) > 0 and any(transaction_ids):
        log_batch_to_dynamodb(transaction_ids, results)

    return json.dumps(results), "application/json"
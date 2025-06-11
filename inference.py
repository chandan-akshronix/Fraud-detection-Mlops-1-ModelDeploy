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

# 1. Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# 2. Initialize DynamoDB table resource (for logging results)
dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
table = dynamodb.Table("StorageTable_Prod")

# 3. sklearn config (if your pipeline uses DataFrame outputs)
from sklearn import set_config
set_config(transform_output="pandas")


def log_batch_to_dynamodb(transaction_ids, results):
    """
    Batch-write prediction results to DynamoDB.
    transaction_ids: pandas Series aligned with results
    results: list of dicts, each containing at least transaction_id and other fields
    """
    try:
        with table.batch_writer() as batch:
            for tid, result in zip(transaction_ids, results):
                # Build item: you can adjust which fields to include
                item = {
                    "TransactionID": str(tid),
                    "Record": {k: str(v) for k, v in result.items()},
                    "Timestamp": datetime.utcnow().isoformat(),
                    # optionally add ModelVersion or other metadata
                }
                batch.put_item(Item=item)
        logger.info(f"Logged {len(results)} items to DynamoDB")
    except ClientError as e:
        logger.error(f"Batch write failed: {e}")


def model_fn(model_dir):
    """
    Load preprocessing pipeline and XGBoost model once.
    """
    # Adjust paths as needed
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))
    logger.info("Preprocessor and model loaded successfully")
    return {"preprocessor": preprocessor, "model": model}


def input_fn(request_body, request_content_type):
    """
    Parse JSON request (single dict or list), extract Transaction ID, and prepare features.
    Returns:
      features_df: pandas DataFrame for model.transform()
      transaction_ids: pandas Series aligned with rows
      raw_records: list of original input dicts (for output/logging)
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    input_data = json.loads(request_body)

    # Build DataFrame and raw_records list
    if isinstance(input_data, dict):
        raw_records = [input_data]
        df = pd.DataFrame(raw_records)
    elif isinstance(input_data, list):
        raw_records = input_data
        df = pd.DataFrame(raw_records)
    else:
        raise ValueError("Input must be a JSON object or array")

    # Extract "Transaction ID" if present
    if "Transaction ID" in df.columns:
        transaction_ids = df["Transaction ID"].astype(str)
        # Drop that column before passing to model
        features_df = df.drop(columns=["Transaction ID"], errors="ignore")
    else:
        # No Transaction ID field
        transaction_ids = pd.Series([None] * len(df))
        features_df = df

    return features_df, transaction_ids.reset_index(drop=True), raw_records


def predict_fn(input_data, models):
    """
    Apply preprocessing and model prediction.
    Returns normalized (preds_array, transaction_ids, raw_records).
    """
    try:
        features_df, transaction_ids, raw_records = input_data

        # 1. Preprocess (vectorized)
        transformed = models["preprocessor"].transform(features_df)

        # 2. XGBoost prediction via Booster
        dmatrix = xgb.DMatrix(transformed)
        preds = models["model"].predict(dmatrix)

        # 3. Normalize preds to 1D numpy array
        if np.isscalar(preds) or isinstance(preds, np.generic):
            preds = np.array([preds])
        else:
            preds = np.asarray(preds)

        # 4. Align transaction_ids and raw_records if lengths mismatch
        #    (usually only for single-record cases)
        n = len(preds)
        # transaction_ids: pandas Series
        if len(transaction_ids) != n:
            if len(transaction_ids) == 1:
                transaction_ids = pd.Series([transaction_ids.iloc[0]] * n)
            else:
                # unexpected mismatch: reset to None
                transaction_ids = pd.Series([None] * n)
        # raw_records: list
        if len(raw_records) != n:
            if len(raw_records) == 1:
                raw_records = [raw_records[0]] * n
            else:
                # unexpected: create empty dicts
                raw_records = [{} for _ in range(n)]

        return preds, transaction_ids.reset_index(drop=True), raw_records

    except Exception as e:
        logger.exception("Prediction failed")
        raise


def output_fn(predictions_with_ids, accept):
    """
    Build JSON response including:
      - transaction_id
      - class (0/1 by threshold 0.5)
      - probability (p*100, rounded)
      - input_features: the original raw dict
    Also batch-log to DynamoDB if any non-null IDs.
    """
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")

    preds, transaction_ids, raw_records = predictions_with_ids

    results = []
    # Iterate aligned preds, IDs, and raw input dicts
    for p, tid, rec in zip(preds, transaction_ids, raw_records):
        # Build result using original rec (dict). If rec lacks some fields, it's as-is.
        result = {
            "transaction_id": tid,
            "class": int(p > 0.5),
            "probability": round(float(p) * 100, 2),
            # "input_features": rec
        }
        results.append(result)

    # Batch-write if any non-null IDs
    if len(results) and any([tid is not None for tid in transaction_ids]):
        # Use pandas Series transaction_ids aligned with results
        log_batch_to_dynamodb(transaction_ids, results)

    return json.dumps(results), "application/json"
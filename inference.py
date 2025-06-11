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

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Initialize DynamoDB table
dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
table = dynamodb.Table("StorageTable_Prod")

# sklearn config
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
                }
                batch.put_item(Item=item)
        logger.info(f"Logged {len(results)} items to DynamoDB")
    except ClientError as e:
        logger.error(f"DynamoDB batch write failed: {e}")


def model_fn(model_dir):
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))
    logger.info("Model and preprocessor loaded")
    return {"preprocessor": preprocessor, "model": model}


def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    input_data = json.loads(request_body)

    # Build DataFrame and raw_records
    if isinstance(input_data, dict):
        raw_records = [input_data]
        df = pd.DataFrame(raw_records)
    elif isinstance(input_data, list):
        raw_records = input_data
        df = pd.DataFrame(raw_records)
    else:
        raise ValueError("Input must be a JSON object or array")

    # logger.info(f"input_fn: DataFrame columns: {df.columns.tolist()}")

    # Robust find of transaction ID column
    trans_col = None
    for col in df.columns:
        normalized = col.strip().lower().replace(" ", "").replace("_", "")
        if normalized == "transactionid":
            trans_col = col
            break

    if trans_col is not None:
        transaction_ids = df[trans_col].astype(str)
        features_df = df.drop(columns=[trans_col], errors="ignore")
        # logger.info(f"input_fn: Using transaction ID column '{trans_col}'")
    else:
        transaction_ids = pd.Series([None] * len(df))
        features_df = df
        logger.warning("input_fn: No transaction ID column found; IDs will be None")

    transaction_ids = transaction_ids.reset_index(drop=True)
    return features_df, transaction_ids, raw_records


def predict_fn(input_data, models):
    try:
        features_df, transaction_ids, raw_records = input_data

        # Preprocess
        transformed = models["preprocessor"].transform(features_df)

        # Predict
        dmatrix = xgb.DMatrix(transformed)
        preds = models["model"].predict(dmatrix)

        # Normalize preds to array
        if np.isscalar(preds) or isinstance(preds, np.generic):
            preds = np.array([preds])
        else:
            preds = np.asarray(preds)

        # Align lengths
        n = len(preds)
        if len(transaction_ids) != n:
            if len(transaction_ids) == 1:
                transaction_ids = pd.Series([transaction_ids.iloc[0]] * n)
            else:
                transaction_ids = pd.Series([None] * n)
        if len(raw_records) != n:
            if len(raw_records) == 1:
                raw_records = [raw_records[0]] * n
            else:
                raw_records = [{} for _ in range(n)]

        transaction_ids = transaction_ids.reset_index(drop=True)
        logger.info(f"predict_fn: preds len={len(preds)}, transaction_ids len={len(transaction_ids)}, raw_records len={len(raw_records)}")
        return preds, transaction_ids, raw_records

    except Exception as e:
        logger.exception("Prediction failed")
        raise


def output_fn(predictions_with_ids, accept):
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")

    preds, transaction_ids, raw_records = predictions_with_ids
    logger.info(f"output_fn: Received {len(preds)} predictions")

    results = []
    for p, tid, rec in zip(preds, transaction_ids, raw_records):
        # Include full input_features if desired:
        result = {
            "transaction_id": tid,
            "class": int(p > 0.5),
            "probability": round(float(p) * 100, 2),
            "input_features": rec  # original dict
        }
        results.append(result)


    if len(results) and any(tid is not None for tid in transaction_ids):
        logger.info("output_fn: Writing to DynamoDB")
        # log_batch_to_dynamodb may store only minimal fields or full raw_records if desired
        log_batch_to_dynamodb(transaction_ids, results)
    else:
        logger.warning("output_fn: No valid transaction IDs; skipping DynamoDB write")

    summary = {
        "processed_count": len(results),
        "transaction_ids": [tid for tid in transaction_ids if tid is not None]
    }
    return json.dumps(summary), "application/json"
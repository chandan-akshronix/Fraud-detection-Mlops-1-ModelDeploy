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
from decimal import Decimal
import concurrent.futures

# 1. Setup logging at WARNING to reduce overhead
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.addHandler(logging.StreamHandler())

# 2. Initialize DynamoDB table resource
dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
table = dynamodb.Table("StorageTable_Prod")

# 3. Thread pool for async logging
_LOG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# 4. sklearn config (if your pipeline outputs pandas DataFrame)
from sklearn import set_config
set_config(transform_output="pandas")


def log_batch_to_dynamodb(transaction_ids, raw_records, preds, prob_percentages):
    """
    Batch-write full details to DynamoDB in background:
      - TransactionID (partition key)
      - PredictionClass (0/1 as Decimal)
      - Probability (Decimal)
      - Timestamp
      - InputFeatures (map of original fields)
    Skips records where TransactionID is None.
    """
    try:
        with table.batch_writer() as batch:
            for tid, rec, pred, prob in zip(transaction_ids, raw_records, preds, prob_percentages):
                if tid is None:
                    # skip writing if no valid ID
                    continue

                # Serialize input features dict to JSON string  
                features_json = json.dumps(rec)
                
                # Build item: use Decimal for numeric attrs
                item = {
                    "TransactionID": str(tid),
                    "PredictionClass": Decimal(str(int(pred > 0.5))),
                    "Probability": Decimal(str(prob)),
                    "Timestamp": datetime.utcnow().isoformat(),
                    "InputFeatures": features_json
                }
                batch.put_item(Item=item)
    except Exception as e:
        logger.error("Async DynamoDB write failed: %s", e)



def model_fn(model_dir):
    """
    Load preprocessing pipeline and XGBoost model once at startup.
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))
    return {"preprocessor": preprocessor, "model": model}


def input_fn(request_body, request_content_type):
    """
    Parse JSON request (single dict or list), extract Transaction ID, and prepare features_df.
    Returns: (features_df, transaction_ids pd.Series, raw_records list).
    """
    if request_content_type != "application/json":
        raise ValueError("Unsupported content type")
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

    # Find Transaction ID column robustly
    trans_col = None
    for col in df.columns:
        if col.strip().lower().replace(" ", "").replace("_", "") == "transactionid":
            trans_col = col
            break

    if trans_col:
        transaction_ids = df[trans_col].astype(str).reset_index(drop=True)
        features_df = df.drop(columns=[trans_col], errors="ignore")
    else:
        transaction_ids = pd.Series([None] * len(df))
        features_df = df

    return features_df, transaction_ids, raw_records

def predict_fn(input_data, models):
    """
    Preprocess and predict. Assumes model.predict returns array matching features_df length.
    Returns (preds np.ndarray, transaction_ids pd.Series, raw_records list).
    """
    features_df, transaction_ids, raw_records = input_data

    # 1. Preprocess vectorized
    transformed = models["preprocessor"].transform(features_df)

    # 2. Predict via Booster
    dmatrix = xgb.DMatrix(transformed)
    preds = models["model"].predict(dmatrix)
    preds = np.asarray(preds)

    # Assume len(preds) == len(transaction_ids) == len(raw_records)
    # If mismatch is possible, add alignment logic here.

    return preds, transaction_ids.reset_index(drop=True), raw_records

def output_fn(predictions_with_ids, accept):
    """
    Return minimal summary immediately and offload full logging to background.
    Response: {"processed_count": N, "transaction_ids": [...]}
    """
    if accept != "application/json":
        raise ValueError("Unsupported accept type")

    preds, transaction_ids, raw_records = predictions_with_ids
    n = len(preds)

    # Build minimal summary
    summary = [
        {
            "class": int(p > 0.5),
            "probability": round(p * 100)
        }
        for p in preds
    ]

    # Offload full logging if any valid IDs
    if n > 0 and any(tid is not None for tid in transaction_ids):
        # Prepare prediction values and probabilities
        # pred > 0.5 yields class; probability percentage = p * 100
        pred_vals = [int(p > 0.5) for p in preds]
        prob_percs = [round(float(p) * 100, 2) for p in preds]
        # Submit background task; does not block response
        _LOG_EXECUTOR.submit(
            log_batch_to_dynamodb,
            transaction_ids,
            raw_records,
            pred_vals,
            prob_percs
        )

    # Immediate return
    return json.dumps(summary), "application/json"
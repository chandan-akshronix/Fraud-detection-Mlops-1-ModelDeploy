import argparse
import json
import logging
import os
import shutil
import tarfile
import tempfile

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug("Getting more packages for token: {}".format(response["NextToken"]))
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            )
            logger.error(error_message)
            raise Exception(error_message)

        # Return the model package ARN
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


def extend_config(args, model_package_arn, stage_config, sklearn_model_data_url, xgboost_image, xgboost_model_data_url):
    """
    Extend the stage configuration with additional parameters and tags.

    Args:
        args: Command-line arguments.
        model_package_arn: ARN of the approved model package.
        stage_config: The original stage configuration dictionary.
        sklearn_model_data_url: S3 URL of the SKLearn model artifact.
        xgboost_image: ECR image URI for the XGBoost model.
        xgboost_model_data_url: S3 URL of the XGBoost model artifact.

    Returns:
        Updated configuration dictionary.
    """
    # Verify that config has parameters and tags sections
    if not "Parameters" in stage_config or not "StageName" in stage_config["Parameters"]:
        raise Exception("Configuration file must include StageName parameter")
    if not "Tags" in stage_config:
        stage_config["Tags"] = {}
    # Define SKLearn image URI (hardcoded for us-west-2; adjust for your region)
    sklearn_image = "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

    # Create new parameters
    new_params = {
        "SageMakerProjectName": args.sagemaker_project_name,
        "SKLearnImage": sklearn_image,
        "SKLearnModelDataUrl": sklearn_model_data_url,
        "XGBoostImage": xgboost_image,
        "XGBoostModelDataUrl": xgboost_model_data_url,
        "ModelExecutionRoleArn": args.model_execution_role,
        "DataCaptureUploadPath": "s3://" + args.s3_bucket + '/datacapture-' + stage_config["Parameters"]["StageName"],
        "ModelPackageName": model_package_arn,
    }
    new_tags = {
        "sagemaker:deployment-stage": stage_config["Parameters"]["StageName"],
        "sagemaker:project-id": args.sagemaker_project_id,
        "sagemaker:project-name": args.sagemaker_project_name,
    }
    # Add tags from Project
    get_pipeline_custom_tags(args, sm_client, new_tags)

    return {
        "Parameters": {**stage_config["Parameters"], **new_params},
        "Tags": {**stage_config.get("Tags", {}), **new_tags},
    }


def get_pipeline_custom_tags(args, sm_client, new_tags):
    try:
        response = sm_client.describe_project(
            ProjectName=args.sagemaker_project_name
        )
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags[project_tag["Key"]] = project_tag["Value"]
    except:
        logger.error("Error getting project tags")
    return new_tags


def get_cfn_style_config(stage_config):
    parameters = []
    for key, value in stage_config["Parameters"].items():
        parameter = {
            "ParameterKey": key,
            "ParameterValue": value
        }
        parameters.append(parameter)
    tags = []
    for key, value in stage_config["Tags"].items():
        tag = {
            "Key": key,
            "Value": value
        }
        tags.append(tag)
    return parameters, tags


def create_cfn_params_tags_file(config, export_params_file, export_tags_file):
    # Write Params and tags in separate file for CFN CLI command
    parameters, tags = get_cfn_style_config(config)
    with open(export_params_file, "w") as f:
        json.dump(parameters, f, indent=4)
    with open(export_tags_file, "w") as f:
        json.dump(tags, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--model-execution-role", type=str, required=True)
    parser.add_argument("--model-package-group-name", type=str, required=True)
    parser.add_argument("--sagemaker-project-id", type=str, required=True)
    parser.add_argument("--sagemaker-project-name", type=str, required=True)
    parser.add_argument("--s3-bucket", type=str, required=True)
    parser.add_argument("--preprocess-s3-path", type=str, required=False, help="S3 path to preprocess.tar.gz (optional)")
    parser.add_argument("--import-staging-config", type=str, default="staging-config.json")
    parser.add_argument("--import-prod-config", type=str, default="prod-config.json")
    parser.add_argument("--export-staging-config", type=str, default="staging-config-export.json")
    parser.add_argument("--export-staging-params", type=str, default="staging-params-export.json")
    parser.add_argument("--export-staging-tags", type=str, default="staging-tags-export.json")
    parser.add_argument("--export-prod-config", type=str, default="prod-config-export.json")
    parser.add_argument("--export-prod-params", type=str, default="prod-params-export.json")
    parser.add_argument("--export-prod-tags", type=str, default="prod-tags-export.json")
    parser.add_argument("--export-cfn-params-tags", type=bool, default=False)
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    # Get the latest approved package
    model_package_arn = get_approved_package(args.model_package_group_name)

    response = sm_client.describe_model_package(ModelPackageName=model_package_arn)
    preprocess_s3_path = args.preprocess_s3_path or response["CustomerMetadataProperties"].get("preprocess_s3_path")
    if not preprocess_s3_path:
        raise ValueError("preprocess_s3_path not provided and not found in model package customer metadata")

    # Process the preprocessing artifact
    s3 = boto3.client("s3")

    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Download the input preprocess tarball
        bucket, key = preprocess_s3_path.replace("s3://", "").split("/", 1)
        local_preprocess_path = os.path.join(tmpdirname, "preprocess.tar.gz")
        s3.download_file(bucket, key, local_preprocess_path)

        # Extract the preprocess tarball
        with tarfile.open(local_preprocess_path, "r:gz") as tar:
            tar.extractall(path=tmpdirname)

        # Ensure model.joblib exists
        preprocess_path = os.path.join(tmpdirname, "preprocessor.pkl")
        model_joblib_path = os.path.join(tmpdirname, "model.joblib")

        if os.path.exists(preprocess_path):
            logger.info("Renaming preprocessor.pkl to model.joblib")
            os.rename(preprocess_path, model_joblib_path)
        elif os.path.exists(model_joblib_path):
            logger.info("model.joblib already exists")
        else:
            logger.error("No preprocessor.pkl or model.joblib found in preprocess.tar.gz")
            raise FileNotFoundError("No model file found")
        
        # Copy inference.py from the same directory as build.py
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of build.py
        source_inference_path = os.path.join(script_dir, "inference.py")
        inference_script_path = os.path.join(tmpdirname, "inference.py")

        if not os.path.exists(inference_source_path):
            logger.error("inference.py not found in script directory: %s", script_dir)
            raise FileNotFoundError("inference.py not found in script directory")
    
        shutil.copy(inference_source_path, inference_script_path)
        logger.info("Copied inference.py to temporary directory")

        # Verify inference.py was copied
        if not os.path.exists(inference_script_path):
            logger.error("Failed to copy inference.py to temporary directory")
            raise FileNotFoundError("inference.py not found after copy")
        
        # Log files in temp directory for debugging
        logger.info("Files in temp directory: %s", os.listdir(tmpdirname))

        # Create a new tarball with model.joblib and inference.py
        sklearn_model_tar_path = os.path.join(tmpdirname, "sklearn_model.tar.gz")
        with tarfile.open(sklearn_model_tar_path, "w:gz") as tar:
            tar.add(model_joblib_path, arcname="model.joblib")
            tar.add(inference_script_path, arcname="inference.py")

        sklearn_model_s3_key = "models/sklearn_model.tar.gz"
        s3.upload_file(sklearn_model_tar_path, args.s3_bucket, sklearn_model_s3_key)
        sklearn_model_data_url = f"s3://{args.s3_bucket}/{sklearn_model_s3_key}"
        logger.info(f"Uploaded model artifact to {sklearn_model_data_url}")

    # Get XGBoost model details from the model package
    xgboost_image = response["InferenceSpecification"]["Containers"][0]["Image"]
    xgboost_model_data_url = response["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    # Write the staging config
    with open(args.import_staging_config, "r") as f:
        staging_config = extend_config(
            args, model_package_arn, json.load(f), sklearn_model_data_url, xgboost_image, xgboost_model_data_url
        )
    logger.debug("Staging config: {}".format(json.dumps(staging_config, indent=4)))
    with open(args.export_staging_config, "w") as f:
        json.dump(staging_config, f, indent=4)
    if args.export_cfn_params_tags:
        create_cfn_params_tags_file(staging_config, args.export_staging_params, args.export_staging_tags)

    # Write the prod config for CodePipeline
    with open(args.import_prod_config, "r") as f:
        prod_config = extend_config(
            args, model_package_arn, json.load(f), sklearn_model_data_url, xgboost_image, xgboost_model_data_url
        )
    logger.debug("Prod config: {}".format(json.dumps(prod_config, indent=4)))
    with open(args.export_prod_config, "w") as f:
        json.dump(prod_config, f, indent=4)
    if args.export_cfn_params_tags:
        create_cfn_params_tags_file(prod_config, args.export_prod_params, args.export_prod_tags)
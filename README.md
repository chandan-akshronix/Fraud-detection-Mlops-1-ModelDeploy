# Fraud-detection-Mlops-1-ModelDeploy

This repository serves as the deployment component of the fraud detection MLOps pipeline. Its primary purpose is to deploy a trained fraud detection model, registered in the SageMaker Model Registry, to an Amazon SageMaker endpoint. The deployment leverages a multi-container architecture with an SKLearn container for preprocessing and an XGBoost container for prediction. Integrated with SageMaker Projects, this repository enables a CI/CD-driven deployment process for both staging and production environments.

## Features

- **Multi-Container Endpoint**: Deploys a SageMaker endpoint with two containers:
  - **SKLearn Container**: Manages data preprocessing using a custom inference script.
  - **XGBoost Container**: Executes predictions with the trained XGBoost model.
- **Environment-Specific Configurations**: Supports distinct configurations for staging and production via JSON files.
- **CI/CD Integration**: Utilizes AWS CodeBuild and CloudFormation within SageMaker Projects for automated deployments.
- **Data Capture**: Enables input and output data capture on the SageMaker endpoint for monitoring.
- **Testing Framework**: Provides a test suite to validate the deployed endpoint’s functionality.

## File Structure

The repository is organized as follows:

```
Fraud-detection-Mlops-1-ModelDeploy/
├── test/
│   ├── buildspec.yml           # CodeBuild specification for testing the deployment
│   └── test.py                 # Script to test the deployed SageMaker endpoint
├── .gitignore                  # Git ignore rules for excluding unnecessary files
├── README.md                   # Project documentation (this file)
├── build.py                    # Script to prepare deployment artifacts and configurations
├── buildspec.yml               # CodeBuild specification for building and packaging
├── custom_transformers.py      # Custom preprocessing logic shared with ModelBuild
├── endpoint-config-template.yml # CloudFormation template for SageMaker resources
├── inference.py                # Inference script for the SKLearn preprocessing container
├── prod-config.json            # Production environment configuration
└── staging-config.json         # Staging environment configuration
```

## File Descriptions

### `test/buildspec.yml`
- Defines the AWS CodeBuild process for testing the deployment, specifying the environment and commands to execute `test.py`.

### `test/test.py`
- Tests the deployed SageMaker endpoint by sending sample requests and verifying responses.

### `.gitignore`
- Excludes unnecessary files (e.g., `.sagemaker-code-config`) from version control.

### `README.md`
- Provides an overview, setup instructions, file descriptions, and usage guidelines for the repository.

### `build.py`
- Prepares deployment artifacts by:
  - Retrieving the latest approved model package from the SageMaker Model Registry.
  - Processing the preprocessing artifact, adding `inference.py` and `custom_transformers.py`.
  - Uploading a new SKLearn artifact to S3.
  - Extending staging and production configurations with necessary parameters.

### `buildspec.yml`
- Defines the AWS CodeBuild process for deployment, installing dependencies, running `build.py`, and packaging the CloudFormation template.

### `custom_transformers.py`
- Contains custom preprocessing logic (e.g., `FrequencyEncoder`, `FeatureEngineeringTransformer`) for consistency with the training phase.

### `endpoint-config-template.yml`
- A CloudFormation template that defines SageMaker resources, including the model, endpoint configuration, and multi-container endpoint.

### `inference.py`
- The inference script for the SKLearn container, responsible for loading the preprocessing pipeline, transforming input data, and preparing it for XGBoost prediction.

### `prod-config.json`
- Configuration file specifying production environment parameters (e.g., instance count, type, data capture settings).

### `staging-config.json`
- Configuration file for the staging environment, similar to `prod-config.json` but with staging-specific settings.

## Getting Started

### Prerequisites
- **AWS Account**: With permissions for SageMaker, S3, and IAM.
- **Python**: Version 3.11 or later.
- **Dependencies**: `boto3`, `botocore`, `awscli`, `sagemaker` (install manually or via `requirements.txt` if provided).
- **AWS CLI**: Configured with valid credentials.
- **Git**: For cloning the repository.

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/chandan-akshronix/Fraud-detection-Mlops-1-ModelDeploy.git
   cd Fraud-detection-Mlops-1-ModelDeploy
   ```

2. **Install Dependencies**:
   - Dependencies are typically installed during the CodeBuild process. For local testing:
     ```bash
     pip install boto3 botocore awscli sagemaker
     ```

## Usage

The deployment is automated using AWS CodeBuild and CloudFormation. Follow these steps:

1. **Prepare Artifacts and Configurations**:
   - Execute `build.py` with the required arguments:
     ```bash
     python build.py --model-execution-role "arn:aws:iam::123456789012:role/SageMakerRole" \
                     --model-package-group-name "FraudPackageGroup" \
                     --sagemaker-project-id "p-12345" \
                     --sagemaker-project-name "FraudDetection" \
                     --s3-bucket "sagemaker-deploy-bucket" \
                     --export-staging-config "staging-config-export.json" \
                     --export-prod-config "prod-config-export.json"
     ```
   - This generates updated configuration files and uploads artifacts to S3.

2. **Deploy the CloudFormation Stack**:
   - Deploy the stack using the AWS CLI:
     ```bash
     aws cloudformation deploy \
         --template-file <EXPORT_TEMPLATE_NAME> \
         --stack-name FraudDetectionDeployment \
         --capabilities CAPABILITY_NAMED_IAM \
         --parameter-overrides file://staging-config-export.json  # or prod-config-export.json
     ```

3. **Test the Endpoint**:
   - Validate the endpoint with the test script:
     ```bash
     python test/test.py --endpoint-name "FraudDetection-staging"
     ```

## Deployment Workflow

1. **Model Package Retrieval**: `build.py` fetches the latest approved model package.
2. **Artifact Preparation**: Processes and uploads artifacts for SKLearn and retrieves XGBoost details.
3. **Configuration Extension**: Updates staging and production configurations.
4. **Infrastructure Deployment**: Deploys SageMaker resources via CloudFormation.
5. **Multi-Container Workflow**: SKLearn preprocesses data, and XGBoost generates predictions.
6. **Testing**: Validates the endpoint’s functionality.

## Integration with ModelBuild

This repository works in tandem with `Fraud-detection-Mlops-1-ModelBuild`:
- **ModelBuild**: Trains and registers the model.
- **ModelDeploy**: Deploys the registered model to SageMaker.
- Shared `custom_transformers.py` ensures preprocessing consistency.

## Contributing

Contributions are encouraged! Refer to `CONTRIBUTING.md` (if available) for guidelines on pull requests and our code of conduct.

## License

This project is distributed under the terms outlined in the `LICENSE` file (if available).

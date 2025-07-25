"""
ML platform integration for ChemiAccel.

Provides native integration with cloud ML platforms including Amazon SageMaker,
Google Vertex AI, and Azure ML for seamless workflow integration.
"""

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MLJobConfig:
    """Configuration for ML platform jobs."""

    job_name: str
    instance_type: str
    instance_count: int = 1
    input_data_uri: str = None
    output_data_uri: str = None
    hyperparameters: Dict[str, Any] = None
    environment_variables: Dict[str, str] = None
    max_runtime_seconds: int = 86400  # 24 hours
    role_arn: str = None  # AWS specific
    service_account: str = None  # GCP specific


@dataclass
class MLJobStatus:
    """ML job status information."""

    job_name: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    failure_reason: Optional[str] = None
    metrics: Dict[str, Any] = None


class MLPlatformProvider(ABC):
    """Abstract base class for ML platform providers."""

    @abstractmethod
    def submit_processing_job(
        self, job_config: MLJobConfig, processing_script: str
    ) -> str:
        """Submit a processing job and return job name."""
        pass

    @abstractmethod
    def get_job_status(self, job_name: str) -> MLJobStatus:
        """Get current job status."""
        pass

    @abstractmethod
    def stop_job(self, job_name: str) -> bool:
        """Stop a running job."""
        pass

    @abstractmethod
    def get_job_logs(self, job_name: str) -> str:
        """Retrieve job logs."""
        pass


class SageMakerIntegration(MLPlatformProvider):
    """Amazon SageMaker integration for ChemiAccel."""

    def __init__(self, region: str = "us-east-1", role_arn: str = None):
        """
        Initialize SageMaker client.

        Args:
            region: AWS region
            role_arn: IAM role ARN for SageMaker execution
        """
        try:
            import boto3

            self.sagemaker_client = boto3.client("sagemaker", region_name=region)
            self.logs_client = boto3.client("logs", region_name=region)
            self.s3_client = boto3.client("s3", region_name=region)
            self.region = region
            self.default_role_arn = role_arn
        except ImportError:
            raise ImportError(
                "boto3 is required for SageMaker integration. Install with: pip install boto3"
            )

    def submit_processing_job(
        self, job_config: MLJobConfig, processing_script: str
    ) -> str:
        """Submit SageMaker Processing Job with GPU support."""
        job_name = job_config.job_name or f"chemi-accel-{uuid.uuid4().hex[:8]}"

        # Prepare processing inputs
        processing_inputs = []
        if job_config.input_data_uri:
            processing_inputs.append(
                {
                    "InputName": "input-data",
                    "S3Input": {
                        "S3Uri": job_config.input_data_uri,
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                }
            )

        # Prepare processing outputs
        processing_outputs = []
        if job_config.output_data_uri:
            processing_outputs.append(
                {
                    "OutputName": "output-data",
                    "S3Output": {
                        "S3Uri": job_config.output_data_uri,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            )

        # Enhanced GPU configuration
        volume_size = 100 if "gpu" in job_config.instance_type.lower() else 30

        # Create processing job with enhanced GPU support
        response = self.sagemaker_client.create_processing_job(
            ProcessingJobName=job_name,
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceType": job_config.instance_type,
                    "InstanceCount": job_config.instance_count,
                    "VolumeSizeInGB": volume_size,
                }
            },
            AppSpecification={
                "ImageUri": self._get_chemi_accel_image_uri(),
                "ContainerEntrypoint": ["python3"],
                "ContainerArguments": ["/opt/ml/code/processing_script.py"],
            },
            ProcessingInputs=processing_inputs,
            ProcessingOutputs=processing_outputs,
            RoleArn=job_config.role_arn or self.default_role_arn,
            Environment={
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "CHEMI_ACCEL_BACKEND": "cuda",
                **(job_config.environment_variables or {}),
            },
            StoppingCondition={"MaxRuntimeInSeconds": job_config.max_runtime_seconds},
            Tags=[
                {"Key": "Project", "Value": "ChemiAccel"},
                {"Key": "Type", "Value": "ProcessingJob"},
                {
                    "Key": "GPU-Enabled",
                    "Value": str("gpu" in job_config.instance_type.lower()),
                },
            ],
        )

        # Upload processing script
        self._upload_processing_script(processing_script, job_name)

        logger.info(f"Submitted SageMaker processing job: {job_name}")
        return job_name

    def get_job_status(self, job_name: str) -> MLJobStatus:
        """Get SageMaker job status."""
        response = self.sagemaker_client.describe_processing_job(
            ProcessingJobName=job_name
        )

        return MLJobStatus(
            job_name=job_name,
            status=response["ProcessingJobStatus"],
            created_at=response["CreationTime"].isoformat(),
            started_at=response.get("ProcessingStartTime", {}).isoformat()
            if response.get("ProcessingStartTime")
            else None,
            completed_at=response.get("ProcessingEndTime", {}).isoformat()
            if response.get("ProcessingEndTime")
            else None,
            failure_reason=response.get("FailureReason"),
            metrics=response.get("ProcessingResources", {}),
        )

    def stop_job(self, job_name: str) -> bool:
        """Stop SageMaker processing job."""
        try:
            self.sagemaker_client.stop_processing_job(ProcessingJobName=job_name)
            return True
        except Exception as e:
            logger.error(f"Failed to stop SageMaker job {job_name}: {e}")
            return False

    def get_job_logs(self, job_name: str) -> str:
        """Get SageMaker job logs."""
        try:
            log_group = "/aws/sagemaker/ProcessingJobs"
            log_stream = f"{job_name}/algo-1-{int(time.time())}"

            response = self.logs_client.get_log_events(
                logGroupName=log_group, logStreamName=log_stream
            )

            logs = []
            for event in response["events"]:
                logs.append(f"[{event['timestamp']}] {event['message']}")

            return "\n".join(logs)
        except Exception as e:
            logger.error(f"Failed to retrieve SageMaker logs for {job_name}: {e}")
            return f"Error retrieving logs: {e}"

    def _get_chemi_accel_image_uri(self) -> str:
        """Get ChemiAccel container image URI for SageMaker."""
        # This would typically be a pre-built ECR image
        account_id = self._get_aws_account_id()
        return f"{account_id}.dkr.ecr.{self.region}.amazonaws.com/chemi-accel-sagemaker:latest"

    def _get_aws_account_id(self) -> str:
        """Get AWS account ID."""
        import boto3

        sts = boto3.client("sts")
        return sts.get_caller_identity()["Account"]

    def _upload_processing_script(self, script_content: str, job_name: str):
        """Upload processing script to S3."""
        import boto3

        s3_client = boto3.client("s3")
        bucket_name = f"chemi-accel-sagemaker-{self.region}"
        key = f"processing-scripts/{job_name}/processing_script.py"

        # Create bucket if it doesn't exist
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except:
            s3_client.create_bucket(Bucket=bucket_name)

        # Upload script
        s3_client.put_object(
            Bucket=bucket_name, Key=key, Body=script_content.encode("utf-8")
        )

    def create_model_endpoint(
        self, model_data_uri: str, endpoint_name: str = None
    ) -> str:
        """Create SageMaker model endpoint for real-time inference."""
        endpoint_name = endpoint_name or f"chemi-accel-endpoint-{uuid.uuid4().hex[:8]}"

        # Create model
        model_name = f"{endpoint_name}-model"
        self.sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": self._get_chemi_accel_image_uri(),
                "ModelDataUrl": model_data_uri,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
                },
            },
            ExecutionRoleArn=self.default_role_arn,
        )

        # Create endpoint configuration
        config_name = f"{endpoint_name}-config"
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "primary",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.g4dn.xlarge",
                    "InitialVariantWeight": 1.0,
                }
            ],
        )

        # Create endpoint
        self.sagemaker_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=config_name
        )

        logger.info(f"Created SageMaker endpoint: {endpoint_name}")
        return endpoint_name

    def create_pipeline_step(
        self,
        step_name: str,
        job_config: MLJobConfig,
        descriptor_config: Dict[str, Any] = None,
    ) -> dict:
        """Create a native SageMaker Pipeline processing step."""
        try:
            from sagemaker.processing import ProcessingInput, ProcessingOutput
            from sagemaker.sklearn.processing import SKLearnProcessor
            from sagemaker.workflow.parameters import ParameterString
            from sagemaker.workflow.steps import ProcessingStep
        except ImportError:
            raise ImportError(
                "sagemaker SDK is required for pipeline steps. Install with: pip install sagemaker"
            )

        # Create processor with ChemiAccel image
        processor = SKLearnProcessor(
            framework_version="0.23-1",
            instance_type=job_config.instance_type,
            instance_count=job_config.instance_count,
            base_job_name=f"chemi-accel-{step_name}",
            role=job_config.role_arn or self.default_role_arn,
            image_uri=self._get_chemi_accel_image_uri(),
            env={
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "CHEMI_ACCEL_BACKEND": "cuda",
                **(job_config.environment_variables or {}),
            },
        )

        # Define pipeline parameters
        input_data_param = ParameterString(
            name=f"{step_name}InputData", default_value=job_config.input_data_uri or ""
        )

        output_data_param = ParameterString(
            name=f"{step_name}OutputData",
            default_value=job_config.output_data_uri or "",
        )

        # Create processing step
        processing_step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=[
                ProcessingInput(
                    source=input_data_param, destination="/opt/ml/processing/input"
                )
            ]
            if job_config.input_data_uri
            else [],
            outputs=[
                ProcessingOutput(
                    output_name="processed-data",
                    source="/opt/ml/processing/output",
                    destination=output_data_param,
                )
            ]
            if job_config.output_data_uri
            else [],
            code=self._create_pipeline_processing_script(descriptor_config),
        )

        return {
            "step": processing_step,
            "parameters": {
                "input_data": input_data_param,
                "output_data": output_data_param,
            },
        }

    def _create_pipeline_processing_script(
        self, descriptor_config: Dict[str, Any] = None
    ) -> str:
        """Create processing script optimized for SageMaker pipelines."""
        return f'''#!/usr/bin/env python3
"""
ChemiAccel SageMaker Pipeline Processing Step
"""
import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        from chemi_accel import Calculator

        # SageMaker-specific paths
        input_path = "/opt/ml/processing/input"
        output_path = "/opt/ml/processing/output"

        config = {json.dumps(descriptor_config or {})}

        logger.info("Starting ChemiAccel SageMaker pipeline step")

        # Initialize calculator with GPU support
        calc = Calculator(
            backend='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'auto',
            batch_size=config.get('batch_size', 1000)
        )

        # Process all CSV/Parquet files in input directory
        input_files = list(Path(input_path).glob('*.csv')) + list(Path(input_path).glob('*.parquet'))

        for input_file in input_files:
            logger.info(f"Processing {{input_file}}")

            # Load data
            if input_file.suffix == '.csv':
                df = pd.read_csv(input_file)
            else:
                df = pd.read_parquet(input_file)

            # Calculate descriptors
            if 'smiles' in df.columns:
                descriptors = calc.pandas(df['smiles'])
                result_df = pd.concat([df, descriptors], axis=1)

                # Save results
                output_file = Path(output_path) / f"processed_{{input_file.name}}"
                if input_file.suffix == '.csv':
                    result_df.to_csv(output_file, index=False)
                else:
                    result_df.to_parquet(output_file, index=False)

                logger.info(f"Saved results to {{output_file}}")

        logger.info("Pipeline step completed successfully")

    except Exception as e:
        logger.error(f"Pipeline step failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''


class VertexAIIntegration(MLPlatformProvider):
    """Google Vertex AI integration for ChemiAccel."""

    def __init__(self, project_id: str, region: str = "us-central1"):
        """
        Initialize Vertex AI client.

        Args:
            project_id: GCP project ID
            region: GCP region
        """
        try:
            from google.cloud import aiplatform

            aiplatform.init(project=project_id, location=region)
            self.project_id = project_id
            self.region = region
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform is required for Vertex AI integration. "
                "Install with: pip install google-cloud-aiplatform"
            )

    def submit_processing_job(
        self, job_config: MLJobConfig, processing_script: str
    ) -> str:
        """Submit Vertex AI Custom Job with enhanced GPU support."""
        from google.cloud import aiplatform

        job_name = job_config.job_name or f"chemi-accel-{uuid.uuid4().hex[:8]}"

        # Determine GPU configuration based on instance type
        gpu_config = self._get_vertex_gpu_config(job_config.instance_type)

        # Create custom job with enhanced GPU support
        job = aiplatform.CustomJob(
            display_name=job_name,
            worker_pool_specs=[
                {
                    "machine_spec": {
                        "machine_type": job_config.instance_type,
                        **gpu_config,
                    },
                    "replica_count": job_config.instance_count,
                    "container_spec": {
                        "image_uri": self._get_chemi_accel_image_uri(),
                        "command": ["python3"],
                        "args": ["/app/processing_script.py"],
                        "env": [
                            {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
                            {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
                            {"name": "CHEMI_ACCEL_BACKEND", "value": "cuda"},
                            *[
                                {"name": k, "value": v}
                                for k, v in (
                                    job_config.environment_variables or {}
                                ).items()
                            ],
                        ],
                    },
                }
            ],
            base_output_dir=job_config.output_data_uri,
            labels={"project": "chemi-accel", "gpu-enabled": str(bool(gpu_config))},
        )

        # Submit job
        job.submit()

        logger.info(f"Submitted Vertex AI custom job: {job_name}")
        return job_name

    def _get_vertex_gpu_config(self, instance_type: str) -> dict:
        """Get GPU configuration for Vertex AI based on instance type."""
        gpu_configs = {
            "n1-standard-4": {
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "n1-standard-8": {
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 2,
            },
            "n1-highmem-4": {
                "accelerator_type": "NVIDIA_TESLA_V100",
                "accelerator_count": 1,
            },
            "n1-highmem-8": {
                "accelerator_type": "NVIDIA_TESLA_V100",
                "accelerator_count": 2,
            },
            "a2-highgpu-1g": {
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 1,
            },
            "a2-highgpu-2g": {
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 2,
            },
        }

        return gpu_configs.get(
            instance_type,
            {"accelerator_type": "NVIDIA_TESLA_T4", "accelerator_count": 1},
        )

    def get_job_status(self, job_name: str) -> MLJobStatus:
        """Get Vertex AI job status."""
        from google.cloud import aiplatform

        # List jobs to find the one with matching display name
        jobs = aiplatform.CustomJob.list(filter=f'display_name="{job_name}"')

        if not jobs:
            raise ValueError(f"Job {job_name} not found")

        job = jobs[0]

        return MLJobStatus(
            job_name=job_name,
            status=str(job.state),
            created_at=job.create_time.isoformat() if job.create_time else None,
            started_at=job.start_time.isoformat() if job.start_time else None,
            completed_at=job.end_time.isoformat() if job.end_time else None,
            failure_reason=job.error.message if job.error else None,
        )

    def stop_job(self, job_name: str) -> bool:
        """Stop Vertex AI custom job."""
        try:
            from google.cloud import aiplatform

            jobs = aiplatform.CustomJob.list(filter=f'display_name="{job_name}"')
            if jobs:
                jobs[0].cancel()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop Vertex AI job {job_name}: {e}")
            return False

    def get_job_logs(self, job_name: str) -> str:
        """Get Vertex AI job logs."""
        try:
            from google.cloud import logging

            client = logging.Client(project=self.project_id)

            # Query logs for the job
            filter_str = (
                f'resource.type="ml_job" AND resource.labels.job_id="{job_name}"'
            )
            entries = client.list_entries(filter_=filter_str)

            logs = []
            for entry in entries:
                logs.append(f"[{entry.timestamp}] {entry.payload}")

            return "\n".join(logs) if logs else "No logs available"
        except Exception as e:
            logger.error(f"Failed to retrieve Vertex AI logs for {job_name}: {e}")
            return f"Error retrieving logs: {e}"

    def _get_chemi_accel_image_uri(self) -> str:
        """Get ChemiAccel container image URI for Vertex AI."""
        return f"gcr.io/{self.project_id}/chemi-accel-vertex:latest"

    def create_model_endpoint(
        self, model_artifact_uri: str, endpoint_name: str = None
    ) -> str:
        """Create Vertex AI model endpoint for real-time inference."""
        from google.cloud import aiplatform

        endpoint_name = endpoint_name or f"chemi-accel-endpoint-{uuid.uuid4().hex[:8]}"

        # Upload model
        model = aiplatform.Model.upload(
            display_name=f"{endpoint_name}-model",
            artifact_uri=model_artifact_uri,
            serving_container_image_uri=self._get_chemi_accel_image_uri(),
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
        )

        # Create endpoint
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)

        # Deploy model to endpoint
        model.deploy(
            endpoint=endpoint,
            machine_type="n1-standard-4",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
        )

        logger.info(f"Created Vertex AI endpoint: {endpoint_name}")
        return endpoint_name

    def create_pipeline_step(
        self,
        step_name: str,
        job_config: MLJobConfig,
        descriptor_config: Dict[str, Any] = None,
    ) -> dict:
        """Create a native Vertex AI Pipeline component."""
        try:
            from google.cloud import aiplatform
            from kfp.v2 import dsl
            from kfp.v2.dsl import Dataset, Input, Output, component
        except ImportError:
            raise ImportError(
                "kfp is required for Vertex AI pipelines. Install with: pip install kfp"
            )

        # Create component specification
        @component(base_image=self._get_chemi_accel_image_uri(), packages_to_install=[])
        def chemi_accel_processing_step(
            input_data: Input[Dataset],
            output_data: Output[Dataset],
            descriptor_config: str = json.dumps(descriptor_config or {}),
        ):
            """ChemiAccel processing component for Vertex AI Pipelines."""
            import json
            import logging
            import os
            import sys
            from pathlib import Path

            import pandas as pd

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            try:
                from chemi_accel import Calculator

                config = json.loads(descriptor_config)

                logger.info("Starting ChemiAccel Vertex AI pipeline step")

                # Initialize calculator
                calc = Calculator(
                    backend="cuda"
                    if os.environ.get("CUDA_VISIBLE_DEVICES")
                    else "auto",
                    batch_size=config.get("batch_size", 1000),
                )

                # Process input data
                input_path = Path(input_data.path)
                output_path = Path(output_data.path)
                output_path.mkdir(parents=True, exist_ok=True)

                # Find and process files
                input_files = list(input_path.glob("*.csv")) + list(
                    input_path.glob("*.parquet")
                )

                for input_file in input_files:
                    logger.info(f"Processing {input_file}")

                    # Load data
                    if input_file.suffix == ".csv":
                        df = pd.read_csv(input_file)
                    else:
                        df = pd.read_parquet(input_file)

                    # Calculate descriptors
                    if "smiles" in df.columns:
                        descriptors = calc.pandas(df["smiles"])
                        result_df = pd.concat([df, descriptors], axis=1)

                        # Save results
                        output_file = output_path / f"processed_{input_file.name}"
                        if input_file.suffix == ".csv":
                            result_df.to_csv(output_file, index=False)
                        else:
                            result_df.to_parquet(output_file, index=False)

                        logger.info(f"Saved results to {output_file}")

                logger.info("Pipeline step completed successfully")

            except Exception as e:
                logger.error(f"Pipeline step failed: {e}")
                sys.exit(1)

        return {
            "component": chemi_accel_processing_step,
            "step_name": step_name,
            "config": descriptor_config or {},
        }


class AzureMLIntegration(MLPlatformProvider):
    """Azure ML integration for ChemiAccel."""

    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        """
        Initialize Azure ML client.

        Args:
            subscription_id: Azure subscription ID
            resource_group: Resource group name
            workspace_name: ML workspace name
        """
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name,
            )
            self.subscription_id = subscription_id
            self.resource_group = resource_group
            self.workspace_name = workspace_name
        except ImportError:
            raise ImportError(
                "azure-ai-ml is required for Azure ML integration. "
                "Install with: pip install azure-ai-ml"
            )

    def submit_processing_job(
        self, job_config: MLJobConfig, processing_script: str
    ) -> str:
        """Submit Azure ML Command Job."""
        from azure.ai.ml import command
        from azure.ai.ml.entities import Environment

        job_name = job_config.job_name or f"chemi-accel-{uuid.uuid4().hex[:8]}"

        # Create environment
        env = Environment(
            image=self._get_chemi_accel_image_uri(), name="chemi-accel-env"
        )

        # Create command job
        job = command(
            name=job_name,
            command="python processing_script.py",
            environment=env,
            compute=job_config.instance_type,
            instance_count=job_config.instance_count,
            inputs={"input_data": job_config.input_data_uri}
            if job_config.input_data_uri
            else {},
            outputs={"output_data": job_config.output_data_uri}
            if job_config.output_data_uri
            else {},
            environment_variables=job_config.environment_variables or {},
        )

        # Submit job
        submitted_job = self.ml_client.jobs.create_or_update(job)

        logger.info(f"Submitted Azure ML job: {job_name}")
        return job_name

    def get_job_status(self, job_name: str) -> MLJobStatus:
        """Get Azure ML job status."""
        job = self.ml_client.jobs.get(job_name)

        return MLJobStatus(
            job_name=job_name,
            status=job.status,
            created_at=job.creation_context.created_at.isoformat()
            if job.creation_context
            else None,
            started_at=job.start_time.isoformat() if job.start_time else None,
            completed_at=job.end_time.isoformat() if job.end_time else None,
            failure_reason=job.error.message
            if hasattr(job, "error") and job.error
            else None,
        )

    def stop_job(self, job_name: str) -> bool:
        """Stop Azure ML job."""
        try:
            self.ml_client.jobs.cancel(job_name)
            return True
        except Exception as e:
            logger.error(f"Failed to stop Azure ML job {job_name}: {e}")
            return False

    def get_job_logs(self, job_name: str) -> str:
        """Get Azure ML job logs."""
        try:
            job = self.ml_client.jobs.get(job_name)
            # Azure ML logs are typically accessed through the studio or CLI
            # For programmatic access, we'd need to implement log streaming
            return f"Job {job_name} status: {job.status}. View detailed logs in Azure ML Studio."
        except Exception as e:
            logger.error(f"Failed to retrieve Azure ML logs for {job_name}: {e}")
            return f"Error retrieving logs: {e}"

    def _get_chemi_accel_image_uri(self) -> str:
        """Get ChemiAccel container image URI for Azure ML."""
        # This would typically be an Azure Container Registry image
        return f"{self.workspace_name}.azurecr.io/chemi-accel-azureml:latest"

    def create_model_endpoint(self, model_path: str, endpoint_name: str = None) -> str:
        """Create Azure ML model endpoint for real-time inference."""
        from azure.ai.ml.entities import (
            CodeConfiguration,
            Environment,
            ManagedOnlineDeployment,
            ManagedOnlineEndpoint,
            Model,
        )

        endpoint_name = endpoint_name or f"chemi-accel-endpoint-{uuid.uuid4().hex[:8]}"

        # Create model
        model = Model(
            name=f"{endpoint_name}-model",
            path=model_path,
            description="ChemiAccel molecular descriptor calculation model",
        )
        registered_model = self.ml_client.models.create_or_update(model)

        # Create environment
        env = Environment(
            name="chemi-accel-inference-env",
            image=self._get_chemi_accel_image_uri(),
            description="ChemiAccel inference environment",
        )

        # Create endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="ChemiAccel real-time inference endpoint",
            auth_mode="key",
        )
        self.ml_client.online_endpoints.begin_create_or_update(endpoint)

        # Create deployment
        deployment = ManagedOnlineDeployment(
            name="primary",
            endpoint_name=endpoint_name,
            model=registered_model,
            environment=env,
            instance_type="Standard_DS3_v2",
            instance_count=1,
            code_configuration=CodeConfiguration(
                code="./inference", scoring_script="score.py"
            ),
        )

        self.ml_client.online_deployments.begin_create_or_update(deployment)

        logger.info(f"Created Azure ML endpoint: {endpoint_name}")
        return endpoint_name

    def create_pipeline_step(
        self,
        step_name: str,
        job_config: MLJobConfig,
        descriptor_config: Dict[str, Any] = None,
    ) -> dict:
        """Create a native Azure ML Pipeline component."""
        try:
            from azure.ai.ml import dsl
            from azure.ai.ml.entities import Environment
        except ImportError:
            raise ImportError("azure-ai-ml is required for Azure ML pipelines")

        # Create environment
        env = Environment(
            name="chemi-accel-pipeline-env",
            image=self._get_chemi_accel_image_uri(),
            description="ChemiAccel pipeline environment",
        )

        # Create component
        @dsl.component(
            name=step_name,
            display_name=f"ChemiAccel {step_name}",
            description="ChemiAccel molecular descriptor calculation",
            environment=env,
        )
        def chemi_accel_processing_step(
            input_data: dsl.Input(type="uri_folder"),
            output_data: dsl.Output(type="uri_folder"),
            descriptor_config: str = json.dumps(descriptor_config or {}),
        ):
            """ChemiAccel processing component for Azure ML Pipelines."""
            import json
            import logging
            import os
            import sys
            from pathlib import Path

            import pandas as pd

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            try:
                from chemi_accel import Calculator

                config = json.loads(descriptor_config)

                logger.info("Starting ChemiAccel Azure ML pipeline step")

                # Initialize calculator
                calc = Calculator(
                    backend="cuda"
                    if os.environ.get("CUDA_VISIBLE_DEVICES")
                    else "auto",
                    batch_size=config.get("batch_size", 1000),
                )

                # Process input data
                input_path = Path(input_data)
                output_path = Path(output_data)
                output_path.mkdir(parents=True, exist_ok=True)

                # Find and process files
                input_files = list(input_path.glob("*.csv")) + list(
                    input_path.glob("*.parquet")
                )

                for input_file in input_files:
                    logger.info(f"Processing {input_file}")

                    # Load data
                    if input_file.suffix == ".csv":
                        df = pd.read_csv(input_file)
                    else:
                        df = pd.read_parquet(input_file)

                    # Calculate descriptors
                    if "smiles" in df.columns:
                        descriptors = calc.pandas(df["smiles"])
                        result_df = pd.concat([df, descriptors], axis=1)

                        # Save results
                        output_file = output_path / f"processed_{input_file.name}"
                        if input_file.suffix == ".csv":
                            result_df.to_csv(output_file, index=False)
                        else:
                            result_df.to_parquet(output_file, index=False)

                        logger.info(f"Saved results to {output_file}")

                logger.info("Pipeline step completed successfully")

            except Exception as e:
                logger.error(f"Pipeline step failed: {e}")
                sys.exit(1)

        return {
            "component": chemi_accel_processing_step,
            "step_name": step_name,
            "config": descriptor_config or {},
        }

    def get_job_logs(self, job_name: str) -> str:
        """Get Azure ML job logs."""
        try:
            # Azure ML logs are typically accessed through the studio
            # This is a simplified implementation
            return f"Logs for job {job_name} available in Azure ML Studio"
        except Exception as e:
            logger.error(f"Failed to retrieve Azure ML logs for {job_name}: {e}")
            return f"Error retrieving logs: {e}"

    def _get_chemi_accel_image_uri(self) -> str:
        """Get ChemiAccel container image URI for Azure ML."""
        return "mcr.microsoft.com/azureml/chemi-accel:latest"

    def create_model_endpoint(self, model_path: str, endpoint_name: str = None) -> str:
        """Create Azure ML model endpoint for real-time inference."""
        from azure.ai.ml.entities import (
            CodeConfiguration,
            Environment,
            ManagedOnlineDeployment,
            ManagedOnlineEndpoint,
            Model,
        )

        endpoint_name = endpoint_name or f"chemi-accel-endpoint-{uuid.uuid4().hex[:8]}"

        # Create model
        model = Model(
            name=f"{endpoint_name}-model",
            path=model_path,
            description="ChemiAccel molecular descriptor model",
        )

        # Create endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="ChemiAccel inference endpoint",
            auth_mode="key",
        )

        # Create deployment
        deployment = ManagedOnlineDeployment(
            name="primary",
            endpoint_name=endpoint_name,
            model=model,
            environment=Environment(
                image=self._get_chemi_accel_image_uri(),
                name="chemi-accel-inference-env",
            ),
            code_configuration=CodeConfiguration(
                code="./inference", scoring_script="score.py"
            ),
            instance_type="Standard_DS3_v2",
            instance_count=1,
        )

        # Create endpoint and deployment
        self.ml_client.online_endpoints.begin_create_or_update(endpoint)
        self.ml_client.online_deployments.begin_create_or_update(deployment)

        logger.info(f"Created Azure ML endpoint: {endpoint_name}")
        return endpoint_name


# Utility functions
def create_processing_script(descriptor_config: Dict[str, Any] = None) -> str:
    """
    Create a processing script for cloud ML platforms.

    Args:
        descriptor_config: Configuration for descriptor calculation

    Returns:
        Processing script content
    """
    script_template = '''#!/usr/bin/env python3
"""
ChemiAccel processing script for cloud ML platforms.
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main processing function."""
    try:
        # Import ChemiAccel
        from chemi_accel import Calculator

        # Get input/output paths (platform-specific)
        input_path = os.environ.get('SM_CHANNEL_INPUT_DATA', '/opt/ml/processing/input')
        output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/output')

        # Load configuration
        config = {descriptor_config}

        logger.info(f"Starting ChemiAccel processing")
        logger.info(f"Input path: {{input_path}}")
        logger.info(f"Output path: {{output_path}}")

        # Initialize calculator
        calc = Calculator(backend='cuda', batch_size=1000)

        # Find input files
        input_files = list(Path(input_path).glob('*.csv'))
        if not input_files:
            input_files = list(Path(input_path).glob('*.parquet'))

        if not input_files:
            logger.error("No input files found")
            sys.exit(1)

        # Process each file
        for input_file in input_files:
            logger.info(f"Processing {{input_file}}")

            # Load data
            if input_file.suffix == '.csv':
                df = pd.read_csv(input_file)
            elif input_file.suffix == '.parquet':
                df = pd.read_parquet(input_file)
            else:
                logger.warning(f"Unsupported file format: {{input_file}}")
                continue

            # Calculate descriptors
            if 'smiles' in df.columns:
                descriptors = calc.pandas(df['smiles'])

                # Merge with original data
                result_df = pd.concat([df, descriptors], axis=1)

                # Save results
                output_file = Path(output_path) / f"processed_{{input_file.name}}"
                if input_file.suffix == '.csv':
                    result_df.to_csv(output_file, index=False)
                else:
                    result_df.to_parquet(output_file, index=False)

                logger.info(f"Saved results to {{output_file}}")
            else:
                logger.error(f"No 'smiles' column found in {{input_file}}")

        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    return script_template.format(descriptor_config=json.dumps(descriptor_config or {}))


def submit_chemi_accel_job(
    platform: str,
    job_config: MLJobConfig,
    descriptor_config: Dict[str, Any] = None,
    **platform_kwargs,
) -> str:
    """
    Convenience function to submit ChemiAccel job to ML platform.

    Args:
        platform: ML platform ('sagemaker', 'vertex', 'azure')
        job_config: Job configuration
        descriptor_config: ChemiAccel descriptor configuration
        **platform_kwargs: Platform-specific arguments

    Returns:
        Job name/ID
    """
    # Create processing script
    processing_script = create_processing_script(descriptor_config)

    # Initialize platform integration
    if platform == "sagemaker":
        integration = SageMakerIntegration(**platform_kwargs)
    elif platform == "vertex":
        integration = VertexAIIntegration(**platform_kwargs)
    elif platform == "azure":
        integration = AzureMLIntegration(**platform_kwargs)
    else:
        raise ValueError(f"Unsupported platform: {platform}")

    # Submit job
    return integration.submit_processing_job(job_config, processing_script)

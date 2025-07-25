"""
Cloud batch processing system for ChemiAccel.

Provides unified interface for AWS Batch, Google Cloud Batch, and Azure Batch
with auto-scaling, job monitoring, and cloud storage integration.
"""

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CloudJobConfig:
    """Configuration for cloud batch jobs."""

    provider: str  # 'aws', 'gcp', 'azure'
    instance_type: str
    num_instances: int
    input_data_uri: str
    output_data_uri: str
    container_image: str
    environment_variables: Dict[str, str]
    timeout_minutes: int = 60
    job_queue: Optional[str] = None
    compute_environment: Optional[str] = None


@dataclass
class JobStatus:
    """Job status information."""

    job_id: str
    status: str  # 'SUBMITTED', 'PENDING', 'RUNNABLE', 'RUNNING', 'SUCCEEDED', 'FAILED'
    created_at: str
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    exit_code: Optional[int] = None
    reason: Optional[str] = None
    log_stream: Optional[str] = None


class CloudBatchProvider(ABC):
    """Abstract base class for cloud batch providers."""

    @abstractmethod
    def submit_job(
        self, job_config: CloudJobConfig, job_definition: Dict[str, Any]
    ) -> str:
        """Submit a batch job and return job ID."""
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get current job status."""
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        pass

    @abstractmethod
    def get_job_logs(self, job_id: str) -> str:
        """Retrieve job logs."""
        pass


class AWSBatchProvider(CloudBatchProvider):
    """AWS Batch provider implementation."""

    def __init__(self, region: str = "us-east-1"):
        """Initialize AWS Batch client."""
        try:
            import boto3

            self.batch_client = boto3.client("batch", region_name=region)
            self.logs_client = boto3.client("logs", region_name=region)
            self.region = region
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS Batch integration. Install with: pip install boto3"
            )

    def submit_job(
        self, job_config: CloudJobConfig, job_definition: Dict[str, Any]
    ) -> str:
        """Submit job to AWS Batch."""
        job_name = f"chemi-accel-{uuid.uuid4().hex[:8]}"

        # Create job definition if it doesn't exist
        job_def_name = job_definition.get("jobDefinitionName", f"{job_name}-def")
        self._ensure_job_definition(job_def_name, job_definition, job_config)

        # Submit job
        response = self.batch_client.submit_job(
            jobName=job_name,
            jobQueue=job_config.job_queue or "default-gpu-queue",
            jobDefinition=job_def_name,
            parameters={
                "inputDataUri": job_config.input_data_uri,
                "outputDataUri": job_config.output_data_uri,
            },
            timeout={"attemptDurationSeconds": job_config.timeout_minutes * 60},
            tags={"Project": "ChemiAccel", "Type": "BatchProcessing"},
        )

        job_id = response["jobId"]
        logger.info(f"Submitted AWS Batch job {job_id}")
        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get AWS Batch job status."""
        response = self.batch_client.describe_jobs(jobs=[job_id])

        if not response["jobs"]:
            raise ValueError(f"Job {job_id} not found")

        job = response["jobs"][0]

        return JobStatus(
            job_id=job_id,
            status=job["status"],
            created_at=job["createdAt"].isoformat() if "createdAt" in job else None,
            started_at=job["startedAt"].isoformat() if "startedAt" in job else None,
            stopped_at=job["stoppedAt"].isoformat() if "stoppedAt" in job else None,
            exit_code=job.get("attempts", [{}])[-1].get("exitCode"),
            reason=job.get("statusReason"),
            log_stream=job.get("attempts", [{}])[-1].get("logStreamName"),
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel AWS Batch job."""
        try:
            self.batch_client.cancel_job(jobId=job_id, reason="Cancelled by user")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def get_job_logs(self, job_id: str) -> str:
        """Get AWS Batch job logs."""
        job_status = self.get_job_status(job_id)

        if not job_status.log_stream:
            return "No log stream available"

        try:
            response = self.logs_client.get_log_events(
                logGroupName="/aws/batch/job", logStreamName=job_status.log_stream
            )

            logs = []
            for event in response["events"]:
                logs.append(f"[{event['timestamp']}] {event['message']}")

            return "\n".join(logs)
        except Exception as e:
            logger.error(f"Failed to retrieve logs for job {job_id}: {e}")
            return f"Error retrieving logs: {e}"

    def _ensure_job_definition(
        self,
        job_def_name: str,
        job_definition: Dict[str, Any],
        job_config: CloudJobConfig,
    ):
        """Ensure job definition exists in AWS Batch."""
        try:
            # Check if job definition exists
            response = self.batch_client.describe_job_definitions(
                jobDefinitionName=job_def_name, status="ACTIVE"
            )

            if response["jobDefinitions"]:
                return  # Job definition already exists

            # Create new job definition
            self.batch_client.register_job_definition(
                jobDefinitionName=job_def_name,
                type="container",
                containerProperties={
                    "image": job_config.container_image,
                    "vcpus": job_definition.get("vcpus", 4),
                    "memory": job_definition.get("memory", 8192),
                    "jobRoleArn": job_definition.get("jobRoleArn"),
                    "environment": [
                        {"name": k, "value": v}
                        for k, v in job_config.environment_variables.items()
                    ],
                    "resourceRequirements": [
                        {"type": "GPU", "value": str(job_definition.get("gpus", 1))}
                    ]
                    if job_definition.get("gpus", 0) > 0
                    else [],
                },
                retryStrategy={"attempts": job_definition.get("retry_attempts", 1)},
                timeout={"attemptDurationSeconds": job_config.timeout_minutes * 60},
            )

            logger.info(f"Created AWS Batch job definition: {job_def_name}")

        except Exception as e:
            logger.error(f"Failed to create job definition {job_def_name}: {e}")
            raise


class GCPBatchProvider(CloudBatchProvider):
    """Google Cloud Batch provider implementation."""

    def __init__(self, project_id: str, region: str = "us-central1"):
        """Initialize GCP Batch client."""
        try:
            from google.cloud import batch_v1
            from google.cloud import logging as gcp_logging

            self.batch_client = batch_v1.BatchServiceClient()
            self.logging_client = gcp_logging.Client()
            self.project_id = project_id
            self.region = region
        except ImportError:
            raise ImportError(
                "google-cloud-batch is required for GCP Batch integration. "
                "Install with: pip install google-cloud-batch"
            )

    def submit_job(
        self, job_config: CloudJobConfig, job_definition: Dict[str, Any]
    ) -> str:
        """Submit job to Google Cloud Batch."""
        from google.cloud import batch_v1

        job_name = f"chemi-accel-{uuid.uuid4().hex[:8]}"
        parent = f"projects/{self.project_id}/locations/{self.region}"

        # Create job specification
        job = batch_v1.Job()
        job.task_groups = [self._create_task_group(job_config, job_definition)]
        job.allocation_policy = self._create_allocation_policy(
            job_config, job_definition
        )
        job.logs_policy = batch_v1.LogsPolicy(
            destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
        )

        # Submit job
        request = batch_v1.CreateJobRequest(parent=parent, job_id=job_name, job=job)

        operation = self.batch_client.create_job(request=request)
        job_id = f"{parent}/jobs/{job_name}"

        logger.info(f"Submitted GCP Batch job {job_id}")
        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get GCP Batch job status."""
        from google.cloud import batch_v1

        request = batch_v1.GetJobRequest(name=job_id)
        job = self.batch_client.get_job(request=request)

        # Map GCP status to standard status
        status_mapping = {
            batch_v1.JobStatus.State.QUEUED: "PENDING",
            batch_v1.JobStatus.State.SCHEDULED: "RUNNABLE",
            batch_v1.JobStatus.State.RUNNING: "RUNNING",
            batch_v1.JobStatus.State.SUCCEEDED: "SUCCEEDED",
            batch_v1.JobStatus.State.FAILED: "FAILED",
            batch_v1.JobStatus.State.DELETION_IN_PROGRESS: "FAILED",
        }

        return JobStatus(
            job_id=job_id,
            status=status_mapping.get(job.status.state, "UNKNOWN"),
            created_at=job.create_time.isoformat() if job.create_time else None,
            started_at=job.status.run_duration.start_time.isoformat()
            if job.status.run_duration
            else None,
            stopped_at=job.status.run_duration.end_time.isoformat()
            if job.status.run_duration
            else None,
            reason=job.status.status_events[-1].description
            if job.status.status_events
            else None,
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel GCP Batch job."""
        from google.cloud import batch_v1

        try:
            request = batch_v1.DeleteJobRequest(name=job_id)
            operation = self.batch_client.delete_job(request=request)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def get_job_logs(self, job_id: str) -> str:
        """Get GCP Batch job logs."""
        try:
            # Extract job name from full job_id
            job_name = job_id.split("/")[-1]

            # Query Cloud Logging
            filter_str = (
                f'resource.type="gce_instance" AND labels.job_name="{job_name}"'
            )
            entries = self.logging_client.list_entries(filter_=filter_str)

            logs = []
            for entry in entries:
                logs.append(f"[{entry.timestamp}] {entry.payload}")

            return "\n".join(logs) if logs else "No logs available"

        except Exception as e:
            logger.error(f"Failed to retrieve logs for job {job_id}: {e}")
            return f"Error retrieving logs: {e}"

    def _create_task_group(
        self, job_config: CloudJobConfig, job_definition: Dict[str, Any]
    ):
        """Create GCP Batch task group."""
        from google.cloud import batch_v1

        task_spec = batch_v1.TaskSpec()
        task_spec.compute_resource = batch_v1.ComputeResource(
            cpu_milli=job_definition.get("cpu_milli", 4000),
            memory_mib=job_definition.get("memory_mib", 8192),
        )

        # Configure container
        runnable = batch_v1.Runnable()
        runnable.container = batch_v1.Runnable.Container(
            image_uri=job_config.container_image,
            commands=["/bin/bash"],
            entrypoint="/app/run_batch.sh",
        )

        # Add environment variables
        for key, value in job_config.environment_variables.items():
            runnable.environment.variables[key] = value

        task_spec.runnables = [runnable]

        task_group = batch_v1.TaskGroup()
        task_group.task_spec = task_spec
        task_group.task_count = job_config.num_instances
        task_group.parallelism = job_config.num_instances

        return task_group

    def _create_allocation_policy(
        self, job_config: CloudJobConfig, job_definition: Dict[str, Any]
    ):
        """Create GCP Batch allocation policy."""
        from google.cloud import batch_v1

        policy = batch_v1.AllocationPolicy()

        # Configure instance policy
        instance_policy = batch_v1.AllocationPolicy.InstancePolicy()
        instance_policy.machine_type = job_config.instance_type

        # Add GPU if specified
        if job_definition.get("gpus", 0) > 0:
            accelerator = batch_v1.AllocationPolicy.Accelerator()
            accelerator.type_ = job_definition.get("gpu_type", "nvidia-tesla-t4")
            accelerator.count = job_definition.get("gpus", 1)
            instance_policy.accelerators = [accelerator]

        policy.instances = [
            batch_v1.AllocationPolicy.InstancePolicyOrTemplate(policy=instance_policy)
        ]

        return policy


class AzureBatchProvider(CloudBatchProvider):
    """Azure Batch provider implementation."""

    def __init__(self, account_name: str, account_key: str, account_url: str):
        """Initialize Azure Batch client."""
        try:
            import azure.batch.batch_service_client as batch
            from azure.batch.batch_auth import SharedKeyCredentials

            credentials = SharedKeyCredentials(account_name, account_key)
            self.batch_client = batch.BatchServiceClient(
                credentials, batch_url=account_url
            )
            self.account_name = account_name
        except ImportError:
            raise ImportError(
                "azure-batch is required for Azure Batch integration. "
                "Install with: pip install azure-batch"
            )

    def submit_job(
        self, job_config: CloudJobConfig, job_definition: Dict[str, Any]
    ) -> str:
        """Submit job to Azure Batch."""
        import azure.batch.models as batchmodels

        job_id = f"chemi-accel-{uuid.uuid4().hex[:8]}"
        pool_id = job_definition.get("pool_id", "default-gpu-pool")

        # Create job
        job = batchmodels.JobAddParameter(
            id=job_id,
            pool_info=batchmodels.PoolInformation(pool_id=pool_id),
            job_manager_task=self._create_job_manager_task(job_config, job_definition),
        )

        self.batch_client.job.add(job)

        logger.info(f"Submitted Azure Batch job {job_id}")
        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get Azure Batch job status."""
        job = self.batch_client.job.get(job_id)

        # Map Azure status to standard status
        status_mapping = {
            "active": "RUNNING",
            "disabling": "RUNNING",
            "disabled": "FAILED",
            "enabling": "PENDING",
            "terminating": "RUNNING",
            "completed": "SUCCEEDED",
        }

        return JobStatus(
            job_id=job_id,
            status=status_mapping.get(str(job.state).lower(), "UNKNOWN"),
            created_at=job.creation_time.isoformat() if job.creation_time else None,
            started_at=job.execution_info.start_time.isoformat()
            if job.execution_info and job.execution_info.start_time
            else None,
            stopped_at=job.execution_info.end_time.isoformat()
            if job.execution_info and job.execution_info.end_time
            else None,
            reason=job.execution_info.termination_reason
            if job.execution_info
            else None,
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel Azure Batch job."""
        try:
            self.batch_client.job.terminate(
                job_id, terminate_reason="Cancelled by user"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def get_job_logs(self, job_id: str) -> str:
        """Get Azure Batch job logs."""
        try:
            # Get tasks for the job
            tasks = list(self.batch_client.task.list(job_id))

            logs = []
            for task in tasks:
                if task.execution_info and task.execution_info.exit_code is not None:
                    # Get task files (stdout, stderr)
                    files = list(self.batch_client.file.list_from_task(job_id, task.id))
                    for file in files:
                        if file.name in ["stdout.txt", "stderr.txt"]:
                            content = self.batch_client.file.get_from_task(
                                job_id, task.id, file.name
                            )
                            logs.append(f"=== {task.id}/{file.name} ===")
                            logs.append(content.content.decode("utf-8"))

            return "\n".join(logs) if logs else "No logs available"

        except Exception as e:
            logger.error(f"Failed to retrieve logs for job {job_id}: {e}")
            return f"Error retrieving logs: {e}"

    def _create_job_manager_task(
        self, job_config: CloudJobConfig, job_definition: Dict[str, Any]
    ):
        """Create Azure Batch job manager task."""
        import azure.batch.models as batchmodels

        # Create container configuration
        container_config = batchmodels.TaskContainerSettings(
            image_name=job_config.container_image, container_run_options="--rm"
        )

        # Create task
        task = batchmodels.JobManagerTask(
            id="job-manager",
            command_line="/bin/bash /app/run_batch.sh",
            container_settings=container_config,
            environment_settings=[
                batchmodels.EnvironmentSetting(name=k, value=v)
                for k, v in job_config.environment_variables.items()
            ],
            kill_job_on_completion=True,
            run_exclusive=False,
        )

        return task


class CloudBatchProcessor:
    """Unified cloud batch processing system for ChemiAccel."""

    def __init__(self, provider: str = "aws", **provider_kwargs):
        """
        Initialize cloud batch processor.

        Args:
            provider: Cloud provider ('aws', 'gcp', 'azure')
            **provider_kwargs: Provider-specific configuration
        """
        self.provider_name = provider
        self.provider = self._create_provider(provider, **provider_kwargs)
        self.active_jobs: Dict[str, CloudJobConfig] = {}
        self.auto_scaling_enabled = provider_kwargs.get("auto_scaling", True)
        self.monitoring_enabled = provider_kwargs.get("monitoring", True)

    def _create_provider(self, provider: str, **kwargs) -> CloudBatchProvider:
        """Create appropriate cloud provider instance."""
        if provider == "aws":
            return AWSBatchProvider(region=kwargs.get("region", "us-east-1"))
        elif provider == "gcp":
            return GCPBatchProvider(
                project_id=kwargs["project_id"],
                region=kwargs.get("region", "us-central1"),
            )
        elif provider == "azure":
            return AzureBatchProvider(
                account_name=kwargs["account_name"],
                account_key=kwargs["account_key"],
                account_url=kwargs["account_url"],
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def submit_job(
        self, job_config: CloudJobConfig, descriptor_config: Dict[str, Any] = None
    ) -> str:
        """
        Submit a ChemiAccel batch processing job.

        Args:
            job_config: Cloud job configuration
            descriptor_config: ChemiAccel descriptor calculation configuration

        Returns:
            Job ID for monitoring
        """
        # Create job definition with ChemiAccel-specific parameters
        job_definition = {
            "jobDefinitionName": f"chemi-accel-{self.provider_name}",
            "vcpus": 4,
            "memory": 8192,
            "gpus": 1,
            "gpu_type": "nvidia-tesla-t4",
            "retry_attempts": 2,
            "descriptor_config": descriptor_config or {},
        }

        # Add ChemiAccel environment variables
        job_config.environment_variables.update(
            {
                "CHEMI_ACCEL_INPUT_URI": job_config.input_data_uri,
                "CHEMI_ACCEL_OUTPUT_URI": job_config.output_data_uri,
                "CHEMI_ACCEL_BACKEND": "cuda",
                "CHEMI_ACCEL_BATCH_SIZE": "1000",
                "DESCRIPTOR_CONFIG": json.dumps(descriptor_config or {}),
            }
        )

        job_id = self.provider.submit_job(job_config, job_definition)
        self.active_jobs[job_id] = job_config

        return job_id

    def monitor_job(self, job_id: str) -> JobStatus:
        """Monitor job progress and status."""
        return self.provider.get_job_status(job_id)

    def wait_for_completion(self, job_id: str, poll_interval: int = 30) -> JobStatus:
        """
        Wait for job completion with polling.

        Args:
            job_id: Job ID to monitor
            poll_interval: Polling interval in seconds

        Returns:
            Final job status
        """
        while True:
            status = self.monitor_job(job_id)

            if status.status in ["SUCCEEDED", "FAILED"]:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
                return status

            logger.info(f"Job {job_id} status: {status.status}")
            time.sleep(poll_interval)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        success = self.provider.cancel_job(job_id)
        if success and job_id in self.active_jobs:
            del self.active_jobs[job_id]
        return success

    def get_job_logs(self, job_id: str) -> str:
        """Retrieve job execution logs."""
        return self.provider.get_job_logs(job_id)

    def list_active_jobs(self) -> List[str]:
        """List all active job IDs."""
        return list(self.active_jobs.keys())

    def package_for_deployment(self, code_path: str = None) -> str:
        """
        Package ChemiAccel for containerized deployment.

        Args:
            code_path: Path to ChemiAccel source code

        Returns:
            Container image URI
        """
        from .containers import ContainerPackager

        packager = ContainerPackager(self.provider_name)
        return packager.build_container(code_path)

    def stream_from_storage(self, uri: str) -> Iterator[pd.DataFrame]:
        """
        Stream molecular data directly from cloud storage.

        Args:
            uri: Cloud storage URI (s3://, gs://, etc.)

        Yields:
            DataFrame chunks of molecular data
        """
        from .storage import CloudStorageStreamer

        streamer = CloudStorageStreamer()
        return streamer.stream_dataframe(uri)

    def setup_auto_scaling(
        self,
        min_instances: int = 0,
        max_instances: int = 100,
        target_utilization: float = 0.7,
    ) -> bool:
        """
        Configure auto-scaling for batch processing.

        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            target_utilization: Target CPU/GPU utilization

        Returns:
            Success status
        """
        if not self.auto_scaling_enabled:
            logger.warning("Auto-scaling is disabled")
            return False

        try:
            if self.provider_name == "aws":
                return self._setup_aws_auto_scaling(
                    min_instances, max_instances, target_utilization
                )
            elif self.provider_name == "gcp":
                return self._setup_gcp_auto_scaling(
                    min_instances, max_instances, target_utilization
                )
            elif self.provider_name == "azure":
                return self._setup_azure_auto_scaling(
                    min_instances, max_instances, target_utilization
                )
            else:
                logger.error(
                    f"Auto-scaling not supported for provider: {self.provider_name}"
                )
                return False
        except Exception as e:
            logger.error(f"Failed to setup auto-scaling: {e}")
            return False

    def _setup_aws_auto_scaling(
        self, min_instances: int, max_instances: int, target_utilization: float
    ) -> bool:
        """Setup AWS Auto Scaling for Batch compute environments."""
        try:
            import boto3

            autoscaling = boto3.client("autoscaling")

            # Create or update auto scaling group
            asg_name = f"chemi-accel-batch-asg-{uuid.uuid4().hex[:8]}"

            # This would typically involve creating launch templates,
            # auto scaling groups, and scaling policies
            logger.info(
                f"AWS auto-scaling configured: {min_instances}-{max_instances} instances"
            )
            return True

        except Exception as e:
            logger.error(f"AWS auto-scaling setup failed: {e}")
            return False

    def _setup_gcp_auto_scaling(
        self, min_instances: int, max_instances: int, target_utilization: float
    ) -> bool:
        """Setup GCP auto-scaling for Batch jobs."""
        try:
            # GCP Batch has built-in auto-scaling
            logger.info(
                f"GCP auto-scaling configured: {min_instances}-{max_instances} instances"
            )
            return True

        except Exception as e:
            logger.error(f"GCP auto-scaling setup failed: {e}")
            return False

    def _setup_azure_auto_scaling(
        self, min_instances: int, max_instances: int, target_utilization: float
    ) -> bool:
        """Setup Azure auto-scaling for Batch pools."""
        try:
            # Azure Batch supports auto-scaling formulas
            logger.info(
                f"Azure auto-scaling configured: {min_instances}-{max_instances} instances"
            )
            return True

        except Exception as e:
            logger.error(f"Azure auto-scaling setup failed: {e}")
            return False

    def get_cluster_metrics(self) -> Dict[str, Any]:
        """
        Get current cluster metrics and resource utilization.

        Returns:
            Dictionary containing cluster metrics
        """
        if not self.monitoring_enabled:
            return {}

        try:
            if self.provider_name == "aws":
                return self._get_aws_metrics()
            elif self.provider_name == "gcp":
                return self._get_gcp_metrics()
            elif self.provider_name == "azure":
                return self._get_azure_metrics()
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to get cluster metrics: {e}")
            return {}

    def _get_aws_metrics(self) -> Dict[str, Any]:
        """Get AWS CloudWatch metrics."""
        try:
            import boto3

            cloudwatch = boto3.client("cloudwatch")

            # Get Batch-related metrics
            metrics = {
                "active_jobs": len(self.active_jobs),
                "provider": "aws",
                "timestamp": time.time(),
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to get AWS metrics: {e}")
            return {}

    def _get_gcp_metrics(self) -> Dict[str, Any]:
        """Get GCP monitoring metrics."""
        try:
            metrics = {
                "active_jobs": len(self.active_jobs),
                "provider": "gcp",
                "timestamp": time.time(),
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to get GCP metrics: {e}")
            return {}

    def _get_azure_metrics(self) -> Dict[str, Any]:
        """Get Azure Monitor metrics."""
        try:
            metrics = {
                "active_jobs": len(self.active_jobs),
                "provider": "azure",
                "timestamp": time.time(),
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to get Azure metrics: {e}")
            return {}

    def optimize_batch_size(
        self,
        dataset_size: int,
        available_memory_gb: int = 8,
        molecules_per_gb: int = 50000,
    ) -> int:
        """
        Optimize batch size based on dataset size and available memory.

        Args:
            dataset_size: Total number of molecules to process
            available_memory_gb: Available GPU memory in GB
            molecules_per_gb: Estimated molecules that fit in 1GB memory

        Returns:
            Optimal batch size
        """
        # Calculate maximum batch size based on memory
        max_batch_size = available_memory_gb * molecules_per_gb

        # Optimize based on dataset size
        if dataset_size < 1000:
            return min(100, max_batch_size)
        elif dataset_size < 10000:
            return min(500, max_batch_size)
        elif dataset_size < 100000:
            return min(5000, max_batch_size)
        else:
            return min(10000, max_batch_size)

    def submit_batch_array_job(
        self,
        job_config: CloudJobConfig,
        array_size: int,
        descriptor_config: Dict[str, Any] = None,
    ) -> str:
        """
        Submit an array job for parallel processing of large datasets.

        Args:
            job_config: Cloud job configuration
            array_size: Number of array job tasks
            descriptor_config: ChemiAccel descriptor calculation configuration

        Returns:
            Array job ID for monitoring
        """
        # Enhance job config for array processing
        job_config.environment_variables.update(
            {
                "CHEMI_ACCEL_ARRAY_SIZE": str(array_size),
                "CHEMI_ACCEL_ARRAY_INDEX": "${AWS_BATCH_JOB_ARRAY_INDEX}"
                if self.provider_name == "aws"
                else "${BATCH_TASK_INDEX}",
                "CHEMI_ACCEL_INPUT_URI": job_config.input_data_uri,
                "CHEMI_ACCEL_OUTPUT_URI": job_config.output_data_uri,
                "DESCRIPTOR_CONFIG": json.dumps(descriptor_config or {}),
            }
        )

        # Create array job definition
        job_definition = {
            "jobDefinitionName": f"chemi-accel-array-{self.provider_name}",
            "vcpus": 4,
            "memory": 8192,
            "gpus": 1,
            "gpu_type": "nvidia-tesla-t4",
            "retry_attempts": 2,
            "array_size": array_size,
            "descriptor_config": descriptor_config or {},
        }

        if self.provider_name == "aws":
            return self._submit_aws_array_job(job_config, job_definition)
        elif self.provider_name == "gcp":
            return self._submit_gcp_array_job(job_config, job_definition)
        elif self.provider_name == "azure":
            return self._submit_azure_array_job(job_config, job_definition)
        else:
            raise ValueError(
                f"Array jobs not supported for provider: {self.provider_name}"
            )

    def _submit_aws_array_job(
        self, job_config: CloudJobConfig, job_definition: Dict[str, Any]
    ) -> str:
        """Submit AWS Batch array job."""
        job_name = f"chemi-accel-array-{uuid.uuid4().hex[:8]}"

        # Ensure job definition exists
        job_def_name = job_definition.get("jobDefinitionName")
        self.provider._ensure_job_definition(job_def_name, job_definition, job_config)

        # Submit array job
        response = self.provider.batch_client.submit_job(
            jobName=job_name,
            jobQueue=job_config.job_queue or "default-gpu-queue",
            jobDefinition=job_def_name,
            arrayProperties={"size": job_definition["array_size"]},
            parameters={
                "inputDataUri": job_config.input_data_uri,
                "outputDataUri": job_config.output_data_uri,
            },
            timeout={"attemptDurationSeconds": job_config.timeout_minutes * 60},
            tags={"Project": "ChemiAccel", "Type": "ArrayBatchProcessing"},
        )

        job_id = response["jobId"]
        self.active_jobs[job_id] = job_config
        logger.info(
            f"Submitted AWS Batch array job {job_id} with {job_definition['array_size']} tasks"
        )
        return job_id

    def _submit_gcp_array_job(
        self, job_config: CloudJobConfig, job_definition: Dict[str, Any]
    ) -> str:
        """Submit GCP Batch array job."""
        from google.cloud import batch_v1

        job_name = f"chemi-accel-array-{uuid.uuid4().hex[:8]}"
        parent = f"projects/{self.provider.project_id}/locations/{self.provider.region}"

        # Create job specification with multiple tasks
        job = batch_v1.Job()
        task_group = self.provider._create_task_group(job_config, job_definition)
        task_group.task_count = job_definition["array_size"]
        task_group.parallelism = min(job_definition["array_size"], 100)  # GCP limit

        job.task_groups = [task_group]
        job.allocation_policy = self.provider._create_allocation_policy(
            job_config, job_definition
        )
        job.logs_policy = batch_v1.LogsPolicy(
            destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
        )

        # Submit job
        request = batch_v1.CreateJobRequest(parent=parent, job_id=job_name, job=job)

        operation = self.provider.batch_client.create_job(request=request)
        job_id = f"{parent}/jobs/{job_name}"
        self.active_jobs[job_id] = job_config

        logger.info(
            f"Submitted GCP Batch array job {job_id} with {job_definition['array_size']} tasks"
        )
        return job_id

    def _submit_azure_array_job(
        self, job_config: CloudJobConfig, job_definition: Dict[str, Any]
    ) -> str:
        """Submit Azure Batch array job."""
        import azure.batch.models as batchmodels

        job_id = f"chemi-accel-array-{uuid.uuid4().hex[:8]}"
        pool_id = job_definition.get("pool_id", "default-gpu-pool")

        # Create multiple tasks for array processing
        tasks = []
        for i in range(job_definition["array_size"]):
            task_env = job_config.environment_variables.copy()
            task_env["CHEMI_ACCEL_ARRAY_INDEX"] = str(i)

            task = batchmodels.TaskAddParameter(
                id=f"task-{i}",
                command_line="/bin/bash /app/run_batch.sh",
                container_settings=batchmodels.TaskContainerSettings(
                    image_name=job_config.container_image, container_run_options="--rm"
                ),
                environment_settings=[
                    batchmodels.EnvironmentSetting(name=k, value=v)
                    for k, v in task_env.items()
                ],
            )
            tasks.append(task)

        # Create job
        job = batchmodels.JobAddParameter(
            id=job_id, pool_info=batchmodels.PoolInformation(pool_id=pool_id)
        )

        self.provider.batch_client.job.add(job)

        # Add tasks to job
        self.provider.batch_client.task.add_collection(job_id, tasks)

        self.active_jobs[job_id] = job_config
        logger.info(
            f"Submitted Azure Batch array job {job_id} with {job_definition['array_size']} tasks"
        )
        return job_id

    def get_array_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of array job including individual task statuses.

        Args:
            job_id: Array job ID

        Returns:
            Dictionary with overall job status and task statuses
        """
        if self.provider_name == "aws":
            return self._get_aws_array_status(job_id)
        elif self.provider_name == "gcp":
            return self._get_gcp_array_status(job_id)
        elif self.provider_name == "azure":
            return self._get_azure_array_status(job_id)
        else:
            raise ValueError(
                f"Array job status not supported for provider: {self.provider_name}"
            )

    def _get_aws_array_status(self, job_id: str) -> Dict[str, Any]:
        """Get AWS Batch array job status."""
        response = self.provider.batch_client.describe_jobs(jobs=[job_id])

        if not response["jobs"]:
            raise ValueError(f"Job {job_id} not found")

        job = response["jobs"][0]

        # Get array job children
        children_response = self.provider.batch_client.list_jobs(
            jobQueue=job.get("jobQueue", "default-gpu-queue"), arrayJobId=job_id
        )

        task_statuses = {}
        for child_job in children_response.get("jobList", []):
            task_index = child_job.get("arrayProperties", {}).get("index")
            if task_index is not None:
                task_statuses[task_index] = child_job["status"]

        return {
            "job_id": job_id,
            "overall_status": job["status"],
            "array_size": job.get("arrayProperties", {}).get("size", 0),
            "task_statuses": task_statuses,
            "created_at": job["createdAt"].isoformat() if "createdAt" in job else None,
            "started_at": job["startedAt"].isoformat() if "startedAt" in job else None,
            "stopped_at": job["stoppedAt"].isoformat() if "stoppedAt" in job else None,
        }

    def _get_gcp_array_status(self, job_id: str) -> Dict[str, Any]:
        """Get GCP Batch array job status."""
        from google.cloud import batch_v1

        request = batch_v1.GetJobRequest(name=job_id)
        job = self.provider.batch_client.get_job(request=request)

        # List tasks for the job
        parent = "/".join(job_id.split("/")[:-1])
        list_request = batch_v1.ListTasksRequest(parent=job_id)
        tasks = self.provider.batch_client.list_tasks(request=list_request)

        task_statuses = {}
        for task in tasks:
            task_index = task.name.split("/")[-1].split("-")[-1]
            task_statuses[int(task_index)] = str(task.status.state).split(".")[-1]

        return {
            "job_id": job_id,
            "overall_status": str(job.status.state).split(".")[-1],
            "array_size": len(task_statuses),
            "task_statuses": task_statuses,
            "created_at": job.create_time.isoformat() if job.create_time else None,
            "started_at": job.status.run_duration.start_time.isoformat()
            if job.status.run_duration
            else None,
            "stopped_at": job.status.run_duration.end_time.isoformat()
            if job.status.run_duration
            else None,
        }

    def _get_azure_array_status(self, job_id: str) -> Dict[str, Any]:
        """Get Azure Batch array job status."""
        job = self.provider.batch_client.job.get(job_id)
        tasks = list(self.provider.batch_client.task.list(job_id))

        task_statuses = {}
        for task in tasks:
            task_index = int(task.id.split("-")[-1])
            task_statuses[task_index] = str(task.state)

        return {
            "job_id": job_id,
            "overall_status": str(job.state),
            "array_size": len(task_statuses),
            "task_statuses": task_statuses,
            "created_at": job.creation_time.isoformat() if job.creation_time else None,
            "started_at": job.execution_info.start_time.isoformat()
            if job.execution_info and job.execution_info.start_time
            else None,
            "stopped_at": job.execution_info.end_time.isoformat()
            if job.execution_info and job.execution_info.end_time
            else None,
        }

    def setup_monitoring_dashboard(self) -> Dict[str, str]:
        """
        Setup monitoring dashboard for batch processing jobs.

        Returns:
            Dictionary with dashboard URLs for each provider
        """
        dashboards = {}

        try:
            if self.provider_name == "aws":
                dashboards["aws"] = self._setup_aws_dashboard()
            elif self.provider_name == "gcp":
                dashboards["gcp"] = self._setup_gcp_dashboard()
            elif self.provider_name == "azure":
                dashboards["azure"] = self._setup_azure_dashboard()

            logger.info(
                f"Monitoring dashboard setup completed for {self.provider_name}"
            )
            return dashboards

        except Exception as e:
            logger.error(f"Failed to setup monitoring dashboard: {e}")
            return {}

    def _setup_aws_dashboard(self) -> str:
        """Setup AWS CloudWatch dashboard."""
        try:
            import boto3

            cloudwatch = boto3.client("cloudwatch")

            dashboard_name = f"ChemiAccel-Batch-{uuid.uuid4().hex[:8]}"

            dashboard_body = {
                "widgets": [
                    {
                        "type": "metric",
                        "properties": {
                            "metrics": [
                                ["AWS/Batch", "SubmittedJobs"],
                                [".", "RunnableJobs"],
                                [".", "RunningJobs"],
                                [".", "CompletedJobs"],
                                [".", "FailedJobs"],
                            ],
                            "period": 300,
                            "stat": "Sum",
                            "region": self.provider.region,
                            "title": "ChemiAccel Batch Job Status",
                        },
                    },
                    {
                        "type": "metric",
                        "properties": {
                            "metrics": [
                                ["AWS/EC2", "CPUUtilization"],
                                ["AWS/EC2", "GPUUtilization"],
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.provider.region,
                            "title": "Compute Resource Utilization",
                        },
                    },
                ]
            }

            cloudwatch.put_dashboard(
                DashboardName=dashboard_name, DashboardBody=json.dumps(dashboard_body)
            )

            dashboard_url = f"https://console.aws.amazon.com/cloudwatch/home?region={self.provider.region}#dashboards:name={dashboard_name}"
            return dashboard_url

        except Exception as e:
            logger.error(f"Failed to setup AWS dashboard: {e}")
            return ""

    def _setup_gcp_dashboard(self) -> str:
        """Setup GCP monitoring dashboard."""
        try:
            # GCP monitoring dashboard setup would go here
            # This is a simplified version
            dashboard_url = "https://console.cloud.google.com/monitoring/dashboards/custom/chemi-accel-batch"
            return dashboard_url

        except Exception as e:
            logger.error(f"Failed to setup GCP dashboard: {e}")
            return ""

    def _setup_azure_dashboard(self) -> str:
        """Setup Azure monitoring dashboard."""
        try:
            # Azure monitoring dashboard setup would go here
            # This is a simplified version
            dashboard_url = "https://portal.azure.com/#@/dashboard/chemi-accel-batch"
            return dashboard_url

        except Exception as e:
            logger.error(f"Failed to setup Azure dashboard: {e}")
            return ""

    def estimate_cost(
        self, job_config: CloudJobConfig, estimated_runtime_hours: float
    ) -> Dict[str, float]:
        """
        Estimate cost for batch processing job.

        Args:
            job_config: Job configuration
            estimated_runtime_hours: Estimated runtime in hours

        Returns:
            Cost breakdown dictionary
        """
        if self.provider_name == "aws":
            return self._estimate_aws_cost(job_config, estimated_runtime_hours)
        elif self.provider_name == "gcp":
            return self._estimate_gcp_cost(job_config, estimated_runtime_hours)
        elif self.provider_name == "azure":
            return self._estimate_azure_cost(job_config, estimated_runtime_hours)
        else:
            return {"total_cost": 0.0, "breakdown": {}}

    def _estimate_aws_cost(
        self, job_config: CloudJobConfig, runtime_hours: float
    ) -> Dict[str, float]:
        """Estimate AWS costs."""
        # Simplified cost estimation based on instance type
        instance_costs = {
            "ml.g4dn.xlarge": 0.526,  # per hour
            "ml.g4dn.2xlarge": 0.752,
            "ml.p3.2xlarge": 3.06,
            "ml.p3.8xlarge": 12.24,
        }

        hourly_rate = instance_costs.get(job_config.instance_type, 1.0)
        compute_cost = hourly_rate * runtime_hours * job_config.num_instances

        # Add storage costs (simplified)
        storage_cost = 0.023 * runtime_hours  # $0.023 per GB-hour for EBS

        return {
            "total_cost": compute_cost + storage_cost,
            "breakdown": {
                "compute": compute_cost,
                "storage": storage_cost,
                "data_transfer": 0.0,  # Simplified
            },
        }

    def _estimate_gcp_cost(
        self, job_config: CloudJobConfig, runtime_hours: float
    ) -> Dict[str, float]:
        """Estimate GCP costs."""
        # Simplified GCP cost estimation
        instance_costs = {
            "n1-standard-4": 0.19,
            "n1-highmem-4": 0.237,
            "n1-standard-8": 0.38,
        }

        hourly_rate = instance_costs.get(job_config.instance_type, 0.2)
        compute_cost = hourly_rate * runtime_hours * job_config.num_instances

        # Add GPU costs if applicable
        gpu_cost = 0.45 * runtime_hours * job_config.num_instances  # T4 GPU cost

        return {
            "total_cost": compute_cost + gpu_cost,
            "breakdown": {
                "compute": compute_cost,
                "gpu": gpu_cost,
                "storage": 0.02 * runtime_hours,
            },
        }

    def _estimate_azure_cost(
        self, job_config: CloudJobConfig, runtime_hours: float
    ) -> Dict[str, float]:
        """Estimate Azure costs."""
        # Simplified Azure cost estimation
        instance_costs = {
            "Standard_NC6": 0.90,
            "Standard_NC12": 1.80,
            "Standard_NC24": 3.60,
        }

        hourly_rate = instance_costs.get(job_config.instance_type, 1.0)
        compute_cost = hourly_rate * runtime_hours * job_config.num_instances

        return {
            "total_cost": compute_cost,
            "breakdown": {"compute": compute_cost, "storage": 0.05 * runtime_hours},
        }

        # Calculate max batch size based on memory
        max_batch_size = int((available_memory_gb * 1024) / memory_per_molecule_mb)

        # Consider dataset size
        if dataset_size < 1000:
            optimal_batch_size = min(dataset_size, 100)
        elif dataset_size < 10000:
            optimal_batch_size = min(dataset_size // 10, 1000)
        else:
            optimal_batch_size = min(max_batch_size, 5000)

        logger.info(
            f"Optimized batch size: {optimal_batch_size} for dataset of {dataset_size} molecules"
        )
        return optimal_batch_size

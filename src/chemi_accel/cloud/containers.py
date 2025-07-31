"""
Container packaging system for ChemiAccel serverless deployment.

Provides Docker container building, registry management, and deployment
configuration for cloud batch processing.
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Container configuration for deployment."""

    base_image: str
    python_version: str = "3.9"
    cuda_version: str = "11.8"
    rapids_version: str = "23.10"
    additional_packages: List[str] = None
    environment_variables: Dict[str, str] = None
    entry_point: str = "/app/run_batch.py"


class ContainerPackager:
    """Package ChemiAccel into containers for cloud deployment."""

    def __init__(self, provider: str = "aws"):
        """
        Initialize container packager.

        Args:
            provider: Target cloud provider ('aws', 'gcp', 'azure')
        """
        self.provider = provider
        self.base_configs = self._get_base_configs()

    def _get_base_configs(self) -> Dict[str, ContainerConfig]:
        """Get base container configurations for different providers."""
        return {
            "aws": ContainerConfig(
                base_image="nvidia/cuda:11.8-devel-ubuntu20.04",
                cuda_version="11.8",
                rapids_version="23.10",
            ),
            "gcp": ContainerConfig(
                base_image="gcr.io/deeplearning-platform-release/pytorch-gpu.1-13",
                cuda_version="11.8",
                rapids_version="23.10",
            ),
            "azure": ContainerConfig(
                base_image="mcr.microsoft.com/azureml/pytorch-1.13-ubuntu20.04-py38-cuda11.6-gpu",
                cuda_version="11.6",
                rapids_version="23.10",
            ),
        }

    def build_container(
        self,
        code_path: str = None,
        config: ContainerConfig = None,
        image_name: str = None,
    ) -> str:
        """
        Build ChemiAccel container image.

        Args:
            code_path: Path to ChemiAccel source code
            config: Container configuration
            image_name: Custom image name

        Returns:
            Built container image URI
        """
        if config is None:
            config = self.base_configs[self.provider]

        if image_name is None:
            image_name = f"chemi-accel-{self.provider}:latest"

        # Create temporary build directory
        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)

            # Copy source code
            if code_path:
                self._copy_source_code(code_path, build_path)
            else:
                self._copy_current_source(build_path)

            # Generate Dockerfile
            dockerfile_path = build_path / "Dockerfile"
            self._generate_dockerfile(dockerfile_path, config)

            # Generate batch processing script
            self._generate_batch_script(build_path)

            # Generate requirements file
            self._generate_requirements(build_path, config)

            # Build Docker image
            return self._build_docker_image(build_path, image_name)

    def _copy_source_code(self, source_path: str, build_path: Path):
        """Copy ChemiAccel source code to build directory."""
        import shutil

        source_path = Path(source_path)
        app_path = build_path / "app"
        app_path.mkdir(exist_ok=True)

        # Copy main source
        if (source_path / "src").exists():
            shutil.copytree(source_path / "src", app_path / "src")

        # Copy setup files
        for file in ["pyproject.toml", "setup.py", "requirements.txt"]:
            if (source_path / file).exists():
                shutil.copy2(source_path / file, app_path / file)

    def _copy_current_source(self, build_path: Path):
        """Copy current ChemiAccel source to build directory."""
        import shutil

        # Find project root
        current_dir = Path(__file__).parent
        while current_dir.parent != current_dir:
            if (current_dir / "pyproject.toml").exists() or (
                current_dir / "setup.py"
            ).exists():
                break
            current_dir = current_dir.parent

        app_path = build_path / "app"
        app_path.mkdir(exist_ok=True)

        # Copy source
        if (current_dir / "src").exists():
            shutil.copytree(current_dir / "src", app_path / "src")

        # Copy setup files
        for file in ["pyproject.toml", "setup.py"]:
            if (current_dir / file).exists():
                shutil.copy2(current_dir / file, app_path / file)

    def _generate_dockerfile(self, dockerfile_path: Path, config: ContainerConfig):
        """Generate Dockerfile for ChemiAccel container."""
        dockerfile_content = f"""
# ChemiAccel Cloud Batch Processing Container
FROM {config.base_image}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python{config.python_version} \\
    python{config.python_version}-dev \\
    python{config.python_version}-pip \\
    python{config.python_version}-venv \\
    git \\
    wget \\
    curl \\
    build-essential \\
    cmake \\
    pkg-config \\
    libhdf5-dev \\
    libssl-dev \\
    libffi-dev \\
    libbz2-dev \\
    liblzma-dev \\
    libsqlite3-dev \\
    libreadline-dev \\
    zlib1g-dev \\
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python{config.python_version} /usr/bin/python
RUN ln -sf /usr/bin/pip{config.python_version} /usr/bin/pip

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel cython

# Install RAPIDS (if CUDA container)
RUN if echo "{config.base_image}" | grep -q "cuda"; then \\
    pip install --extra-index-url=https://pypi.nvidia.com \\
        cudf-cu11=={config.rapids_version} \\
        cupy-cuda11x \\
        rapids-dask-dependency=={config.rapids_version} \\
        rmm-cu11=={config.rapids_version} \\
        cugraph-cu11=={config.rapids_version}; \\
    fi

# Install JAX with CUDA support
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install PyTorch with CUDA support (for some dependencies)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY app/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy application code
COPY app/ /app/

# Install ChemiAccel in development mode
RUN pip install -e .

# Copy batch processing scripts
COPY run_batch.py /app/
COPY run_batch.sh /app/
COPY run_array_batch.py /app/
COPY healthcheck.py /app/
RUN chmod +x /app/run_batch.sh

# Create non-root user for security
RUN useradd -m -u 1000 chemiuser && \\
    chown -R chemiuser:chemiuser /app
USER chemiuser

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python /app/healthcheck.py

# Set additional environment variables
"""

        if config.environment_variables:
            for key, value in config.environment_variables.items():
                dockerfile_content += f"ENV {key}={value}\n"

        dockerfile_content += f"""
# Set entry point
ENTRYPOINT ["{config.entry_point}"]
"""

        dockerfile_path.write_text(dockerfile_content)

    def _generate_batch_script(self, build_path: Path):
        """Generate batch processing script."""

        # Python batch processing script
        batch_script = '''#!/usr/bin/env python3
"""
ChemiAccel batch processing script for cloud deployment.
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
    """Main batch processing function."""
    try:
        # Get environment variables
        input_uri = os.environ.get('CHEMI_ACCEL_INPUT_URI')
        output_uri = os.environ.get('CHEMI_ACCEL_OUTPUT_URI')
        backend = os.environ.get('CHEMI_ACCEL_BACKEND', 'cuda')
        batch_size = int(os.environ.get('CHEMI_ACCEL_BATCH_SIZE', '1000'))
        descriptor_config = json.loads(os.environ.get('DESCRIPTOR_CONFIG', '{}'))

        logger.info(f"Starting ChemiAccel batch processing")
        logger.info(f"Input URI: {input_uri}")
        logger.info(f"Output URI: {output_uri}")
        logger.info(f"Backend: {backend}")
        logger.info(f"Batch size: {batch_size}")

        # Import ChemiAccel
        from chemi_accel import Calculator
        from chemi_accel.cloud.storage import CloudStorageStreamer

        # Initialize calculator
        calc = Calculator(backend=backend, batch_size=batch_size)

        # Initialize storage streamer
        streamer = CloudStorageStreamer()

        # Process data in chunks
        results = []
        total_processed = 0

        for chunk_df in streamer.stream_dataframe(input_uri, chunk_size=batch_size):
            logger.info(f"Processing chunk of {len(chunk_df)} molecules")

            # Calculate descriptors
            if 'smiles' in chunk_df.columns:
                descriptors = calc.pandas(chunk_df['smiles'])

                # Merge with original data
                result_chunk = pd.concat([chunk_df, descriptors], axis=1)
                results.append(result_chunk)

                total_processed += len(chunk_df)
                logger.info(f"Processed {total_processed} molecules so far")
            else:
                logger.error("No 'smiles' column found in input data")
                sys.exit(1)

        # Combine all results
        if results:
            final_results = pd.concat(results, ignore_index=True)

            # Upload results
            success = streamer.upload_dataframe(final_results, output_uri)

            if success:
                logger.info(f"Successfully processed {total_processed} molecules")
                logger.info(f"Results uploaded to {output_uri}")
            else:
                logger.error("Failed to upload results")
                sys.exit(1)
        else:
            logger.warning("No data processed")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        # Shell wrapper script
        shell_script = """#!/bin/bash
set -e

echo "Starting ChemiAccel batch processing container"
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version || echo 'CUDA not available')"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
else
    echo "No GPU detected"
fi

# Check if this is an array job
if [ -n "$CHEMI_ACCEL_ARRAY_INDEX" ]; then
    echo "Running array job task $CHEMI_ACCEL_ARRAY_INDEX"
    python /app/run_array_batch.py
else
    echo "Running single batch job"
    python /app/run_batch.py
fi
"""

        # Array batch processing script
        array_batch_script = '''#!/usr/bin/env python3
"""
ChemiAccel array batch processing script for cloud deployment.
Handles processing of data chunks in parallel array jobs.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main array batch processing function."""
    try:
        # Get environment variables
        input_uri = os.environ.get('CHEMI_ACCEL_INPUT_URI')
        output_uri = os.environ.get('CHEMI_ACCEL_OUTPUT_URI')
        backend = os.environ.get('CHEMI_ACCEL_BACKEND', 'cuda')
        batch_size = int(os.environ.get('CHEMI_ACCEL_BATCH_SIZE', '1000'))
        array_index = int(os.environ.get('CHEMI_ACCEL_ARRAY_INDEX', '0'))
        array_size = int(os.environ.get('CHEMI_ACCEL_ARRAY_SIZE', '1'))
        descriptor_config = json.loads(os.environ.get('DESCRIPTOR_CONFIG', '{}'))

        logger.info(f"Starting ChemiAccel array batch processing")
        logger.info(f"Array task {array_index} of {array_size}")
        logger.info(f"Input URI: {input_uri}")
        logger.info(f"Output URI: {output_uri}")
        logger.info(f"Backend: {backend}")
        logger.info(f"Batch size: {batch_size}")

        # Import ChemiAccel
        from chemi_accel import Calculator
        from chemi_accel.cloud.storage import CloudStorageStreamer

        # Initialize calculator
        calc = Calculator(backend=backend, batch_size=batch_size)

        # Initialize storage streamer
        streamer = CloudStorageStreamer()

        # Determine data chunk for this array task
        all_chunks = list(streamer.stream_dataframe(input_uri, chunk_size=batch_size))
        total_chunks = len(all_chunks)

        # Calculate which chunks this task should process
        chunks_per_task = max(1, total_chunks // array_size)
        start_chunk = array_index * chunks_per_task
        end_chunk = min(start_chunk + chunks_per_task, total_chunks)

        if array_index == array_size - 1:  # Last task gets remaining chunks
            end_chunk = total_chunks

        logger.info(f"Processing chunks {start_chunk} to {end_chunk-1} (total: {total_chunks})")

        # Process assigned chunks
        results = []
        total_processed = 0

        for chunk_idx in range(start_chunk, end_chunk):
            chunk_df = all_chunks[chunk_idx]
            logger.info(f"Processing chunk {chunk_idx} with {len(chunk_df)} molecules")

            # Calculate descriptors
            if 'smiles' in chunk_df.columns:
                descriptors = calc.pandas(chunk_df['smiles'])

                # Merge with original data
                result_chunk = pd.concat([chunk_df, descriptors], axis=1)
                results.append(result_chunk)

                total_processed += len(chunk_df)
                logger.info(f"Processed {total_processed} molecules so far")
            else:
                logger.error("No 'smiles' column found in input data")
                sys.exit(1)

        # Combine results for this task
        if results:
            final_results = pd.concat(results, ignore_index=True)

            # Create task-specific output URI
            output_parts = output_uri.rsplit('.', 1)
            if len(output_parts) == 2:
                task_output_uri = f"{output_parts[0]}_task_{array_index}.{output_parts[1]}"
            else:
                task_output_uri = f"{output_uri}_task_{array_index}"

            # Upload results
            success = streamer.upload_dataframe(final_results, task_output_uri)

            if success:
                logger.info(f"Successfully processed {total_processed} molecules")
                logger.info(f"Results uploaded to {task_output_uri}")

                # Write completion marker
                completion_marker = {
                    'task_index': array_index,
                    'molecules_processed': total_processed,
                    'output_uri': task_output_uri,
                    'status': 'completed'
                }

                marker_uri = f"{output_uri.rsplit('.', 1)[0]}_task_{array_index}_complete.json"
                with open('/tmp/completion_marker.json', 'w') as f:
                    json.dump(completion_marker, f)

                # Upload completion marker
                streamer._get_provider(marker_uri).upload_file('/tmp/completion_marker.json', marker_uri)

            else:
                logger.error("Failed to upload results")
                sys.exit(1)
        else:
            logger.warning("No data processed")

    except Exception as e:
        logger.error(f"Array batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        # Health check script
        healthcheck_script = '''#!/usr/bin/env python3
"""
Health check script for ChemiAccel container.
"""

import sys
import logging

def main():
    """Perform health check."""
    try:
        # Check if ChemiAccel can be imported
        import chemi_accel

        # Check if basic functionality works
        from chemi_accel import Calculator
        calc = Calculator(backend='cpu')  # Use CPU for health check

        # Test with a simple molecule
        import pandas as pd
        test_smiles = pd.Series(['CCO'])  # ethanol
        result = calc.pandas(test_smiles)

        if len(result) > 0:
            print("Health check passed")
            sys.exit(0)
        else:
            print("Health check failed: no results")
            sys.exit(1)

    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        # Write scripts
        (build_path / "run_batch.py").write_text(batch_script)
        (build_path / "run_batch.sh").write_text(shell_script)
        (build_path / "run_array_batch.py").write_text(array_batch_script)
        (build_path / "healthcheck.py").write_text(healthcheck_script)

    def _generate_requirements(self, build_path: Path, config: ContainerConfig):
        """Generate requirements.txt for container."""
        requirements = [
            # Core scientific computing
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "scipy>=1.9.0",
            "scikit-learn>=1.1.0",
            # Chemistry libraries
            "rdkit-pypi>=2022.9.1",
            # Cloud storage
            "boto3>=1.26.0",  # AWS
            "botocore>=1.29.0",
            "google-cloud-storage>=2.7.0",  # GCP
            "google-cloud-batch>=0.17.0",  # GCP Batch
            "google-cloud-logging>=3.5.0",  # GCP Logging
            "azure-storage-blob>=12.14.0",  # Azure Blob
            "azure-batch>=13.0.0",  # Azure Batch
            "azure-identity>=1.12.0",  # Azure Auth
            # Data formats
            "pyarrow>=10.0.0",  # Parquet support
            "h5py>=3.7.0",  # HDF5 support
            "openpyxl>=3.0.0",  # Excel support
            # Web framework (for health checks and monitoring)
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
            "pydantic>=1.10.0",
            "requests>=2.28.0",
            # Monitoring and logging
            "psutil>=5.9.0",  # System monitoring
            "prometheus-client>=0.16.0",  # Metrics
            # Parallel processing
            "dask[complete]>=2023.1.0",
            "distributed>=2023.1.0",
            # GPU libraries (will be installed conditionally)
            "numba>=0.57.0",
            # Utilities
            "tqdm>=4.64.0",  # Progress bars
            "click>=8.1.0",  # CLI
            "pyyaml>=6.0",  # YAML support
            "python-dotenv>=1.0.0",  # Environment variables
        ]

        # Add provider-specific requirements
        if self.provider == "aws":
            requirements.extend(
                ["awscli>=2.11.0", "boto3[crt]>=1.26.0", "s3fs>=2023.1.0"]
            )
        elif self.provider == "gcp":
            requirements.extend(["google-cloud-sdk", "gcsfs>=2023.1.0"])
        elif self.provider == "azure":
            requirements.extend(["azure-cli>=2.46.0", "adlfs>=2023.1.0"])

        if config.additional_packages:
            requirements.extend(config.additional_packages)

        requirements_path = build_path / "app" / "requirements.txt"
        requirements_path.write_text("\\n".join(requirements))

    def _build_docker_image(self, build_path: Path, image_name: str) -> str:
        """Build Docker image."""
        try:
            # Build command
            cmd = ["docker", "build", "-t", image_name, str(build_path)]

            logger.info(f"Building Docker image: {image_name}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            logger.info("Docker build completed successfully")
            return image_name

        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise

    def push_to_registry(self, image_name: str, registry_uri: str = None) -> str:
        """
        Push container image to cloud registry.

        Args:
            image_name: Local image name
            registry_uri: Target registry URI

        Returns:
            Registry image URI
        """
        if registry_uri is None:
            registry_uri = self._get_default_registry_uri(image_name)

        try:
            # Tag image for registry
            tag_cmd = ["docker", "tag", image_name, registry_uri]
            subprocess.run(tag_cmd, check=True)

            # Push to registry
            push_cmd = ["docker", "push", registry_uri]
            subprocess.run(push_cmd, check=True)

            logger.info(f"Successfully pushed image to {registry_uri}")
            return registry_uri

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push image to registry: {e}")
            raise

    def _get_default_registry_uri(self, image_name: str) -> str:
        """Get default registry URI for provider."""
        if self.provider == "aws":
            # AWS ECR format: account.dkr.ecr.region.amazonaws.com/repository:tag
            account_id = self._get_aws_account_id()
            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            repo_name = image_name.split(":")[0]
            tag = image_name.split(":")[1] if ":" in image_name else "latest"
            return f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}"

        elif self.provider == "gcp":
            # GCR format: gcr.io/project-id/image:tag
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable required")
            return f"gcr.io/{project_id}/{image_name}"

        elif self.provider == "azure":
            # ACR format: registry.azurecr.io/repository:tag
            registry_name = os.environ.get("AZURE_CONTAINER_REGISTRY")
            if not registry_name:
                raise ValueError(
                    "AZURE_CONTAINER_REGISTRY environment variable required"
                )
            return f"{registry_name}.azurecr.io/{image_name}"

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_aws_account_id(self) -> str:
        """Get AWS account ID."""
        try:
            import boto3

            sts = boto3.client("sts")
            return sts.get_caller_identity()["Account"]
        except Exception as e:
            logger.error(f"Failed to get AWS account ID: {e}")
            raise

    def create_deployment_config(
        self, image_uri: str, job_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create deployment configuration for cloud provider.

        Args:
            image_uri: Container image URI
            job_config: Job configuration parameters

        Returns:
            Provider-specific deployment configuration
        """
        if self.provider == "aws":
            return self._create_aws_config(image_uri, job_config)
        elif self.provider == "gcp":
            return self._create_gcp_config(image_uri, job_config)
        elif self.provider == "azure":
            return self._create_azure_config(image_uri, job_config)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _create_aws_config(
        self, image_uri: str, job_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create AWS Batch job definition."""
        return {
            "jobDefinitionName": job_config.get("job_name", "chemi-accel-batch"),
            "type": "container",
            "containerProperties": {
                "image": image_uri,
                "vcpus": job_config.get("vcpus", 4),
                "memory": job_config.get("memory", 8192),
                "resourceRequirements": (
                    [{"type": "GPU", "value": str(job_config.get("gpus", 1))}]
                    if job_config.get("gpus", 0) > 0
                    else []
                ),
                "jobRoleArn": job_config.get("job_role_arn"),
                "executionRoleArn": job_config.get("execution_role_arn"),
            },
            "retryStrategy": {"attempts": job_config.get("retry_attempts", 2)},
            "timeout": {
                "attemptDurationSeconds": job_config.get("timeout_seconds", 3600)
            },
        }

    def _create_gcp_config(
        self, image_uri: str, job_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create GCP Batch job configuration."""
        return {
            "taskGroups": [
                {
                    "taskSpec": {
                        "runnables": [
                            {
                                "container": {
                                    "imageUri": image_uri,
                                    "commands": ["/bin/bash"],
                                    "entrypoint": "/app/run_batch.sh",
                                }
                            }
                        ],
                        "computeResource": {
                            "cpuMilli": job_config.get("cpu_milli", 4000),
                            "memoryMib": job_config.get("memory_mib", 8192),
                        },
                    },
                    "taskCount": job_config.get("task_count", 1),
                    "parallelism": job_config.get("parallelism", 1),
                }
            ],
            "allocationPolicy": {
                "instances": [
                    {
                        "policy": {
                            "machineType": job_config.get(
                                "machine_type", "n1-standard-4"
                            ),
                            "accelerators": (
                                [
                                    {
                                        "type": job_config.get(
                                            "gpu_type", "nvidia-tesla-t4"
                                        ),
                                        "count": job_config.get("gpus", 1),
                                    }
                                ]
                                if job_config.get("gpus", 0) > 0
                                else []
                            ),
                        }
                    }
                ]
            },
        }

    def _create_azure_config(
        self, image_uri: str, job_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create Azure Batch job configuration."""
        return {
            "id": job_config.get("job_name", "chemi-accel-batch"),
            "poolInfo": {"poolId": job_config.get("pool_id", "default-gpu-pool")},
            "jobManagerTask": {
                "id": "job-manager",
                "commandLine": "/bin/bash /app/run_batch.sh",
                "containerSettings": {
                    "imageName": image_uri,
                    "containerRunOptions": "--rm",
                },
                "resourceFiles": [],
                "environmentSettings": [
                    {"name": k, "value": v}
                    for k, v in job_config.get("environment", {}).items()
                ],
                "killJobOnCompletion": True,
            },
        }


# Utility functions
def build_chemi_accel_container(
    provider: str = "aws", code_path: str = None, push_to_registry: bool = False
) -> str:
    """
    Convenience function to build ChemiAccel container.

    Args:
        provider: Cloud provider ('aws', 'gcp', 'azure')
        code_path: Path to ChemiAccel source code
        push_to_registry: Whether to push to cloud registry

    Returns:
        Container image URI
    """
    packager = ContainerPackager(provider)
    image_name = packager.build_container(code_path)

    if push_to_registry:
        return packager.push_to_registry(image_name)

    return image_name

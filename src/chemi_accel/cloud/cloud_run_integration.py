"""
Google Cloud Run integration for ChemiAccel.

This module provides utilities for deploying ChemiAccel as serverless functions
on Google Cloud Run, including job submission and container management.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from ..core.exceptions import ChemiAccelError
from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class CloudRunJobConfig:
    """Configuration for Google Cloud Run jobs."""
    project_id: str
    region: str = 'us-central1'
    job_name: str = 'chemiaccel-job'
    image_uri: str = None
    cpu: str = '2'
    memory: str = '4Gi'
    max_retries: int = 3
    task_count: int = 1
    parallelism: int = 1
    task_timeout: str = '3600s'
    service_account: Optional[str] = None
    env_vars: Dict[str, str] = None


class CloudRunIntegration:
    """Google Cloud Run integration for serverless ChemiAccel processing."""
    
    def __init__(self, project_id: str, region: str = 'us-central1'):
        """Initialize Cloud Run integration."""
        self.project_id = project_id
        self.region = region
        self.gcloud_available = self._check_gcloud_availability()
        
        if not self.gcloud_available:
            logger.warning("gcloud CLI not available. Some features may not work.")
    
    def _check_gcloud_availability(self) -> bool:
        """Check if gcloud CLI is available."""
        try:
            subprocess.run(['gcloud', '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def build_container_image(self, 
                            dockerfile_path: str,
                            image_name: str,
                            tag: str = 'latest') -> str:
        """Build and push container image to Google Container Registry."""
        if not self.gcloud_available:
            raise ChemiAccelError("gcloud CLI not available")
        
        # Full image URI
        image_uri = f"gcr.io/{self.project_id}/{image_name}:{tag}"
        
        try:
            # Build image
            logger.info(f"Building container image: {image_uri}")
            subprocess.run([
                'gcloud', 'builds', 'submit',
                '--project', self.project_id,
                '--tag', image_uri,
                str(Path(dockerfile_path).parent)
            ], check=True)
            
            logger.info(f"Successfully built and pushed image: {image_uri}")
            return image_uri
            
        except subprocess.CalledProcessError as e:
            raise ChemiAccelError(f"Failed to build container image: {e}")
    
    def create_job_definition(self, config: CloudRunJobConfig) -> Dict[str, Any]:
        """Create Cloud Run job definition."""
        env_vars = config.env_vars or {}
        
        job_spec = {
            "apiVersion": "run.googleapis.com/v1",
            "kind": "Job",
            "metadata": {
                "name": config.job_name,
                "namespace": self.project_id
            },
            "spec": {
                "template": {
                    "spec": {
                        "template": {
                            "spec": {
                                "restartPolicy": "OnFailure",
                                "containers": [{
                                    "name": "chemiaccel-container",
                                    "image": config.image_uri,
                                    "resources": {
                                        "requests": {
                                            "cpu": config.cpu,
                                            "memory": config.memory
                                        },
                                        "limits": {
                                            "cpu": config.cpu,
                                            "memory": config.memory
                                        }
                                    },
                                    "env": [
                                        {"name": k, "value": v} 
                                        for k, v in env_vars.items()
                                    ]
                                }],
                                "timeoutSeconds": int(config.task_timeout.rstrip('s'))
                            }
                        }
                    }
                },
                "backoffLimit": config.max_retries,
                "completions": config.task_count,
                "parallelism": config.parallelism
            }
        }
        
        if config.service_account:
            job_spec["spec"]["template"]["spec"]["template"]["spec"]["serviceAccountName"] = config.service_account
        
        return job_spec
    
    def submit_job(self, config: CloudRunJobConfig, 
                   job_args: List[str] = None) -> str:
        """Submit job to Cloud Run."""
        if not self.gcloud_available:
            raise ChemiAccelError("gcloud CLI not available")
        
        # Create job definition
        job_spec = self.create_job_definition(config)
        
        # Add command arguments if provided
        if job_args:
            job_spec["spec"]["template"]["spec"]["template"]["spec"]["containers"][0]["args"] = job_args
        
        # Write job spec to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(job_spec, f)
            job_file = f.name
        
        try:
            # Submit job using gcloud
            result = subprocess.run([
                'gcloud', 'run', 'jobs', 'replace', job_file,
                '--project', self.project_id,
                '--region', self.region
            ], capture_output=True, text=True, check=True)
            
            # Execute the job
            exec_result = subprocess.run([
                'gcloud', 'run', 'jobs', 'execute', config.job_name,
                '--project', self.project_id,
                '--region', self.region,
                '--wait'
            ], capture_output=True, text=True, check=True)
            
            # Extract execution name from output
            execution_name = None
            for line in exec_result.stdout.split('\n'):
                if 'Execution' in line and 'created' in line:
                    # Parse execution name from output
                    parts = line.split()
                    for part in parts:
                        if config.job_name in part:
                            execution_name = part
                            break
            
            logger.info(f"Cloud Run job submitted: {execution_name}")
            return execution_name or config.job_name
            
        except subprocess.CalledProcessError as e:
            raise ChemiAccelError(f"Failed to submit Cloud Run job: {e.stderr}")
        finally:
            # Clean up temporary file
            os.unlink(job_file)
    
    def check_job_status(self, job_name: str) -> Dict[str, Any]:
        """Check status of Cloud Run job."""
        if not self.gcloud_available:
            return {"status": "unknown", "reason": "gcloud CLI not available"}
        
        try:
            result = subprocess.run([
                'gcloud', 'run', 'jobs', 'executions', 'list',
                '--job', job_name,
                '--project', self.project_id,
                '--region', self.region,
                '--format', 'json'
            ], capture_output=True, text=True, check=True)
            
            executions = json.loads(result.stdout)
            
            if executions:
                latest_execution = executions[0]  # Most recent execution
                return {
                    "status": latest_execution.get("status", {}).get("conditions", [{}])[0].get("type", "unknown"),
                    "reason": latest_execution.get("status", {}).get("conditions", [{}])[0].get("reason", ""),
                    "execution_name": latest_execution.get("metadata", {}).get("name", ""),
                    "start_time": latest_execution.get("status", {}).get("startTime", ""),
                    "completion_time": latest_execution.get("status", {}).get("completionTime", "")
                }
            else:
                return {"status": "not_found", "reason": "No executions found"}
                
        except subprocess.CalledProcessError as e:
            return {"status": "error", "reason": f"Failed to check status: {e.stderr}"}
    
    def get_job_logs(self, execution_name: str) -> str:
        """Get logs from Cloud Run job execution."""
        if not self.gcloud_available:
            return "gcloud CLI not available"
        
        try:
            result = subprocess.run([
                'gcloud', 'logging', 'read',
                f'resource.type="cloud_run_job" AND resource.labels.job_name="{execution_name}"',
                '--project', self.project_id,
                '--format', 'value(textPayload)',
                '--limit', '1000'
            ], capture_output=True, text=True, check=True)
            
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            return f"Failed to get logs: {e.stderr}"
    
    def cancel_job(self, execution_name: str) -> bool:
        """Cancel Cloud Run job execution."""
        if not self.gcloud_available:
            return False
        
        try:
            subprocess.run([
                'gcloud', 'run', 'jobs', 'executions', 'cancel', execution_name,
                '--project', self.project_id,
                '--region', self.region
            ], check=True)
            
            logger.info(f"Cancelled Cloud Run job execution: {execution_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel job execution: {e}")
            return False
    
    def create_dockerfile(self, output_path: str, 
                         python_version: str = '3.9',
                         cuda_support: bool = False) -> str:
        """Create Dockerfile for ChemiAccel container."""
        
        if cuda_support:
            base_image = f"nvidia/cuda:11.8-devel-ubuntu20.04"
            cuda_setup = """
# Install CUDA dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \\
    python3 python3-pip python3-dev \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*
"""
        else:
            base_image = f"python:{python_version}-slim"
            cuda_setup = ""
        
        dockerfile_content = f"""FROM {base_image}

{cuda_setup}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc g++ \\
    librdkit-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ChemiAccel
COPY . .
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV CHEMIACCEL_BACKEND={"cuda" if cuda_support else "cpu"}

# Default command
CMD ["python", "-m", "chemi_accel.cli.batch_process"]
"""
        
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        logger.info(f"Created Dockerfile at {output_path}")
        return output_path
    
    def create_requirements_file(self, output_path: str, 
                               include_cuda: bool = False) -> str:
        """Create requirements.txt for container."""
        
        requirements = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "rdkit-pypi>=2022.3.0",
            "scikit-learn>=1.0.0",
            "psutil>=5.8.0",
            "pyyaml>=6.0",
            "click>=8.0.0"
        ]
        
        if include_cuda:
            requirements.extend([
                "cupy-cuda11x>=10.0.0",
                "cudf-cu11>=22.0.0",
                "numba>=0.56.0"
            ])
        
        # Optional dependencies
        optional_requirements = [
            "dask[distributed]>=2022.1.0",
            "ray>=2.0.0",
            "mpi4py>=3.1.0"
        ]
        
        all_requirements = requirements + optional_requirements
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(all_requirements))
        
        logger.info(f"Created requirements.txt at {output_path}")
        return output_path


def create_cloud_run_deployment(project_id: str,
                               region: str = 'us-central1',
                               job_name: str = 'chemiaccel-job',
                               cuda_support: bool = False) -> CloudRunIntegration:
    """Create a complete Cloud Run deployment setup."""
    
    integration = CloudRunIntegration(project_id, region)
    
    # Create deployment directory
    deploy_dir = Path('cloud_run_deployment')
    deploy_dir.mkdir(exist_ok=True)
    
    # Create Dockerfile
    dockerfile_path = deploy_dir / 'Dockerfile'
    integration.create_dockerfile(str(dockerfile_path), cuda_support=cuda_support)
    
    # Create requirements.txt
    requirements_path = deploy_dir / 'requirements.txt'
    integration.create_requirements_file(str(requirements_path), include_cuda=cuda_support)
    
    # Create deployment script
    deploy_script = deploy_dir / 'deploy.sh'
    with open(deploy_script, 'w') as f:
        f.write(f"""#!/bin/bash
# ChemiAccel Cloud Run Deployment Script

set -e

PROJECT_ID="{project_id}"
REGION="{region}"
JOB_NAME="{job_name}"
IMAGE_NAME="chemiaccel"
IMAGE_TAG="latest"

echo "Building and deploying ChemiAccel to Cloud Run..."

# Build and push container image
gcloud builds submit --project $PROJECT_ID --tag gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG .

# Create job configuration
cat > job.yaml << EOF
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: $JOB_NAME
spec:
  template:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: chemiaccel-container
            image: gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG
            resources:
              requests:
                cpu: "2"
                memory: "4Gi"
              limits:
                cpu: "4"
                memory: "8Gi"
            env:
            - name: CHEMIACCEL_BACKEND
              value: "{'cuda' if cuda_support else 'cpu'}"
            - name: GOOGLE_CLOUD_PROJECT
              value: $PROJECT_ID
EOF

# Deploy job
gcloud run jobs replace job.yaml --project $PROJECT_ID --region $REGION

echo "Deployment completed!"
echo "To execute the job, run:"
echo "gcloud run jobs execute $JOB_NAME --project $PROJECT_ID --region $REGION --wait"
""")
    
    # Make script executable
    deploy_script.chmod(0o755)
    
    logger.info(f"Cloud Run deployment created in {deploy_dir}")
    logger.info(f"Run {deploy_script} to deploy to Google Cloud Run")
    
    return integration


# Example usage functions
def example_batch_processing_job():
    """Example of setting up a batch processing job on Cloud Run."""
    
    # Configuration
    project_id = "your-gcp-project-id"
    region = "us-central1"
    
    # Create integration
    integration = CloudRunIntegration(project_id, region)
    
    # Build container image (assumes Dockerfile exists)
    image_uri = integration.build_container_image(
        dockerfile_path="./Dockerfile",
        image_name="chemiaccel-batch",
        tag="v1.0"
    )
    
    # Configure job
    config = CloudRunJobConfig(
        project_id=project_id,
        region=region,
        job_name="chemiaccel-batch-job",
        image_uri=image_uri,
        cpu="4",
        memory="8Gi",
        task_count=10,  # Process 10 chunks in parallel
        parallelism=5,  # Run 5 tasks simultaneously
        env_vars={
            "INPUT_BUCKET": "gs://your-bucket/molecules.csv",
            "OUTPUT_BUCKET": "gs://your-bucket/results/",
            "CHUNK_SIZE": "1000"
        }
    )
    
    # Submit job
    execution_name = integration.submit_job(config, [
        "--input", "gs://your-bucket/molecules.csv",
        "--output", "gs://your-bucket/results/",
        "--chunk-size", "1000"
    ])
    
    print(f"Job submitted: {execution_name}")
    
    # Monitor job
    while True:
        status = integration.check_job_status(config.job_name)
        print(f"Job status: {status['status']}")
        
        if status['status'] in ['Succeeded', 'Failed']:
            break
        
        time.sleep(30)  # Check every 30 seconds
    
    # Get logs
    logs = integration.get_job_logs(execution_name)
    print("Job logs:")
    print(logs)


if __name__ == '__main__':
    # Example deployment
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'your-project-id')
    
    print("Creating Cloud Run deployment for ChemiAccel...")
    integration = create_cloud_run_deployment(
        project_id=project_id,
        cuda_support=False  # Set to True for GPU support
    )
    
    print("Deployment setup completed!")
    print("Check the 'cloud_run_deployment' directory for deployment files.")
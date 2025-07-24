# ML Platform Integration

ChemiAccel provides native integration with major cloud ML platforms, enabling seamless deployment and execution of molecular descriptor calculations at scale.

## Supported Platforms

### Amazon SageMaker
- **Processing Jobs**: GPU-accelerated batch processing
- **Pipeline Steps**: Native SageMaker Pipeline integration
- **Model Endpoints**: Real-time inference endpoints
- **Features**: Auto-scaling, S3 integration, CloudWatch monitoring

### Google Vertex AI
- **Custom Jobs**: Flexible job execution with GPU support
- **Pipeline Components**: Kubeflow Pipelines integration
- **Model Endpoints**: Managed prediction endpoints
- **Features**: TPU support, GCS integration, Cloud Logging

### Azure ML
- **Command Jobs**: Scalable compute with GPU instances
- **Pipeline Components**: Azure ML Pipeline integration
- **Model Endpoints**: Managed online endpoints
- **Features**: Azure Blob integration, Application Insights monitoring

## Quick Start

### Installation

```bash
# Install with cloud ML platform support
pip install chemi-accel[cloud]

# Platform-specific dependencies
pip install boto3  # For SageMaker
pip install google-cloud-aiplatform  # For Vertex AI
pip install azure-ai-ml  # For Azure ML
```

### Basic Usage

```python
from chemi_accel.cloud.ml_platforms import submit_chemi_accel_job, MLJobConfig

# Configure job
job_config = MLJobConfig(
    job_name="molecular-descriptors",
    instance_type="ml.g4dn.xlarge",  # GPU instance
    input_data_uri="s3://my-bucket/compounds.csv",
    output_data_uri="s3://my-bucket/results/",
    role_arn="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Descriptor configuration
descriptor_config = {
    'descriptors': ['molecular_weight', 'logp', 'tpsa'],
    'batch_size': 1000,
    'backend': 'cuda'
}

# Submit to SageMaker
job_name = submit_chemi_accel_job(
    platform='sagemaker',
    job_config=job_config,
    descriptor_config=descriptor_config,
    region='us-east-1'
)
```

## Platform-Specific Guides

### Amazon SageMaker

#### Processing Jobs

```python
from chemi_accel.cloud.ml_platforms import SageMakerIntegration, MLJobConfig

# Initialize integration
sagemaker = SageMakerIntegration(
    region='us-east-1',
    role_arn='arn:aws:iam::123456789012:role/SageMakerRole'
)

# Configure job
job_config = MLJobConfig(
    job_name="chemi-accel-processing",
    instance_type="ml.g4dn.xlarge",
    instance_count=2,
    input_data_uri="s3://my-bucket/input/",
    output_data_uri="s3://my-bucket/output/",
    environment_variables={
        'CUDA_VISIBLE_DEVICES': '0',
        'CHEMI_ACCEL_BACKEND': 'cuda'
    }
)

# Submit job
job_name = sagemaker.submit_processing_job(job_config, processing_script)

# Monitor job
status = sagemaker.get_job_status(job_name)
print(f"Job status: {status.status}")
```

#### Pipeline Integration

```python
from sagemaker.workflow.pipeline import Pipeline

# Create pipeline step
step_result = sagemaker.create_pipeline_step(
    'molecular-descriptors',
    job_config,
    descriptor_config
)

# Create pipeline
pipeline = Pipeline(
    name="chemi-accel-pipeline",
    steps=[step_result['step']]
)

# Execute pipeline
pipeline.create(role_arn=job_config.role_arn)
execution = pipeline.start()
```

#### Instance Types

| Instance Type | vCPUs | Memory | GPU | Use Case |
|---------------|-------|--------|-----|----------|
| ml.g4dn.xlarge | 4 | 16 GB | 1x T4 | Small-medium datasets |
| ml.g4dn.2xlarge | 8 | 32 GB | 1x T4 | Medium datasets |
| ml.p3.2xlarge | 8 | 61 GB | 1x V100 | Large datasets |
| ml.p3.8xlarge | 32 | 244 GB | 4x V100 | Very large datasets |

### Google Vertex AI

#### Custom Jobs

```python
from chemi_accel.cloud.ml_platforms import VertexAIIntegration

# Initialize integration
vertex = VertexAIIntegration(
    project_id='my-gcp-project',
    region='us-central1'
)

# Configure job
job_config = MLJobConfig(
    job_name="chemi-accel-vertex",
    instance_type="n1-standard-4",
    input_data_uri="gs://my-bucket/input/",
    output_data_uri="gs://my-bucket/output/"
)

# Submit job
job_name = vertex.submit_processing_job(job_config, processing_script)
```

#### Pipeline Components

```python
from google.cloud import aiplatform
from kfp.v2 import dsl

# Create component
component_result = vertex.create_pipeline_step(
    'molecular-descriptors',
    job_config,
    descriptor_config
)

# Define pipeline
@dsl.pipeline(name="chemi-accel-pipeline")
def molecular_pipeline():
    return component_result['component']()

# Submit pipeline
job = aiplatform.PipelineJob(
    display_name="chemi-accel-pipeline",
    template_path="pipeline.json"
)
job.submit()
```

#### Machine Types

| Machine Type | vCPUs | Memory | GPU | Use Case |
|--------------|-------|--------|-----|----------|
| n1-standard-4 | 4 | 15 GB | 1x T4 | Small datasets |
| n1-highmem-4 | 4 | 26 GB | 1x V100 | Medium datasets |
| a2-highgpu-1g | 12 | 85 GB | 1x A100 | Large datasets |
| a2-highgpu-2g | 24 | 170 GB | 2x A100 | Very large datasets |

### Azure ML

#### Command Jobs

```python
from chemi_accel.cloud.ml_platforms import AzureMLIntegration

# Initialize integration
azure = AzureMLIntegration(
    subscription_id='my-subscription',
    resource_group='my-resource-group',
    workspace_name='my-workspace'
)

# Configure job
job_config = MLJobConfig(
    job_name="chemi-accel-azure",
    instance_type="Standard_NC6",
    input_data_uri="azureml://datastores/workspaceblobstore/paths/input/",
    output_data_uri="azureml://datastores/workspaceblobstore/paths/output/"
)

# Submit job
job_name = azure.submit_processing_job(job_config, processing_script)
```

#### Pipeline Components

```python
from azure.ai.ml import dsl

# Create component
component_result = azure.create_pipeline_step(
    'molecular-descriptors',
    job_config,
    descriptor_config
)

# Define pipeline
@dsl.pipeline(name="chemi-accel-pipeline")
def molecular_pipeline():
    return component_result['component']()

# Submit pipeline
pipeline_job = azure.ml_client.jobs.create_or_update(molecular_pipeline)
```

#### VM Sizes

| VM Size | vCPUs | Memory | GPU | Use Case |
|---------|-------|--------|-----|----------|
| Standard_NC6 | 6 | 56 GB | 1x K80 | Small datasets |
| Standard_NC12 | 12 | 112 GB | 2x K80 | Medium datasets |
| Standard_NC24 | 24 | 224 GB | 4x K80 | Large datasets |
| Standard_ND40rs_v2 | 40 | 672 GB | 8x V100 | Very large datasets |

## Command Line Interface

### Installation

```bash
# Add CLI entry point to setup.py
pip install -e .
```

### Usage

```bash
# Create configuration template
chemi-accel ml create-config --output config.json

# Submit job to SageMaker
chemi-accel ml submit sagemaker \
  --job-name my-job \
  --instance-type ml.g4dn.xlarge \
  --input-data s3://bucket/input/ \
  --output-data s3://bucket/output/ \
  --config config.json \
  --role-arn arn:aws:iam::123456789012:role/SageMakerRole \
  --monitor

# Check job status
chemi-accel ml status sagemaker my-job --region us-east-1

# Get job logs
chemi-accel ml logs sagemaker my-job --region us-east-1

# Stop running job
chemi-accel ml stop sagemaker my-job --region us-east-1
```

## Configuration

### Descriptor Configuration

```json
{
  "descriptors": [
    "molecular_weight",
    "logp",
    "tpsa",
    "num_rotatable_bonds",
    "morgan_fingerprint",
    "constitutional_descriptors"
  ],
  "batch_size": 1000,
  "backend": "cuda",
  "fingerprint_params": {
    "morgan_radius": 2,
    "morgan_bits": 2048
  },
  "3d_descriptors": {
    "enabled": true,
    "conformer_generation": {
      "method": "distance_geometry",
      "num_conformers": 10
    }
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU device selection | `0` |
| `CHEMI_ACCEL_BACKEND` | Backend selection | `auto` |
| `CHEMI_ACCEL_BATCH_SIZE` | Batch size override | `1000` |
| `CHEMI_ACCEL_LOG_LEVEL` | Logging level | `INFO` |

## Best Practices

### Performance Optimization

1. **Choose appropriate instance types**:
   - Use GPU instances for large datasets (>10K molecules)
   - Use CPU instances for small datasets or data preprocessing

2. **Optimize batch sizes**:
   - Larger batches for GPU processing
   - Consider memory constraints

3. **Use appropriate storage**:
   - Store data in same region as compute
   - Use efficient formats (Parquet > CSV)

### Cost Optimization

1. **Right-size instances**:
   - Start with smaller instances and scale up
   - Use spot instances when available

2. **Optimize data transfer**:
   - Minimize cross-region transfers
   - Use compression for large datasets

3. **Monitor resource usage**:
   - Set up billing alerts
   - Use auto-scaling policies

### Security

1. **Use IAM roles and policies**:
   - Principle of least privilege
   - Separate roles for different environments

2. **Encrypt data**:
   - Enable encryption at rest and in transit
   - Use platform-specific encryption services

3. **Network security**:
   - Use VPCs and private subnets
   - Configure security groups appropriately

## Troubleshooting

### Common Issues

#### Job Fails to Start
- Check IAM permissions
- Verify instance type availability
- Ensure input data exists and is accessible

#### Out of Memory Errors
- Reduce batch size
- Use larger instance type
- Enable streaming for large datasets

#### Slow Performance
- Check GPU utilization
- Optimize data loading
- Consider parallel processing

#### Network Timeouts
- Increase timeout settings
- Check network connectivity
- Use retry mechanisms

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check job logs
integration = SageMakerIntegration(region='us-east-1')
logs = integration.get_job_logs(job_name)
print(logs)

# Monitor job status
while True:
    status = integration.get_job_status(job_name)
    print(f"Status: {status.status}")
    if status.status in ['Completed', 'Failed']:
        break
    time.sleep(30)
```

## Examples

See the [examples/ml_platform_workflows.py](../examples/ml_platform_workflows.py) file for comprehensive examples including:

- Basic job submission
- Pipeline integration
- Multi-platform comparison
- Real-world drug discovery workflow

## API Reference

### Classes

- `MLJobConfig`: Job configuration data class
- `MLJobStatus`: Job status information
- `SageMakerIntegration`: Amazon SageMaker integration
- `VertexAIIntegration`: Google Vertex AI integration
- `AzureMLIntegration`: Azure ML integration

### Functions

- `submit_chemi_accel_job()`: Convenience function for job submission
- `create_processing_script()`: Generate processing script

For detailed API documentation, see the docstrings in the source code.
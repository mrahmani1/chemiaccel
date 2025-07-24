#!/usr/bin/env python3
"""
ChemiAccel Cloud Batch Processing Demo

This script demonstrates how to use ChemiAccel's cloud batch processing
capabilities for large-scale molecular descriptor calculations.
"""

import pandas as pd
import json
from pathlib import Path

# Import ChemiAccel cloud components
from chemi_accel.cloud import (
    CloudBatchProcessor,
    CloudJobConfig,
    ContainerPackager,
    CloudStorageStreamer
)


def demo_job_configuration():
    """Demonstrate job configuration for different cloud providers."""
    print("=== Cloud Job Configuration Demo ===")
    
    # AWS Batch configuration
    aws_config = CloudJobConfig(
        provider='aws',
        instance_type='ml.g4dn.xlarge',
        num_instances=2,
        input_data_uri='s3://my-bucket/molecules.csv',
        output_data_uri='s3://my-bucket/results/descriptors.parquet',
        container_image='123456789.dkr.ecr.us-east-1.amazonaws.com/chemi-accel:latest',
        environment_variables={
            'CHEMI_ACCEL_BACKEND': 'cuda',
            'BATCH_SIZE': '1000'
        },
        timeout_minutes=120
    )
    
    print(f"AWS Config: {aws_config.provider} - {aws_config.instance_type}")
    print(f"Input: {aws_config.input_data_uri}")
    print(f"Output: {aws_config.output_data_uri}")
    
    # GCP Batch configuration
    gcp_config = CloudJobConfig(
        provider='gcp',
        instance_type='n1-standard-4',
        num_instances=1,
        input_data_uri='gs://my-bucket/molecules.csv',
        output_data_uri='gs://my-bucket/results/descriptors.parquet',
        container_image='gcr.io/my-project/chemi-accel:latest',
        environment_variables={
            'CHEMI_ACCEL_BACKEND': 'jax',
            'BATCH_SIZE': '2000'
        }
    )
    
    print(f"\\nGCP Config: {gcp_config.provider} - {gcp_config.instance_type}")
    print(f"Input: {gcp_config.input_data_uri}")
    print(f"Output: {gcp_config.output_data_uri}")
    
    # Azure Batch configuration
    azure_config = CloudJobConfig(
        provider='azure',
        instance_type='Standard_NC6',
        num_instances=1,
        input_data_uri='https://myaccount.blob.core.windows.net/data/molecules.csv',
        output_data_uri='https://myaccount.blob.core.windows.net/results/descriptors.parquet',
        container_image='myregistry.azurecr.io/chemi-accel:latest',
        environment_variables={
            'CHEMI_ACCEL_BACKEND': 'cuda'
        }
    )
    
    print(f"\\nAzure Config: {azure_config.provider} - {azure_config.instance_type}")
    print(f"Input: {azure_config.input_data_uri}")
    print(f"Output: {azure_config.output_data_uri}")


def demo_container_packaging():
    """Demonstrate container packaging for different providers."""
    print("\\n=== Container Packaging Demo ===")
    
    # AWS container packaging
    print("\\n1. AWS Container Packaging:")
    aws_packager = ContainerPackager(provider='aws')
    aws_config = aws_packager.base_configs['aws']
    
    print(f"Base image: {aws_config.base_image}")
    print(f"Python version: {aws_config.python_version}")
    print(f"CUDA version: {aws_config.cuda_version}")
    print(f"RAPIDS version: {aws_config.rapids_version}")
    
    # GCP container packaging
    print("\\n2. GCP Container Packaging:")
    gcp_packager = ContainerPackager(provider='gcp')
    gcp_config = gcp_packager.base_configs['gcp']
    
    print(f"Base image: {gcp_config.base_image}")
    print(f"Python version: {gcp_config.python_version}")
    print(f"CUDA version: {gcp_config.cuda_version}")
    
    # Azure container packaging
    print("\\n3. Azure Container Packaging:")
    azure_packager = ContainerPackager(provider='azure')
    azure_config = azure_packager.base_configs['azure']
    
    print(f"Base image: {azure_config.base_image}")
    print(f"Python version: {azure_config.python_version}")
    print(f"CUDA version: {azure_config.cuda_version}")


def demo_batch_processing_workflow():
    """Demonstrate the complete batch processing workflow."""
    print("\\n=== Batch Processing Workflow Demo ===")
    
    # Note: This is a demonstration - actual execution would require cloud credentials
    print("\\n1. Initialize Cloud Batch Processor")
    print("   processor = CloudBatchProcessor(provider='aws', region='us-east-1')")
    
    print("\\n2. Configure Job Parameters")
    job_config = CloudJobConfig(
        provider='aws',
        instance_type='ml.g4dn.xlarge',
        num_instances=1,
        input_data_uri='s3://demo-bucket/molecules.csv',
        output_data_uri='s3://demo-bucket/results.parquet',
        container_image='demo-image:latest',
        environment_variables={}
    )
    
    descriptor_config = {
        'descriptors': [
            'molecular_weight',
            'logp',
            'tpsa',
            'num_rotatable_bonds',
            'num_aromatic_rings'
        ],
        'backend': 'cuda',
        'batch_size': 1000
    }
    
    print(f"   Job config: {job_config.provider} - {job_config.instance_type}")
    print(f"   Descriptors: {descriptor_config['descriptors']}")
    
    print("\\n3. Submit Batch Job")
    print("   job_id = processor.submit_job(job_config, descriptor_config)")
    print("   # Returns: job-12345678")
    
    print("\\n4. Monitor Job Progress")
    print("   status = processor.monitor_job(job_id)")
    print("   # Returns: JobStatus(job_id='job-12345678', status='RUNNING', ...)")
    
    print("\\n5. Wait for Completion")
    print("   final_status = processor.wait_for_completion(job_id)")
    print("   # Returns: JobStatus(job_id='job-12345678', status='SUCCEEDED', ...)")
    
    print("\\n6. Retrieve Results")
    print("   results = processor.stream_from_storage(job_config.output_data_uri)")
    print("   # Returns: Iterator of DataFrame chunks with calculated descriptors")


def demo_cost_estimation():
    """Demonstrate cost estimation for different providers."""
    print("\\n=== Cost Estimation Demo ===")
    
    # Create sample job configurations
    configs = {
        'aws': CloudJobConfig(
            provider='aws',
            instance_type='ml.g4dn.xlarge',
            num_instances=2,
            input_data_uri='s3://bucket/input.csv',
            output_data_uri='s3://bucket/output.parquet',
            container_image='image:latest',
            environment_variables={}
        ),
        'gcp': CloudJobConfig(
            provider='gcp',
            instance_type='n1-standard-4',
            num_instances=2,
            input_data_uri='gs://bucket/input.csv',
            output_data_uri='gs://bucket/output.parquet',
            container_image='image:latest',
            environment_variables={}
        ),
        'azure': CloudJobConfig(
            provider='azure',
            instance_type='Standard_NC6',
            num_instances=2,
            input_data_uri='https://account.blob.core.windows.net/container/input.csv',
            output_data_uri='https://account.blob.core.windows.net/container/output.parquet',
            container_image='image:latest',
            environment_variables={}
        )
    }
    
    runtime_hours = 2.0
    
    print(f"\\nEstimated costs for {runtime_hours} hour job with 2 instances:")
    
    for provider, config in configs.items():
        print(f"\\n{provider.upper()}:")
        print(f"  Instance type: {config.instance_type}")
        print(f"  Instances: {config.num_instances}")
        
        # Note: Actual cost calculation would require processor initialization
        # This is just for demonstration
        if provider == 'aws':
            estimated_cost = 0.526 * runtime_hours * config.num_instances  # ml.g4dn.xlarge rate
            print(f"  Estimated cost: ${estimated_cost:.2f}")
        elif provider == 'gcp':
            estimated_cost = (0.19 + 0.45) * runtime_hours * config.num_instances  # compute + GPU
            print(f"  Estimated cost: ${estimated_cost:.2f}")
        elif provider == 'azure':
            estimated_cost = 0.90 * runtime_hours * config.num_instances  # Standard_NC6 rate
            print(f"  Estimated cost: ${estimated_cost:.2f}")


def demo_array_job_processing():
    """Demonstrate array job processing for large datasets."""
    print("\\n=== Array Job Processing Demo ===")
    
    print("\\nArray jobs allow parallel processing of large datasets:")
    print("- Split large dataset into chunks")
    print("- Process chunks in parallel across multiple instances")
    print("- Combine results automatically")
    
    print("\\nExample: Processing 1 million molecules")
    print("1. Split into 100 chunks of 10,000 molecules each")
    print("2. Submit array job with 100 tasks")
    print("3. Each task processes one chunk independently")
    print("4. Results are combined into final output")
    
    print("\\nCode example:")
    print("   job_id = processor.submit_batch_array_job(")
    print("       job_config=config,")
    print("       array_size=100,")
    print("       descriptor_config=descriptor_config")
    print("   )")
    
    print("\\nMonitoring array jobs:")
    print("   array_status = processor.get_array_job_status(job_id)")
    print("   # Returns overall status plus individual task statuses")


def demo_storage_streaming():
    """Demonstrate cloud storage streaming capabilities."""
    print("\\n=== Cloud Storage Streaming Demo ===")
    
    print("\\nSupported storage formats:")
    print("- AWS S3: s3://bucket/file.csv")
    print("- Google Cloud Storage: gs://bucket/file.csv")
    print("- Azure Blob Storage: https://account.blob.core.windows.net/container/file.csv")
    
    print("\\nSupported file formats:")
    print("- CSV: .csv, .csv.gz")
    print("- Parquet: .parquet, .parquet.gz")
    print("- SMILES: .smi, .smiles, .smi.gz")
    
    print("\\nStreaming example:")
    print("   streamer = CloudStorageStreamer()")
    print("   for chunk in streamer.stream_dataframe('s3://bucket/molecules.csv'):")
    print("       # Process chunk of molecules")
    print("       descriptors = calculator.pandas(chunk['smiles'])")
    print("       # Upload results")
    print("       streamer.upload_dataframe(descriptors, 's3://bucket/results.parquet')")


def main():
    """Run all demonstrations."""
    print("ChemiAccel Cloud Batch Processing Demo")
    print("=" * 50)
    
    demo_job_configuration()
    demo_container_packaging()
    demo_batch_processing_workflow()
    demo_cost_estimation()
    demo_array_job_processing()
    demo_storage_streaming()
    
    print("\\n" + "=" * 50)
    print("Demo completed!")
    print("\\nTo use ChemiAccel cloud batch processing in production:")
    print("1. Set up cloud credentials (AWS, GCP, or Azure)")
    print("2. Create compute environments and job queues")
    print("3. Build and push container images")
    print("4. Submit jobs using the CloudBatchProcessor")
    print("\\nFor more information, see the ChemiAccel documentation.")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
CLI tool for ChemiAccel ML platform integration.

Provides command-line interface for submitting jobs to various cloud ML platforms.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

from chemi_accel.cloud.ml_platforms import (
    MLJobConfig, SageMakerIntegration, VertexAIIntegration, 
    AzureMLIntegration, submit_chemi_accel_job
)


def create_job_config(args) -> MLJobConfig:
    """Create MLJobConfig from command line arguments."""
    return MLJobConfig(
        job_name=args.job_name,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        input_data_uri=args.input_data,
        output_data_uri=args.output_data,
        hyperparameters=json.loads(args.hyperparameters) if args.hyperparameters else None,
        environment_variables=json.loads(args.env_vars) if args.env_vars else None,
        max_runtime_seconds=args.max_runtime,
        role_arn=args.role_arn,
        service_account=args.service_account
    )


def load_descriptor_config(config_file: str) -> Dict[str, Any]:
    """Load descriptor configuration from file."""
    if not config_file:
        return {
            'descriptors': ['molecular_weight', 'logp', 'tpsa'],
            'batch_size': 1000,
            'backend': 'auto'
        }
    
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix == '.json':
            return json.load(f)
        elif config_path.suffix in ['.yml', '.yaml']:
            import yaml
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def submit_job_command(args):
    """Submit ChemiAccel job to ML platform."""
    try:
        # Create job configuration
        job_config = create_job_config(args)
        
        # Load descriptor configuration
        descriptor_config = load_descriptor_config(args.config)
        
        # Platform-specific arguments
        platform_kwargs = {}
        
        if args.platform == 'sagemaker':
            platform_kwargs.update({
                'region': args.region or 'us-east-1',
                'role_arn': args.role_arn
            })
        elif args.platform == 'vertex':
            platform_kwargs.update({
                'project_id': args.project_id,
                'region': args.region or 'us-central1'
            })
        elif args.platform == 'azure':
            platform_kwargs.update({
                'subscription_id': args.subscription_id,
                'resource_group': args.resource_group,
                'workspace_name': args.workspace_name
            })
        
        # Submit job
        job_name = submit_chemi_accel_job(
            platform=args.platform,
            job_config=job_config,
            descriptor_config=descriptor_config,
            **platform_kwargs
        )
        
        print(f"✓ Successfully submitted job: {job_name}")
        
        # Monitor job if requested
        if args.monitor:
            monitor_job(args.platform, job_name, platform_kwargs)
        
        return job_name
        
    except Exception as e:
        print(f"✗ Failed to submit job: {e}", file=sys.stderr)
        sys.exit(1)


def monitor_job(platform: str, job_name: str, platform_kwargs: Dict[str, Any]):
    """Monitor job status until completion."""
    print(f"Monitoring job: {job_name}")
    
    # Initialize platform integration
    if platform == 'sagemaker':
        integration = SageMakerIntegration(**platform_kwargs)
    elif platform == 'vertex':
        integration = VertexAIIntegration(**platform_kwargs)
    elif platform == 'azure':
        integration = AzureMLIntegration(**platform_kwargs)
    else:
        print(f"Unsupported platform for monitoring: {platform}")
        return
    
    # Monitor loop
    while True:
        try:
            status = integration.get_job_status(job_name)
            print(f"Status: {status.status}")
            
            # Check for completion
            completion_states = {
                'sagemaker': ['Completed', 'Failed', 'Stopped'],
                'vertex': ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'],
                'azure': ['Completed', 'Failed', 'Canceled']
            }
            
            if status.status in completion_states.get(platform, []):
                if 'Completed' in status.status or 'SUCCEEDED' in status.status:
                    print("✓ Job completed successfully!")
                else:
                    print(f"✗ Job failed: {status.failure_reason}")
                    # Get logs for debugging
                    logs = integration.get_job_logs(job_name)
                    if logs:
                        print("Recent logs:")
                        print(logs[-1000:])  # Last 1000 characters
                break
            
            time.sleep(30)  # Wait 30 seconds before next check
            
        except KeyboardInterrupt:
            print("\nMonitoring interrupted by user")
            break
        except Exception as e:
            print(f"Error monitoring job: {e}")
            break


def status_command(args):
    """Get job status."""
    try:
        # Platform-specific arguments
        platform_kwargs = {}
        
        if args.platform == 'sagemaker':
            platform_kwargs.update({
                'region': args.region or 'us-east-1',
                'role_arn': args.role_arn
            })
            integration = SageMakerIntegration(**platform_kwargs)
        elif args.platform == 'vertex':
            platform_kwargs.update({
                'project_id': args.project_id,
                'region': args.region or 'us-central1'
            })
            integration = VertexAIIntegration(**platform_kwargs)
        elif args.platform == 'azure':
            platform_kwargs.update({
                'subscription_id': args.subscription_id,
                'resource_group': args.resource_group,
                'workspace_name': args.workspace_name
            })
            integration = AzureMLIntegration(**platform_kwargs)
        else:
            raise ValueError(f"Unsupported platform: {args.platform}")
        
        # Get status
        status = integration.get_job_status(args.job_name)
        
        print(f"Job: {status.job_name}")
        print(f"Status: {status.status}")
        print(f"Created: {status.created_at}")
        if status.started_at:
            print(f"Started: {status.started_at}")
        if status.completed_at:
            print(f"Completed: {status.completed_at}")
        if status.failure_reason:
            print(f"Failure reason: {status.failure_reason}")
        
    except Exception as e:
        print(f"✗ Failed to get job status: {e}", file=sys.stderr)
        sys.exit(1)


def logs_command(args):
    """Get job logs."""
    try:
        # Platform-specific arguments
        platform_kwargs = {}
        
        if args.platform == 'sagemaker':
            platform_kwargs.update({
                'region': args.region or 'us-east-1',
                'role_arn': args.role_arn
            })
            integration = SageMakerIntegration(**platform_kwargs)
        elif args.platform == 'vertex':
            platform_kwargs.update({
                'project_id': args.project_id,
                'region': args.region or 'us-central1'
            })
            integration = VertexAIIntegration(**platform_kwargs)
        elif args.platform == 'azure':
            platform_kwargs.update({
                'subscription_id': args.subscription_id,
                'resource_group': args.resource_group,
                'workspace_name': args.workspace_name
            })
            integration = AzureMLIntegration(**platform_kwargs)
        else:
            raise ValueError(f"Unsupported platform: {args.platform}")
        
        # Get logs
        logs = integration.get_job_logs(args.job_name)
        print(logs)
        
    except Exception as e:
        print(f"✗ Failed to get job logs: {e}", file=sys.stderr)
        sys.exit(1)


def stop_command(args):
    """Stop running job."""
    try:
        # Platform-specific arguments
        platform_kwargs = {}
        
        if args.platform == 'sagemaker':
            platform_kwargs.update({
                'region': args.region or 'us-east-1',
                'role_arn': args.role_arn
            })
            integration = SageMakerIntegration(**platform_kwargs)
        elif args.platform == 'vertex':
            platform_kwargs.update({
                'project_id': args.project_id,
                'region': args.region or 'us-central1'
            })
            integration = VertexAIIntegration(**platform_kwargs)
        elif args.platform == 'azure':
            platform_kwargs.update({
                'subscription_id': args.subscription_id,
                'resource_group': args.resource_group,
                'workspace_name': args.workspace_name
            })
            integration = AzureMLIntegration(**platform_kwargs)
        else:
            raise ValueError(f"Unsupported platform: {args.platform}")
        
        # Stop job
        success = integration.stop_job(args.job_name)
        
        if success:
            print(f"✓ Successfully stopped job: {args.job_name}")
        else:
            print(f"✗ Failed to stop job: {args.job_name}")
            sys.exit(1)
        
    except Exception as e:
        print(f"✗ Failed to stop job: {e}", file=sys.stderr)
        sys.exit(1)


def create_config_template(args):
    """Create configuration template file."""
    template = {
        "descriptors": [
            "molecular_weight",
            "logp", 
            "tpsa",
            "num_rotatable_bonds",
            "num_aromatic_rings",
            "morgan_fingerprint",
            "constitutional_descriptors",
            "topological_descriptors"
        ],
        "batch_size": 1000,
        "backend": "cuda",
        "fingerprint_params": {
            "morgan_radius": 2,
            "morgan_bits": 2048
        },
        "3d_descriptors": {
            "enabled": False,
            "conformer_generation": {
                "method": "distance_geometry",
                "num_conformers": 10
            }
        },
        "polymer_support": {
            "enabled": False,
            "chain_length_limit": 100
        }
    }
    
    output_file = args.output or "chemi_accel_config.json"
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✓ Created configuration template: {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ChemiAccel ML Platform Integration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit job to SageMaker
  chemi-accel ml submit sagemaker --job-name my-job --instance-type ml.g4dn.xlarge \\
    --input-data s3://bucket/input/ --output-data s3://bucket/output/ \\
    --role-arn arn:aws:iam::123456789012:role/SageMakerRole

  # Submit job to Vertex AI
  chemi-accel ml submit vertex --job-name my-job --instance-type n1-standard-4 \\
    --input-data gs://bucket/input/ --output-data gs://bucket/output/ \\
    --project-id my-project

  # Submit job to Azure ML
  chemi-accel ml submit azure --job-name my-job --instance-type Standard_NC6 \\
    --input-data azureml://datastores/workspaceblobstore/paths/input/ \\
    --subscription-id my-subscription --resource-group my-rg --workspace-name my-workspace

  # Monitor job status
  chemi-accel ml status sagemaker my-job --region us-east-1

  # Get job logs
  chemi-accel ml logs vertex my-job --project-id my-project

  # Stop running job
  chemi-accel ml stop azure my-job --subscription-id my-subscription
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit job to ML platform')
    submit_parser.add_argument('platform', choices=['sagemaker', 'vertex', 'azure'],
                              help='ML platform to use')
    submit_parser.add_argument('--job-name', required=True,
                              help='Job name')
    submit_parser.add_argument('--instance-type', required=True,
                              help='Instance type (e.g., ml.g4dn.xlarge, n1-standard-4, Standard_NC6)')
    submit_parser.add_argument('--instance-count', type=int, default=1,
                              help='Number of instances')
    submit_parser.add_argument('--input-data',
                              help='Input data URI')
    submit_parser.add_argument('--output-data',
                              help='Output data URI')
    submit_parser.add_argument('--config',
                              help='Descriptor configuration file (JSON/YAML)')
    submit_parser.add_argument('--hyperparameters',
                              help='Hyperparameters as JSON string')
    submit_parser.add_argument('--env-vars',
                              help='Environment variables as JSON string')
    submit_parser.add_argument('--max-runtime', type=int, default=86400,
                              help='Maximum runtime in seconds')
    submit_parser.add_argument('--monitor', action='store_true',
                              help='Monitor job until completion')
    
    # Platform-specific arguments
    submit_parser.add_argument('--region',
                              help='AWS region or GCP region')
    submit_parser.add_argument('--role-arn',
                              help='AWS IAM role ARN (SageMaker)')
    submit_parser.add_argument('--project-id',
                              help='GCP project ID (Vertex AI)')
    submit_parser.add_argument('--subscription-id',
                              help='Azure subscription ID')
    submit_parser.add_argument('--resource-group',
                              help='Azure resource group')
    submit_parser.add_argument('--workspace-name',
                              help='Azure ML workspace name')
    submit_parser.add_argument('--service-account',
                              help='GCP service account')
    
    submit_parser.set_defaults(func=submit_job_command)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get job status')
    status_parser.add_argument('platform', choices=['sagemaker', 'vertex', 'azure'])
    status_parser.add_argument('job_name', help='Job name')
    status_parser.add_argument('--region', help='AWS/GCP region')
    status_parser.add_argument('--role-arn', help='AWS IAM role ARN')
    status_parser.add_argument('--project-id', help='GCP project ID')
    status_parser.add_argument('--subscription-id', help='Azure subscription ID')
    status_parser.add_argument('--resource-group', help='Azure resource group')
    status_parser.add_argument('--workspace-name', help='Azure ML workspace name')
    status_parser.set_defaults(func=status_command)
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Get job logs')
    logs_parser.add_argument('platform', choices=['sagemaker', 'vertex', 'azure'])
    logs_parser.add_argument('job_name', help='Job name')
    logs_parser.add_argument('--region', help='AWS/GCP region')
    logs_parser.add_argument('--role-arn', help='AWS IAM role ARN')
    logs_parser.add_argument('--project-id', help='GCP project ID')
    logs_parser.add_argument('--subscription-id', help='Azure subscription ID')
    logs_parser.add_argument('--resource-group', help='Azure resource group')
    logs_parser.add_argument('--workspace-name', help='Azure ML workspace name')
    logs_parser.set_defaults(func=logs_command)
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop running job')
    stop_parser.add_argument('platform', choices=['sagemaker', 'vertex', 'azure'])
    stop_parser.add_argument('job_name', help='Job name')
    stop_parser.add_argument('--region', help='AWS/GCP region')
    stop_parser.add_argument('--role-arn', help='AWS IAM role ARN')
    stop_parser.add_argument('--project-id', help='GCP project ID')
    stop_parser.add_argument('--subscription-id', help='Azure subscription ID')
    stop_parser.add_argument('--resource-group', help='Azure resource group')
    stop_parser.add_argument('--workspace-name', help='Azure ML workspace name')
    stop_parser.set_defaults(func=stop_command)
    
    # Config template command
    config_parser = subparsers.add_parser('create-config', help='Create configuration template')
    config_parser.add_argument('--output', help='Output file path')
    config_parser.set_defaults(func=create_config_template)
    
    # Parse arguments and execute command
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
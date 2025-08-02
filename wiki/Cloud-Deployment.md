# Cloud Deployment

ChemiAccel provides native integration with major cloud platforms, enabling scalable molecular descriptor calculations and AI-driven discovery workflows. This guide covers deployment strategies for AWS, Google Cloud Platform, and Microsoft Azure.

## Overview

Cloud deployment capabilities include:
- **Multi-Platform Support**: Native integration with AWS, GCP, and Azure
- **Serverless Computing**: Lambda, Cloud Functions, and Azure Functions
- **Managed Services**: SageMaker, Vertex AI, Azure ML integration
- **Auto-Scaling**: Dynamic resource allocation based on workload
- **Cost Optimization**: Spot instances, preemptible VMs, and reserved capacity
- **Data Pipeline Integration**: Cloud-native data processing workflows

## Quick Start

### Basic Cloud Setup

```python
from chemi_accel.cloud import CloudDeployment
from chemi_accel import Calculator

# Initialize cloud deployment manager
cloud = CloudDeployment(
    provider="aws",  # or "gcp", "azure"
    region="us-west-2",
    credentials_file="cloud_credentials.json"
)

# Deploy ChemiAccel as a service
deployment_config = {
    "instance_type": "g4dn.xlarge",
    "min_instances": 1,
    "max_instances": 10,
    "target_utilization": 70
}

service_endpoint = cloud.deploy_service(
    service_name="chemiaccel-api",
    config=deployment_config
)

print(f"ChemiAccel API deployed at: {service_endpoint}")
```

## AWS Integration

### Amazon SageMaker Deployment

```python
from chemi_accel.cloud.aws import SageMakerDeployment, S3DataManager
import boto3

class ChemiAccelSageMakerService:
    """ChemiAccel deployment on Amazon SageMaker."""
    
    def __init__(self, region="us-west-2"):
        self.region = region
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.s3_manager = S3DataManager()
        self.deployment = SageMakerDeployment(region=region)
    
    def create_training_job(self, training_config):
        """Create SageMaker training job for ChemiAccel models."""
        
        job_config = {
            "TrainingJobName": f"chemiaccel-training-{int(time.time())}",
            "AlgorithmSpecification": {
                "TrainingImage": "chemiaccel/sagemaker:latest",
                "TrainingInputMode": "File"
            },
            "RoleArn": training_config["execution_role"],
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": training_config["training_data_uri"],
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "text/csv",
                    "CompressionType": "None"
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": training_config["model_output_uri"]
            },
            "ResourceConfig": {
                "InstanceType": training_config.get("instance_type", "ml.g4dn.xlarge"),
                "InstanceCount": training_config.get("instance_count", 1),
                "VolumeSizeInGB": training_config.get("volume_size", 30)
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": training_config.get("max_runtime", 86400)
            },
            "HyperParameters": training_config.get("hyperparameters", {})
        }
        
        response = self.sagemaker_client.create_training_job(**job_config)
        return response["TrainingJobArn"]
    
    def deploy_endpoint(self, model_data_uri, endpoint_config):
        """Deploy ChemiAccel model as SageMaker endpoint."""
        
        model_name = f"chemiaccel-model-{int(time.time())}"
        endpoint_config_name = f"chemiaccel-config-{int(time.time())}"
        endpoint_name = f"chemiaccel-endpoint-{int(time.time())}"
        
        # Create model
        model_config = {
            "ModelName": model_name,
            "PrimaryContainer": {
                "Image": "chemiaccel/sagemaker-inference:latest",
                "ModelDataUrl": model_data_uri,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code"
                }
            },
            "ExecutionRoleArn": endpoint_config["execution_role"]
        }
        
        self.sagemaker_client.create_model(**model_config)
        
        # Create endpoint configuration
        endpoint_config_params = {
            "EndpointConfigName": endpoint_config_name,
            "ProductionVariants": [
                {
                    "VariantName": "primary",
                    "ModelName": model_name,
                    "InitialInstanceCount": endpoint_config.get("initial_instance_count", 1),
                    "InstanceType": endpoint_config.get("instance_type", "ml.m5.large"),
                    "InitialVariantWeight": 1
                }
            ]
        }
        
        if endpoint_config.get("enable_auto_scaling", False):
            endpoint_config_params["ProductionVariants"][0]["InitialInstanceCount"] = 1
        
        self.sagemaker_client.create_endpoint_config(**endpoint_config_params)
        
        # Create endpoint
        endpoint_params = {
            "EndpointName": endpoint_name,
            "EndpointConfigName": endpoint_config_name
        }
        
        self.sagemaker_client.create_endpoint(**endpoint_params)
        
        return endpoint_name
    
    def batch_processing_job(self, job_config):
        """Submit batch processing job for large-scale descriptor calculation."""
        
        job_name = f"chemiaccel-batch-{int(time.time())}"
        
        batch_job_config = {
            "TransformJobName": job_name,
            "ModelName": job_config["model_name"],
            "TransformInput": {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": job_config["input_data_uri"]
                    }
                },
                "ContentType": "text/csv",
                "SplitType": "Line"
            },
            "TransformOutput": {
                "S3OutputPath": job_config["output_data_uri"],
                "AssembleWith": "Line"
            },
            "TransformResources": {
                "InstanceType": job_config.get("instance_type", "ml.m5.large"),
                "InstanceCount": job_config.get("instance_count", 1)
            }
        }
        
        response = self.sagemaker_client.create_transform_job(**batch_job_config)
        return response["TransformJobArn"]

# Example: Deploy molecular property prediction model
sagemaker_service = ChemiAccelSageMakerService(region="us-west-2")

# Training configuration
training_config = {
    "training_data_uri": "s3://my-bucket/training-data/molecules.csv",
    "model_output_uri": "s3://my-bucket/models/",
    "execution_role": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    "instance_type": "ml.g4dn.xlarge",
    "instance_count": 1,
    "hyperparameters": {
        "epochs": "100",
        "batch_size": "64",
        "learning_rate": "0.001"
    }
}

# Submit training job
training_job_arn = sagemaker_service.create_training_job(training_config)
print(f"Training job submitted: {training_job_arn}")

# Deploy trained model
endpoint_config = {
    "execution_role": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    "instance_type": "ml.m5.large",
    "initial_instance_count": 1,
    "enable_auto_scaling": True
}

model_uri = "s3://my-bucket/models/model.tar.gz"
endpoint_name = sagemaker_service.deploy_endpoint(model_uri, endpoint_config)
print(f"Endpoint deployed: {endpoint_name}")
```

### AWS Lambda Serverless Functions

```python
from chemi_accel.cloud.aws import LambdaDeployment
import json

class ChemiAccelLambdaFunction:
    """Serverless ChemiAccel functions on AWS Lambda."""
    
    def __init__(self):
        self.lambda_deployment = LambdaDeployment()
    
    def create_descriptor_calculation_function(self):
        """Create Lambda function for molecular descriptor calculation."""
        
        function_code = '''
import json
from chemi_accel import Calculator

def lambda_handler(event, context):
    try:
        # Parse input
        molecules = event.get('molecules', [])
        descriptors = event.get('descriptors', 'default')
        
        if not molecules:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No molecules provided'})
            }
        
        # Calculate descriptors
        calc = Calculator(backend="cpu")  # Lambda uses CPU
        results = calc.pandas(molecules, descriptors=descriptors)
        
        # Convert to JSON-serializable format
        results_dict = results.to_dict('records')
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'results': results_dict,
                'count': len(results_dict)
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
        '''
        
        function_config = {
            "FunctionName": "chemiaccel-descriptor-calc",
            "Runtime": "python3.9",
            "Role": "arn:aws:iam::123456789012:role/LambdaExecutionRole",
            "Handler": "lambda_function.lambda_handler",
            "Code": {"ZipFile": function_code.encode()},
            "Description": "ChemiAccel molecular descriptor calculation",
            "Timeout": 300,  # 5 minutes
            "MemorySize": 3008,  # Maximum memory
            "Environment": {
                "Variables": {
                    "PYTHONPATH": "/opt/python"
                }
            }
        }
        
        return self.lambda_deployment.create_function(function_config)
    
    def create_batch_processor_function(self):
        """Create Lambda function triggered by S3 for batch processing."""
        
        batch_function_code = '''
import json
import boto3
import pandas as pd
from chemi_accel import Calculator

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    try:
        # Parse S3 event
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            
            # Download input file
            response = s3_client.get_object(Bucket=bucket, Key=key)
            input_data = pd.read_csv(response['Body'])
            
            # Process molecules
            molecules = input_data['smiles'].tolist()
            calc = Calculator(backend="cpu")
            results = calc.pandas(molecules, descriptors="comprehensive")
            
            # Upload results
            output_key = key.replace('input/', 'output/').replace('.csv', '_descriptors.csv')
            results_csv = results.to_csv(index=False)
            
            s3_client.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=results_csv,
                ContentType='text/csv'
            )
            
            print(f"Processed {len(molecules)} molecules from {key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Batch processing completed'})
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
        '''
        
        batch_config = {
            "FunctionName": "chemiaccel-batch-processor",
            "Runtime": "python3.9",
            "Role": "arn:aws:iam::123456789012:role/LambdaS3ExecutionRole",
            "Handler": "lambda_function.lambda_handler",
            "Code": {"ZipFile": batch_function_code.encode()},
            "Timeout": 900,  # 15 minutes
            "MemorySize": 3008
        }
        
        return self.lambda_deployment.create_function(batch_config)
    
    def setup_api_gateway(self, lambda_function_name):
        """Set up API Gateway for Lambda function."""
        
        api_config = {
            "function_name": lambda_function_name,
            "api_name": "chemiaccel-api",
            "stage_name": "prod",
            "cors_enabled": True,
            "rate_limit": 1000,  # requests per day
            "burst_limit": 20    # concurrent requests
        }
        
        api_url = self.lambda_deployment.create_api_gateway(api_config)
        return api_url

# Deploy serverless functions
lambda_service = ChemiAccelLambdaFunction()

# Create descriptor calculation function
descriptor_function = lambda_service.create_descriptor_calculation_function()
print(f"Descriptor function created: {descriptor_function['FunctionArn']}")

# Create batch processing function
batch_function = lambda_service.create_batch_processor_function()
print(f"Batch function created: {batch_function['FunctionArn']}")

# Set up API Gateway
api_url = lambda_service.setup_api_gateway("chemiaccel-descriptor-calc")
print(f"API Gateway URL: {api_url}")
```

### EC2 Auto Scaling Deployment

```python
from chemi_accel.cloud.aws import EC2AutoScaling
import boto3

class ChemiAccelAutoScalingCluster:
    """Auto-scaling ChemiAccel cluster on EC2."""
    
    def __init__(self, region="us-west-2"):
        self.region = region
        self.ec2_client = boto3.client('ec2', region_name=region)
        self.autoscaling_client = boto3.client('autoscaling', region_name=region)
        self.elbv2_client = boto3.client('elbv2', region_name=region)
        
        self.auto_scaling = EC2AutoScaling(region=region)
    
    def create_launch_template(self):
        """Create launch template for ChemiAccel instances."""
        
        user_data_script = '''#!/bin/bash
# Install dependencies
yum update -y
yum install -y python3 python3-pip git

# Install CUDA (for GPU instances)
wget https://developer.download.nvidia.com/compute/cuda/repos/amazon2/x86_64/cuda-repo-amazon2-11-4.x86_64.rpm
rpm -i cuda-repo-amazon2-11-4.x86_64.rpm
yum clean all
yum install -y cuda

# Install ChemiAccel
pip3 install chemi-accel[cuda]

# Create service user
useradd -m chemiaccel

# Create systemd service
cat > /etc/systemd/system/chemiaccel.service << EOF
[Unit]
Description=ChemiAccel API Service
After=network.target

[Service]
Type=simple
User=chemiaccel
WorkingDirectory=/home/chemiaccel
ExecStart=/usr/local/bin/python3 -m chemi_accel.api.server --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Start service
systemctl enable chemiaccel
systemctl start chemiaccel

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U amazon-cloudwatch-agent.rpm
'''
        
        launch_template_config = {
            "LaunchTemplateName": "chemiaccel-template",
            "LaunchTemplateData": {
                "ImageId": "ami-0c55b159cbfafe1d0",  # Amazon Linux 2
                "InstanceType": "g4dn.xlarge",
                "SecurityGroupIds": ["sg-chemiaccel"],
                "UserData": base64.b64encode(user_data_script.encode()).decode(),
                "IamInstanceProfile": {
                    "Name": "ChemiAccelInstanceProfile"
                },
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": "ChemiAccel-Worker"},
                            {"Key": "Service", "Value": "ChemiAccel"}
                        ]
                    }
                ],
                "Monitoring": {"Enabled": True}
            }
        }
        
        response = self.ec2_client.create_launch_template(**launch_template_config)
        return response["LaunchTemplate"]["LaunchTemplateId"]
    
    def create_auto_scaling_group(self, launch_template_id, vpc_config):
        """Create auto scaling group for ChemiAccel cluster."""
        
        # Create target group for load balancer
        target_group_response = self.elbv2_client.create_target_group(
            Name="chemiaccel-targets",
            Protocol="HTTP",
            Port=8000,
            VpcId=vpc_config["vpc_id"],
            HealthCheckPath="/health",
            HealthCheckIntervalSeconds=30,
            HealthyThresholdCount=2,
            UnhealthyThresholdCount=5
        )
        
        target_group_arn = target_group_response["TargetGroups"][0]["TargetGroupArn"]
        
        # Create auto scaling group
        asg_config = {
            "AutoScalingGroupName": "chemiaccel-asg",
            "LaunchTemplate": {
                "LaunchTemplateId": launch_template_id,
                "Version": "$Latest"
            },
            "MinSize": 1,
            "MaxSize": 20,
            "DesiredCapacity": 2,
            "VPCZoneIdentifier": ",".join(vpc_config["subnet_ids"]),
            "TargetGroupARNs": [target_group_arn],
            "HealthCheckType": "ELB",
            "HealthCheckGracePeriod": 300,
            "Tags": [
                {
                    "Key": "Name",
                    "Value": "ChemiAccel-ASG",
                    "PropagateAtLaunch": True,
                    "ResourceId": "chemiaccel-asg",
                    "ResourceType": "auto-scaling-group"
                }
            ]
        }
        
        self.autoscaling_client.create_auto_scaling_group(**asg_config)
        
        return target_group_arn
    
    def create_scaling_policies(self):
        """Create scaling policies based on CPU and queue depth."""
        
        # Scale up policy
        scale_up_policy = {
            "AutoScalingGroupName": "chemiaccel-asg",
            "PolicyName": "chemiaccel-scale-up",
            "PolicyType": "TargetTrackingScaling",
            "TargetTrackingConfiguration": {
                "TargetValue": 70.0,
                "PredefinedMetricSpecification": {
                    "PredefinedMetricType": "ASGAverageCPUUtilization"
                },
                "ScaleOutCooldown": 300,
                "ScaleInCooldown": 300
            }
        }
        
        # Custom metric scaling (based on queue depth)
        queue_scaling_policy = {
            "AutoScalingGroupName": "chemiaccel-asg",
            "PolicyName": "chemiaccel-queue-scaling",
            "PolicyType": "TargetTrackingScaling",
            "TargetTrackingConfiguration": {
                "TargetValue": 10.0,
                "CustomizedMetricSpecification": {
                    "MetricName": "QueueDepth",
                    "Namespace": "ChemiAccel/Processing",
                    "Statistic": "Average"
                },
                "ScaleOutCooldown": 180,
                "ScaleInCooldown": 600
            }
        }
        
        self.autoscaling_client.put_scaling_policy(**scale_up_policy)
        self.autoscaling_client.put_scaling_policy(**queue_scaling_policy)

# Deploy auto-scaling cluster
cluster = ChemiAccelAutoScalingCluster()

# Create launch template
template_id = cluster.create_launch_template()
print(f"Launch template created: {template_id}")

# Set up VPC configuration
vpc_config = {
    "vpc_id": "vpc-12345678",
    "subnet_ids": ["subnet-12345678", "subnet-87654321"]
}

# Create auto scaling group
target_group_arn = cluster.create_auto_scaling_group(template_id, vpc_config)
print(f"Auto scaling group created with target group: {target_group_arn}")

# Set up scaling policies
cluster.create_scaling_policies()
print("Scaling policies configured")
```

## Google Cloud Platform Integration

### Vertex AI Deployment

```python
from chemi_accel.cloud.gcp import VertexAIDeployment, CloudStorageManager
from google.cloud import aiplatform

class ChemiAccelVertexAI:
    """ChemiAccel deployment on Google Cloud Vertex AI."""
    
    def __init__(self, project_id, region="us-central1"):
        self.project_id = project_id
        self.region = region
        
        aiplatform.init(project=project_id, location=region)
        
        self.vertex_deployment = VertexAIDeployment(project_id, region)
        self.storage_manager = CloudStorageManager(project_id)
    
    def create_custom_training_job(self, training_config):
        """Create custom training job for ChemiAccel models."""
        
        # Define custom container
        container_spec = aiplatform.CustomContainerTrainingJob(
            display_name="chemiaccel-training",
            container_uri="gcr.io/{}/chemiaccel-trainer:latest".format(self.project_id),
            model_serving_container_image_uri="gcr.io/{}/chemiaccel-predictor:latest".format(self.project_id),
            model_serving_container_predict_route="/predict",
            model_serving_container_health_route="/health"
        )
        
        # Training arguments
        training_args = [
            "--training-data-uri", training_config["training_data_uri"],
            "--model-output-uri", training_config["model_output_uri"],
            "--epochs", str(training_config.get("epochs", 100)),
            "--batch-size", str(training_config.get("batch_size", 64)),
            "--learning-rate", str(training_config.get("learning_rate", 0.001))
        ]
        
        # Submit training job
        model = container_spec.run(
            args=training_args,
            replica_count=training_config.get("replica_count", 1),
            machine_type=training_config.get("machine_type", "n1-standard-4"),
            accelerator_type=training_config.get("accelerator_type", "NVIDIA_TESLA_T4"),
            accelerator_count=training_config.get("accelerator_count", 1),
            base_output_dir=training_config["model_output_uri"]
        )
        
        return model
    
    def deploy_endpoint(self, model, endpoint_config):
        """Deploy trained model to Vertex AI endpoint."""
        
        endpoint = aiplatform.Endpoint.create(
            display_name="chemiaccel-endpoint",
            project=self.project_id,
            location=self.region
        )
        
        # Deploy model to endpoint
        endpoint.deploy(
            model=model,
            deployed_model_display_name="chemiaccel-predictor",
            machine_type=endpoint_config.get("machine_type", "n1-standard-2"),
            min_replica_count=endpoint_config.get("min_replicas", 1),
            max_replica_count=endpoint_config.get("max_replicas", 10),
            accelerator_type=endpoint_config.get("accelerator_type"),
            accelerator_count=endpoint_config.get("accelerator_count", 0)
        )
        
        return endpoint
    
    def create_batch_prediction_job(self, model, job_config):
        """Create batch prediction job for large-scale processing."""
        
        batch_job = aiplatform.BatchPredictionJob.create(
            job_display_name="chemiaccel-batch-prediction",
            model_name=model.resource_name,
            instances_format="csv",
            predictions_format="csv",
            gcs_source=job_config["input_uri"],
            gcs_destination_prefix=job_config["output_uri"],
            machine_type=job_config.get("machine_type", "n1-standard-4"),
            accelerator_type=job_config.get("accelerator_type"),
            accelerator_count=job_config.get("accelerator_count", 0),
            starting_replica_count=job_config.get("replica_count", 1),
            max_replica_count=job_config.get("max_replicas", 10)
        )
        
        return batch_job

# Example: Deploy on Vertex AI
vertex_service = ChemiAccelVertexAI(
    project_id="my-gcp-project",
    region="us-central1"
)

# Training configuration
training_config = {
    "training_data_uri": "gs://my-bucket/training-data/",
    "model_output_uri": "gs://my-bucket/models/",
    "epochs": 150,
    "batch_size": 128,
    "learning_rate": 0.001,
    "machine_type": "n1-standard-8",
    "accelerator_type": "NVIDIA_TESLA_V100",
    "accelerator_count": 1
}

# Train model
trained_model = vertex_service.create_custom_training_job(training_config)
print(f"Training job submitted: {trained_model.resource_name}")

# Deploy endpoint
endpoint_config = {
    "machine_type": "n1-standard-4",
    "min_replicas": 1,
    "max_replicas": 20,
    "accelerator_type": "NVIDIA_TESLA_T4",
    "accelerator_count": 1
}

endpoint = vertex_service.deploy_endpoint(trained_model, endpoint_config)
print(f"Endpoint deployed: {endpoint.resource_name}")
```

### Cloud Functions Serverless

```python
from chemi_accel.cloud.gcp import CloudFunctionsDeployment
import functions_framework
import json

class ChemiAccelCloudFunctions:
    """Serverless ChemiAccel functions on Google Cloud Functions."""
    
    def __init__(self, project_id):
        self.project_id = project_id
        self.functions_deployment = CloudFunctionsDeployment(project_id)
    
    def create_descriptor_function(self):
        """Create Cloud Function for descriptor calculation."""
        
        function_code = '''
import json
from chemi_accel import Calculator
import functions_framework

@functions_framework.http
def calculate_descriptors(request):
    """HTTP Cloud Function for descriptor calculation."""
    
    # Handle CORS
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers for main request
    headers = {'Access-Control-Allow-Origin': '*'}
    
    try:
        request_json = request.get_json(silent=True)
        
        if not request_json:
            return json.dumps({'error': 'No JSON data provided'}), 400, headers
        
        molecules = request_json.get('molecules', [])
        descriptors = request_json.get('descriptors', 'default')
        
        if not molecules:
            return json.dumps({'error': 'No molecules provided'}), 400, headers
        
        # Calculate descriptors
        calc = Calculator(backend="cpu")
        results = calc.pandas(molecules, descriptors=descriptors)
        
        # Convert to JSON-serializable format
        results_dict = results.to_dict('records')
        
        response_data = {
            'results': results_dict,
            'count': len(results_dict),
            'descriptors_calculated': descriptors
        }
        
        return json.dumps(response_data), 200, headers
        
    except Exception as e:
        error_response = {'error': str(e)}
        return json.dumps(error_response), 500, headers
        '''
        
        function_config = {
            "name": "chemiaccel-descriptors",
            "source_code": function_code,
            "entry_point": "calculate_descriptors",
            "runtime": "python39",
            "memory": "2048MB",
            "timeout": "540s",
            "environment_variables": {
                "PYTHONPATH": "/workspace"
            },
            "trigger": "http"
        }
        
        return self.functions_deployment.deploy_function(function_config)
    
    def create_pubsub_batch_function(self):
        """Create Pub/Sub triggered function for batch processing."""
        
        batch_function_code = '''
import json
import base64
from google.cloud import storage
import pandas as pd
from chemi_accel import Calculator
import functions_framework

@functions_framework.cloud_event
def process_batch(cloud_event):
    """Pub/Sub triggered function for batch processing."""
    
    try:
        # Decode Pub/Sub message
        message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode()
        job_config = json.loads(message_data)
        
        # Initialize storage client
        storage_client = storage.Client()
        
        # Download input file
        bucket = storage_client.bucket(job_config["input_bucket"])
        blob = bucket.blob(job_config["input_file"])
        input_data = pd.read_csv(blob.open())
        
        # Process molecules
        molecules = input_data['smiles'].tolist()
        calc = Calculator(backend="cpu")
        results = calc.pandas(molecules, descriptors=job_config.get('descriptors', 'comprehensive'))
        
        # Upload results
        output_bucket = storage_client.bucket(job_config["output_bucket"])
        output_blob = output_bucket.blob(job_config["output_file"])
        
        output_csv = results.to_csv(index=False)
        output_blob.upload_from_string(output_csv, content_type='text/csv')
        
        print(f"Processed {len(molecules)} molecules")
        print(f"Results uploaded to gs://{job_config['output_bucket']}/{job_config['output_file']}")
        
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        raise
        '''
        
        batch_config = {
            "name": "chemiaccel-batch-processor",
            "source_code": batch_function_code,
            "entry_point": "process_batch",
            "runtime": "python39",
            "memory": "4096MB",
            "timeout": "540s",
            "trigger": "pubsub",
            "topic": f"projects/{self.project_id}/topics/chemiaccel-batch"
        }
        
        return self.functions_deployment.deploy_function(batch_config)

# Deploy Cloud Functions
functions_service = ChemiAccelCloudFunctions(project_id="my-gcp-project")

# Deploy descriptor calculation function
descriptor_function = functions_service.create_descriptor_function()
print(f"Descriptor function deployed: {descriptor_function}")

# Deploy batch processing function
batch_function = functions_service.create_pubsub_batch_function()
print(f"Batch function deployed: {batch_function}")
```

## Microsoft Azure Integration

### Azure Machine Learning Deployment

```python
from chemi_accel.cloud.azure import AzureMLDeployment
from azureml.core import Workspace, Environment, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute

class ChemiAccelAzureML:
    """ChemiAccel deployment on Azure Machine Learning."""
    
    def __init__(self, subscription_id, resource_group, workspace_name):
        self.workspace = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name
        )
        
        self.azure_deployment = AzureMLDeployment(self.workspace)
    
    def create_compute_cluster(self, cluster_config):
        """Create Azure ML compute cluster."""
        
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=cluster_config.get("vm_size", "Standard_NC6s_v3"),
            min_nodes=cluster_config.get("min_nodes", 0),
            max_nodes=cluster_config.get("max_nodes", 10),
            idle_seconds_before_scaledown=cluster_config.get("idle_seconds", 1800),
            tier=cluster_config.get("tier", "Dedicated")
        )
        
        cluster_name = cluster_config.get("name", "chemiaccel-cluster")
        
        if cluster_name not in self.workspace.compute_targets:
            compute_target = ComputeTarget.create(
                self.workspace,
                cluster_name,
                compute_config
            )
            compute_target.wait_for_completion(show_output=True)
        else:
            compute_target = self.workspace.compute_targets[cluster_name]
        
        return compute_target
    
    def create_training_environment(self):
        """Create training environment with ChemiAccel dependencies."""
        
        environment = Environment(name="chemiaccel-env")
        
        # Define conda dependencies
        conda_deps = {
            "channels": ["conda-forge", "pytorch"],
            "dependencies": [
                "python=3.9",
                "pip",
                {
                    "pip": [
                        "chemi-accel[azure]",
                        "azureml-sdk",
                        "mlflow",
                        "tensorboard"
                    ]
                }
            ]
        }
        
        environment.python.conda_dependencies = conda_deps
        environment.docker.enabled = True
        environment.docker.base_image = "mcr.microsoft.com/azureml/pytorch-1.10-ubuntu18.04-py37-cuda11-gpu"
        
        return environment
    
    def submit_training_job(self, training_config):
        """Submit training job to Azure ML."""
        
        from azureml.core import ScriptRunConfig
        from azureml.core.runconfig import MpiConfiguration
        
        # Create experiment
        experiment = Experiment(self.workspace, training_config["experiment_name"])
        
        # Set up distributed training if specified
        distributed_config = None
        if training_config.get("distributed", False):
            distributed_config = MpiConfiguration(
                node_count=training_config.get("node_count", 2),
                process_count_per_node=training_config.get("processes_per_node", 1)
            )
        
        # Create script run configuration
        script_config = ScriptRunConfig(
            source_directory=training_config["source_directory"],
            script=training_config["training_script"],
            arguments=training_config.get("arguments", []),
            compute_target=training_config["compute_target"],
            environment=training_config["environment"],
            distributed_job_config=distributed_config
        )
        
        # Submit run
        run = experiment.submit(script_config)
        return run
    
    def deploy_real_time_endpoint(self, model, endpoint_config):
        """Deploy model as real-time endpoint."""
        
        from azureml.core.webservice import AciWebservice, Webservice
        from azureml.core.model import InferenceConfig
        
        # Inference configuration
        inference_config = InferenceConfig(
            entry_script=endpoint_config["inference_script"],
            environment=endpoint_config["environment"]
        )
        
        # Deployment configuration
        if endpoint_config.get("deployment_type") == "aci":
            deployment_config = AciWebservice.deploy_configuration(
                cpu_cores=endpoint_config.get("cpu_cores", 1),
                memory_gb=endpoint_config.get("memory_gb", 1),
                tags=endpoint_config.get("tags", {}),
                description="ChemiAccel real-time endpoint"
            )
        else:
            # AKS deployment
            from azureml.core.webservice import AksWebservice
            deployment_config = AksWebservice.deploy_configuration(
                cpu_cores=endpoint_config.get("cpu_cores", 2),
                memory_gb=endpoint_config.get("memory_gb", 4),
                gpu_cores=endpoint_config.get("gpu_cores", 0),
                autoscale_enabled=endpoint_config.get("autoscale", True),
                autoscale_min_replicas=endpoint_config.get("min_replicas", 1),
                autoscale_max_replicas=endpoint_config.get("max_replicas", 10)
            )
        
        # Deploy service
        service = Model.deploy(
            workspace=self.workspace,
            name=endpoint_config["service_name"],
            models=[model],
            inference_config=inference_config,
            deployment_config=deployment_config,
            overwrite=True
        )
        
        service.wait_for_deployment(show_output=True)
        return service
    
    def create_batch_endpoint(self, model, batch_config):
        """Create batch inference endpoint."""
        
        from azureml.core import Environment
        from azureml.core.compute import ComputeTarget
        from azureml.pipeline.core import Pipeline
        from azureml.pipeline.steps import ParallelRunStep, ParallelRunConfig
        
        # Parallel run configuration
        parallel_config = ParallelRunConfig(
            source_directory=batch_config["source_directory"],
            entry_script=batch_config["batch_script"],
            mini_batch_size=batch_config.get("mini_batch_size", "1000KB"),
            error_threshold=batch_config.get("error_threshold", 10),
            output_action="append_row",
            environment=batch_config["environment"],
            compute_target=batch_config["compute_target"],
            node_count=batch_config.get("node_count", 4)
        )
        
        # Create parallel run step
        batch_step = ParallelRunStep(
            name="chemiaccel-batch-inference",
            parallel_run_config=parallel_config,
            inputs=[batch_config["input_dataset"]],
            output=batch_config["output_dataset"],
            allow_reuse=False
        )
        
        # Create and publish pipeline
        pipeline = Pipeline(workspace=self.workspace, steps=[batch_step])
        published_pipeline = pipeline.publish(
            name="ChemiAccel Batch Inference Pipeline",
            description="Batch inference pipeline for molecular descriptor calculation"
        )
        
        return published_pipeline

# Example: Deploy on Azure ML
azure_service = ChemiAccelAzureML(
    subscription_id="your-subscription-id",
    resource_group="chemiaccel-rg",
    workspace_name="chemiaccel-workspace"
)

# Create compute cluster
cluster_config = {
    "name": "chemiaccel-gpu-cluster",
    "vm_size": "Standard_NC12s_v3",
    "min_nodes": 0,
    "max_nodes": 20,
    "idle_seconds": 1800
}

compute_cluster = azure_service.create_compute_cluster(cluster_config)
print(f"Compute cluster created: {compute_cluster.name}")

# Create training environment
training_env = azure_service.create_training_environment()
training_env.register(azure_service.workspace)
print(f"Training environment registered: {training_env.name}")

# Submit training job
training_config = {
    "experiment_name": "chemiaccel-training",
    "source_directory": "./training",
    "training_script": "train.py",
    "compute_target": compute_cluster,
    "environment": training_env,
    "distributed": True,
    "node_count": 2,
    "arguments": [
        "--data-path", "data/molecules.csv",
        "--output-path", "outputs/model",
        "--epochs", "100"
    ]
}

training_run = azure_service.submit_training_job(training_config)
print(f"Training job submitted: {training_run.id}")
```

## Best Practices

### 1. Cost Optimization
```python
# Use spot instances for training workloads
spot_config = {
    "instance_type": "g4dn.xlarge",
    "spot_price": "0.50",  # Maximum price
    "interruption_handling": "terminate"
}
```

### 2. Security Configuration
```python
# Always use IAM roles and least privilege access
security_config = {
    "vpc_config": {
        "security_group_ids": ["sg-private"],
        "subnet_ids": ["subnet-private-1", "subnet-private-2"]
    },
    "encryption": {
        "kms_key_id": "arn:aws:kms:region:account:key/key-id",
        "encrypt_inter_container_traffic": True
    }
}
```

### 3. Monitoring and Logging
```python
# Implement comprehensive monitoring
monitoring_config = {
    "cloudwatch_logs": True,
    "metrics_enabled": True,
    "custom_metrics": ["queue_depth", "processing_rate", "error_rate"],
    "alerts": [
        {"metric": "error_rate", "threshold": 0.05, "action": "scale_up"},
        {"metric": "queue_depth", "threshold": 100, "action": "scale_up"}
    ]
}
```

---

*For distributed computing and HPC integration, see the [Distributed Computing](Distributed-Computing) and [HPC Integration](HPC-Integration) guides.*

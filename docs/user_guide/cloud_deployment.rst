Cloud Deployment Guide
======================

This guide covers deploying ChemiAccel workflows on major cloud platforms including AWS, Google Cloud Platform, and Microsoft Azure.

Overview
--------

ChemiAccel provides native integrations with cloud ML platforms for:

* **Scalable batch processing** of large molecular datasets
* **GPU-accelerated computation** using cloud GPU instances
* **Serverless processing** for cost-effective smaller workloads
* **Multi-cloud deployment** for optimal cost and performance
* **Integration with ML platforms** like SageMaker, Vertex AI, Azure ML

Architecture Patterns
----------------------

Single Cloud Deployment
~~~~~~~~~~~~~~~~~~~~~~~~

Best for organizations already committed to a single cloud provider:

.. code-block:: text

    [Data Input] → [Cloud Storage] → [ML Platform] → [ChemiAccel Processing] → [Results Storage]

Multi-Cloud Hybrid
~~~~~~~~~~~~~~~~~~~

Optimal cost and performance by leveraging multiple providers:

.. code-block:: text

    [Load Balancer] → [AWS SageMaker] → [ChemiAccel CUDA]
                   → [GCP Vertex AI] → [ChemiAccel JAX/TPU]  
                   → [Azure ML] → [ChemiAccel CPU/GPU]

Edge-Cloud Hybrid
~~~~~~~~~~~~~~~~~~

Local preprocessing with cloud-scale processing:

.. code-block:: text

    [Local Processing] → [Data Upload] → [Cloud Batch] → [Results Download]

AWS Deployment
--------------

Prerequisites
~~~~~~~~~~~~~

1. AWS CLI installed and configured
2. Appropriate IAM permissions for SageMaker, S3, and EC2
3. Docker installed for container builds

Quick Start with SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.ml_platforms import submit_chemi_accel_job, MLJobConfig

    # Configure job
    job_config = MLJobConfig(
        platform='sagemaker',
        instance_type='ml.g4dn.xlarge',
        instance_count=2,
        input_data_uri='s3://my-bucket/compounds.csv',
        output_data_uri='s3://my-bucket/results/',
        max_runtime_seconds=3600
    )

    # Submit job
    job_name = submit_chemi_accel_job(
        job_config=job_config,
        descriptor_config={
            'descriptors': ['molecular_weight', 'logp', 'tpsa', 'morgan_fingerprint'],
            'backend': 'cuda',
            'batch_size': 1000
        }
    )

    print(f"Job submitted: {job_name}")

Custom SageMaker Training
~~~~~~~~~~~~~~~~~~~~~~~~~

For custom models and workflows:

.. code-block:: python

    import sagemaker
    from sagemaker.estimator import Estimator

    # Define custom estimator
    estimator = Estimator(
        image_uri='your-account.dkr.ecr.us-west-2.amazonaws.com/chemi-accel:latest',
        role=sagemaker.get_execution_role(),
        instance_count=2,
        instance_type='ml.g4dn.xlarge',
        base_job_name='chemi-accel-training'
    )

    # Set hyperparameters
    estimator.set_hyperparameters(
        backend='cuda',
        batch_size=5000,
        descriptors='molecular_weight,logp,tpsa',
        output_format='parquet'
    )

    # Start training
    estimator.fit({
        'training': 's3://my-bucket/training-data/',
        'validation': 's3://my-bucket/validation-data/'
    })

Batch Processing with AWS Batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large-scale processing:

.. code-block:: bash

    # Create compute environment
    aws batch create-compute-environment \
        --compute-environment-name chemi-accel-gpu \
        --type MANAGED \
        --state ENABLED \
        --compute-resources type=EC2,minvCpus=0,maxvCpus=256,desiredvCpus=0,instanceTypes=g4dn.xlarge

    # Create job queue
    aws batch create-job-queue \
        --job-queue-name chemi-accel-queue \
        --state ENABLED \
        --priority 1 \
        --compute-environment-order order=1,computeEnvironment=chemi-accel-gpu

    # Submit job
    aws batch submit-job \
        --job-name chemi-accel-descriptors \
        --job-queue chemi-accel-queue \
        --job-definition chemi-accel-job-def \
        --parameters inputS3Path=s3://my-bucket/compounds.csv,outputS3Path=s3://my-bucket/results/

Google Cloud Platform Deployment
---------------------------------

Prerequisites
~~~~~~~~~~~~~

1. Google Cloud SDK installed
2. Project with Vertex AI API enabled
3. Service account with appropriate permissions

Vertex AI Custom Jobs
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from google.cloud import aiplatform

    # Initialize Vertex AI
    aiplatform.init(
        project='my-gcp-project',
        location='us-central1'
    )

    # Create custom job
    job = aiplatform.CustomJob(
        display_name='ChemiAccel Molecular Descriptors',
        worker_pool_specs=[{
            'machine_spec': {
                'machine_type': 'n1-standard-4',
                'accelerator_type': 'NVIDIA_TESLA_T4',
                'accelerator_count': 1
            },
            'replica_count': 2,
            'container_spec': {
                'image_uri': 'gcr.io/my-project/chemi-accel:latest',
                'args': [
                    '--input-path=gs://my-bucket/compounds.csv',
                    '--output-path=gs://my-bucket/results/',
                    '--backend=cuda'
                ]
            }
        }]
    )

    # Submit job
    job.run(sync=False)

TPU Processing with JAX
~~~~~~~~~~~~~~~~~~~~~~~

For TPU-accelerated processing:

.. code-block:: python

    from chemi_accel import Calculator

    # Configure for TPU
    calc = Calculator(backend='jax')

    # Process on TPU
    import jax
    print(f"JAX devices: {jax.devices()}")

    # Large-scale processing
    results = calc.jax(large_smiles_list)

Cloud Run for Serverless
~~~~~~~~~~~~~~~~~~~~~~~~

Deploy as serverless container:

.. code-block:: bash

    # Build container
    docker build -t gcr.io/my-project/chemi-accel-api .

    # Push to registry
    docker push gcr.io/my-project/chemi-accel-api

    # Deploy to Cloud Run
    gcloud run deploy chemi-accel-api \
        --image gcr.io/my-project/chemi-accel-api \
        --platform managed \
        --region us-central1 \
        --memory 4Gi \
        --cpu 2 \
        --max-instances 100

Microsoft Azure Deployment
---------------------------

Prerequisites
~~~~~~~~~~~~~

1. Azure CLI installed
2. Azure ML workspace created
3. Appropriate RBAC permissions

Azure ML Compute Jobs
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from azure.ai.ml import MLClient, command
    from azure.identity import DefaultAzureCredential

    # Connect to workspace
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id="your-subscription-id",
        resource_group_name="my-resource-group",
        workspace_name="my-ml-workspace"
    )

    # Define job
    job = command(
        inputs={
            "input_data": "azureml:molecular_dataset:1",
            "descriptors": "molecular_weight,logp,tpsa"
        },
        code="./src",
        command="python main.py --input ${{inputs.input_data}} --descriptors ${{inputs.descriptors}}",
        environment="azureml:chemi-accel-gpu:1",
        compute="gpu-cluster",
        instance_count=2
    )

    # Submit job
    ml_client.create_or_update(job)

Container Instances
~~~~~~~~~~~~~~~~~~~

For simpler deployments:

.. code-block:: bash

    # Create container group
    az container create \
        --resource-group my-resource-group \
        --name chemi-accel-batch \
        --image myregistry.azurecr.io/chemi-accel:latest \
        --cpu 4 \
        --memory 16 \
        --gpu-count 1 \
        --gpu-sku V100 \
        --environment-variables \
            CHEMI_ACCEL_BACKEND=cuda \
            INPUT_PATH=https://myaccount.blob.core.windows.net/data/compounds.csv

Multi-Cloud Deployment
-----------------------

Cost Optimization Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.ml_platforms import MultiCloudOptimizer

    optimizer = MultiCloudOptimizer()

    # Define workload
    workload = {
        'molecules': 1000000,
        'descriptors': ['molecular_weight', 'logp', 'tpsa', 'morgan_fingerprint'],
        'deadline': '2024-12-31T23:59:59Z',
        'budget': 500  # USD
    }

    # Get recommendations
    recommendations = optimizer.optimize_deployment(workload)

    print("Optimal deployment strategy:")
    for provider, config in recommendations.items():
        print(f"{provider}: {config['instances']} x {config['instance_type']}")
        print(f"  Cost: ${config['cost']:.2f}")
        print(f"  Time: {config['estimated_time']:.1f} hours")

Load Balancing
~~~~~~~~~~~~~~

Distribute workload across providers:

.. code-block:: python

    from chemi_accel.cloud.distributed_computing import MultiCloudLoadBalancer

    balancer = MultiCloudLoadBalancer()

    # Configure providers
    providers = {
        'aws': {'weight': 0.4, 'max_instances': 10},
        'gcp': {'weight': 0.4, 'max_instances': 8},
        'azure': {'weight': 0.2, 'max_instances': 6}
    }

    # Submit distributed job
    job_id = balancer.submit_distributed_job(
        providers=providers,
        input_data='s3://shared-bucket/compounds.csv',
        output_location='gs://results-bucket/',
        job_config={
            'descriptors': ['molecular_weight', 'logp', 'tpsa'],
            'backend': 'cuda'
        }
    )

    # Monitor progress
    status = balancer.get_job_status(job_id)
    print(f"Overall progress: {status['progress']:.1%}")

Best Practices
--------------

Cost Management
~~~~~~~~~~~~~~~

1. **Use spot instances** for non-critical workloads
2. **Implement auto-scaling** based on queue depth
3. **Choose appropriate instance types** based on workload characteristics
4. **Monitor usage** and set billing alerts

.. code-block:: python

    # Spot instance configuration
    spot_config = {
        'aws': {'spot_fleet': True, 'max_price': 0.5},
        'gcp': {'preemptible': True},
        'azure': {'low_priority': True}
    }

Security
~~~~~~~~

1. **Use IAM roles** instead of access keys
2. **Encrypt data** in transit and at rest
3. **Implement VPC/network isolation**
4. **Regular security audits**

.. code-block:: python

    # Security configuration
    security_config = {
        'encryption': {
            'in_transit': True,
            'at_rest': True,
            'key_management': 'customer_managed'
        },
        'network': {
            'vpc_isolation': True,
            'private_subnets': True
        },
        'access_control': {
            'iam_roles': True,
            'mfa_required': True
        }
    }

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Data locality** - process data near storage
2. **Batch size tuning** for optimal GPU utilization
3. **Parallel processing** across multiple instances
4. **Caching** frequently accessed data

.. code-block:: python

    # Performance optimization
    perf_config = {
        'data_locality': True,
        'batch_size': 'auto',  # Auto-tune based on GPU memory
        'parallel_workers': 4,
        'cache_size': '8GB'
    }

Monitoring and Logging
~~~~~~~~~~~~~~~~~~~~~~

Implement comprehensive monitoring:

.. code-block:: python

    from chemi_accel.cloud.monitoring import CloudMonitor

    monitor = CloudMonitor()

    # Configure metrics
    metrics = [
        'molecules_processed_per_second',
        'gpu_utilization',
        'memory_usage',
        'cost_per_molecule',
        'error_rate'
    ]

    # Set up alerts
    alerts = {
        'high_error_rate': {'threshold': 0.05, 'action': 'email'},
        'high_cost': {'threshold': 100, 'action': 'pause_jobs'},
        'low_throughput': {'threshold': 100, 'action': 'scale_up'}
    }

    monitor.configure_monitoring(metrics, alerts)

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Out of Memory Errors:**

.. code-block:: python

    # Reduce batch size
    calc = Calculator(backend='cuda', config={'batch_size': 500})

    # Enable memory monitoring
    calc.backend.enable_memory_monitoring()

**Timeout Issues:**

.. code-block:: python

    # Increase timeout
    job_config.max_runtime_seconds = 7200  # 2 hours

    # Use checkpointing
    job_config.enable_checkpointing = True
    job_config.checkpoint_interval = 1800  # 30 minutes

**Network Connectivity:**

.. code-block:: bash

    # Test connectivity
    curl -I https://s3.amazonaws.com/my-bucket/

    # Check VPC configuration
    aws ec2 describe-vpcs
    aws ec2 describe-security-groups

**Cost Overruns:**

.. code-block:: python

    # Set budget alerts
    budget_config = {
        'monthly_limit': 1000,
        'alert_thresholds': [0.5, 0.8, 0.95],
        'auto_stop_at': 1.0
    }

For additional help, see the :doc:`../troubleshooting` guide or contact support.
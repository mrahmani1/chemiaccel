Cloud Computing API Reference
==============================

Cloud-native workflows and ML platform integrations for scalable molecular discovery.

ML Platform Integration
-----------------------

.. automodule:: chemi_accel.cloud.ml_platforms
    :members:
    :undoc-members:
    :show-inheritance:

Batch Processing
----------------

.. automodule:: chemi_accel.cloud.batch_processor
    :members:
    :undoc-members:
    :show-inheritance:

Distributed Computing
---------------------

.. automodule:: chemi_accel.cloud.distributed_computing
    :members:
    :undoc-members:
    :show-inheritance:

HPC Integration
---------------

.. automodule:: chemi_accel.cloud.hpc_integration
    :members:
    :undoc-members:
    :show-inheritance:

Cloud Storage
-------------

.. automodule:: chemi_accel.cloud.storage
    :members:
    :undoc-members:
    :show-inheritance:

Container Integration
---------------------

.. automodule:: chemi_accel.cloud.containers
    :members:
    :undoc-members:
    :show-inheritance:

Cloud Run Integration
---------------------

.. automodule:: chemi_accel.cloud.cloud_run_integration
    :members:
    :undoc-members:
    :show-inheritance:

Usage Examples
--------------

SageMaker Integration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.ml_platforms import SageMakerIntegration

    sagemaker = SageMakerIntegration(
        role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
        region='us-west-2'
    )
    
    # Submit batch job
    job_config = {
        'instance_type': 'ml.g4dn.xlarge',
        'instance_count': 2,
        'input_data_uri': 's3://my-bucket/compounds.csv',
        'output_data_uri': 's3://my-bucket/results/',
        'max_runtime_seconds': 3600
    }
    
    job_name = sagemaker.submit_batch_job(
        job_config=job_config,
        descriptor_config={
            'descriptors': ['molecular_weight', 'logp', 'tpsa', 'morgan_fingerprint'],
            'backend': 'cuda'
        }
    )
    
    print(f"Submitted job: {job_name}")
    
    # Monitor job progress
    status = sagemaker.get_job_status(job_name)
    print(f"Job status: {status}")

Vertex AI Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.ml_platforms import VertexAIIntegration

    vertex_ai = VertexAIIntegration(
        project_id='my-gcp-project',
        region='us-central1'
    )
    
    # Custom training job
    training_config = {
        'machine_type': 'n1-standard-4',
        'accelerator_type': 'NVIDIA_TESLA_T4',
        'accelerator_count': 1,
        'python_package_uris': ['gs://my-bucket/chemi-accel-package.tar.gz']
    }
    
    job_id = vertex_ai.submit_custom_job(
        display_name='ChemiAccel Property Prediction',
        training_config=training_config,
        args=['--dataset=gs://my-bucket/training_data.csv']
    )
    
    print(f"Training job ID: {job_id}")

Azure ML Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.ml_platforms import AzureMLIntegration

    azure_ml = AzureMLIntegration(
        subscription_id='your-subscription-id',
        resource_group='my-resource-group',
        workspace_name='my-ml-workspace'
    )
    
    # Submit compute job
    compute_config = {
        'compute_target': 'gpu-cluster',
        'vm_size': 'Standard_NC6s_v3',
        'min_nodes': 0,
        'max_nodes': 4
    }
    
    experiment_name = azure_ml.create_experiment('molecular-descriptors')
    run = azure_ml.submit_run(
        experiment_name=experiment_name,
        compute_config=compute_config,
        script_path='train_model.py'
    )
    
    print(f"Run ID: {run.id}")

Distributed Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.batch_processor import DistributedBatchProcessor

    processor = DistributedBatchProcessor(
        backend='cuda',
        cluster_config={
            'nodes': 4,
            'gpus_per_node': 2,
            'memory_per_node': '32GB'
        }
    )
    
    # Process large dataset
    large_dataset = 's3://my-bucket/million_compounds.csv'
    
    job_id = processor.submit_batch_job(
        input_data=large_dataset,
        output_location='s3://my-bucket/results/',
        descriptors=['molecular_weight', 'logp', 'tpsa'],
        chunk_size=10000
    )
    
    # Monitor progress
    progress = processor.get_progress(job_id)
    print(f"Progress: {progress['completed']}/{progress['total']} chunks")

HPC Cluster Integration  
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.hpc_integration import SLURMIntegration

    slurm = SLURMIntegration(
        cluster_endpoint='login.hpc.university.edu',
        username='researcher'
    )
    
    # Submit SLURM job
    job_script = """
    #!/bin/bash
    #SBATCH --job-name=chemi-accel
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=4
    #SBATCH --gres=gpu:4
    #SBATCH --time=02:00:00
    
    module load cuda/11.2
    module load python/3.9
    
    python -m chemi_accel.distributed \
        --backend cuda \
        --input-file compounds.csv \
        --output-file results.csv \
        --descriptors molecular_weight,logp,tpsa
    """
    
    job_id = slurm.submit_job(job_script)
    print(f"SLURM job ID: {job_id}")
    
    # Check job status
    status = slurm.get_job_status(job_id)
    print(f"Job status: {status}")

Cloud Storage Integration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.storage import CloudStorage

    # S3 storage
    s3_storage = CloudStorage(provider='aws')
    
    # Upload dataset
    s3_storage.upload_file(
        local_path='local_compounds.csv',
        remote_path='s3://my-bucket/compounds.csv'
    )
    
    # Download results
    s3_storage.download_file(
        remote_path='s3://my-bucket/results/descriptors.csv',
        local_path='results.csv'
    )
    
    # Google Cloud Storage
    gcs_storage = CloudStorage(provider='gcp')
    
    # List files
    files = gcs_storage.list_files('gs://my-bucket/datasets/')
    print(f"Found {len(files)} files")

Container Deployment
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.containers import DockerDeployment

    docker = DockerDeployment()
    
    # Build custom image
    dockerfile_content = """
    FROM chemi-accel/base:latest
    
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    COPY my_analysis.py .
    CMD ["python", "my_analysis.py"]
    """
    
    image_name = docker.build_image(
        dockerfile_content=dockerfile_content,
        image_name='my-chemi-accel-analysis',
        tag='v1.0'
    )
    
    # Deploy to cloud
    deployment = docker.deploy_to_cloud(
        image_name=image_name,
        cloud_provider='aws',
        instance_type='g4dn.xlarge'
    )
    
    print(f"Deployment URL: {deployment['endpoint']}")

Serverless Processing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.batch_processor import ServerlessBatchProcessor

    serverless = ServerlessBatchProcessor(provider='aws')
    
    # Create serverless function
    function_config = {
        'runtime': 'python3.9',
        'memory': 3008,  # MB
        'timeout': 900,  # seconds
        'environment': {
            'CHEMI_ACCEL_BACKEND': 'cpu'
        }
    }
    
    function_arn = serverless.create_function(
        function_name='chemi-accel-processor',
        config=function_config,
        code_package='chemi_accel_lambda.zip'
    )
    
    # Process data in batches
    results = serverless.process_batch(
        function_arn=function_arn,
        input_data=['CCO', 'CCN', 'CC(=O)O'],
        batch_size=100
    )
    
    print(f"Processed {len(results)} molecules")

Cost Optimization
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.ml_platforms import CostOptimizer

    optimizer = CostOptimizer()
    
    # Estimate costs for different configurations
    configurations = [
        {'provider': 'aws', 'instance': 'g4dn.xlarge', 'hours': 2},
        {'provider': 'gcp', 'instance': 'n1-standard-4-gpu', 'hours': 2},
        {'provider': 'azure', 'instance': 'Standard_NC6s_v3', 'hours': 2}
    ]
    
    cost_estimates = optimizer.estimate_costs(configurations)
    
    print("Cost Estimates:")
    for config, cost in cost_estimates.items():
        print(f"  {config}: ${cost:.2f}")
    
    # Recommend optimal configuration
    optimal = optimizer.recommend_optimal_config(
        workload_size=100000,  # molecules
        time_constraint=3600,  # seconds
        budget=100  # dollars
    )
    
    print(f"Recommended: {optimal['provider']} {optimal['instance']}")

Multi-Cloud Deployment
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.cloud.ml_platforms import MultiCloudManager

    manager = MultiCloudManager()
    
    # Configure multiple providers
    providers = {
        'aws': {'region': 'us-west-2', 'credentials': 'aws_creds'},
        'gcp': {'region': 'us-central1', 'credentials': 'gcp_creds'},
        'azure': {'region': 'eastus', 'credentials': 'azure_creds'}
    }
    
    manager.configure_providers(providers)
    
    # Distribute workload across clouds
    workload = {
        'total_molecules': 1000000,
        'descriptors': ['molecular_weight', 'logp', 'tpsa'],
        'deadline': '2024-12-31T23:59:59Z'
    }
    
    deployment_plan = manager.create_deployment_plan(workload)
    
    print("Multi-cloud deployment plan:")
    for provider, allocation in deployment_plan.items():
        print(f"  {provider}: {allocation['molecules']} molecules")
        print(f"    Instance: {allocation['instance_type']}")
        print(f"    Estimated cost: ${allocation['cost']:.2f}")
    
    # Execute deployment
    results = manager.execute_deployment(deployment_plan)
    print(f"Total processing time: {results['total_time']:.1f} seconds")
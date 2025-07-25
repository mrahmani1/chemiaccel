#!/usr/bin/env python3
"""
ML Platform Workflow Examples for ChemiAccel

This module demonstrates how to use ChemiAccel with various cloud ML platforms
including Amazon SageMaker, Google Vertex AI, and Azure ML.
"""


import pandas as pd
from chemi_accel.cloud.ml_platforms import (
    AzureMLIntegration,
    MLJobConfig,
    SageMakerIntegration,
    VertexAIIntegration,
    submit_chemi_accel_job,
)


def sagemaker_processing_example():
    """Example: Submit ChemiAccel job to Amazon SageMaker."""
    print("=== Amazon SageMaker Processing Job Example ===")

    # Configure job
    job_config = MLJobConfig(
        job_name="chemi-accel-sagemaker-demo",
        instance_type="ml.g4dn.xlarge",  # GPU instance
        instance_count=1,
        input_data_uri="s3://my-bucket/molecular-data/compounds.csv",
        output_data_uri="s3://my-bucket/results/",
        environment_variables={
            "CUDA_VISIBLE_DEVICES": "0",
            "CHEMI_ACCEL_BACKEND": "cuda",
        },
        max_runtime_seconds=3600,
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    )

    # Descriptor configuration
    descriptor_config = {
        "descriptors": [
            "molecular_weight",
            "logp",
            "tpsa",
            "num_rotatable_bonds",
            "num_aromatic_rings",
            "morgan_fingerprint",
        ],
        "batch_size": 1000,
        "backend": "cuda",
    }

    # Submit job using convenience function
    job_name = submit_chemi_accel_job(
        platform="sagemaker",
        job_config=job_config,
        descriptor_config=descriptor_config,
        region="us-east-1",
        role_arn=job_config.role_arn,
    )

    print(f"Submitted SageMaker job: {job_name}")

    # Monitor job status
    integration = SageMakerIntegration(region="us-east-1", role_arn=job_config.role_arn)

    import time

    while True:
        status = integration.get_job_status(job_name)
        print(f"Job status: {status.status}")

        if status.status in ["Completed", "Failed", "Stopped"]:
            break

        time.sleep(30)

    if status.status == "Completed":
        print("Job completed successfully!")
        print("Results available at:", job_config.output_data_uri)
    else:
        print(f"Job failed: {status.failure_reason}")
        logs = integration.get_job_logs(job_name)
        print("Logs:", logs)


def sagemaker_pipeline_example():
    """Example: Create SageMaker Pipeline with ChemiAccel step."""
    print("=== Amazon SageMaker Pipeline Example ===")

    try:
        from sagemaker.workflow.parameters import ParameterString
        from sagemaker.workflow.pipeline import Pipeline
    except ImportError:
        print("SageMaker SDK not available. Install with: pip install sagemaker")
        return

    # Initialize SageMaker integration
    integration = SageMakerIntegration(
        region="us-east-1",
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    )

    # Job configuration
    job_config = MLJobConfig(
        job_name="chemi-accel-pipeline-step",
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    )

    # Descriptor configuration
    descriptor_config = {
        "descriptors": ["molecular_weight", "logp", "tpsa"],
        "batch_size": 1000,
    }

    # Create pipeline step
    step_result = integration.create_pipeline_step(
        "molecular-descriptors", job_config, descriptor_config
    )

    # Pipeline parameters
    input_data_param = ParameterString(
        name="InputData", default_value="s3://my-bucket/input/"
    )

    output_data_param = ParameterString(
        name="OutputData", default_value="s3://my-bucket/output/"
    )

    # Create pipeline
    pipeline = Pipeline(
        name="chemi-accel-molecular-descriptors",
        parameters=[input_data_param, output_data_param],
        steps=[step_result["step"]],
    )

    # Submit pipeline
    pipeline.create(role_arn=job_config.role_arn)
    execution = pipeline.start()

    print(f"Started pipeline execution: {execution.arn}")


def vertex_ai_example():
    """Example: Submit ChemiAccel job to Google Vertex AI."""
    print("=== Google Vertex AI Custom Job Example ===")

    # Configure job
    job_config = MLJobConfig(
        job_name="chemi-accel-vertex-demo",
        instance_type="n1-standard-4",  # With GPU
        instance_count=1,
        input_data_uri="gs://my-bucket/molecular-data/compounds.csv",
        output_data_uri="gs://my-bucket/results/",
        environment_variables={
            "CUDA_VISIBLE_DEVICES": "0",
            "CHEMI_ACCEL_BACKEND": "cuda",
        },
    )

    # Descriptor configuration
    descriptor_config = {
        "descriptors": [
            "constitutional_descriptors",
            "topological_descriptors",
            "geometrical_3d_descriptors",
        ],
        "batch_size": 500,
        "backend": "cuda",
    }

    # Submit job
    job_name = submit_chemi_accel_job(
        platform="vertex",
        job_config=job_config,
        descriptor_config=descriptor_config,
        project_id="my-gcp-project",
        region="us-central1",
    )

    print(f"Submitted Vertex AI job: {job_name}")

    # Monitor job
    integration = VertexAIIntegration(project_id="my-gcp-project", region="us-central1")

    import time

    while True:
        status = integration.get_job_status(job_name)
        print(f"Job status: {status.status}")

        if "SUCCEEDED" in status.status or "FAILED" in status.status:
            break

        time.sleep(30)

    print("Job completed!")


def vertex_ai_pipeline_example():
    """Example: Create Vertex AI Pipeline with ChemiAccel component."""
    print("=== Google Vertex AI Pipeline Example ===")

    try:
        from google.cloud import aiplatform
        from kfp.v2 import dsl
    except ImportError:
        print(
            "Vertex AI SDK not available. Install with: pip install google-cloud-aiplatform kfp"
        )
        return

    # Initialize Vertex AI
    aiplatform.init(project="my-gcp-project", location="us-central1")

    # Initialize integration
    integration = VertexAIIntegration(project_id="my-gcp-project", region="us-central1")

    # Job configuration
    job_config = MLJobConfig(
        job_name="chemi-accel-vertex-pipeline",
        instance_type="n1-standard-4",
        instance_count=1,
    )

    # Create component
    component_result = integration.create_pipeline_step(
        "molecular-descriptors",
        job_config,
        {"descriptors": ["molecular_weight", "logp"], "batch_size": 1000},
    )

    # Define pipeline
    @dsl.pipeline(
        name="chemi-accel-molecular-analysis",
        description="Molecular descriptor calculation pipeline",
    )
    def molecular_analysis_pipeline(
        input_data: str = "gs://my-bucket/input/",
        output_data: str = "gs://my-bucket/output/",
    ):
        descriptor_step = component_result["component"](
            input_data=input_data, output_data=output_data
        )
        return descriptor_step.outputs

    # Compile and run pipeline
    from kfp.v2 import compiler

    compiler.Compiler().compile(
        pipeline_func=molecular_analysis_pipeline,
        package_path="molecular_analysis_pipeline.json",
    )

    # Submit pipeline
    job = aiplatform.PipelineJob(
        display_name="chemi-accel-molecular-analysis",
        template_path="molecular_analysis_pipeline.json",
        parameter_values={
            "input_data": "gs://my-bucket/input/",
            "output_data": "gs://my-bucket/output/",
        },
    )

    job.submit()
    print(f"Submitted Vertex AI pipeline: {job.display_name}")


def azure_ml_example():
    """Example: Submit ChemiAccel job to Azure ML."""
    print("=== Azure ML Command Job Example ===")

    # Configure job
    job_config = MLJobConfig(
        job_name="chemi-accel-azure-demo",
        instance_type="Standard_NC6",  # GPU instance
        instance_count=1,
        input_data_uri="azureml://datastores/workspaceblobstore/paths/molecular-data/",
        output_data_uri="azureml://datastores/workspaceblobstore/paths/results/",
        environment_variables={
            "CUDA_VISIBLE_DEVICES": "0",
            "CHEMI_ACCEL_BACKEND": "cuda",
        },
    )

    # Descriptor configuration
    descriptor_config = {
        "descriptors": ["admet_descriptors", "drug_likeness_scores"],
        "batch_size": 800,
        "backend": "cuda",
    }

    # Submit job
    job_name = submit_chemi_accel_job(
        platform="azure",
        job_config=job_config,
        descriptor_config=descriptor_config,
        subscription_id="my-subscription-id",
        resource_group="my-resource-group",
        workspace_name="my-ml-workspace",
    )

    print(f"Submitted Azure ML job: {job_name}")

    # Monitor job
    integration = AzureMLIntegration(
        subscription_id="my-subscription-id",
        resource_group="my-resource-group",
        workspace_name="my-ml-workspace",
    )

    import time

    while True:
        status = integration.get_job_status(job_name)
        print(f"Job status: {status.status}")

        if status.status in ["Completed", "Failed", "Canceled"]:
            break

        time.sleep(30)

    print("Job completed!")


def azure_ml_pipeline_example():
    """Example: Create Azure ML Pipeline with ChemiAccel component."""
    print("=== Azure ML Pipeline Example ===")

    try:
        from azure.ai.ml import dsl
        from azure.ai.ml.entities import Pipeline
    except ImportError:
        print("Azure ML SDK not available. Install with: pip install azure-ai-ml")
        return

    # Initialize integration
    integration = AzureMLIntegration(
        subscription_id="my-subscription-id",
        resource_group="my-resource-group",
        workspace_name="my-ml-workspace",
    )

    # Job configuration
    job_config = MLJobConfig(
        job_name="chemi-accel-azure-pipeline",
        instance_type="Standard_NC6",
        instance_count=1,
    )

    # Create component
    component_result = integration.create_pipeline_step(
        "molecular-descriptors",
        job_config,
        {"descriptors": ["polymer_descriptors"], "batch_size": 500},
    )

    # Define pipeline
    @dsl.pipeline(
        name="chemi-accel-polymer-analysis",
        description="Polymer descriptor calculation pipeline",
    )
    def polymer_analysis_pipeline(
        input_data: dsl.Input(type="uri_folder"),
        output_data: dsl.Output(type="uri_folder"),
    ):
        descriptor_step = component_result["component"](
            input_data=input_data, output_data=output_data
        )
        return {"processed_data": descriptor_step.outputs.output_data}

    # Create pipeline
    pipeline = Pipeline(
        name="chemi-accel-polymer-analysis",
        description="Polymer molecular analysis pipeline",
        jobs=polymer_analysis_pipeline,
    )

    # Submit pipeline
    pipeline_job = integration.ml_client.jobs.create_or_update(pipeline)
    print(f"Submitted Azure ML pipeline: {pipeline_job.name}")


def multi_platform_comparison():
    """Example: Compare performance across multiple ML platforms."""
    print("=== Multi-Platform Performance Comparison ===")

    # Sample molecular data
    molecules = [
        "CCO",  # ethanol
        "CC(=O)O",  # acetic acid
        "c1ccccc1",  # benzene
        "CCN(CC)CC",  # triethylamine
        "CC(C)O",  # isopropanol
    ] * 1000  # Scale up for performance testing

    # Create test dataset
    df = pd.DataFrame({"smiles": molecules})

    # Common descriptor configuration
    descriptor_config = {
        "descriptors": [
            "molecular_weight",
            "logp",
            "tpsa",
            "num_rotatable_bonds",
            "morgan_fingerprint",
            "constitutional_descriptors",
        ],
        "batch_size": 1000,
        "backend": "cuda",
    }

    platforms = [
        {
            "name": "SageMaker",
            "platform": "sagemaker",
            "instance_type": "ml.g4dn.xlarge",
            "kwargs": {
                "region": "us-east-1",
                "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
            },
        },
        {
            "name": "Vertex AI",
            "platform": "vertex",
            "instance_type": "n1-standard-4",
            "kwargs": {"project_id": "my-gcp-project", "region": "us-central1"},
        },
        {
            "name": "Azure ML",
            "platform": "azure",
            "instance_type": "Standard_NC6",
            "kwargs": {
                "subscription_id": "my-subscription",
                "resource_group": "my-rg",
                "workspace_name": "my-workspace",
            },
        },
    ]

    results = []

    for platform_config in platforms:
        print(f"\nTesting {platform_config['name']}...")

        # Configure job
        job_config = MLJobConfig(
            job_name=f"chemi-accel-perf-test-{platform_config['name'].lower()}",
            instance_type=platform_config["instance_type"],
            instance_count=1,
            environment_variables={"CUDA_VISIBLE_DEVICES": "0"},
        )

        try:
            # Submit job (in practice, you'd upload data first)
            job_name = submit_chemi_accel_job(
                platform=platform_config["platform"],
                job_config=job_config,
                descriptor_config=descriptor_config,
                **platform_config["kwargs"],
            )

            results.append(
                {
                    "platform": platform_config["name"],
                    "job_name": job_name,
                    "status": "submitted",
                }
            )

            print(f"✓ Submitted job: {job_name}")

        except Exception as e:
            print(f"✗ Failed to submit job: {e}")
            results.append(
                {
                    "platform": platform_config["name"],
                    "job_name": None,
                    "status": "failed",
                    "error": str(e),
                }
            )

    # Print summary
    print("\n=== Submission Summary ===")
    for result in results:
        status_icon = "✓" if result["status"] == "submitted" else "✗"
        print(f"{status_icon} {result['platform']}: {result.get('job_name', 'Failed')}")


def real_world_drug_discovery_pipeline():
    """Example: Real-world drug discovery pipeline using multiple platforms."""
    print("=== Real-World Drug Discovery Pipeline ===")

    # Stage 1: Data preprocessing on SageMaker
    print("Stage 1: Data preprocessing and curation...")

    preprocessing_config = MLJobConfig(
        job_name="drug-discovery-preprocessing",
        instance_type="ml.m5.2xlarge",  # CPU instance for data processing
        instance_count=1,
        input_data_uri="s3://drug-discovery-bucket/raw-compounds/",
        output_data_uri="s3://drug-discovery-bucket/curated-compounds/",
        environment_variables={"CHEMI_ACCEL_BACKEND": "cpu"},
    )

    curation_config = {
        "operations": ["validate_smiles", "remove_salts", "standardize_tautomers"],
        "filters": ["pains", "reactive_groups"],
        "batch_size": 5000,
    }

    # Stage 2: Descriptor calculation on Vertex AI (GPU)
    print("Stage 2: Molecular descriptor calculation...")

    descriptor_job_config = MLJobConfig(
        job_name="drug-discovery-descriptors",
        instance_type="n1-standard-4",  # GPU instance
        instance_count=2,  # Parallel processing
        input_data_uri="gs://drug-discovery-bucket/curated-compounds/",
        output_data_uri="gs://drug-discovery-bucket/descriptors/",
        environment_variables={"CHEMI_ACCEL_BACKEND": "cuda"},
    )

    descriptor_config = {
        "descriptors": [
            "admet_descriptors",
            "drug_likeness_scores",
            "constitutional_descriptors",
            "topological_descriptors",
            "morgan_fingerprint",
            "maccs_keys",
        ],
        "batch_size": 1000,
        "include_3d": True,
    }

    # Stage 3: ADMET prediction on Azure ML
    print("Stage 3: ADMET property prediction...")

    admet_job_config = MLJobConfig(
        job_name="drug-discovery-admet",
        instance_type="Standard_NC12",  # High-performance GPU
        instance_count=1,
        input_data_uri="azureml://datastores/workspaceblobstore/paths/descriptors/",
        output_data_uri="azureml://datastores/workspaceblobstore/paths/admet-predictions/",
        environment_variables={"CHEMI_ACCEL_BACKEND": "cuda"},
    )

    admet_config = {
        "models": ["solubility", "permeability", "clearance", "toxicity"],
        "uncertainty_quantification": True,
        "batch_size": 500,
    }

    # Submit all stages
    stages = [
        (
            "sagemaker",
            preprocessing_config,
            curation_config,
            {
                "region": "us-east-1",
                "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
            },
        ),
        (
            "vertex",
            descriptor_job_config,
            descriptor_config,
            {"project_id": "drug-discovery-project", "region": "us-central1"},
        ),
        (
            "azure",
            admet_job_config,
            admet_config,
            {
                "subscription_id": "drug-discovery-subscription",
                "resource_group": "drug-discovery-rg",
                "workspace_name": "drug-discovery-workspace",
            },
        ),
    ]

    job_names = []
    for platform, job_config, config, kwargs in stages:
        try:
            job_name = submit_chemi_accel_job(
                platform=platform,
                job_config=job_config,
                descriptor_config=config,
                **kwargs,
            )
            job_names.append((platform, job_name))
            print(f"✓ Submitted {platform} job: {job_name}")
        except Exception as e:
            print(f"✗ Failed to submit {platform} job: {e}")

    print(
        f"\nSubmitted {len(job_names)} jobs across {len(set(j[0] for j in job_names))} platforms"
    )
    return job_names


if __name__ == "__main__":
    print("ChemiAccel ML Platform Integration Examples")
    print("=" * 50)

    # Note: These examples require proper cloud credentials and resources
    # Uncomment the examples you want to run:

    # sagemaker_processing_example()
    # sagemaker_pipeline_example()
    # vertex_ai_example()
    # vertex_ai_pipeline_example()
    # azure_ml_example()
    # azure_ml_pipeline_example()
    # multi_platform_comparison()
    # real_world_drug_discovery_pipeline()

    print("\nTo run these examples, ensure you have:")
    print("1. Proper cloud credentials configured")
    print("2. Required ML platform SDKs installed")
    print("3. Cloud storage buckets/containers set up")
    print("4. Appropriate IAM roles and permissions")

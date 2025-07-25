"""
Cloud-native infrastructure for ChemiAccel.

This module provides cloud batch processing, ML platform integration,
HPC cluster support, and distributed computing capabilities for large-scale molecular analysis.
"""

from .batch_processor import CloudBatchProcessor, CloudJobConfig, JobStatus
from .containers import ContainerPackager, build_chemi_accel_container
from .distributed_computing import (
    ClusterConfig,
    DaskIntegration,
    DistributedChemiAccelCalculator,
    RayIntegration,
    distributed_calculate_descriptors,
    process_large_dataset,
    setup_dask_cluster,
    setup_ray_cluster,
)
from .hpc_integration import (
    HPCIntegration,
    HPCJobConfig,
    MPIIntegration,
    ParallelIOManager,
    PBSScheduler,
    SLURMScheduler,
    submit_hpc_job,
)
from .ml_platforms import (
    AzureMLIntegration,
    MLJobConfig,
    MLJobStatus,
    SageMakerIntegration,
    VertexAIIntegration,
    submit_chemi_accel_job,
)
from .storage import (
    CloudStorageStreamer,
    stream_molecules_from_cloud,
    upload_results_to_cloud,
)

__all__ = [
    # Cloud Batch Processing
    "CloudBatchProcessor",
    "CloudJobConfig",
    "JobStatus",
    # ML Platform Integration
    "SageMakerIntegration",
    "VertexAIIntegration",
    "AzureMLIntegration",
    "MLJobConfig",
    "MLJobStatus",
    "submit_chemi_accel_job",
    # Cloud Storage
    "CloudStorageStreamer",
    "stream_molecules_from_cloud",
    "upload_results_to_cloud",
    # Container Packaging
    "ContainerPackager",
    "build_chemi_accel_container",
    # HPC Integration
    "HPCIntegration",
    "MPIIntegration",
    "ParallelIOManager",
    "HPCJobConfig",
    "SLURMScheduler",
    "PBSScheduler",
    "submit_hpc_job",
    # Distributed Computing
    "DaskIntegration",
    "RayIntegration",
    "DistributedChemiAccelCalculator",
    "ClusterConfig",
    "setup_dask_cluster",
    "setup_ray_cluster",
    "distributed_calculate_descriptors",
    "process_large_dataset",
]

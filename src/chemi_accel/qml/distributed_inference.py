"""
Scalable QML inference and distributed execution system.

This module provides massively parallel QML inference across multiple GPUs or TPUs
using Dask/Ray for distributed QML inference with optimized batching and data transfer.
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    dist = None
    mp = None
    DDP = None

try:
    import dask
    import dask.distributed as dd
    from dask import delayed

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dask = None
    dd = None
    delayed = None

try:
    import ray
    from ray.util.multiprocessing import Pool as RayPool

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None
    RayPool = None

try:
    import jax
    import jax.numpy as jnp
    from jax import device_put, pmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    pmap = None
    device_put = None

from ..cloud.distributed_computing import ClusterConfig
from ..utils.distributed import LoadBalancer, PerformanceMetrics, PerformanceMonitor
from .base import BaseQMLModel, QMLModelError, QMLPrediction, QuantumProperty
from .memory_manager import MemoryAwareProcessor, MemoryStrategy
from .on_the_fly import GeometryInput


class DistributionBackend(Enum):
    """Supported distributed computing backends."""

    DASK = "dask"
    RAY = "ray"
    TORCH_DDP = "torch_ddp"
    JAX_PMAP = "jax_pmap"
    MULTIPROCESSING = "multiprocessing"


class ScalingStrategy(Enum):
    """Scaling strategies for distributed inference."""

    DATA_PARALLEL = "data_parallel"  # Distribute data across workers
    MODEL_PARALLEL = "model_parallel"  # Distribute model across workers
    PIPELINE_PARALLEL = "pipeline_parallel"  # Pipeline stages across workers
    HYBRID = "hybrid"  # Combination of strategies


class DataTransferMode(Enum):
    """Data transfer optimization modes."""

    EAGER = "eager"  # Transfer data immediately
    LAZY = "lazy"  # Transfer data when needed
    STREAMING = "streaming"  # Stream data continuously
    CACHED = "cached"  # Cache frequently used data
    ZERO_COPY = "zero_copy"  # Use zero-copy transfers when possible


@dataclass
class DistributedConfig:
    """Configuration for distributed QML inference."""

    # Distribution settings
    backend: DistributionBackend = DistributionBackend.DASK
    scaling_strategy: ScalingStrategy = ScalingStrategy.DATA_PARALLEL
    num_workers: int = 4
    devices_per_worker: int = 1

    # Performance settings
    batch_size: int = 64
    max_batch_size: int = 512
    prefetch_factor: int = 2
    data_transfer_mode: DataTransferMode = DataTransferMode.STREAMING

    # Memory management
    memory_strategy: MemoryStrategy = MemoryStrategy.BALANCED
    max_memory_per_worker: str = "8GB"
    enable_memory_monitoring: bool = True

    # Network and communication
    communication_backend: str = "nccl"  # nccl, gloo, mpi
    compression: bool = True
    compression_algorithm: str = "lz4"

    # Fault tolerance
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_checkpointing: bool = False
    checkpoint_interval: int = 100

    # Load balancing
    enable_dynamic_batching: bool = True
    load_balance_threshold: float = 0.8
    auto_scaling: bool = False
    min_workers: int = 1
    max_workers: int = 16


@dataclass
class WorkerInfo:
    """Information about a distributed worker."""

    worker_id: str
    device_ids: List[str]
    memory_available: float
    compute_capability: float
    status: str = "idle"
    current_batch_size: int = 0
    total_processed: int = 0
    error_count: int = 0
    last_heartbeat: float = field(default_factory=time.time)


@dataclass
class DistributedBatch:
    """Batch prepared for distributed processing."""

    batch_id: str
    geometries: List[GeometryInput]
    properties: List[QuantumProperty]
    worker_assignments: Dict[str, List[int]]  # worker_id -> molecule indices
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: float = field(default_factory=time.time)


@dataclass
class InferenceResult:
    """Result from distributed inference."""

    batch_id: str
    predictions: List[QMLPrediction]
    worker_results: Dict[str, Any]
    execution_metrics: PerformanceMetrics
    transfer_metrics: Dict[str, float]
    errors: List[Dict[str, Any]] = field(default_factory=list)


class OptimizedDataLoader:
    """Optimized data loader for distributed QML inference."""

    def __init__(
        self, config: DistributedConfig, prefetch_factor: int = 2, num_workers: int = 4
    ):
        """
        Initialize optimized data loader.

        Args:
            config: Distributed configuration
            prefetch_factor: Number of batches to prefetch
            num_workers: Number of data loading workers
        """
        self.config = config
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._cache = {}
        self._cache_size = 1000

    def create_optimized_batches(
        self,
        geometries: List[GeometryInput],
        properties: List[QuantumProperty],
        worker_info: Dict[str, WorkerInfo],
    ) -> List[DistributedBatch]:
        """
        Create optimized batches for distributed processing.

        Args:
            geometries: Input molecular geometries
            properties: Properties to predict
            worker_info: Information about available workers

        Returns:
            List of optimized distributed batches
        """
        if not worker_info:
            raise QMLModelError("No workers available for batch creation")

        # Calculate optimal batch sizes per worker
        total_capacity = sum(
            info.compute_capability * self.config.devices_per_worker
            for info in worker_info.values()
        )

        batches = []
        batch_assignments = self._calculate_load_balanced_assignments(
            geometries, worker_info, total_capacity
        )

        # Create batches with worker assignments
        for i, (worker_id, molecule_indices) in enumerate(batch_assignments.items()):
            if not molecule_indices:
                continue

            batch_geometries = [geometries[idx] for idx in molecule_indices]

            batch = DistributedBatch(
                batch_id=f"batch_{i}_{uuid.uuid4().hex[:8]}",
                geometries=batch_geometries,
                properties=properties,
                worker_assignments={worker_id: list(range(len(batch_geometries)))},
                metadata={
                    "original_indices": molecule_indices,
                    "target_worker": worker_id,
                    "estimated_compute_time": self._estimate_compute_time(
                        len(batch_geometries), worker_info[worker_id]
                    ),
                },
            )

            batches.append(batch)

        return batches

    def _calculate_load_balanced_assignments(
        self,
        geometries: List[GeometryInput],
        worker_info: Dict[str, WorkerInfo],
        total_capacity: float,
    ) -> Dict[str, List[int]]:
        """Calculate load-balanced assignments of molecules to workers."""
        assignments = {worker_id: [] for worker_id in worker_info.keys()}

        # Sort workers by capacity (highest first)
        sorted_workers = sorted(
            worker_info.items(),
            key=lambda x: x[1].compute_capability * self.config.devices_per_worker,
            reverse=True,
        )

        # Assign molecules using load balancing
        molecule_indices = list(range(len(geometries)))
        worker_loads = dict.fromkeys(worker_info.keys(), 0.0)

        for mol_idx in molecule_indices:
            # Estimate computational cost for this molecule
            mol_cost = self._estimate_molecule_cost(geometries[mol_idx])

            # Find worker with lowest current load
            best_worker = min(
                sorted_workers,
                key=lambda x: worker_loads[x[0]] / (x[1].compute_capability + 1e-6),
            )[0]

            assignments[best_worker].append(mol_idx)
            worker_loads[best_worker] += mol_cost

        return assignments

    def _estimate_molecule_cost(self, geometry: GeometryInput) -> float:
        """Estimate computational cost for a molecule."""
        # Simple heuristic based on number of atoms
        num_atoms = len(geometry.atomic_numbers)

        # Cost scales approximately with N^2 to N^3 for QML models
        base_cost = num_atoms**2.5

        # Adjust for molecular complexity (more bonds = higher cost)
        complexity_factor = 1.0 + (num_atoms - 1) * 0.1

        return base_cost * complexity_factor

    def _estimate_compute_time(self, batch_size: int, worker: WorkerInfo) -> float:
        """Estimate computation time for a batch on a specific worker."""
        base_time_per_molecule = 0.1  # seconds

        # Scale by worker capability
        time_per_molecule = base_time_per_molecule / (worker.compute_capability + 0.1)

        # Account for batch efficiency (larger batches are more efficient)
        batch_efficiency = min(1.0, batch_size / 32.0)

        return (batch_size * time_per_molecule) / batch_efficiency

    def prefetch_data(
        self, batches: List[DistributedBatch], target_devices: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Prefetch data to target devices.

        Args:
            batches: Batches to prefetch
            target_devices: Mapping of worker_id to device_ids

        Returns:
            Prefetched data handles
        """
        if self.config.data_transfer_mode == DataTransferMode.EAGER:
            return self._eager_prefetch(batches, target_devices)
        elif self.config.data_transfer_mode == DataTransferMode.LAZY:
            return self._lazy_prefetch(batches, target_devices)
        elif self.config.data_transfer_mode == DataTransferMode.STREAMING:
            return self._streaming_prefetch(batches, target_devices)
        else:
            return {}

    def _eager_prefetch(
        self, batches: List[DistributedBatch], target_devices: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Eagerly transfer all data to target devices."""
        prefetched_data = {}

        if not TORCH_AVAILABLE:
            return prefetched_data

        for batch in batches:
            batch_data = {}

            for worker_id, mol_indices in batch.worker_assignments.items():
                if worker_id in target_devices:
                    device_ids = target_devices[worker_id]
                    primary_device = device_ids[0] if device_ids else "cpu"

                    # Prepare tensors for this worker
                    geometries = [batch.geometries[i] for i in mol_indices]
                    tensor_data = self._geometries_to_tensors(
                        geometries, primary_device
                    )

                    batch_data[worker_id] = {
                        "tensors": tensor_data,
                        "device": primary_device,
                        "molecule_count": len(geometries),
                    }

            prefetched_data[batch.batch_id] = batch_data

        return prefetched_data

    def _lazy_prefetch(
        self, batches: List[DistributedBatch], target_devices: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Set up lazy data transfer (transfer when requested)."""
        return {
            batch.batch_id: {
                "batch": batch,
                "target_devices": target_devices,
                "transfer_fn": self._create_lazy_transfer_fn(batch, target_devices),
            }
            for batch in batches
        }

    def _streaming_prefetch(
        self, batches: List[DistributedBatch], target_devices: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Set up streaming data transfer."""
        # This would implement streaming transfer using queues/pipes
        return {
            batch.batch_id: {
                "streaming_queue": self._create_streaming_queue(batch, target_devices),
                "batch_info": batch,
            }
            for batch in batches
        }

    def _geometries_to_tensors(
        self, geometries: List[GeometryInput], device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """Convert geometries to optimized tensor format."""
        if not TORCH_AVAILABLE:
            raise QMLModelError("PyTorch not available for tensor conversion")

        max_atoms = max(len(geom.atomic_numbers) for geom in geometries)
        batch_size = len(geometries)

        # Preallocate tensors
        coordinates = torch.zeros(
            (batch_size, max_atoms, 3), dtype=torch.float32, device=device
        )
        atomic_numbers = torch.zeros(
            (batch_size, max_atoms), dtype=torch.long, device=device
        )
        atom_mask = torch.zeros(
            (batch_size, max_atoms), dtype=torch.bool, device=device
        )
        charges = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Fill tensors
        for i, geom in enumerate(geometries):
            n_atoms = len(geom.atomic_numbers)
            coordinates[i, :n_atoms] = torch.tensor(
                geom.coordinates, dtype=torch.float32
            )
            atomic_numbers[i, :n_atoms] = torch.tensor(
                geom.atomic_numbers, dtype=torch.long
            )
            atom_mask[i, :n_atoms] = True
            charges[i] = geom.charge

        return {
            "coordinates": coordinates,
            "atomic_numbers": atomic_numbers,
            "atom_mask": atom_mask,
            "charges": charges,
            "batch_size": batch_size,
            "max_atoms": max_atoms,
        }

    def _create_lazy_transfer_fn(
        self, batch: DistributedBatch, target_devices: Dict[str, List[str]]
    ):
        """Create lazy transfer function."""

        def transfer_fn(worker_id: str):
            if worker_id in batch.worker_assignments and worker_id in target_devices:
                mol_indices = batch.worker_assignments[worker_id]
                geometries = [batch.geometries[i] for i in mol_indices]
                device = (
                    target_devices[worker_id][0] if target_devices[worker_id] else "cpu"
                )
                return self._geometries_to_tensors(geometries, device)
            return None

        return transfer_fn

    def _create_streaming_queue(
        self, batch: DistributedBatch, target_devices: Dict[str, List[str]]
    ):
        """Create streaming queue for gradual data transfer."""
        # Simplified implementation - would use actual streaming queues
        return {"batch_id": batch.batch_id, "ready": True, "data": batch}


class DistributedQMLInferenceEngine:
    """
    Main engine for scalable distributed QML inference.

    Provides massively parallel QML inference across multiple GPUs or TPUs
    with optimized batching, data transfer, and fault tolerance.
    """

    def __init__(self, model: BaseQMLModel, config: DistributedConfig = None):
        """
        Initialize distributed QML inference engine.

        Args:
            model: Base QML model for inference
            config: Distributed configuration
        """
        self.base_model = model
        self.config = config or DistributedConfig()

        # Initialize components
        self.data_loader = OptimizedDataLoader(self.config)
        self.memory_processor = MemoryAwareProcessor(
            strategy=self.config.memory_strategy
        )
        self.performance_monitor = PerformanceMonitor()
        self.load_balancer = LoadBalancer()

        # Worker management
        self.workers: Dict[str, WorkerInfo] = {}
        self.distributed_models: Dict[str, Any] = {}
        self._cluster_client = None
        self._executor = None

        # State management
        self._initialized = False
        self._running = False

        # Metrics
        self._total_predictions = 0
        self._total_batches = 0
        self._execution_times = []

        logger = logging.getLogger(__name__)

    def initialize(self, cluster_address: Optional[str] = None) -> bool:
        """
        Initialize distributed infrastructure.

        Args:
            cluster_address: Address of existing cluster to connect to

        Returns:
            Success status
        """
        try:
            # Initialize backend-specific infrastructure
            success = self._initialize_backend(cluster_address)

            if success:
                # Discover and register workers
                self._discover_workers()

                # Initialize distributed models
                self._initialize_distributed_models()

                # Start monitoring
                self.performance_monitor.start_monitoring()

                self._initialized = True
                logging.info(
                    f"Distributed QML inference engine initialized with {len(self.workers)} workers"
                )

            return success

        except Exception as e:
            logging.error(f"Failed to initialize distributed engine: {e}")
            return False

    def _initialize_backend(self, cluster_address: Optional[str] = None) -> bool:
        """Initialize specific distributed backend."""
        if self.config.backend == DistributionBackend.DASK:
            return self._initialize_dask(cluster_address)
        elif self.config.backend == DistributionBackend.RAY:
            return self._initialize_ray(cluster_address)
        elif self.config.backend == DistributionBackend.TORCH_DDP:
            return self._initialize_torch_ddp()
        elif self.config.backend == DistributionBackend.JAX_PMAP:
            return self._initialize_jax_pmap()
        elif self.config.backend == DistributionBackend.MULTIPROCESSING:
            return self._initialize_multiprocessing()
        else:
            raise QMLModelError(f"Unsupported backend: {self.config.backend}")

    def _initialize_dask(self, cluster_address: Optional[str] = None) -> bool:
        """Initialize Dask distributed backend."""
        if not DASK_AVAILABLE:
            logging.error("Dask not available")
            return False

        try:
            from ..cloud.distributed_computing import DaskIntegration

            cluster_config = ClusterConfig(
                cluster_type="dask",
                scheduler_address=cluster_address,
                n_workers=self.config.num_workers,
                memory_per_worker=self.config.max_memory_per_worker,
            )

            self._dask_integration = DaskIntegration(cluster_config)

            if cluster_address:
                success = self._dask_integration.setup_cluster("existing")
            else:
                success = self._dask_integration.setup_cluster("local")

            if success:
                self._cluster_client = self._dask_integration.client
                logging.info("Dask cluster initialized successfully")

            return success

        except Exception as e:
            logging.error(f"Failed to initialize Dask: {e}")
            return False

    def _initialize_ray(self, cluster_address: Optional[str] = None) -> bool:
        """Initialize Ray distributed backend."""
        if not RAY_AVAILABLE:
            logging.error("Ray not available")
            return False

        try:
            from ..cloud.distributed_computing import RayIntegration

            cluster_config = ClusterConfig(
                cluster_type="ray",
                scheduler_address=cluster_address,
                n_workers=self.config.num_workers,
            )

            self._ray_integration = RayIntegration(cluster_config)

            if cluster_address:
                success = self._ray_integration.setup_cluster("existing")
            else:
                success = self._ray_integration.setup_cluster("local")

            return success

        except Exception as e:
            logging.error(f"Failed to initialize Ray: {e}")
            return False

    def _initialize_torch_ddp(self) -> bool:
        """Initialize PyTorch DistributedDataParallel."""
        if not TORCH_AVAILABLE:
            logging.error("PyTorch not available")
            return False

        try:
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.communication_backend,
                    init_method="env://",
                    world_size=self.config.num_workers,
                    rank=0,  # This would be set appropriately in multi-process setup
                )

            logging.info("PyTorch DDP initialized")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize PyTorch DDP: {e}")
            return False

    def _initialize_jax_pmap(self) -> bool:
        """Initialize JAX pmap for distributed computation."""
        if not JAX_AVAILABLE:
            logging.error("JAX not available")
            return False

        try:
            # Check available devices
            devices = jax.devices()
            if len(devices) < self.config.num_workers:
                logging.warning(
                    f"Requested {self.config.num_workers} workers but only {len(devices)} devices available"
                )

            self._jax_devices = devices
            logging.info(f"JAX pmap initialized with {len(devices)} devices")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize JAX pmap: {e}")
            return False

    def _initialize_multiprocessing(self) -> bool:
        """Initialize multiprocessing backend."""
        try:
            self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
            logging.info(
                f"Multiprocessing initialized with {self.config.num_workers} workers"
            )
            return True

        except Exception as e:
            logging.error(f"Failed to initialize multiprocessing: {e}")
            return False

    def _discover_workers(self):
        """Discover and register available workers."""
        if self.config.backend == DistributionBackend.DASK and self._cluster_client:
            self._discover_dask_workers()
        elif self.config.backend == DistributionBackend.RAY:
            self._discover_ray_workers()
        elif self.config.backend == DistributionBackend.TORCH_DDP:
            self._discover_torch_workers()
        elif self.config.backend == DistributionBackend.JAX_PMAP:
            self._discover_jax_workers()
        else:
            self._create_local_workers()

    def _discover_dask_workers(self):
        """Discover Dask workers."""
        if self._cluster_client:
            scheduler_info = self._cluster_client.scheduler_info()

            for worker_id, worker_info in scheduler_info["workers"].items():
                # Estimate GPU devices (simplified)
                gpu_count = self.config.devices_per_worker
                device_ids = [f"cuda:{i}" for i in range(gpu_count)]

                self.workers[worker_id] = WorkerInfo(
                    worker_id=worker_id,
                    device_ids=device_ids,
                    memory_available=worker_info.get("memory_limit", 0)
                    / 1e9,  # Convert to GB
                    compute_capability=1.0,  # Default capability
                    status="ready",
                )

    def _discover_ray_workers(self):
        """Discover Ray workers."""
        if RAY_AVAILABLE and ray.is_initialized():
            nodes = ray.nodes()

            for i, node in enumerate(nodes):
                if node["Alive"]:
                    worker_id = f"ray_worker_{i}"
                    gpu_count = node.get("Resources", {}).get("GPU", 0)
                    device_ids = [f"cuda:{j}" for j in range(int(gpu_count))]

                    self.workers[worker_id] = WorkerInfo(
                        worker_id=worker_id,
                        device_ids=device_ids,
                        memory_available=node.get("Resources", {}).get("memory", 0)
                        / 1e9,
                        compute_capability=1.0,
                        status="ready",
                    )

    def _discover_torch_workers(self):
        """Discover PyTorch DDP workers."""
        if TORCH_AVAILABLE and dist.is_initialized():
            world_size = dist.get_world_size()

            for rank in range(world_size):
                worker_id = f"torch_worker_{rank}"
                device_ids = [f"cuda:{rank}"] if torch.cuda.is_available() else ["cpu"]

                self.workers[worker_id] = WorkerInfo(
                    worker_id=worker_id,
                    device_ids=device_ids,
                    memory_available=8.0,  # Default 8GB
                    compute_capability=1.0,
                    status="ready",
                )

    def _discover_jax_workers(self):
        """Discover JAX devices as workers."""
        if JAX_AVAILABLE and hasattr(self, "_jax_devices"):
            for i, device in enumerate(self._jax_devices):
                worker_id = f"jax_device_{i}"

                self.workers[worker_id] = WorkerInfo(
                    worker_id=worker_id,
                    device_ids=[str(device)],
                    memory_available=8.0,  # Estimate
                    compute_capability=1.0,
                    status="ready",
                )

    def _create_local_workers(self):
        """Create local multiprocessing workers."""
        for i in range(self.config.num_workers):
            worker_id = f"local_worker_{i}"

            # Try to assign GPUs if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_id = i % torch.cuda.device_count()
                device_ids = [f"cuda:{gpu_id}"]
            else:
                device_ids = ["cpu"]

            self.workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                device_ids=device_ids,
                memory_available=4.0,  # Default 4GB per worker
                compute_capability=1.0,
                status="ready",
            )

    def _initialize_distributed_models(self):
        """Initialize distributed model instances."""
        for worker_id, worker_info in self.workers.items():
            try:
                # Create model instance for this worker
                if (
                    self.config.backend == DistributionBackend.TORCH_DDP
                    and TORCH_AVAILABLE
                ):
                    # Wrap model with DDP
                    device = (
                        worker_info.device_ids[0] if worker_info.device_ids else "cpu"
                    )
                    model_copy = self._create_model_copy(device)
                    if hasattr(model_copy, "_model") and isinstance(
                        model_copy._model, torch.nn.Module
                    ):
                        wrapped_model = DDP(model_copy._model)
                        model_copy._model = wrapped_model
                    self.distributed_models[worker_id] = model_copy
                else:
                    # Create regular model copy
                    self.distributed_models[worker_id] = self._create_model_copy()

                logging.debug(f"Initialized model for worker {worker_id}")

            except Exception as e:
                logging.error(f"Failed to initialize model for worker {worker_id}: {e}")

    def _create_model_copy(self, device: str = "cpu") -> BaseQMLModel:
        """Create a copy of the base model for distributed execution."""
        # This is a simplified implementation
        # In practice, you'd need to properly serialize/deserialize the model
        model_copy = self.base_model.__class__(
            model_path=self.base_model.model_path,
            device=device,
            precision=self.base_model.precision,
            cache_predictions=False,  # Disable caching for distributed models
        )

        if hasattr(self.base_model, "_model") and self.base_model._model is not None:
            model_copy._model = self.base_model._model
            model_copy._is_loaded = True

        return model_copy

    def predict_distributed(
        self,
        geometries: List[GeometryInput],
        properties: List[QuantumProperty],
        timeout: Optional[float] = None,
    ) -> InferenceResult:
        """
        Perform distributed QML inference.

        Args:
            geometries: Input molecular geometries
            properties: Properties to predict
            timeout: Maximum execution time

        Returns:
            Distributed inference result
        """
        if not self._initialized:
            raise QMLModelError("Distributed engine not initialized")

        start_time = time.time()
        batch_id = f"distributed_batch_{uuid.uuid4().hex[:8]}"

        try:
            # Create optimized batches
            with self.performance_monitor.memory_managed_processing():
                batches = self.data_loader.create_optimized_batches(
                    geometries, properties, self.workers
                )

            if not batches:
                raise QMLModelError("No batches created for processing")

            # Execute distributed inference
            if self.config.backend == DistributionBackend.DASK:
                result = self._execute_dask_inference(batches, properties, timeout)
            elif self.config.backend == DistributionBackend.RAY:
                result = self._execute_ray_inference(batches, properties, timeout)
            elif self.config.backend == DistributionBackend.TORCH_DDP:
                result = self._execute_torch_ddp_inference(batches, properties, timeout)
            elif self.config.backend == DistributionBackend.JAX_PMAP:
                result = self._execute_jax_pmap_inference(batches, properties, timeout)
            else:
                result = self._execute_multiprocessing_inference(
                    batches, properties, timeout
                )

            execution_time = time.time() - start_time

            # Create comprehensive result
            execution_metrics = PerformanceMetrics(
                total_time=execution_time,
                compute_time=execution_time,
                communication_time=0.0,  # Would be measured in practice
                io_time=0.0,
                memory_usage_mb=0.0,  # Would be measured
                cpu_utilization=0.0,
                throughput_molecules_per_sec=len(geometries) / execution_time
                if execution_time > 0
                else 0,
            )

            # Update internal metrics
            self._total_predictions += len(geometries)
            self._total_batches += len(batches)
            self._execution_times.append(execution_time)

            return InferenceResult(
                batch_id=batch_id,
                predictions=result.get("predictions", []),
                worker_results=result.get("worker_results", {}),
                execution_metrics=execution_metrics,
                transfer_metrics=result.get("transfer_metrics", {}),
                errors=result.get("errors", []),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"Distributed inference failed: {e}")

            return InferenceResult(
                batch_id=batch_id,
                predictions=[],
                worker_results={},
                execution_metrics=PerformanceMetrics(
                    total_time=execution_time,
                    compute_time=0.0,
                    communication_time=0.0,
                    io_time=0.0,
                    memory_usage_mb=0.0,
                    cpu_utilization=0.0,
                ),
                transfer_metrics={},
                errors=[{"error": str(e), "timestamp": time.time()}],
            )

    def _execute_dask_inference(
        self,
        batches: List[DistributedBatch],
        properties: List[QuantumProperty],
        timeout: Optional[float],
    ) -> Dict[str, Any]:
        """Execute inference using Dask distributed."""
        if not self._cluster_client:
            raise QMLModelError("Dask client not initialized")

        try:
            # Create delayed tasks
            futures = []

            for batch in batches:
                # Create delayed function for batch processing
                @delayed
                def process_batch_delayed(batch_data, props, model_info):
                    # This would run on remote workers
                    from chemi_accel.qml.base import BaseQMLModel

                    # Reconstruct model (simplified)
                    model = BaseQMLModel(
                        model_path=model_info.get("model_path"),
                        device=model_info.get("device", "cpu"),
                    )

                    # Process batch
                    predictions = model.batch_predict(
                        batch_data["geometries"],
                        props,
                        batch_size=len(batch_data["geometries"]),
                    )

                    return {
                        "batch_id": batch_data["batch_id"],
                        "predictions": predictions,
                        "worker_id": batch_data.get("worker_id"),
                        "processing_time": 0.0,  # Would measure actual time
                    }

                # Prepare batch data for serialization
                batch_data = {
                    "batch_id": batch.batch_id,
                    "geometries": batch.geometries,
                    "worker_id": list(batch.worker_assignments.keys())[0]
                    if batch.worker_assignments
                    else None,
                }

                model_info = {
                    "model_path": str(self.base_model.model_path)
                    if self.base_model.model_path
                    else None,
                    "device": "cuda"
                    if "cuda" in str(batch.worker_assignments)
                    else "cpu",
                }

                future = process_batch_delayed(batch_data, properties, model_info)
                futures.append(future)

            # Execute and gather results
            results = dd.compute(*futures, scheduler=self._cluster_client)

            # Combine results
            all_predictions = []
            worker_results = {}

            for result in results:
                if "predictions" in result:
                    all_predictions.extend(result["predictions"])
                if "worker_id" in result:
                    worker_results[result["worker_id"]] = result

            return {
                "predictions": all_predictions,
                "worker_results": worker_results,
                "transfer_metrics": {},
            }

        except Exception as e:
            logging.error(f"Dask execution failed: {e}")
            return {
                "predictions": [],
                "worker_results": {},
                "errors": [{"error": str(e)}],
            }

    def _execute_ray_inference(
        self,
        batches: List[DistributedBatch],
        properties: List[QuantumProperty],
        timeout: Optional[float],
    ) -> Dict[str, Any]:
        """Execute inference using Ray distributed."""
        if not RAY_AVAILABLE or not ray.is_initialized():
            raise QMLModelError("Ray not initialized")

        try:
            # Define remote worker class
            @ray.remote(num_gpus=1 if self.config.devices_per_worker > 0 else 0)
            class QMLWorker:
                def __init__(self, model_info):
                    from chemi_accel.qml.base import BaseQMLModel

                    self.model = BaseQMLModel(
                        model_path=model_info.get("model_path"),
                        device="cuda" if model_info.get("use_gpu") else "cpu",
                    )

                def process_batch(self, geometries, properties):
                    predictions = self.model.batch_predict(
                        geometries, properties, batch_size=len(geometries)
                    )
                    return predictions

            # Create workers
            model_info = {
                "model_path": str(self.base_model.model_path)
                if self.base_model.model_path
                else None,
                "use_gpu": self.config.devices_per_worker > 0,
            }

            workers = [
                QMLWorker.remote(model_info)
                for _ in range(min(len(batches), self.config.num_workers))
            ]

            # Submit tasks
            futures = []
            for i, batch in enumerate(batches):
                worker = workers[i % len(workers)]
                future = worker.process_batch.remote(batch.geometries, properties)
                futures.append(future)

            # Get results
            results = ray.get(futures)

            # Combine predictions
            all_predictions = []
            for result in results:
                if isinstance(result, list):
                    all_predictions.extend(result)
                else:
                    all_predictions.append(result)

            return {
                "predictions": all_predictions,
                "worker_results": {
                    f"ray_worker_{i}": result for i, result in enumerate(results)
                },
                "transfer_metrics": {},
            }

        except Exception as e:
            logging.error(f"Ray execution failed: {e}")
            return {
                "predictions": [],
                "worker_results": {},
                "errors": [{"error": str(e)}],
            }

    def _execute_torch_ddp_inference(
        self,
        batches: List[DistributedBatch],
        properties: List[QuantumProperty],
        timeout: Optional[float],
    ) -> Dict[str, Any]:
        """Execute inference using PyTorch DistributedDataParallel."""
        # Simplified implementation for single-process demonstration
        # In practice, this would coordinate across multiple processes

        try:
            all_predictions = []
            worker_results = {}

            for i, batch in enumerate(batches):
                worker_id = f"torch_worker_{i % len(self.workers)}"

                if worker_id in self.distributed_models:
                    model = self.distributed_models[worker_id]
                    predictions = model.batch_predict(
                        batch.geometries, properties, batch_size=len(batch.geometries)
                    )

                    all_predictions.extend(predictions)
                    worker_results[worker_id] = {
                        "batch_id": batch.batch_id,
                        "predictions": predictions,
                        "processing_time": 0.0,
                    }

            return {
                "predictions": all_predictions,
                "worker_results": worker_results,
                "transfer_metrics": {},
            }

        except Exception as e:
            logging.error(f"PyTorch DDP execution failed: {e}")
            return {
                "predictions": [],
                "worker_results": {},
                "errors": [{"error": str(e)}],
            }

    def _execute_jax_pmap_inference(
        self,
        batches: List[DistributedBatch],
        properties: List[QuantumProperty],
        timeout: Optional[float],
    ) -> Dict[str, Any]:
        """Execute inference using JAX pmap."""
        # This would implement JAX pmap-based distributed processing
        # Simplified implementation

        try:
            all_predictions = []
            worker_results = {}

            for i, batch in enumerate(batches):
                worker_id = f"jax_device_{i % len(self._jax_devices)}"

                if worker_id in self.distributed_models:
                    model = self.distributed_models[worker_id]
                    predictions = model.batch_predict(
                        batch.geometries, properties, batch_size=len(batch.geometries)
                    )

                    all_predictions.extend(predictions)
                    worker_results[worker_id] = {
                        "batch_id": batch.batch_id,
                        "predictions": predictions,
                    }

            return {
                "predictions": all_predictions,
                "worker_results": worker_results,
                "transfer_metrics": {},
            }

        except Exception as e:
            logging.error(f"JAX pmap execution failed: {e}")
            return {
                "predictions": [],
                "worker_results": {},
                "errors": [{"error": str(e)}],
            }

    def _execute_multiprocessing_inference(
        self,
        batches: List[DistributedBatch],
        properties: List[QuantumProperty],
        timeout: Optional[float],
    ) -> Dict[str, Any]:
        """Execute inference using multiprocessing."""
        if not self._executor:
            raise QMLModelError("Multiprocessing executor not initialized")

        try:
            # Submit tasks to thread pool
            futures = []

            for batch in batches:
                future = self._executor.submit(
                    self._process_batch_multiprocessing, batch, properties
                )
                futures.append((batch.batch_id, future))

            # Collect results
            all_predictions = []
            worker_results = {}
            errors = []

            for batch_id, future in futures:
                try:
                    result = future.result(timeout=timeout)
                    all_predictions.extend(result["predictions"])
                    worker_results[batch_id] = result

                except Exception as e:
                    errors.append(
                        {
                            "batch_id": batch_id,
                            "error": str(e),
                            "timestamp": time.time(),
                        }
                    )

            return {
                "predictions": all_predictions,
                "worker_results": worker_results,
                "transfer_metrics": {},
                "errors": errors,
            }

        except Exception as e:
            logging.error(f"Multiprocessing execution failed: {e}")
            return {
                "predictions": [],
                "worker_results": {},
                "errors": [{"error": str(e)}],
            }

    def _process_batch_multiprocessing(
        self, batch: DistributedBatch, properties: List[QuantumProperty]
    ) -> Dict[str, Any]:
        """Process a single batch in multiprocessing mode."""
        start_time = time.time()

        try:
            # Use base model for prediction
            predictions = self.base_model.batch_predict(
                batch.geometries, properties, batch_size=len(batch.geometries)
            )

            processing_time = time.time() - start_time

            return {
                "batch_id": batch.batch_id,
                "predictions": predictions,
                "processing_time": processing_time,
                "molecule_count": len(batch.geometries),
            }

        except Exception as e:
            return {
                "batch_id": batch.batch_id,
                "predictions": [],
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self._execution_times:
            return {}

        avg_execution_time = np.mean(self._execution_times)
        total_execution_time = sum(self._execution_times)

        return {
            "total_predictions": self._total_predictions,
            "total_batches": self._total_batches,
            "average_execution_time": avg_execution_time,
            "total_execution_time": total_execution_time,
            "throughput": self._total_predictions / total_execution_time
            if total_execution_time > 0
            else 0,
            "average_batch_size": self._total_predictions / self._total_batches
            if self._total_batches > 0
            else 0,
            "worker_count": len(self.workers),
            "backend": self.config.backend.value,
            "scaling_strategy": self.config.scaling_strategy.value,
        }

    def get_worker_status(self) -> Dict[str, WorkerInfo]:
        """Get current status of all workers."""
        # Update worker heartbeats and status
        current_time = time.time()

        for worker_id, worker_info in self.workers.items():
            # Simple health check - in practice would ping workers
            if current_time - worker_info.last_heartbeat > 30.0:  # 30 second timeout
                worker_info.status = "disconnected"
            elif worker_info.error_count > 5:
                worker_info.status = "error"
            elif worker_info.current_batch_size > 0:
                worker_info.status = "processing"
            else:
                worker_info.status = "idle"

        return self.workers.copy()

    def shutdown(self):
        """Shutdown distributed infrastructure."""
        if not self._initialized:
            return

        try:
            self._running = False

            # Shutdown performance monitoring
            self.performance_monitor.stop_monitoring()

            # Shutdown backend-specific resources
            if self.config.backend == DistributionBackend.DASK and hasattr(
                self, "_dask_integration"
            ):
                self._dask_integration.shutdown_cluster()

            elif self.config.backend == DistributionBackend.RAY and hasattr(
                self, "_ray_integration"
            ):
                self._ray_integration.shutdown_cluster()

            elif (
                self.config.backend == DistributionBackend.TORCH_DDP and TORCH_AVAILABLE
            ):
                if dist.is_initialized():
                    dist.destroy_process_group()

            elif (
                self.config.backend == DistributionBackend.MULTIPROCESSING
                and self._executor
            ):
                self._executor.shutdown(wait=True)

            # Clear resources
            self.workers.clear()
            self.distributed_models.clear()

            self._initialized = False
            logging.info("Distributed QML inference engine shutdown complete")

        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Convenience functions
def create_distributed_engine(
    model: BaseQMLModel, backend: str = "dask", num_workers: int = 4, **kwargs
) -> DistributedQMLInferenceEngine:
    """
    Create a distributed QML inference engine with sensible defaults.

    Args:
        model: Base QML model
        backend: Distribution backend ('dask', 'ray', 'torch_ddp', 'jax_pmap', 'multiprocessing')
        num_workers: Number of workers
        **kwargs: Additional configuration options

    Returns:
        Configured distributed inference engine
    """
    config = DistributedConfig(
        backend=DistributionBackend(backend), num_workers=num_workers, **kwargs
    )

    return DistributedQMLInferenceEngine(model, config)


def distributed_predict(
    model: BaseQMLModel,
    geometries: List[GeometryInput],
    properties: List[QuantumProperty],
    backend: str = "dask",
    num_workers: int = 4,
    **kwargs,
) -> InferenceResult:
    """
    Convenience function for one-off distributed prediction.

    Args:
        model: QML model
        geometries: Input geometries
        properties: Properties to predict
        backend: Distribution backend
        num_workers: Number of workers
        **kwargs: Additional configuration

    Returns:
        Inference result
    """
    with create_distributed_engine(model, backend, num_workers, **kwargs) as engine:
        return engine.predict_distributed(geometries, properties)

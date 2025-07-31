"""
Optimized batching and data transfer mechanisms for scalable QML inference.

This module provides advanced batching strategies and high-performance data transfer
optimizations to maximize throughput for large-scale QML calculations.
"""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None
    DataLoader = None
    Dataset = None

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from .base import QuantumProperty
from .on_the_fly import GeometryInput


class BatchingStrategy(Enum):
    """Advanced batching strategies for optimal throughput."""

    FIXED_SIZE = "fixed_size"  # Fixed batch sizes
    ADAPTIVE_SIZE = "adaptive_size"  # Adaptive based on complexity
    MEMORY_AWARE = "memory_aware"  # Memory-constrained batching
    LATENCY_OPTIMIZED = "latency_optimized"  # Minimize latency
    THROUGHPUT_OPTIMIZED = "throughput_optimized"  # Maximize throughput
    LOAD_BALANCED = "load_balanced"  # Balance across workers
    HIERARCHICAL = "hierarchical"  # Multi-level batching


class DataLayout(Enum):
    """Memory layout strategies for optimal access patterns."""

    ARRAY_OF_STRUCTURES = "aos"  # Array of Structures (default)
    STRUCTURE_OF_ARRAYS = "soa"  # Structure of Arrays (vectorized)
    BLOCKED = "blocked"  # Blocked layout for cache efficiency
    COMPRESSED = "compressed"  # Compressed representation
    SPARSE = "sparse"  # Sparse representation for large molecules


class TransferOptimization(Enum):
    """Data transfer optimization techniques."""

    NONE = "none"  # No optimization
    PINNED_MEMORY = "pinned_memory"  # Use pinned memory for faster transfers
    ZERO_COPY = "zero_copy"  # Zero-copy transfers when possible
    COMPRESSION = "compression"  # Compress data during transfer
    PREFETCH = "prefetch"  # Prefetch next batch
    PIPELINE = "pipeline"  # Pipeline transfers with computation
    ASYNC_TRANSFER = "async_transfer"  # Asynchronous transfers


@dataclass
class BatchingConfig:
    """Configuration for optimized batching."""

    # Batching strategy
    strategy: BatchingStrategy = BatchingStrategy.ADAPTIVE_SIZE
    base_batch_size: int = 64
    min_batch_size: int = 8
    max_batch_size: int = 512

    # Memory constraints
    max_memory_per_batch: str = "2GB"
    memory_safety_factor: float = 0.8
    enable_memory_profiling: bool = True

    # Performance tuning
    target_utilization: float = 0.85
    latency_threshold: float = 1.0  # seconds
    throughput_threshold: float = 100.0  # molecules/second

    # Data layout and transfer
    data_layout: DataLayout = DataLayout.STRUCTURE_OF_ARRAYS
    transfer_optimization: TransferOptimization = TransferOptimization.PIPELINE
    enable_compression: bool = True
    compression_level: int = 3

    # Advanced features
    enable_prefetching: bool = True
    prefetch_factor: int = 2
    enable_caching: bool = True
    cache_size: int = 1000

    # Load balancing
    enable_dynamic_sizing: bool = True
    size_adjustment_factor: float = 0.1
    load_balance_window: int = 10


@dataclass
class BatchMetadata:
    """Metadata for batch processing optimization."""

    batch_id: str
    molecule_complexities: List[float]
    estimated_memory: float
    estimated_compute_time: float
    priority: int = 1
    creation_time: float = field(default_factory=time.time)
    target_devices: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TransferMetrics:
    """Metrics for data transfer performance."""

    transfer_time: float = 0.0
    transfer_size_mb: float = 0.0
    compression_ratio: float = 1.0
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class ComplexityEstimator:
    """Estimate computational complexity of molecular batches."""

    def __init__(self):
        """Initialize complexity estimator."""
        self._atom_weights = {
            1: 1.0,  # H
            6: 2.0,  # C
            7: 2.5,  # N
            8: 2.5,  # O
            16: 3.0,  # S
            15: 3.5,  # P
            # Add more elements as needed
        }
        self._complexity_cache = {}
        self._cache_lock = threading.RLock()

    def estimate_molecule_complexity(self, geometry: GeometryInput) -> float:
        """
        Estimate computational complexity for a single molecule.

        Args:
            geometry: Molecular geometry

        Returns:
            Complexity score (higher = more expensive)
        """
        # Create cache key
        cache_key = self._create_cache_key(geometry)

        with self._cache_lock:
            if cache_key in self._complexity_cache:
                return self._complexity_cache[cache_key]

        # Calculate complexity
        complexity = self._calculate_complexity(geometry)

        with self._cache_lock:
            if len(self._complexity_cache) < 10000:  # Limit cache size
                self._complexity_cache[cache_key] = complexity

        return complexity

    def _create_cache_key(self, geometry: GeometryInput) -> str:
        """Create cache key for geometry."""
        atomic_signature = tuple(sorted(geometry.atomic_numbers))
        atom_count = len(geometry.atomic_numbers)
        return f"{atom_count}_{hash(atomic_signature)}"

    def _calculate_complexity(self, geometry: GeometryInput) -> float:
        """Calculate complexity score."""
        atom_count = len(geometry.atomic_numbers)

        # Base complexity scales with atom count
        base_complexity = atom_count**2.2  # Slightly better than N^2.5

        # Weight by atom types
        type_complexity = sum(
            self._atom_weights.get(atomic_num, 1.0)
            for atomic_num in geometry.atomic_numbers
        )

        # Consider molecular structure
        structure_complexity = self._estimate_structure_complexity(geometry)

        # Consider charge and multiplicity
        charge_factor = 1.0 + abs(geometry.charge) * 0.1
        mult_factor = 1.0 + (geometry.multiplicity - 1) * 0.2

        return (
            base_complexity
            * type_complexity
            * structure_complexity
            * charge_factor
            * mult_factor
        )

    def _estimate_structure_complexity(self, geometry: GeometryInput) -> float:
        """Estimate structural complexity from coordinates."""
        if len(geometry.coordinates) < 2:
            return 1.0

        # Calculate average interatomic distances
        coords = geometry.coordinates
        distances = []

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)

        if not distances:
            return 1.0

        # Use distance distribution as complexity measure
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)

        # Higher std suggests more complex structure
        structure_factor = 1.0 + std_dist / (avg_dist + 1e-6)

        return min(structure_factor, 3.0)  # Cap at 3x

    def estimate_batch_complexity(self, geometries: List[GeometryInput]) -> float:
        """Estimate total batch complexity."""
        if not geometries:
            return 0.0

        individual_complexities = [
            self.estimate_molecule_complexity(geom) for geom in geometries
        ]

        # Account for batch efficiency (larger batches are more efficient)
        batch_size = len(geometries)
        efficiency_factor = min(1.0, batch_size / 32.0) * 0.9 + 0.1

        total_complexity = sum(individual_complexities)

        return total_complexity / efficiency_factor


class MemoryEstimator:
    """Estimate memory requirements for molecular batches."""

    def __init__(self, safety_factor: float = 0.8):
        """
        Initialize memory estimator.

        Args:
            safety_factor: Safety factor for memory estimates
        """
        self.safety_factor = safety_factor
        self._base_memory_per_atom = 64  # bytes per atom (rough estimate)

    def estimate_geometry_memory(
        self, geometry: GeometryInput, precision: str = "float32"
    ) -> float:
        """
        Estimate memory usage for a single geometry.

        Args:
            geometry: Molecular geometry
            precision: Numerical precision

        Returns:
            Memory usage in MB
        """
        atom_count = len(geometry.atomic_numbers)

        # Base memory for coordinates, atomic numbers, etc.
        if precision == "float64":
            bytes_per_float = 8
        else:
            bytes_per_float = 4

        # Coordinate memory: N_atoms * 3 * bytes_per_float
        coord_memory = atom_count * 3 * bytes_per_float

        # Atomic numbers: N_atoms * 4 bytes (int32)
        atomic_num_memory = atom_count * 4

        # Additional metadata and temporary arrays
        overhead_memory = atom_count * self._base_memory_per_atom

        total_bytes = coord_memory + atomic_num_memory + overhead_memory
        total_mb = total_bytes / (1024 * 1024)

        return total_mb / self.safety_factor

    def estimate_batch_memory(
        self,
        geometries: List[GeometryInput],
        precision: str = "float32",
        data_layout: DataLayout = DataLayout.STRUCTURE_OF_ARRAYS,
    ) -> float:
        """
        Estimate memory usage for a batch.

        Args:
            geometries: List of geometries
            precision: Numerical precision
            data_layout: Memory layout strategy

        Returns:
            Memory usage in MB
        """
        if not geometries:
            return 0.0

        # Individual memory estimates
        individual_memory = sum(
            self.estimate_geometry_memory(geom, precision) for geom in geometries
        )

        # Layout-specific overhead
        layout_factor = self._get_layout_factor(data_layout, len(geometries))

        # Padding and alignment overhead
        max_atoms = max(len(geom.atomic_numbers) for geom in geometries)
        avg_atoms = np.mean([len(geom.atomic_numbers) for geom in geometries])
        padding_factor = max_atoms / avg_atoms if avg_atoms > 0 else 1.0

        total_memory = individual_memory * layout_factor * padding_factor

        return total_memory

    def _get_layout_factor(self, layout: DataLayout, batch_size: int) -> float:
        """Get memory overhead factor for different layouts."""
        if layout == DataLayout.ARRAY_OF_STRUCTURES:
            return 1.1  # Minimal overhead
        elif layout == DataLayout.STRUCTURE_OF_ARRAYS:
            return 1.05  # Better cache efficiency, slight overhead
        elif layout == DataLayout.BLOCKED:
            return 1.2  # Block overhead
        elif layout == DataLayout.COMPRESSED:
            return 0.6  # Compression saves memory
        elif layout == DataLayout.SPARSE:
            sparsity = min(0.5, 32.0 / batch_size)  # Estimate sparsity
            return sparsity
        else:
            return 1.1


class OptimizedBatchCreator:
    """Create optimized batches for distributed QML inference."""

    def __init__(self, config: BatchingConfig = None):
        """
        Initialize batch creator.

        Args:
            config: Batching configuration
        """
        self.config = config or BatchingConfig()
        self.complexity_estimator = ComplexityEstimator()
        self.memory_estimator = MemoryEstimator(
            safety_factor=self.config.memory_safety_factor
        )

        # Performance tracking
        self._batch_history = []
        self._performance_metrics = {}
        self._adaptive_params = {
            "current_batch_size": self.config.base_batch_size,
            "success_rate": 1.0,
            "avg_utilization": 0.8,
        }

        # Threading for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)

    def create_optimized_batches(
        self,
        geometries: List[GeometryInput],
        properties: List[QuantumProperty],
        device_constraints: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[List[GeometryInput], BatchMetadata]]:
        """
        Create optimized batches from input geometries.

        Args:
            geometries: Input molecular geometries
            properties: Properties to predict
            device_constraints: Device-specific constraints

        Returns:
            List of (batch_geometries, metadata) tuples
        """
        if not geometries:
            return []

        # Estimate complexity for all molecules
        complexities = [
            self.complexity_estimator.estimate_molecule_complexity(geom)
            for geom in geometries
        ]

        # Apply batching strategy
        if self.config.strategy == BatchingStrategy.FIXED_SIZE:
            batches = self._create_fixed_size_batches(geometries, complexities)
        elif self.config.strategy == BatchingStrategy.ADAPTIVE_SIZE:
            batches = self._create_adaptive_batches(
                geometries, complexities, device_constraints
            )
        elif self.config.strategy == BatchingStrategy.MEMORY_AWARE:
            batches = self._create_memory_aware_batches(geometries, complexities)
        elif self.config.strategy == BatchingStrategy.LATENCY_OPTIMIZED:
            batches = self._create_latency_optimized_batches(geometries, complexities)
        elif self.config.strategy == BatchingStrategy.THROUGHPUT_OPTIMIZED:
            batches = self._create_throughput_optimized_batches(
                geometries, complexities
            )
        elif self.config.strategy == BatchingStrategy.LOAD_BALANCED:
            batches = self._create_load_balanced_batches(
                geometries, complexities, device_constraints
            )
        elif self.config.strategy == BatchingStrategy.HIERARCHICAL:
            batches = self._create_hierarchical_batches(geometries, complexities)
        else:
            batches = self._create_adaptive_batches(
                geometries, complexities, device_constraints
            )

        return batches

    def _create_fixed_size_batches(
        self, geometries: List[GeometryInput], complexities: List[float]
    ) -> List[Tuple[List[GeometryInput], BatchMetadata]]:
        """Create batches with fixed size."""
        batch_size = self.config.base_batch_size
        batches = []

        for i in range(0, len(geometries), batch_size):
            batch_geoms = geometries[i : i + batch_size]
            batch_complexities = complexities[i : i + batch_size]

            metadata = self._create_batch_metadata(batch_geoms, batch_complexities)
            batches.append((batch_geoms, metadata))

        return batches

    def _create_adaptive_batches(
        self,
        geometries: List[GeometryInput],
        complexities: List[float],
        device_constraints: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[List[GeometryInput], BatchMetadata]]:
        """Create batches with adaptive sizing based on complexity."""
        batches = []
        current_batch = []
        current_complexities = []
        current_complexity = 0.0

        # Sort by complexity for better batching
        sorted_items = sorted(zip(geometries, complexities), key=lambda x: x[1])

        # Target complexity per batch
        avg_complexity = np.mean(complexities)
        target_complexity = avg_complexity * self._adaptive_params["current_batch_size"]

        for geom, complexity in sorted_items:
            # Check if adding this molecule exceeds limits
            new_complexity = current_complexity + complexity
            new_batch_size = len(current_batch) + 1

            # Memory check
            test_batch = current_batch + [geom]
            estimated_memory = self.memory_estimator.estimate_batch_memory(
                test_batch, "float32", self.config.data_layout
            )

            memory_limit_mb = self._parse_memory_limit(self.config.max_memory_per_batch)

            should_split = (
                new_batch_size >= self.config.max_batch_size
                or new_complexity > target_complexity * 1.5
                or estimated_memory > memory_limit_mb
            )

            if should_split and current_batch:
                # Finalize current batch
                metadata = self._create_batch_metadata(
                    current_batch, current_complexities
                )
                batches.append((current_batch, metadata))

                # Start new batch
                current_batch = [geom]
                current_complexities = [complexity]
                current_complexity = complexity
            else:
                # Add to current batch
                current_batch.append(geom)
                current_complexities.append(complexity)
                current_complexity = new_complexity

        # Add final batch
        if current_batch:
            metadata = self._create_batch_metadata(current_batch, current_complexities)
            batches.append((current_batch, metadata))

        return batches

    def _create_memory_aware_batches(
        self, geometries: List[GeometryInput], complexities: List[float]
    ) -> List[Tuple[List[GeometryInput], BatchMetadata]]:
        """Create batches optimized for memory constraints."""
        batches = []
        current_batch = []
        current_complexities = []

        memory_limit_mb = self._parse_memory_limit(self.config.max_memory_per_batch)

        for geom, complexity in zip(geometries, complexities):
            test_batch = current_batch + [geom]
            estimated_memory = self.memory_estimator.estimate_batch_memory(
                test_batch, "float32", self.config.data_layout
            )

            if estimated_memory > memory_limit_mb and current_batch:
                # Finalize current batch
                metadata = self._create_batch_metadata(
                    current_batch, current_complexities
                )
                batches.append((current_batch, metadata))

                # Start new batch
                current_batch = [geom]
                current_complexities = [complexity]
            else:
                current_batch.append(geom)
                current_complexities.append(complexity)

        # Add final batch
        if current_batch:
            metadata = self._create_batch_metadata(current_batch, current_complexities)
            batches.append((current_batch, metadata))

        return batches

    def _create_latency_optimized_batches(
        self, geometries: List[GeometryInput], complexities: List[float]
    ) -> List[Tuple[List[GeometryInput], BatchMetadata]]:
        """Create batches optimized for low latency."""
        # Use smaller batches for lower latency
        target_batch_size = min(self.config.base_batch_size // 2, 32)

        batches = []
        for i in range(0, len(geometries), target_batch_size):
            batch_geoms = geometries[i : i + target_batch_size]
            batch_complexities = complexities[i : i + target_batch_size]

            metadata = self._create_batch_metadata(batch_geoms, batch_complexities)
            metadata.priority = 10  # High priority for low latency

            batches.append((batch_geoms, metadata))

        return batches

    def _create_throughput_optimized_batches(
        self, geometries: List[GeometryInput], complexities: List[float]
    ) -> List[Tuple[List[GeometryInput], BatchMetadata]]:
        """Create batches optimized for maximum throughput."""
        # Use larger batches for better throughput
        target_batch_size = min(self.config.max_batch_size, len(geometries))

        batches = []
        for i in range(0, len(geometries), target_batch_size):
            batch_geoms = geometries[i : i + target_batch_size]
            batch_complexities = complexities[i : i + target_batch_size]

            metadata = self._create_batch_metadata(batch_geoms, batch_complexities)
            batches.append((batch_geoms, metadata))

        return batches

    def _create_load_balanced_batches(
        self,
        geometries: List[GeometryInput],
        complexities: List[float],
        device_constraints: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[List[GeometryInput], BatchMetadata]]:
        """Create load-balanced batches across devices."""
        if not device_constraints:
            return self._create_adaptive_batches(geometries, complexities, None)

        num_devices = device_constraints.get("num_devices", 1)
        device_capabilities = device_constraints.get(
            "capabilities", [1.0] * num_devices
        )

        # Sort by complexity
        sorted_items = sorted(
            zip(geometries, complexities),
            key=lambda x: x[1],
            reverse=True,  # Start with most complex
        )

        # Initialize device loads
        device_loads = [0.0] * num_devices
        device_batches = [[] for _ in range(num_devices)]
        device_batch_complexities = [[] for _ in range(num_devices)]

        # Assign molecules to devices using greedy load balancing
        for geom, complexity in sorted_items:
            # Find device with lowest normalized load
            normalized_loads = [
                load / capability
                for load, capability in zip(device_loads, device_capabilities)
            ]

            best_device = np.argmin(normalized_loads)

            device_batches[best_device].append(geom)
            device_batch_complexities[best_device].append(complexity)
            device_loads[best_device] += complexity

        # Create final batches
        batches = []
        for device_id, (batch_geoms, batch_complexities) in enumerate(
            zip(device_batches, device_batch_complexities)
        ):
            if batch_geoms:
                metadata = self._create_batch_metadata(batch_geoms, batch_complexities)
                metadata.target_devices = [f"device_{device_id}"]
                batches.append((batch_geoms, metadata))

        return batches

    def _create_hierarchical_batches(
        self, geometries: List[GeometryInput], complexities: List[float]
    ) -> List[Tuple[List[GeometryInput], BatchMetadata]]:
        """Create hierarchical batches for multi-level processing."""
        # Group by complexity ranges
        complexity_ranges = [
            (0.0, np.percentile(complexities, 33)),
            (np.percentile(complexities, 33), np.percentile(complexities, 67)),
            (np.percentile(complexities, 67), max(complexities)),
        ]

        batches = []

        for i, (min_complexity, max_complexity) in enumerate(complexity_ranges):
            # Filter geometries in this complexity range
            range_items = [
                (geom, comp)
                for geom, comp in zip(geometries, complexities)
                if min_complexity <= comp <= max_complexity
            ]

            if not range_items:
                continue

            range_geoms = [item[0] for item in range_items]
            range_complexities = [item[1] for item in range_items]

            # Create sub-batches within this range
            range_batch_size = self.config.base_batch_size // (
                i + 1
            )  # Smaller batches for higher complexity

            for j in range(0, len(range_geoms), range_batch_size):
                batch_geoms = range_geoms[j : j + range_batch_size]
                batch_complexities = range_complexities[j : j + range_batch_size]

                metadata = self._create_batch_metadata(batch_geoms, batch_complexities)
                metadata.priority = 3 - i  # Higher priority for lower complexity

                batches.append((batch_geoms, metadata))

        return batches

    def _create_batch_metadata(
        self, geometries: List[GeometryInput], complexities: List[float]
    ) -> BatchMetadata:
        """Create metadata for a batch."""
        estimated_memory = self.memory_estimator.estimate_batch_memory(
            geometries, "float32", self.config.data_layout
        )

        # Estimate compute time based on complexity
        total_complexity = sum(complexities)
        base_time_per_complexity = 0.001  # seconds per complexity unit
        estimated_compute_time = total_complexity * base_time_per_complexity

        return BatchMetadata(
            batch_id=str(uuid.uuid4()),
            molecule_complexities=complexities.copy(),
            estimated_memory=estimated_memory,
            estimated_compute_time=estimated_compute_time,
            priority=1,
        )

    def _parse_memory_limit(self, memory_str: str) -> float:
        """Parse memory limit string to MB."""
        memory_str = memory_str.upper()

        if memory_str.endswith("GB"):
            return float(memory_str[:-2]) * 1024
        elif memory_str.endswith("MB"):
            return float(memory_str[:-2])
        elif memory_str.endswith("KB"):
            return float(memory_str[:-2]) / 1024
        else:
            # Assume MB
            return float(memory_str)

    def update_adaptive_parameters(self, batch_results: List[Dict[str, Any]]):
        """Update adaptive parameters based on batch performance."""
        if not batch_results:
            return

        # Calculate performance metrics
        success_count = sum(
            1 for result in batch_results if result.get("success", True)
        )
        success_rate = success_count / len(batch_results)

        avg_utilization = np.mean(
            [result.get("utilization", 0.8) for result in batch_results]
        )

        avg_throughput = np.mean(
            [result.get("throughput", 0.0) for result in batch_results]
        )

        # Update adaptive parameters
        self._adaptive_params["success_rate"] = (
            0.8 * self._adaptive_params["success_rate"] + 0.2 * success_rate
        )

        self._adaptive_params["avg_utilization"] = (
            0.8 * self._adaptive_params["avg_utilization"] + 0.2 * avg_utilization
        )

        # Adjust batch size based on performance
        current_size = self._adaptive_params["current_batch_size"]

        if success_rate > 0.9 and avg_utilization < self.config.target_utilization:
            # Increase batch size
            new_size = min(
                int(current_size * (1 + self.config.size_adjustment_factor)),
                self.config.max_batch_size,
            )
        elif success_rate < 0.7 or avg_utilization > 0.95:
            # Decrease batch size
            new_size = max(
                int(current_size * (1 - self.config.size_adjustment_factor)),
                self.config.min_batch_size,
            )
        else:
            new_size = current_size

        self._adaptive_params["current_batch_size"] = new_size

        # Store performance history
        self._batch_history.append(
            {
                "timestamp": time.time(),
                "success_rate": success_rate,
                "avg_utilization": avg_utilization,
                "avg_throughput": avg_throughput,
                "batch_size": current_size,
            }
        )

        # Limit history size
        if len(self._batch_history) > 100:
            self._batch_history = self._batch_history[-100:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "adaptive_params": self._adaptive_params.copy(),
            "config": self.config,
            "batch_history": self._batch_history[-10:],  # Last 10 entries
            "cache_stats": {
                "complexity_cache_size": len(
                    self.complexity_estimator._complexity_cache
                )
            },
        }


class DataTransferOptimizer:
    """Optimize data transfers for distributed QML inference."""

    def __init__(self, config: BatchingConfig = None):
        """
        Initialize data transfer optimizer.

        Args:
            config: Batching configuration
        """
        self.config = config or BatchingConfig()
        self._transfer_cache = {}
        self._cache_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=8)

    def optimize_transfers(
        self,
        batches: List[Tuple[List[GeometryInput], BatchMetadata]],
        target_devices: List[str],
    ) -> Dict[str, Any]:
        """
        Optimize data transfers for batches.

        Args:
            batches: List of (geometries, metadata) tuples
            target_devices: Target devices for transfer

        Returns:
            Optimized transfer plan
        """
        transfer_plan = {
            "batches": [],
            "transfer_order": [],
            "memory_layout": self.config.data_layout,
            "optimization": self.config.transfer_optimization,
            "estimated_time": 0.0,
            "estimated_bandwidth": 0.0,
        }

        # Apply transfer optimizations
        if self.config.transfer_optimization == TransferOptimization.PREFETCH:
            transfer_plan = self._optimize_prefetch(
                batches, target_devices, transfer_plan
            )
        elif self.config.transfer_optimization == TransferOptimization.PIPELINE:
            transfer_plan = self._optimize_pipeline(
                batches, target_devices, transfer_plan
            )
        elif self.config.transfer_optimization == TransferOptimization.ZERO_COPY:
            transfer_plan = self._optimize_zero_copy(
                batches, target_devices, transfer_plan
            )
        elif self.config.transfer_optimization == TransferOptimization.COMPRESSION:
            transfer_plan = self._optimize_compression(
                batches, target_devices, transfer_plan
            )
        else:
            transfer_plan = self._optimize_basic(batches, target_devices, transfer_plan)

        return transfer_plan

    def _optimize_prefetch(
        self,
        batches: List[Tuple[List[GeometryInput], BatchMetadata]],
        target_devices: List[str],
        transfer_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Optimize with prefetching."""
        prefetch_queue = []

        for i, (geometries, metadata) in enumerate(batches):
            # Estimate transfer time
            transfer_size = self._estimate_transfer_size(geometries)
            estimated_time = transfer_size / 1000.0  # Rough estimate: 1GB/s

            # Add prefetch information
            prefetch_info = {
                "batch_index": i,
                "batch_id": metadata.batch_id,
                "transfer_size_mb": transfer_size,
                "estimated_time": estimated_time,
                "prefetch_priority": metadata.priority,
                "target_device": (
                    target_devices[i % len(target_devices)] if target_devices else "cpu"
                ),
            }

            prefetch_queue.append(prefetch_info)

        # Sort by priority and estimated time
        prefetch_queue.sort(key=lambda x: (x["prefetch_priority"], x["estimated_time"]))

        transfer_plan["prefetch_queue"] = prefetch_queue
        transfer_plan["transfer_order"] = [
            item["batch_index"] for item in prefetch_queue
        ]
        transfer_plan["estimated_time"] = sum(
            item["estimated_time"] for item in prefetch_queue
        )

        return transfer_plan

    def _optimize_pipeline(
        self,
        batches: List[Tuple[List[GeometryInput], BatchMetadata]],
        target_devices: List[str],
        transfer_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Optimize with pipelined transfers."""
        pipeline_stages = []

        for i, (geometries, metadata) in enumerate(batches):
            # Create pipeline stage
            stage = {
                "batch_index": i,
                "batch_id": metadata.batch_id,
                "stage_type": "transfer_compute",
                "dependencies": [i - 1] if i > 0 else [],
                "estimated_duration": metadata.estimated_compute_time,
                "target_device": (
                    target_devices[i % len(target_devices)] if target_devices else "cpu"
                ),
            }

            pipeline_stages.append(stage)

        transfer_plan["pipeline_stages"] = pipeline_stages
        transfer_plan["parallelism_factor"] = (
            min(len(target_devices), len(batches)) if target_devices else 1
        )

        # Estimate pipelined execution time
        if pipeline_stages:
            max_stage_time = max(
                stage["estimated_duration"] for stage in pipeline_stages
            )
            parallelism = transfer_plan["parallelism_factor"]
            transfer_plan["estimated_time"] = (
                max_stage_time + (len(pipeline_stages) - 1) / parallelism
            )

        return transfer_plan

    def _optimize_zero_copy(
        self,
        batches: List[Tuple[List[GeometryInput], BatchMetadata]],
        target_devices: List[str],
        transfer_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Optimize with zero-copy transfers where possible."""
        zero_copy_batches = []

        for i, (geometries, metadata) in enumerate(batches):
            # Check if zero-copy is possible
            can_zero_copy = self._can_use_zero_copy(geometries, target_devices)

            batch_info = {
                "batch_index": i,
                "batch_id": metadata.batch_id,
                "zero_copy_eligible": can_zero_copy,
                "transfer_method": "zero_copy" if can_zero_copy else "standard",
                "estimated_speedup": 3.0 if can_zero_copy else 1.0,
            }

            zero_copy_batches.append(batch_info)

        transfer_plan["zero_copy_info"] = zero_copy_batches

        # Estimate time savings
        total_speedup = sum(batch["estimated_speedup"] for batch in zero_copy_batches)
        baseline_time = len(batches) * 1.0  # Baseline transfer time
        transfer_plan["estimated_time"] = baseline_time / (total_speedup / len(batches))

        return transfer_plan

    def _optimize_compression(
        self,
        batches: List[Tuple[List[GeometryInput], BatchMetadata]],
        target_devices: List[str],
        transfer_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Optimize with compression."""
        compression_info = []

        for i, (geometries, metadata) in enumerate(batches):
            # Estimate compression ratio
            transfer_size = self._estimate_transfer_size(geometries)
            compression_ratio = self._estimate_compression_ratio(geometries)

            compressed_size = transfer_size / compression_ratio
            compression_time = transfer_size * 0.0001  # Rough estimate

            batch_info = {
                "batch_index": i,
                "batch_id": metadata.batch_id,
                "original_size_mb": transfer_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": compression_ratio,
                "compression_time": compression_time,
                "net_benefit": (transfer_size - compressed_size) * 0.001
                - compression_time,
            }

            compression_info.append(batch_info)

        transfer_plan["compression_info"] = compression_info

        # Calculate total time with compression
        total_time = sum(
            info["compressed_size_mb"] * 0.001 + info["compression_time"]
            for info in compression_info
        )

        transfer_plan["estimated_time"] = total_time

        return transfer_plan

    def _optimize_basic(
        self,
        batches: List[Tuple[List[GeometryInput], BatchMetadata]],
        target_devices: List[str],
        transfer_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Basic transfer optimization."""
        basic_info = []

        for i, (geometries, metadata) in enumerate(batches):
            transfer_size = self._estimate_transfer_size(geometries)

            batch_info = {
                "batch_index": i,
                "batch_id": metadata.batch_id,
                "transfer_size_mb": transfer_size,
                "target_device": (
                    target_devices[i % len(target_devices)] if target_devices else "cpu"
                ),
            }

            basic_info.append(batch_info)

        transfer_plan["basic_info"] = basic_info
        transfer_plan["estimated_time"] = (
            sum(info["transfer_size_mb"] for info in basic_info) * 0.001
        )

        return transfer_plan

    def _estimate_transfer_size(self, geometries: List[GeometryInput]) -> float:
        """Estimate transfer size in MB."""
        if not geometries:
            return 0.0

        total_atoms = sum(len(geom.atomic_numbers) for geom in geometries)

        # Rough estimate: coordinates (3*4 bytes) + atomic numbers (4 bytes) + overhead
        bytes_per_atom = 3 * 4 + 4 + 8  # 24 bytes per atom
        total_bytes = total_atoms * bytes_per_atom

        return total_bytes / (1024 * 1024)  # Convert to MB

    def _can_use_zero_copy(
        self, geometries: List[GeometryInput], target_devices: List[str]
    ) -> bool:
        """Check if zero-copy transfer is possible."""
        # Simplified check - in practice would consider memory layout, device types, etc.
        if not target_devices:
            return False

        # Zero-copy is possible for GPU-to-GPU transfers or shared memory
        return any("cuda" in device for device in target_devices)

    def _estimate_compression_ratio(self, geometries: List[GeometryInput]) -> float:
        """Estimate compression ratio for molecular data."""
        if not geometries:
            return 1.0

        # Molecular coordinate data typically compresses well
        # Estimate based on data regularity
        total_atoms = sum(len(geom.atomic_numbers) for geom in geometries)
        unique_elements = len(
            set(atom for geom in geometries for atom in geom.atomic_numbers)
        )

        # More diverse elements = better compression for atomic numbers
        element_diversity = unique_elements / total_atoms if total_atoms > 0 else 1.0

        # Coordinate compression depends on molecular structure
        base_compression = 2.5  # Typical compression ratio
        diversity_bonus = element_diversity * 0.5

        return min(base_compression + diversity_bonus, 4.0)  # Cap at 4:1 ratio


# Convenience functions
def create_optimized_batches(
    geometries: List[GeometryInput],
    properties: List[QuantumProperty],
    strategy: str = "adaptive_size",
    batch_size: int = 64,
    memory_limit: str = "2GB",
    **kwargs,
) -> List[Tuple[List[GeometryInput], BatchMetadata]]:
    """
    Convenience function to create optimized batches.

    Args:
        geometries: Input geometries
        properties: Properties to predict
        strategy: Batching strategy
        batch_size: Base batch size
        memory_limit: Memory limit per batch
        **kwargs: Additional configuration options

    Returns:
        List of optimized batches with metadata
    """
    config = BatchingConfig(
        strategy=BatchingStrategy(strategy),
        base_batch_size=batch_size,
        max_memory_per_batch=memory_limit,
        **kwargs,
    )

    creator = OptimizedBatchCreator(config)
    return creator.create_optimized_batches(geometries, properties)


def optimize_data_transfers(
    batches: List[Tuple[List[GeometryInput], BatchMetadata]],
    target_devices: List[str],
    optimization: str = "pipeline",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to optimize data transfers.

    Args:
        batches: Batches to optimize
        target_devices: Target devices
        optimization: Transfer optimization strategy
        **kwargs: Additional configuration options

    Returns:
        Optimized transfer plan
    """
    config = BatchingConfig(
        transfer_optimization=TransferOptimization(optimization), **kwargs
    )

    optimizer = DataTransferOptimizer(config)
    return optimizer.optimize_transfers(batches, target_devices)

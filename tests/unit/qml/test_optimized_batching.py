"""
Unit tests for optimized batching and data transfer mechanisms.
"""

import time

import numpy as np
import pytest
from chemi_accel.qml.base import QuantumProperty
from chemi_accel.qml.on_the_fly import GeometryInput
from chemi_accel.qml.optimized_batching import (
    BatchingConfig,
    BatchingStrategy,
    BatchMetadata,
    ComplexityEstimator,
    DataLayout,
    DataTransferOptimizer,
    MemoryEstimator,
    OptimizedBatchCreator,
    TransferOptimization,
    create_optimized_batches,
    optimize_data_transfers,
)


@pytest.fixture
def sample_geometries():
    """Create sample molecular geometries with varying complexity."""
    geometries = []

    # Simple molecule (water)
    geometries.append(
        GeometryInput(
            coordinates=np.array(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]  # O  # H  # H
            ),
            atomic_numbers=np.array([8, 1, 1]),
            charge=0,
            multiplicity=1,
        )
    )

    # Medium molecule (ethanol)
    geometries.append(
        GeometryInput(
            coordinates=np.array(
                [
                    [1.2, -0.2, 0.0],  # C
                    [0.0, 0.6, 0.0],  # C
                    [-1.3, -0.2, 0.0],  # O
                    [2.1, 0.4, 0.0],  # H
                    [1.2, -0.8, 0.9],  # H
                    [1.2, -0.8, -0.9],  # H
                    [0.0, 1.3, 0.9],  # H
                    [0.0, 1.3, -0.9],  # H
                    [-2.0, 0.3, 0.0],  # H
                ]
            ),
            atomic_numbers=np.array([6, 6, 8, 1, 1, 1, 1, 1, 1]),
            charge=0,
            multiplicity=1,
        )
    )

    # Large molecule (caffeine-like)
    large_coords = np.random.randn(24, 3) * 2  # Random but realistic scale
    large_atoms = np.array(
        [6, 6, 6, 6, 6, 6, 7, 7, 8, 8] + [1] * 14
    )  # C, N, O, H atoms

    geometries.append(
        GeometryInput(
            coordinates=large_coords,
            atomic_numbers=large_atoms,
            charge=0,
            multiplicity=1,
        )
    )

    return geometries


@pytest.fixture
def batching_config():
    """Create a test batching configuration."""
    return BatchingConfig(
        strategy=BatchingStrategy.ADAPTIVE_SIZE,
        base_batch_size=4,
        min_batch_size=2,
        max_batch_size=8,
        max_memory_per_batch="512MB",
        data_layout=DataLayout.STRUCTURE_OF_ARRAYS,
        transfer_optimization=TransferOptimization.PIPELINE,
    )


class TestBatchingConfig:
    """Test batching configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BatchingConfig()

        assert config.strategy == BatchingStrategy.ADAPTIVE_SIZE
        assert config.base_batch_size == 64
        assert config.min_batch_size == 8
        assert config.max_batch_size == 512
        assert config.data_layout == DataLayout.STRUCTURE_OF_ARRAYS
        assert config.transfer_optimization == TransferOptimization.PIPELINE

    def test_custom_config(self):
        """Test custom configuration."""
        config = BatchingConfig(
            strategy=BatchingStrategy.MEMORY_AWARE,
            base_batch_size=32,
            max_memory_per_batch="1GB",
            enable_compression=True,
        )

        assert config.strategy == BatchingStrategy.MEMORY_AWARE
        assert config.base_batch_size == 32
        assert config.max_memory_per_batch == "1GB"
        assert config.enable_compression is True


class TestBatchMetadata:
    """Test batch metadata."""

    def test_metadata_creation(self):
        """Test batch metadata creation."""
        complexities = [1.0, 2.5, 1.8]

        metadata = BatchMetadata(
            batch_id="test_batch_1",
            molecule_complexities=complexities,
            estimated_memory=256.0,
            estimated_compute_time=1.5,
            priority=2,
        )

        assert metadata.batch_id == "test_batch_1"
        assert metadata.molecule_complexities == complexities
        assert metadata.estimated_memory == 256.0
        assert metadata.estimated_compute_time == 1.5
        assert metadata.priority == 2
        assert metadata.creation_time > 0


class TestComplexityEstimator:
    """Test molecular complexity estimation."""

    def test_estimator_creation(self):
        """Test complexity estimator creation."""
        estimator = ComplexityEstimator()

        assert hasattr(estimator, "_atom_weights")
        assert hasattr(estimator, "_complexity_cache")
        assert estimator._atom_weights[1] == 1.0  # Hydrogen
        assert estimator._atom_weights[6] == 2.0  # Carbon

    def test_molecule_complexity_estimation(self, sample_geometries):
        """Test complexity estimation for individual molecules."""
        estimator = ComplexityEstimator()

        complexities = []
        for geom in sample_geometries:
            complexity = estimator.estimate_molecule_complexity(geom)
            complexities.append(complexity)

            assert complexity > 0
            assert isinstance(complexity, float)

        # Larger molecules should generally have higher complexity
        assert complexities[2] > complexities[1] > complexities[0]

    def test_batch_complexity_estimation(self, sample_geometries):
        """Test batch complexity estimation."""
        estimator = ComplexityEstimator()

        batch_complexity = estimator.estimate_batch_complexity(sample_geometries)

        assert batch_complexity > 0
        assert isinstance(batch_complexity, float)

        # Batch complexity should be roughly the sum of individual complexities
        individual_sum = sum(
            estimator.estimate_molecule_complexity(geom) for geom in sample_geometries
        )

        # Should be within reasonable range due to batch efficiency factors
        assert batch_complexity > individual_sum * 0.5
        assert batch_complexity < individual_sum * 1.5

    def test_complexity_caching(self, sample_geometries):
        """Test complexity caching mechanism."""
        estimator = ComplexityEstimator()

        # Calculate complexity first time
        geom = sample_geometries[0]
        complexity1 = estimator.estimate_molecule_complexity(geom)

        # Check that it was cached
        assert len(estimator._complexity_cache) > 0

        # Calculate again - should use cache
        complexity2 = estimator.estimate_molecule_complexity(geom)

        assert complexity1 == complexity2

    def test_empty_geometry_handling(self):
        """Test handling of empty geometry list."""
        estimator = ComplexityEstimator()

        batch_complexity = estimator.estimate_batch_complexity([])
        assert batch_complexity == 0.0


class TestMemoryEstimator:
    """Test memory requirement estimation."""

    def test_estimator_creation(self):
        """Test memory estimator creation."""
        estimator = MemoryEstimator(safety_factor=0.9)

        assert estimator.safety_factor == 0.9
        assert hasattr(estimator, "_base_memory_per_atom")

    def test_geometry_memory_estimation(self, sample_geometries):
        """Test memory estimation for individual geometries."""
        estimator = MemoryEstimator()

        for geom in sample_geometries:
            memory_mb = estimator.estimate_geometry_memory(geom)

            assert memory_mb > 0
            assert isinstance(memory_mb, float)

            # Memory should scale roughly with number of atoms
            atom_count = len(geom.atomic_numbers)
            assert memory_mb > atom_count * 0.001  # At least some memory per atom

    def test_batch_memory_estimation(self, sample_geometries):
        """Test memory estimation for batches."""
        estimator = MemoryEstimator()

        # Test different data layouts
        layouts = [
            DataLayout.ARRAY_OF_STRUCTURES,
            DataLayout.STRUCTURE_OF_ARRAYS,
            DataLayout.COMPRESSED,
        ]

        for layout in layouts:
            memory_mb = estimator.estimate_batch_memory(
                sample_geometries, precision="float32", data_layout=layout
            )

            assert memory_mb > 0
            assert isinstance(memory_mb, float)

        # Compressed layout should use less memory
        aos_memory = estimator.estimate_batch_memory(
            sample_geometries, data_layout=DataLayout.ARRAY_OF_STRUCTURES
        )
        compressed_memory = estimator.estimate_batch_memory(
            sample_geometries, data_layout=DataLayout.COMPRESSED
        )

        assert compressed_memory < aos_memory

    def test_precision_impact(self, sample_geometries):
        """Test impact of precision on memory estimation."""
        estimator = MemoryEstimator()

        float32_memory = estimator.estimate_batch_memory(
            sample_geometries, precision="float32"
        )
        float64_memory = estimator.estimate_batch_memory(
            sample_geometries, precision="float64"
        )

        # float64 should use more memory than float32
        assert float64_memory > float32_memory

    def test_empty_batch_handling(self):
        """Test handling of empty batch."""
        estimator = MemoryEstimator()

        memory_mb = estimator.estimate_batch_memory([])
        assert memory_mb == 0.0


class TestOptimizedBatchCreator:
    """Test optimized batch creation."""

    def test_creator_initialization(self, batching_config):
        """Test batch creator initialization."""
        creator = OptimizedBatchCreator(batching_config)

        assert creator.config == batching_config
        assert hasattr(creator, "complexity_estimator")
        assert hasattr(creator, "memory_estimator")
        assert hasattr(creator, "_adaptive_params")

    def test_fixed_size_batching(self, sample_geometries):
        """Test fixed size batching strategy."""
        config = BatchingConfig(strategy=BatchingStrategy.FIXED_SIZE, base_batch_size=2)
        creator = OptimizedBatchCreator(config)

        batches = creator.create_optimized_batches(
            sample_geometries, [QuantumProperty.TOTAL_ENERGY]
        )

        assert len(batches) > 0

        # Check batch sizes (last batch might be smaller)
        for i, (batch_geoms, metadata) in enumerate(batches[:-1]):
            assert len(batch_geoms) == 2

        # Check that all geometries are included
        total_molecules = sum(len(batch_geoms) for batch_geoms, _ in batches)
        assert total_molecules == len(sample_geometries)

    def test_adaptive_size_batching(self, sample_geometries):
        """Test adaptive size batching strategy."""
        config = BatchingConfig(
            strategy=BatchingStrategy.ADAPTIVE_SIZE,
            base_batch_size=4,
            min_batch_size=1,
            max_batch_size=6,
        )
        creator = OptimizedBatchCreator(config)

        batches = creator.create_optimized_batches(
            sample_geometries, [QuantumProperty.TOTAL_ENERGY]
        )

        assert len(batches) > 0

        # Check that batch sizes are within limits
        for batch_geoms, metadata in batches:
            assert len(batch_geoms) >= config.min_batch_size
            assert len(batch_geoms) <= config.max_batch_size

        # Check metadata
        for batch_geoms, metadata in batches:
            assert isinstance(metadata, BatchMetadata)
            assert len(metadata.molecule_complexities) == len(batch_geoms)
            assert metadata.estimated_memory > 0
            assert metadata.estimated_compute_time > 0

    def test_memory_aware_batching(self, sample_geometries):
        """Test memory-aware batching strategy."""
        config = BatchingConfig(
            strategy=BatchingStrategy.MEMORY_AWARE,
            max_memory_per_batch="100MB",  # Small limit to force multiple batches
        )
        creator = OptimizedBatchCreator(config)

        # Create many geometries to test memory constraints
        many_geometries = sample_geometries * 10

        batches = creator.create_optimized_batches(
            many_geometries, [QuantumProperty.TOTAL_ENERGY]
        )

        assert len(batches) > 1  # Should create multiple batches due to memory limit

        # Check that batches respect memory limits
        for batch_geoms, metadata in batches:
            assert metadata.estimated_memory <= 100.0  # 100MB limit

    def test_latency_optimized_batching(self, sample_geometries):
        """Test latency-optimized batching strategy."""
        config = BatchingConfig(
            strategy=BatchingStrategy.LATENCY_OPTIMIZED, base_batch_size=8
        )
        creator = OptimizedBatchCreator(config)

        batches = creator.create_optimized_batches(
            sample_geometries, [QuantumProperty.TOTAL_ENERGY]
        )

        assert len(batches) > 0

        # Latency-optimized should create smaller batches with higher priority
        for batch_geoms, metadata in batches:
            assert len(batch_geoms) <= 4  # Should be smaller than base size
            assert metadata.priority >= 1

    def test_throughput_optimized_batching(self, sample_geometries):
        """Test throughput-optimized batching strategy."""
        config = BatchingConfig(
            strategy=BatchingStrategy.THROUGHPUT_OPTIMIZED,
            base_batch_size=2,
            max_batch_size=10,
        )
        creator = OptimizedBatchCreator(config)

        batches = creator.create_optimized_batches(
            sample_geometries, [QuantumProperty.TOTAL_ENERGY]
        )

        assert len(batches) > 0

        # Throughput-optimized should create larger batches
        for batch_geoms, metadata in batches:
            # Should use larger batch sizes for better throughput
            assert len(batch_geoms) >= config.base_batch_size

    def test_load_balanced_batching(self, sample_geometries):
        """Test load-balanced batching strategy."""
        config = BatchingConfig(strategy=BatchingStrategy.LOAD_BALANCED)
        creator = OptimizedBatchCreator(config)

        # Mock device constraints
        device_constraints = {
            "num_devices": 2,
            "capabilities": [1.0, 2.0],  # Different device capabilities
        }

        batches = creator.create_optimized_batches(
            sample_geometries, [QuantumProperty.TOTAL_ENERGY], device_constraints
        )

        assert len(batches) > 0

        # Check that batches have device assignments
        for batch_geoms, metadata in batches:
            assert len(metadata.target_devices) > 0

    def test_hierarchical_batching(self, sample_geometries):
        """Test hierarchical batching strategy."""
        config = BatchingConfig(
            strategy=BatchingStrategy.HIERARCHICAL, base_batch_size=2
        )
        creator = OptimizedBatchCreator(config)

        # Create more geometries for better hierarchical splitting
        many_geometries = sample_geometries * 5

        batches = creator.create_optimized_batches(
            many_geometries, [QuantumProperty.TOTAL_ENERGY]
        )

        assert len(batches) > 0

        # Check that batches have different priorities
        priorities = [metadata.priority for _, metadata in batches]
        assert len(set(priorities)) > 1  # Should have multiple priority levels

    def test_empty_geometry_handling(self):
        """Test handling of empty geometry list."""
        creator = OptimizedBatchCreator()

        batches = creator.create_optimized_batches([], [QuantumProperty.TOTAL_ENERGY])

        assert len(batches) == 0

    def test_adaptive_parameter_update(self):
        """Test adaptive parameter updates."""
        creator = OptimizedBatchCreator()

        # Mock batch results
        batch_results = [
            {"success": True, "utilization": 0.8, "throughput": 100.0},
            {"success": True, "utilization": 0.9, "throughput": 120.0},
            {"success": False, "utilization": 0.6, "throughput": 50.0},
        ]

        initial_batch_size = creator._adaptive_params["current_batch_size"]

        creator.update_adaptive_parameters(batch_results)

        # Parameters should be updated
        assert "success_rate" in creator._adaptive_params
        assert "avg_utilization" in creator._adaptive_params
        assert len(creator._batch_history) > 0

    def test_performance_metrics(self, sample_geometries):
        """Test performance metrics collection."""
        creator = OptimizedBatchCreator()

        # Create some batches to generate metrics
        creator.create_optimized_batches(
            sample_geometries, [QuantumProperty.TOTAL_ENERGY]
        )

        metrics = creator.get_performance_metrics()

        assert "adaptive_params" in metrics
        assert "config" in metrics
        assert "cache_stats" in metrics


class TestDataTransferOptimizer:
    """Test data transfer optimization."""

    def test_optimizer_creation(self, batching_config):
        """Test data transfer optimizer creation."""
        optimizer = DataTransferOptimizer(batching_config)

        assert optimizer.config == batching_config
        assert hasattr(optimizer, "_transfer_cache")

    def test_basic_transfer_optimization(self, sample_geometries):
        """Test basic transfer optimization."""
        optimizer = DataTransferOptimizer()

        # Create mock batches
        batch_metadata = BatchMetadata(
            batch_id="test_batch",
            molecule_complexities=[1.0] * len(sample_geometries),
            estimated_memory=100.0,
            estimated_compute_time=1.0,
        )

        batches = [(sample_geometries, batch_metadata)]
        target_devices = ["cuda:0", "cuda:1"]

        transfer_plan = optimizer.optimize_transfers(batches, target_devices)

        assert "batches" in transfer_plan
        assert "transfer_order" in transfer_plan
        assert "memory_layout" in transfer_plan
        assert "optimization" in transfer_plan
        assert "estimated_time" in transfer_plan
        assert transfer_plan["estimated_time"] >= 0

    def test_prefetch_optimization(self, sample_geometries):
        """Test prefetch transfer optimization."""
        config = BatchingConfig(transfer_optimization=TransferOptimization.PREFETCH)
        optimizer = DataTransferOptimizer(config)

        # Create multiple batches
        batches = []
        for i in range(3):
            metadata = BatchMetadata(
                batch_id=f"batch_{i}",
                molecule_complexities=[1.0],
                estimated_memory=50.0,
                estimated_compute_time=1.0,
                priority=i + 1,
            )
            batches.append(([sample_geometries[0]], metadata))

        target_devices = ["cuda:0"]

        transfer_plan = optimizer.optimize_transfers(batches, target_devices)

        assert "prefetch_queue" in transfer_plan
        assert len(transfer_plan["prefetch_queue"]) == len(batches)

        # Check that prefetch queue is properly ordered
        prefetch_queue = transfer_plan["prefetch_queue"]
        assert all("batch_index" in item for item in prefetch_queue)
        assert all("estimated_time" in item for item in prefetch_queue)

    def test_pipeline_optimization(self, sample_geometries):
        """Test pipeline transfer optimization."""
        config = BatchingConfig(transfer_optimization=TransferOptimization.PIPELINE)
        optimizer = DataTransferOptimizer(config)

        # Create batches
        batches = []
        for i in range(2):
            metadata = BatchMetadata(
                batch_id=f"batch_{i}",
                molecule_complexities=[1.0],
                estimated_memory=50.0,
                estimated_compute_time=2.0,
            )
            batches.append(([sample_geometries[0]], metadata))

        target_devices = ["cuda:0", "cuda:1"]

        transfer_plan = optimizer.optimize_transfers(batches, target_devices)

        assert "pipeline_stages" in transfer_plan
        assert "parallelism_factor" in transfer_plan
        assert len(transfer_plan["pipeline_stages"]) == len(batches)

        # Check pipeline stage structure
        for stage in transfer_plan["pipeline_stages"]:
            assert "batch_index" in stage
            assert "stage_type" in stage
            assert "estimated_duration" in stage

    def test_zero_copy_optimization(self, sample_geometries):
        """Test zero-copy transfer optimization."""
        config = BatchingConfig(transfer_optimization=TransferOptimization.ZERO_COPY)
        optimizer = DataTransferOptimizer(config)

        batches = [
            (
                sample_geometries,
                BatchMetadata(
                    batch_id="test_batch",
                    molecule_complexities=[1.0] * len(sample_geometries),
                    estimated_memory=100.0,
                    estimated_compute_time=1.0,
                ),
            )
        ]

        target_devices = ["cuda:0", "cuda:1"]

        transfer_plan = optimizer.optimize_transfers(batches, target_devices)

        assert "zero_copy_info" in transfer_plan
        assert len(transfer_plan["zero_copy_info"]) == len(batches)

        # Check zero-copy information
        for info in transfer_plan["zero_copy_info"]:
            assert "zero_copy_eligible" in info
            assert "transfer_method" in info
            assert "estimated_speedup" in info

    def test_compression_optimization(self, sample_geometries):
        """Test compression transfer optimization."""
        config = BatchingConfig(transfer_optimization=TransferOptimization.COMPRESSION)
        optimizer = DataTransferOptimizer(config)

        batches = [
            (
                sample_geometries,
                BatchMetadata(
                    batch_id="test_batch",
                    molecule_complexities=[1.0] * len(sample_geometries),
                    estimated_memory=100.0,
                    estimated_compute_time=1.0,
                ),
            )
        ]

        target_devices = ["cuda:0"]

        transfer_plan = optimizer.optimize_transfers(batches, target_devices)

        assert "compression_info" in transfer_plan
        assert len(transfer_plan["compression_info"]) == len(batches)

        # Check compression information
        for info in transfer_plan["compression_info"]:
            assert "compression_ratio" in info
            assert "compressed_size_mb" in info
            assert "compression_time" in info
            assert info["compression_ratio"] > 1.0  # Should provide some compression


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_optimized_batches(self, sample_geometries):
        """Test create_optimized_batches convenience function."""
        batches = create_optimized_batches(
            sample_geometries,
            [QuantumProperty.TOTAL_ENERGY],
            strategy="fixed_size",
            batch_size=2,
            memory_limit="512MB",
        )

        assert len(batches) > 0
        assert all(isinstance(batch[0], list) for batch in batches)
        assert all(isinstance(batch[1], BatchMetadata) for batch in batches)

    def test_optimize_data_transfers(self, sample_geometries):
        """Test optimize_data_transfers convenience function."""
        # First create batches
        batches = create_optimized_batches(
            sample_geometries, [QuantumProperty.TOTAL_ENERGY], batch_size=2
        )

        # Then optimize transfers
        transfer_plan = optimize_data_transfers(
            batches, ["cuda:0", "cuda:1"], optimization="prefetch"
        )

        assert "estimated_time" in transfer_plan
        assert "optimization" in transfer_plan
        assert transfer_plan["optimization"] == TransferOptimization.PREFETCH


@pytest.mark.integration
class TestBatchingIntegration:
    """Integration tests for batching functionality."""

    def test_end_to_end_batching(self, sample_geometries):
        """Test end-to-end batching workflow."""
        # Create optimized batches
        batches = create_optimized_batches(
            sample_geometries,
            [QuantumProperty.TOTAL_ENERGY, QuantumProperty.HOMO_ENERGY],
            strategy="adaptive_size",
            batch_size=2,
            memory_limit="1GB",
        )

        assert len(batches) > 0

        # Optimize transfers
        transfer_plan = optimize_data_transfers(
            batches, ["cuda:0", "cuda:1"], optimization="pipeline"
        )

        assert transfer_plan["estimated_time"] > 0

        # Verify all molecules are included
        total_molecules = sum(len(batch_geoms) for batch_geoms, _ in batches)
        assert total_molecules == len(sample_geometries)

    def test_performance_comparison(self, sample_geometries):
        """Test performance comparison between strategies."""
        strategies = [
            BatchingStrategy.FIXED_SIZE,
            BatchingStrategy.ADAPTIVE_SIZE,
            BatchingStrategy.MEMORY_AWARE,
        ]

        results = {}

        for strategy in strategies:
            config = BatchingConfig(
                strategy=strategy, base_batch_size=2, max_memory_per_batch="1GB"
            )

            creator = OptimizedBatchCreator(config)

            start_time = time.time()
            batches = creator.create_optimized_batches(
                sample_geometries, [QuantumProperty.TOTAL_ENERGY]
            )
            end_time = time.time()

            results[strategy] = {
                "num_batches": len(batches),
                "creation_time": end_time - start_time,
                "total_molecules": sum(len(batch_geoms) for batch_geoms, _ in batches),
            }

        # All strategies should process all molecules
        molecule_counts = [result["total_molecules"] for result in results.values()]
        assert all(count == len(sample_geometries) for count in molecule_counts)

        # All strategies should create valid batches
        batch_counts = [result["num_batches"] for result in results.values()]
        assert all(count > 0 for count in batch_counts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

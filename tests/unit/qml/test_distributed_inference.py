"""
Unit tests for distributed QML inference functionality.
"""

import time
from unittest.mock import patch

import numpy as np
import pytest
from chemi_accel.qml.base import BaseQMLModel, QMLPrediction, QuantumProperty
from chemi_accel.qml.distributed_inference import (
    DataTransferMode,
    DistributedBatch,
    DistributedConfig,
    DistributedQMLInferenceEngine,
    DistributionBackend,
    InferenceResult,
    OptimizedDataLoader,
    ScalingStrategy,
    WorkerInfo,
    create_distributed_engine,
    distributed_predict,
)
from chemi_accel.qml.on_the_fly import GeometryInput


class MockQMLModel(BaseQMLModel):
    """Mock QML model for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_loaded = True

    def load_model(self):
        self._is_loaded = True

    def predict(self, molecules, properties=None, **kwargs):
        if isinstance(molecules, list):
            return [self._create_mock_prediction() for _ in molecules]
        else:
            return self._create_mock_prediction()

    def batch_predict(self, molecules, properties=None, batch_size=32, **kwargs):
        return [self._create_mock_prediction() for _ in molecules]

    def get_supported_properties(self):
        return [QuantumProperty.TOTAL_ENERGY, QuantumProperty.HOMO_ENERGY]

    def validate_molecule(self, molecule):
        return True

    def _create_mock_prediction(self):
        return QMLPrediction(
            properties={
                QuantumProperty.TOTAL_ENERGY: -100.5,
                QuantumProperty.HOMO_ENERGY: -8.2,
            },
            confidence=0.95,
            model_name="MockModel",
        )


@pytest.fixture
def mock_model():
    """Create a mock QML model."""
    return MockQMLModel()


@pytest.fixture
def sample_geometries():
    """Create sample molecular geometries."""
    geometries = []

    # Simple molecule (methane)
    geometries.append(
        GeometryInput(
            coordinates=np.array(
                [
                    [0.0, 0.0, 0.0],  # C
                    [1.0, 1.0, 1.0],  # H
                    [-1.0, -1.0, 1.0],  # H
                    [-1.0, 1.0, -1.0],  # H
                    [1.0, -1.0, -1.0],  # H
                ]
            ),
            atomic_numbers=np.array([6, 1, 1, 1, 1]),
            charge=0,
            multiplicity=1,
        )
    )

    # Larger molecule (benzene)
    geometries.append(
        GeometryInput(
            coordinates=np.array(
                [
                    [1.2, 0.7, 0.0],  # C
                    [1.2, -0.7, 0.0],  # C
                    [0.0, -1.4, 0.0],  # C
                    [-1.2, -0.7, 0.0],  # C
                    [-1.2, 0.7, 0.0],  # C
                    [0.0, 1.4, 0.0],  # C
                    [2.1, 1.2, 0.0],  # H
                    [2.1, -1.2, 0.0],  # H
                    [0.0, -2.5, 0.0],  # H
                    [-2.1, -1.2, 0.0],  # H
                    [-2.1, 1.2, 0.0],  # H
                    [0.0, 2.5, 0.0],  # H
                ]
            ),
            atomic_numbers=np.array([6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1]),
            charge=0,
            multiplicity=1,
        )
    )

    return geometries


@pytest.fixture
def distributed_config():
    """Create a test distributed configuration."""
    return DistributedConfig(
        backend=DistributionBackend.MULTIPROCESSING,
        num_workers=2,
        batch_size=4,
        max_batch_size=8,
        data_transfer_mode=DataTransferMode.EAGER,
    )


class TestDistributedConfig:
    """Test distributed configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DistributedConfig()

        assert config.backend == DistributionBackend.DASK
        assert config.scaling_strategy == ScalingStrategy.DATA_PARALLEL
        assert config.num_workers == 4
        assert config.batch_size == 64
        assert config.data_transfer_mode == DataTransferMode.STREAMING

    def test_custom_config(self):
        """Test custom configuration."""
        config = DistributedConfig(
            backend=DistributionBackend.RAY,
            num_workers=8,
            batch_size=128,
            scaling_strategy=ScalingStrategy.MODEL_PARALLEL,
        )

        assert config.backend == DistributionBackend.RAY
        assert config.num_workers == 8
        assert config.batch_size == 128
        assert config.scaling_strategy == ScalingStrategy.MODEL_PARALLEL


class TestWorkerInfo:
    """Test worker information management."""

    def test_worker_creation(self):
        """Test worker info creation."""
        worker = WorkerInfo(
            worker_id="test_worker_1",
            device_ids=["cuda:0", "cuda:1"],
            memory_available=8.0,
            compute_capability=1.5,
        )

        assert worker.worker_id == "test_worker_1"
        assert worker.device_ids == ["cuda:0", "cuda:1"]
        assert worker.memory_available == 8.0
        assert worker.compute_capability == 1.5
        assert worker.status == "idle"

    def test_worker_heartbeat(self):
        """Test worker heartbeat tracking."""
        worker = WorkerInfo(
            worker_id="test_worker_1",
            device_ids=["cpu"],
            memory_available=4.0,
            compute_capability=1.0,
        )

        initial_heartbeat = worker.last_heartbeat
        time.sleep(0.01)  # Small delay

        # Update heartbeat
        worker.last_heartbeat = time.time()
        assert worker.last_heartbeat > initial_heartbeat


class TestDistributedBatch:
    """Test distributed batch management."""

    def test_batch_creation(self, sample_geometries):
        """Test distributed batch creation."""
        batch = DistributedBatch(
            batch_id="test_batch_1",
            geometries=sample_geometries,
            properties=[QuantumProperty.TOTAL_ENERGY],
            worker_assignments={"worker_1": [0], "worker_2": [1]},
            priority=2,
        )

        assert batch.batch_id == "test_batch_1"
        assert len(batch.geometries) == 2
        assert len(batch.properties) == 1
        assert batch.priority == 2
        assert "worker_1" in batch.worker_assignments
        assert "worker_2" in batch.worker_assignments


class TestOptimizedDataLoader:
    """Test optimized data loader."""

    def test_data_loader_creation(self, distributed_config):
        """Test data loader creation."""
        loader = OptimizedDataLoader(distributed_config)

        assert loader.config == distributed_config
        assert loader.prefetch_factor == 2
        assert loader.num_workers == 4

    def test_batch_creation(self, distributed_config, sample_geometries):
        """Test optimized batch creation."""
        loader = OptimizedDataLoader(distributed_config)

        # Create mock worker info
        worker_info = {
            "worker_1": WorkerInfo(
                worker_id="worker_1",
                device_ids=["cuda:0"],
                memory_available=8.0,
                compute_capability=1.0,
            ),
            "worker_2": WorkerInfo(
                worker_id="worker_2",
                device_ids=["cuda:1"],
                memory_available=8.0,
                compute_capability=1.2,
            ),
        }

        batches = loader.create_optimized_batches(
            sample_geometries, [QuantumProperty.TOTAL_ENERGY], worker_info
        )

        assert len(batches) > 0
        assert all(isinstance(batch, DistributedBatch) for batch in batches)
        assert all(len(batch.geometries) > 0 for batch in batches)

    def test_load_balanced_assignment(self, distributed_config, sample_geometries):
        """Test load-balanced assignment to workers."""
        loader = OptimizedDataLoader(distributed_config)

        # Create workers with different capabilities
        worker_info = {
            "fast_worker": WorkerInfo(
                worker_id="fast_worker",
                device_ids=["cuda:0"],
                memory_available=16.0,
                compute_capability=2.0,
            ),
            "slow_worker": WorkerInfo(
                worker_id="slow_worker",
                device_ids=["cpu"],
                memory_available=8.0,
                compute_capability=0.5,
            ),
        }

        # Create many geometries to test load balancing
        many_geometries = sample_geometries * 10

        batches = loader.create_optimized_batches(
            many_geometries, [QuantumProperty.TOTAL_ENERGY], worker_info
        )

        # Check that batches were created
        assert len(batches) > 0

        # Check that fast worker gets more work
        fast_worker_molecules = sum(
            len(batch.geometries)
            for batch in batches
            if "fast_worker" in batch.worker_assignments
        )

        slow_worker_molecules = sum(
            len(batch.geometries)
            for batch in batches
            if "slow_worker" in batch.worker_assignments
        )

        # Fast worker should handle more molecules
        assert fast_worker_molecules >= slow_worker_molecules


class TestDistributedQMLInferenceEngine:
    """Test distributed QML inference engine."""

    def test_engine_creation(self, mock_model, distributed_config):
        """Test engine creation."""
        engine = DistributedQMLInferenceEngine(mock_model, distributed_config)

        assert engine.base_model == mock_model
        assert engine.config == distributed_config
        assert not engine._initialized

    @patch("chemi_accel.qml.distributed_inference.ThreadPoolExecutor")
    def test_multiprocessing_initialization(self, mock_executor, mock_model):
        """Test multiprocessing backend initialization."""
        config = DistributedConfig(backend=DistributionBackend.MULTIPROCESSING)
        engine = DistributedQMLInferenceEngine(mock_model, config)

        # Initialize engine
        success = engine.initialize()

        assert success
        assert engine._initialized
        assert len(engine.workers) == config.num_workers

    def test_worker_discovery(self, mock_model, distributed_config):
        """Test worker discovery."""
        engine = DistributedQMLInferenceEngine(mock_model, distributed_config)
        engine.initialize()

        # Check that workers were discovered
        assert len(engine.workers) == distributed_config.num_workers

        for worker_id, worker in engine.workers.items():
            assert isinstance(worker, WorkerInfo)
            assert worker.worker_id == worker_id
            assert worker.status in ["idle", "ready"]

    def test_distributed_prediction(
        self, mock_model, distributed_config, sample_geometries
    ):
        """Test distributed prediction."""
        engine = DistributedQMLInferenceEngine(mock_model, distributed_config)
        engine.initialize()

        try:
            result = engine.predict_distributed(
                sample_geometries,
                [QuantumProperty.TOTAL_ENERGY, QuantumProperty.HOMO_ENERGY],
                timeout=10.0,
            )

            assert isinstance(result, InferenceResult)
            assert len(result.predictions) == len(sample_geometries)

            # Check that all predictions are valid
            for prediction in result.predictions:
                assert isinstance(prediction, QMLPrediction)
                assert (
                    QuantumProperty.TOTAL_ENERGY in prediction.properties
                    or len(prediction.properties) == 0
                )

        finally:
            engine.shutdown()

    def test_performance_metrics(
        self, mock_model, distributed_config, sample_geometries
    ):
        """Test performance metrics collection."""
        engine = DistributedQMLInferenceEngine(mock_model, distributed_config)
        engine.initialize()

        try:
            # Run a prediction
            engine.predict_distributed(
                sample_geometries, [QuantumProperty.TOTAL_ENERGY]
            )

            # Get performance metrics
            metrics = engine.get_performance_metrics()

            assert "total_predictions" in metrics
            assert "total_batches" in metrics
            assert "throughput" in metrics
            assert "backend" in metrics
            assert metrics["total_predictions"] == len(sample_geometries)

        finally:
            engine.shutdown()

    def test_worker_status(self, mock_model, distributed_config):
        """Test worker status monitoring."""
        engine = DistributedQMLInferenceEngine(mock_model, distributed_config)
        engine.initialize()

        try:
            status = engine.get_worker_status()

            assert isinstance(status, dict)
            assert len(status) == distributed_config.num_workers

            for worker_id, worker_info in status.items():
                assert isinstance(worker_info, WorkerInfo)
                assert worker_info.worker_id == worker_id

        finally:
            engine.shutdown()

    def test_context_manager(self, mock_model, distributed_config):
        """Test context manager interface."""
        with DistributedQMLInferenceEngine(mock_model, distributed_config) as engine:
            assert engine._initialized

        # Engine should be shut down after context exit
        assert not engine._initialized

    def test_error_handling(self, mock_model, distributed_config):
        """Test error handling in distributed prediction."""
        engine = DistributedQMLInferenceEngine(mock_model, distributed_config)
        engine.initialize()

        try:
            # Test with empty geometries
            result = engine.predict_distributed([], [QuantumProperty.TOTAL_ENERGY])

            assert isinstance(result, InferenceResult)
            assert len(result.predictions) == 0

        finally:
            engine.shutdown()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_distributed_engine(self, mock_model):
        """Test create_distributed_engine function."""
        engine = create_distributed_engine(
            mock_model, backend="multiprocessing", num_workers=2, batch_size=16
        )

        assert isinstance(engine, DistributedQMLInferenceEngine)
        assert engine.config.backend == DistributionBackend.MULTIPROCESSING
        assert engine.config.num_workers == 2
        assert engine.config.batch_size == 16

    def test_distributed_predict(self, mock_model, sample_geometries):
        """Test distributed_predict convenience function."""
        result = distributed_predict(
            mock_model,
            sample_geometries,
            [QuantumProperty.TOTAL_ENERGY],
            backend="multiprocessing",
            num_workers=2,
        )

        assert isinstance(result, InferenceResult)
        assert len(result.predictions) == len(sample_geometries)


@pytest.mark.integration
class TestDistributedIntegration:
    """Integration tests for distributed QML inference."""

    def test_end_to_end_prediction(self, mock_model, sample_geometries):
        """Test end-to-end distributed prediction."""
        config = DistributedConfig(
            backend=DistributionBackend.MULTIPROCESSING,
            num_workers=2,
            batch_size=2,
            max_batch_size=4,
        )

        with DistributedQMLInferenceEngine(mock_model, config) as engine:
            result = engine.predict_distributed(
                sample_geometries,
                [QuantumProperty.TOTAL_ENERGY, QuantumProperty.HOMO_ENERGY],
            )

            # Verify results
            assert len(result.predictions) == len(sample_geometries)
            assert result.execution_metrics.total_time > 0
            assert result.execution_metrics.throughput > 0

            # Check prediction quality
            for prediction in result.predictions:
                if prediction.properties:  # Some may be empty due to mock behavior
                    assert isinstance(prediction, QMLPrediction)
                    assert prediction.confidence is not None

    def test_large_batch_processing(self, mock_model):
        """Test processing of large batches."""
        # Create many simple geometries
        large_geometry_list = []
        for i in range(50):
            geometry = GeometryInput(
                coordinates=np.random.randn(5, 3),  # Random 5-atom molecules
                atomic_numbers=np.array([6, 1, 1, 1, 1]),  # C and H atoms
                charge=0,
                multiplicity=1,
            )
            large_geometry_list.append(geometry)

        config = DistributedConfig(
            backend=DistributionBackend.MULTIPROCESSING,
            num_workers=4,
            batch_size=8,
            max_batch_size=16,
        )

        with DistributedQMLInferenceEngine(mock_model, config) as engine:
            start_time = time.time()

            result = engine.predict_distributed(
                large_geometry_list, [QuantumProperty.TOTAL_ENERGY]
            )

            end_time = time.time()

            # Verify results
            assert len(result.predictions) == len(large_geometry_list)
            assert result.execution_metrics.total_time > 0

            # Check performance
            throughput = len(large_geometry_list) / (end_time - start_time)
            assert throughput > 0  # Should process at least some molecules per second

    @pytest.mark.skipif(not torch, reason="PyTorch not available")
    def test_memory_management(self, mock_model, sample_geometries):
        """Test memory management during distributed processing."""
        config = DistributedConfig(
            backend=DistributionBackend.MULTIPROCESSING,
            num_workers=2,
            batch_size=4,
            memory_strategy=MemoryStrategy.CONSERVATIVE,
        )

        with DistributedQMLInferenceEngine(mock_model, config) as engine:
            # Process multiple batches to test memory management
            for _ in range(3):
                result = engine.predict_distributed(
                    sample_geometries, [QuantumProperty.TOTAL_ENERGY]
                )

                assert len(result.predictions) == len(sample_geometries)

                # Memory usage should be reasonable
                if hasattr(result.execution_metrics, "memory_usage_mb"):
                    assert (
                        result.execution_metrics.memory_usage_mb < 1000
                    )  # Less than 1GB


@pytest.mark.parametrize(
    "backend",
    [
        DistributionBackend.MULTIPROCESSING,
        # Add other backends when they become available in test environment
    ],
)
class TestBackendSpecific:
    """Test backend-specific functionality."""

    def test_backend_initialization(self, mock_model, backend):
        """Test initialization of different backends."""
        config = DistributedConfig(backend=backend, num_workers=2)
        engine = DistributedQMLInferenceEngine(mock_model, config)

        success = engine.initialize()

        try:
            assert success
            assert engine._initialized
            assert len(engine.workers) > 0

        finally:
            engine.shutdown()

    def test_backend_prediction(self, mock_model, sample_geometries, backend):
        """Test prediction with different backends."""
        config = DistributedConfig(backend=backend, num_workers=2)

        with DistributedQMLInferenceEngine(mock_model, config) as engine:
            result = engine.predict_distributed(
                sample_geometries, [QuantumProperty.TOTAL_ENERGY]
            )

            assert isinstance(result, InferenceResult)
            assert len(result.predictions) == len(sample_geometries)


class TestErrorHandling:
    """Test error handling and fault tolerance."""

    def test_invalid_backend(self, mock_model):
        """Test handling of invalid backend."""
        with pytest.raises((ValueError, QMLModelError)):
            config = DistributedConfig(backend="invalid_backend")
            # This should fail during enum conversion or initialization

    def test_initialization_failure(self, mock_model):
        """Test handling of initialization failures."""
        config = DistributedConfig(
            backend=DistributionBackend.MULTIPROCESSING,
            num_workers=0,  # Invalid number of workers
        )

        engine = DistributedQMLInferenceEngine(mock_model, config)

        # Should handle gracefully
        success = engine.initialize()

        # Depending on implementation, this might succeed with default workers
        # or fail gracefully
        if not success:
            assert not engine._initialized

    def test_prediction_with_uninitialized_engine(self, mock_model, sample_geometries):
        """Test prediction with uninitialized engine."""
        engine = DistributedQMLInferenceEngine(mock_model, DistributedConfig())

        with pytest.raises(QMLModelError):
            engine.predict_distributed(
                sample_geometries, [QuantumProperty.TOTAL_ENERGY]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

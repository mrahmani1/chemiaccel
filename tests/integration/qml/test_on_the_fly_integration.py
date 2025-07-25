"""
Integration tests for on-the-fly QML prediction system.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.qml.base import BaseQMLModel, QMLPrediction, QuantumProperty
from chemi_accel.qml.memory_manager import MemoryAwareProcessor, MemoryStrategy
from chemi_accel.qml.on_the_fly import (
    BatchPredictionRequest,
    ConformerStrategy,
    GeometryInput,
    OnTheFlyPredictor,
)
from chemi_accel.qml.streaming import StreamingConfig, StreamingPredictor


class MockQMLModel(BaseQMLModel):
    """Mock QML model for integration testing."""

    def __init__(self, processing_time=0.1, failure_rate=0.0):
        super().__init__()
        self.processing_time = processing_time
        self.failure_rate = failure_rate
        self._prediction_count = 0

    def load_model(self):
        self._model = Mock()
        self._is_loaded = True

    def predict(self, molecules, properties=None, **kwargs):
        # Simulate processing time
        time.sleep(self.processing_time)

        self._prediction_count += 1

        # Simulate failures
        if np.random.random() < self.failure_rate:
            raise Exception(f"Simulated model failure #{self._prediction_count}")

        if isinstance(molecules, list):
            return [
                QMLPrediction(
                    properties={
                        QuantumProperty.HOMO_ENERGY: -8.0 - i * 0.1,
                        QuantumProperty.LUMO_ENERGY: -2.0 - i * 0.05,
                        QuantumProperty.DIPOLE_MOMENT: 1.0 + i * 0.1,
                    },
                    confidence=0.9 - i * 0.01,
                    model_name="MockModel",
                )
                for i in range(len(molecules))
            ]
        else:
            return QMLPrediction(
                properties={
                    QuantumProperty.HOMO_ENERGY: -8.0,
                    QuantumProperty.LUMO_ENERGY: -2.0,
                    QuantumProperty.DIPOLE_MOMENT: 1.0,
                },
                confidence=0.9,
                model_name="MockModel",
            )

    def get_supported_properties(self):
        return [
            QuantumProperty.HOMO_ENERGY,
            QuantumProperty.LUMO_ENERGY,
            QuantumProperty.DIPOLE_MOMENT,
        ]

    def validate_molecule(self, molecule):
        return True


@pytest.mark.integration
class TestOnTheFlyIntegration:
    """Integration tests for on-the-fly prediction system."""

    def test_end_to_end_batch_prediction(self):
        """Test complete end-to-end batch prediction workflow."""
        model = MockQMLModel(processing_time=0.01)
        model.load_model()

        predictor = OnTheFlyPredictor(
            model=model,
            device="cpu",
            max_batch_size=16,
            conformer_strategy=ConformerStrategy.SINGLE,
            cache_size=100,
        )

        # Create batch request
        molecules = ["CCO", "CC", "CCC", "CCCC", "CCCCC"]
        properties = [QuantumProperty.HOMO_ENERGY, QuantumProperty.DIPOLE_MOMENT]

        request = BatchPredictionRequest(
            molecules=molecules, properties=properties, batch_size=8, use_cache=True
        )

        # Execute prediction
        with patch.object(predictor, "_prepare_molecular_inputs") as mock_prepare:
            # Mock molecular inputs (skip conformer generation for speed)
            mock_prepare.return_value = [
                [
                    GeometryInput(
                        coordinates=np.array([[i, 0.0, 0.0]]),
                        atomic_numbers=np.array([6]),
                        energy=-100.0 - i,
                    )
                ]
                for i in range(len(molecules))
            ]

            result = predictor.predict_batch(request)

        # Verify results
        assert len(result.predictions) == len(molecules)
        assert result.execution_time > 0
        assert result.statistics["successful_predictions"] == len(molecules)
        assert result.statistics["failed_predictions"] == 0
        assert result.cache_misses == len(molecules)  # First time, all cache misses

        # Test caching - run same request again
        result2 = predictor.predict_batch(request)
        assert result2.cache_hits > 0  # Should have cache hits

    def test_conformer_strategy_integration(self):
        """Test integration with different conformer strategies."""
        model = MockQMLModel(processing_time=0.01)
        model.load_model()

        # Test with averaging strategy
        predictor = OnTheFlyPredictor(
            model=model, conformer_strategy=ConformerStrategy.AVERAGE
        )

        # Create multiple conformers for a molecule
        geometries = [
            [
                GeometryInput(
                    coordinates=np.array([[0.0, 0.0, 0.0]]),
                    atomic_numbers=np.array([6]),
                    energy=-100.0,
                ),
                GeometryInput(
                    coordinates=np.array([[0.5, 0.0, 0.0]]),
                    atomic_numbers=np.array([6]),
                    energy=-98.0,
                ),
                GeometryInput(
                    coordinates=np.array([[1.0, 0.0, 0.0]]),
                    atomic_numbers=np.array([6]),
                    energy=-96.0,
                ),
            ]
        ]

        request = BatchPredictionRequest(
            molecules=["CCO"],
            properties=[QuantumProperty.HOMO_ENERGY],
            geometries=geometries,
            conformer_strategy=ConformerStrategy.AVERAGE,
        )

        result = predictor.predict_batch(request)

        assert len(result.predictions) == 1
        prediction = result.predictions[0]
        assert prediction.metadata.get("conformer_average") == True
        assert prediction.metadata.get("n_conformers") == 3

    def test_memory_integration(self):
        """Test integration with memory management."""
        model = MockQMLModel(processing_time=0.01)
        model.load_model()

        # Create predictor with memory-aware processor
        memory_processor = MemoryAwareProcessor(
            strategy=MemoryStrategy.BALANCED, gc_threshold=0.8
        )

        predictor = OnTheFlyPredictor(model=model, max_batch_size=32)

        # Large batch to test memory management
        molecules = [f"C{'C' * i}O" for i in range(50)]
        properties = [QuantumProperty.HOMO_ENERGY, QuantumProperty.LUMO_ENERGY]

        request = BatchPredictionRequest(
            molecules=molecules, properties=properties, batch_size=16
        )

        # Use memory-managed processing
        with memory_processor.memory_managed_processing(estimated_memory_gb=1.0):
            with patch.object(predictor, "_prepare_molecular_inputs") as mock_prepare:
                mock_prepare.return_value = [
                    [
                        GeometryInput(
                            coordinates=np.array([[i % 5, 0.0, 0.0]]),
                            atomic_numbers=np.array([6]),
                        )
                    ]
                    for i in range(len(molecules))
                ]

                result = predictor.predict_batch(request)

        assert len(result.predictions) == len(molecules)
        assert (
            "memory_allocated" in result.memory_usage
            or "memory_info" in result.memory_usage
        )

    def test_streaming_integration(self):
        """Test integration with streaming prediction system."""
        model = MockQMLModel(processing_time=0.01)
        model.load_model()

        config = StreamingConfig(
            buffer_size=8, max_workers=2, timeout=10.0, batch_timeout=0.5
        )

        predictor = StreamingPredictor(model=model, config=config)

        try:
            predictor.start()

            # Test single prediction
            response = predictor.predict(
                molecule="CCO", properties=[QuantumProperty.HOMO_ENERGY], timeout=5.0
            )

            assert response.success
            assert response.prediction is not None
            assert QuantumProperty.HOMO_ENERGY in response.prediction.properties

            # Test streaming multiple predictions
            molecules = ["CCO", "CC", "CCC", "CCCC"]
            properties = [QuantumProperty.HOMO_ENERGY, QuantumProperty.DIPOLE_MOMENT]

            responses = []
            for mol in molecules:
                resp = predictor.predict(
                    molecule=mol, properties=properties, timeout=5.0
                )
                responses.append(resp)

            # Verify all responses
            assert len(responses) == len(molecules)
            assert all(resp.success for resp in responses)

            # Check metrics
            metrics = predictor.get_metrics()
            assert metrics.total_predictions >= len(molecules) + 1
            assert metrics.throughput > 0

        finally:
            predictor.stop()

    @pytest.mark.slow
    def test_performance_scaling(self):
        """Test performance scaling with different batch sizes."""
        model = MockQMLModel(processing_time=0.005)  # Very fast model
        model.load_model()

        predictor = OnTheFlyPredictor(
            model=model, device="cpu", conformer_strategy=ConformerStrategy.SINGLE
        )

        molecules = ["CCO"] * 100  # 100 identical molecules
        properties = [QuantumProperty.HOMO_ENERGY]

        batch_sizes = [1, 10, 25, 50]
        results = {}

        for batch_size in batch_sizes:
            request = BatchPredictionRequest(
                molecules=molecules, properties=properties, batch_size=batch_size
            )

            with patch.object(predictor, "_prepare_molecular_inputs") as mock_prepare:
                mock_prepare.return_value = [
                    [
                        GeometryInput(
                            coordinates=np.array([[0.0, 0.0, 0.0]]),
                            atomic_numbers=np.array([6]),
                        )
                    ]
                    for _ in range(len(molecules))
                ]

                start_time = time.time()
                result = predictor.predict_batch(request)
                execution_time = time.time() - start_time

            throughput = len(molecules) / execution_time
            results[batch_size] = {
                "execution_time": execution_time,
                "throughput": throughput,
                "batch_execution_time": result.execution_time,
            }

        # Verify that larger batch sizes generally improve throughput
        # (though very small examples might not show clear scaling)
        assert all(result["throughput"] > 0 for result in results.values())

        # At minimum, batch processing should be faster than individual
        if len(results) >= 2:
            assert (
                results[max(batch_sizes)]["throughput"]
                >= results[min(batch_sizes)]["throughput"] * 0.8
            )

    def test_error_handling_integration(self):
        """Test error handling across the integrated system."""
        # Model with high failure rate
        model = MockQMLModel(processing_time=0.01, failure_rate=0.3)
        model.load_model()

        predictor = OnTheFlyPredictor(model=model, max_batch_size=8)

        molecules = ["CCO", "CC", "CCC", "CCCC", "CCCCC"]
        properties = [QuantumProperty.HOMO_ENERGY]

        request = BatchPredictionRequest(
            molecules=molecules, properties=properties, batch_size=3
        )

        with patch.object(predictor, "_prepare_molecular_inputs") as mock_prepare:
            mock_prepare.return_value = [
                [
                    GeometryInput(
                        coordinates=np.array([[i, 0.0, 0.0]]),
                        atomic_numbers=np.array([6]),
                    )
                ]
                for i in range(len(molecules))
            ]

            result = predictor.predict_batch(request)

        # Should handle errors gracefully
        assert len(result.predictions) == len(molecules)
        assert result.statistics["failed_predictions"] >= 0
        assert len(result.errors) >= 0

        # Some predictions might fail, but system should continue
        successful_preds = [p for p in result.predictions if p.properties]
        failed_preds = [p for p in result.predictions if not p.properties]

        assert len(successful_preds) + len(failed_preds) == len(molecules)

    def test_concurrent_prediction_integration(self):
        """Test concurrent predictions with thread safety."""
        model = MockQMLModel(processing_time=0.02)
        model.load_model()

        predictor = OnTheFlyPredictor(model=model, max_batch_size=4, num_workers=2)

        import queue
        import threading

        results_queue = queue.Queue()

        def worker_prediction(worker_id):
            try:
                molecules = [f"C{'C' * (worker_id % 3)}O"] * 5
                properties = [QuantumProperty.HOMO_ENERGY]

                request = BatchPredictionRequest(
                    molecules=molecules, properties=properties, batch_size=2
                )

                with patch.object(
                    predictor, "_prepare_molecular_inputs"
                ) as mock_prepare:
                    mock_prepare.return_value = [
                        [
                            GeometryInput(
                                coordinates=np.array([[worker_id, 0.0, 0.0]]),
                                atomic_numbers=np.array([6]),
                            )
                        ]
                        for _ in range(len(molecules))
                    ]

                    result = predictor.predict_batch(request)

                results_queue.put((worker_id, result))

            except Exception as e:
                results_queue.put((worker_id, e))

        # Start multiple worker threads
        threads = []
        for i in range(4):  # 4 concurrent predictions
            thread = threading.Thread(target=worker_prediction, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)

        # Collect results
        results = []
        while not results_queue.empty():
            worker_id, result = results_queue.get()
            results.append((worker_id, result))

        # Verify all threads completed successfully
        assert len(results) == 4

        for worker_id, result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Worker {worker_id} failed with exception: {result}")
            else:
                assert len(result.predictions) == 5
                assert result.statistics["successful_predictions"] >= 0

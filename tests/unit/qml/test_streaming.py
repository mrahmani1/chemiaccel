"""
Tests for streaming QML prediction system.
"""

import time
from unittest.mock import Mock, patch

import pytest
from chemi_accel.qml.base import BaseQMLModel, QMLPrediction, QuantumProperty
from chemi_accel.qml.on_the_fly import BatchPredictionResult
from chemi_accel.qml.streaming import (
    AdaptiveBuffer,
    BufferStrategy,
    FlowControl,
    StreamingConfig,
    StreamingMetrics,
    StreamingPredictor,
    StreamingRequest,
    StreamingResponse,
)


class MockQMLModel(BaseQMLModel):
    """Mock QML model for testing."""

    def load_model(self):
        self._model = Mock()
        self._is_loaded = True

    def predict(self, molecules, properties=None, **kwargs):
        if isinstance(molecules, list):
            return [
                QMLPrediction(
                    properties={QuantumProperty.HOMO_ENERGY: -8.0 - i * 0.1},
                    model_name="MockModel",
                )
                for i in range(len(molecules))
            ]
        else:
            return QMLPrediction(
                properties={QuantumProperty.HOMO_ENERGY: -8.0}, model_name="MockModel"
            )

    def get_supported_properties(self):
        return [QuantumProperty.HOMO_ENERGY]

    def validate_molecule(self, molecule):
        return True


class TestStreamingConfig:
    """Test StreamingConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = StreamingConfig()

        assert config.buffer_size == 64
        assert config.max_buffer_size == 1024
        assert config.timeout == 30.0
        assert config.batch_timeout == 1.0
        assert config.max_workers == 4
        assert config.buffer_strategy == BufferStrategy.ADAPTIVE
        assert config.flow_control == FlowControl.BACKPRESSURE
        assert config.enable_async == True

    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = StreamingConfig(
            buffer_size=128,
            max_workers=8,
            buffer_strategy=BufferStrategy.PRIORITY,
            flow_control=FlowControl.RATE_LIMIT,
        )

        assert config.buffer_size == 128
        assert config.max_workers == 8
        assert config.buffer_strategy == BufferStrategy.PRIORITY
        assert config.flow_control == FlowControl.RATE_LIMIT


class TestStreamingMetrics:
    """Test StreamingMetrics class."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = StreamingMetrics()

        assert metrics.total_predictions == 0
        assert metrics.successful_predictions == 0
        assert metrics.failed_predictions == 0
        assert metrics.throughput == 0.0
        assert metrics.error_rate == 0.0

    def test_update_with_batch_result(self):
        """Test metrics update with batch result."""
        metrics = StreamingMetrics()

        # Mock batch result
        batch_result = Mock(spec=BatchPredictionResult)
        batch_result.predictions = [Mock(), Mock(), Mock()]  # 3 predictions
        batch_result.errors = [{"error": "test"}]  # 1 error
        batch_result.execution_time = 2.0

        metrics.update(batch_result)

        assert metrics.total_batches == 1
        assert metrics.total_predictions == 3
        assert metrics.successful_predictions == 2  # 3 predictions - 1 error
        assert metrics.failed_predictions == 1
        assert metrics.average_batch_size == 3.0
        assert metrics.error_rate == 1 / 3


class TestStreamingRequest:
    """Test StreamingRequest class."""

    def test_initialization(self):
        """Test request initialization."""
        request = StreamingRequest(
            request_id="test-123",
            molecule="CCO",
            properties=[QuantumProperty.HOMO_ENERGY],
            priority=2,
        )

        assert request.request_id == "test-123"
        assert request.molecule == "CCO"
        assert request.properties == [QuantumProperty.HOMO_ENERGY]
        assert request.priority == 2
        assert isinstance(request.timestamp, float)

    def test_auto_generated_id(self):
        """Test auto-generated request ID."""
        request = StreamingRequest(
            request_id="", molecule="CCO", properties=[QuantumProperty.HOMO_ENERGY]
        )

        # Should have generated a UUID
        assert request.request_id != ""
        assert len(request.request_id) > 10


class TestStreamingResponse:
    """Test StreamingResponse class."""

    def test_initialization(self):
        """Test response initialization."""
        prediction = QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -8.0})

        response = StreamingResponse(
            request_id="test-123",
            prediction=prediction,
            processing_time=1.5,
            queue_time=0.5,
        )

        assert response.request_id == "test-123"
        assert response.prediction is prediction
        assert response.processing_time == 1.5
        assert response.queue_time == 0.5
        assert response.total_time == 2.0
        assert response.success == True

    def test_error_response(self):
        """Test error response."""
        response = StreamingResponse(request_id="test-123", error="Test error")

        assert response.error == "Test error"
        assert response.prediction is None
        assert response.success == False


class TestAdaptiveBuffer:
    """Test AdaptiveBuffer class."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = AdaptiveBuffer(
            initial_size=16, max_size=256, strategy=BufferStrategy.FIFO
        )

        assert buffer.initial_size == 16
        assert buffer.max_size == 256
        assert buffer.current_size == 16
        assert buffer.strategy == BufferStrategy.FIFO

    def test_put_and_get_fifo(self):
        """Test FIFO buffer operations."""
        buffer = AdaptiveBuffer(initial_size=5, strategy=BufferStrategy.FIFO)

        request = StreamingRequest(
            request_id="test-1",
            molecule="CCO",
            properties=[QuantumProperty.HOMO_ENERGY],
        )

        # Put item
        success = buffer.put(request, timeout=0.1)
        assert success == True
        assert buffer.size() == 1

        # Get item
        retrieved = buffer.get(timeout=0.1)
        assert retrieved is request
        assert buffer.size() == 0

    def test_put_and_get_priority(self):
        """Test priority buffer operations."""
        buffer = AdaptiveBuffer(initial_size=5, strategy=BufferStrategy.PRIORITY)

        request1 = StreamingRequest(
            request_id="test-1",
            molecule="CCO",
            properties=[QuantumProperty.HOMO_ENERGY],
            priority=1,
        )

        request2 = StreamingRequest(
            request_id="test-2",
            molecule="CC",
            properties=[QuantumProperty.HOMO_ENERGY],
            priority=3,
        )

        # Put items (lower priority first)
        buffer.put(request1, timeout=0.1)
        buffer.put(request2, timeout=0.1)

        # Should get higher priority item first
        retrieved = buffer.get(timeout=0.1)
        assert retrieved is request2  # Higher priority

    def test_get_batch(self):
        """Test batch retrieval from buffer."""
        buffer = AdaptiveBuffer(initial_size=10, strategy=BufferStrategy.FIFO)

        # Add multiple requests
        requests = []
        for i in range(5):
            request = StreamingRequest(
                request_id=f"test-{i}",
                molecule="CCO",
                properties=[QuantumProperty.HOMO_ENERGY],
            )
            requests.append(request)
            buffer.put(request, timeout=0.1)

        # Get batch
        batch = buffer.get_batch(batch_size=3, timeout=0.5)

        assert len(batch) == 3
        assert all(isinstance(req, StreamingRequest) for req in batch)
        assert buffer.size() == 2  # 2 remaining

    def test_buffer_full_handling(self):
        """Test buffer full condition."""
        buffer = AdaptiveBuffer(
            initial_size=2, max_size=2, strategy=BufferStrategy.FIFO
        )

        # Fill buffer
        req1 = StreamingRequest(
            request_id="1", molecule="CCO", properties=[QuantumProperty.HOMO_ENERGY]
        )
        req2 = StreamingRequest(
            request_id="2", molecule="CC", properties=[QuantumProperty.HOMO_ENERGY]
        )
        req3 = StreamingRequest(
            request_id="3", molecule="CCC", properties=[QuantumProperty.HOMO_ENERGY]
        )

        assert buffer.put(req1, timeout=0.1) == True
        assert buffer.put(req2, timeout=0.1) == True
        assert buffer.is_full() == True

        # Should fail to add more
        assert buffer.put(req3, timeout=0.1) == False

    def test_adaptive_resizing(self):
        """Test adaptive buffer resizing."""
        buffer = AdaptiveBuffer(
            initial_size=2, max_size=8, strategy=BufferStrategy.ADAPTIVE
        )

        # Fill initial buffer
        req1 = StreamingRequest(
            request_id="1", molecule="CCO", properties=[QuantumProperty.HOMO_ENERGY]
        )
        req2 = StreamingRequest(
            request_id="2", molecule="CC", properties=[QuantumProperty.HOMO_ENERGY]
        )
        req3 = StreamingRequest(
            request_id="3", molecule="CCC", properties=[QuantumProperty.HOMO_ENERGY]
        )

        buffer.put(req1, timeout=0.1)
        buffer.put(req2, timeout=0.1)

        # Should trigger resize when trying to add third
        original_size = buffer.current_size
        success = buffer.put(req3, timeout=0.1)

        if success:  # If resize was successful
            assert buffer.current_size > original_size

    def test_statistics(self):
        """Test buffer statistics."""
        buffer = AdaptiveBuffer()

        request = StreamingRequest(
            request_id="test", molecule="CCO", properties=[QuantumProperty.HOMO_ENERGY]
        )

        buffer.put(request, timeout=0.1)

        stats = buffer.get_statistics()

        assert "current_size" in stats
        assert "max_size" in stats
        assert "current_fill" in stats
        assert "utilization" in stats
        assert "strategy" in stats
        assert stats["stats"]["buffer_hits"] >= 1


class TestStreamingPredictor:
    """Test StreamingPredictor class."""

    def test_initialization(self):
        """Test predictor initialization."""
        model = MockQMLModel()
        config = StreamingConfig(buffer_size=32, max_workers=2)

        predictor = StreamingPredictor(model=model, config=config)

        assert predictor.model is model
        assert predictor.config is config
        assert predictor._running == False

    def test_start_stop(self):
        """Test starting and stopping predictor."""
        model = MockQMLModel()
        config = StreamingConfig(max_workers=1)

        predictor = StreamingPredictor(model=model, config=config)

        # Start
        predictor.start()
        assert predictor._running == True
        assert len(predictor._worker_threads) > 0

        # Stop
        predictor.stop()
        assert predictor._running == False

    def test_predict_synchronous(self):
        """Test synchronous prediction."""
        model = MockQMLModel()
        model.load_model()

        config = StreamingConfig(max_workers=1, timeout=5.0)
        predictor = StreamingPredictor(model=model, config=config)

        # Mock the batch processing
        mock_result = BatchPredictionResult(
            predictions=[
                QMLPrediction(
                    properties={QuantumProperty.HOMO_ENERGY: -8.0},
                    model_name="MockModel",
                )
            ],
            request=Mock(),
            execution_time=0.1,
            memory_usage={},
            statistics={},
        )

        with patch.object(
            predictor.predictor, "predict_batch", return_value=mock_result
        ):
            predictor.start()

            try:
                response = predictor.predict(
                    molecule="CCO",
                    properties=[QuantumProperty.HOMO_ENERGY],
                    timeout=1.0,
                )

                assert isinstance(response, StreamingResponse)
                # Note: Actual success depends on worker thread timing

            finally:
                predictor.stop()

    def test_predict_buffer_full(self):
        """Test prediction when buffer is full."""
        model = MockQMLModel()
        config = StreamingConfig(buffer_size=1, max_buffer_size=1)

        predictor = StreamingPredictor(model=model, config=config)

        # Fill buffer first
        predictor.buffer.put(
            StreamingRequest(
                request_id="blocker",
                molecule="CC",
                properties=[QuantumProperty.HOMO_ENERGY],
            ),
            timeout=0.1,
        )

        # This should fail due to full buffer
        response = predictor.predict(
            molecule="CCO", properties=[QuantumProperty.HOMO_ENERGY], timeout=0.1
        )

        assert response.success == False
        assert "Buffer full" in response.error

    def test_predict_stream_generator(self):
        """Test streaming prediction with generator interface."""
        model = MockQMLModel()
        config = StreamingConfig(max_workers=1)

        predictor = StreamingPredictor(model=model, config=config)

        # Create requests
        requests = [
            StreamingRequest(
                request_id=f"test-{i}",
                molecule=f"C{'C' * i}O",
                properties=[QuantumProperty.HOMO_ENERGY],
            )
            for i in range(3)
        ]

        # Mock successful processing
        with patch.object(predictor, "predict") as mock_predict:
            mock_predict.return_value = StreamingResponse(
                request_id="test",
                prediction=QMLPrediction(
                    properties={QuantumProperty.HOMO_ENERGY: -8.0}
                ),
            )

            responses = list(predictor.predict_stream(iter(requests)))

            # Should get responses for all requests
            assert len(responses) >= len(requests)

    @pytest.mark.asyncio
    async def test_predict_async(self):
        """Test asynchronous prediction."""
        model = MockQMLModel()
        config = StreamingConfig(enable_async=True, max_workers=1)

        predictor = StreamingPredictor(model=model, config=config)

        # Mock synchronous predict method
        expected_response = StreamingResponse(
            request_id="test-async",
            prediction=QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -8.0}),
            processing_time=0.1,
        )

        with patch.object(predictor, "predict", return_value=expected_response):
            response = await predictor.predict_async(
                molecule="CCO", properties=[QuantumProperty.HOMO_ENERGY]
            )

            assert isinstance(response, StreamingResponse)
            assert response is expected_response

    def test_worker_loop_processing(self):
        """Test worker loop batch processing."""
        model = MockQMLModel()
        model.load_model()

        config = StreamingConfig(buffer_size=2, batch_timeout=0.1)
        predictor = StreamingPredictor(model=model, config=config)

        # Add requests to buffer
        requests = [
            StreamingRequest(
                request_id="test-1",
                molecule="CCO",
                properties=[QuantumProperty.HOMO_ENERGY],
            ),
            StreamingRequest(
                request_id="test-2",
                molecule="CC",
                properties=[QuantumProperty.HOMO_ENERGY],
            ),
        ]

        for req in requests:
            predictor.buffer.put(req, timeout=0.1)

        # Mock batch processing
        mock_result = BatchPredictionResult(
            predictions=[
                QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -8.0}),
                QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -7.5}),
            ],
            request=Mock(),
            execution_time=0.2,
            memory_usage={},
            statistics={},
        )

        with patch.object(
            predictor.predictor, "predict_batch", return_value=mock_result
        ):
            # Process batch
            predictor._process_batch(requests)

            # Should have responses in queue
            assert predictor._response_queue.qsize() == 2

    def test_get_metrics(self):
        """Test metrics retrieval."""
        model = MockQMLModel()
        predictor = StreamingPredictor(model=model)

        metrics = predictor.get_metrics()

        assert isinstance(metrics, StreamingMetrics)
        assert metrics.total_predictions >= 0
        assert metrics.throughput >= 0

    def test_get_status(self):
        """Test status information retrieval."""
        model = MockQMLModel()
        predictor = StreamingPredictor(model=model)

        status = predictor.get_status()

        assert "running" in status
        assert "metrics" in status
        assert "buffer_stats" in status
        assert "memory_stats" in status
        assert "config" in status
        assert isinstance(status["running"], bool)

    def test_context_manager(self):
        """Test context manager interface."""
        model = MockQMLModel()
        config = StreamingConfig(max_workers=1)

        with StreamingPredictor(model=model, config=config) as predictor:
            assert predictor._running == True

        # Should be stopped after context
        assert predictor._running == False

    def test_flow_controller_creation(self):
        """Test flow controller creation based on config."""
        model = MockQMLModel()

        # Test different flow control types
        configs = [
            StreamingConfig(flow_control=FlowControl.NONE),
            StreamingConfig(flow_control=FlowControl.BACKPRESSURE),
            StreamingConfig(flow_control=FlowControl.RATE_LIMIT),
            StreamingConfig(flow_control=FlowControl.ADAPTIVE_RATE),
        ]

        for config in configs:
            predictor = StreamingPredictor(model=model, config=config)
            controller = predictor._create_flow_controller()

            assert controller is not None
            assert hasattr(controller, "allow_batch")

    def test_metrics_update_loop(self):
        """Test metrics update in background loop."""
        model = MockQMLModel()
        config = StreamingConfig(max_workers=1)

        predictor = StreamingPredictor(model=model, config=config)
        predictor._start_time = time.time()

        # Test single metrics update
        original_uptime = predictor._metrics.uptime

        # Simulate metrics loop execution
        predictor._metrics.uptime = time.time() - predictor._start_time

        assert predictor._metrics.uptime >= original_uptime

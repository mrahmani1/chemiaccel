"""
Unit tests for real-time data streaming module.
"""

import asyncio
import json
import time
from collections import deque
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from src.chemi_accel.core.exceptions import ChemiAccelError
from src.chemi_accel.digital_twin.manufacturing_integration import (
    ProcessConditions,
    ProcessOutput,
)
from src.chemi_accel.digital_twin.realtime_data_streaming import (
    AnomalyDetectionProcessor,
    DataExporter,
    DataFormat,
    DataPacket,
    DataStreamManager,
    PropertyAggregationProcessor,
    QualityOfService,
    RealTimeDataConnector,
    StreamingConfig,
    StreamingMode,
)
from src.chemi_accel.digital_twin.realtime_process_simulation import (
    MaterialProperty,
    MicrostructureState,
    PropertyType,
    SimulationState,
)


class TestStreamingConfig:
    """Test StreamingConfig dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        config = StreamingConfig()

        assert config.mode == StreamingMode.PUSH
        assert config.data_format == DataFormat.JSON
        assert config.qos == QualityOfService.RELIABLE
        assert config.buffer_size == 1000
        assert config.batch_size == 10
        assert config.enable_persistence == True

    def test_custom_initialization(self):
        """Test custom initialization."""
        config = StreamingConfig(
            mode=StreamingMode.PULL,
            data_format=DataFormat.BINARY,
            buffer_size=500,
            batch_size=5,
            enable_persistence=False,
        )

        assert config.mode == StreamingMode.PULL
        assert config.data_format == DataFormat.BINARY
        assert config.buffer_size == 500
        assert config.batch_size == 5
        assert config.enable_persistence == False


class TestDataPacket:
    """Test DataPacket dataclass."""

    def test_initialization(self):
        """Test packet initialization."""
        packet = DataPacket(
            source="test_source",
            data_type="test_data",
            payload={"key": "value"},
            metadata={"info": "test"},
        )

        assert packet.source == "test_source"
        assert packet.data_type == "test_data"
        assert packet.payload == {"key": "value"}
        assert packet.metadata == {"info": "test"}
        assert packet.id is not None
        assert packet.timestamp > 0

    def test_to_dict(self):
        """Test dictionary conversion."""
        packet = DataPacket(source="test", data_type="data", payload={"value": 123})

        packet_dict = packet.to_dict()

        assert "id" in packet_dict
        assert "timestamp" in packet_dict
        assert packet_dict["source"] == "test"
        assert packet_dict["data_type"] == "data"
        assert packet_dict["payload"] == {"value": 123}

    def test_from_dict(self):
        """Test creation from dictionary."""
        packet_dict = {
            "id": "test-id",
            "timestamp": 1234567890.0,
            "source": "test",
            "data_type": "data",
            "payload": {"value": 456},
            "metadata": {},
            "sequence_number": 0,
            "checksum": None,
        }

        packet = DataPacket.from_dict(packet_dict)

        assert packet.id == "test-id"
        assert packet.timestamp == 1234567890.0
        assert packet.source == "test"
        assert packet.payload == {"value": 456}

    def test_serialize_json(self):
        """Test JSON serialization."""
        packet = DataPacket(source="test", data_type="data", payload={"value": 789})

        serialized = packet.serialize(DataFormat.JSON)

        assert isinstance(serialized, bytes)

        # Test deserialization
        deserialized = DataPacket.deserialize(serialized, DataFormat.JSON)
        assert deserialized.source == "test"
        assert deserialized.payload == {"value": 789}

    def test_serialize_binary(self):
        """Test binary serialization."""
        packet = DataPacket(source="test", data_type="data", payload={"value": 999})

        serialized = packet.serialize(DataFormat.BINARY)

        assert isinstance(serialized, bytes)

        # Test deserialization
        deserialized = DataPacket.deserialize(serialized, DataFormat.BINARY)
        assert deserialized.source == "test"
        assert deserialized.payload == {"value": 999}


class TestPropertyAggregationProcessor:
    """Test PropertyAggregationProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create PropertyAggregationProcessor instance."""
        return PropertyAggregationProcessor(window_size=5)

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.window_size == 5
        assert "mean" in processor.aggregation_functions
        assert len(processor.property_windows) == 0

    @pytest.mark.asyncio
    async def test_process_simulation_state(self, processor):
        """Test processing simulation state packet."""
        packet = DataPacket(
            data_type="simulation_state",
            payload={
                "material_properties": {
                    "tensile_strength": {"value": 100.0},
                    "elastic_modulus": {"value": 2000.0},
                }
            },
        )

        result = await processor.process(packet)

        # First call should not return aggregation (not enough data)
        assert result is None

        # Add more data points
        for i in range(4):
            packet.payload["material_properties"]["tensile_strength"]["value"] = (
                100.0 + i
            )
            await processor.process(packet)

        # Now should have aggregation
        result = await processor.process(packet)
        assert result is not None
        assert result.data_type == "aggregated_properties"
        assert "aggregated_properties" in result.payload

    @pytest.mark.asyncio
    async def test_process_non_simulation_state(self, processor):
        """Test processing non-simulation state packet."""
        packet = DataPacket(data_type="other_data", payload={"some": "data"})

        result = await processor.process(packet)
        assert result is None

    def test_get_processor_info(self, processor):
        """Test getting processor information."""
        info = processor.get_processor_info()

        assert info["type"] == "PropertyAggregationProcessor"
        assert info["window_size"] == 5
        assert "aggregation_functions" in info


class TestAnomalyDetectionProcessor:
    """Test AnomalyDetectionProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create AnomalyDetectionProcessor instance."""
        return AnomalyDetectionProcessor(threshold_multiplier=2.0, min_samples=5)

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.threshold_multiplier == 2.0
        assert processor.min_samples == 5
        assert len(processor.property_baselines) == 0

    @pytest.mark.asyncio
    async def test_process_normal_values(self, processor):
        """Test processing normal values (no anomalies)."""
        # Add normal values
        for i in range(10):
            packet = DataPacket(
                data_type="simulation_state",
                payload={
                    "material_properties": {
                        "tensile_strength": {"value": 100.0 + np.random.normal(0, 1)}
                    }
                },
            )
            result = await processor.process(packet)

        # Should establish baseline but no anomalies
        assert "tensile_strength" in processor.property_baselines
        # Last result should be None (no anomalies)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_anomalous_values(self, processor):
        """Test processing anomalous values."""
        # Add normal values to establish baseline
        for i in range(10):
            packet = DataPacket(
                data_type="simulation_state",
                payload={"material_properties": {"tensile_strength": {"value": 100.0}}},
            )
            await processor.process(packet)

        # Add anomalous value
        anomaly_packet = DataPacket(
            data_type="simulation_state",
            payload={
                "material_properties": {
                    "tensile_strength": {"value": 200.0}  # Much higher than baseline
                }
            },
        )

        result = await processor.process(anomaly_packet)

        assert result is not None
        assert result.data_type == "anomalies"
        assert "anomalies" in result.payload
        assert "tensile_strength" in result.payload["anomalies"]

    def test_get_processor_info(self, processor):
        """Test getting processor information."""
        info = processor.get_processor_info()

        assert info["type"] == "AnomalyDetectionProcessor"
        assert info["threshold_multiplier"] == 2.0
        assert info["min_samples"] == 5


class TestDataStreamManager:
    """Test DataStreamManager class."""

    @pytest.fixture
    def config(self):
        """Create streaming configuration."""
        return StreamingConfig(
            buffer_size=100,
            batch_size=5,
            enable_persistence=False,  # Disable for testing
        )

    @pytest.fixture
    def stream_manager(self, config):
        """Create DataStreamManager instance."""
        return DataStreamManager(config)

    def test_initialization(self, stream_manager, config):
        """Test manager initialization."""
        assert stream_manager.config == config
        assert len(stream_manager.streams) == 0
        assert len(stream_manager.processors) == 0
        assert not stream_manager.is_running

    def test_add_remove_processor(self, stream_manager):
        """Test adding and removing processors."""
        processor = PropertyAggregationProcessor()

        stream_manager.add_processor(processor)
        assert processor in stream_manager.processors

        stream_manager.remove_processor(processor)
        assert processor not in stream_manager.processors

    def test_subscribe_unsubscribe(self, stream_manager):
        """Test subscribing and unsubscribing to data types."""
        callback_calls = []

        def test_callback(packet):
            callback_calls.append(packet)

        # Subscribe
        stream_manager.subscribe("test_data", test_callback)
        assert test_callback in stream_manager.subscribers["test_data"]

        # Unsubscribe
        stream_manager.unsubscribe("test_data", test_callback)
        assert test_callback not in stream_manager.subscribers["test_data"]

    def test_start_stop_streaming(self, stream_manager):
        """Test starting and stopping streaming."""
        stream_manager.start_streaming()
        assert stream_manager.is_running == True
        assert stream_manager.processing_thread is not None

        stream_manager.stop_streaming()
        assert stream_manager.is_running == False

    def test_ingest_simulation_state(self, stream_manager):
        """Test ingesting simulation state."""
        state = SimulationState(
            time=1.0,
            process_conditions=ProcessConditions(
                temperature=298.15, pressure=1e5, material_composition={"polymer": 1.0}
            ),
            material_properties={
                "tensile_strength": MaterialProperty(
                    "tensile_strength", PropertyType.MECHANICAL, 100.0, "MPa"
                )
            },
            microstructure=MicrostructureState(
                grain_size=5.0,
                grain_orientation=np.array([1.0, 0.0, 0.0]),
                phase_fractions={"alpha": 1.0},
                defect_density=0.001,
                crystallinity=0.8,
            ),
        )

        initial_queue_size = stream_manager.processing_queue.qsize()
        stream_manager.ingest_simulation_state(state)

        assert stream_manager.processing_queue.qsize() == initial_queue_size + 1

    def test_ingest_process_output(self, stream_manager):
        """Test ingesting process output."""
        output = ProcessOutput(
            part_quality=0.85, dimensional_accuracy=0.9, surface_finish=0.8
        )

        initial_queue_size = stream_manager.processing_queue.qsize()
        stream_manager.ingest_process_output(output, time.time())

        assert stream_manager.processing_queue.qsize() == initial_queue_size + 1

    def test_get_metrics(self, stream_manager):
        """Test getting metrics."""
        metrics = stream_manager.get_metrics()

        assert "packets_processed" in metrics
        assert "packets_dropped" in metrics
        assert "processing_time_avg" in metrics
        assert "buffer_utilization" in metrics
        assert "error_count" in metrics

    def test_get_stream_info(self, stream_manager):
        """Test getting stream information."""
        info = stream_manager.get_stream_info()

        assert "total_streams" in info
        assert "active_processors" in info
        assert "subscriber_count" in info
        assert "buffer_size" in info
        assert "is_running" in info


class TestRealTimeDataConnector:
    """Test RealTimeDataConnector class."""

    @pytest.fixture
    def mock_simulation_core(self):
        """Create mock simulation core."""
        mock_core = Mock()
        mock_core.add_update_callback = Mock()
        return mock_core

    @pytest.fixture
    def mock_stream_manager(self):
        """Create mock stream manager."""
        mock_manager = Mock(spec=DataStreamManager)
        mock_manager.add_processor = Mock()
        mock_manager.start_streaming = Mock()
        mock_manager.stop_streaming = Mock()
        return mock_manager

    @pytest.fixture
    def connector(self, mock_simulation_core, mock_stream_manager):
        """Create RealTimeDataConnector instance."""
        return RealTimeDataConnector(mock_simulation_core, mock_stream_manager)

    def test_initialization(self, connector, mock_simulation_core, mock_stream_manager):
        """Test connector initialization."""
        assert connector.simulation_core == mock_simulation_core
        assert connector.stream_manager == mock_stream_manager

        # Check that callback was registered
        mock_simulation_core.add_update_callback.assert_called_once()

    def test_on_simulation_update(self, connector):
        """Test simulation update handling."""
        state = SimulationState(
            time=2.0,
            process_conditions=ProcessConditions(
                temperature=350.0, pressure=2e5, material_composition={"polymer": 1.0}
            ),
            material_properties={},
            microstructure=MicrostructureState(
                grain_size=8.0,
                grain_orientation=np.array([0.0, 1.0, 0.0]),
                phase_fractions={"beta": 1.0},
                defect_density=0.005,
                crystallinity=0.7,
            ),
        )

        # This should not raise an exception
        connector._on_simulation_update(state)

        # Check that stream manager's ingest method was called
        connector.stream_manager.ingest_simulation_state.assert_called_once_with(state)

    def test_setup_default_processing_pipeline(self, connector):
        """Test setting up default processing pipeline."""
        connector.setup_default_processing_pipeline()

        # Should have added processors
        assert connector.stream_manager.add_processor.call_count >= 2

    def test_start_stop_streaming(self, connector):
        """Test starting and stopping streaming."""
        connector.start_streaming()
        connector.stream_manager.start_streaming.assert_called_once()

        connector.stop_streaming()
        connector.stream_manager.stop_streaming.assert_called_once()


class TestDataExporter:
    """Test DataExporter class."""

    @pytest.fixture
    def mock_stream_manager(self):
        """Create mock stream manager."""
        mock_manager = Mock(spec=DataStreamManager)
        mock_manager.subscribe = Mock()
        return mock_manager

    @pytest.fixture
    def exporter(self, mock_stream_manager):
        """Create DataExporter instance."""
        return DataExporter(mock_stream_manager)

    def test_initialization(self, exporter, mock_stream_manager):
        """Test exporter initialization."""
        assert exporter.stream_manager == mock_stream_manager
        assert len(exporter.export_targets) == 0

    def test_add_export_target(self, exporter):
        """Test adding export target."""

        def dummy_export(packet):
            pass

        exporter.add_export_target("test_target", dummy_export)

        assert "test_target" in exporter.export_targets
        assert exporter.export_targets["test_target"] == dummy_export

        # Should have subscribed to all data types
        exporter.stream_manager.subscribe.assert_called_once_with(
            "*", unittest.mock.ANY
        )

    def test_export_to_target(self, exporter):
        """Test exporting to specific target."""
        export_calls = []

        def test_export(packet):
            export_calls.append(packet)

        exporter.export_targets["test"] = test_export

        packet = DataPacket(data_type="test_data")
        exporter._export_to_target("test", packet)

        assert len(export_calls) == 1
        assert export_calls[0] == packet

    def test_export_example_functions(self, exporter):
        """Test example export functions."""
        packet = DataPacket(data_type="test_data")

        # These should not raise exceptions
        exporter.export_to_database(packet)
        exporter.export_to_file(packet)
        exporter.export_to_api(packet)


class TestStreamingIntegration:
    """Integration tests for streaming components."""

    @pytest.fixture
    def full_streaming_setup(self):
        """Create full streaming setup."""
        config = StreamingConfig(buffer_size=50, batch_size=3, enable_persistence=False)

        stream_manager = DataStreamManager(config)

        # Add processors
        agg_processor = PropertyAggregationProcessor(window_size=3)
        anomaly_processor = AnomalyDetectionProcessor(min_samples=3)

        stream_manager.add_processor(agg_processor)
        stream_manager.add_processor(anomaly_processor)

        return stream_manager

    def test_end_to_end_processing(self, full_streaming_setup):
        """Test end-to-end data processing."""
        stream_manager = full_streaming_setup

        # Set up callback to capture results
        processed_packets = []

        def capture_callback(packet):
            processed_packets.append(packet)

        stream_manager.subscribe("aggregated_properties", capture_callback)
        stream_manager.subscribe("anomalies", capture_callback)

        # Start streaming
        stream_manager.start_streaming()

        try:
            # Generate some test data
            for i in range(10):
                state = SimulationState(
                    time=i * 0.1,
                    process_conditions=ProcessConditions(
                        temperature=298.15 + i,
                        pressure=1e5,
                        material_composition={"polymer": 1.0},
                    ),
                    material_properties={
                        "tensile_strength": MaterialProperty(
                            "tensile_strength",
                            PropertyType.MECHANICAL,
                            100.0 + i,
                            "MPa",
                        )
                    },
                    microstructure=MicrostructureState(
                        grain_size=5.0,
                        grain_orientation=np.array([1.0, 0.0, 0.0]),
                        phase_fractions={"alpha": 1.0},
                        defect_density=0.001,
                        crystallinity=0.8,
                    ),
                )

                stream_manager.ingest_simulation_state(state)

            # Wait a bit for processing
            time.sleep(0.5)

            # Should have processed some packets
            assert stream_manager.metrics["packets_processed"] > 0

        finally:
            stream_manager.stop_streaming()


if __name__ == "__main__":
    pytest.main([__file__])

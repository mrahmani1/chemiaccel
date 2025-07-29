"""
Unit tests for sensor data integration module.
"""

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from src.chemi_accel.core.exceptions import ChemiAccelError
from src.chemi_accel.digital_twin.realtime_data_streaming import StreamingConfig
from src.chemi_accel.digital_twin.sensor_data_integration import (
    DataQuality,
    MockSensorInterface,
    SensorConfiguration,
    SensorDataBatch,
    SensorDataBuffer,
    SensorDataManager,
    SensorDataValidator,
    SensorProtocol,
    SensorReading,
    SensorType,
)


class TestSensorConfiguration:
    """Test SensorConfiguration dataclass."""

    def test_initialization(self):
        """Test sensor configuration initialization."""
        config = SensorConfiguration(
            sensor_id="temp_01",
            sensor_type=SensorType.TEMPERATURE,
            units="°C",
            sampling_rate=10.0,
            protocol=SensorProtocol.MODBUS,
            connection_params={"host": "192.168.1.100", "port": 502},
        )

        assert config.sensor_id == "temp_01"
        assert config.sensor_type == SensorType.TEMPERATURE
        assert config.units == "°C"
        assert config.sampling_rate == 10.0
        assert config.protocol == SensorProtocol.MODBUS
        assert config.enable_compression == True
        assert config.enable_filtering == True

    def test_default_quality_thresholds(self):
        """Test default quality thresholds initialization."""
        config = SensorConfiguration(
            sensor_id="pressure_01",
            sensor_type=SensorType.PRESSURE,
            units="bar",
            sampling_rate=5.0,
            protocol=SensorProtocol.OPC_UA,
            connection_params={},
        )

        assert "min_value" in config.quality_thresholds
        assert "max_value" in config.quality_thresholds
        assert "max_rate_change" in config.quality_thresholds
        assert "noise_threshold" in config.quality_thresholds


class TestSensorReading:
    """Test SensorReading dataclass."""

    def test_initialization(self):
        """Test sensor reading initialization."""
        reading = SensorReading(
            sensor_id="temp_01",
            timestamp=time.time(),
            value=25.5,
            units="°C",
            quality=DataQuality.EXCELLENT,
            uncertainty=0.1,
        )

        assert reading.sensor_id == "temp_01"
        assert reading.value == 25.5
        assert reading.units == "°C"
        assert reading.quality == DataQuality.EXCELLENT
        assert reading.uncertainty == 0.1

    def test_to_dict(self):
        """Test dictionary conversion."""
        reading = SensorReading(
            sensor_id="pressure_01", timestamp=1234567890.0, value=1.013, units="bar"
        )

        reading_dict = reading.to_dict()

        assert reading_dict["sensor_id"] == "pressure_01"
        assert reading_dict["timestamp"] == 1234567890.0
        assert reading_dict["value"] == 1.013
        assert reading_dict["units"] == "bar"
        assert reading_dict["quality"] == "good"

    def test_from_dict(self):
        """Test creation from dictionary."""
        reading_dict = {
            "sensor_id": "flow_01",
            "timestamp": 1234567890.0,
            "value": 50.0,
            "units": "L/min",
            "quality": "excellent",
            "uncertainty": 0.5,
            "metadata": {"source": "test"},
        }

        reading = SensorReading.from_dict(reading_dict)

        assert reading.sensor_id == "flow_01"
        assert reading.value == 50.0
        assert reading.quality == DataQuality.EXCELLENT
        assert reading.uncertainty == 0.5

    def test_array_value_handling(self):
        """Test handling of array values."""
        array_value = np.array([1.0, 2.0, 3.0])
        reading = SensorReading(
            sensor_id="multi_sensor",
            timestamp=time.time(),
            value=array_value,
            units="various",
        )

        reading_dict = reading.to_dict()
        assert reading_dict["value"] == [1.0, 2.0, 3.0]

        reconstructed = SensorReading.from_dict(reading_dict)
        assert np.array_equal(reconstructed.value, array_value)


class TestSensorDataBatch:
    """Test SensorDataBatch dataclass."""

    def test_initialization(self):
        """Test batch initialization."""
        batch = SensorDataBatch([])

        assert len(batch.readings) == 0
        assert isinstance(batch.batch_timestamp, float)
        assert isinstance(batch.quality_metrics, dict)

    def test_add_reading(self):
        """Test adding readings to batch."""
        batch = SensorDataBatch([])

        reading1 = SensorReading(
            "sensor1", time.time(), 10.0, "units", DataQuality.EXCELLENT
        )
        reading2 = SensorReading(
            "sensor2", time.time(), 20.0, "units", DataQuality.GOOD
        )

        batch.add_reading(reading1)
        batch.add_reading(reading2)

        assert len(batch.readings) == 2
        assert batch.quality_metrics["total_readings"] == 2
        assert batch.quality_metrics["excellent_ratio"] == 0.5
        assert batch.quality_metrics["good_ratio"] == 0.5


class TestMockSensorInterface:
    """Test MockSensorInterface class."""

    @pytest.fixture
    def sensor_configs(self):
        """Create test sensor configurations."""
        return [
            SensorConfiguration(
                sensor_id="temp_01",
                sensor_type=SensorType.TEMPERATURE,
                units="°C",
                sampling_rate=10.0,
                protocol=SensorProtocol.MODBUS,
                connection_params={},
            ),
            SensorConfiguration(
                sensor_id="pressure_01",
                sensor_type=SensorType.PRESSURE,
                units="bar",
                sampling_rate=5.0,
                protocol=SensorProtocol.OPC_UA,
                connection_params={},
            ),
        ]

    @pytest.fixture
    def mock_interface(self, sensor_configs):
        """Create MockSensorInterface instance."""
        return MockSensorInterface(sensor_configs)

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, mock_interface):
        """Test sensor connection lifecycle."""
        assert not mock_interface.is_connected()

        # Connect
        result = await mock_interface.connect()
        assert result == True
        assert mock_interface.is_connected()

        # Disconnect
        await mock_interface.disconnect()
        assert not mock_interface.is_connected()

    @pytest.mark.asyncio
    async def test_sensor_reading(self, mock_interface):
        """Test sensor data reading."""
        await mock_interface.connect()

        reading = await mock_interface.read_sensor("temp_01")

        assert reading.sensor_id == "temp_01"
        assert reading.units == "°C"
        assert isinstance(reading.value, float)
        assert reading.quality in [
            DataQuality.EXCELLENT,
            DataQuality.GOOD,
            DataQuality.POOR,
        ]
        assert reading.timestamp > 0

    @pytest.mark.asyncio
    async def test_unknown_sensor_error(self, mock_interface):
        """Test error for unknown sensor."""
        await mock_interface.connect()

        with pytest.raises(ChemiAccelError):
            await mock_interface.read_sensor("unknown_sensor")

    @pytest.mark.asyncio
    async def test_disconnected_read_error(self, mock_interface):
        """Test error when reading from disconnected interface."""
        with pytest.raises(ChemiAccelError):
            await mock_interface.read_sensor("temp_01")

    def test_realistic_sensor_values(self, mock_interface):
        """Test that generated sensor values are realistic."""
        # Test temperature values
        elapsed = 3600  # 1 hour
        temp_value = mock_interface._generate_base_value(
            SensorType.TEMPERATURE, elapsed
        )
        assert -50 <= temp_value <= 200  # Reasonable temperature range

        # Test pressure values
        pressure_value = mock_interface._generate_base_value(
            SensorType.PRESSURE, elapsed
        )
        assert 0.5 <= pressure_value <= 2.0  # Reasonable pressure range

        # Test flow rate values
        flow_value = mock_interface._generate_base_value(SensorType.FLOW_RATE, elapsed)
        assert 0 <= flow_value <= 100  # Reasonable flow range


class TestSensorDataValidator:
    """Test SensorDataValidator class."""

    @pytest.fixture
    def validator(self):
        """Create SensorDataValidator instance."""
        return SensorDataValidator(window_size=5)

    @pytest.fixture
    def sensor_config(self):
        """Create test sensor configuration."""
        return SensorConfiguration(
            sensor_id="test_sensor",
            sensor_type=SensorType.TEMPERATURE,
            units="°C",
            sampling_rate=10.0,
            protocol=SensorProtocol.MODBUS,
            connection_params={},
            quality_thresholds={
                "min_value": 0.0,
                "max_value": 100.0,
                "max_rate_change": 10.0,
                "noise_threshold": 2.0,
            },
        )

    def test_valid_reading(self, validator, sensor_config):
        """Test validation of valid reading."""
        reading = SensorReading(
            sensor_id="test_sensor",
            timestamp=time.time(),
            value=25.0,
            units="°C",
            quality=DataQuality.GOOD,
        )

        validated = validator.validate_reading(reading, sensor_config)

        assert validated.sensor_id == "test_sensor"
        assert validated.quality in [DataQuality.EXCELLENT, DataQuality.GOOD]
        assert "validation_score" in validated.metadata

    def test_out_of_range_reading(self, validator, sensor_config):
        """Test validation of out-of-range reading."""
        reading = SensorReading(
            sensor_id="test_sensor",
            timestamp=time.time(),
            value=150.0,  # Above max_value
            units="°C",
            quality=DataQuality.GOOD,
        )

        validated = validator.validate_reading(reading, sensor_config)

        assert validated.quality in [DataQuality.POOR, DataQuality.INVALID]
        assert "out_of_range" in validated.metadata["validation_flags"]

    def test_high_rate_change(self, validator, sensor_config):
        """Test validation with high rate of change."""
        # Add first reading
        reading1 = SensorReading("test_sensor", time.time(), 20.0, "°C")
        validator.validate_reading(reading1, sensor_config)

        # Add second reading with large change
        reading2 = SensorReading("test_sensor", time.time(), 50.0, "°C")  # 30°C change
        validated = validator.validate_reading(reading2, sensor_config)

        assert "high_rate_change" in validated.metadata["validation_flags"]
        assert validated.quality in [DataQuality.ACCEPTABLE, DataQuality.POOR]

    def test_sensor_statistics(self, validator, sensor_config):
        """Test sensor statistics calculation."""
        # Add multiple readings
        values = [20.0, 21.0, 22.0, 23.0, 24.0]
        for value in values:
            reading = SensorReading("test_sensor", time.time(), value, "°C")
            validator.validate_reading(reading, sensor_config)

        stats = validator.get_sensor_statistics("test_sensor")

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "trend" in stats
        assert "stability" in stats

        assert abs(stats["mean"] - 22.0) < 0.1
        assert stats["min"] == 20.0
        assert stats["max"] == 24.0


class TestSensorDataBuffer:
    """Test SensorDataBuffer class."""

    @pytest.fixture
    def buffer(self):
        """Create SensorDataBuffer instance."""
        return SensorDataBuffer(max_size=5, flush_interval=0.1)

    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.max_size == 5
        assert buffer.flush_interval == 0.1
        assert len(buffer.buffer) == 0
        assert len(buffer.flush_callbacks) == 0

    def test_add_reading(self, buffer):
        """Test adding readings to buffer."""
        reading = SensorReading("test_sensor", time.time(), 25.0, "°C")
        buffer.add_reading(reading)

        assert len(buffer.buffer) == 1

    def test_size_based_flush(self, buffer):
        """Test buffer flush based on size."""
        flushed_batches = []

        def capture_flush(batch):
            flushed_batches.append(batch)

        buffer.add_flush_callback(capture_flush)

        # Add readings up to max size
        for i in range(6):  # One more than max_size
            reading = SensorReading(f"sensor_{i}", time.time(), float(i), "units")
            buffer.add_reading(reading)

        # Should have flushed at least once
        assert len(flushed_batches) > 0
        assert len(buffer.buffer) < buffer.max_size

    def test_time_based_flush(self, buffer):
        """Test buffer flush based on time interval."""
        flushed_batches = []

        def capture_flush(batch):
            flushed_batches.append(batch)

        buffer.add_flush_callback(capture_flush)

        # Add one reading
        reading = SensorReading("test_sensor", time.time(), 25.0, "°C")
        buffer.add_reading(reading)

        # Wait for flush interval
        time.sleep(0.15)

        # Add another reading to trigger flush check
        reading2 = SensorReading("test_sensor", time.time(), 26.0, "°C")
        buffer.add_reading(reading2)

        # Should have flushed due to time
        assert len(flushed_batches) > 0

    def test_force_flush(self, buffer):
        """Test forced buffer flush."""
        reading = SensorReading("test_sensor", time.time(), 25.0, "°C")
        buffer.add_reading(reading)

        batch = buffer.force_flush()

        assert batch is not None
        assert len(batch.readings) == 1
        assert len(buffer.buffer) == 0

    def test_buffer_info(self, buffer):
        """Test buffer information retrieval."""
        info = buffer.get_buffer_info()

        assert "current_size" in info
        assert "max_size" in info
        assert "utilization" in info
        assert "time_since_last_flush" in info
        assert "flush_interval" in info

        assert info["max_size"] == 5
        assert 0 <= info["utilization"] <= 1


class TestSensorDataManager:
    """Test SensorDataManager class."""

    @pytest.fixture
    def streaming_config(self):
        """Create streaming configuration."""
        return StreamingConfig(
            buffer_size=10, flush_interval=0.1, enable_persistence=False
        )

    @pytest.fixture
    def sensor_manager(self, streaming_config):
        """Create SensorDataManager instance."""
        return SensorDataManager(streaming_config)

    @pytest.fixture
    def mock_sensor_configs(self):
        """Create mock sensor configurations."""
        return [
            SensorConfiguration(
                sensor_id="temp_01",
                sensor_type=SensorType.TEMPERATURE,
                units="°C",
                sampling_rate=2.0,  # Low rate for testing
                protocol=SensorProtocol.MODBUS,
                connection_params={},
            ),
            SensorConfiguration(
                sensor_id="pressure_01",
                sensor_type=SensorType.PRESSURE,
                units="bar",
                sampling_rate=1.0,  # Low rate for testing
                protocol=SensorProtocol.OPC_UA,
                connection_params={},
            ),
        ]

    def test_initialization(self, sensor_manager, streaming_config):
        """Test manager initialization."""
        assert sensor_manager.config == streaming_config
        assert len(sensor_manager.sensor_configs) == 0
        assert len(sensor_manager.sensor_interfaces) == 0
        assert not sensor_manager.is_running

    def test_add_remove_sensor(self, sensor_manager, mock_sensor_configs):
        """Test adding and removing sensors."""
        config = mock_sensor_configs[0]
        interface = MockSensorInterface([config])

        # Add sensor
        sensor_manager.add_sensor(config, interface)
        assert config.sensor_id in sensor_manager.sensor_configs
        assert config.sensor_id in sensor_manager.sensor_interfaces

        # Remove sensor
        sensor_manager.remove_sensor(config.sensor_id)
        assert config.sensor_id not in sensor_manager.sensor_configs
        assert config.sensor_id not in sensor_manager.sensor_interfaces

    @pytest.mark.asyncio
    async def test_collection_lifecycle(self, sensor_manager, mock_sensor_configs):
        """Test sensor data collection lifecycle."""
        # Add sensors
        for config in mock_sensor_configs:
            interface = MockSensorInterface([config])
            sensor_manager.add_sensor(config, interface)

        # Start collection
        await sensor_manager.start_collection()
        assert sensor_manager.is_running

        # Let it run for a short time
        await asyncio.sleep(0.5)

        # Stop collection
        await sensor_manager.stop_collection()
        assert not sensor_manager.is_running

    @pytest.mark.asyncio
    async def test_data_collection(self, sensor_manager, mock_sensor_configs):
        """Test actual data collection."""
        collected_batches = []

        def capture_data(batch):
            collected_batches.append(batch)

        sensor_manager.add_data_callback(capture_data)

        # Add sensors
        for config in mock_sensor_configs:
            interface = MockSensorInterface([config])
            sensor_manager.add_sensor(config, interface)

        # Start collection
        await sensor_manager.start_collection()

        # Wait for some data
        await asyncio.sleep(1.0)

        # Stop collection
        await sensor_manager.stop_collection()

        # Should have collected some data
        assert len(collected_batches) > 0
        assert sensor_manager.metrics["readings_collected"] > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, sensor_manager):
        """Test error handling in data collection."""
        error_messages = []

        def capture_error(message):
            error_messages.append(message)

        sensor_manager.add_error_callback(capture_error)

        # Add sensor with bad configuration
        bad_config = SensorConfiguration(
            sensor_id="bad_sensor",
            sensor_type=SensorType.TEMPERATURE,
            units="°C",
            sampling_rate=10.0,
            protocol=SensorProtocol.MODBUS,
            connection_params={},
        )

        # Create interface that will fail
        bad_interface = Mock()
        bad_interface.connect = AsyncMock(return_value=True)
        bad_interface.read_sensor = AsyncMock(side_effect=Exception("Sensor error"))
        bad_interface.is_connected = Mock(return_value=True)
        bad_interface.disconnect = AsyncMock()

        sensor_manager.add_sensor(bad_config, bad_interface)

        # Start collection
        await sensor_manager.start_collection()

        # Wait for errors
        await asyncio.sleep(0.5)

        # Stop collection
        await sensor_manager.stop_collection()

        # Should have captured errors
        assert len(error_messages) > 0
        assert sensor_manager.metrics["collection_errors"] > 0

    @pytest.mark.asyncio
    async def test_latest_readings(self, sensor_manager, mock_sensor_configs):
        """Test getting latest sensor readings."""
        # Add sensors
        for config in mock_sensor_configs:
            interface = MockSensorInterface([config])
            sensor_manager.add_sensor(config, interface)

        # Start collection to connect sensors
        await sensor_manager.start_collection()
        await asyncio.sleep(0.1)  # Brief pause for connection

        # Get latest readings
        readings = await sensor_manager.get_latest_readings()

        await sensor_manager.stop_collection()

        # Should have readings for connected sensors
        assert len(readings) <= len(mock_sensor_configs)
        for sensor_id, reading in readings.items():
            assert isinstance(reading, SensorReading)
            assert reading.sensor_id == sensor_id

    def test_metrics_tracking(self, sensor_manager):
        """Test metrics tracking."""
        metrics = sensor_manager.get_metrics()

        assert "readings_collected" in metrics
        assert "readings_processed" in metrics
        assert "data_quality_average" in metrics
        assert "collection_errors" in metrics
        assert "buffer_info" in metrics
        assert "active_sensors" in metrics
        assert "total_sensors" in metrics
        assert "is_running" in metrics

    def test_sensor_status(self, sensor_manager, mock_sensor_configs):
        """Test sensor status reporting."""
        # Add sensors
        for config in mock_sensor_configs:
            interface = MockSensorInterface([config])
            sensor_manager.add_sensor(config, interface)

        status = sensor_manager.get_sensor_status()

        assert len(status) == len(mock_sensor_configs)
        for sensor_id in status:
            assert "connected" in status[sensor_id]
            assert "statistics" in status[sensor_id]
            assert "config" in status[sensor_id]


if __name__ == "__main__":
    pytest.main([__file__])

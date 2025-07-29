"""
Sensor Data Integration Module for Real-time Manufacturing Digital Twin

This module provides comprehensive sensor data integration capabilities including:
- Real-time sensor data streaming from manufacturing equipment
- Robust data validation and quality assessment
- Sensor fusion and multi-modal data processing
- Real-time data buffering and synchronization
- Equipment status monitoring and diagnostics

Key Features:
- Multiple sensor protocol support (OPC-UA, Modbus, MQTT, etc.)
- Adaptive sampling rates and data compression
- Fault detection and sensor diagnostics
- Real-time data quality metrics
- Historical data archiving with efficient storage
"""

import asyncio
import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# ChemiAccel imports
from ..core.exceptions import ChemiAccelError
from ..manufacturing_integration import ProcessConditions
from .realtime_data_streaming import DataPacket, StreamingConfig


class SensorType(Enum):
    """Types of manufacturing sensors."""

    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW_RATE = "flow_rate"
    VIBRATION = "vibration"
    FORCE = "force"
    DISPLACEMENT = "displacement"
    STRAIN = "strain"
    ELECTRICAL = "electrical"
    OPTICAL = "optical"
    CHEMICAL = "chemical"
    ACOUSTIC = "acoustic"
    IMAGING = "imaging"


class DataQuality(Enum):
    """Data quality levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


class SensorProtocol(Enum):
    """Communication protocols for sensors."""

    OPC_UA = "opc_ua"
    MODBUS = "modbus"
    MQTT = "mqtt"
    HTTP_REST = "http_rest"
    SERIAL = "serial"
    ETHERNET_IP = "ethernet_ip"
    TCP_IP = "tcp_ip"
    UDP = "udp"


@dataclass
class SensorConfiguration:
    """Configuration for individual sensors."""

    sensor_id: str
    sensor_type: SensorType
    units: str
    sampling_rate: float  # Hz
    protocol: SensorProtocol
    connection_params: Dict[str, Any]
    calibration_data: Optional[Dict[str, float]] = None
    quality_thresholds: Optional[Dict[str, float]] = None
    enable_compression: bool = True
    enable_filtering: bool = True
    filter_window_size: int = 5

    def __post_init__(self):
        """Initialize default quality thresholds if not provided."""
        if self.quality_thresholds is None:
            self.quality_thresholds = {
                "min_value": -1e6,
                "max_value": 1e6,
                "max_rate_change": 1000.0,
                "noise_threshold": 0.1,
            }


@dataclass
class SensorReading:
    """Individual sensor reading with metadata."""

    sensor_id: str
    timestamp: float
    value: Union[float, np.ndarray]
    units: str
    quality: DataQuality = DataQuality.GOOD
    uncertainty: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        if isinstance(self.value, np.ndarray):
            result["value"] = self.value.tolist()
        result["quality"] = self.quality.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorReading":
        """Create from dictionary."""
        if "quality" in data and isinstance(data["quality"], str):
            data["quality"] = DataQuality(data["quality"])
        if isinstance(data.get("value"), list):
            data["value"] = np.array(data["value"])
        return cls(**data)


@dataclass
class SensorDataBatch:
    """Batch of sensor readings for efficient processing."""

    readings: List[SensorReading]
    batch_timestamp: float = field(default_factory=time.time)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    def add_reading(self, reading: SensorReading):
        """Add reading to batch."""
        self.readings.append(reading)
        self._update_quality_metrics()

    def _update_quality_metrics(self):
        """Update batch quality metrics."""
        if not self.readings:
            return

        qualities = [r.quality for r in self.readings]
        self.quality_metrics = {
            "total_readings": len(self.readings),
            "excellent_ratio": sum(1 for q in qualities if q == DataQuality.EXCELLENT)
            / len(qualities),
            "good_ratio": sum(1 for q in qualities if q == DataQuality.GOOD)
            / len(qualities),
            "acceptable_ratio": sum(1 for q in qualities if q == DataQuality.ACCEPTABLE)
            / len(qualities),
            "poor_ratio": sum(1 for q in qualities if q == DataQuality.POOR)
            / len(qualities),
            "invalid_ratio": sum(1 for q in qualities if q == DataQuality.INVALID)
            / len(qualities),
        }


class SensorInterface(ABC):
    """Abstract base class for sensor communication interfaces."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to sensor."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from sensor."""
        pass

    @abstractmethod
    async def read_sensor(self, sensor_id: str) -> SensorReading:
        """Read data from specific sensor."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if interface is connected."""
        pass


class MockSensorInterface(SensorInterface):
    """Mock sensor interface for testing and demonstration."""

    def __init__(self, sensor_configs: List[SensorConfiguration]):
        self.sensor_configs = {config.sensor_id: config for config in sensor_configs}
        self.connected = False
        self.start_time = time.time()

    async def connect(self) -> bool:
        """Connect to mock sensors."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.connected = True
        return True

    async def disconnect(self):
        """Disconnect from mock sensors."""
        self.connected = False

    async def read_sensor(self, sensor_id: str) -> SensorReading:
        """Generate mock sensor reading."""
        if not self.connected:
            raise ChemiAccelError("Sensor interface not connected")

        if sensor_id not in self.sensor_configs:
            raise ChemiAccelError(f"Unknown sensor: {sensor_id}")

        config = self.sensor_configs[sensor_id]
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Generate realistic sensor data based on type
        base_value = self._generate_base_value(config.sensor_type, elapsed)
        noise = np.random.normal(0, 0.02 * abs(base_value))
        value = base_value + noise

        # Simulate occasional poor quality readings
        quality = DataQuality.GOOD
        if np.random.random() < 0.05:  # 5% chance of poor quality
            quality = DataQuality.POOR
            value += np.random.normal(0, 0.1 * abs(base_value))
        elif np.random.random() < 0.1:  # 10% chance of excellent quality
            quality = DataQuality.EXCELLENT

        return SensorReading(
            sensor_id=sensor_id,
            timestamp=current_time,
            value=value,
            units=config.units,
            quality=quality,
            uncertainty=abs(noise),
            metadata={"interface": "mock", "protocol": config.protocol.value},
        )

    def _generate_base_value(self, sensor_type: SensorType, elapsed: float) -> float:
        """Generate base sensor value with realistic patterns."""
        if sensor_type == SensorType.TEMPERATURE:
            # Temperature with daily cycle and process variations
            daily_cycle = 20 * np.sin(2 * np.pi * elapsed / 86400)  # 24-hour cycle
            process_cycle = 10 * np.sin(
                2 * np.pi * elapsed / 3600
            )  # 1-hour process cycle
            return 25.0 + daily_cycle + process_cycle  # Celsius

        elif sensor_type == SensorType.PRESSURE:
            # Pressure with process-dependent variations
            process_variation = 0.1 * np.sin(2 * np.pi * elapsed / 1800)  # 30-min cycle
            return 1.01325 + process_variation  # Bar

        elif sensor_type == SensorType.FLOW_RATE:
            # Flow rate with step changes and disturbances
            base_flow = 50.0  # L/min
            step_changes = 10.0 * np.sin(2 * np.pi * elapsed / 7200)  # 2-hour steps
            return base_flow + step_changes

        elif sensor_type == SensorType.VIBRATION:
            # Vibration with machinery-related frequencies
            fundamental = 0.5 * np.sin(2 * np.pi * elapsed * 60)  # 60 Hz
            harmonic = 0.2 * np.sin(2 * np.pi * elapsed * 120)  # 120 Hz
            return fundamental + harmonic

        elif sensor_type == SensorType.FORCE:
            # Force with load-dependent variations
            cyclic_load = 1000 * (
                1 + 0.3 * np.sin(2 * np.pi * elapsed / 600)
            )  # 10-min cycle
            return cyclic_load  # N

        else:
            # Generic sinusoidal pattern
            return 10.0 * np.sin(2 * np.pi * elapsed / 3600)

    def is_connected(self) -> bool:
        """Check connection status."""
        return self.connected


class SensorDataValidator:
    """Validates sensor data quality and detects anomalies."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.sensor_histories = defaultdict(lambda: deque(maxlen=window_size))
        self.sensor_statistics = defaultdict(dict)

    def validate_reading(
        self, reading: SensorReading, config: SensorConfiguration
    ) -> SensorReading:
        """Validate and potentially correct sensor reading."""
        validated_reading = SensorReading(
            sensor_id=reading.sensor_id,
            timestamp=reading.timestamp,
            value=reading.value,
            units=reading.units,
            quality=reading.quality,
            uncertainty=reading.uncertainty,
            metadata=reading.metadata.copy(),
        )

        # Store reading for trend analysis
        self.sensor_histories[reading.sensor_id].append(reading.value)

        # Perform validation checks
        quality_score = 1.0
        validation_flags = []

        # Range check
        thresholds = config.quality_thresholds
        if not (thresholds["min_value"] <= reading.value <= thresholds["max_value"]):
            quality_score *= 0.3
            validation_flags.append("out_of_range")

        # Rate of change check
        if len(self.sensor_histories[reading.sensor_id]) >= 2:
            prev_value = list(self.sensor_histories[reading.sensor_id])[-2]
            rate_change = abs(reading.value - prev_value)
            if rate_change > thresholds["max_rate_change"]:
                quality_score *= 0.5
                validation_flags.append("high_rate_change")

        # Noise check using recent statistics
        if len(self.sensor_histories[reading.sensor_id]) >= 3:
            recent_values = list(self.sensor_histories[reading.sensor_id])
            std_dev = np.std(recent_values)
            if std_dev > thresholds["noise_threshold"]:
                quality_score *= 0.7
                validation_flags.append("high_noise")

        # Update quality based on validation
        if quality_score > 0.9:
            validated_reading.quality = DataQuality.EXCELLENT
        elif quality_score > 0.7:
            validated_reading.quality = DataQuality.GOOD
        elif quality_score > 0.5:
            validated_reading.quality = DataQuality.ACCEPTABLE
        elif quality_score > 0.3:
            validated_reading.quality = DataQuality.POOR
        else:
            validated_reading.quality = DataQuality.INVALID

        # Add validation metadata
        validated_reading.metadata.update(
            {
                "validation_score": quality_score,
                "validation_flags": validation_flags,
                "validator_timestamp": time.time(),
            }
        )

        return validated_reading

    def get_sensor_statistics(self, sensor_id: str) -> Dict[str, float]:
        """Get statistical summary for sensor."""
        if sensor_id not in self.sensor_histories:
            return {}

        values = list(self.sensor_histories[sensor_id])
        if not values:
            return {}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "trend": self._calculate_trend(values),
            "stability": 1.0 / (1.0 + np.std(values)),  # Higher is more stable
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)."""
        if len(values) < 2:
            return 0.0

        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]

        # Normalize to [-1, 1]
        return np.tanh(slope)


class SensorDataBuffer:
    """Thread-safe buffer for sensor data with automatic flushing."""

    def __init__(self, max_size: int = 1000, flush_interval: float = 1.0):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.flush_callbacks = []
        self.last_flush_time = time.time()

    def add_reading(self, reading: SensorReading):
        """Add reading to buffer."""
        with self.lock:
            self.buffer.append(reading)
            self._check_flush_conditions()

    def add_batch(self, batch: SensorDataBatch):
        """Add batch of readings to buffer."""
        with self.lock:
            for reading in batch.readings:
                self.buffer.append(reading)
            self._check_flush_conditions()

    def _check_flush_conditions(self):
        """Check if buffer should be flushed."""
        current_time = time.time()
        time_since_flush = current_time - self.last_flush_time

        if len(self.buffer) >= self.max_size or time_since_flush >= self.flush_interval:
            self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffer contents to callbacks."""
        if not self.buffer:
            return

        # Create batch from current buffer
        readings = list(self.buffer)
        batch = SensorDataBatch(readings)

        # Clear buffer
        self.buffer.clear()
        self.last_flush_time = time.time()

        # Notify callbacks
        for callback in self.flush_callbacks:
            try:
                callback(batch)
            except Exception as e:
                logging.error(f"Error in flush callback: {e}")

    def add_flush_callback(self, callback: Callable[[SensorDataBatch], None]):
        """Add callback to be called when buffer is flushed."""
        self.flush_callbacks.append(callback)

    def force_flush(self) -> Optional[SensorDataBatch]:
        """Force flush of current buffer contents."""
        with self.lock:
            if self.buffer:
                self._flush_buffer()
                return SensorDataBatch(list(self.buffer))
        return None

    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer status information."""
        with self.lock:
            return {
                "current_size": len(self.buffer),
                "max_size": self.max_size,
                "utilization": len(self.buffer) / self.max_size,
                "time_since_last_flush": time.time() - self.last_flush_time,
                "flush_interval": self.flush_interval,
            }


class SensorDataManager:
    """Main manager for sensor data integration and streaming."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.sensor_configs = {}
        self.sensor_interfaces = {}
        self.validator = SensorDataValidator()
        self.buffer = SensorDataBuffer(
            max_size=config.buffer_size, flush_interval=config.flush_interval
        )

        self.is_running = False
        self.collection_tasks = {}
        self.data_callbacks = []
        self.error_callbacks = []

        # Performance metrics
        self.metrics = {
            "readings_collected": 0,
            "readings_processed": 0,
            "data_quality_average": 0.0,
            "collection_errors": 0,
            "last_collection_time": 0.0,
        }

        # Setup buffer callbacks
        self.buffer.add_flush_callback(self._on_buffer_flush)

    def add_sensor(self, config: SensorConfiguration, interface: SensorInterface):
        """Add sensor configuration and interface."""
        self.sensor_configs[config.sensor_id] = config
        self.sensor_interfaces[config.sensor_id] = interface

    def remove_sensor(self, sensor_id: str):
        """Remove sensor configuration."""
        if sensor_id in self.sensor_configs:
            del self.sensor_configs[sensor_id]
        if sensor_id in self.sensor_interfaces:
            del self.sensor_interfaces[sensor_id]
        if sensor_id in self.collection_tasks:
            self.collection_tasks[sensor_id].cancel()
            del self.collection_tasks[sensor_id]

    async def start_collection(self):
        """Start sensor data collection."""
        if self.is_running:
            return

        # Connect all sensor interfaces
        for sensor_id, interface in self.sensor_interfaces.items():
            try:
                await interface.connect()
                logging.info(f"Connected to sensor: {sensor_id}")
            except Exception as e:
                logging.error(f"Failed to connect to sensor {sensor_id}: {e}")
                self._notify_error_callbacks(f"Connection error for {sensor_id}: {e}")

        # Start collection tasks for each sensor
        for sensor_id in self.sensor_configs:
            task = asyncio.create_task(self._collect_sensor_data(sensor_id))
            self.collection_tasks[sensor_id] = task

        self.is_running = True
        logging.info("Started sensor data collection")

    async def stop_collection(self):
        """Stop sensor data collection."""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel all collection tasks
        for task in self.collection_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self.collection_tasks:
            await asyncio.gather(
                *self.collection_tasks.values(), return_exceptions=True
            )

        self.collection_tasks.clear()

        # Disconnect all sensor interfaces
        for sensor_id, interface in self.sensor_interfaces.items():
            try:
                await interface.disconnect()
                logging.info(f"Disconnected from sensor: {sensor_id}")
            except Exception as e:
                logging.error(f"Error disconnecting from sensor {sensor_id}: {e}")

        # Force flush remaining data
        self.buffer.force_flush()

        logging.info("Stopped sensor data collection")

    async def _collect_sensor_data(self, sensor_id: str):
        """Collection loop for individual sensor."""
        config = self.sensor_configs[sensor_id]
        interface = self.sensor_interfaces[sensor_id]
        collection_interval = 1.0 / config.sampling_rate

        while self.is_running:
            try:
                start_time = time.time()

                # Read sensor data
                reading = await interface.read_sensor(sensor_id)

                # Validate reading
                validated_reading = self.validator.validate_reading(reading, config)

                # Add to buffer
                self.buffer.add_reading(validated_reading)

                # Update metrics
                self.metrics["readings_collected"] += 1
                self.metrics["last_collection_time"] = time.time()

                # Calculate sleep time to maintain sampling rate
                elapsed = time.time() - start_time
                sleep_time = max(0, collection_interval - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.metrics["collection_errors"] += 1
                error_msg = f"Error collecting data from sensor {sensor_id}: {e}"
                logging.error(error_msg)
                self._notify_error_callbacks(error_msg)

                # Wait before retrying
                await asyncio.sleep(1.0)

    def _on_buffer_flush(self, batch: SensorDataBatch):
        """Handle buffer flush events."""
        self.metrics["readings_processed"] += len(batch.readings)

        # Calculate data quality metrics
        quality_scores = []
        for reading in batch.readings:
            score = 1.0
            if reading.quality == DataQuality.EXCELLENT:
                score = 1.0
            elif reading.quality == DataQuality.GOOD:
                score = 0.8
            elif reading.quality == DataQuality.ACCEPTABLE:
                score = 0.6
            elif reading.quality == DataQuality.POOR:
                score = 0.3
            else:  # INVALID
                score = 0.0
            quality_scores.append(score)

        if quality_scores:
            self.metrics["data_quality_average"] = np.mean(quality_scores)

        # Notify data callbacks
        for callback in self.data_callbacks:
            try:
                callback(batch)
            except Exception as e:
                logging.error(f"Error in data callback: {e}")

    def add_data_callback(self, callback: Callable[[SensorDataBatch], None]):
        """Add callback for processed sensor data."""
        self.data_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str], None]):
        """Add callback for error notifications."""
        self.error_callbacks.append(callback)

    def _notify_error_callbacks(self, error_message: str):
        """Notify error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error_message)
            except Exception as e:
                logging.error(f"Error in error callback: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get collection metrics."""
        return {
            **self.metrics,
            "buffer_info": self.buffer.get_buffer_info(),
            "active_sensors": len(
                [s for s in self.sensor_interfaces.values() if s.is_connected()]
            ),
            "total_sensors": len(self.sensor_configs),
            "is_running": self.is_running,
        }

    def get_sensor_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sensors."""
        status = {}
        for sensor_id in self.sensor_configs:
            interface = self.sensor_interfaces.get(sensor_id)
            status[sensor_id] = {
                "connected": interface.is_connected() if interface else False,
                "statistics": self.validator.get_sensor_statistics(sensor_id),
                "config": asdict(self.sensor_configs[sensor_id]),
            }
        return status

    async def get_latest_readings(
        self, sensor_ids: Optional[List[str]] = None
    ) -> Dict[str, SensorReading]:
        """Get latest readings from specified sensors."""
        if sensor_ids is None:
            sensor_ids = list(self.sensor_configs.keys())

        readings = {}
        for sensor_id in sensor_ids:
            if sensor_id in self.sensor_interfaces:
                try:
                    interface = self.sensor_interfaces[sensor_id]
                    if interface.is_connected():
                        reading = await interface.read_sensor(sensor_id)
                        validated_reading = self.validator.validate_reading(
                            reading, self.sensor_configs[sensor_id]
                        )
                        readings[sensor_id] = validated_reading
                except Exception as e:
                    logging.error(f"Error reading sensor {sensor_id}: {e}")

        return readings

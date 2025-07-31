"""
Real-time Data Streaming and Update Mechanisms

This module provides comprehensive data streaming capabilities for real-time process simulation,
including data ingestion, processing, and distribution to visualization and analysis systems.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.exceptions import ChemiAccelError
from .manufacturing_integration import ProcessConditions, ProcessOutput
from .realtime_process_simulation import (
    MaterialProperty,
    MicrostructureState,
    RealTimeSimulationCore,
    SimulationConfig,
    SimulationState,
)


class StreamingMode(Enum):
    """Data streaming modes."""

    PUSH = "push"
    PULL = "pull"
    HYBRID = "hybrid"
    BATCH = "batch"


class DataFormat(Enum):
    """Data serialization formats."""

    JSON = "json"
    BINARY = "binary"
    PROTOBUF = "protobuf"
    ARROW = "arrow"


class QualityOfService(Enum):
    """Quality of service levels for data streaming."""

    BEST_EFFORT = "best_effort"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"
    RELIABLE = "reliable"


@dataclass
class StreamingConfig:
    """Configuration for data streaming system."""

    mode: StreamingMode = StreamingMode.PUSH
    data_format: DataFormat = DataFormat.JSON
    qos: QualityOfService = QualityOfService.RELIABLE
    buffer_size: int = 1000
    batch_size: int = 10
    flush_interval: float = 1.0  # seconds
    compression_enabled: bool = True
    encryption_enabled: bool = False
    max_retries: int = 3
    retry_delay: float = 0.1
    enable_persistence: bool = True
    persistence_path: str = "streaming_data"
    max_concurrent_streams: int = 10
    heartbeat_interval: float = 30.0


@dataclass
class DataPacket:
    """Container for streaming data packets."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str = "simulation"
    data_type: str = "simulation_state"
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sequence_number: int = 0
    checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPacket":
        """Create from dictionary."""
        return cls(**data)

    def serialize(self, format: DataFormat = DataFormat.JSON) -> bytes:
        """Serialize packet to bytes."""
        if format == DataFormat.JSON:
            return json.dumps(self.to_dict(), default=str).encode("utf-8")
        elif format == DataFormat.BINARY:
            import pickle

            return pickle.dumps(self.to_dict())
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def deserialize(
        cls, data: bytes, format: DataFormat = DataFormat.JSON
    ) -> "DataPacket":
        """Deserialize packet from bytes."""
        if format == DataFormat.JSON:
            packet_dict = json.loads(data.decode("utf-8"))
            return cls.from_dict(packet_dict)
        elif format == DataFormat.BINARY:
            import pickle

            packet_dict = pickle.loads(data)
            return cls.from_dict(packet_dict)
        else:
            raise ValueError(f"Unsupported format: {format}")


class StreamProcessor(ABC):
    """Abstract base class for stream processors."""

    @abstractmethod
    async def process(self, packet: DataPacket) -> Optional[DataPacket]:
        """Process a data packet."""
        pass

    @abstractmethod
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information."""
        pass


class PropertyAggregationProcessor(StreamProcessor):
    """Processor for aggregating material properties."""

    def __init__(
        self, window_size: int = 10, aggregation_functions: Optional[List[str]] = None
    ):
        self.window_size = window_size
        self.aggregation_functions = aggregation_functions or [
            "mean",
            "std",
            "min",
            "max",
        ]
        self.property_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    async def process(self, packet: DataPacket) -> Optional[DataPacket]:
        """Process property data for aggregation."""
        if packet.data_type != "simulation_state":
            return None

        properties = packet.payload.get("material_properties", {})
        aggregated_properties = {}

        for prop_name, prop_data in properties.items():
            if isinstance(prop_data, dict) and "value" in prop_data:
                value = prop_data["value"]
                if isinstance(value, (int, float)):
                    # Add to window
                    self.property_windows[prop_name].append(value)

                    # Calculate aggregations
                    window_values = list(self.property_windows[prop_name])
                    if len(window_values) >= 2:
                        aggregated_properties[prop_name] = {
                            "mean": np.mean(window_values),
                            "std": np.std(window_values),
                            "min": np.min(window_values),
                            "max": np.max(window_values),
                            "trend": (
                                np.polyfit(range(len(window_values)), window_values, 1)[
                                    0
                                ]
                                if len(window_values) > 1
                                else 0.0
                            ),
                        }

        if aggregated_properties:
            return DataPacket(
                source="property_aggregator",
                data_type="aggregated_properties",
                payload={"aggregated_properties": aggregated_properties},
                metadata={"window_size": self.window_size},
            )

        return None

    def get_processor_info(self) -> Dict[str, Any]:
        return {
            "type": "PropertyAggregationProcessor",
            "window_size": self.window_size,
            "aggregation_functions": self.aggregation_functions,
        }


class AnomalyDetectionProcessor(StreamProcessor):
    """Processor for detecting anomalies in simulation data."""

    def __init__(self, threshold_multiplier: float = 2.0, min_samples: int = 10):
        self.threshold_multiplier = threshold_multiplier
        self.min_samples = min_samples
        self.property_baselines: Dict[str, Dict[str, float]] = {}
        self.property_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    async def process(self, packet: DataPacket) -> Optional[DataPacket]:
        """Process data for anomaly detection."""
        if packet.data_type != "simulation_state":
            return None

        properties = packet.payload.get("material_properties", {})
        anomalies = {}

        for prop_name, prop_data in properties.items():
            if isinstance(prop_data, dict) and "value" in prop_data:
                value = prop_data["value"]
                if isinstance(value, (int, float)):
                    # Update history
                    self.property_history[prop_name].append(value)

                    # Calculate baseline if we have enough samples
                    if len(self.property_history[prop_name]) >= self.min_samples:
                        values = list(self.property_history[prop_name])
                        mean_val = np.mean(values)
                        std_val = np.std(values)

                        self.property_baselines[prop_name] = {
                            "mean": mean_val,
                            "std": std_val,
                            "upper_threshold": mean_val
                            + self.threshold_multiplier * std_val,
                            "lower_threshold": mean_val
                            - self.threshold_multiplier * std_val,
                        }

                        # Check for anomaly
                        baseline = self.property_baselines[prop_name]
                        if (
                            value > baseline["upper_threshold"]
                            or value < baseline["lower_threshold"]
                        ):
                            anomalies[prop_name] = {
                                "value": value,
                                "baseline_mean": baseline["mean"],
                                "deviation": abs(value - baseline["mean"])
                                / max(baseline["std"], 1e-10),
                                "severity": (
                                    "high"
                                    if abs(value - baseline["mean"])
                                    > 3 * baseline["std"]
                                    else "medium"
                                ),
                            }

        if anomalies:
            return DataPacket(
                source="anomaly_detector",
                data_type="anomalies",
                payload={"anomalies": anomalies},
                metadata={"threshold_multiplier": self.threshold_multiplier},
            )

        return None

    def get_processor_info(self) -> Dict[str, Any]:
        return {
            "type": "AnomalyDetectionProcessor",
            "threshold_multiplier": self.threshold_multiplier,
            "min_samples": self.min_samples,
            "baselines_count": len(self.property_baselines),
        }


class DataStreamManager:
    """Main manager for data streaming operations."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.streams: Dict[str, DataStream] = {}
        self.processors: List[StreamProcessor] = []
        self.subscribers: Dict[str, List[Callable[[DataPacket], None]]] = defaultdict(
            list
        )

        # Buffer management
        self.buffer: deque = deque(maxlen=config.buffer_size)
        self.processing_queue: Queue = Queue(maxsize=config.buffer_size)

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_streams)
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False

        # Persistence
        self.persistence_enabled = config.enable_persistence
        if self.persistence_enabled:
            import os

            os.makedirs(config.persistence_path, exist_ok=True)

        # Metrics
        self.metrics = {
            "packets_processed": 0,
            "packets_dropped": 0,
            "processing_time_avg": 0.0,
            "buffer_utilization": 0.0,
            "error_count": 0,
        }

        self.logger = logging.getLogger(__name__)

    def add_processor(self, processor: StreamProcessor):
        """Add a stream processor."""
        self.processors.append(processor)
        self.logger.info("Added processor: %s", processor.get_processor_info()["type"])

    def remove_processor(self, processor: StreamProcessor):
        """Remove a stream processor."""
        if processor in self.processors:
            self.processors.remove(processor)

    def subscribe(self, data_type: str, callback: Callable[[DataPacket], None]):
        """Subscribe to data packets of specific type."""
        self.subscribers[data_type].append(callback)
        self.logger.info("Added subscriber for data type: %s", data_type)

    def unsubscribe(self, data_type: str, callback: Callable[[DataPacket], None]):
        """Unsubscribe from data packets."""
        if callback in self.subscribers[data_type]:
            self.subscribers[data_type].remove(callback)

    def start_streaming(self):
        """Start the streaming system."""
        if self.is_running:
            return

        self.is_running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.logger.info("Data streaming started")

    def stop_streaming(self):
        """Stop the streaming system."""
        self.is_running = False

        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        self.logger.info("Data streaming stopped")

    def _processing_loop(self):
        """Main processing loop for streaming data."""
        batch_buffer = []
        last_flush_time = time.time()

        while self.is_running:
            try:
                # Get packet from queue with timeout
                try:
                    packet = self.processing_queue.get(timeout=0.1)
                except Empty:
                    # Check if we need to flush batch
                    if (
                        batch_buffer
                        and time.time() - last_flush_time > self.config.flush_interval
                    ):
                        self._process_batch(batch_buffer)
                        batch_buffer.clear()
                        last_flush_time = time.time()
                    continue

                # Add to batch
                batch_buffer.append(packet)

                # Process batch if full or flush interval reached
                if (
                    len(batch_buffer) >= self.config.batch_size
                    or time.time() - last_flush_time > self.config.flush_interval
                ):
                    self._process_batch(batch_buffer)
                    batch_buffer.clear()
                    last_flush_time = time.time()

            except Exception as e:
                self.logger.error("Processing loop error: %s", e)
                self.metrics["error_count"] += 1

        # Process any remaining packets
        if batch_buffer:
            self._process_batch(batch_buffer)

    def _process_batch(self, batch: List[DataPacket]):
        """Process a batch of data packets."""
        start_time = time.time()

        for packet in batch:
            try:
                # Apply processors
                processed_packets = [packet]

                for processor in self.processors:
                    new_packets = []
                    for p in processed_packets:
                        result = asyncio.run(processor.process(p))
                        if result:
                            new_packets.append(result)

                    # Add new packets to the list
                    processed_packets.extend(new_packets)

                # Distribute to subscribers
                for p in processed_packets:
                    for callback in self.subscribers[p.data_type]:
                        try:
                            callback(p)
                        except Exception as e:
                            self.logger.warning("Subscriber callback error: %s", e)

                    # Generic subscribers (for all data types)
                    for callback in self.subscribers.get("*", []):
                        try:
                            callback(p)
                        except Exception as e:
                            self.logger.warning(
                                "Generic subscriber callback error: %s", e
                            )

                # Persist if enabled
                if self.persistence_enabled:
                    self._persist_packet(packet)

                self.metrics["packets_processed"] += 1

            except Exception as e:
                self.logger.error("Packet processing error: %s", e)
                self.metrics["packets_dropped"] += 1

        # Update metrics
        processing_time = time.time() - start_time
        self.metrics["processing_time_avg"] = (
            self.metrics["processing_time_avg"] * 0.9
        ) + (processing_time * 0.1)
        self.metrics["buffer_utilization"] = (
            self.processing_queue.qsize() / self.config.buffer_size
        )

    def _persist_packet(self, packet: DataPacket):
        """Persist packet to storage."""
        try:
            timestamp_str = str(int(packet.timestamp))
            filename = f"{timestamp_str}_{packet.id}.json"
            filepath = f"{self.config.persistence_path}/{filename}"

            with open(filepath, "w") as f:
                json.dump(packet.to_dict(), f, default=str)

        except Exception as e:
            self.logger.warning("Persistence error: %s", e)

    def ingest_simulation_state(self, state: SimulationState):
        """Ingest simulation state as data packet."""
        packet = DataPacket(
            source="simulation_core",
            data_type="simulation_state",
            payload={
                "time": state.time,
                "material_properties": {
                    name: prop.to_dict()
                    for name, prop in state.material_properties.items()
                },
                "microstructure": {
                    "grain_size": state.microstructure.grain_size,
                    "crystallinity": state.microstructure.crystallinity,
                    "defect_density": state.microstructure.defect_density,
                    "phase_fractions": state.microstructure.phase_fractions,
                },
                "process_conditions": {
                    "temperature": state.process_conditions.temperature,
                    "pressure": state.process_conditions.pressure,
                    "flow_rate": getattr(state.process_conditions, "flow_rate", 0.0),
                },
            },
            metadata={"simulation_time": state.time},
        )

        self._enqueue_packet(packet)

    def ingest_process_output(self, output: ProcessOutput, timestamp: float):
        """Ingest process output as data packet."""
        packet = DataPacket(
            source="process_output",
            data_type="process_output",
            payload={
                "part_quality": output.part_quality,
                "dimensional_accuracy": getattr(output, "dimensional_accuracy", None),
                "surface_finish": getattr(output, "surface_finish", None),
                "processing_time": getattr(output, "processing_time", None),
                "energy_consumption": getattr(output, "energy_consumption", None),
            },
            metadata={"timestamp": timestamp},
        )

        self._enqueue_packet(packet)

    def _enqueue_packet(self, packet: DataPacket):
        """Enqueue packet for processing."""
        try:
            self.processing_queue.put_nowait(packet)
        except:
            # Queue is full, drop oldest packet
            try:
                self.processing_queue.get_nowait()
                self.processing_queue.put_nowait(packet)
                self.metrics["packets_dropped"] += 1
            except:
                pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming metrics."""
        return self.metrics.copy()

    def get_stream_info(self) -> Dict[str, Any]:
        """Get information about active streams."""
        return {
            "total_streams": len(self.streams),
            "active_processors": len(self.processors),
            "subscriber_count": sum(len(subs) for subs in self.subscribers.values()),
            "buffer_size": self.processing_queue.qsize(),
            "is_running": self.is_running,
        }


class RealTimeDataConnector:
    """Connector for integrating simulation with streaming system."""

    def __init__(
        self, simulation_core: RealTimeSimulationCore, stream_manager: DataStreamManager
    ):
        self.simulation_core = simulation_core
        self.stream_manager = stream_manager

        # Register callback with simulation
        simulation_core.add_update_callback(self._on_simulation_update)

        self.logger = logging.getLogger(__name__)

    def _on_simulation_update(self, state: SimulationState):
        """Handle simulation state updates."""
        try:
            # Ingest simulation state
            self.stream_manager.ingest_simulation_state(state)

        except Exception as e:
            self.logger.error("Error ingesting simulation state: %s", e)

    def setup_default_processing_pipeline(self):
        """Setup default data processing pipeline."""

        # Add property aggregation processor
        aggregation_processor = PropertyAggregationProcessor(
            window_size=10, aggregation_functions=["mean", "std", "min", "max"]
        )
        self.stream_manager.add_processor(aggregation_processor)

        # Add anomaly detection processor
        anomaly_processor = AnomalyDetectionProcessor(
            threshold_multiplier=2.5, min_samples=20
        )
        self.stream_manager.add_processor(anomaly_processor)

        self.logger.info("Default processing pipeline setup complete")

    def start_streaming(self):
        """Start the streaming system."""
        self.stream_manager.start_streaming()

    def stop_streaming(self):
        """Stop the streaming system."""
        self.stream_manager.stop_streaming()


# Export streaming data to external systems
class DataExporter:
    """Export streaming data to external systems."""

    def __init__(self, stream_manager: DataStreamManager):
        self.stream_manager = stream_manager
        self.export_targets: Dict[str, Callable[[DataPacket], None]] = {}

    def add_export_target(
        self, name: str, export_function: Callable[[DataPacket], None]
    ):
        """Add an export target."""
        self.export_targets[name] = export_function

        # Subscribe to all data types
        self.stream_manager.subscribe(
            "*", lambda packet: self._export_to_target(name, packet)
        )

    def _export_to_target(self, target_name: str, packet: DataPacket):
        """Export packet to specific target."""
        try:
            export_function = self.export_targets[target_name]
            export_function(packet)
        except Exception as e:
            logging.getLogger(__name__).error(f"Export to {target_name} failed: {e}")

    def export_to_database(self, packet: DataPacket):
        """Example export function for database."""
        # This would implement actual database export
        print(f"Exporting to database: {packet.data_type} at {packet.timestamp}")

    def export_to_file(self, packet: DataPacket):
        """Example export function for file."""
        # This would implement file export
        print(f"Exporting to file: {packet.data_type} at {packet.timestamp}")

    def export_to_api(self, packet: DataPacket):
        """Example export function for REST API."""
        # This would implement API export
        print(f"Exporting to API: {packet.data_type} at {packet.timestamp}")

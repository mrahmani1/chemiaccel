"""
Sensor-Digital Twin Integration Module

This module provides comprehensive integration between sensor data and the digital twin,
enabling real-time model updates, state synchronization, and closed-loop control.

Key Features:
- Real-time digital twin state synchronization with sensor data
- Sensor-driven model parameter estimation and updating
- State estimation and sensor fusion algorithms
- Anomaly detection and model drift compensation
- Predictive maintenance and equipment health monitoring
- Closed-loop control integration with manufacturing processes
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# ChemiAccel imports
from ..core.exceptions import ChemiAccelError
from ..manufacturing_integration import ProcessConditions, ProcessOutput
from .predictive_control import (
    AdaptiveController,
    ControlAction,
    ModelPredictiveController,
)
from .realtime_data_streaming import DataStreamManager
from .realtime_process_simulation import (
    MaterialProperty,
    RealTimeSimulationCore,
    SimulationState,
)
from .sensor_data_integration import SensorDataBatch, SensorDataManager, SensorReading


class UpdateStrategy(Enum):
    """Strategies for updating digital twin from sensor data."""

    IMMEDIATE = "immediate"  # Update immediately upon sensor data
    BATCH = "batch"  # Update in batches
    SCHEDULED = "scheduled"  # Update on schedule
    TRIGGERED = "triggered"  # Update when conditions are met
    ADAPTIVE = "adaptive"  # Adapt update frequency based on conditions


class FusionMethod(Enum):
    """Methods for sensor data fusion."""

    WEIGHTED_AVERAGE = "weighted_average"
    KALMAN_FILTER = "kalman_filter"
    PARTICLE_FILTER = "particle_filter"
    NEURAL_FUSION = "neural_fusion"
    BAYESIAN_FUSION = "bayesian_fusion"


class StateEstimationMethod(Enum):
    """Methods for state estimation."""

    DIRECT_MEASUREMENT = "direct"
    INTERPOLATION = "interpolation"
    EXTRAPOLATION = "extrapolation"
    MODEL_BASED = "model_based"
    HYBRID = "hybrid"


@dataclass
class SensorTwinMapping:
    """Mapping between sensor data and digital twin parameters."""

    sensor_id: str
    twin_parameter: str
    mapping_function: Optional[Callable[[float], float]] = None
    uncertainty_model: Optional[Callable[[float], float]] = None
    validation_range: Optional[Tuple[float, float]] = None
    trust_factor: float = 1.0  # 0-1, higher means more trust in sensor

    def apply_mapping(self, sensor_value: float) -> Tuple[float, float]:
        """Apply mapping to convert sensor value to twin parameter."""
        # Apply mapping function if provided
        if self.mapping_function:
            twin_value = self.mapping_function(sensor_value)
        else:
            twin_value = sensor_value

        # Calculate uncertainty
        if self.uncertainty_model:
            uncertainty = self.uncertainty_model(sensor_value)
        else:
            uncertainty = 0.1 * abs(twin_value)  # Default 10% uncertainty

        # Apply trust factor
        uncertainty = uncertainty / max(self.trust_factor, 0.1)

        return twin_value, uncertainty

    def is_valid(self, sensor_value: float) -> bool:
        """Check if sensor value is within validation range."""
        if self.validation_range is None:
            return True
        return self.validation_range[0] <= sensor_value <= self.validation_range[1]


@dataclass
class TwinUpdateEvent:
    """Event representing a digital twin update from sensor data."""

    timestamp: float
    sensor_id: str
    twin_parameter: str
    old_value: float
    new_value: float
    confidence: float
    update_source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_update_magnitude(self) -> float:
        """Get magnitude of the update."""
        return abs(self.new_value - self.old_value)


class StateEstimator:
    """Advanced state estimator for digital twin using sensor fusion."""

    def __init__(self, fusion_method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE):
        self.fusion_method = fusion_method
        self.sensor_weights = {}
        self.state_history = deque(maxlen=100)
        self.estimation_errors = deque(maxlen=1000)

        # Kalman filter parameters (if used)
        self.kalman_state = None
        self.kalman_covariance = None
        self.process_noise = 0.01
        self.measurement_noise = 0.1

    def add_sensor_weight(self, sensor_id: str, weight: float):
        """Add or update sensor weight for fusion."""
        self.sensor_weights[sensor_id] = max(0.0, min(1.0, weight))

    def estimate_state(
        self,
        sensor_readings: Dict[str, SensorReading],
        twin_mappings: Dict[str, SensorTwinMapping],
        current_twin_state: SimulationState,
    ) -> Dict[str, Tuple[float, float]]:
        """Estimate digital twin state parameters from sensor data."""
        estimated_parameters = {}

        # Group sensors by twin parameter
        parameter_sensors = {}
        for sensor_id, mapping in twin_mappings.items():
            if sensor_id in sensor_readings:
                param = mapping.twin_parameter
                if param not in parameter_sensors:
                    parameter_sensors[param] = []
                parameter_sensors[param].append(
                    (sensor_id, mapping, sensor_readings[sensor_id])
                )

        # Estimate each parameter
        for parameter, sensor_data in parameter_sensors.items():
            if self.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
                estimated_value, uncertainty = self._weighted_average_fusion(
                    sensor_data
                )
            elif self.fusion_method == FusionMethod.KALMAN_FILTER:
                estimated_value, uncertainty = self._kalman_filter_fusion(
                    parameter, sensor_data, current_twin_state
                )
            elif self.fusion_method == FusionMethod.NEURAL_FUSION:
                estimated_value, uncertainty = self._neural_fusion(sensor_data)
            else:
                # Default to weighted average
                estimated_value, uncertainty = self._weighted_average_fusion(
                    sensor_data
                )

            estimated_parameters[parameter] = (estimated_value, uncertainty)

        # Store estimation for history
        self.state_history.append(
            {
                "timestamp": time.time(),
                "estimated_parameters": estimated_parameters,
                "sensor_count": len(sensor_readings),
            }
        )

        return estimated_parameters

    def _weighted_average_fusion(
        self, sensor_data: List[Tuple[str, SensorTwinMapping, SensorReading]]
    ) -> Tuple[float, float]:
        """Fuse sensor data using weighted average."""
        total_weight = 0.0
        weighted_sum = 0.0
        uncertainty_sum = 0.0

        for sensor_id, mapping, reading in sensor_data:
            if not mapping.is_valid(reading.value):
                continue

            twin_value, uncertainty = mapping.apply_mapping(reading.value)
            weight = self.sensor_weights.get(sensor_id, 1.0) * mapping.trust_factor

            # Adjust weight based on reading quality
            if reading.quality.value == "excellent":
                weight *= 1.2
            elif reading.quality.value == "poor":
                weight *= 0.5
            elif reading.quality.value == "invalid":
                continue

            weighted_sum += twin_value * weight
            total_weight += weight
            uncertainty_sum += uncertainty * weight

        if total_weight == 0:
            return 0.0, 1.0  # No valid data

        estimated_value = weighted_sum / total_weight
        estimated_uncertainty = uncertainty_sum / total_weight

        return estimated_value, estimated_uncertainty

    def _kalman_filter_fusion(
        self,
        parameter: str,
        sensor_data: List[Tuple[str, SensorTwinMapping, SensorReading]],
        current_state: SimulationState,
    ) -> Tuple[float, float]:
        """Fuse sensor data using Kalman filtering."""
        # Initialize Kalman filter if needed
        if self.kalman_state is None:
            self.kalman_state = {}
            self.kalman_covariance = {}

        if parameter not in self.kalman_state:
            # Initialize with current twin state
            if parameter == "temperature":
                initial_value = current_state.process_conditions.temperature
            elif parameter == "pressure":
                initial_value = current_state.process_conditions.pressure
            else:
                initial_value = 0.0

            self.kalman_state[parameter] = initial_value
            self.kalman_covariance[parameter] = 1.0

        # Prediction step (assume constant model for simplicity)
        predicted_state = self.kalman_state[parameter]
        predicted_covariance = self.kalman_covariance[parameter] + self.process_noise

        # Update step with sensor measurements
        for sensor_id, mapping, reading in sensor_data:
            if not mapping.is_valid(reading.value):
                continue

            measurement, measurement_uncertainty = mapping.apply_mapping(reading.value)
            measurement_noise = max(self.measurement_noise, measurement_uncertainty)

            # Kalman gain
            kalman_gain = predicted_covariance / (
                predicted_covariance + measurement_noise
            )

            # State update
            predicted_state = predicted_state + kalman_gain * (
                measurement - predicted_state
            )
            predicted_covariance = (1 - kalman_gain) * predicted_covariance

        # Store updated state
        self.kalman_state[parameter] = predicted_state
        self.kalman_covariance[parameter] = predicted_covariance

        return predicted_state, np.sqrt(predicted_covariance)

    def _neural_fusion(
        self, sensor_data: List[Tuple[str, SensorTwinMapping, SensorReading]]
    ) -> Tuple[float, float]:
        """Fuse sensor data using neural network (simplified implementation)."""
        # For demonstration, use weighted average with learned weights
        values = []
        uncertainties = []

        for sensor_id, mapping, reading in sensor_data:
            if mapping.is_valid(reading.value):
                twin_value, uncertainty = mapping.apply_mapping(reading.value)
                values.append(twin_value)
                uncertainties.append(uncertainty)

        if not values:
            return 0.0, 1.0

        # Simple neural-like fusion (could be replaced with actual neural network)
        weights = np.exp(-np.array(uncertainties))  # Lower uncertainty = higher weight
        weights = weights / np.sum(weights)

        fused_value = np.sum(np.array(values) * weights)
        fused_uncertainty = np.sqrt(np.sum((np.array(uncertainties) * weights) ** 2))

        return float(fused_value), float(fused_uncertainty)

    def get_estimation_quality(self) -> Dict[str, float]:
        """Get quality metrics for state estimation."""
        if not self.estimation_errors:
            return {"estimation_accuracy": 1.0}

        recent_errors = list(self.estimation_errors)
        mean_error = np.mean(recent_errors)
        std_error = np.std(recent_errors)

        return {
            "estimation_accuracy": max(0.0, 1.0 - mean_error),
            "estimation_consistency": max(0.0, 1.0 - std_error),
            "sample_count": len(recent_errors),
        }


class DigitalTwinUpdater:
    """Manages updates to digital twin based on sensor feedback."""

    def __init__(
        self,
        simulation_core: RealTimeSimulationCore,
        update_strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE,
    ):
        self.simulation_core = simulation_core
        self.update_strategy = update_strategy
        self.sensor_mappings = {}
        self.state_estimator = StateEstimator()
        self.update_history = deque(maxlen=1000)
        self.update_callbacks = []

        # Update scheduling
        self.last_update_time = 0.0
        self.update_interval = 1.0  # seconds
        self.pending_updates = {}

        # Performance tracking
        self.update_metrics = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "average_update_time": 0.0,
            "last_update_timestamp": 0.0,
        }

    def add_sensor_mapping(self, mapping: SensorTwinMapping):
        """Add sensor-to-twin parameter mapping."""
        self.sensor_mappings[mapping.sensor_id] = mapping

        # Set sensor weight in state estimator
        self.state_estimator.add_sensor_weight(mapping.sensor_id, mapping.trust_factor)

    def remove_sensor_mapping(self, sensor_id: str):
        """Remove sensor mapping."""
        if sensor_id in self.sensor_mappings:
            del self.sensor_mappings[sensor_id]

    async def update_twin_from_sensors(
        self, sensor_data: Dict[str, SensorReading]
    ) -> List[TwinUpdateEvent]:
        """Update digital twin state based on sensor data."""
        start_time = time.time()
        update_events = []

        try:
            # Get current twin state
            current_state = self.simulation_core.current_state
            if current_state is None:
                logging.warning("No current simulation state available for update")
                return update_events

            # Filter sensor data based on available mappings
            mapped_sensors = {
                sensor_id: reading
                for sensor_id, reading in sensor_data.items()
                if sensor_id in self.sensor_mappings
            }

            if not mapped_sensors:
                return update_events

            # Estimate twin parameters from sensor data
            estimated_parameters = self.state_estimator.estimate_state(
                mapped_sensors, self.sensor_mappings, current_state
            )

            # Apply updates based on strategy
            if self.update_strategy == UpdateStrategy.IMMEDIATE:
                update_events = await self._apply_immediate_updates(
                    estimated_parameters, current_state
                )
            elif self.update_strategy == UpdateStrategy.BATCH:
                self._accumulate_batch_updates(estimated_parameters)
                update_events = await self._apply_batch_updates_if_ready()
            elif self.update_strategy == UpdateStrategy.ADAPTIVE:
                update_events = await self._apply_adaptive_updates(
                    estimated_parameters, current_state, sensor_data
                )
            else:
                update_events = await self._apply_immediate_updates(
                    estimated_parameters, current_state
                )

            # Update metrics
            self.update_metrics["total_updates"] += len(update_events)
            self.update_metrics["successful_updates"] += len(update_events)
            self.update_metrics["last_update_timestamp"] = time.time()

            # Store update history
            for event in update_events:
                self.update_history.append(event)

            # Notify callbacks
            for callback in self.update_callbacks:
                try:
                    callback(update_events)
                except Exception as e:
                    logging.error(f"Error in update callback: {e}")

        except Exception as e:
            self.update_metrics["failed_updates"] += 1
            logging.error(f"Error updating twin from sensors: {e}")
            raise ChemiAccelError(f"Twin update failed: {e}")

        finally:
            # Update timing metrics
            update_time = time.time() - start_time
            self.update_metrics["average_update_time"] = (
                0.9 * self.update_metrics["average_update_time"] + 0.1 * update_time
            )

        return update_events

    async def _apply_immediate_updates(
        self,
        estimated_parameters: Dict[str, Tuple[float, float]],
        current_state: SimulationState,
    ) -> List[TwinUpdateEvent]:
        """Apply updates immediately."""
        update_events = []

        for parameter, (new_value, uncertainty) in estimated_parameters.items():
            old_value = self._get_current_parameter_value(parameter, current_state)
            confidence = max(0.0, 1.0 - uncertainty / max(abs(new_value), 1.0))

            # Apply update if confidence is sufficient
            if confidence > 0.5:
                success = await self._update_simulation_parameter(
                    parameter, new_value, current_state
                )

                if success:
                    update_events.append(
                        TwinUpdateEvent(
                            timestamp=time.time(),
                            sensor_id="fused",
                            twin_parameter=parameter,
                            old_value=old_value,
                            new_value=new_value,
                            confidence=confidence,
                            update_source="immediate",
                            metadata={"uncertainty": uncertainty},
                        )
                    )

        return update_events

    def _accumulate_batch_updates(
        self, estimated_parameters: Dict[str, Tuple[float, float]]
    ):
        """Accumulate updates for batch processing."""
        for parameter, (value, uncertainty) in estimated_parameters.items():
            if parameter not in self.pending_updates:
                self.pending_updates[parameter] = []

            self.pending_updates[parameter].append(
                {"timestamp": time.time(), "value": value, "uncertainty": uncertainty}
            )

    async def _apply_batch_updates_if_ready(self) -> List[TwinUpdateEvent]:
        """Apply batch updates if conditions are met."""
        current_time = time.time()

        # Check if it's time for batch update
        if current_time - self.last_update_time < self.update_interval:
            return []

        update_events = []
        current_state = self.simulation_core.current_state

        for parameter, updates in self.pending_updates.items():
            if not updates:
                continue

            # Average the pending updates
            values = [u["value"] for u in updates]
            uncertainties = [u["uncertainty"] for u in updates]

            avg_value = np.mean(values)
            avg_uncertainty = np.mean(uncertainties)
            confidence = max(0.0, 1.0 - avg_uncertainty / max(abs(avg_value), 1.0))

            old_value = self._get_current_parameter_value(parameter, current_state)

            # Apply update
            if confidence > 0.5:
                success = await self._update_simulation_parameter(
                    parameter, avg_value, current_state
                )

                if success:
                    update_events.append(
                        TwinUpdateEvent(
                            timestamp=current_time,
                            sensor_id="batch",
                            twin_parameter=parameter,
                            old_value=old_value,
                            new_value=avg_value,
                            confidence=confidence,
                            update_source="batch",
                            metadata={
                                "uncertainty": avg_uncertainty,
                                "batch_size": len(updates),
                            },
                        )
                    )

        # Clear pending updates
        self.pending_updates.clear()
        self.last_update_time = current_time

        return update_events

    async def _apply_adaptive_updates(
        self,
        estimated_parameters: Dict[str, Tuple[float, float]],
        current_state: SimulationState,
        sensor_data: Dict[str, SensorReading],
    ) -> List[TwinUpdateEvent]:
        """Apply updates using adaptive strategy."""
        update_events = []

        # Determine update urgency based on sensor data quality and changes
        urgency_score = self._calculate_update_urgency(
            estimated_parameters, current_state, sensor_data
        )

        # Adapt update frequency based on urgency
        if urgency_score > 0.8:
            # High urgency - immediate update
            update_events = await self._apply_immediate_updates(
                estimated_parameters, current_state
            )
        elif urgency_score > 0.5:
            # Medium urgency - batch update with shorter interval
            self.update_interval = 0.5
            self._accumulate_batch_updates(estimated_parameters)
            update_events = await self._apply_batch_updates_if_ready()
        else:
            # Low urgency - normal batch processing
            self.update_interval = 2.0
            self._accumulate_batch_updates(estimated_parameters)
            update_events = await self._apply_batch_updates_if_ready()

        return update_events

    def _calculate_update_urgency(
        self,
        estimated_parameters: Dict[str, Tuple[float, float]],
        current_state: SimulationState,
        sensor_data: Dict[str, SensorReading],
    ) -> float:
        """Calculate urgency score for updates."""
        urgency_factors = []

        # Factor 1: Magnitude of parameter changes
        for parameter, (new_value, uncertainty) in estimated_parameters.items():
            old_value = self._get_current_parameter_value(parameter, current_state)
            relative_change = abs(new_value - old_value) / max(abs(old_value), 1.0)
            urgency_factors.append(min(1.0, relative_change * 2.0))

        # Factor 2: Sensor data quality
        quality_scores = []
        for reading in sensor_data.values():
            if reading.quality.value == "excellent":
                quality_scores.append(1.0)
            elif reading.quality.value == "good":
                quality_scores.append(0.8)
            elif reading.quality.value == "acceptable":
                quality_scores.append(0.6)
            else:
                quality_scores.append(0.2)

        avg_quality = np.mean(quality_scores) if quality_scores else 0.5
        urgency_factors.append(avg_quality)

        # Factor 3: Time since last update
        time_factor = min(1.0, (time.time() - self.last_update_time) / 10.0)
        urgency_factors.append(time_factor)

        return np.mean(urgency_factors)

    def _get_current_parameter_value(
        self, parameter: str, current_state: SimulationState
    ) -> float:
        """Get current value of simulation parameter."""
        if parameter == "temperature":
            return current_state.process_conditions.temperature
        elif parameter == "pressure":
            return current_state.process_conditions.pressure
        elif parameter.startswith("composition_"):
            component = parameter.replace("composition_", "")
            return current_state.process_conditions.material_composition.get(
                component, 0.0
            )
        elif parameter in current_state.material_properties:
            return current_state.material_properties[parameter].value
        else:
            return 0.0

    async def _update_simulation_parameter(
        self, parameter: str, new_value: float, current_state: SimulationState
    ) -> bool:
        """Update simulation parameter."""
        try:
            if parameter == "temperature":
                # Update process conditions
                new_conditions = ProcessConditions(
                    temperature=new_value,
                    pressure=current_state.process_conditions.pressure,
                    material_composition=current_state.process_conditions.material_composition,
                )
                # Here you would update the simulation with new conditions
                # For now, we'll just log the update
                logging.info(
                    f"Updated temperature from {current_state.process_conditions.temperature:.2f} to {new_value:.2f}"
                )
                return True

            elif parameter == "pressure":
                new_conditions = ProcessConditions(
                    temperature=current_state.process_conditions.temperature,
                    pressure=new_value,
                    material_composition=current_state.process_conditions.material_composition,
                )
                logging.info(
                    f"Updated pressure from {current_state.process_conditions.pressure:.2f} to {new_value:.2f}"
                )
                return True

            elif parameter.startswith("composition_"):
                component = parameter.replace("composition_", "")
                new_composition = (
                    current_state.process_conditions.material_composition.copy()
                )
                old_value = new_composition.get(component, 0.0)
                new_composition[component] = new_value

                new_conditions = ProcessConditions(
                    temperature=current_state.process_conditions.temperature,
                    pressure=current_state.process_conditions.pressure,
                    material_composition=new_composition,
                )
                logging.info(
                    f"Updated composition {component} from {old_value:.3f} to {new_value:.3f}"
                )
                return True

            else:
                logging.warning(f"Unknown parameter for update: {parameter}")
                return False

        except Exception as e:
            logging.error(f"Failed to update parameter {parameter}: {e}")
            return False

    def add_update_callback(self, callback: Callable[[List[TwinUpdateEvent]], None]):
        """Add callback for twin update events."""
        self.update_callbacks.append(callback)

    def get_update_metrics(self) -> Dict[str, Any]:
        """Get update performance metrics."""
        return {
            **self.update_metrics,
            "update_strategy": self.update_strategy.value,
            "mapped_sensors": len(self.sensor_mappings),
            "pending_updates": len(self.pending_updates),
            "estimation_quality": self.state_estimator.get_estimation_quality(),
        }


class SensorTwinIntegrationManager:
    """Main manager for sensor-digital twin integration."""

    def __init__(
        self,
        simulation_core: RealTimeSimulationCore,
        sensor_manager: SensorDataManager,
        stream_manager: DataStreamManager,
    ):
        self.simulation_core = simulation_core
        self.sensor_manager = sensor_manager
        self.stream_manager = stream_manager

        # Initialize components
        self.twin_updater = DigitalTwinUpdater(simulation_core)
        self.predictive_controller = None  # Will be set if control is enabled

        # Integration state
        self.is_running = False
        self.integration_task = None
        self.last_sensor_data = {}

        # Performance metrics
        self.integration_metrics = {
            "sensor_twin_updates": 0,
            "control_actions_generated": 0,
            "integration_errors": 0,
            "last_integration_time": 0.0,
        }

        # Setup callbacks
        self.sensor_manager.add_data_callback(self._on_sensor_data_batch)
        self.twin_updater.add_update_callback(self._on_twin_update)

    def set_predictive_controller(self, controller: ModelPredictiveController):
        """Set predictive controller for closed-loop operation."""
        self.predictive_controller = controller

    def add_sensor_twin_mapping(self, mapping: SensorTwinMapping):
        """Add sensor-to-twin parameter mapping."""
        self.twin_updater.add_sensor_mapping(mapping)

    async def start_integration(self):
        """Start sensor-twin integration."""
        if self.is_running:
            return

        self.is_running = True

        # Start sensor data collection
        await self.sensor_manager.start_collection()

        # Start stream processing
        self.stream_manager.start_streaming()

        # Start integration loop
        self.integration_task = asyncio.create_task(self._integration_loop())

        logging.info("Started sensor-twin integration")

    async def stop_integration(self):
        """Stop sensor-twin integration."""
        if not self.is_running:
            return

        self.is_running = False

        # Stop integration loop
        if self.integration_task:
            self.integration_task.cancel()
            try:
                await self.integration_task
            except asyncio.CancelledError:
                pass

        # Stop sensor collection
        await self.sensor_manager.stop_collection()

        # Stop stream processing
        self.stream_manager.stop_streaming()

        logging.info("Stopped sensor-twin integration")

    async def _integration_loop(self):
        """Main integration loop."""
        while self.is_running:
            try:
                start_time = time.time()

                # Get latest sensor readings
                latest_readings = await self.sensor_manager.get_latest_readings()

                if latest_readings:
                    # Update digital twin from sensor data
                    update_events = await self.twin_updater.update_twin_from_sensors(
                        latest_readings
                    )

                    if update_events:
                        self.integration_metrics["sensor_twin_updates"] += len(
                            update_events
                        )
                        logging.debug(
                            f"Applied {len(update_events)} twin updates from sensors"
                        )

                    # Generate control actions if controller is available
                    if (
                        self.predictive_controller
                        and self.simulation_core.current_state
                    ):
                        control_actions = (
                            await self.predictive_controller.compute_control_actions(
                                latest_readings, self.simulation_core.current_state
                            )
                        )

                        if control_actions:
                            self.integration_metrics[
                                "control_actions_generated"
                            ] += len(control_actions)
                            await self._handle_control_actions(control_actions)

                    self.last_sensor_data = latest_readings

                self.integration_metrics["last_integration_time"] = time.time()

                # Sleep to maintain integration frequency
                elapsed = time.time() - start_time
                sleep_time = max(0.1, 1.0 - elapsed)  # 1 Hz integration
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.integration_metrics["integration_errors"] += 1
                logging.error(f"Error in integration loop: {e}")
                await asyncio.sleep(1.0)  # Wait before retrying

    def _on_sensor_data_batch(self, batch: SensorDataBatch):
        """Handle sensor data batch events."""
        # Convert batch to latest readings format
        latest_readings = {}
        for reading in batch.readings:
            latest_readings[reading.sensor_id] = reading

        # Store for integration loop
        self.last_sensor_data.update(latest_readings)

    def _on_twin_update(self, update_events: List[TwinUpdateEvent]):
        """Handle twin update events."""
        for event in update_events:
            logging.info(
                f"Twin parameter '{event.twin_parameter}' updated: "
                f"{event.old_value:.3f} -> {event.new_value:.3f} "
                f"(confidence: {event.confidence:.2f})"
            )

    async def _handle_control_actions(self, control_actions: List[ControlAction]):
        """Handle generated control actions."""
        for action in control_actions:
            if action.confidence > 0.7 and action.urgency > 0.5:
                logging.info(
                    f"High-priority control action: {action.parameter_name} = "
                    f"{action.recommended_value:.3f} (was {action.current_value:.3f})"
                )

                # In a real implementation, you would apply the control action
                # to the actual manufacturing equipment here

            elif action.confidence > 0.5:
                logging.info(
                    f"Control suggestion: {action.parameter_name} = "
                    f"{action.recommended_value:.3f} - {action.rationale}"
                )

    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and metrics."""
        return {
            "is_running": self.is_running,
            "metrics": self.integration_metrics,
            "sensor_status": self.sensor_manager.get_metrics(),
            "twin_updater_status": self.twin_updater.get_update_metrics(),
            "controller_status": (
                self.predictive_controller.get_controller_status()
                if self.predictive_controller
                else None
            ),
            "last_sensor_count": len(self.last_sensor_data),
            "integration_quality": self._calculate_integration_quality(),
        }

    def _calculate_integration_quality(self) -> float:
        """Calculate overall integration quality score."""
        factors = []

        # Sensor data quality
        if self.last_sensor_data:
            quality_scores = []
            for reading in self.last_sensor_data.values():
                if reading.quality.value == "excellent":
                    quality_scores.append(1.0)
                elif reading.quality.value == "good":
                    quality_scores.append(0.8)
                elif reading.quality.value == "acceptable":
                    quality_scores.append(0.6)
                else:
                    quality_scores.append(0.2)
            factors.append(np.mean(quality_scores))

        # Twin update success rate
        total_updates = self.integration_metrics["sensor_twin_updates"]
        if total_updates > 0:
            update_success_rate = (
                self.twin_updater.update_metrics["successful_updates"] / total_updates
            )
            factors.append(update_success_rate)

        # Integration error rate
        total_operations = max(
            1, total_updates + self.integration_metrics["control_actions_generated"]
        )
        error_rate = self.integration_metrics["integration_errors"] / total_operations
        factors.append(1.0 - min(1.0, error_rate))

        return np.mean(factors) if factors else 0.5

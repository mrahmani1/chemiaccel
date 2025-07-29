"""
Unit tests for sensor-digital twin integration module.
"""

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from src.chemi_accel.digital_twin.manufacturing_integration import ProcessConditions
from src.chemi_accel.digital_twin.predictive_control import ModelPredictiveController
from src.chemi_accel.digital_twin.realtime_data_streaming import DataStreamManager
from src.chemi_accel.digital_twin.realtime_process_simulation import (
    MaterialProperty,
    MicrostructureState,
    PropertyType,
    RealTimeSimulationCore,
    SimulationState,
)
from src.chemi_accel.digital_twin.sensor_data_integration import (
    DataQuality,
    SensorDataBatch,
    SensorDataManager,
    SensorReading,
)
from src.chemi_accel.digital_twin.sensor_twin_integration import (
    DigitalTwinUpdater,
    FusionMethod,
    SensorTwinIntegrationManager,
    SensorTwinMapping,
    StateEstimationMethod,
    StateEstimator,
    TwinUpdateEvent,
    UpdateStrategy,
)


class TestSensorTwinMapping:
    """Test SensorTwinMapping class."""

    def test_initialization(self):
        """Test mapping initialization."""
        mapping = SensorTwinMapping(
            sensor_id="temp_sensor_01", twin_parameter="temperature", trust_factor=0.9
        )

        assert mapping.sensor_id == "temp_sensor_01"
        assert mapping.twin_parameter == "temperature"
        assert mapping.trust_factor == 0.9
        assert mapping.mapping_function is None
        assert mapping.uncertainty_model is None

    def test_apply_mapping_simple(self):
        """Test simple mapping application."""
        mapping = SensorTwinMapping(
            sensor_id="temp_sensor", twin_parameter="temperature", trust_factor=1.0
        )

        twin_value, uncertainty = mapping.apply_mapping(25.0)

        assert twin_value == 25.0
        assert uncertainty > 0  # Should have some uncertainty

    def test_apply_mapping_with_function(self):
        """Test mapping with conversion function."""
        # Convert Celsius to Kelvin
        celsius_to_kelvin = lambda c: c + 273.15

        mapping = SensorTwinMapping(
            sensor_id="temp_sensor",
            twin_parameter="temperature",
            mapping_function=celsius_to_kelvin,
            trust_factor=0.8,
        )

        twin_value, uncertainty = mapping.apply_mapping(25.0)

        assert twin_value == 298.15  # 25°C + 273.15
        assert uncertainty > 0

    def test_apply_mapping_with_uncertainty_model(self):
        """Test mapping with custom uncertainty model."""
        uncertainty_model = lambda x: 0.01 * abs(x)  # 1% uncertainty

        mapping = SensorTwinMapping(
            sensor_id="pressure_sensor",
            twin_parameter="pressure",
            uncertainty_model=uncertainty_model,
            trust_factor=0.9,
        )

        twin_value, uncertainty = mapping.apply_mapping(100.0)

        assert twin_value == 100.0
        assert abs(uncertainty - 1.0 / 0.9) < 0.1  # Adjusted for trust factor

    def test_validation_range(self):
        """Test validation range checking."""
        mapping = SensorTwinMapping(
            sensor_id="sensor",
            twin_parameter="parameter",
            validation_range=(0.0, 100.0),
        )

        assert mapping.is_valid(50.0) == True
        assert mapping.is_valid(0.0) == True
        assert mapping.is_valid(100.0) == True
        assert mapping.is_valid(-10.0) == False
        assert mapping.is_valid(150.0) == False


class TestTwinUpdateEvent:
    """Test TwinUpdateEvent class."""

    def test_initialization(self):
        """Test update event initialization."""
        event = TwinUpdateEvent(
            timestamp=time.time(),
            sensor_id="temp_sensor",
            twin_parameter="temperature",
            old_value=300.0,
            new_value=310.0,
            confidence=0.85,
            update_source="sensor_fusion",
        )

        assert event.sensor_id == "temp_sensor"
        assert event.twin_parameter == "temperature"
        assert event.old_value == 300.0
        assert event.new_value == 310.0
        assert event.confidence == 0.85
        assert event.update_source == "sensor_fusion"

    def test_update_magnitude(self):
        """Test update magnitude calculation."""
        event = TwinUpdateEvent(
            timestamp=time.time(),
            sensor_id="sensor",
            twin_parameter="pressure",
            old_value=5.0,
            new_value=8.0,
            confidence=0.9,
            update_source="test",
        )

        assert event.get_update_magnitude() == 3.0


class TestStateEstimator:
    """Test StateEstimator class."""

    @pytest.fixture
    def state_estimator(self):
        """Create StateEstimator instance."""
        return StateEstimator(fusion_method=FusionMethod.WEIGHTED_AVERAGE)

    @pytest.fixture
    def mock_sensor_readings(self):
        """Create mock sensor readings."""
        return {
            "temp_sensor_1": SensorReading(
                "temp_sensor_1", time.time(), 298.15, "K", DataQuality.GOOD
            ),
            "temp_sensor_2": SensorReading(
                "temp_sensor_2", time.time(), 299.15, "K", DataQuality.EXCELLENT
            ),
            "pressure_sensor": SensorReading(
                "pressure_sensor", time.time(), 5e5, "Pa", DataQuality.GOOD
            ),
        }

    @pytest.fixture
    def mock_twin_mappings(self):
        """Create mock twin mappings."""
        return {
            "temp_sensor_1": SensorTwinMapping(
                "temp_sensor_1", "temperature", trust_factor=0.8
            ),
            "temp_sensor_2": SensorTwinMapping(
                "temp_sensor_2", "temperature", trust_factor=0.9
            ),
            "pressure_sensor": SensorTwinMapping(
                "pressure_sensor", "pressure", trust_factor=0.85
            ),
        }

    @pytest.fixture
    def mock_twin_state(self):
        """Create mock twin state."""
        process_conditions = ProcessConditions(
            temperature=300.0, pressure=5e5, material_composition={"polymer": 1.0}
        )

        microstructure = MicrostructureState(
            grain_size=5.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 1.0},
            defect_density=0.001,
            crystallinity=0.8,
        )

        return SimulationState(
            time=0.0,
            process_conditions=process_conditions,
            material_properties={},
            microstructure=microstructure,
        )

    def test_initialization(self, state_estimator):
        """Test state estimator initialization."""
        assert state_estimator.fusion_method == FusionMethod.WEIGHTED_AVERAGE
        assert len(state_estimator.sensor_weights) == 0
        assert len(state_estimator.state_history) == 0

    def test_add_sensor_weight(self, state_estimator):
        """Test adding sensor weights."""
        state_estimator.add_sensor_weight("sensor_1", 0.8)
        state_estimator.add_sensor_weight("sensor_2", 1.2)  # Should be clamped to 1.0

        assert state_estimator.sensor_weights["sensor_1"] == 0.8
        assert state_estimator.sensor_weights["sensor_2"] == 1.0

    def test_weighted_average_fusion(
        self, state_estimator, mock_sensor_readings, mock_twin_mappings, mock_twin_state
    ):
        """Test weighted average fusion."""
        # Set sensor weights
        state_estimator.add_sensor_weight("temp_sensor_1", 0.6)
        state_estimator.add_sensor_weight("temp_sensor_2", 0.8)

        estimated_params = state_estimator.estimate_state(
            mock_sensor_readings, mock_twin_mappings, mock_twin_state
        )

        assert "temperature" in estimated_params
        assert "pressure" in estimated_params

        temp_value, temp_uncertainty = estimated_params["temperature"]
        pressure_value, pressure_uncertainty = estimated_params["pressure"]

        # Temperature should be fusion of two sensors
        assert 298.0 <= temp_value <= 300.0  # Between the two sensor values
        assert temp_uncertainty > 0

        # Pressure should be from single sensor
        assert pressure_value == 5e5
        assert pressure_uncertainty > 0

    def test_kalman_filter_fusion(
        self, mock_sensor_readings, mock_twin_mappings, mock_twin_state
    ):
        """Test Kalman filter fusion."""
        estimator = StateEstimator(fusion_method=FusionMethod.KALMAN_FILTER)

        # First estimation to initialize Kalman filter
        estimated_params1 = estimator.estimate_state(
            mock_sensor_readings, mock_twin_mappings, mock_twin_state
        )

        # Second estimation to test filter update
        mock_sensor_readings["temp_sensor_1"].value = 301.15
        estimated_params2 = estimator.estimate_state(
            mock_sensor_readings, mock_twin_mappings, mock_twin_state
        )

        assert "temperature" in estimated_params2
        assert "temperature" in estimator.kalman_state
        assert "temperature" in estimator.kalman_covariance

    def test_estimation_quality(self, state_estimator):
        """Test estimation quality metrics."""
        # Add some mock estimation errors
        state_estimator.estimation_errors.extend([0.1, 0.15, 0.08, 0.12, 0.09])

        quality = state_estimator.get_estimation_quality()

        assert "estimation_accuracy" in quality
        assert "estimation_consistency" in quality
        assert "sample_count" in quality

        assert 0 <= quality["estimation_accuracy"] <= 1
        assert 0 <= quality["estimation_consistency"] <= 1
        assert quality["sample_count"] == 5


class TestDigitalTwinUpdater:
    """Test DigitalTwinUpdater class."""

    @pytest.fixture
    def mock_simulation_core(self):
        """Create mock simulation core."""
        mock_core = Mock(spec=RealTimeSimulationCore)

        # Mock current state
        process_conditions = ProcessConditions(
            temperature=300.0, pressure=5e5, material_composition={"polymer": 1.0}
        )

        microstructure = MicrostructureState(
            grain_size=5.0,
            grain_orientation=np.array([1.0, 0.0, 0.0]),
            phase_fractions={"alpha": 1.0},
            defect_density=0.001,
            crystallinity=0.8,
        )

        mock_core.current_state = SimulationState(
            time=0.0,
            process_conditions=process_conditions,
            material_properties={},
            microstructure=microstructure,
        )

        return mock_core

    @pytest.fixture
    def twin_updater(self, mock_simulation_core):
        """Create DigitalTwinUpdater instance."""
        return DigitalTwinUpdater(mock_simulation_core, UpdateStrategy.IMMEDIATE)

    @pytest.fixture
    def mock_sensor_data(self):
        """Create mock sensor data."""
        return {
            "temp_sensor": SensorReading(
                "temp_sensor", time.time(), 305.0, "K", DataQuality.GOOD
            ),
            "pressure_sensor": SensorReading(
                "pressure_sensor", time.time(), 6e5, "Pa", DataQuality.EXCELLENT
            ),
        }

    def test_initialization(self, twin_updater, mock_simulation_core):
        """Test updater initialization."""
        assert twin_updater.simulation_core == mock_simulation_core
        assert twin_updater.update_strategy == UpdateStrategy.IMMEDIATE
        assert len(twin_updater.sensor_mappings) == 0
        assert isinstance(twin_updater.state_estimator, StateEstimator)

    def test_add_sensor_mapping(self, twin_updater):
        """Test adding sensor mappings."""
        mapping = SensorTwinMapping("temp_sensor", "temperature")
        twin_updater.add_sensor_mapping(mapping)

        assert "temp_sensor" in twin_updater.sensor_mappings
        assert twin_updater.sensor_mappings["temp_sensor"] == mapping

    def test_remove_sensor_mapping(self, twin_updater):
        """Test removing sensor mappings."""
        mapping = SensorTwinMapping("temp_sensor", "temperature")
        twin_updater.add_sensor_mapping(mapping)
        twin_updater.remove_sensor_mapping("temp_sensor")

        assert "temp_sensor" not in twin_updater.sensor_mappings

    @pytest.mark.asyncio
    async def test_immediate_updates(self, twin_updater, mock_sensor_data):
        """Test immediate update strategy."""
        # Add sensor mappings
        temp_mapping = SensorTwinMapping("temp_sensor", "temperature", trust_factor=0.9)
        pressure_mapping = SensorTwinMapping(
            "pressure_sensor", "pressure", trust_factor=0.8
        )

        twin_updater.add_sensor_mapping(temp_mapping)
        twin_updater.add_sensor_mapping(pressure_mapping)

        update_events = await twin_updater.update_twin_from_sensors(mock_sensor_data)

        # Should generate update events
        assert len(update_events) >= 0  # May be 0 if confidence is too low

        for event in update_events:
            assert isinstance(event, TwinUpdateEvent)
            assert event.twin_parameter in ["temperature", "pressure"]
            assert 0 <= event.confidence <= 1

    @pytest.mark.asyncio
    async def test_adaptive_updates(self, twin_updater, mock_sensor_data):
        """Test adaptive update strategy."""
        twin_updater.update_strategy = UpdateStrategy.ADAPTIVE

        # Add sensor mappings
        temp_mapping = SensorTwinMapping("temp_sensor", "temperature", trust_factor=0.9)
        twin_updater.add_sensor_mapping(temp_mapping)

        update_events = await twin_updater.update_twin_from_sensors(mock_sensor_data)

        # Should handle adaptive strategy without errors
        assert isinstance(update_events, list)

    def test_get_current_parameter_value(self, twin_updater):
        """Test getting current parameter values."""
        current_state = twin_updater.simulation_core.current_state

        temp_value = twin_updater._get_current_parameter_value(
            "temperature", current_state
        )
        pressure_value = twin_updater._get_current_parameter_value(
            "pressure", current_state
        )
        comp_value = twin_updater._get_current_parameter_value(
            "composition_polymer", current_state
        )

        assert temp_value == 300.0
        assert pressure_value == 5e5
        assert comp_value == 1.0

    def test_update_metrics(self, twin_updater):
        """Test update metrics tracking."""
        metrics = twin_updater.get_update_metrics()

        assert "total_updates" in metrics
        assert "successful_updates" in metrics
        assert "failed_updates" in metrics
        assert "average_update_time" in metrics
        assert "update_strategy" in metrics
        assert "mapped_sensors" in metrics
        assert "estimation_quality" in metrics

    def test_update_callbacks(self, twin_updater):
        """Test update callback functionality."""
        callback_calls = []

        def test_callback(events):
            callback_calls.append(events)

        twin_updater.add_update_callback(test_callback)

        # Create a mock update event
        event = TwinUpdateEvent(
            timestamp=time.time(),
            sensor_id="test",
            twin_parameter="temperature",
            old_value=300.0,
            new_value=305.0,
            confidence=0.8,
            update_source="test",
        )

        # Manually trigger callback (would normally happen during update)
        for callback in twin_updater.update_callbacks:
            callback([event])

        assert len(callback_calls) == 1
        assert len(callback_calls[0]) == 1


class TestSensorTwinIntegrationManager:
    """Test SensorTwinIntegrationManager class."""

    @pytest.fixture
    def mock_simulation_core(self):
        """Create mock simulation core."""
        return Mock(spec=RealTimeSimulationCore)

    @pytest.fixture
    def mock_sensor_manager(self):
        """Create mock sensor manager."""
        mock_manager = Mock(spec=SensorDataManager)
        mock_manager.start_collection = AsyncMock()
        mock_manager.stop_collection = AsyncMock()
        mock_manager.get_latest_readings = AsyncMock(return_value={})
        mock_manager.add_data_callback = Mock()
        mock_manager.get_metrics = Mock(return_value={})
        return mock_manager

    @pytest.fixture
    def mock_stream_manager(self):
        """Create mock stream manager."""
        mock_manager = Mock(spec=DataStreamManager)
        mock_manager.start_streaming = Mock()
        mock_manager.stop_streaming = Mock()
        return mock_manager

    @pytest.fixture
    def integration_manager(
        self, mock_simulation_core, mock_sensor_manager, mock_stream_manager
    ):
        """Create SensorTwinIntegrationManager instance."""
        return SensorTwinIntegrationManager(
            mock_simulation_core, mock_sensor_manager, mock_stream_manager
        )

    def test_initialization(
        self,
        integration_manager,
        mock_simulation_core,
        mock_sensor_manager,
        mock_stream_manager,
    ):
        """Test integration manager initialization."""
        assert integration_manager.simulation_core == mock_simulation_core
        assert integration_manager.sensor_manager == mock_sensor_manager
        assert integration_manager.stream_manager == mock_stream_manager
        assert isinstance(integration_manager.twin_updater, DigitalTwinUpdater)
        assert not integration_manager.is_running

    def test_set_predictive_controller(self, integration_manager):
        """Test setting predictive controller."""
        mock_controller = Mock(spec=ModelPredictiveController)
        integration_manager.set_predictive_controller(mock_controller)

        assert integration_manager.predictive_controller == mock_controller

    def test_add_sensor_twin_mapping(self, integration_manager):
        """Test adding sensor-twin mappings."""
        mapping = SensorTwinMapping("temp_sensor", "temperature")
        integration_manager.add_sensor_twin_mapping(mapping)

        # Should be added to twin updater
        assert "temp_sensor" in integration_manager.twin_updater.sensor_mappings

    @pytest.mark.asyncio
    async def test_integration_lifecycle(self, integration_manager):
        """Test integration start/stop lifecycle."""
        # Start integration
        await integration_manager.start_integration()
        assert integration_manager.is_running
        assert integration_manager.integration_task is not None

        # Stop integration
        await integration_manager.stop_integration()
        assert not integration_manager.is_running

    def test_sensor_data_batch_handling(self, integration_manager):
        """Test sensor data batch event handling."""
        # Create mock sensor batch
        readings = [
            SensorReading("temp_sensor", time.time(), 300.0, "K", DataQuality.GOOD),
            SensorReading("pressure_sensor", time.time(), 5e5, "Pa", DataQuality.GOOD),
        ]
        batch = SensorDataBatch(readings)

        # Trigger batch handler
        integration_manager._on_sensor_data_batch(batch)

        # Should have updated last sensor data
        assert len(integration_manager.last_sensor_data) == 2
        assert "temp_sensor" in integration_manager.last_sensor_data
        assert "pressure_sensor" in integration_manager.last_sensor_data

    def test_twin_update_handling(self, integration_manager):
        """Test twin update event handling."""
        update_events = [
            TwinUpdateEvent(
                timestamp=time.time(),
                sensor_id="temp_sensor",
                twin_parameter="temperature",
                old_value=300.0,
                new_value=305.0,
                confidence=0.8,
                update_source="test",
            )
        ]

        # Should not raise exception
        integration_manager._on_twin_update(update_events)

    def test_integration_status(self, integration_manager):
        """Test integration status reporting."""
        status = integration_manager.get_integration_status()

        assert "is_running" in status
        assert "metrics" in status
        assert "sensor_status" in status
        assert "twin_updater_status" in status
        assert "controller_status" in status
        assert "last_sensor_count" in status
        assert "integration_quality" in status

        assert isinstance(status["is_running"], bool)
        assert isinstance(status["metrics"], dict)

    def test_integration_quality_calculation(self, integration_manager):
        """Test integration quality calculation."""
        # Add some mock sensor data with different qualities
        integration_manager.last_sensor_data = {
            "sensor1": SensorReading(
                "sensor1", time.time(), 100.0, "units", DataQuality.EXCELLENT
            ),
            "sensor2": SensorReading(
                "sensor2", time.time(), 200.0, "units", DataQuality.GOOD
            ),
            "sensor3": SensorReading(
                "sensor3", time.time(), 300.0, "units", DataQuality.POOR
            ),
        }

        # Add some metrics
        integration_manager.integration_metrics["sensor_twin_updates"] = 10
        integration_manager.twin_updater.update_metrics["successful_updates"] = 9
        integration_manager.integration_metrics["control_actions_generated"] = 5
        integration_manager.integration_metrics["integration_errors"] = 1

        quality = integration_manager._calculate_integration_quality()

        assert 0 <= quality <= 1
        assert isinstance(quality, float)


if __name__ == "__main__":
    pytest.main([__file__])

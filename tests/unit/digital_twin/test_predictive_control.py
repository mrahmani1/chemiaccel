"""
Unit tests for predictive control module.
"""

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch

from src.chemi_accel.digital_twin.manufacturing_integration import ProcessConditions
from src.chemi_accel.digital_twin.predictive_control import (
    AdaptiveController,
    ConstraintType,
    ControlAction,
    ControllerConfig,
    ControlMode,
    ControlObjective,
    ModelPredictiveController,
    NeuralProcessModel,
    ProcessConstraint,
)
from src.chemi_accel.digital_twin.realtime_process_simulation import (
    MaterialProperty,
    MicrostructureState,
    PropertyType,
    SimulationState,
)
from src.chemi_accel.digital_twin.sensor_data_integration import (
    DataQuality,
    SensorDataBatch,
    SensorReading,
)


class TestProcessConstraint:
    """Test ProcessConstraint class."""

    def test_initialization(self):
        """Test constraint initialization."""
        constraint = ProcessConstraint(
            parameter_name="temperature",
            constraint_type=ConstraintType.SAFETY,
            min_value=200.0,
            max_value=500.0,
            tolerance=5.0,
            priority=1,
        )

        assert constraint.parameter_name == "temperature"
        assert constraint.constraint_type == ConstraintType.SAFETY
        assert constraint.min_value == 200.0
        assert constraint.max_value == 500.0
        assert constraint.tolerance == 5.0
        assert constraint.priority == 1

    def test_violation_detection(self):
        """Test constraint violation detection."""
        constraint = ProcessConstraint(
            parameter_name="pressure",
            constraint_type=ConstraintType.HARD,
            min_value=1.0,
            max_value=10.0,
            tolerance=0.1,
        )

        # Valid values
        assert not constraint.is_violated(5.0)
        assert not constraint.is_violated(1.05)  # Within tolerance
        assert not constraint.is_violated(9.95)  # Within tolerance

        # Invalid values
        assert constraint.is_violated(0.8)  # Below min - tolerance
        assert constraint.is_violated(10.2)  # Above max + tolerance

    def test_violation_severity(self):
        """Test violation severity calculation."""
        constraint = ProcessConstraint(
            parameter_name="flow_rate",
            constraint_type=ConstraintType.SOFT,
            min_value=10.0,
            max_value=100.0,
            tolerance=1.0,
        )

        # No violation
        assert constraint.get_violation_severity(50.0) == 0.0
        assert constraint.get_violation_severity(10.5) == 0.0  # Within tolerance

        # Violations
        assert constraint.get_violation_severity(8.0) > 0.0  # Below min
        assert constraint.get_violation_severity(110.0) > 0.0  # Above max

        # Severity should be higher for larger violations
        mild_violation = constraint.get_violation_severity(9.0)
        severe_violation = constraint.get_violation_severity(5.0)
        assert severe_violation > mild_violation


class TestControlAction:
    """Test ControlAction class."""

    def test_initialization(self):
        """Test control action initialization."""
        action = ControlAction(
            parameter_name="temperature",
            current_value=300.0,
            recommended_value=350.0,
            confidence=0.85,
            urgency=0.6,
            rationale="Improve product quality",
        )

        assert action.parameter_name == "temperature"
        assert action.current_value == 300.0
        assert action.recommended_value == 350.0
        assert action.confidence == 0.85
        assert action.urgency == 0.6
        assert action.rationale == "Improve product quality"
        assert isinstance(action.timestamp, float)

    def test_adjustment_magnitude(self):
        """Test adjustment magnitude calculation."""
        action = ControlAction(
            parameter_name="pressure",
            current_value=5.0,
            recommended_value=8.0,
            confidence=0.9,
            urgency=0.7,
            rationale="Test",
        )

        assert action.get_adjustment_magnitude() == 3.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        action = ControlAction(
            parameter_name="flow_rate",
            current_value=50.0,
            recommended_value=55.0,
            confidence=0.8,
            urgency=0.5,
            rationale="Optimization",
            expected_benefit={"quality": 0.1},
            risk_assessment={"stability": 0.05},
        )

        action_dict = action.to_dict()

        assert action_dict["parameter_name"] == "flow_rate"
        assert action_dict["current_value"] == 50.0
        assert action_dict["recommended_value"] == 55.0
        assert action_dict["confidence"] == 0.8
        assert "timestamp" in action_dict
        assert "expected_benefit" in action_dict
        assert "risk_assessment" in action_dict


class TestControllerConfig:
    """Test ControllerConfig class."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = ControllerConfig()

        assert config.prediction_horizon == 10
        assert config.control_horizon == 5
        assert config.sample_time == 1.0
        assert ControlObjective.QUALITY_MAXIMIZATION in config.objectives
        assert config.max_iterations == 100
        assert config.enable_constraint_relaxation == True

    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        objectives = [
            ControlObjective.ENERGY_EFFICIENCY,
            ControlObjective.COST_MINIMIZATION,
        ]
        weights = {"energy_efficiency": 0.6, "cost_min": 0.4}

        config = ControllerConfig(
            prediction_horizon=20,
            control_horizon=8,
            objectives=objectives,
            objective_weights=weights,
        )

        assert config.prediction_horizon == 20
        assert config.control_horizon == 8
        assert config.objectives == objectives
        assert config.objective_weights == weights

    def test_objective_weights_initialization(self):
        """Test automatic objective weights initialization."""
        objectives = [
            ControlObjective.QUALITY_MAXIMIZATION,
            ControlObjective.ENERGY_EFFICIENCY,
            ControlObjective.DEFECT_MINIMIZATION,
        ]

        config = ControllerConfig(objectives=objectives)

        # Should have equal weights for all objectives
        assert len(config.objective_weights) == 3
        for weight in config.objective_weights.values():
            assert abs(weight - 1.0 / 3.0) < 1e-6


class TestNeuralProcessModel:
    """Test NeuralProcessModel class."""

    @pytest.fixture
    def process_model(self):
        """Create NeuralProcessModel instance."""
        return NeuralProcessModel(input_dim=5, output_dim=3, hidden_dims=[16, 8])

    def test_initialization(self, process_model):
        """Test model initialization."""
        assert process_model.input_dim == 5
        assert process_model.output_dim == 3
        assert process_model.network is not None
        assert process_model.optimizer is not None
        assert process_model.criterion is not None

    def test_predict_shape(self, process_model):
        """Test prediction output shape."""
        current_state = {"temperature": 300.0, "pressure": 5.0, "flow_rate": 50.0}
        control_inputs = np.array([1.0, 2.0])
        horizon = 5

        predictions = process_model.predict(current_state, control_inputs, horizon)

        assert predictions.shape[0] == horizon
        assert predictions.shape[1] == process_model.output_dim

    def test_update_model(self, process_model):
        """Test model updating with sensor data."""
        # Create mock sensor data
        readings = [
            SensorReading("sensor1", time.time(), 25.0, "°C", DataQuality.GOOD),
            SensorReading("sensor1", time.time() + 1, 26.0, "°C", DataQuality.GOOD),
        ]
        batch = SensorDataBatch(readings)

        # Create mock simulation state
        mock_state = Mock()
        mock_state.process_conditions.temperature = 300.0
        mock_state.process_conditions.pressure = 5e5

        # Update model (should not raise exception)
        initial_errors = len(process_model.prediction_errors)
        process_model.update_model(batch, mock_state)

        # Should have recorded prediction error
        assert len(process_model.prediction_errors) >= initial_errors

    def test_uncertainty_estimation(self, process_model):
        """Test model uncertainty estimation."""
        # Add some mock prediction errors
        process_model.prediction_errors.extend([0.1, 0.2, 0.15, 0.12, 0.18])

        uncertainty = process_model.get_model_uncertainty()

        assert "prediction_uncertainty" in uncertainty
        assert "prediction_variability" in uncertainty
        assert "confidence" in uncertainty
        assert "training_samples" in uncertainty

        assert 0 <= uncertainty["confidence"] <= 1
        assert uncertainty["training_samples"] == 5


class TestModelPredictiveController:
    """Test ModelPredictiveController class."""

    @pytest.fixture
    def controller_config(self):
        """Create controller configuration."""
        return ControllerConfig(
            prediction_horizon=5,
            control_horizon=3,
            objectives=[
                ControlObjective.QUALITY_MAXIMIZATION,
                ControlObjective.ENERGY_EFFICIENCY,
            ],
            objective_weights={"quality_max": 0.7, "energy_efficiency": 0.3},
        )

    @pytest.fixture
    def process_model(self):
        """Create mock process model."""
        mock_model = Mock()
        mock_model.predict.return_value = np.random.random((5, 3))
        mock_model.get_model_uncertainty.return_value = {
            "prediction_uncertainty": 0.1,
            "confidence": 0.9,
        }
        return mock_model

    @pytest.fixture
    def controller(self, controller_config, process_model):
        """Create ModelPredictiveController instance."""
        return ModelPredictiveController(controller_config, process_model)

    @pytest.fixture
    def mock_sensor_data(self):
        """Create mock sensor data."""
        return {
            "temp_sensor": SensorReading(
                "temp_sensor", time.time(), 350.0, "K", DataQuality.GOOD
            ),
            "pressure_sensor": SensorReading(
                "pressure_sensor", time.time(), 5e5, "Pa", DataQuality.GOOD
            ),
        }

    @pytest.fixture
    def mock_simulation_state(self):
        """Create mock simulation state."""
        process_conditions = ProcessConditions(
            temperature=350.0,
            pressure=5e5,
            material_composition={"polymer": 0.8, "additive": 0.2},
        )

        material_properties = {
            "tensile_strength": MaterialProperty(
                "tensile_strength", PropertyType.MECHANICAL, 100.0, "MPa"
            )
        }

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
            material_properties=material_properties,
            microstructure=microstructure,
        )

    def test_initialization(self, controller, controller_config, process_model):
        """Test controller initialization."""
        assert controller.config == controller_config
        assert controller.process_model == process_model
        assert len(controller.constraints) == 0
        assert len(controller.control_bounds) == 0
        assert len(controller.current_state) == 0

    def test_add_constraint(self, controller):
        """Test adding process constraints."""
        constraint = ProcessConstraint(
            parameter_name="temperature",
            constraint_type=ConstraintType.SAFETY,
            min_value=273.15,
            max_value=773.15,
        )

        controller.add_constraint(constraint)

        assert len(controller.constraints) == 1
        assert controller.constraints[0] == constraint

    def test_set_control_bounds(self, controller):
        """Test setting control parameter bounds."""
        controller.set_control_bounds("temperature", 200.0, 600.0)
        controller.set_control_bounds("pressure", 1e5, 10e5)

        assert "temperature" in controller.control_bounds
        assert "pressure" in controller.control_bounds
        assert controller.control_bounds["temperature"] == (200.0, 600.0)
        assert controller.control_bounds["pressure"] == (1e5, 10e5)

    @pytest.mark.asyncio
    async def test_compute_control_actions(
        self, controller, mock_sensor_data, mock_simulation_state
    ):
        """Test control action computation."""
        # Set some control bounds
        controller.set_control_bounds("temperature", 273.15, 773.15)
        controller.set_control_bounds("pressure", 1e5, 50e6)

        actions = await controller.compute_control_actions(
            mock_sensor_data, mock_simulation_state
        )

        assert isinstance(actions, list)
        # Should generate some control actions
        for action in actions:
            assert isinstance(action, ControlAction)
            assert action.parameter_name in [
                "temperature",
                "pressure",
            ] or action.parameter_name.startswith("composition_")
            assert 0 <= action.confidence <= 1
            assert 0 <= action.urgency <= 1

    @pytest.mark.asyncio
    async def test_temperature_optimization(
        self, controller, mock_sensor_data, mock_simulation_state
    ):
        """Test temperature control optimization."""
        controller.set_control_bounds("temperature", 273.15, 773.15)

        # Add temperature constraint
        temp_constraint = ProcessConstraint(
            parameter_name="temperature",
            constraint_type=ConstraintType.HARD,
            min_value=300.0,
            max_value=500.0,
        )
        controller.add_constraint(temp_constraint)

        action = await controller._optimize_temperature_control(
            mock_simulation_state, mock_sensor_data, 0.8
        )

        if action:  # May be None if no improvement found
            assert action.parameter_name == "temperature"
            assert 300.0 <= action.recommended_value <= 500.0  # Within constraints
            assert 0 <= action.confidence <= 1

    @pytest.mark.asyncio
    async def test_pressure_optimization(
        self, controller, mock_sensor_data, mock_simulation_state
    ):
        """Test pressure control optimization."""
        controller.set_control_bounds("pressure", 1e5, 50e6)

        action = await controller._optimize_pressure_control(
            mock_simulation_state, mock_sensor_data, 0.8
        )

        if action:  # May be None if no improvement found
            assert action.parameter_name == "pressure"
            assert 1e5 <= action.recommended_value <= 50e6  # Within bounds
            assert 0 <= action.confidence <= 1

    def test_constraint_checking(self, controller):
        """Test constraint violation checking."""
        # Add temperature constraints
        temp_constraint = ProcessConstraint(
            parameter_name="temperature",
            constraint_type=ConstraintType.SAFETY,
            min_value=273.15,
            max_value=573.15,
        )
        controller.add_constraint(temp_constraint)

        # Test valid temperature
        violation = controller._check_temperature_constraints(400.0)
        assert violation == 0.0

        # Test invalid temperature
        violation = controller._check_temperature_constraints(700.0)
        assert violation > 0.0

    def test_prediction_uncertainty(self, controller):
        """Test prediction uncertainty calculation."""
        uncertainty = controller._get_prediction_uncertainty(400.0, 350.0)

        assert 0 <= uncertainty <= 0.5  # Capped at 50%

        # Larger changes should have higher uncertainty
        small_change_uncertainty = controller._get_prediction_uncertainty(351.0, 350.0)
        large_change_uncertainty = controller._get_prediction_uncertainty(450.0, 350.0)
        assert large_change_uncertainty > small_change_uncertainty

    def test_controller_status(self, controller):
        """Test controller status reporting."""
        status = controller.get_controller_status()

        assert "config" in status
        assert "constraints" in status
        assert "recent_actions" in status
        assert "model_uncertainty" in status
        assert "current_state" in status
        assert "control_bounds" in status
        assert "last_update" in status


class TestAdaptiveController:
    """Test AdaptiveController class."""

    @pytest.fixture
    def base_controller(self):
        """Create base controller for adaptation."""
        config = ControllerConfig()
        model = Mock()
        model.get_model_uncertainty.return_value = {"confidence": 0.8}
        return ModelPredictiveController(config, model)

    @pytest.fixture
    def adaptive_controller(self, base_controller):
        """Create AdaptiveController instance."""
        return AdaptiveController(base_controller)

    def test_initialization(self, adaptive_controller, base_controller):
        """Test adaptive controller initialization."""
        assert adaptive_controller.base_controller == base_controller
        assert adaptive_controller.adaptation_rate == 0.1
        assert adaptive_controller.performance_threshold == 0.8
        assert len(adaptive_controller.adaptation_history) == 0

    @pytest.mark.asyncio
    async def test_adapt_control_strategy(self, adaptive_controller):
        """Test control strategy adaptation."""
        # Test with good performance (no adaptation needed)
        good_performance = {"quality": 0.9, "efficiency": 0.85}
        await adaptive_controller.adapt_control_strategy(good_performance)

        assert len(adaptive_controller.adaptation_history) == 1

        # Test with poor performance (adaptation needed)
        poor_performance = {"quality": 0.6, "efficiency": 0.5}
        initial_horizon = adaptive_controller.base_controller.config.prediction_horizon
        await adaptive_controller.adapt_control_strategy(poor_performance)

        assert len(adaptive_controller.adaptation_history) == 2
        # Should have adapted parameters for poor performance
        if poor_performance["quality"] < 0.7:
            assert (
                adaptive_controller.base_controller.config.prediction_horizon
                >= initial_horizon
            )

    @pytest.mark.asyncio
    async def test_parameter_adjustment(self, adaptive_controller):
        """Test controller parameter adjustment."""
        performance_details = {"quality": 0.5, "efficiency": 0.4}

        initial_horizon = adaptive_controller.base_controller.config.prediction_horizon
        initial_weights = (
            adaptive_controller.base_controller.config.objective_weights.copy()
        )

        await adaptive_controller._adjust_controller_parameters(
            0.45, performance_details
        )

        # Should have increased prediction horizon for poor quality
        assert (
            adaptive_controller.base_controller.config.prediction_horizon
            >= initial_horizon
        )

        # Should have adjusted objective weights
        for objective in adaptive_controller.base_controller.config.objectives:
            if objective == ControlObjective.QUALITY_MAXIMIZATION:
                current_weight = (
                    adaptive_controller.base_controller.config.objective_weights.get(
                        objective.value, 1.0
                    )
                )
                initial_weight = initial_weights.get(objective.value, 1.0)
                if performance_details["quality"] < 0.6:
                    assert current_weight >= initial_weight

    def test_adaptation_summary(self, adaptive_controller):
        """Test adaptation summary generation."""
        # Add some mock adaptation history
        for i in range(5):
            adaptive_controller.adaptation_history.append(
                {
                    "timestamp": time.time() - i,
                    "performance_score": 0.7 + i * 0.05,
                    "performance_details": {"quality": 0.7 + i * 0.05},
                }
            )

        summary = adaptive_controller.get_adaptation_summary()

        assert "adaptation_events" in summary
        assert "average_performance" in summary
        assert "performance_trend" in summary
        assert "last_adaptation" in summary
        assert "current_config" in summary

        assert summary["adaptation_events"] == 5
        assert 0 <= summary["average_performance"] <= 1


if __name__ == "__main__":
    pytest.main([__file__])

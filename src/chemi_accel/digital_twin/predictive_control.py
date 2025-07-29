"""
Predictive Control Module for Manufacturing Digital Twin

This module implements advanced predictive control algorithms that leverage
sensor data to update the digital twin and provide real-time process adjustments.

Key Features:
- Model Predictive Control (MPC) for multi-variable optimization
- Adaptive control algorithms that learn from sensor feedback
- Real-time parameter estimation and model updating
- Process constraint handling and safety monitoring
- Multi-objective optimization for quality, efficiency, and cost
- Uncertainty quantification in control decisions
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
import torch.nn as nn

# ChemiAccel imports
from ..core.exceptions import ChemiAccelError
from ..manufacturing_integration import ProcessConditions, ProcessOutput
from .realtime_process_simulation import MaterialProperty, SimulationState
from .sensor_data_integration import SensorDataBatch, SensorReading


class ControlObjective(Enum):
    """Control objectives for optimization."""

    QUALITY_MAXIMIZATION = "quality_max"
    COST_MINIMIZATION = "cost_min"
    ENERGY_EFFICIENCY = "energy_efficiency"
    THROUGHPUT_MAXIMIZATION = "throughput_max"
    DEFECT_MINIMIZATION = "defect_min"
    STABILITY_MAXIMIZATION = "stability_max"


class ControlMode(Enum):
    """Control operating modes."""

    AUTOMATIC = "automatic"
    SEMI_AUTOMATIC = "semi_automatic"
    MANUAL = "manual"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class ConstraintType(Enum):
    """Types of process constraints."""

    HARD = "hard"  # Must not be violated
    SOFT = "soft"  # Preferred but can be relaxed
    SAFETY = "safety"  # Critical safety constraints


@dataclass
class ProcessConstraint:
    """Process constraint definition."""

    parameter_name: str
    constraint_type: ConstraintType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    tolerance: float = 0.0
    priority: int = 1  # Higher priority = more important

    def is_violated(self, value: float) -> bool:
        """Check if constraint is violated."""
        if self.min_value is not None and value < self.min_value - self.tolerance:
            return True
        if self.max_value is not None and value > self.max_value + self.tolerance:
            return True
        return False

    def get_violation_severity(self, value: float) -> float:
        """Get severity of constraint violation (0 = no violation, 1 = maximum)."""
        if not self.is_violated(value):
            return 0.0

        severity = 0.0
        if self.min_value is not None and value < self.min_value - self.tolerance:
            severity = max(
                severity, (self.min_value - value) / max(abs(self.min_value), 1.0)
            )
        if self.max_value is not None and value > self.max_value + self.tolerance:
            severity = max(
                severity, (value - self.max_value) / max(abs(self.max_value), 1.0)
            )

        return min(1.0, severity)


@dataclass
class ControlAction:
    """Recommended control action."""

    parameter_name: str
    current_value: float
    recommended_value: float
    confidence: float  # 0-1
    urgency: float  # 0-1
    rationale: str
    expected_benefit: Dict[str, float] = field(default_factory=dict)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get_adjustment_magnitude(self) -> float:
        """Get magnitude of recommended adjustment."""
        return abs(self.recommended_value - self.current_value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ControllerConfig:
    """Configuration for predictive controller."""

    prediction_horizon: int = 10  # Time steps
    control_horizon: int = 5  # Control moves
    sample_time: float = 1.0  # seconds
    objectives: List[ControlObjective] = field(
        default_factory=lambda: [ControlObjective.QUALITY_MAXIMIZATION]
    )
    objective_weights: Dict[str, float] = field(default_factory=dict)
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    enable_constraint_relaxation: bool = True
    uncertainty_threshold: float = 0.8

    def __post_init__(self):
        """Initialize default objective weights."""
        if not self.objective_weights:
            # Equal weights for all objectives
            weight = 1.0 / len(self.objectives)
            self.objective_weights = {obj.value: weight for obj in self.objectives}


class ProcessModel(ABC):
    """Abstract base class for process models used in predictive control."""

    @abstractmethod
    def predict(
        self, current_state: Dict[str, float], control_inputs: np.ndarray, horizon: int
    ) -> np.ndarray:
        """Predict future process states."""
        pass

    @abstractmethod
    def update_model(
        self, sensor_data: SensorDataBatch, simulation_state: SimulationState
    ):
        """Update model parameters based on new data."""
        pass

    @abstractmethod
    def get_model_uncertainty(self) -> Dict[str, float]:
        """Get current model uncertainty estimates."""
        pass


class NeuralProcessModel(ProcessModel):
    """Neural network-based process model for predictive control."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build neural network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Model uncertainty estimation
        self.prediction_errors = deque(maxlen=1000)
        self.update_count = 0

    def predict(
        self, current_state: Dict[str, float], control_inputs: np.ndarray, horizon: int
    ) -> np.ndarray:
        """Predict future process states using neural network."""
        self.network.eval()

        # Convert inputs to tensor
        state_values = np.array(list(current_state.values()))
        input_tensor = (
            torch.FloatTensor(np.concatenate([state_values, control_inputs.flatten()]))
            .unsqueeze(0)
            .to(self.device)
        )

        predictions = []
        current_input = input_tensor

        with torch.no_grad():
            for _ in range(horizon):
                pred = self.network(current_input)
                predictions.append(pred.cpu().numpy())

                # Update input for next prediction (simplified - assumes control inputs change)
                next_state = pred[:, : len(state_values)]
                next_controls = control_inputs.flatten()  # Simplified
                current_input = (
                    torch.FloatTensor(
                        np.concatenate(
                            [next_state.cpu().numpy().flatten(), next_controls]
                        )
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )

        return np.array(predictions).squeeze()

    def update_model(
        self, sensor_data: SensorDataBatch, simulation_state: SimulationState
    ):
        """Update neural network based on sensor feedback."""
        if len(sensor_data.readings) < 2:
            return  # Need at least two readings for training

        self.network.train()

        # Prepare training data from sensor readings
        inputs = []
        targets = []

        for i in range(len(sensor_data.readings) - 1):
            current_reading = sensor_data.readings[i]
            next_reading = sensor_data.readings[i + 1]

            # Simple input: current sensor value + process conditions
            process_values = [
                simulation_state.process_conditions.temperature,
                simulation_state.process_conditions.pressure,
            ]

            input_vec = np.array([current_reading.value] + process_values)
            target_vec = np.array([next_reading.value])

            inputs.append(input_vec)
            targets.append(target_vec)

        if not inputs:
            return

        # Convert to tensors
        input_tensor = torch.FloatTensor(inputs).to(self.device)
        target_tensor = torch.FloatTensor(targets).to(self.device)

        # Training step
        self.optimizer.zero_grad()
        predictions = self.network(input_tensor)
        loss = self.criterion(predictions, target_tensor)
        loss.backward()
        self.optimizer.step()

        # Update uncertainty estimates
        error = torch.abs(predictions - target_tensor).mean().item()
        self.prediction_errors.append(error)
        self.update_count += 1

        if self.update_count % 10 == 0:
            logging.info(f"Neural model updated. Loss: {loss.item():.6f}")

    def get_model_uncertainty(self) -> Dict[str, float]:
        """Get model uncertainty estimates."""
        if not self.prediction_errors:
            return {"prediction_uncertainty": 1.0}

        recent_errors = list(self.prediction_errors)
        mean_error = np.mean(recent_errors)
        std_error = np.std(recent_errors)

        return {
            "prediction_uncertainty": mean_error,
            "prediction_variability": std_error,
            "confidence": max(0.0, 1.0 - mean_error),
            "training_samples": len(self.prediction_errors),
        }


class ModelPredictiveController:
    """Model Predictive Controller for multi-variable process optimization."""

    def __init__(self, config: ControllerConfig, process_model: ProcessModel):
        self.config = config
        self.process_model = process_model
        self.constraints = []
        self.current_state = {}
        self.control_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=100)

        # Optimization parameters
        self.control_bounds = {}  # parameter_name -> (min, max)
        self.state_weights = {}  # for objective function

    def add_constraint(self, constraint: ProcessConstraint):
        """Add process constraint."""
        self.constraints.append(constraint)

    def set_control_bounds(self, parameter_name: str, min_val: float, max_val: float):
        """Set bounds for control parameters."""
        self.control_bounds[parameter_name] = (min_val, max_val)

    def set_state_weights(self, weights: Dict[str, float]):
        """Set weights for state variables in objective function."""
        self.state_weights = weights

    async def compute_control_actions(
        self,
        current_sensor_data: Dict[str, SensorReading],
        simulation_state: SimulationState,
    ) -> List[ControlAction]:
        """Compute optimal control actions using MPC."""
        # Update current state from sensor data
        self.current_state = {
            sensor_id: reading.value
            for sensor_id, reading in current_sensor_data.items()
        }

        # Add process conditions to state
        self.current_state.update(
            {
                "temperature": simulation_state.process_conditions.temperature,
                "pressure": simulation_state.process_conditions.pressure,
            }
        )

        # Get model uncertainty
        model_uncertainty = self.process_model.get_model_uncertainty()
        base_confidence = model_uncertainty.get("confidence", 0.5)

        # Generate control actions for each controllable parameter
        control_actions = []

        # Temperature control
        temp_action = await self._optimize_temperature_control(
            simulation_state, current_sensor_data, base_confidence
        )
        if temp_action:
            control_actions.append(temp_action)

        # Pressure control
        pressure_action = await self._optimize_pressure_control(
            simulation_state, current_sensor_data, base_confidence
        )
        if pressure_action:
            control_actions.append(pressure_action)

        # Material composition control (if applicable)
        composition_actions = await self._optimize_composition_control(
            simulation_state, current_sensor_data, base_confidence
        )
        control_actions.extend(composition_actions)

        # Store control history
        self.control_history.append(
            {
                "timestamp": time.time(),
                "actions": [action.to_dict() for action in control_actions],
                "state": self.current_state.copy(),
                "model_uncertainty": model_uncertainty,
            }
        )

        return control_actions

    async def _optimize_temperature_control(
        self,
        simulation_state: SimulationState,
        sensor_data: Dict[str, SensorReading],
        base_confidence: float,
    ) -> Optional[ControlAction]:
        """Optimize temperature control parameter."""
        current_temp = simulation_state.process_conditions.temperature

        # Define temperature search range
        temp_bounds = self.control_bounds.get(
            "temperature", (273.15, 773.15)
        )  # 0-500°C
        temp_range = np.linspace(temp_bounds[0], temp_bounds[1], 20)

        best_temp = current_temp
        best_objective = float("-inf")
        best_rationale = "No improvement found"

        # Evaluate different temperature setpoints
        for candidate_temp in temp_range:
            if abs(candidate_temp - current_temp) < 1.0:  # Skip small changes
                continue

            # Predict outcome with this temperature
            objective_value = await self._evaluate_control_objective(
                {"temperature": candidate_temp}, simulation_state, sensor_data
            )

            # Check constraints
            constraint_violation = self._check_temperature_constraints(candidate_temp)

            if constraint_violation == 0.0 and objective_value > best_objective:
                best_objective = objective_value
                best_temp = candidate_temp
                best_rationale = f"Temperature optimization for {', '.join([obj.value for obj in self.config.objectives])}"

        # Generate control action if significant improvement
        if abs(best_temp - current_temp) > 5.0:  # Minimum 5K change
            confidence = base_confidence * (
                1.0 - self._get_prediction_uncertainty(best_temp, current_temp)
            )
            urgency = min(
                1.0, abs(best_temp - current_temp) / 100.0
            )  # Higher for larger changes

            return ControlAction(
                parameter_name="temperature",
                current_value=current_temp,
                recommended_value=best_temp,
                confidence=confidence,
                urgency=urgency,
                rationale=best_rationale,
                expected_benefit={"objective_improvement": best_objective},
                risk_assessment={"constraint_risk": 0.0},
            )

        return None

    async def _optimize_pressure_control(
        self,
        simulation_state: SimulationState,
        sensor_data: Dict[str, SensorReading],
        base_confidence: float,
    ) -> Optional[ControlAction]:
        """Optimize pressure control parameter."""
        current_pressure = simulation_state.process_conditions.pressure

        # Define pressure search range
        pressure_bounds = self.control_bounds.get(
            "pressure", (1e5, 50e6)
        )  # 1 bar to 500 bar
        pressure_range = np.linspace(pressure_bounds[0], pressure_bounds[1], 15)

        best_pressure = current_pressure
        best_objective = float("-inf")
        best_rationale = "No improvement found"

        # Evaluate different pressure setpoints
        for candidate_pressure in pressure_range:
            if abs(candidate_pressure - current_pressure) < 0.1e5:  # Skip small changes
                continue

            # Predict outcome with this pressure
            objective_value = await self._evaluate_control_objective(
                {"pressure": candidate_pressure}, simulation_state, sensor_data
            )

            # Check constraints
            constraint_violation = self._check_pressure_constraints(candidate_pressure)

            if constraint_violation == 0.0 and objective_value > best_objective:
                best_objective = objective_value
                best_pressure = candidate_pressure
                best_rationale = f"Pressure optimization for improved {', '.join([obj.value for obj in self.config.objectives])}"

        # Generate control action if significant improvement
        if abs(best_pressure - current_pressure) > 1e5:  # Minimum 1 bar change
            confidence = base_confidence * (
                1.0 - self._get_prediction_uncertainty(best_pressure, current_pressure)
            )
            urgency = min(
                1.0, abs(best_pressure - current_pressure) / 10e6
            )  # Higher for larger changes

            return ControlAction(
                parameter_name="pressure",
                current_value=current_pressure,
                recommended_value=best_pressure,
                confidence=confidence,
                urgency=urgency,
                rationale=best_rationale,
                expected_benefit={"objective_improvement": best_objective},
                risk_assessment={"constraint_risk": 0.0},
            )

        return None

    async def _optimize_composition_control(
        self,
        simulation_state: SimulationState,
        sensor_data: Dict[str, SensorReading],
        base_confidence: float,
    ) -> List[ControlAction]:
        """Optimize material composition control."""
        actions = []
        current_composition = simulation_state.process_conditions.material_composition

        # Only optimize if we have multiple components
        if len(current_composition) < 2:
            return actions

        # For demonstration, optimize the first two components
        components = list(current_composition.keys())[:2]
        if len(components) < 2:
            return actions

        comp1, comp2 = components
        current_frac1 = current_composition[comp1]
        current_frac2 = current_composition[comp2]

        # Search for optimal composition ratio
        best_frac1 = current_frac1
        best_objective = float("-inf")

        # Constrain to valid composition range
        for frac1 in np.linspace(0.1, 0.9, 9):
            frac2 = 1.0 - frac1  # Simplified two-component system

            if abs(frac1 - current_frac1) < 0.05:  # Skip small changes
                continue

            # Predict outcome with this composition
            objective_value = await self._evaluate_control_objective(
                {"composition_" + comp1: frac1, "composition_" + comp2: frac2},
                simulation_state,
                sensor_data,
            )

            if objective_value > best_objective:
                best_objective = objective_value
                best_frac1 = frac1

        # Generate actions if significant improvement
        if abs(best_frac1 - current_frac1) > 0.05:
            best_frac2 = 1.0 - best_frac1

            # Action for first component
            actions.append(
                ControlAction(
                    parameter_name=f"composition_{comp1}",
                    current_value=current_frac1,
                    recommended_value=best_frac1,
                    confidence=base_confidence
                    * 0.8,  # Lower confidence for composition changes
                    urgency=0.3,  # Composition changes are usually less urgent
                    rationale=f"Composition optimization to improve {', '.join([obj.value for obj in self.config.objectives])}",
                    expected_benefit={"objective_improvement": best_objective},
                )
            )

            # Action for second component
            actions.append(
                ControlAction(
                    parameter_name=f"composition_{comp2}",
                    current_value=current_frac2,
                    recommended_value=best_frac2,
                    confidence=base_confidence * 0.8,
                    urgency=0.3,
                    rationale=f"Composition balance for {comp1} optimization",
                    expected_benefit={"objective_improvement": best_objective},
                )
            )

        return actions

    async def _evaluate_control_objective(
        self,
        control_changes: Dict[str, float],
        simulation_state: SimulationState,
        sensor_data: Dict[str, SensorReading],
    ) -> float:
        """Evaluate control objective for given parameter changes."""
        objective_value = 0.0

        for objective in self.config.objectives:
            weight = self.config.objective_weights.get(objective.value, 1.0)

            if objective == ControlObjective.QUALITY_MAXIMIZATION:
                quality_score = self._predict_quality_impact(
                    control_changes, simulation_state
                )
                objective_value += weight * quality_score

            elif objective == ControlObjective.ENERGY_EFFICIENCY:
                efficiency_score = self._predict_efficiency_impact(
                    control_changes, simulation_state
                )
                objective_value += weight * efficiency_score

            elif objective == ControlObjective.DEFECT_MINIMIZATION:
                defect_score = self._predict_defect_impact(
                    control_changes, simulation_state
                )
                objective_value += weight * (1.0 - defect_score)  # Minimize defects

            elif objective == ControlObjective.STABILITY_MAXIMIZATION:
                stability_score = self._predict_stability_impact(
                    control_changes, sensor_data
                )
                objective_value += weight * stability_score

        return objective_value

    def _predict_quality_impact(
        self, control_changes: Dict[str, float], simulation_state: SimulationState
    ) -> float:
        """Predict impact on product quality."""
        # Simplified quality model based on process conditions
        temp_change = control_changes.get("temperature", 0.0)
        pressure_change = control_changes.get("pressure", 0.0)

        current_temp = simulation_state.process_conditions.temperature
        current_pressure = simulation_state.process_conditions.pressure

        # Optimal temperature range for quality (example: polymer processing)
        optimal_temp_range = (450.0, 550.0)  # Kelvin
        new_temp = current_temp + temp_change

        if optimal_temp_range[0] <= new_temp <= optimal_temp_range[1]:
            temp_quality = 1.0 - abs(new_temp - np.mean(optimal_temp_range)) / (
                optimal_temp_range[1] - optimal_temp_range[0]
            )
        else:
            temp_quality = 0.5  # Sub-optimal

        # Pressure impact on quality
        new_pressure = current_pressure + pressure_change
        pressure_quality = max(
            0.0, min(1.0, new_pressure / 20e6)
        )  # Normalized to 20 MPa

        return 0.6 * temp_quality + 0.4 * pressure_quality

    def _predict_efficiency_impact(
        self, control_changes: Dict[str, float], simulation_state: SimulationState
    ) -> float:
        """Predict impact on energy efficiency."""
        temp_change = control_changes.get("temperature", 0.0)
        pressure_change = control_changes.get("pressure", 0.0)

        # Higher temperatures and pressures generally reduce efficiency
        temp_efficiency = max(0.0, 1.0 - abs(temp_change) / 100.0)
        pressure_efficiency = max(0.0, 1.0 - abs(pressure_change) / 5e6)

        return 0.5 * temp_efficiency + 0.5 * pressure_efficiency

    def _predict_defect_impact(
        self, control_changes: Dict[str, float], simulation_state: SimulationState
    ) -> float:
        """Predict impact on defect formation."""
        # Extreme conditions typically increase defects
        temp_change = control_changes.get("temperature", 0.0)
        pressure_change = control_changes.get("pressure", 0.0)

        temp_defect_risk = min(1.0, abs(temp_change) / 200.0)
        pressure_defect_risk = min(1.0, abs(pressure_change) / 10e6)

        return 0.6 * temp_defect_risk + 0.4 * pressure_defect_risk

    def _predict_stability_impact(
        self, control_changes: Dict[str, float], sensor_data: Dict[str, SensorReading]
    ) -> float:
        """Predict impact on process stability."""
        # Smaller changes generally improve stability
        total_change = sum(abs(change) for change in control_changes.values())
        normalized_change = min(1.0, total_change / 1000.0)  # Normalize

        return 1.0 - normalized_change

    def _check_temperature_constraints(self, temperature: float) -> float:
        """Check temperature constraints and return violation severity."""
        max_violation = 0.0

        for constraint in self.constraints:
            if constraint.parameter_name == "temperature":
                violation = constraint.get_violation_severity(temperature)
                if constraint.constraint_type == ConstraintType.SAFETY:
                    violation *= 2.0  # Safety constraints are more important
                max_violation = max(max_violation, violation)

        return max_violation

    def _check_pressure_constraints(self, pressure: float) -> float:
        """Check pressure constraints and return violation severity."""
        max_violation = 0.0

        for constraint in self.constraints:
            if constraint.parameter_name == "pressure":
                violation = constraint.get_violation_severity(pressure)
                if constraint.constraint_type == ConstraintType.SAFETY:
                    violation *= 2.0  # Safety constraints are more important
                max_violation = max(max_violation, violation)

        return max_violation

    def _get_prediction_uncertainty(
        self, new_value: float, current_value: float
    ) -> float:
        """Get prediction uncertainty for control change."""
        # Larger changes have higher uncertainty
        change_magnitude = abs(new_value - current_value)
        relative_change = change_magnitude / max(abs(current_value), 1.0)

        return min(0.5, relative_change * 0.1)  # Cap at 50% uncertainty

    def get_controller_status(self) -> Dict[str, Any]:
        """Get current controller status and metrics."""
        recent_actions = (
            list(self.control_history)[-10:] if self.control_history else []
        )

        return {
            "config": asdict(self.config),
            "constraints": len(self.constraints),
            "recent_actions": len(recent_actions),
            "model_uncertainty": self.process_model.get_model_uncertainty(),
            "current_state": self.current_state,
            "control_bounds": self.control_bounds,
            "last_update": max([action["timestamp"] for action in recent_actions])
            if recent_actions
            else 0.0,
        }


class AdaptiveController:
    """Adaptive controller that learns and adjusts control strategy over time."""

    def __init__(self, base_controller: ModelPredictiveController):
        self.base_controller = base_controller
        self.adaptation_rate = 0.1
        self.performance_threshold = 0.8
        self.adaptation_history = deque(maxlen=100)

    async def adapt_control_strategy(self, recent_performance: Dict[str, float]):
        """Adapt control strategy based on recent performance."""
        # Calculate performance score
        performance_score = np.mean(list(recent_performance.values()))

        # Store adaptation event
        self.adaptation_history.append(
            {
                "timestamp": time.time(),
                "performance_score": performance_score,
                "performance_details": recent_performance,
            }
        )

        # Adapt if performance is below threshold
        if performance_score < self.performance_threshold:
            await self._adjust_controller_parameters(
                performance_score, recent_performance
            )

    async def _adjust_controller_parameters(
        self, performance_score: float, performance_details: Dict[str, float]
    ):
        """Adjust controller parameters based on poor performance."""
        # Increase prediction horizon if quality is poor
        if performance_details.get("quality", 1.0) < 0.7:
            self.base_controller.config.prediction_horizon = min(
                20, int(self.base_controller.config.prediction_horizon * 1.2)
            )
            logging.info(
                f"Increased prediction horizon to {self.base_controller.config.prediction_horizon}"
            )

        # Adjust objective weights based on performance
        for objective in self.base_controller.config.objectives:
            current_weight = self.base_controller.config.objective_weights.get(
                objective.value, 1.0
            )

            if (
                objective == ControlObjective.QUALITY_MAXIMIZATION
                and performance_details.get("quality", 1.0) < 0.6
            ):
                # Increase quality weight if quality is poor
                new_weight = min(2.0, current_weight * 1.2)
                self.base_controller.config.objective_weights[
                    objective.value
                ] = new_weight

            elif (
                objective == ControlObjective.ENERGY_EFFICIENCY
                and performance_details.get("efficiency", 1.0) < 0.5
            ):
                # Increase efficiency weight if efficiency is poor
                new_weight = min(2.0, current_weight * 1.1)
                self.base_controller.config.objective_weights[
                    objective.value
                ] = new_weight

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation history and current state."""
        if not self.adaptation_history:
            return {"adaptation_events": 0}

        recent_adaptations = list(self.adaptation_history)
        performance_scores = [
            event["performance_score"] for event in recent_adaptations
        ]

        return {
            "adaptation_events": len(recent_adaptations),
            "average_performance": np.mean(performance_scores),
            "performance_trend": np.polyfit(
                range(len(performance_scores)), performance_scores, 1
            )[0]
            if len(performance_scores) > 1
            else 0.0,
            "last_adaptation": recent_adaptations[-1]["timestamp"]
            if recent_adaptations
            else 0.0,
            "current_config": asdict(self.base_controller.config),
        }

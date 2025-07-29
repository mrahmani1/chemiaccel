"""
Real-time Process Simulation and Property Evolution

This module provides a comprehensive real-time simulation core for digital manufacturing twins,
enabling dynamic prediction of material properties and microstructure evolution during processing.
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..core.exceptions import ChemiAccelError
from .manufacturing_integration import ProcessConditions, ProcessOutput, ProcessType
from .multiscale_framework import LengthScale, MultiScaleFramework, TimeScale


class SimulationMode(Enum):
    """Simulation execution modes."""

    REAL_TIME = "real_time"
    ACCELERATED = "accelerated"
    STEP_BY_STEP = "step_by_step"
    BATCH = "batch"


class PropertyType(Enum):
    """Types of material properties to track."""

    MECHANICAL = "mechanical"
    THERMAL = "thermal"
    ELECTRICAL = "electrical"
    OPTICAL = "optical"
    CHEMICAL = "chemical"
    MICROSTRUCTURAL = "microstructural"


class UpdateStrategy(Enum):
    """Property update strategies."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    EVENT_DRIVEN = "event_driven"
    ADAPTIVE = "adaptive"


@dataclass
class SimulationConfig:
    """Configuration for real-time simulation."""

    mode: SimulationMode = SimulationMode.REAL_TIME
    time_step: float = 0.01  # seconds
    max_simulation_time: float = 3600.0  # 1 hour
    update_frequency: float = 10.0  # Hz
    property_update_strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE
    enable_gpu_acceleration: bool = True
    max_concurrent_simulations: int = 4
    enable_visualization: bool = True
    save_trajectory: bool = True
    convergence_tolerance: float = 1e-6
    adaptive_time_stepping: bool = True
    min_time_step: float = 1e-4
    max_time_step: float = 1.0


@dataclass
class MaterialProperty:
    """Container for material property data."""

    name: str
    property_type: PropertyType
    value: Union[float, np.ndarray]
    units: str
    uncertainty: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    spatial_distribution: Optional[np.ndarray] = None
    gradient: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "property_type": self.property_type.value,
            "value": self.value.tolist()
            if isinstance(self.value, np.ndarray)
            else self.value,
            "units": self.units,
            "uncertainty": self.uncertainty,
            "timestamp": self.timestamp,
        }


@dataclass
class MicrostructureState:
    """Microstructure state representation."""

    grain_size: float
    grain_orientation: np.ndarray
    phase_fractions: Dict[str, float]
    defect_density: float
    crystallinity: float
    molecular_orientation: Optional[np.ndarray] = None
    chain_length_distribution: Optional[np.ndarray] = None
    crosslink_density: Optional[float] = None

    def evolve(
        self, process_conditions: ProcessConditions, time_step: float
    ) -> "MicrostructureState":
        """Evolve microstructure based on process conditions."""
        # Simplified evolution model - in practice, this would use sophisticated
        # microstructure evolution equations

        # Temperature-dependent grain growth
        temperature_effect = np.exp(-5000 / (8.314 * process_conditions.temperature))
        new_grain_size = self.grain_size + 0.1 * temperature_effect * time_step

        # Pressure-dependent crystallinity for polymers
        pressure_effect = 1.0 + 0.0001 * (process_conditions.pressure - 1e5)
        new_crystallinity = min(1.0, self.crystallinity * pressure_effect)

        # Evolution of phase fractions based on cooling rate
        cooling_rate = getattr(process_conditions, "cooling_rate", 1.0)
        phase_evolution = {
            phase: fraction * (1.0 + 0.01 * cooling_rate * time_step)
            for phase, fraction in self.phase_fractions.items()
        }
        # Normalize phase fractions
        total_fraction = sum(phase_evolution.values())
        if total_fraction > 0:
            phase_evolution = {
                k: v / total_fraction for k, v in phase_evolution.items()
            }

        return MicrostructureState(
            grain_size=new_grain_size,
            grain_orientation=self.grain_orientation,
            phase_fractions=phase_evolution,
            defect_density=max(0, self.defect_density - 0.01 * time_step),
            crystallinity=new_crystallinity,
            molecular_orientation=self.molecular_orientation,
            chain_length_distribution=self.chain_length_distribution,
            crosslink_density=self.crosslink_density,
        )


@dataclass
class SimulationState:
    """Complete state of the simulation at a given time."""

    time: float
    process_conditions: ProcessConditions
    material_properties: Dict[str, MaterialProperty]
    microstructure: MicrostructureState
    temperature_field: Optional[np.ndarray] = None
    stress_field: Optional[np.ndarray] = None
    velocity_field: Optional[np.ndarray] = None
    composition_field: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "time": self.time,
            "process_conditions": self.process_conditions.__dict__,
            "material_properties": {
                k: v.to_dict() for k, v in self.material_properties.items()
            },
            "microstructure": self.microstructure.__dict__,
        }


class PropertyEvolutionModel(ABC):
    """Abstract base class for property evolution models."""

    @abstractmethod
    def predict_evolution(
        self,
        current_properties: Dict[str, MaterialProperty],
        process_conditions: ProcessConditions,
        microstructure: MicrostructureState,
        time_step: float,
    ) -> Dict[str, MaterialProperty]:
        """Predict property evolution over time step."""
        pass

    @abstractmethod
    def get_property_types(self) -> List[PropertyType]:
        """Get the property types handled by this model."""
        pass


class MechanicalPropertyModel(PropertyEvolutionModel):
    """Model for mechanical property evolution."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._setup_neural_models()

    def _setup_neural_models(self):
        """Setup neural network models for property prediction."""
        # Simplified neural network for demonstration
        self.strength_model = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        ).to(self.device)

        self.modulus_model = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        ).to(self.device)

    def predict_evolution(
        self,
        current_properties: Dict[str, MaterialProperty],
        process_conditions: ProcessConditions,
        microstructure: MicrostructureState,
        time_step: float,
    ) -> Dict[str, MaterialProperty]:
        """Predict mechanical property evolution."""

        # Extract current values
        current_strength = current_properties.get(
            "tensile_strength",
            MaterialProperty("tensile_strength", PropertyType.MECHANICAL, 100.0, "MPa"),
        )
        current_modulus = current_properties.get(
            "elastic_modulus",
            MaterialProperty("elastic_modulus", PropertyType.MECHANICAL, 2000.0, "MPa"),
        )

        # Prepare input features
        features = torch.tensor(
            [
                process_conditions.temperature,
                process_conditions.pressure,
                getattr(process_conditions, "strain_rate", 0.1),
                microstructure.grain_size,
                microstructure.crystallinity,
                microstructure.defect_density,
                current_strength.value,
                current_modulus.value,
                time_step,
                np.mean(list(microstructure.phase_fractions.values())),
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # Predict new values
        with torch.no_grad():
            strength_change = self.strength_model(features).item()
            modulus_change = self.modulus_model(features).item()

        # Apply evolution equations
        new_strength = max(0, current_strength.value + strength_change * time_step)
        new_modulus = max(0, current_modulus.value + modulus_change * time_step)

        return {
            "tensile_strength": MaterialProperty(
                "tensile_strength",
                PropertyType.MECHANICAL,
                new_strength,
                "MPa",
                uncertainty=abs(strength_change) * 0.1,
                timestamp=time.time(),
            ),
            "elastic_modulus": MaterialProperty(
                "elastic_modulus",
                PropertyType.MECHANICAL,
                new_modulus,
                "MPa",
                uncertainty=abs(modulus_change) * 0.1,
                timestamp=time.time(),
            ),
        }

    def get_property_types(self) -> List[PropertyType]:
        return [PropertyType.MECHANICAL]


class ThermalPropertyModel(PropertyEvolutionModel):
    """Model for thermal property evolution."""

    def predict_evolution(
        self,
        current_properties: Dict[str, MaterialProperty],
        process_conditions: ProcessConditions,
        microstructure: MicrostructureState,
        time_step: float,
    ) -> Dict[str, MaterialProperty]:
        """Predict thermal property evolution."""

        # Get current thermal properties
        current_conductivity = current_properties.get(
            "thermal_conductivity",
            MaterialProperty(
                "thermal_conductivity", PropertyType.THERMAL, 0.2, "W/m·K"
            ),
        )
        current_capacity = current_properties.get(
            "heat_capacity",
            MaterialProperty("heat_capacity", PropertyType.THERMAL, 1000.0, "J/kg·K"),
        )

        # Temperature-dependent evolution
        temp_factor = (
            process_conditions.temperature / 298.15
        )  # Relative to room temperature

        # Conductivity evolution (simplified model)
        conductivity_change = -0.001 * (temp_factor - 1.0) * time_step
        new_conductivity = max(0.01, current_conductivity.value + conductivity_change)

        # Heat capacity evolution
        capacity_change = 10.0 * (temp_factor - 1.0) * time_step
        new_capacity = max(500.0, current_capacity.value + capacity_change)

        return {
            "thermal_conductivity": MaterialProperty(
                "thermal_conductivity",
                PropertyType.THERMAL,
                new_conductivity,
                "W/m·K",
                uncertainty=abs(conductivity_change) * 0.1,
                timestamp=time.time(),
            ),
            "heat_capacity": MaterialProperty(
                "heat_capacity",
                PropertyType.THERMAL,
                new_capacity,
                "J/kg·K",
                uncertainty=abs(capacity_change) * 0.1,
                timestamp=time.time(),
            ),
        }

    def get_property_types(self) -> List[PropertyType]:
        return [PropertyType.THERMAL]


class RealTimeSimulationCore:
    """Core engine for real-time process simulation."""

    def __init__(
        self,
        config: SimulationConfig,
        multiscale_framework: MultiScaleFramework,
        property_models: Optional[List[PropertyEvolutionModel]] = None,
    ):
        self.config = config
        self.multiscale_framework = multiscale_framework
        self.property_models = property_models or self._create_default_models()

        # Simulation state
        self.current_state: Optional[SimulationState] = None
        self.simulation_history: List[SimulationState] = []
        self.is_running = False
        self.start_time = 0.0

        # Threading and async support
        self.update_queue: Queue = Queue()
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_simulations
        )
        self.update_callbacks: List[Callable[[SimulationState], None]] = []

        # Performance tracking
        self.performance_metrics = {
            "simulation_time": 0.0,
            "real_time": 0.0,
            "time_ratio": 0.0,
            "updates_per_second": 0.0,
            "memory_usage": 0.0,
        }

        self.logger = logging.getLogger(__name__)

    def _create_default_models(self) -> List[PropertyEvolutionModel]:
        """Create default property evolution models."""
        device = (
            "cuda"
            if self.config.enable_gpu_acceleration and torch.cuda.is_available()
            else "cpu"
        )

        return [MechanicalPropertyModel(device=device), ThermalPropertyModel()]

    def initialize_simulation(
        self,
        initial_conditions: ProcessConditions,
        initial_microstructure: MicrostructureState,
        initial_properties: Optional[Dict[str, MaterialProperty]] = None,
    ) -> SimulationState:
        """Initialize the simulation with initial conditions."""

        if initial_properties is None:
            initial_properties = self._create_default_properties()

        self.current_state = SimulationState(
            time=0.0,
            process_conditions=initial_conditions,
            material_properties=initial_properties,
            microstructure=initial_microstructure,
        )

        self.simulation_history = [self.current_state]
        self.start_time = time.time()

        self.logger.info(
            "Simulation initialized with %d property models", len(self.property_models)
        )
        return self.current_state

    def _create_default_properties(self) -> Dict[str, MaterialProperty]:
        """Create default material properties."""
        return {
            "tensile_strength": MaterialProperty(
                "tensile_strength", PropertyType.MECHANICAL, 100.0, "MPa"
            ),
            "elastic_modulus": MaterialProperty(
                "elastic_modulus", PropertyType.MECHANICAL, 2000.0, "MPa"
            ),
            "thermal_conductivity": MaterialProperty(
                "thermal_conductivity", PropertyType.THERMAL, 0.2, "W/m·K"
            ),
            "heat_capacity": MaterialProperty(
                "heat_capacity", PropertyType.THERMAL, 1000.0, "J/kg·K"
            ),
            "density": MaterialProperty(
                "density", PropertyType.MECHANICAL, 1000.0, "kg/m³"
            ),
            "viscosity": MaterialProperty(
                "viscosity", PropertyType.MECHANICAL, 0.001, "Pa·s"
            ),
        }

    async def run_simulation(
        self,
        duration: Optional[float] = None,
        process_updates: Optional[List[Tuple[float, ProcessConditions]]] = None,
    ) -> List[SimulationState]:
        """Run the real-time simulation."""

        if self.current_state is None:
            raise ChemiAccelError(
                "Simulation not initialized. Call initialize_simulation first."
            )

        self.is_running = True
        duration = duration or self.config.max_simulation_time

        # Setup process updates schedule
        update_schedule = process_updates or []
        update_index = 0

        try:
            while self.is_running and self.current_state.time < duration:
                loop_start_time = time.time()

                # Check for scheduled process updates
                if (
                    update_index < len(update_schedule)
                    and self.current_state.time >= update_schedule[update_index][0]
                ):
                    self.current_state.process_conditions = update_schedule[
                        update_index
                    ][1]
                    update_index += 1

                # Determine time step (adaptive if enabled)
                time_step = self._calculate_time_step()

                # Evolve simulation state
                next_state = await self._evolve_state(self.current_state, time_step)

                # Update current state
                self.current_state = next_state
                self.simulation_history.append(next_state)

                # Notify callbacks
                for callback in self.update_callbacks:
                    try:
                        callback(next_state)
                    except Exception as e:
                        self.logger.warning("Callback error: %s", e)

                # Performance tracking
                loop_time = time.time() - loop_start_time
                self._update_performance_metrics(loop_time, time_step)

                # Control simulation speed based on mode
                if self.config.mode == SimulationMode.REAL_TIME:
                    await asyncio.sleep(max(0, time_step - loop_time))
                elif self.config.mode == SimulationMode.STEP_BY_STEP:
                    # Wait for external signal to continue
                    await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error("Simulation error: %s", e)
            raise ChemiAccelError(f"Simulation failed: {e}")

        finally:
            self.is_running = False

        return self.simulation_history

    def _calculate_time_step(self) -> float:
        """Calculate adaptive time step based on current conditions."""
        if not self.config.adaptive_time_stepping:
            return self.config.time_step

        # Adaptive time stepping based on property change rates
        base_step = self.config.time_step

        if len(self.simulation_history) < 2:
            return base_step

        # Calculate maximum property change rate
        current_props = self.current_state.material_properties
        previous_props = self.simulation_history[-2].material_properties

        max_change_rate = 0.0
        for prop_name in current_props:
            if prop_name in previous_props:
                current_val = current_props[prop_name].value
                previous_val = previous_props[prop_name].value

                if isinstance(current_val, (int, float)) and isinstance(
                    previous_val, (int, float)
                ):
                    change_rate = abs(current_val - previous_val) / max(
                        abs(previous_val), 1e-10
                    )
                    max_change_rate = max(max_change_rate, change_rate)

        # Adapt time step based on change rate
        if max_change_rate > 0.1:  # High change rate
            adaptive_step = base_step * 0.5
        elif max_change_rate < 0.01:  # Low change rate
            adaptive_step = base_step * 2.0
        else:
            adaptive_step = base_step

        # Clamp to bounds
        return np.clip(
            adaptive_step, self.config.min_time_step, self.config.max_time_step
        )

    async def _evolve_state(
        self, current_state: SimulationState, time_step: float
    ) -> SimulationState:
        """Evolve the simulation state by one time step."""

        # Evolve microstructure
        new_microstructure = current_state.microstructure.evolve(
            current_state.process_conditions, time_step
        )

        # Evolve material properties using all models
        new_properties = current_state.material_properties.copy()

        for model in self.property_models:
            model_properties = model.predict_evolution(
                current_state.material_properties,
                current_state.process_conditions,
                current_state.microstructure,
                time_step,
            )
            new_properties.update(model_properties)

        # Create new simulation state
        new_state = SimulationState(
            time=current_state.time + time_step,
            process_conditions=current_state.process_conditions,
            material_properties=new_properties,
            microstructure=new_microstructure,
        )

        return new_state

    def _update_performance_metrics(self, loop_time: float, time_step: float):
        """Update simulation performance metrics."""
        current_real_time = time.time() - self.start_time

        self.performance_metrics.update(
            {
                "simulation_time": self.current_state.time,
                "real_time": current_real_time,
                "time_ratio": self.current_state.time / max(current_real_time, 1e-10),
                "updates_per_second": 1.0 / max(loop_time, 1e-10),
                "memory_usage": len(self.simulation_history) * 1000,  # Rough estimate
            }
        )

    def add_update_callback(self, callback: Callable[[SimulationState], None]):
        """Add callback for simulation state updates."""
        self.update_callbacks.append(callback)

    def remove_update_callback(self, callback: Callable[[SimulationState], None]):
        """Remove update callback."""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)

    def pause_simulation(self):
        """Pause the simulation."""
        self.is_running = False

    def resume_simulation(self):
        """Resume the simulation."""
        self.is_running = True

    def stop_simulation(self):
        """Stop the simulation."""
        self.is_running = False

    def get_current_state(self) -> Optional[SimulationState]:
        """Get the current simulation state."""
        return self.current_state

    def get_simulation_history(self) -> List[SimulationState]:
        """Get the complete simulation history."""
        return self.simulation_history

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def export_trajectory(self, filepath: str, format: str = "json"):
        """Export simulation trajectory to file."""
        import json

        if format == "json":
            trajectory_data = {
                "config": self.config.__dict__,
                "performance_metrics": self.performance_metrics,
                "history": [state.to_dict() for state in self.simulation_history],
            }

            with open(filepath, "w") as f:
                json.dump(trajectory_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_property_evolution(
        self, property_name: str
    ) -> Tuple[List[float], List[float]]:
        """Get evolution of specific property over time."""
        times = []
        values = []

        for state in self.simulation_history:
            if property_name in state.material_properties:
                times.append(state.time)
                prop_value = state.material_properties[property_name].value
                if isinstance(prop_value, (int, float)):
                    values.append(prop_value)
                else:
                    values.append(np.mean(prop_value))

        return times, values

    def get_microstructure_evolution(
        self, parameter: str
    ) -> Tuple[List[float], List[float]]:
        """Get evolution of microstructure parameter over time."""
        times = []
        values = []

        for state in self.simulation_history:
            times.append(state.time)
            if hasattr(state.microstructure, parameter):
                value = getattr(state.microstructure, parameter)
                if isinstance(value, (int, float)):
                    values.append(value)
                elif isinstance(value, dict):
                    values.append(sum(value.values()))
                else:
                    values.append(
                        np.mean(value) if hasattr(value, "__iter__") else float(value)
                    )

        return times, values


class SimulationMonitor:
    """Monitor and analyze real-time simulation performance."""

    def __init__(self, simulation_core: RealTimeSimulationCore):
        self.simulation_core = simulation_core
        self.performance_history: List[Dict[str, float]] = []
        self.alert_thresholds = {
            "time_ratio": 0.1,  # Simulation should be at least 10% of real-time
            "memory_usage": 1e6,  # Memory usage limit
            "updates_per_second": 1.0,  # Minimum update rate
        }

        # Register as callback
        simulation_core.add_update_callback(self._record_performance)

    def _record_performance(self, state: SimulationState):
        """Record performance metrics."""
        metrics = self.simulation_core.get_performance_metrics()
        metrics["timestamp"] = time.time()
        self.performance_history.append(metrics)

        # Check for performance issues
        self._check_performance_alerts(metrics)

    def _check_performance_alerts(self, metrics: Dict[str, float]):
        """Check for performance issues and log alerts."""
        logger = logging.getLogger(__name__)

        if metrics["time_ratio"] < self.alert_thresholds["time_ratio"]:
            logger.warning(
                "Simulation running slower than real-time: %.2fx", metrics["time_ratio"]
            )

        if metrics["memory_usage"] > self.alert_thresholds["memory_usage"]:
            logger.warning("High memory usage: %.2f MB", metrics["memory_usage"] / 1e6)

        if metrics["updates_per_second"] < self.alert_thresholds["updates_per_second"]:
            logger.warning("Low update rate: %.2f Hz", metrics["updates_per_second"])

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {"message": "No performance data available"}

        metrics = ["time_ratio", "updates_per_second", "memory_usage"]
        summary = {}

        for metric in metrics:
            values = [
                entry[metric] for entry in self.performance_history if metric in entry
            ]
            if values:
                summary[metric] = {
                    "current": values[-1],
                    "average": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std": np.std(values),
                }

        return summary

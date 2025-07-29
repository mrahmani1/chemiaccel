"""
Reinforcement Learning for Process Control in ChemiAccel.

This module provides advanced RL capabilities for chemical process optimization including:
- RL agents for multi-step synthesis optimization
- Continuous manufacturing process control
- Cost-aware and constraint-based experimental planning
- Real-time feedback loops and model updates
- GPU-accelerated training and inference
"""

import asyncio
import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical, Normal

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. RL capabilities will be limited.")

try:
    import flax.linen as jnn
    import jax
    import jax.numpy as jnp
    import optax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. Some RL backends will be disabled.")

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. GPU acceleration will be limited.")

try:
    import gymnasium as gym

    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym

        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        warnings.warn(
            "Gymnasium/OpenAI Gym not available. Some environments will be disabled."
        )

from ..core.exceptions import ChemiAccelError
from .bayesian_optimization import ParameterSpace
from .multi_objective_optimization import MultiObjectiveResult, ObjectiveConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLError(ChemiAccelError):
    """Raised when RL operations fail."""

    pass


class AgentType(Enum):
    """Available RL agent types."""

    DQN = "dqn"
    A2C = "a2c"
    PPO = "ppo"
    SAC = "sac"
    TD3 = "td3"
    MADDPG = "maddpg"  # Multi-agent


class EnvironmentType(Enum):
    """Available environment types."""

    SYNTHESIS = "synthesis"
    CONTINUOUS_MANUFACTURING = "continuous_manufacturing"
    BATCH_PROCESS = "batch_process"
    MULTI_STEP_SYNTHESIS = "multi_step_synthesis"
    COST_AWARE_DESIGN = "cost_aware_design"


class ConstraintType(Enum):
    """Available constraint types."""

    SAFETY = "safety"
    COST = "cost"
    TIME = "time"
    MATERIAL = "material"
    ENVIRONMENTAL = "environmental"
    QUALITY = "quality"
    THROUGHPUT = "throughput"


@dataclass
class ExperimentalCost:
    """Model for experimental costs."""

    material_cost: float = 0.0
    energy_cost: float = 0.0
    time_cost: float = 0.0
    equipment_cost: float = 0.0
    labor_cost: float = 0.0
    environmental_cost: float = 0.0

    def total_cost(self) -> float:
        """Calculate total experimental cost."""
        return (
            self.material_cost
            + self.energy_cost
            + self.time_cost
            + self.equipment_cost
            + self.labor_cost
            + self.environmental_cost
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "material_cost": self.material_cost,
            "energy_cost": self.energy_cost,
            "time_cost": self.time_cost,
            "equipment_cost": self.equipment_cost,
            "labor_cost": self.labor_cost,
            "environmental_cost": self.environmental_cost,
            "total_cost": self.total_cost(),
        }


@dataclass
class ProcessConstraint:
    """Process constraint definition."""

    name: str
    constraint_type: ConstraintType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    tolerance: float = 0.1
    weight: float = 1.0
    violation_penalty: float = 100.0

    def evaluate(self, value: float) -> Tuple[bool, float]:
        """Evaluate constraint satisfaction.

        Returns:
            Tuple of (is_satisfied, penalty)
        """
        penalty = 0.0
        satisfied = True

        if self.min_value is not None and value < self.min_value:
            penalty += self.violation_penalty * abs(value - self.min_value)
            satisfied = False

        if self.max_value is not None and value > self.max_value:
            penalty += self.violation_penalty * abs(value - self.max_value)
            satisfied = False

        if self.target_value is not None:
            deviation = abs(value - self.target_value)
            if deviation > self.tolerance:
                penalty += self.violation_penalty * deviation
                satisfied = False

        return satisfied, penalty * self.weight


@dataclass
class RealTimeData:
    """Real-time experimental data point."""

    timestamp: float
    experiment_id: str
    measurements: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "experiment_id": self.experiment_id,
            "measurements": self.measurements,
            "metadata": self.metadata or {},
        }


@dataclass
class RLConfig:
    """Configuration for RL agents."""

    agent_type: AgentType = AgentType.PPO
    environment_type: EnvironmentType = EnvironmentType.SYNTHESIS

    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"

    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005

    # PPO-specific
    ppo_epochs: int = 10
    ppo_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # Training schedule
    max_episodes: int = 1000
    max_steps_per_episode: int = 200
    update_frequency: int = 1000

    # GPU settings
    use_gpu: bool = True
    device: str = "auto"

    # Environment settings
    cost_weight: float = 0.1
    constraint_weight: float = 1.0
    safety_margin: float = 0.2

    # Real-time settings
    data_buffer_size: int = 10000
    model_update_frequency: int = 100
    prediction_window: int = 10


class ChemicalEnvironment(ABC):
    """Abstract base class for chemical process environments."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.state_dim = 0
        self.action_dim = 0
        self.action_space = None
        self.observation_space = None
        self.current_state = None
        self.episode_step = 0
        self.total_reward = 0.0

        self.constraints: List[ProcessConstraint] = []
        self.cost_model: Optional[Callable] = None
        self.history: List[Dict[str, Any]] = []

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action and return next state, reward, done, info."""
        pass

    @abstractmethod
    def render(self, mode: str = "human") -> Any:
        """Render environment state."""
        pass

    def add_constraint(self, constraint: ProcessConstraint):
        """Add process constraint."""
        self.constraints.append(constraint)

    def set_cost_model(self, cost_model: Callable[[np.ndarray], ExperimentalCost]):
        """Set cost evaluation model."""
        self.cost_model = cost_model

    def evaluate_constraints(
        self, measurements: Dict[str, float]
    ) -> Tuple[bool, float]:
        """Evaluate all constraints and return satisfaction status and total penalty."""
        total_penalty = 0.0
        all_satisfied = True

        for constraint in self.constraints:
            if constraint.name in measurements:
                satisfied, penalty = constraint.evaluate(measurements[constraint.name])
                total_penalty += penalty
                if not satisfied:
                    all_satisfied = False

        return all_satisfied, total_penalty

    def calculate_cost(self, action: np.ndarray) -> ExperimentalCost:
        """Calculate experimental cost for given action."""
        if self.cost_model:
            return self.cost_model(action)
        return ExperimentalCost()


class MultiStepSynthesisEnvironment(ChemicalEnvironment):
    """Environment for multi-step synthesis optimization."""

    def __init__(self, config: RLConfig, reaction_network: Optional[Dict] = None):
        super().__init__(config)

        # Define state space (concentrations, temperature, pressure, etc.)
        self.state_dim = 10  # Example: 6 concentrations + T + P + pH + time
        self.action_dim = 4  # Example: flow rates, temperature, catalyst

        self.reaction_network = reaction_network or self._default_reaction_network()
        self.target_product = "product"
        self.max_steps = 20

        # Initialize state bounds
        self.state_bounds = {
            "concentrations": (0.0, 2.0),  # M
            "temperature": (273.15, 473.15),  # K
            "pressure": (1e5, 10e5),  # Pa
            "pH": (1.0, 14.0),
            "time": (0.0, 86400.0),  # seconds
        }

        # Action bounds
        self.action_bounds = {
            "flow_rate": (0.1, 10.0),  # mL/min
            "temperature_change": (-10.0, 10.0),  # K/step
            "catalyst_addition": (0.0, 0.1),  # mol%
            "pH_adjustment": (-1.0, 1.0),  # pH units
        }

        self.reset()

    def _default_reaction_network(self) -> Dict:
        """Default reaction network for demonstration."""
        return {
            "reactions": [
                {"reactants": ["A"], "products": ["B"], "rate_constant": 0.1},
                {"reactants": ["B"], "products": ["C"], "rate_constant": 0.05},
                {"reactants": ["C"], "products": ["product"], "rate_constant": 0.08},
                {
                    "reactants": ["B", "C"],
                    "products": ["byproduct"],
                    "rate_constant": 0.02,
                },
            ],
            "initial_concentrations": {
                "A": 1.0,
                "B": 0.0,
                "C": 0.0,
                "product": 0.0,
                "byproduct": 0.0,
            },
        }

    def reset(self) -> np.ndarray:
        """Reset synthesis environment."""
        self.episode_step = 0
        self.total_reward = 0.0

        # Initialize concentrations
        initial_conc = self.reaction_network["initial_concentrations"]
        concentrations = [
            initial_conc.get(species, 0.0)
            for species in ["A", "B", "C", "product", "byproduct"]
        ]

        # Initial conditions
        self.current_state = np.array(
            [
                *concentrations,  # 5 concentrations
                298.15,  # temperature (K)
                1e5,  # pressure (Pa)
                7.0,  # pH
                0.0,  # time (s)
                0.0,  # synthesis progress
            ],
            dtype=np.float32,
        )

        self.history = []
        return self.current_state.copy()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute synthesis step."""
        self.episode_step += 1

        # Parse action
        flow_rate_change = action[0]
        temp_change = action[1]
        catalyst_addition = action[2]
        ph_adjustment = action[3]

        # Update state based on action
        new_state = self.current_state.copy()

        # Update process conditions
        new_state[5] += temp_change  # temperature
        new_state[7] += ph_adjustment  # pH
        new_state[8] += 300.0  # time increment (5 min)

        # Simulate reaction kinetics
        dt = 300.0  # time step in seconds
        concentrations = new_state[:5]
        temperature = new_state[5]

        # Temperature-dependent rate constants (Arrhenius)
        base_rates = [0.1, 0.05, 0.08, 0.02]
        rate_constants = [
            r * np.exp(-5000 / 8.314 * (1 / temperature - 1 / 298.15))
            for r in base_rates
        ]

        # Catalyst effect
        if catalyst_addition > 0:
            rate_constants = [r * (1 + 2 * catalyst_addition) for r in rate_constants]

        # pH effects
        ph_factor = np.exp(-abs(new_state[7] - 7.0) * 0.1)
        rate_constants = [r * ph_factor for r in rate_constants]

        # Reaction kinetics (simplified)
        dA_dt = -rate_constants[0] * concentrations[0]
        dB_dt = (
            rate_constants[0] * concentrations[0]
            - rate_constants[1] * concentrations[1]
            - rate_constants[3] * concentrations[1] * concentrations[2]
        )
        dC_dt = (
            rate_constants[1] * concentrations[1]
            - rate_constants[2] * concentrations[2]
            - rate_constants[3] * concentrations[1] * concentrations[2]
        )
        dP_dt = rate_constants[2] * concentrations[2]
        dBy_dt = rate_constants[3] * concentrations[1] * concentrations[2]

        # Update concentrations
        new_state[0] += dA_dt * dt
        new_state[1] += dB_dt * dt
        new_state[2] += dC_dt * dt
        new_state[3] += dP_dt * dt
        new_state[4] += dBy_dt * dt

        # Ensure non-negative concentrations
        new_state[:5] = np.maximum(new_state[:5], 0.0)

        # Update synthesis progress
        new_state[9] = new_state[3] / (
            new_state[3] + new_state[4] + 1e-8
        )  # Selectivity

        # Calculate reward
        reward = self._calculate_reward(new_state, action)

        # Check termination conditions
        done = (
            self.episode_step >= self.max_steps
            or new_state[3] > 0.8
            or new_state[4] > 0.5  # High product concentration
        )  # Too much byproduct

        # Update state
        self.current_state = new_state
        self.total_reward += reward

        # Store history
        info = {
            "product_concentration": new_state[3],
            "byproduct_concentration": new_state[4],
            "selectivity": new_state[9],
            "temperature": new_state[5],
            "pH": new_state[7],
            "step_reward": reward,
            "total_reward": self.total_reward,
        }

        self.history.append(info.copy())

        return self.current_state.copy(), reward, done, info

    def _calculate_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Calculate reward for synthesis step."""
        product_conc = state[3]
        byproduct_conc = state[4]
        selectivity = state[9]
        temperature = state[5]

        # Primary reward: product formation
        product_reward = product_conc * 10.0

        # Selectivity bonus
        selectivity_reward = selectivity * 5.0

        # Efficiency penalty (avoid extreme conditions)
        temp_penalty = (
            abs(temperature - 323.15) / 50.0
        )  # Penalty for deviation from 50°C

        # Cost penalty
        cost = self.calculate_cost(action)
        cost_penalty = cost.total_cost() * self.config.cost_weight

        # Constraint evaluation
        measurements = {"temperature": temperature, "product": product_conc}
        _, constraint_penalty = self.evaluate_constraints(measurements)
        constraint_penalty *= self.config.constraint_weight

        total_reward = (
            product_reward
            + selectivity_reward
            - temp_penalty
            - cost_penalty
            - constraint_penalty
        )

        return total_reward

    def render(self, mode: str = "human") -> Any:
        """Render synthesis state."""
        if mode == "human":
            print(f"Step: {self.episode_step}")
            print(f"Product: {self.current_state[3]:.3f} M")
            print(f"Byproduct: {self.current_state[4]:.3f} M")
            print(f"Selectivity: {self.current_state[9]:.3f}")
            print(f"Temperature: {self.current_state[5]:.1f} K")
            print(f"Total Reward: {self.total_reward:.2f}")

        return self.current_state.copy()


class ContinuousManufacturingEnvironment(ChemicalEnvironment):
    """Environment for continuous manufacturing process control."""

    def __init__(self, config: RLConfig, process_config: Optional[Dict] = None):
        super().__init__(config)

        self.state_dim = 15  # Extended state for continuous process
        self.action_dim = 6  # More control variables

        self.process_config = process_config or self._default_process_config()
        self.target_throughput = 100.0  # kg/h
        self.target_quality = 0.95

        # Process disturbances
        self.disturbance_probability = 0.1
        self.current_disturbance = None

        self.reset()

    def _default_process_config(self) -> Dict:
        """Default continuous process configuration."""
        return {
            "reactor_volume": 1000.0,  # L
            "residence_time": 3600.0,  # s
            "heat_capacity": 4184.0,  # J/kg/K
            "heat_transfer_coeff": 1000.0,  # W/m²/K
            "nominal_flow_rate": 100.0,  # L/h
            "nominal_temperature": 350.0,  # K
        }

    def reset(self) -> np.ndarray:
        """Reset continuous process."""
        self.episode_step = 0
        self.total_reward = 0.0
        self.current_disturbance = None

        # Initialize state: [flows, temperatures, concentrations, pressures, quality metrics]
        self.current_state = np.array(
            [
                100.0,  # feed_flow (L/h)
                98.0,  # product_flow (L/h)
                350.0,  # reactor_temperature (K)
                345.0,  # cooling_temperature (K)
                1.5,  # feed_concentration (M)
                0.05,  # product_concentration (M)
                2e5,  # reactor_pressure (Pa)
                1.8e5,  # system_pressure (Pa)
                0.95,  # conversion
                0.92,  # selectivity
                0.94,  # overall_quality
                0.0,  # fouling_factor
                0.0,  # vibration_level
                100.0,  # throughput (kg/h)
                0.0,  # process_time (h)
            ],
            dtype=np.float32,
        )

        self.history = []
        return self.current_state.copy()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute continuous process control step."""
        self.episode_step += 1

        # Parse control actions
        feed_flow_change = action[0] * 10.0  # Flow rate adjustment
        temp_setpoint_change = action[1] * 5.0  # Temperature setpoint
        cooling_flow_change = action[2] * 20.0  # Cooling flow
        pressure_valve_change = action[3] * 0.1  # Pressure control
        feed_conc_change = action[4] * 0.1  # Feed concentration
        residence_time_change = action[5] * 0.1  # Residence time control

        # Apply disturbances
        self._apply_disturbances()

        # Update process state
        new_state = self.current_state.copy()

        # Flow dynamics
        new_state[0] += feed_flow_change
        new_state[1] = new_state[0] * 0.98  # Slight holdup

        # Temperature dynamics
        new_state[2] += temp_setpoint_change
        new_state[3] += cooling_flow_change * 0.1

        # Concentration dynamics (simplified reactor model)
        residence_time = (
            self.process_config["reactor_volume"] / max(new_state[0], 1.0) * 3600
        )
        reaction_rate = 0.1 * np.exp(-5000 / 8.314 * (1 / new_state[2] - 1 / 350.0))

        conversion = 1 - np.exp(-reaction_rate * residence_time)
        new_state[8] = min(conversion, 0.99)

        # Update concentrations
        new_state[5] = new_state[4] * new_state[8]

        # Pressure dynamics
        new_state[6] += pressure_valve_change * 1e4
        new_state[7] = new_state[6] * 0.9

        # Quality metrics
        temp_factor = np.exp(-abs(new_state[2] - 350.0) / 20.0)
        pressure_factor = np.exp(-abs(new_state[6] - 2e5) / 5e4)
        new_state[9] = 0.95 * temp_factor * pressure_factor  # Selectivity
        new_state[10] = new_state[8] * new_state[9]  # Overall quality

        # Process degradation
        new_state[11] += 0.001  # Fouling accumulation
        new_state[12] = np.random.normal(0.0, 0.1)  # Vibration noise

        # Throughput calculation
        new_state[13] = new_state[1] * new_state[5] * 0.1  # Simplified kg/h

        # Time increment
        new_state[14] += 0.1  # 6 minutes per step

        # Apply physical constraints
        new_state = self._apply_physical_constraints(new_state)

        # Calculate reward
        reward = self._calculate_continuous_reward(new_state, action)

        # Check termination
        done = (
            self.episode_step >= 500
            or new_state[2] > 400.0  # 50 hours
            or new_state[2] < 300.0
            or new_state[6] > 5e5  # Temperature limits
            or new_state[6] < 1e5
            or new_state[11] > 0.5  # Pressure limits
        )  # Excessive fouling

        self.current_state = new_state
        self.total_reward += reward

        # Create info dictionary
        info = {
            "throughput": new_state[13],
            "quality": new_state[10],
            "conversion": new_state[8],
            "selectivity": new_state[9],
            "temperature": new_state[2],
            "pressure": new_state[6],
            "fouling": new_state[11],
            "step_reward": reward,
            "total_reward": self.total_reward,
            "disturbance": self.current_disturbance,
        }

        self.history.append(info.copy())

        return self.current_state.copy(), reward, done, info

    def _apply_disturbances(self):
        """Apply process disturbances."""
        if np.random.random() < self.disturbance_probability:
            disturbance_type = np.random.choice(
                ["feed_upset", "cooling_failure", "fouling_spike"]
            )

            if disturbance_type == "feed_upset":
                self.current_state[4] *= np.random.uniform(
                    0.8, 1.2
                )  # Feed concentration variation
                self.current_disturbance = "feed_upset"
            elif disturbance_type == "cooling_failure":
                self.current_state[3] += np.random.uniform(
                    5, 15
                )  # Cooling temperature increase
                self.current_disturbance = "cooling_failure"
            elif disturbance_type == "fouling_spike":
                self.current_state[11] += np.random.uniform(
                    0.05, 0.1
                )  # Sudden fouling increase
                self.current_disturbance = "fouling_spike"
        else:
            self.current_disturbance = None

    def _apply_physical_constraints(self, state: np.ndarray) -> np.ndarray:
        """Apply physical constraints to state."""
        constrained_state = state.copy()

        # Flow rate constraints
        constrained_state[0] = np.clip(constrained_state[0], 10.0, 500.0)
        constrained_state[1] = np.clip(constrained_state[1], 5.0, 450.0)

        # Temperature constraints
        constrained_state[2] = np.clip(constrained_state[2], 300.0, 400.0)
        constrained_state[3] = np.clip(constrained_state[3], 280.0, 380.0)

        # Concentration constraints
        constrained_state[4] = np.clip(constrained_state[4], 0.1, 3.0)
        constrained_state[5] = np.clip(constrained_state[5], 0.0, 2.5)

        # Pressure constraints
        constrained_state[6] = np.clip(constrained_state[6], 1e5, 5e5)
        constrained_state[7] = np.clip(constrained_state[7], 0.8e5, 4.5e5)

        # Quality metrics [0, 1]
        constrained_state[8:11] = np.clip(constrained_state[8:11], 0.0, 1.0)

        # Fouling factor [0, 1]
        constrained_state[11] = np.clip(constrained_state[11], 0.0, 1.0)

        return constrained_state

    def _calculate_continuous_reward(
        self, state: np.ndarray, action: np.ndarray
    ) -> float:
        """Calculate reward for continuous process."""
        throughput = state[13]
        quality = state[10]
        temperature = state[2]
        pressure = state[6]
        fouling = state[11]

        # Primary objectives
        throughput_reward = (throughput / self.target_throughput) * 10.0
        quality_reward = (quality / self.target_quality) * 8.0

        # Efficiency penalties
        temp_penalty = abs(temperature - 350.0) / 50.0
        pressure_penalty = abs(pressure - 2e5) / 1e5
        fouling_penalty = fouling * 5.0

        # Control effort penalty
        control_penalty = np.sum(np.abs(action)) * 0.1

        # Cost evaluation
        cost = self.calculate_cost(action)
        cost_penalty = cost.total_cost() * self.config.cost_weight

        # Constraint evaluation
        measurements = {
            "temperature": temperature,
            "pressure": pressure,
            "throughput": throughput,
            "quality": quality,
        }
        _, constraint_penalty = self.evaluate_constraints(measurements)
        constraint_penalty *= self.config.constraint_weight

        total_reward = (
            throughput_reward
            + quality_reward
            - temp_penalty
            - pressure_penalty
            - fouling_penalty
            - control_penalty
            - cost_penalty
            - constraint_penalty
        )

        return total_reward

    def render(self, mode: str = "human") -> Any:
        """Render continuous process state."""
        if mode == "human":
            print(f"Time: {self.current_state[14]:.1f} h")
            print(f"Throughput: {self.current_state[13]:.1f} kg/h")
            print(f"Quality: {self.current_state[10]:.3f}")
            print(f"Temperature: {self.current_state[2]:.1f} K")
            print(f"Pressure: {self.current_state[6]:.0f} Pa")
            print(f"Fouling: {self.current_state[11]:.3f}")
            if self.current_disturbance:
                print(f"Disturbance: {self.current_disturbance}")

        return self.current_state.copy()


class CostAwareAcquisitionFunction:
    """Cost-aware acquisition function for experimental design."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.cost_models: Dict[str, Callable] = {}
        self.constraint_models: Dict[str, ProcessConstraint] = {}

    def add_cost_model(self, name: str, cost_model: Callable):
        """Add cost evaluation model."""
        self.cost_models[name] = cost_model

    def add_constraint(self, constraint: ProcessConstraint):
        """Add process constraint."""
        self.constraint_models[constraint.name] = constraint

    def evaluate(
        self,
        candidates: np.ndarray,
        surrogate_mean: np.ndarray,
        surrogate_variance: np.ndarray,
        best_value: float,
    ) -> np.ndarray:
        """Evaluate cost-aware acquisition function."""
        n_candidates = len(candidates)

        # Standard Expected Improvement
        std = np.sqrt(surrogate_variance)
        z = (surrogate_mean - best_value) / (std + 1e-8)
        ei = (surrogate_mean - best_value) * self._normal_cdf(
            z
        ) + std * self._normal_pdf(z)

        # Cost penalty
        cost_penalties = np.zeros(n_candidates)
        for i, candidate in enumerate(candidates):
            total_cost = 0.0
            for cost_model in self.cost_models.values():
                cost = cost_model(candidate)
                if isinstance(cost, ExperimentalCost):
                    total_cost += cost.total_cost()
                else:
                    total_cost += float(cost)
            cost_penalties[i] = total_cost

        # Normalize cost penalties
        if cost_penalties.max() > 0:
            cost_penalties = cost_penalties / cost_penalties.max()

        # Constraint penalties
        constraint_penalties = np.zeros(n_candidates)
        for i, candidate in enumerate(candidates):
            total_penalty = 0.0
            for j, constraint in enumerate(self.constraint_models.values()):
                if j < len(candidate):  # Check if we have enough dimensions
                    _, penalty = constraint.evaluate(candidate[j])
                    total_penalty += penalty
            constraint_penalties[i] = total_penalty

        # Normalize constraint penalties
        if constraint_penalties.max() > 0:
            constraint_penalties = constraint_penalties / constraint_penalties.max()

        # Combined acquisition function
        cost_aware_ei = (
            ei
            - self.config.cost_weight * cost_penalties
            - self.config.constraint_weight * constraint_penalties
        )

        return cost_aware_ei

    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """Standard normal CDF."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


class RealTimeDataManager:
    """Manager for real-time experimental data streams."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.data_buffer: deque = deque(maxlen=config.data_buffer_size)
        self.subscribers: List[Callable] = []
        self.is_running = False
        self.update_queue = asyncio.Queue()

        # Data processing
        self.preprocessing_functions: List[Callable] = []
        self.validation_rules: List[Callable] = []

        # Model update tracking
        self.last_model_update = 0
        self.pending_updates = 0

    def add_subscriber(self, callback: Callable[[RealTimeData], Awaitable[None]]):
        """Add data subscriber."""
        self.subscribers.append(callback)

    def add_preprocessing(self, func: Callable[[RealTimeData], RealTimeData]):
        """Add data preprocessing function."""
        self.preprocessing_functions.append(func)

    def add_validation_rule(self, rule: Callable[[RealTimeData], bool]):
        """Add data validation rule."""
        self.validation_rules.append(rule)

    async def ingest_data(self, data: RealTimeData):
        """Ingest new real-time data point."""
        try:
            # Validate data
            for rule in self.validation_rules:
                if not rule(data):
                    logger.warning(f"Data validation failed for {data.experiment_id}")
                    return

            # Preprocess data
            processed_data = data
            for func in self.preprocessing_functions:
                processed_data = func(processed_data)

            # Add to buffer
            self.data_buffer.append(processed_data)

            # Notify subscribers
            for subscriber in self.subscribers:
                await subscriber(processed_data)

            # Check if model update is needed
            self.pending_updates += 1
            if self.pending_updates >= self.config.model_update_frequency:
                await self.update_queue.put("update_models")
                self.pending_updates = 0

        except Exception as e:
            logger.error(f"Error ingesting data: {e}")

    async def start_streaming(self):
        """Start real-time data streaming."""
        self.is_running = True
        logger.info("Real-time data streaming started")

        # Start update processor
        asyncio.create_task(self._process_updates())

    async def stop_streaming(self):
        """Stop real-time data streaming."""
        self.is_running = False
        logger.info("Real-time data streaming stopped")

    async def _process_updates(self):
        """Process model updates."""
        while self.is_running:
            try:
                update_type = await asyncio.wait_for(
                    self.update_queue.get(), timeout=1.0
                )

                if update_type == "update_models":
                    await self._trigger_model_updates()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing updates: {e}")

    async def _trigger_model_updates(self):
        """Trigger model updates based on recent data."""
        if len(self.data_buffer) < 10:
            return

        # Extract recent data for model updates
        recent_data = list(self.data_buffer)[-self.config.prediction_window :]

        # Convert to arrays for model training
        features = []
        targets = []

        for data_point in recent_data:
            if (
                "features" in data_point.measurements
                and "target" in data_point.measurements
            ):
                features.append(data_point.measurements["features"])
                targets.append(data_point.measurements["target"])

        if len(features) > 0:
            logger.info(f"Updating models with {len(features)} new data points")
            # Here you would trigger actual model updates
            self.last_model_update = time.time()

    def get_recent_data(self, n_points: Optional[int] = None) -> List[RealTimeData]:
        """Get recent data points."""
        if n_points is None:
            n_points = min(100, len(self.data_buffer))

        return list(self.data_buffer)[-n_points:]

    def get_data_statistics(self) -> Dict[str, Any]:
        """Get data buffer statistics."""
        if not self.data_buffer:
            return {"status": "empty"}

        recent_data = self.get_recent_data(100)

        # Calculate basic statistics
        timestamps = [d.timestamp for d in recent_data]
        data_rate = (
            len(recent_data) / max(1, timestamps[-1] - timestamps[0])
            if len(timestamps) > 1
            else 0
        )

        # Measurement statistics
        all_measurements = {}
        for data_point in recent_data:
            for key, value in data_point.measurements.items():
                if key not in all_measurements:
                    all_measurements[key] = []
                all_measurements[key].append(value)

        measurement_stats = {}
        for key, values in all_measurements.items():
            if values:
                measurement_stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values),
                }

        return {
            "buffer_size": len(self.data_buffer),
            "data_rate": data_rate,
            "last_update": self.last_model_update,
            "pending_updates": self.pending_updates,
            "measurement_stats": measurement_stats,
        }


# Neural Network Architectures for RL Agents
if TORCH_AVAILABLE:

    class PolicyNetwork(nn.Module):
        """Policy network for continuous control."""

        def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
            super().__init__()

            layers = []
            prev_size = state_dim

            for hidden_size in hidden_sizes:
                layers.extend(
                    [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(0.1)]
                )
                prev_size = hidden_size

            self.shared_layers = nn.Sequential(*layers)

            # Policy head (outputs mean and log_std for continuous actions)
            self.policy_mean = nn.Linear(prev_size, action_dim)
            self.policy_log_std = nn.Linear(prev_size, action_dim)

            # Value head
            self.value_head = nn.Linear(prev_size, 1)

        def forward(
            self, state: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass returning action mean, log_std, and state value."""
            features = self.shared_layers(state)

            action_mean = torch.tanh(self.policy_mean(features))
            action_log_std = torch.clamp(self.policy_log_std(features), -20, 2)
            state_value = self.value_head(features)

            return action_mean, action_log_std, state_value

    class QNetwork(nn.Module):
        """Q-network for value estimation."""

        def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
            super().__init__()

            layers = []
            prev_size = state_dim + action_dim

            for hidden_size in hidden_sizes:
                layers.extend(
                    [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(0.1)]
                )
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            """Forward pass returning Q-value."""
            x = torch.cat([state, action], dim=-1)
            return self.network(x)


class RLAgent(ABC):
    """Abstract base class for RL agents."""

    def __init__(self, config: RLConfig, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.device = self._setup_device()
        self.episode_count = 0
        self.step_count = 0

        # Training metrics
        self.training_metrics: Dict[str, List[float]] = defaultdict(list)

    def _setup_device(self) -> str:
        """Setup compute device."""
        if not self.config.use_gpu:
            return "cpu"

        if self.config.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            elif CUPY_AVAILABLE:
                return "cuda"

        return self.config.device if self.config.device != "auto" else "cpu"

    @abstractmethod
    async def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> np.ndarray:
        """Select action given current state."""
        pass

    @abstractmethod
    async def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update agent parameters with batch of experiences."""
        pass

    @abstractmethod
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint."""
        pass


if TORCH_AVAILABLE:

    class PPOAgent(RLAgent):
        """Proximal Policy Optimization agent."""

        def __init__(self, config: RLConfig, state_dim: int, action_dim: int):
            super().__init__(config, state_dim, action_dim)

            # Networks
            self.policy_network = PolicyNetwork(
                state_dim, action_dim, config.hidden_sizes
            ).to(self.device)

            self.optimizer = optim.Adam(
                self.policy_network.parameters(), lr=config.learning_rate
            )

            # Experience buffer
            self.buffer_size = config.buffer_size
            self.clear_buffer()

        def clear_buffer(self):
            """Clear experience buffer."""
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.log_probs = []
            self.dones = []

        async def select_action(
            self, state: np.ndarray, training: bool = True
        ) -> np.ndarray:
            """Select action using policy network."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_mean, action_log_std, state_value = self.policy_network(
                    state_tensor
                )

                if training:
                    # Sample from policy distribution
                    action_std = torch.exp(action_log_std)
                    action_dist = Normal(action_mean, action_std)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
                else:
                    # Use mean action for evaluation
                    action = action_mean
                    log_prob = None

                action = torch.clamp(action, -1.0, 1.0)

            # Store for training
            if training and log_prob is not None:
                self.states.append(state.copy())
                self.actions.append(action.squeeze(0).cpu().numpy())
                self.values.append(state_value.item())
                self.log_probs.append(log_prob.item())

            return action.squeeze(0).cpu().numpy()

        def store_reward_and_done(self, reward: float, done: bool):
            """Store reward and done flag."""
            self.rewards.append(reward)
            self.dones.append(done)

        async def update(
            self, batch: Optional[Dict[str, np.ndarray]] = None
        ) -> Dict[str, float]:
            """Update PPO agent."""
            if len(self.states) < self.config.batch_size:
                return {}

            # Calculate returns and advantages
            returns = self._calculate_returns()
            advantages = self._calculate_advantages(returns)

            # Convert to tensors
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
            old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)

            # Normalize advantages
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )

            # PPO updates
            metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}

            for epoch in range(self.config.ppo_epochs):
                # Forward pass
                action_mean, action_log_std, state_values = self.policy_network(states)

                # Calculate new log probabilities
                action_std = torch.exp(action_log_std)
                action_dist = Normal(action_mean, action_std)
                new_log_probs = action_dist.log_prob(actions).sum(dim=-1)

                # Ratio for importance sampling
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate loss
                surr1 = ratio * advantages_tensor
                surr2 = (
                    torch.clamp(
                        ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip
                    )
                    * advantages_tensor
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(state_values.squeeze(), returns_tensor)

                # Entropy loss for exploration
                entropy = action_dist.entropy().mean()
                entropy_loss = -self.config.entropy_coef * entropy

                # Total loss
                total_loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    + entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
                self.optimizer.step()

                # Store metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy_loss"] += entropy_loss.item()

            # Average metrics over epochs
            for key in metrics:
                metrics[key] /= self.config.ppo_epochs
                self.training_metrics[key].append(metrics[key])

            # Clear buffer
            self.clear_buffer()

            return metrics

        def _calculate_returns(self) -> List[float]:
            """Calculate discounted returns."""
            returns = []
            discounted_sum = 0

            for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
                if done:
                    discounted_sum = 0
                discounted_sum = reward + self.config.gamma * discounted_sum
                returns.insert(0, discounted_sum)

            return returns

        def _calculate_advantages(self, returns: List[float]) -> List[float]:
            """Calculate advantages using TD error."""
            advantages = []

            for i, (ret, value) in enumerate(zip(returns, self.values)):
                advantage = ret - value
                advantages.append(advantage)

            return advantages

        def save_checkpoint(self, filepath: str):
            """Save PPO checkpoint."""
            checkpoint = {
                "policy_network": self.policy_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "episode_count": self.episode_count,
                "step_count": self.step_count,
                "training_metrics": dict(self.training_metrics),
            }
            torch.save(checkpoint, filepath)

        def load_checkpoint(self, filepath: str):
            """Load PPO checkpoint."""
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_network.load_state_dict(checkpoint["policy_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.episode_count = checkpoint.get("episode_count", 0)
            self.step_count = checkpoint.get("step_count", 0)
            self.training_metrics = defaultdict(
                list, checkpoint.get("training_metrics", {})
            )


class ProcessControlTrainer:
    """Trainer for RL process control agents."""

    def __init__(
        self, agent: RLAgent, environment: ChemicalEnvironment, config: RLConfig
    ):
        self.agent = agent
        self.environment = environment
        self.config = config

        # Training state
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.best_reward = -np.inf

        # Real-time data integration
        self.data_manager: Optional[RealTimeDataManager] = None

    def set_data_manager(self, data_manager: RealTimeDataManager):
        """Set real-time data manager."""
        self.data_manager = data_manager

        # Subscribe to data updates
        data_manager.add_subscriber(self._process_real_time_data)

    async def _process_real_time_data(self, data: RealTimeData):
        """Process incoming real-time data."""
        # Update environment parameters based on real data
        if hasattr(self.environment, "update_from_real_data"):
            self.environment.update_from_real_data(data)

    async def train(self) -> Dict[str, Any]:
        """Train RL agent."""
        logger.info(f"Starting RL training for {self.config.max_episodes} episodes")

        training_start_time = time.time()

        for episode in range(self.config.max_episodes):
            episode_start_time = time.time()

            # Reset environment
            state = self.environment.reset()
            episode_reward = 0.0
            episode_length = 0

            # Episode loop
            for step in range(self.config.max_steps_per_episode):
                # Select action
                action = await self.agent.select_action(state, training=True)

                # Execute action
                next_state, reward, done, info = self.environment.step(action)

                # Store experience for agent
                if hasattr(self.agent, "store_reward_and_done"):
                    self.agent.store_reward_and_done(reward, done)

                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_length += 1
                self.agent.step_count += 1

                # Update agent
                if self.agent.step_count % self.config.update_frequency == 0:
                    update_metrics = await self.agent.update()

                    if update_metrics:
                        logger.debug(f"Agent update metrics: {update_metrics}")

                if done:
                    break

            # Episode completed
            self.agent.episode_count += 1
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Update best reward
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward

                # Save best model
                best_model_path = f"best_agent_{self.config.agent_type.value}.pt"
                self.agent.save_checkpoint(best_model_path)

            # Logging
            episode_time = time.time() - episode_start_time
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])

                logger.info(
                    f"Episode {episode}: "
                    f"Reward = {episode_reward:.2f}, "
                    f"Avg Reward = {avg_reward:.2f}, "
                    f"Length = {episode_length}, "
                    f"Time = {episode_time:.2f}s"
                )

        training_time = time.time() - training_start_time

        # Training summary
        training_summary = {
            "total_episodes": len(self.episode_rewards),
            "total_steps": self.agent.step_count,
            "best_reward": self.best_reward,
            "average_reward": np.mean(self.episode_rewards),
            "final_reward": self.episode_rewards[-1] if self.episode_rewards else 0.0,
            "training_time": training_time,
            "convergence_episode": self._find_convergence_episode(),
            "agent_metrics": dict(self.agent.training_metrics)
            if hasattr(self.agent, "training_metrics")
            else {},
        }

        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Best reward: {self.best_reward:.2f}")

        return training_summary

    def _find_convergence_episode(self) -> Optional[int]:
        """Find episode where training converged."""
        if len(self.episode_rewards) < 100:
            return None

        # Look for stable performance in last 100 episodes
        recent_rewards = self.episode_rewards[-100:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        # Check if performance is stable (low variance)
        if std_reward < 0.1 * abs(mean_reward):
            return len(self.episode_rewards) - 100

        return None

    async def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate trained agent."""
        logger.info(f"Evaluating agent for {num_episodes} episodes")

        eval_rewards = []
        eval_lengths = []

        for episode in range(num_episodes):
            state = self.environment.reset()
            episode_reward = 0.0
            episode_length = 0

            for step in range(self.config.max_steps_per_episode):
                # Select action (no exploration)
                action = await self.agent.select_action(state, training=False)

                # Execute action
                next_state, reward, done, info = self.environment.step(action)

                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    break

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

        evaluation_summary = {
            "num_episodes": num_episodes,
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "min_reward": np.min(eval_rewards),
            "max_reward": np.max(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "std_length": np.std(eval_lengths),
            "rewards": eval_rewards,
            "lengths": eval_lengths,
        }

        logger.info(
            f"Evaluation completed: "
            f"Mean reward = {evaluation_summary['mean_reward']:.2f} ± "
            f"{evaluation_summary['std_reward']:.2f}"
        )

        return evaluation_summary


# Utility functions for common use cases
async def train_synthesis_agent(
    reaction_network: Optional[Dict] = None,
    config: Optional[RLConfig] = None,
    constraints: Optional[List[ProcessConstraint]] = None,
) -> Tuple[RLAgent, Dict[str, Any]]:
    """Train RL agent for multi-step synthesis optimization."""

    if config is None:
        config = RLConfig(
            agent_type=AgentType.PPO,
            environment_type=EnvironmentType.MULTI_STEP_SYNTHESIS,
            max_episodes=500,
            max_steps_per_episode=100,
        )

    # Create environment
    env = MultiStepSynthesisEnvironment(config, reaction_network)

    # Add constraints
    if constraints:
        for constraint in constraints:
            env.add_constraint(constraint)

    # Create agent
    if not TORCH_AVAILABLE:
        raise RLError("PyTorch is required for RL training")

    agent = PPOAgent(config, env.state_dim, env.action_dim)

    # Create trainer
    trainer = ProcessControlTrainer(agent, env, config)

    # Train
    training_summary = await trainer.train()

    return agent, training_summary


async def train_continuous_manufacturing_agent(
    process_config: Optional[Dict] = None,
    config: Optional[RLConfig] = None,
    constraints: Optional[List[ProcessConstraint]] = None,
) -> Tuple[RLAgent, Dict[str, Any]]:
    """Train RL agent for continuous manufacturing control."""

    if config is None:
        config = RLConfig(
            agent_type=AgentType.PPO,
            environment_type=EnvironmentType.CONTINUOUS_MANUFACTURING,
            max_episodes=1000,
            max_steps_per_episode=500,
        )

    # Create environment
    env = ContinuousManufacturingEnvironment(config, process_config)

    # Add constraints
    if constraints:
        for constraint in constraints:
            env.add_constraint(constraint)

    # Create agent
    if not TORCH_AVAILABLE:
        raise RLError("PyTorch is required for RL training")

    agent = PPOAgent(config, env.state_dim, env.action_dim)

    # Create trainer
    trainer = ProcessControlTrainer(agent, env, config)

    # Train
    training_summary = await trainer.train()

    return agent, training_summary


def create_cost_aware_acquisition_function(
    cost_models: Dict[str, Callable],
    constraints: List[ProcessConstraint],
    config: Optional[RLConfig] = None,
) -> CostAwareAcquisitionFunction:
    """Create cost-aware acquisition function for experimental design."""

    if config is None:
        config = RLConfig()

    acq_func = CostAwareAcquisitionFunction(config)

    # Add cost models
    for name, model in cost_models.items():
        acq_func.add_cost_model(name, model)

    # Add constraints
    for constraint in constraints:
        acq_func.add_constraint(constraint)

    return acq_func


def create_real_time_data_manager(
    config: Optional[RLConfig] = None,
) -> RealTimeDataManager:
    """Create real-time data manager for experimental feedback."""

    if config is None:
        config = RLConfig()

    data_manager = RealTimeDataManager(config)

    # Add default preprocessing
    def standardize_measurements(data: RealTimeData) -> RealTimeData:
        """Standardize measurement keys."""
        standardized = {}
        for key, value in data.measurements.items():
            standardized_key = key.lower().replace(" ", "_").replace("-", "_")
            standardized[standardized_key] = (
                float(value) if isinstance(value, (int, float)) else value
            )

        return RealTimeData(
            timestamp=data.timestamp,
            experiment_id=data.experiment_id,
            measurements=standardized,
            metadata=data.metadata,
        )

    def validate_measurements(data: RealTimeData) -> bool:
        """Validate measurement data."""
        if not data.measurements:
            return False

        # Check for reasonable timestamps
        current_time = time.time()
        if abs(data.timestamp - current_time) > 86400:  # More than 1 day difference
            return False

        # Check for numeric measurements
        for key, value in data.measurements.items():
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    return False

        return True

    data_manager.add_preprocessing(standardize_measurements)
    data_manager.add_validation_rule(validate_measurements)

    return data_manager


# Example cost models
def material_cost_model(action: np.ndarray) -> ExperimentalCost:
    """Example material cost model."""
    # Simple linear cost model based on reagent consumption
    material_cost = np.sum(np.abs(action)) * 10.0  # $10 per unit action

    return ExperimentalCost(
        material_cost=material_cost,
        energy_cost=0.0,
        time_cost=5.0,  # $5 per step
        equipment_cost=2.0,  # $2 per step
        labor_cost=20.0,  # $20 per step
    )


def energy_cost_model(action: np.ndarray) -> ExperimentalCost:
    """Example energy cost model."""
    # Energy cost based on temperature and flow rate changes
    temp_energy = (
        abs(action[1]) * 50.0 if len(action) > 1 else 0.0
    )  # Higher temp costs more
    flow_energy = (
        abs(action[0]) * 5.0 if len(action) > 0 else 0.0
    )  # Higher flow costs more

    return ExperimentalCost(
        material_cost=0.0,
        energy_cost=temp_energy + flow_energy,
        time_cost=0.0,
        equipment_cost=1.0,  # Equipment wear
        labor_cost=0.0,
        environmental_cost=temp_energy * 0.1,  # Environmental impact
    )


# Example constraints
def create_safety_constraints() -> List[ProcessConstraint]:
    """Create common safety constraints."""
    return [
        ProcessConstraint(
            name="temperature",
            constraint_type=ConstraintType.SAFETY,
            min_value=273.15,  # 0°C
            max_value=473.15,  # 200°C
            violation_penalty=1000.0,
        ),
        ProcessConstraint(
            name="pressure",
            constraint_type=ConstraintType.SAFETY,
            min_value=1e5,  # 1 bar
            max_value=10e5,  # 10 bar
            violation_penalty=1000.0,
        ),
        ProcessConstraint(
            name="pH",
            constraint_type=ConstraintType.SAFETY,
            min_value=1.0,
            max_value=14.0,
            violation_penalty=500.0,
        ),
    ]


def create_quality_constraints() -> List[ProcessConstraint]:
    """Create quality-related constraints."""
    return [
        ProcessConstraint(
            name="purity",
            constraint_type=ConstraintType.QUALITY,
            min_value=0.95,
            target_value=0.99,
            tolerance=0.02,
            violation_penalty=200.0,
        ),
        ProcessConstraint(
            name="yield",
            constraint_type=ConstraintType.QUALITY,
            min_value=0.80,
            target_value=0.95,
            tolerance=0.05,
            violation_penalty=150.0,
        ),
    ]


def create_cost_constraints(
    max_cost_per_step: float = 100.0,
) -> List[ProcessConstraint]:
    """Create cost-related constraints."""
    return [
        ProcessConstraint(
            name="step_cost",
            constraint_type=ConstraintType.COST,
            max_value=max_cost_per_step,
            violation_penalty=50.0,
        )
    ]

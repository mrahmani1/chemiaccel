"""
Reinforcement Learning for Process Optimization

GPU-accelerated reinforcement learning agents for exploring and optimizing
manufacturing process parameter spaces. Includes state-of-the-art RL algorithms
with custom reward functions based on material properties and manufacturing objectives.

Key Features:
- GPU-accelerated RL agents (PPO, DDPG, SAC)
- Custom reward functions for material properties and manufacturing objectives
- Process parameter space exploration with adaptive sampling
- Multi-objective optimization with Pareto efficiency
- Real-time process control integration
- Hierarchical RL for multi-scale optimization
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal

from ..core.exceptions import ChemiAccelError
from .manufacturing_integration import (
    ManufacturingDigitalTwin,
    ProcessConditions,
    ProcessOutput,
    ProcessType,
)
from .multiscale_framework import MultiScaleFramework

logger = logging.getLogger(__name__)


class RLAlgorithm(Enum):
    """Supported RL algorithms."""

    PPO = "ppo"  # Proximal Policy Optimization
    DDPG = "ddpg"  # Deep Deterministic Policy Gradient
    SAC = "sac"  # Soft Actor-Critic
    TD3 = "td3"  # Twin Delayed Deep Deterministic Policy Gradient
    A3C = "a3c"  # Asynchronous Advantage Actor-Critic


class OptimizationObjective(Enum):
    """Manufacturing optimization objectives."""

    QUALITY = "quality"  # Part quality maximization
    YIELD = "yield"  # Production yield optimization
    ENERGY_EFFICIENCY = "energy_efficiency"  # Energy consumption minimization
    CYCLE_TIME = "cycle_time"  # Processing time minimization
    MATERIAL_UTILIZATION = "material_utilization"  # Material waste reduction
    MULTI_OBJECTIVE = "multi_objective"  # Pareto-optimal solutions


@dataclass
class RLConfig:
    """Configuration for RL training and optimization."""

    algorithm: RLAlgorithm = RLAlgorithm.PPO
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "relu"
    dropout_rate: float = 0.1

    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005

    # PPO specific
    ppo_epochs: int = 10
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # DDPG/SAC specific
    noise_std: float = 0.1
    target_update_frequency: int = 1

    # Training schedule
    max_episodes: int = 1000
    max_steps_per_episode: int = 500
    eval_frequency: int = 50
    save_frequency: int = 100

    # Exploration
    initial_exploration_steps: int = 1000
    exploration_decay: float = 0.99
    min_exploration: float = 0.1

    # Multi-objective
    pareto_archive_size: int = 100
    crowding_distance_weight: float = 0.1


@dataclass
class ProcessState:
    """State representation for RL environment."""

    # Process conditions
    temperature: float
    pressure: float
    flow_rate: float
    shear_rate: float
    residence_time: float

    # Material properties
    viscosity: float
    density: float
    thermal_conductivity: float

    # Current quality metrics
    part_quality: float
    dimensional_accuracy: float
    surface_finish: float

    # Historical performance
    quality_trend: List[float] = field(default_factory=list)
    energy_consumption_trend: List[float] = field(default_factory=list)

    # Multi-scale state
    atomistic_energy: float = 0.0
    mesoscopic_stress: float = 0.0
    continuum_strain: float = 0.0

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert state to tensor representation."""
        state_vector = [
            self.temperature,
            self.pressure,
            self.flow_rate,
            self.shear_rate,
            self.residence_time,
            self.viscosity,
            self.density,
            self.thermal_conductivity,
            self.part_quality,
            self.dimensional_accuracy,
            self.surface_finish,
            self.atomistic_energy,
            self.mesoscopic_stress,
            self.continuum_strain,
        ]

        # Add trend statistics
        if self.quality_trend:
            state_vector.extend(
                [
                    np.mean(self.quality_trend[-5:]),  # Recent quality average
                    (
                        np.std(self.quality_trend[-5:])
                        if len(self.quality_trend) > 1
                        else 0.0
                    ),
                    (
                        self.quality_trend[-1] - self.quality_trend[-2]
                        if len(self.quality_trend) > 1
                        else 0.0
                    ),
                ]
            )
        else:
            state_vector.extend([0.0, 0.0, 0.0])

        if self.energy_consumption_trend:
            state_vector.extend(
                [
                    np.mean(self.energy_consumption_trend[-5:]),
                    (
                        np.std(self.energy_consumption_trend[-5:])
                        if len(self.energy_consumption_trend) > 1
                        else 0.0
                    ),
                ]
            )
        else:
            state_vector.extend([0.0, 0.0])

        return torch.tensor(state_vector, dtype=torch.float32, device=device)

    @property
    def state_size(self) -> int:
        """Get state vector size."""
        return 19  # Base features + trend statistics


@dataclass
class ProcessAction:
    """Action representation for RL environment."""

    # Process parameter changes (normalized -1 to 1)
    temperature_delta: float
    pressure_delta: float
    flow_rate_delta: float
    shear_rate_delta: float
    residence_time_delta: float

    # Control signals
    heating_power: float = 0.0
    cooling_rate: float = 0.0
    pump_speed: float = 0.0

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert action to tensor representation."""
        action_vector = [
            self.temperature_delta,
            self.pressure_delta,
            self.flow_rate_delta,
            self.shear_rate_delta,
            self.residence_time_delta,
            self.heating_power,
            self.cooling_rate,
            self.pump_speed,
        ]
        return torch.tensor(action_vector, dtype=torch.float32, device=device)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "ProcessAction":
        """Create action from tensor."""
        action_data = tensor.cpu().numpy().tolist()
        return cls(
            temperature_delta=action_data[0],
            pressure_delta=action_data[1],
            flow_rate_delta=action_data[2],
            shear_rate_delta=action_data[3],
            residence_time_delta=action_data[4],
            heating_power=action_data[5] if len(action_data) > 5 else 0.0,
            cooling_rate=action_data[6] if len(action_data) > 6 else 0.0,
            pump_speed=action_data[7] if len(action_data) > 7 else 0.0,
        )

    @property
    def action_size(self) -> int:
        """Get action vector size."""
        return 8


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def calculate_reward(
        self,
        state: ProcessState,
        action: ProcessAction,
        next_state: ProcessState,
        process_output: ProcessOutput,
    ) -> float:
        """Calculate reward for state transition."""
        pass

    @abstractmethod
    def get_objectives(self) -> List[OptimizationObjective]:
        """Get optimization objectives handled by this reward function."""
        pass


class QualityRewardFunction(RewardFunction):
    """Reward function focused on part quality optimization."""

    def __init__(
        self,
        quality_weight: float = 1.0,
        dimensional_weight: float = 0.8,
        surface_weight: float = 0.6,
        target_quality: float = 0.95,
        penalty_threshold: float = 0.5,
    ):
        self.quality_weight = quality_weight
        self.dimensional_weight = dimensional_weight
        self.surface_weight = surface_weight
        self.target_quality = target_quality
        self.penalty_threshold = penalty_threshold

    def calculate_reward(
        self,
        state: ProcessState,
        action: ProcessAction,
        next_state: ProcessState,
        process_output: ProcessOutput,
    ) -> float:
        """Calculate quality-based reward."""
        # Primary quality reward
        quality_score = (
            self.quality_weight * process_output.part_quality
            + self.dimensional_weight * process_output.dimensional_accuracy
            + self.surface_weight * process_output.surface_finish
        ) / (self.quality_weight + self.dimensional_weight + self.surface_weight)

        # Bonus for reaching target quality
        target_bonus = 0.0
        if quality_score >= self.target_quality:
            target_bonus = (
                1.0
                * (quality_score - self.target_quality)
                / (1.0 - self.target_quality)
            )

        # Penalty for poor quality
        quality_penalty = 0.0
        if quality_score < self.penalty_threshold:
            quality_penalty = (
                -2.0 * (self.penalty_threshold - quality_score) / self.penalty_threshold
            )

        # Improvement reward
        improvement_reward = 0.0
        if state.part_quality > 0:
            improvement = process_output.part_quality - state.part_quality
            improvement_reward = 0.5 * improvement

        # Stability reward (penalize large fluctuations)
        stability_reward = 0.0
        if len(next_state.quality_trend) > 2:
            recent_std = np.std(next_state.quality_trend[-3:])
            stability_reward = -0.1 * recent_std

        total_reward = (
            quality_score
            + target_bonus
            + quality_penalty
            + improvement_reward
            + stability_reward
        )
        return np.clip(total_reward, -5.0, 5.0)

    def get_objectives(self) -> List[OptimizationObjective]:
        """Get quality optimization objectives."""
        return [OptimizationObjective.QUALITY]


class EnergyEfficiencyRewardFunction(RewardFunction):
    """Reward function focused on energy efficiency optimization."""

    def __init__(
        self,
        energy_weight: float = 1.0,
        quality_threshold: float = 0.7,
        baseline_energy: float = 1000.0,
        efficiency_target: float = 0.8,
    ):
        self.energy_weight = energy_weight
        self.quality_threshold = quality_threshold
        self.baseline_energy = baseline_energy
        self.efficiency_target = efficiency_target

    def calculate_reward(
        self,
        state: ProcessState,
        action: ProcessAction,
        next_state: ProcessState,
        process_output: ProcessOutput,
    ) -> float:
        """Calculate energy efficiency reward."""
        # Energy efficiency calculation
        energy_consumption = process_output.energy_consumption
        if energy_consumption <= 0:
            energy_consumption = self.baseline_energy

        efficiency = min(1.0, self.baseline_energy / energy_consumption)

        # Quality constraint
        quality_factor = 1.0
        if process_output.part_quality < self.quality_threshold:
            quality_factor = process_output.part_quality / self.quality_threshold

        # Energy savings reward
        energy_reward = self.energy_weight * efficiency * quality_factor

        # Bonus for exceeding efficiency target
        efficiency_bonus = 0.0
        if efficiency >= self.efficiency_target:
            efficiency_bonus = (
                0.5
                * (efficiency - self.efficiency_target)
                / (1.0 - self.efficiency_target)
            )

        # Penalty for excessive energy use
        energy_penalty = 0.0
        if efficiency < 0.5:
            energy_penalty = -1.0 * (0.5 - efficiency) / 0.5

        total_reward = energy_reward + efficiency_bonus + energy_penalty
        return np.clip(total_reward, -3.0, 3.0)

    def get_objectives(self) -> List[OptimizationObjective]:
        """Get energy efficiency objectives."""
        return [OptimizationObjective.ENERGY_EFFICIENCY]


class MultiObjectiveRewardFunction(RewardFunction):
    """Multi-objective reward function with Pareto optimization."""

    def __init__(
        self,
        objective_weights: Dict[OptimizationObjective, float] = None,
        pareto_weight: float = 0.3,
        diversity_weight: float = 0.1,
    ):
        if objective_weights is None:
            objective_weights = {
                OptimizationObjective.QUALITY: 0.4,
                OptimizationObjective.ENERGY_EFFICIENCY: 0.3,
                OptimizationObjective.CYCLE_TIME: 0.2,
                OptimizationObjective.YIELD: 0.1,
            }

        self.objective_weights = objective_weights
        self.pareto_weight = pareto_weight
        self.diversity_weight = diversity_weight
        self.pareto_archive = []

    def calculate_reward(
        self,
        state: ProcessState,
        action: ProcessAction,
        next_state: ProcessState,
        process_output: ProcessOutput,
    ) -> float:
        """Calculate multi-objective reward."""
        objectives = self._calculate_objectives(next_state, process_output)

        # Weighted sum reward
        weighted_reward = sum(
            weight * objectives.get(obj, 0.0)
            for obj, weight in self.objective_weights.items()
        )

        # Pareto efficiency bonus
        pareto_bonus = self._calculate_pareto_bonus(objectives)

        # Diversity bonus
        diversity_bonus = self._calculate_diversity_bonus(objectives)

        total_reward = (
            weighted_reward
            + self.pareto_weight * pareto_bonus
            + self.diversity_weight * diversity_bonus
        )

        # Update Pareto archive
        self._update_pareto_archive(objectives)

        return np.clip(total_reward, -5.0, 5.0)

    def _calculate_objectives(
        self, state: ProcessState, output: ProcessOutput
    ) -> Dict[OptimizationObjective, float]:
        """Calculate individual objective values."""
        objectives = {}

        # Quality objective
        quality_score = (
            output.part_quality + output.dimensional_accuracy + output.surface_finish
        ) / 3.0
        objectives[OptimizationObjective.QUALITY] = quality_score

        # Energy efficiency objective
        energy_consumption = (
            output.energy_consumption if output.energy_consumption > 0 else 1000.0
        )
        efficiency = min(1.0, 1000.0 / energy_consumption)
        objectives[OptimizationObjective.ENERGY_EFFICIENCY] = efficiency

        # Cycle time objective (faster is better)
        processing_time = (
            output.processing_time if output.processing_time > 0 else 100.0
        )
        cycle_time_score = min(1.0, 50.0 / processing_time)
        objectives[OptimizationObjective.CYCLE_TIME] = cycle_time_score

        # Yield objective (simplified)
        yield_score = quality_score * efficiency
        objectives[OptimizationObjective.YIELD] = yield_score

        return objectives

    def _calculate_pareto_bonus(
        self, objectives: Dict[OptimizationObjective, float]
    ) -> float:
        """Calculate Pareto efficiency bonus."""
        if not self.pareto_archive:
            return 0.5  # First solution gets bonus

        # Check if current solution dominates any in archive
        dominates_count = 0
        dominated_by_count = 0

        current_values = [
            objectives.get(obj, 0.0) for obj in self.objective_weights.keys()
        ]

        for archive_solution in self.pareto_archive:
            archive_values = [
                archive_solution.get(obj, 0.0) for obj in self.objective_weights.keys()
            ]

            if self._dominates(current_values, archive_values):
                dominates_count += 1
            elif self._dominates(archive_values, current_values):
                dominated_by_count += 1

        if dominated_by_count == 0:  # Non-dominated solution
            return 1.0 + 0.1 * dominates_count
        else:
            return -0.1 * dominated_by_count

    def _dominates(self, solution_a: List[float], solution_b: List[float]) -> bool:
        """Check if solution A dominates solution B."""
        better_in_one = False
        for a, b in zip(solution_a, solution_b):
            if a < b:  # Worse in any objective
                return False
            elif a > b:  # Better in this objective
                better_in_one = True
        return better_in_one

    def _calculate_diversity_bonus(
        self, objectives: Dict[OptimizationObjective, float]
    ) -> float:
        """Calculate diversity bonus to encourage exploration."""
        if len(self.pareto_archive) < 2:
            return 0.0

        current_values = [
            objectives.get(obj, 0.0) for obj in self.objective_weights.keys()
        ]

        # Calculate minimum distance to archive solutions
        min_distance = float("inf")
        for archive_solution in self.pareto_archive:
            archive_values = [
                archive_solution.get(obj, 0.0) for obj in self.objective_weights.keys()
            ]
            distance = np.linalg.norm(
                np.array(current_values) - np.array(archive_values)
            )
            min_distance = min(min_distance, distance)

        # Normalize and return diversity bonus
        return min(1.0, min_distance / 2.0)

    def _update_pareto_archive(self, objectives: Dict[OptimizationObjective, float]):
        """Update Pareto archive with new solution."""
        current_values = [
            objectives.get(obj, 0.0) for obj in self.objective_weights.keys()
        ]

        # Remove dominated solutions from archive
        self.pareto_archive = [
            sol
            for sol in self.pareto_archive
            if not self._dominates(
                current_values,
                [sol.get(obj, 0.0) for obj in self.objective_weights.keys()],
            )
        ]

        # Add current solution if it's not dominated
        is_dominated = any(
            self._dominates(
                [sol.get(obj, 0.0) for obj in self.objective_weights.keys()],
                current_values,
            )
            for sol in self.pareto_archive
        )

        if not is_dominated:
            self.pareto_archive.append(objectives.copy())

        # Limit archive size
        if len(self.pareto_archive) > 100:
            # Remove solutions with lowest crowding distance
            self.pareto_archive = self._select_diverse_solutions(
                self.pareto_archive, 100
            )

    def _select_diverse_solutions(
        self, solutions: List[Dict], max_size: int
    ) -> List[Dict]:
        """Select diverse solutions using crowding distance."""
        if len(solutions) <= max_size:
            return solutions

        # Calculate crowding distances
        distances = []
        for i, sol in enumerate(solutions):
            distance = 0.0
            sol_values = [sol.get(obj, 0.0) for obj in self.objective_weights.keys()]

            for j, other_sol in enumerate(solutions):
                if i != j:
                    other_values = [
                        other_sol.get(obj, 0.0) for obj in self.objective_weights.keys()
                    ]
                    distance += np.linalg.norm(
                        np.array(sol_values) - np.array(other_values)
                    )

            distances.append((distance, i))

        # Select solutions with highest crowding distances
        distances.sort(reverse=True)
        selected_indices = [idx for _, idx in distances[:max_size]]
        return [solutions[i] for i in selected_indices]

    def get_objectives(self) -> List[OptimizationObjective]:
        """Get multi-objective optimization objectives."""
        return list(self.objective_weights.keys())


class Actor(nn.Module):
    """Actor network for policy-based RL algorithms."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] = [256, 256],
        activation: str = "relu",
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        # Build network layers
        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    self.activation,
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_size = hidden_size

        # Output layers for mean and log_std
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_size, action_size)
        self.log_std_head = nn.Linear(prev_size, action_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get action distribution parameters."""
        features = self.backbone(state)
        mean = torch.tanh(self.mean_head(features))  # Bounded actions
        log_std = torch.clamp(self.log_std_head(features), -20, 2)
        return mean, log_std

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy distribution."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """Critic network for value-based RL algorithms."""

    def __init__(
        self,
        state_size: int,
        action_size: int = None,
        hidden_sizes: List[int] = [256, 256],
        activation: str = "relu",
        dropout_rate: float = 0.1,
        output_size: int = 1,
    ):
        super().__init__()

        input_size = state_size
        if action_size is not None:  # Q-function critic
            input_size += action_size

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        # Build network
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    self.activation,
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """Forward pass to get value estimate."""
        if action is not None:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms."""

    def __init__(
        self, capacity: int, state_size: int, action_size: int, device: str = "cpu"
    ):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.pos = 0

        # Pre-allocate tensors
        self.states = torch.zeros((capacity, state_size), device=device)
        self.actions = torch.zeros((capacity, action_size), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.next_states = torch.zeros((capacity, state_size), device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Add experience to buffer."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        """Get buffer size."""
        return self.size


class ProcessEnvironment:
    """RL environment for manufacturing process optimization."""

    def __init__(
        self,
        digital_twin: ManufacturingDigitalTwin,
        process_type: ProcessType,
        reward_function: RewardFunction,
        parameter_ranges: Dict[str, Tuple[float, float]],
        max_steps: int = 100,
    ):
        self.digital_twin = digital_twin
        self.process_type = process_type
        self.reward_function = reward_function
        self.parameter_ranges = parameter_ranges
        self.max_steps = max_steps

        self.current_step = 0
        self.current_state = None
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []

        # Initialize baseline conditions
        self.baseline_conditions = ProcessConditions(
            temperature=473.15,
            pressure=2e5,
            flow_rate=0.1,
            shear_rate=100.0,
            residence_time=60.0,
        )

        # Set initial state
        self.reset()

    def reset(self) -> ProcessState:
        """Reset environment to initial state."""
        self.current_step = 0
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []

        # Set random initial conditions within ranges
        initial_conditions = self._randomize_conditions(self.baseline_conditions)
        self.digital_twin.set_process_conditions(initial_conditions)

        # Run initial simulation to get baseline state
        try:
            initial_output = self.digital_twin.run_manufacturing_simulation(
                self.process_type
            )
        except Exception as e:
            logger.warning(f"Initial simulation failed: {e}. Using default output.")
            initial_output = ProcessOutput(
                part_quality=0.5,
                dimensional_accuracy=0.5,
                surface_finish=0.5,
                processing_time=100.0,
                energy_consumption=1000.0,
            )

        # Create initial state
        self.current_state = ProcessState(
            temperature=initial_conditions.temperature or 473.15,
            pressure=initial_conditions.pressure or 2e5,
            flow_rate=initial_conditions.flow_rate or 0.1,
            shear_rate=initial_conditions.shear_rate or 100.0,
            residence_time=initial_conditions.residence_time or 60.0,
            viscosity=1000.0,  # Default material properties
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=initial_output.part_quality,
            dimensional_accuracy=initial_output.dimensional_accuracy,
            surface_finish=initial_output.surface_finish,
            quality_trend=[initial_output.part_quality],
            energy_consumption_trend=[initial_output.energy_consumption],
        )

        return self.current_state

    def step(self, action: ProcessAction) -> Tuple[ProcessState, float, bool, Dict]:
        """Take environment step with given action."""
        self.current_step += 1

        # Apply action to update process conditions
        new_conditions = self._apply_action(action)
        self.digital_twin.set_process_conditions(new_conditions)

        # Run simulation
        try:
            process_output = self.digital_twin.run_manufacturing_simulation(
                self.process_type
            )
        except Exception as e:
            logger.warning(f"Simulation failed: {e}. Using penalty output.")
            process_output = ProcessOutput(
                part_quality=0.1,
                dimensional_accuracy=0.1,
                surface_finish=0.1,
                processing_time=1000.0,
                energy_consumption=5000.0,
            )

        # Update multi-scale state if available
        multiscale_state = self._get_multiscale_state()

        # Create next state
        next_state = ProcessState(
            temperature=new_conditions.temperature or self.current_state.temperature,
            pressure=new_conditions.pressure or self.current_state.pressure,
            flow_rate=new_conditions.flow_rate or self.current_state.flow_rate,
            shear_rate=new_conditions.shear_rate or self.current_state.shear_rate,
            residence_time=new_conditions.residence_time
            or self.current_state.residence_time,
            viscosity=self.current_state.viscosity,  # Updated by digital twin
            density=self.current_state.density,
            thermal_conductivity=self.current_state.thermal_conductivity,
            part_quality=process_output.part_quality,
            dimensional_accuracy=process_output.dimensional_accuracy,
            surface_finish=process_output.surface_finish,
            quality_trend=self.current_state.quality_trend
            + [process_output.part_quality],
            energy_consumption_trend=self.current_state.energy_consumption_trend
            + [process_output.energy_consumption],
            atomistic_energy=multiscale_state.get("atomistic_energy", 0.0),
            mesoscopic_stress=multiscale_state.get("mesoscopic_stress", 0.0),
            continuum_strain=multiscale_state.get("continuum_strain", 0.0),
        )

        # Calculate reward
        reward = self.reward_function.calculate_reward(
            self.current_state, action, next_state, process_output
        )

        # Check if episode is done
        done = (
            self.current_step >= self.max_steps
            or process_output.part_quality < 0.1
            or process_output.energy_consumption  # Quality too low
            > 10000.0  # Energy too high
        )

        # Create info dictionary
        info = {
            "step": self.current_step,
            "process_output": process_output,
            "multiscale_state": multiscale_state,
            "conditions": new_conditions,
        }

        # Store episode data
        self.episode_rewards.append(reward)
        self.episode_states.append(next_state)
        self.episode_actions.append(action)

        # Update current state
        self.current_state = next_state

        return next_state, reward, done, info

    def _apply_action(self, action: ProcessAction) -> ProcessConditions:
        """Apply action to current process conditions."""
        current_conditions = (
            self.digital_twin.process_conditions or self.baseline_conditions
        )

        # Calculate new parameter values
        new_temperature = self._apply_delta(
            current_conditions.temperature or 473.15,
            action.temperature_delta,
            self.parameter_ranges.get("temperature", (400.0, 550.0)),
        )

        new_pressure = self._apply_delta(
            current_conditions.pressure or 2e5,
            action.pressure_delta,
            self.parameter_ranges.get("pressure", (1e5, 5e5)),
        )

        new_flow_rate = self._apply_delta(
            current_conditions.flow_rate or 0.1,
            action.flow_rate_delta,
            self.parameter_ranges.get("flow_rate", (0.01, 0.5)),
        )

        new_shear_rate = self._apply_delta(
            current_conditions.shear_rate or 100.0,
            action.shear_rate_delta,
            self.parameter_ranges.get("shear_rate", (10.0, 500.0)),
        )

        new_residence_time = self._apply_delta(
            current_conditions.residence_time or 60.0,
            action.residence_time_delta,
            self.parameter_ranges.get("residence_time", (10.0, 300.0)),
        )

        return ProcessConditions(
            temperature=new_temperature,
            pressure=new_pressure,
            flow_rate=new_flow_rate,
            shear_rate=new_shear_rate,
            residence_time=new_residence_time,
            melt_temperature=current_conditions.melt_temperature,
            mold_temperature=current_conditions.mold_temperature,
            cycle_time=current_conditions.cycle_time,
        )

    def _apply_delta(
        self, current_value: float, delta: float, value_range: Tuple[float, float]
    ) -> float:
        """Apply normalized delta to parameter value."""
        min_val, max_val = value_range

        # Scale delta based on range
        scaled_delta = delta * (max_val - min_val) * 0.1  # 10% of range per step
        new_value = current_value + scaled_delta

        # Clip to valid range
        return np.clip(new_value, min_val, max_val)

    def _randomize_conditions(
        self, base_conditions: ProcessConditions
    ) -> ProcessConditions:
        """Create randomized initial conditions."""
        randomized = ProcessConditions()

        for param, value_range in self.parameter_ranges.items():
            min_val, max_val = value_range
            random_value = np.random.uniform(min_val, max_val)
            setattr(randomized, param, random_value)

        # Keep non-optimized parameters from base
        for attr in ["melt_temperature", "mold_temperature", "cycle_time"]:
            if hasattr(base_conditions, attr):
                setattr(randomized, attr, getattr(base_conditions, attr))

        return randomized

    def _get_multiscale_state(self) -> Dict[str, float]:
        """Get current multiscale framework state."""
        try:
            # Get state from multiscale framework
            ms_framework = self.digital_twin.multiscale_framework
            if hasattr(ms_framework, "get_current_state"):
                ms_state = ms_framework.get_current_state()

                return {
                    "atomistic_energy": ms_state.get("atomistic_model", {}).get(
                        "energy", 0.0
                    ),
                    "mesoscopic_stress": ms_state.get("mesoscopic_model", {}).get(
                        "stress", 0.0
                    ),
                    "continuum_strain": ms_state.get("continuum_model", {}).get(
                        "strain", 0.0
                    ),
                }
        except Exception as e:
            logger.debug(f"Could not get multiscale state: {e}")

        return {
            "atomistic_energy": 0.0,
            "mesoscopic_stress": 0.0,
            "continuum_strain": 0.0,
        }

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of current episode."""
        if not self.episode_rewards:
            return {"message": "No episode data available"}

        return {
            "total_steps": len(self.episode_rewards),
            "total_reward": sum(self.episode_rewards),
            "average_reward": np.mean(self.episode_rewards),
            "final_quality": (
                self.current_state.part_quality if self.current_state else 0.0
            ),
            "quality_improvement": (
                self.current_state.part_quality - self.episode_states[0].part_quality
                if self.current_state and self.episode_states
                else 0.0
            ),
            "reward_trend": (
                self.episode_rewards[-10:]
                if len(self.episode_rewards) >= 10
                else self.episode_rewards
            ),
        }

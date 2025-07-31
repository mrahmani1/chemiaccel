"""
Multi-Objective Experimental Optimization & Pareto Front Navigation.

This module provides advanced experimental design capabilities with multi-objective
optimization and interactive Pareto front navigation for conflicting objectives.
It extends ChemiAccel's optimization framework to experimental conditions and provides
tools for exploring trade-offs in the experimental space.
"""

import asyncio
import json
import logging
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Optional dependencies with graceful fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some features will be disabled.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualization will be disabled.")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Some data handling features will be disabled.")

# Set up logging
logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Types of experimental objectives."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    TARGET = "target"


class ExperimentalParameter(Enum):
    """Types of experimental parameters."""

    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    CONCENTRATION = "concentration"
    TIME = "time"
    PH = "ph"
    FLOW_RATE = "flow_rate"
    CATALYST_LOADING = "catalyst_loading"
    SOLVENT_RATIO = "solvent_ratio"
    AGITATION_SPEED = "agitation_speed"
    ATMOSPHERE = "atmosphere"


class OptimizationAlgorithm(Enum):
    """Multi-objective optimization algorithms."""

    NSGA_II = "nsga_ii"
    NSGA_III = "nsga_iii"
    MOEA_D = "moea_d"
    SPEA2 = "spea2"
    PARETOARCHIVE = "pareto_archive"


class ExperimentalOptimizationError(Exception):
    """Base exception for experimental optimization errors."""

    pass


@dataclass
class ExperimentalObjective:
    """Configuration for experimental objectives."""

    name: str
    parameter: str
    objective_type: ObjectiveType
    weight: float = 1.0
    target_value: Optional[float] = None
    tolerance: Optional[float] = None
    units: Optional[str] = None
    priority: int = 1


@dataclass
class ExperimentalConstraint:
    """Configuration for experimental constraints."""

    name: str
    parameter: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    penalty_weight: float = 1.0
    constraint_type: str = "bounds"


@dataclass
class ExperimentalCondition:
    """Represents a set of experimental conditions."""

    parameters: Dict[str, float]
    objectives: Dict[str, float] = field(default_factory=dict)
    constraints_satisfied: bool = True
    feasible: bool = True
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParetoPoint:
    """Represents a point on the Pareto front."""

    conditions: ExperimentalCondition
    objective_values: List[float]
    rank: int = 0
    crowding_distance: float = 0.0
    dominated_count: int = 0
    dominates: List[int] = field(default_factory=list)


@dataclass
class ParetoFront:
    """Represents a Pareto front with multiple solutions."""

    points: List[ParetoPoint]
    objectives: List[ExperimentalObjective]
    hypervolume: float = 0.0
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalSpace:
    """Defines the experimental parameter space."""

    parameters: Dict[str, Tuple[float, float]]  # parameter_name: (min, max)
    discrete_parameters: Dict[str, List[Any]] = field(default_factory=dict)
    constraints: List[ExperimentalConstraint] = field(default_factory=list)
    resolution: Dict[str, float] = field(default_factory=dict)


class ExperimentalModel(ABC):
    """Abstract base class for experimental models."""

    @abstractmethod
    def predict(self, conditions: ExperimentalCondition) -> Dict[str, float]:
        """Predict experimental outcomes for given conditions."""
        pass

    @abstractmethod
    def update(
        self, conditions: List[ExperimentalCondition], outcomes: List[Dict[str, float]]
    ) -> None:
        """Update model with new experimental data."""
        pass

    @abstractmethod
    def get_uncertainty(self, conditions: ExperimentalCondition) -> Dict[str, float]:
        """Get prediction uncertainty for given conditions."""
        pass


class SurrogateExperimentalModel(ExperimentalModel):
    """Surrogate model for experimental predictions."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.models = {}
        self.training_data = []
        self.fitted = False

    def predict(self, conditions: ExperimentalCondition) -> Dict[str, float]:
        """Predict experimental outcomes."""
        if not self.fitted:
            # Return dummy predictions if not fitted
            return dict.fromkeys(self.models.keys(), 0.0)

        predictions = {}
        for objective, model in self.models.items():
            if TORCH_AVAILABLE and self.use_gpu:
                # GPU-accelerated prediction
                x = torch.tensor(
                    list(conditions.parameters.values()), dtype=torch.float32
                )
                if torch.cuda.is_available():
                    x = x.cuda()
                with torch.no_grad():
                    pred = model(x.unsqueeze(0)).cpu().numpy()[0]
                predictions[objective] = float(pred)
            else:
                # Simple linear model fallback
                x = np.array(list(conditions.parameters.values()))
                pred = np.dot(model, x) if hasattr(model, "__len__") else 0.0
                predictions[objective] = float(pred)

        return predictions

    def update(
        self, conditions: List[ExperimentalCondition], outcomes: List[Dict[str, float]]
    ) -> None:
        """Update model with new data."""
        self.training_data.extend(zip(conditions, outcomes))

        if len(self.training_data) < 2:
            return

        # Extract features and targets
        X = []
        y_dict = {}

        for cond, outcome in self.training_data:
            X.append(list(cond.parameters.values()))
            for obj, value in outcome.items():
                if obj not in y_dict:
                    y_dict[obj] = []
                y_dict[obj].append(value)

        X = np.array(X)

        # Train simple models for each objective
        for objective, y_values in y_dict.items():
            y = np.array(y_values)

            if TORCH_AVAILABLE and self.use_gpu:
                # Simple neural network
                model = nn.Sequential(
                    nn.Linear(X.shape[1], 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

                if torch.cuda.is_available():
                    model = model.cuda()

                optimizer = optim.Adam(model.parameters(), lr=0.01)
                criterion = nn.MSELoss()

                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

                if torch.cuda.is_available():
                    X_tensor = X_tensor.cuda()
                    y_tensor = y_tensor.cuda()

                # Simple training loop
                for epoch in range(100):
                    optimizer.zero_grad()
                    pred = model(X_tensor)
                    loss = criterion(pred, y_tensor)
                    loss.backward()
                    optimizer.step()

                self.models[objective] = model
            else:
                # Linear regression fallback
                if X.shape[0] >= X.shape[1]:
                    try:
                        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                        self.models[objective] = coeffs
                    except np.linalg.LinAlgError:
                        self.models[objective] = np.zeros(X.shape[1])
                else:
                    self.models[objective] = np.zeros(X.shape[1])

        self.fitted = True

    def get_uncertainty(self, conditions: ExperimentalCondition) -> Dict[str, float]:
        """Get prediction uncertainty."""
        # Simple uncertainty estimation
        uncertainty = {}
        for objective in self.models.keys():
            # Use distance to nearest training point as uncertainty measure
            if self.training_data:
                x = np.array(list(conditions.parameters.values()))
                training_X = np.array(
                    [list(cond.parameters.values()) for cond, _ in self.training_data]
                )
                distances = cdist([x], training_X)[0]
                min_distance = np.min(distances) if len(distances) > 0 else 1.0
                uncertainty[objective] = min_distance * 0.1
            else:
                uncertainty[objective] = 1.0

        return uncertainty


class NonDominatedSorting:
    """Non-dominated sorting for multi-objective optimization."""

    @staticmethod
    def fast_non_dominated_sort(population: List[ParetoPoint]) -> List[List[int]]:
        """Perform fast non-dominated sorting."""
        fronts = [[]]

        for i, point in enumerate(population):
            point.dominated_count = 0
            point.dominates = []

            for j, other_point in enumerate(population):
                if i != j:
                    if NonDominatedSorting._dominates(point, other_point):
                        point.dominates.append(j)
                    elif NonDominatedSorting._dominates(other_point, point):
                        point.dominated_count += 1

            if point.dominated_count == 0:
                point.rank = 0
                fronts[0].append(i)

        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in population[i].dominates:
                    population[j].dominated_count -= 1
                    if population[j].dominated_count == 0:
                        population[j].rank = current_front + 1
                        next_front.append(j)

            current_front += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    @staticmethod
    def _dominates(point1: ParetoPoint, point2: ParetoPoint) -> bool:
        """Check if point1 dominates point2."""
        better_in_any = False

        for i, (obj1, obj2) in enumerate(
            zip(point1.objective_values, point2.objective_values)
        ):
            if obj1 > obj2:  # Assuming maximization
                return False
            elif obj1 < obj2:
                better_in_any = True

        return better_in_any


class CrowdingDistance:
    """Crowding distance calculation for diversity preservation."""

    @staticmethod
    def calculate_crowding_distance(
        front: List[int], population: List[ParetoPoint]
    ) -> None:
        """Calculate crowding distance for a front."""
        if len(front) <= 2:
            for i in front:
                population[i].crowding_distance = float("inf")
            return

        # Initialize distances
        for i in front:
            population[i].crowding_distance = 0.0

        num_objectives = len(population[front[0]].objective_values)

        for obj_idx in range(num_objectives):
            # Sort by objective value
            front.sort(key=lambda x: population[x].objective_values[obj_idx])

            # Set boundary points to infinity
            population[front[0]].crowding_distance = float("inf")
            population[front[-1]].crowding_distance = float("inf")

            # Calculate objective range
            obj_min = population[front[0]].objective_values[obj_idx]
            obj_max = population[front[-1]].objective_values[obj_idx]
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # Calculate distances for intermediate points
            for i in range(1, len(front) - 1):
                if population[front[i]].crowding_distance != float("inf"):
                    distance = (
                        population[front[i + 1]].objective_values[obj_idx]
                        - population[front[i - 1]].objective_values[obj_idx]
                    )
                    population[front[i]].crowding_distance += distance / obj_range


class HypervolumeCalculator:
    """Hypervolume calculation for Pareto front quality assessment."""

    @staticmethod
    def calculate_hypervolume(
        pareto_front: ParetoFront, reference_point: Optional[List[float]] = None
    ) -> float:
        """Calculate hypervolume of Pareto front."""
        if not pareto_front.points:
            return 0.0

        # Extract objective values
        objectives = np.array([point.objective_values for point in pareto_front.points])

        if reference_point is None:
            # Use worst point as reference
            reference_point = np.max(objectives, axis=0) + 1.0

        reference_point = np.array(reference_point)

        # Simple hypervolume calculation (works well for 2-3 objectives)
        if objectives.shape[1] == 2:
            return HypervolumeCalculator._hypervolume_2d(objectives, reference_point)
        elif objectives.shape[1] == 3:
            return HypervolumeCalculator._hypervolume_3d(objectives, reference_point)
        else:
            # Use Monte Carlo approximation for higher dimensions
            return HypervolumeCalculator._hypervolume_monte_carlo(
                objectives, reference_point
            )

    @staticmethod
    def _hypervolume_2d(objectives: np.ndarray, reference: np.ndarray) -> float:
        """Calculate 2D hypervolume."""
        # Sort by first objective
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_objectives = objectives[sorted_indices]

        hypervolume = 0.0
        prev_x = reference[0]

        for point in sorted_objectives:
            if point[1] < reference[1]:
                width = prev_x - point[0]
                height = reference[1] - point[1]
                hypervolume += width * height
                prev_x = point[0]

        return hypervolume

    @staticmethod
    def _hypervolume_3d(objectives: np.ndarray, reference: np.ndarray) -> float:
        """Calculate 3D hypervolume using inclusion-exclusion principle."""
        hypervolume = 0.0
        n_points = len(objectives)

        # Single points
        for i in range(n_points):
            point = objectives[i]
            if np.all(point <= reference):
                volume = np.prod(reference - point)
                hypervolume += volume

        # Pairwise intersections (subtract)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                intersection = np.minimum(objectives[i], objectives[j])
                if np.all(intersection <= reference):
                    volume = np.prod(reference - intersection)
                    hypervolume -= volume

        # Triple intersections (add back)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                for k in range(j + 1, n_points):
                    intersection = np.minimum.reduce(
                        [objectives[i], objectives[j], objectives[k]]
                    )
                    if np.all(intersection <= reference):
                        volume = np.prod(reference - intersection)
                        hypervolume += volume

        return hypervolume

    @staticmethod
    def _hypervolume_monte_carlo(
        objectives: np.ndarray, reference: np.ndarray, n_samples: int = 100000
    ) -> float:
        """Monte Carlo approximation for high-dimensional hypervolume."""
        # Generate random points in the reference space
        n_objectives = len(reference)
        random_points = np.random.uniform(
            low=np.min(objectives, axis=0),
            high=reference,
            size=(n_samples, n_objectives),
        )

        # Count points dominated by at least one Pareto point
        dominated_count = 0
        for point in random_points:
            for pareto_point in objectives:
                if np.all(pareto_point <= point):
                    dominated_count += 1
                    break

        # Calculate hypervolume as fraction of reference volume
        reference_volume = np.prod(reference - np.min(objectives, axis=0))
        return (dominated_count / n_samples) * reference_volume


class ExperimentalOptimizer:
    """Multi-objective experimental optimizer with Pareto front navigation."""

    def __init__(
        self,
        experimental_space: ExperimentalSpace,
        objectives: List[ExperimentalObjective],
        model: ExperimentalModel,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA_II,
        population_size: int = 100,
        max_generations: int = 100,
        use_gpu: bool = True,
    ):
        self.experimental_space = experimental_space
        self.objectives = objectives
        self.model = model
        self.algorithm = algorithm
        self.population_size = population_size
        self.max_generations = max_generations
        self.use_gpu = use_gpu and TORCH_AVAILABLE

        self.current_population = []
        self.pareto_history = []
        self.best_hypervolume = 0.0

    def optimize(
        self, initial_conditions: Optional[List[ExperimentalCondition]] = None
    ) -> ParetoFront:
        """Run multi-objective optimization."""
        logger.info(f"Starting experimental optimization with {self.algorithm.value}")

        # Initialize population
        if initial_conditions:
            self.current_population = self._conditions_to_pareto_points(
                initial_conditions
            )
        else:
            self.current_population = self._generate_initial_population()

        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate population
            self._evaluate_population()

            # Non-dominated sorting
            fronts = NonDominatedSorting.fast_non_dominated_sort(
                self.current_population
            )

            # Calculate crowding distances
            for front in fronts:
                CrowdingDistance.calculate_crowding_distance(
                    front, self.current_population
                )

            # Create Pareto front
            if fronts:
                pareto_points = [self.current_population[i] for i in fronts[0]]
                pareto_front = ParetoFront(
                    points=pareto_points,
                    objectives=self.objectives,
                    generation=generation,
                )

                # Calculate hypervolume
                pareto_front.hypervolume = HypervolumeCalculator.calculate_hypervolume(
                    pareto_front
                )
                self.pareto_history.append(pareto_front)

                if pareto_front.hypervolume > self.best_hypervolume:
                    self.best_hypervolume = pareto_front.hypervolume

                logger.info(
                    f"Generation {generation}: "
                    f"Pareto front size = {len(pareto_points)}, "
                    f"Hypervolume = {pareto_front.hypervolume:.4f}"
                )

            # Generate next population
            if generation < self.max_generations - 1:
                self.current_population = self._generate_next_population(fronts)

        # Return final Pareto front
        final_fronts = NonDominatedSorting.fast_non_dominated_sort(
            self.current_population
        )
        if final_fronts:
            final_pareto_points = [self.current_population[i] for i in final_fronts[0]]
            final_pareto_front = ParetoFront(
                points=final_pareto_points,
                objectives=self.objectives,
                generation=self.max_generations,
            )
            final_pareto_front.hypervolume = (
                HypervolumeCalculator.calculate_hypervolume(final_pareto_front)
            )
            return final_pareto_front
        else:
            return ParetoFront(points=[], objectives=self.objectives)

    def _generate_initial_population(self) -> List[ParetoPoint]:
        """Generate initial population of experimental conditions."""
        population = []

        for _ in range(self.population_size):
            parameters = {}

            # Sample continuous parameters
            for param_name, (
                min_val,
                max_val,
            ) in self.experimental_space.parameters.items():
                parameters[param_name] = np.random.uniform(min_val, max_val)

            # Sample discrete parameters
            for (
                param_name,
                values,
            ) in self.experimental_space.discrete_parameters.items():
                parameters[param_name] = np.random.choice(values)

            condition = ExperimentalCondition(parameters=parameters)
            pareto_point = ParetoPoint(
                conditions=condition, objective_values=[0.0] * len(self.objectives)
            )
            population.append(pareto_point)

        return population

    def _conditions_to_pareto_points(
        self, conditions: List[ExperimentalCondition]
    ) -> List[ParetoPoint]:
        """Convert experimental conditions to Pareto points."""
        pareto_points = []

        for condition in conditions:
            pareto_point = ParetoPoint(
                conditions=condition, objective_values=[0.0] * len(self.objectives)
            )
            pareto_points.append(pareto_point)

        return pareto_points

    def _evaluate_population(self) -> None:
        """Evaluate objective values for population."""
        for point in self.current_population:
            predictions = self.model.predict(point.conditions)

            objective_values = []
            for objective in self.objectives:
                if objective.name in predictions:
                    value = predictions[objective.name]

                    # Handle different objective types
                    if objective.objective_type == ObjectiveType.MINIMIZE:
                        objective_values.append(-value)  # Convert to maximization
                    elif objective.objective_type == ObjectiveType.MAXIMIZE:
                        objective_values.append(value)
                    elif objective.objective_type == ObjectiveType.TARGET:
                        if objective.target_value is not None:
                            distance = abs(value - objective.target_value)
                            objective_values.append(-distance)  # Minimize distance
                        else:
                            objective_values.append(value)
                else:
                    objective_values.append(0.0)

            point.objective_values = objective_values

            # Check constraints
            point.conditions.constraints_satisfied = self._check_constraints(
                point.conditions
            )
            point.conditions.feasible = point.conditions.constraints_satisfied

    def _check_constraints(self, condition: ExperimentalCondition) -> bool:
        """Check if experimental condition satisfies constraints."""
        for constraint in self.experimental_space.constraints:
            param_value = condition.parameters.get(constraint.parameter)

            if param_value is None:
                continue

            if constraint.min_value is not None and param_value < constraint.min_value:
                return False

            if constraint.max_value is not None and param_value > constraint.max_value:
                return False

            if (
                constraint.allowed_values is not None
                and param_value not in constraint.allowed_values
            ):
                return False

        return True

    def _generate_next_population(self, fronts: List[List[int]]) -> List[ParetoPoint]:
        """Generate next population using NSGA-II selection."""
        new_population = []

        # Add entire fronts until population is filled
        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend([self.current_population[i] for i in front])
            else:
                # Add partial front based on crowding distance
                remaining_slots = self.population_size - len(new_population)
                front_points = [self.current_population[i] for i in front]
                front_points.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_population.extend(front_points[:remaining_slots])
                break

        # Apply variation operators (crossover and mutation)
        offspring = self._generate_offspring(new_population)

        return new_population + offspring

    def _generate_offspring(self, parents: List[ParetoPoint]) -> List[ParetoPoint]:
        """Generate offspring through crossover and mutation."""
        offspring = []
        n_offspring = self.population_size - len(parents)

        for _ in range(n_offspring):
            # Select two parents
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)

            # Crossover
            child_parameters = {}
            for param_name in parent1.conditions.parameters.keys():
                if np.random.random() < 0.5:
                    child_parameters[param_name] = parent1.conditions.parameters[
                        param_name
                    ]
                else:
                    child_parameters[param_name] = parent2.conditions.parameters[
                        param_name
                    ]

            # Mutation
            for param_name, value in child_parameters.items():
                if np.random.random() < 0.1:  # Mutation probability
                    if param_name in self.experimental_space.parameters:
                        min_val, max_val = self.experimental_space.parameters[
                            param_name
                        ]
                        noise = np.random.normal(0, (max_val - min_val) * 0.1)
                        child_parameters[param_name] = np.clip(
                            value + noise, min_val, max_val
                        )
                    elif param_name in self.experimental_space.discrete_parameters:
                        values = self.experimental_space.discrete_parameters[param_name]
                        child_parameters[param_name] = np.random.choice(values)

            child_condition = ExperimentalCondition(parameters=child_parameters)
            child_point = ParetoPoint(
                conditions=child_condition,
                objective_values=[0.0] * len(self.objectives),
            )
            offspring.append(child_point)

        return offspring


class ParetoNavigator:
    """Interactive tools for navigating Pareto fronts in experimental space."""

    def __init__(self, pareto_front: ParetoFront):
        self.pareto_front = pareto_front
        self.selected_points = []
        self.filters = {}

    def filter_by_parameter(
        self, parameter: str, min_val: float, max_val: float
    ) -> List[ParetoPoint]:
        """Filter Pareto points by parameter range."""
        filtered_points = []

        for point in self.pareto_front.points:
            param_value = point.conditions.parameters.get(parameter)
            if param_value is not None and min_val <= param_value <= max_val:
                filtered_points.append(point)

        return filtered_points

    def filter_by_objective(
        self, objective_idx: int, min_val: float, max_val: float
    ) -> List[ParetoPoint]:
        """Filter Pareto points by objective value range."""
        filtered_points = []

        for point in self.pareto_front.points:
            if objective_idx < len(point.objective_values):
                obj_value = point.objective_values[objective_idx]
                if min_val <= obj_value <= max_val:
                    filtered_points.append(point)

        return filtered_points

    def find_knee_points(
        self, weights: Optional[List[float]] = None
    ) -> List[ParetoPoint]:
        """Find knee points on the Pareto front."""
        if len(self.pareto_front.points) < 3:
            return self.pareto_front.points

        if weights is None:
            weights = [1.0] * len(self.pareto_front.objectives)

        # Calculate weighted distances from utopia point
        objectives = np.array(
            [point.objective_values for point in self.pareto_front.points]
        )
        utopia = np.max(objectives, axis=0)

        distances = []
        for point_obj in objectives:
            weighted_distance = np.sqrt(
                np.sum(
                    [(u - p) ** 2 * w for u, p, w in zip(utopia, point_obj, weights)]
                )
            )
            distances.append(weighted_distance)

        # Find points with minimum distance (knee points)
        min_distance = np.min(distances)
        knee_indices = np.where(np.array(distances) <= min_distance * 1.1)[0]

        return [self.pareto_front.points[i] for i in knee_indices]

    def get_trade_off_analysis(self) -> Dict[str, Any]:
        """Analyze trade-offs between objectives."""
        if len(self.pareto_front.points) < 2:
            return {"error": "Insufficient points for trade-off analysis"}

        objectives = np.array(
            [point.objective_values for point in self.pareto_front.points]
        )

        analysis = {"objective_ranges": {}, "correlations": {}, "diversity_metrics": {}}

        # Objective ranges
        for i, objective in enumerate(self.pareto_front.objectives):
            obj_values = objectives[:, i]
            analysis["objective_ranges"][objective.name] = {
                "min": float(np.min(obj_values)),
                "max": float(np.max(obj_values)),
                "mean": float(np.mean(obj_values)),
                "std": float(np.std(obj_values)),
            }

        # Correlation analysis
        if objectives.shape[1] > 1:
            correlation_matrix = np.corrcoef(objectives.T)
            for i, obj1 in enumerate(self.pareto_front.objectives):
                for j, obj2 in enumerate(self.pareto_front.objectives):
                    if i < j:
                        key = f"{obj1.name}_vs_{obj2.name}"
                        analysis["correlations"][key] = float(correlation_matrix[i, j])

        # Diversity metrics
        if len(objectives) > 1:
            distances = cdist(objectives, objectives)
            analysis["diversity_metrics"] = {
                "mean_distance": float(np.mean(distances[distances > 0])),
                "min_distance": float(np.min(distances[distances > 0])),
                "max_distance": float(np.max(distances)),
                "spread": float(np.max(distances)),
            }

        return analysis

    def create_interactive_plot(self) -> Optional[Any]:
        """Create interactive Plotly visualization."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Interactive plotting disabled.")
            return None

        if len(self.pareto_front.objectives) < 2:
            logger.warning("Need at least 2 objectives for plotting.")
            return None

        objectives = np.array(
            [point.objective_values for point in self.pareto_front.points]
        )

        if len(self.pareto_front.objectives) == 2:
            return self._create_2d_plot(objectives)
        elif len(self.pareto_front.objectives) == 3:
            return self._create_3d_plot(objectives)
        else:
            return self._create_parallel_coordinates_plot()

    def _create_2d_plot(self, objectives: np.ndarray) -> Any:
        """Create 2D scatter plot."""
        fig = go.Figure()

        # Extract hover text with experimental parameters
        hover_text = []
        for point in self.pareto_front.points:
            params_str = "<br>".join(
                [f"{k}: {v:.3f}" for k, v in point.conditions.parameters.items()]
            )
            hover_text.append(f"Parameters:<br>{params_str}")

        fig.add_trace(
            go.Scatter(
                x=objectives[:, 0],
                y=objectives[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=objectives[:, 0],
                    colorscale="Viridis",
                    showscale=True,
                ),
                text=hover_text,
                hovertemplate="<b>%{text}</b><br>"
                + f"{self.pareto_front.objectives[0].name}: %{{x:.3f}}<br>"
                + f"{self.pareto_front.objectives[1].name}: %{{y:.3f}}<extra></extra>",
                name="Pareto Front",
            )
        )

        fig.update_layout(
            title="2D Pareto Front",
            xaxis_title=self.pareto_front.objectives[0].name,
            yaxis_title=self.pareto_front.objectives[1].name,
            hovermode="closest",
        )

        return fig

    def _create_3d_plot(self, objectives: np.ndarray) -> Any:
        """Create 3D scatter plot."""
        fig = go.Figure()

        # Extract hover text
        hover_text = []
        for point in self.pareto_front.points:
            params_str = "<br>".join(
                [f"{k}: {v:.3f}" for k, v in point.conditions.parameters.items()]
            )
            hover_text.append(f"Parameters:<br>{params_str}")

        fig.add_trace(
            go.Scatter3d(
                x=objectives[:, 0],
                y=objectives[:, 1],
                z=objectives[:, 2],
                mode="markers",
                marker=dict(
                    size=8, color=objectives[:, 0], colorscale="Viridis", showscale=True
                ),
                text=hover_text,
                hovertemplate="<b>%{text}</b><br>"
                + f"{self.pareto_front.objectives[0].name}: %{{x:.3f}}<br>"
                + f"{self.pareto_front.objectives[1].name}: %{{y:.3f}}<br>"
                + f"{self.pareto_front.objectives[2].name}: %{{z:.3f}}<extra></extra>",
                name="Pareto Front",
            )
        )

        fig.update_layout(
            title="3D Pareto Front",
            scene=dict(
                xaxis_title=self.pareto_front.objectives[0].name,
                yaxis_title=self.pareto_front.objectives[1].name,
                zaxis_title=self.pareto_front.objectives[2].name,
            ),
        )

        return fig

    def _create_parallel_coordinates_plot(self) -> Any:
        """Create parallel coordinates plot for high-dimensional objectives."""
        if not PANDAS_AVAILABLE:
            logger.warning(
                "Pandas not available. Cannot create parallel coordinates plot."
            )
            return None

        # Prepare data
        data = []
        for point in self.pareto_front.points:
            row = {}
            for i, objective in enumerate(self.pareto_front.objectives):
                row[objective.name] = point.objective_values[i]

            # Add key parameters
            for param, value in point.conditions.parameters.items():
                if len(row) < 10:  # Limit to prevent overcrowding
                    row[f"param_{param}"] = value

            data.append(row)

        df = pd.DataFrame(data)

        fig = px.parallel_coordinates(
            df, title="Multi-Objective Pareto Front", color=df.columns[0]
        )

        return fig


# Utility functions
def create_chemical_synthesis_objectives() -> List[ExperimentalObjective]:
    """Create typical objectives for chemical synthesis optimization."""
    return [
        ExperimentalObjective(
            name="yield",
            parameter="product_yield",
            objective_type=ObjectiveType.MAXIMIZE,
            weight=1.0,
            units="%",
        ),
        ExperimentalObjective(
            name="selectivity",
            parameter="product_selectivity",
            objective_type=ObjectiveType.MAXIMIZE,
            weight=0.8,
            units="%",
        ),
        ExperimentalObjective(
            name="reaction_time",
            parameter="time",
            objective_type=ObjectiveType.MINIMIZE,
            weight=0.6,
            units="hours",
        ),
        ExperimentalObjective(
            name="cost",
            parameter="total_cost",
            objective_type=ObjectiveType.MINIMIZE,
            weight=0.7,
            units="$/mol",
        ),
    ]


def create_polymer_synthesis_objectives() -> List[ExperimentalObjective]:
    """Create objectives for polymer synthesis optimization."""
    return [
        ExperimentalObjective(
            name="molecular_weight",
            parameter="mn",
            objective_type=ObjectiveType.TARGET,
            target_value=50000,
            tolerance=5000,
            weight=1.0,
            units="g/mol",
        ),
        ExperimentalObjective(
            name="polydispersity",
            parameter="pdi",
            objective_type=ObjectiveType.MINIMIZE,
            weight=0.9,
            units="",
        ),
        ExperimentalObjective(
            name="conversion",
            parameter="monomer_conversion",
            objective_type=ObjectiveType.MAXIMIZE,
            weight=0.8,
            units="%",
        ),
        ExperimentalObjective(
            name="processing_time",
            parameter="polymerization_time",
            objective_type=ObjectiveType.MINIMIZE,
            weight=0.5,
            units="hours",
        ),
    ]


def create_standard_experimental_space() -> ExperimentalSpace:
    """Create standard experimental parameter space."""
    return ExperimentalSpace(
        parameters={
            "temperature": (20.0, 200.0),
            "pressure": (1.0, 10.0),
            "concentration": (0.01, 2.0),
            "time": (0.5, 24.0),
            "ph": (1.0, 14.0),
        },
        discrete_parameters={
            "catalyst": ["Pd", "Pt", "Rh", "Ru"],
            "solvent": ["DMF", "THF", "DCM", "MeOH", "H2O"],
        },
        constraints=[
            ExperimentalConstraint(
                name="temperature_pressure_safety",
                parameter="temperature",
                min_value=20.0,
                max_value=150.0,
                constraint_type="safety",
            ),
            ExperimentalConstraint(
                name="concentration_range",
                parameter="concentration",
                min_value=0.01,
                max_value=1.0,
                constraint_type="feasibility",
            ),
        ],
    )


async def run_experimental_optimization(
    experimental_space: ExperimentalSpace,
    objectives: List[ExperimentalObjective],
    model: ExperimentalModel,
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA_II,
    population_size: int = 100,
    max_generations: int = 50,
) -> ParetoFront:
    """Run experimental optimization asynchronously."""
    optimizer = ExperimentalOptimizer(
        experimental_space=experimental_space,
        objectives=objectives,
        model=model,
        algorithm=algorithm,
        population_size=population_size,
        max_generations=max_generations,
    )

    return optimizer.optimize()


def analyze_pareto_trade_offs(pareto_front: ParetoFront) -> Dict[str, Any]:
    """Analyze trade-offs in a Pareto front."""
    navigator = ParetoNavigator(pareto_front)
    return navigator.get_trade_off_analysis()


def find_optimal_experimental_conditions(
    pareto_front: ParetoFront, preference_weights: Optional[List[float]] = None
) -> ExperimentalCondition:
    """Find optimal experimental conditions based on preferences."""
    navigator = ParetoNavigator(pareto_front)

    if preference_weights is None:
        # Use equal weights
        preference_weights = [1.0] * len(pareto_front.objectives)

    knee_points = navigator.find_knee_points(preference_weights)

    if knee_points:
        return knee_points[0].conditions
    elif pareto_front.points:
        return pareto_front.points[0].conditions
    else:
        raise ExperimentalOptimizationError("No feasible experimental conditions found")

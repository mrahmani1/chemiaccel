"""
Multi-Objective Optimization Framework for ChemiAccel.

This module provides comprehensive multi-objective optimization capabilities including:
- GPU-accelerated Pareto front identification
- Non-dominated sorting algorithms
- Interactive visualization tools for trade-off analysis
- Scalarization functions for generative model guidance
- Performance optimization for large-scale problems
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. GPU acceleration will be disabled.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualization will be disabled.")

from ..core.exceptions import ChemiAccelError


class MultiObjectiveOptimizationError(ChemiAccelError):
    """Raised when multi-objective optimization operations fail."""

    pass


@dataclass
class ObjectiveConfig:
    """Configuration for a single objective."""

    name: str
    minimize: bool = True
    weight: float = 1.0
    target: Optional[float] = None
    bounds: Optional[Tuple[float, float]] = None
    normalization: str = "none"  # 'none', 'minmax', 'zscore'


@dataclass
class ParetoFrontResult:
    """Results from Pareto front analysis."""

    pareto_indices: np.ndarray
    pareto_molecules: List[str]
    pareto_objectives: np.ndarray
    dominated_indices: np.ndarray
    crowding_distances: np.ndarray
    hypervolume: float
    computation_time: float


@dataclass
class MultiObjectiveResult:
    """Comprehensive multi-objective optimization results."""

    molecules: List[str]
    objective_values: np.ndarray
    pareto_front: ParetoFrontResult
    scalarized_scores: Optional[np.ndarray] = None
    diversity_metrics: Optional[Dict[str, float]] = None
    convergence_metrics: Optional[Dict[str, Any]] = None


class NonDominatedSorting:
    """GPU-accelerated non-dominated sorting algorithms."""

    def __init__(self, use_gpu: bool = True):
        """Initialize non-dominated sorting."""
        self.use_gpu = use_gpu and CUPY_AVAILABLE

    def fast_non_dominated_sort(self, objectives: np.ndarray) -> List[np.ndarray]:
        """
        Fast non-dominated sorting algorithm (NSGA-II style).

        Args:
            objectives: Array of shape (n_points, n_objectives)

        Returns:
            List of arrays, each containing indices of solutions in that front
        """
        if self.use_gpu:
            return self._gpu_fast_non_dominated_sort(objectives)
        else:
            return self._cpu_fast_non_dominated_sort(objectives)

    def _cpu_fast_non_dominated_sort(self, objectives: np.ndarray) -> List[np.ndarray]:
        """CPU implementation of fast non-dominated sorting."""
        n_points, n_objectives = objectives.shape

        # Initialize data structures
        domination_count = np.zeros(n_points, dtype=int)
        dominated_solutions = [[] for _ in range(n_points)]
        fronts = [[]]

        # Find first front
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        # Find subsequent fronts
        front_idx = 0
        while front_idx < len(fronts) and len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            if next_front:
                fronts.append(next_front)
            front_idx += 1

        # Remove empty last front if it exists
        if fronts and not fronts[-1]:
            fronts.pop()

        return [np.array(front) for front in fronts]

    def _gpu_fast_non_dominated_sort(self, objectives: np.ndarray) -> List[np.ndarray]:
        """GPU implementation of fast non-dominated sorting."""
        if not CUPY_AVAILABLE:
            # Fall back to CPU implementation
            return self._cpu_fast_non_dominated_sort(objectives)

        objectives_gpu = cp.asarray(objectives)
        n_points, n_objectives = objectives_gpu.shape

        # GPU kernel for domination comparison
        domination_matrix = self._compute_domination_matrix_gpu(objectives_gpu)

        # Convert back to CPU for front extraction
        domination_matrix_cpu = cp.asnumpy(domination_matrix)

        # Extract fronts using CPU algorithm
        return self._extract_fronts_from_domination_matrix(domination_matrix_cpu)

    def _compute_domination_matrix_gpu(self, objectives) -> np.ndarray:
        """Compute domination matrix on GPU."""
        if not CUPY_AVAILABLE:
            # Fall back to CPU implementation
            return self._compute_domination_matrix_cpu(objectives)

        n_points = objectives.shape[0]

        # Create domination matrix: domination_matrix[i, j] = 1 if i dominates j
        domination_matrix = cp.zeros((n_points, n_points), dtype=cp.int32)

        # Use broadcasting for efficient comparison
        for i in range(n_points):
            obj_i = objectives[i : i + 1, :]  # Shape: (1, n_objectives)

            # Check if obj_i dominates all other objectives
            better_or_equal = obj_i <= objectives  # Shape: (n_points, n_objectives)
            strictly_better = obj_i < objectives  # Shape: (n_points, n_objectives)

            # i dominates j if i is better or equal in all objectives and strictly better in at least one
            dominates = cp.all(better_or_equal, axis=1) & cp.any(
                strictly_better, axis=1
            )
            domination_matrix[i, :] = dominates.astype(cp.int32)

        return domination_matrix

    def _compute_domination_matrix_cpu(self, objectives: np.ndarray) -> np.ndarray:
        """Compute domination matrix on CPU."""
        n_points = objectives.shape[0]

        # Create domination matrix: domination_matrix[i, j] = 1 if i dominates j
        domination_matrix = np.zeros((n_points, n_points), dtype=np.int32)

        # Use broadcasting for efficient comparison
        for i in range(n_points):
            obj_i = objectives[i : i + 1, :]  # Shape: (1, n_objectives)

            # Check if obj_i dominates all other objectives
            better_or_equal = obj_i <= objectives  # Shape: (n_points, n_objectives)
            strictly_better = obj_i < objectives  # Shape: (n_points, n_objectives)

            # i dominates j if i is better or equal in all objectives and strictly better in at least one
            dominates = np.all(better_or_equal, axis=1) & np.any(
                strictly_better, axis=1
            )
            domination_matrix[i, :] = dominates.astype(np.int32)

        return domination_matrix

    def _extract_fronts_from_domination_matrix(
        self, domination_matrix: np.ndarray
    ) -> List[np.ndarray]:
        """Extract fronts from domination matrix."""
        n_points = domination_matrix.shape[0]
        assigned = np.zeros(n_points, dtype=bool)
        fronts = []

        while not np.all(assigned):
            # Find points not dominated by any unassigned point
            current_front = []

            for i in range(n_points):
                if assigned[i]:
                    continue

                # Check if i is dominated by any unassigned point
                dominated = False
                for j in range(n_points):
                    if not assigned[j] and j != i and domination_matrix[j, i]:
                        dominated = True
                        break

                if not dominated:
                    current_front.append(i)

            if not current_front:
                break

            fronts.append(np.array(current_front))
            assigned[current_front] = True

        return fronts

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2."""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)


class CrowdingDistance:
    """Calculate crowding distance for diversity preservation."""

    def __init__(self, use_gpu: bool = True):
        """Initialize crowding distance calculator."""
        self.use_gpu = use_gpu and CUPY_AVAILABLE

    def calculate_crowding_distance(
        self, objectives: np.ndarray, front_indices: np.ndarray
    ) -> np.ndarray:
        """
        Calculate crowding distance for solutions in a front.

        Args:
            objectives: Array of shape (n_points, n_objectives)
            front_indices: Indices of solutions in the current front

        Returns:
            Crowding distances for solutions in the front
        """
        if self.use_gpu:
            return self._gpu_crowding_distance(objectives, front_indices)
        else:
            return self._cpu_crowding_distance(objectives, front_indices)

    def _cpu_crowding_distance(
        self, objectives: np.ndarray, front_indices: np.ndarray
    ) -> np.ndarray:
        """CPU implementation of crowding distance calculation."""
        n_front = len(front_indices)
        n_objectives = objectives.shape[1]

        if n_front <= 2:
            return np.full(n_front, float("inf"))

        crowding_distances = np.zeros(n_front)
        front_objectives = objectives[front_indices]

        for obj_idx in range(n_objectives):
            # Sort by objective value
            sorted_indices = np.argsort(front_objectives[:, obj_idx])

            # Boundary points get infinite distance
            crowding_distances[sorted_indices[0]] = float("inf")
            crowding_distances[sorted_indices[-1]] = float("inf")

            # Calculate crowding distance for interior points
            obj_range = (
                front_objectives[sorted_indices[-1], obj_idx]
                - front_objectives[sorted_indices[0], obj_idx]
            )

            if obj_range > 0:
                for i in range(1, n_front - 1):
                    distance = (
                        front_objectives[sorted_indices[i + 1], obj_idx]
                        - front_objectives[sorted_indices[i - 1], obj_idx]
                    ) / obj_range
                    crowding_distances[sorted_indices[i]] += distance

        return crowding_distances

    def _gpu_crowding_distance(
        self, objectives: np.ndarray, front_indices: np.ndarray
    ) -> np.ndarray:
        """GPU implementation of crowding distance calculation."""
        if not CUPY_AVAILABLE:
            # Fall back to CPU implementation
            return self._cpu_crowding_distance(objectives, front_indices)

        objectives_gpu = cp.asarray(objectives)
        front_indices_gpu = cp.asarray(front_indices)

        n_front = len(front_indices)
        n_objectives = objectives_gpu.shape[1]

        if n_front <= 2:
            return np.full(n_front, float("inf"))

        crowding_distances = cp.zeros(n_front)
        front_objectives = objectives_gpu[front_indices_gpu]

        for obj_idx in range(n_objectives):
            # Sort by objective value
            sorted_indices = cp.argsort(front_objectives[:, obj_idx])

            # Boundary points get infinite distance
            crowding_distances[sorted_indices[0]] = float("inf")
            crowding_distances[sorted_indices[-1]] = float("inf")

            # Calculate crowding distance for interior points
            obj_range = (
                front_objectives[sorted_indices[-1], obj_idx]
                - front_objectives[sorted_indices[0], obj_idx]
            )

            if obj_range > 0:
                for i in range(1, n_front - 1):
                    distance = (
                        front_objectives[sorted_indices[i + 1], obj_idx]
                        - front_objectives[sorted_indices[i - 1], obj_idx]
                    ) / obj_range
                    crowding_distances[sorted_indices[i]] += distance

        return cp.asnumpy(crowding_distances)


class HypervolumeCalculator:
    """Calculate hypervolume indicator for multi-objective optimization."""

    def __init__(self, reference_point: Optional[np.ndarray] = None):
        """
        Initialize hypervolume calculator.

        Args:
            reference_point: Reference point for hypervolume calculation
        """
        self.reference_point = reference_point

    def calculate_hypervolume(
        self, objectives: np.ndarray, pareto_indices: np.ndarray
    ) -> float:
        """
        Calculate hypervolume of Pareto front.

        Args:
            objectives: Array of shape (n_points, n_objectives)
            pareto_indices: Indices of Pareto optimal solutions

        Returns:
            Hypervolume value
        """
        pareto_objectives = objectives[pareto_indices]

        if self.reference_point is None:
            # Use worst values as reference point
            self.reference_point = np.max(objectives, axis=0) + 1.0

        return self._calculate_hypervolume_recursive(
            pareto_objectives, self.reference_point
        )

    def _calculate_hypervolume_recursive(
        self, points: np.ndarray, reference: np.ndarray
    ) -> float:
        """Recursive hypervolume calculation (WFG algorithm)."""
        n_points, n_objectives = points.shape

        if n_points == 0:
            return 0.0

        if n_objectives == 1:
            return np.max(reference[0] - points[:, 0])

        # Sort points by last objective
        sorted_indices = np.argsort(points[:, -1])
        sorted_points = points[sorted_indices]

        hypervolume = 0.0
        prev_point = reference.copy()

        for i in range(n_points):
            current_point = sorted_points[i]

            if i == 0 or current_point[-1] < prev_point[-1]:
                # Calculate contribution of this point
                temp_reference = prev_point.copy()
                temp_reference[-1] = current_point[-1]

                # Recursive call for lower dimensions
                dominated_points = []
                for j in range(i + 1, n_points):
                    if np.all(sorted_points[j, :-1] >= current_point[:-1]):
                        dominated_points.append(sorted_points[j, :-1])

                if dominated_points:
                    dominated_points = np.array(dominated_points)
                    sub_hypervolume = self._calculate_hypervolume_recursive(
                        dominated_points, temp_reference[:-1]
                    )
                else:
                    sub_hypervolume = np.prod(temp_reference[:-1] - current_point[:-1])

                hypervolume += (prev_point[-1] - current_point[-1]) * sub_hypervolume
                prev_point = current_point.copy()

        return hypervolume


class ScalarizationFunctions:
    """Scalarization functions for converting multi-objective to single-objective."""

    @staticmethod
    def weighted_sum(objectives: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Weighted sum scalarization.

        Args:
            objectives: Array of shape (n_points, n_objectives)
            weights: Array of shape (n_objectives,)

        Returns:
            Scalarized scores of shape (n_points,)
        """
        return np.sum(objectives * weights, axis=1)

    @staticmethod
    def tchebycheff(
        objectives: np.ndarray, weights: np.ndarray, ideal_point: np.ndarray
    ) -> np.ndarray:
        """
        Tchebycheff scalarization.

        Args:
            objectives: Array of shape (n_points, n_objectives)
            weights: Array of shape (n_objectives,)
            ideal_point: Array of shape (n_objectives,)

        Returns:
            Scalarized scores of shape (n_points,)
        """
        weighted_diff = weights * np.abs(objectives - ideal_point)
        return np.max(weighted_diff, axis=1)

    @staticmethod
    def achievement_scalarizing_function(
        objectives: np.ndarray,
        weights: np.ndarray,
        reference_point: np.ndarray,
        rho: float = 0.001,
    ) -> np.ndarray:
        """
        Achievement scalarizing function.

        Args:
            objectives: Array of shape (n_points, n_objectives)
            weights: Array of shape (n_objectives,)
            reference_point: Array of shape (n_objectives,)
            rho: Augmentation parameter

        Returns:
            Scalarized scores of shape (n_points,)
        """
        weighted_diff = weights * (objectives - reference_point)
        max_term = np.max(weighted_diff, axis=1)
        sum_term = rho * np.sum(weighted_diff, axis=1)

        return max_term + sum_term

    @staticmethod
    def penalty_boundary_intersection(
        objectives: np.ndarray,
        weights: np.ndarray,
        ideal_point: np.ndarray,
        theta: float = 5.0,
    ) -> np.ndarray:
        """
        Penalty Boundary Intersection (PBI) scalarization.

        Args:
            objectives: Array of shape (n_points, n_objectives)
            weights: Array of shape (n_objectives,)
            ideal_point: Array of shape (n_objectives,)
            theta: Penalty parameter

        Returns:
            Scalarized scores of shape (n_points,)
        """
        # Normalize weights
        weights_norm = weights / np.linalg.norm(weights)

        # Translate objectives
        translated_obj = objectives - ideal_point

        # Calculate d1 (distance along weight vector)
        d1 = np.abs(np.sum(translated_obj * weights_norm, axis=1))

        # Calculate d2 (distance perpendicular to weight vector)
        proj = (
            np.sum(translated_obj * weights_norm, axis=1, keepdims=True) * weights_norm
        )
        d2 = np.linalg.norm(translated_obj - proj, axis=1)

        return d1 + theta * d2


class InteractiveVisualization:
    """Interactive visualization tools for multi-objective optimization."""

    def __init__(self):
        """Initialize visualization tools."""
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available. Interactive visualization disabled.")

    def plot_pareto_front_2d(
        self,
        objectives: np.ndarray,
        pareto_indices: np.ndarray,
        objective_names: List[str],
        molecules: Optional[List[str]] = None,
    ) -> Optional[go.Figure]:
        """
        Plot 2D Pareto front.

        Args:
            objectives: Array of shape (n_points, 2)
            pareto_indices: Indices of Pareto optimal solutions
            objective_names: Names of the two objectives
            molecules: Optional list of molecule identifiers

        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE or objectives.shape[1] != 2:
            return None

        fig = go.Figure()

        # Plot all points
        hover_text = (
            molecules
            if molecules
            else [f"Molecule {i}" for i in range(len(objectives))]
        )

        fig.add_trace(
            go.Scatter(
                x=objectives[:, 0],
                y=objectives[:, 1],
                mode="markers",
                marker=dict(color="lightblue", size=8, opacity=0.6),
                text=hover_text,
                hovertemplate=f"{objective_names[0]}: %{{x}}<br>{objective_names[1]}: %{{y}}<br>%{{text}}<extra></extra>",
                name="All Solutions",
            )
        )

        # Highlight Pareto front
        pareto_objectives = objectives[pareto_indices]
        pareto_hover = [hover_text[i] for i in pareto_indices]

        fig.add_trace(
            go.Scatter(
                x=pareto_objectives[:, 0],
                y=pareto_objectives[:, 1],
                mode="markers+lines",
                marker=dict(color="red", size=12, symbol="diamond"),
                line=dict(color="red", width=2),
                text=pareto_hover,
                hovertemplate=f"{objective_names[0]}: %{{x}}<br>{objective_names[1]}: %{{y}}<br>%{{text}}<extra></extra>",
                name="Pareto Front",
            )
        )

        fig.update_layout(
            title="2D Pareto Front Visualization",
            xaxis_title=objective_names[0],
            yaxis_title=objective_names[1],
            hovermode="closest",
            showlegend=True,
        )

        return fig

    def plot_pareto_front_3d(
        self,
        objectives: np.ndarray,
        pareto_indices: np.ndarray,
        objective_names: List[str],
        molecules: Optional[List[str]] = None,
    ) -> Optional[go.Figure]:
        """
        Plot 3D Pareto front.

        Args:
            objectives: Array of shape (n_points, 3)
            pareto_indices: Indices of Pareto optimal solutions
            objective_names: Names of the three objectives
            molecules: Optional list of molecule identifiers

        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE or objectives.shape[1] != 3:
            return None

        fig = go.Figure()

        # Plot all points
        hover_text = (
            molecules
            if molecules
            else [f"Molecule {i}" for i in range(len(objectives))]
        )

        fig.add_trace(
            go.Scatter3d(
                x=objectives[:, 0],
                y=objectives[:, 1],
                z=objectives[:, 2],
                mode="markers",
                marker=dict(color="lightblue", size=5, opacity=0.6),
                text=hover_text,
                hovertemplate=f"{objective_names[0]}: %{{x}}<br>{objective_names[1]}: %{{y}}<br>{objective_names[2]}: %{{z}}<br>%{{text}}<extra></extra>",
                name="All Solutions",
            )
        )

        # Highlight Pareto front
        pareto_objectives = objectives[pareto_indices]
        pareto_hover = [hover_text[i] for i in pareto_indices]

        fig.add_trace(
            go.Scatter3d(
                x=pareto_objectives[:, 0],
                y=pareto_objectives[:, 1],
                z=pareto_objectives[:, 2],
                mode="markers",
                marker=dict(color="red", size=8, symbol="diamond"),
                text=pareto_hover,
                hovertemplate=f"{objective_names[0]}: %{{x}}<br>{objective_names[1]}: %{{y}}<br>{objective_names[2]}: %{{z}}<br>%{{text}}<extra></extra>",
                name="Pareto Front",
            )
        )

        fig.update_layout(
            title="3D Pareto Front Visualization",
            scene=dict(
                xaxis_title=objective_names[0],
                yaxis_title=objective_names[1],
                zaxis_title=objective_names[2],
            ),
            showlegend=True,
        )

        return fig

    def plot_parallel_coordinates(
        self,
        objectives: np.ndarray,
        pareto_indices: np.ndarray,
        objective_names: List[str],
        molecules: Optional[List[str]] = None,
    ) -> Optional[go.Figure]:
        """
        Plot parallel coordinates for high-dimensional objectives.

        Args:
            objectives: Array of shape (n_points, n_objectives)
            pareto_indices: Indices of Pareto optimal solutions
            objective_names: Names of objectives
            molecules: Optional list of molecule identifiers

        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Create dimensions for parallel coordinates
        dimensions = []
        for i, name in enumerate(objective_names):
            dimensions.append(
                dict(
                    range=[objectives[:, i].min(), objectives[:, i].max()],
                    label=name,
                    values=objectives[:, i],
                )
            )

        # Color by Pareto membership
        colors = np.zeros(len(objectives))
        colors[pareto_indices] = 1

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=colors,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Pareto Front"),
                ),
                dimensions=dimensions,
            )
        )

        fig.update_layout(
            title="Parallel Coordinates Plot - Multi-Objective Trade-offs"
        )

        return fig


class MPOptimizer:
    """
    Multi-objective optimization framework with Pareto front identification.

    This class provides comprehensive multi-objective optimization capabilities
    including GPU-accelerated algorithms, interactive visualization, and
    scalarization functions for generative model guidance.
    """

    def __init__(
        self,
        objectives: List[ObjectiveConfig],
        use_gpu: bool = True,
        reference_point: Optional[np.ndarray] = None,
    ):
        """
        Initialize multi-objective optimizer.

        Args:
            objectives: List of objective configurations
            use_gpu: Whether to use GPU acceleration
            reference_point: Reference point for hypervolume calculation
        """
        self.objectives = objectives
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        self._validate_objectives()

        # Initialize components
        self.non_dominated_sorting = NonDominatedSorting(use_gpu=self.use_gpu)
        self.crowding_distance = CrowdingDistance(use_gpu=self.use_gpu)
        self.hypervolume_calculator = HypervolumeCalculator(reference_point)
        self.scalarization = ScalarizationFunctions()
        self.visualization = InteractiveVisualization()

        # Optimization state
        self.optimization_history = []
        self.best_hypervolume = -np.inf

    def _validate_objectives(self):
        """Validate objective configurations."""
        if not self.objectives:
            raise MultiObjectiveOptimizationError(
                "At least one objective must be specified"
            )

        for obj in self.objectives:
            if not isinstance(obj, ObjectiveConfig):
                raise MultiObjectiveOptimizationError(
                    "All objectives must be ObjectiveConfig instances"
                )

    def optimize(
        self,
        molecules: List[str],
        properties: Dict[str, np.ndarray],
        population_size: Optional[int] = None,
    ) -> MultiObjectiveResult:
        """
        Perform multi-objective optimization.

        Args:
            molecules: List of molecule identifiers
            properties: Dictionary mapping property names to arrays
            population_size: Optional population size limit

        Returns:
            MultiObjectiveResult with optimization results
        """
        start_time = time.time()

        # Extract and normalize objective values
        objective_values = self._extract_objective_values(properties)
        objective_values = self._normalize_objectives(objective_values)

        # Apply population size limit if specified
        if population_size and len(molecules) > population_size:
            indices = np.random.choice(len(molecules), population_size, replace=False)
            molecules = [molecules[i] for i in indices]
            objective_values = objective_values[indices]

        # Find Pareto front
        pareto_result = self._find_pareto_front(molecules, objective_values)

        # Calculate scalarized scores
        scalarized_scores = self._calculate_scalarized_scores(objective_values)

        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(
            objective_values, pareto_result
        )

        computation_time = time.time() - start_time

        # Create result
        result = MultiObjectiveResult(
            molecules=molecules,
            objective_values=objective_values,
            pareto_front=pareto_result,
            scalarized_scores=scalarized_scores,
            diversity_metrics=diversity_metrics,
            convergence_metrics={"computation_time": computation_time},
        )

        # Update optimization history
        self.optimization_history.append(result)

        return result

    def _extract_objective_values(
        self, properties: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Extract objective values from properties."""
        n_molecules = len(next(iter(properties.values())))
        n_objectives = len(self.objectives)

        objective_values = np.zeros((n_molecules, n_objectives))

        for i, obj_config in enumerate(self.objectives):
            if obj_config.name not in properties:
                raise MultiObjectiveOptimizationError(
                    f"Property '{obj_config.name}' not found in properties"
                )

            values = properties[obj_config.name].copy()

            # Apply target transformation if specified
            if obj_config.target is not None:
                values = np.abs(values - obj_config.target)

            # Apply minimization/maximization
            if not obj_config.minimize:
                values = -values

            # Apply weight
            values = obj_config.weight * values

            objective_values[:, i] = values

        return objective_values

    def _normalize_objectives(self, objective_values: np.ndarray) -> np.ndarray:
        """Normalize objective values based on configuration."""
        normalized_values = objective_values.copy()

        for i, obj_config in enumerate(self.objectives):
            if obj_config.normalization == "minmax":
                min_val = np.min(objective_values[:, i])
                max_val = np.max(objective_values[:, i])
                if max_val > min_val:
                    normalized_values[:, i] = (objective_values[:, i] - min_val) / (
                        max_val - min_val
                    )

            elif obj_config.normalization == "zscore":
                mean_val = np.mean(objective_values[:, i])
                std_val = np.std(objective_values[:, i])
                if std_val > 0:
                    normalized_values[:, i] = (
                        objective_values[:, i] - mean_val
                    ) / std_val

        return normalized_values

    def _find_pareto_front(
        self, molecules: List[str], objective_values: np.ndarray
    ) -> ParetoFrontResult:
        """Find Pareto front using non-dominated sorting."""
        start_time = time.time()

        # Perform non-dominated sorting
        fronts = self.non_dominated_sorting.fast_non_dominated_sort(objective_values)

        if not fronts or len(fronts[0]) == 0:
            raise MultiObjectiveOptimizationError("No Pareto optimal solutions found")

        pareto_indices = fronts[0]
        pareto_molecules = [molecules[i] for i in pareto_indices]
        pareto_objectives = objective_values[pareto_indices]

        # Calculate crowding distances
        crowding_distances = self.crowding_distance.calculate_crowding_distance(
            objective_values, pareto_indices
        )

        # Calculate hypervolume
        hypervolume = self.hypervolume_calculator.calculate_hypervolume(
            objective_values, pareto_indices
        )

        # Find dominated solutions
        all_indices = set(range(len(molecules)))
        pareto_set = set(pareto_indices)
        dominated_indices = np.array(list(all_indices - pareto_set))

        computation_time = time.time() - start_time

        return ParetoFrontResult(
            pareto_indices=pareto_indices,
            pareto_molecules=pareto_molecules,
            pareto_objectives=pareto_objectives,
            dominated_indices=dominated_indices,
            crowding_distances=crowding_distances,
            hypervolume=hypervolume,
            computation_time=computation_time,
        )

    def _calculate_scalarized_scores(self, objective_values: np.ndarray) -> np.ndarray:
        """Calculate scalarized scores using weighted sum."""
        weights = np.array([obj.weight for obj in self.objectives])
        weights = weights / np.sum(weights)  # Normalize weights

        return self.scalarization.weighted_sum(objective_values, weights)

    def _calculate_diversity_metrics(
        self, objective_values: np.ndarray, pareto_result: ParetoFrontResult
    ) -> Dict[str, float]:
        """Calculate diversity metrics for the solution set."""
        metrics = {}

        # Spacing metric (uniformity of distribution)
        if len(pareto_result.pareto_indices) > 1:
            pareto_obj = pareto_result.pareto_objectives
            distances = []

            for i in range(len(pareto_obj)):
                min_dist = float("inf")
                for j in range(len(pareto_obj)):
                    if i != j:
                        dist = np.linalg.norm(pareto_obj[i] - pareto_obj[j])
                        min_dist = min(min_dist, dist)
                distances.append(min_dist)

            mean_distance = np.mean(distances)
            spacing = np.sqrt(np.mean([(d - mean_distance) ** 2 for d in distances]))
            metrics["spacing"] = spacing
        else:
            metrics["spacing"] = 0.0

        # Spread metric (extent of Pareto front)
        if len(pareto_result.pareto_objectives) > 0:
            obj_ranges = np.max(pareto_result.pareto_objectives, axis=0) - np.min(
                pareto_result.pareto_objectives, axis=0
            )
            metrics["spread"] = np.mean(obj_ranges)
        else:
            metrics["spread"] = 0.0

        # Hypervolume
        metrics["hypervolume"] = pareto_result.hypervolume

        # Number of Pareto solutions
        metrics["n_pareto_solutions"] = len(pareto_result.pareto_indices)

        return metrics

    def get_scalarization_guidance(
        self, objective_values: np.ndarray, method: str = "weighted_sum", **kwargs
    ) -> np.ndarray:
        """
        Get scalarization scores for generative model guidance.

        Args:
            objective_values: Array of shape (n_points, n_objectives)
            method: Scalarization method ('weighted_sum', 'tchebycheff', 'asf', 'pbi')
            **kwargs: Additional parameters for scalarization method

        Returns:
            Scalarized scores for guiding generative models
        """
        weights = kwargs.get("weights", np.ones(len(self.objectives)))
        weights = weights / np.sum(weights)  # Normalize

        if method == "weighted_sum":
            return self.scalarization.weighted_sum(objective_values, weights)

        elif method == "tchebycheff":
            ideal_point = kwargs.get("ideal_point", np.min(objective_values, axis=0))
            return self.scalarization.tchebycheff(
                objective_values, weights, ideal_point
            )

        elif method == "asf":
            reference_point = kwargs.get(
                "reference_point", np.mean(objective_values, axis=0)
            )
            rho = kwargs.get("rho", 0.001)
            return self.scalarization.achievement_scalarizing_function(
                objective_values, weights, reference_point, rho
            )

        elif method == "pbi":
            ideal_point = kwargs.get("ideal_point", np.min(objective_values, axis=0))
            theta = kwargs.get("theta", 5.0)
            return self.scalarization.penalty_boundary_intersection(
                objective_values, weights, ideal_point, theta
            )

        else:
            raise MultiObjectiveOptimizationError(
                f"Unknown scalarization method: {method}"
            )

    def visualize_tradeoffs(self, result: MultiObjectiveResult) -> Dict[str, Any]:
        """
        Create interactive visualizations for trade-off analysis.

        Args:
            result: MultiObjectiveResult from optimization

        Returns:
            Dictionary of visualization figures
        """
        objective_names = [obj.name for obj in self.objectives]
        visualizations = {}

        # 2D Pareto front
        if len(self.objectives) == 2:
            fig_2d = self.visualization.plot_pareto_front_2d(
                result.objective_values,
                result.pareto_front.pareto_indices,
                objective_names,
                result.molecules,
            )
            if fig_2d:
                visualizations["pareto_front_2d"] = fig_2d

        # 3D Pareto front
        elif len(self.objectives) == 3:
            fig_3d = self.visualization.plot_pareto_front_3d(
                result.objective_values,
                result.pareto_front.pareto_indices,
                objective_names,
                result.molecules,
            )
            if fig_3d:
                visualizations["pareto_front_3d"] = fig_3d

        # Parallel coordinates for high-dimensional objectives
        if len(self.objectives) > 3:
            fig_parallel = self.visualization.plot_parallel_coordinates(
                result.objective_values,
                result.pareto_front.pareto_indices,
                objective_names,
                result.molecules,
            )
            if fig_parallel:
                visualizations["parallel_coordinates"] = fig_parallel

        return visualizations

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of multi-objective optimization results."""
        if not self.optimization_history:
            return {"message": "No optimization runs completed"}

        latest_result = self.optimization_history[-1]

        summary = {
            "n_objectives": len(self.objectives),
            "objective_names": [obj.name for obj in self.objectives],
            "n_molecules": len(latest_result.molecules),
            "n_pareto_solutions": len(latest_result.pareto_front.pareto_indices),
            "hypervolume": latest_result.pareto_front.hypervolume,
            "diversity_metrics": latest_result.diversity_metrics,
            "computation_time": latest_result.convergence_metrics.get(
                "computation_time", 0
            ),
            "gpu_acceleration": self.use_gpu,
        }

        # Hypervolume history
        if len(self.optimization_history) > 1:
            hypervolumes = [
                result.pareto_front.hypervolume for result in self.optimization_history
            ]
            summary["hypervolume_history"] = hypervolumes
            summary["hypervolume_improvement"] = hypervolumes[-1] - hypervolumes[0]

        return summary

    def export_pareto_front(
        self,
        result: MultiObjectiveResult,
        filepath: str,
        include_properties: bool = True,
    ) -> None:
        """
        Export Pareto front solutions to file.

        Args:
            result: MultiObjectiveResult from optimization
            filepath: Output file path
            include_properties: Whether to include all objective values
        """
        pareto_data = {
            "molecule": result.pareto_front.pareto_molecules,
            "crowding_distance": result.pareto_front.crowding_distances,
        }

        # Add objective values
        for i, obj in enumerate(self.objectives):
            pareto_data[
                f"objective_{obj.name}"
            ] = result.pareto_front.pareto_objectives[:, i]

        # Add scalarized scores if available
        if result.scalarized_scores is not None:
            pareto_indices = result.pareto_front.pareto_indices
            pareto_data["scalarized_score"] = result.scalarized_scores[pareto_indices]

        df = pd.DataFrame(pareto_data)

        if filepath.endswith(".csv"):
            df.to_csv(filepath, index=False)
        elif filepath.endswith(".json"):
            df.to_json(filepath, orient="records", indent=2)
        else:
            raise MultiObjectiveOptimizationError(
                "Unsupported file format. Use .csv or .json"
            )

        print(f"Pareto front exported to {filepath}")


# Utility functions
def create_drug_discovery_objectives() -> List[ObjectiveConfig]:
    """Create standard drug discovery objectives."""
    return [
        ObjectiveConfig(
            name="molecular_weight",
            minimize=True,
            weight=1.0,
            target=350.0,
            normalization="minmax",
        ),
        ObjectiveConfig(
            name="logp", minimize=True, weight=1.0, target=2.5, normalization="minmax"
        ),
        ObjectiveConfig(
            name="tpsa", minimize=True, weight=1.0, target=70.0, normalization="minmax"
        ),
        ObjectiveConfig(
            name="synthetic_accessibility",
            minimize=True,
            weight=2.0,
            normalization="minmax",
        ),
    ]


def create_polymer_objectives() -> List[ObjectiveConfig]:
    """Create polymer-specific objectives."""
    return [
        ObjectiveConfig(
            name="glass_transition_temp",
            minimize=False,
            weight=1.0,
            target=100.0,
            normalization="zscore",
        ),
        ObjectiveConfig(
            name="youngs_modulus", minimize=False, weight=1.0, normalization="zscore"
        ),
        ObjectiveConfig(
            name="synthetic_accessibility",
            minimize=True,
            weight=1.5,
            normalization="minmax",
        ),
    ]


def run_multi_objective_optimization(
    molecules: List[str],
    properties: Dict[str, np.ndarray],
    objectives: List[ObjectiveConfig],
    use_gpu: bool = True,
    population_size: Optional[int] = None,
) -> MultiObjectiveResult:
    """
    Run complete multi-objective optimization workflow.

    Args:
        molecules: List of molecule identifiers
        properties: Dictionary of molecular properties
        objectives: List of objective configurations
        use_gpu: Whether to use GPU acceleration
        population_size: Optional population size limit

    Returns:
        MultiObjectiveResult with optimization results
    """
    optimizer = MPOptimizer(objectives=objectives, use_gpu=use_gpu)
    result = optimizer.optimize(molecules, properties, population_size)

    print("Multi-objective optimization completed:")
    print(f"  - {len(result.molecules)} molecules evaluated")
    print(
        f"  - {len(result.pareto_front.pareto_indices)} Pareto optimal solutions found"
    )
    print(f"  - Hypervolume: {result.pareto_front.hypervolume:.4f}")
    print(
        f"  - Computation time: {result.convergence_metrics['computation_time']:.2f}s"
    )

    return result

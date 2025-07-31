"""
Process Parameter Space Exploration

Advanced algorithms for exploring manufacturing process parameter spaces,
including adaptive sampling, multi-objective optimization, and uncertainty-aware
exploration strategies specifically designed for RL-guided process optimization.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import differential_evolution, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from ..core.exceptions import ChemiAccelError
from .manufacturing_integration import ProcessConditions, ProcessOutput, ProcessType
from .rl_process_optimization import OptimizationObjective, ProcessAction, ProcessState

logger = logging.getLogger(__name__)


class ExplorationStrategy(Enum):
    """Parameter space exploration strategies."""

    RANDOM = "random"  # Random sampling
    LATIN_HYPERCUBE = "latin_hypercube"  # Latin Hypercube Sampling
    SOBOL = "sobol"  # Sobol sequences
    GAUSSIAN_PROCESS = "gaussian_process"  # GP-guided exploration
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"  # Uncertainty-based sampling
    ADAPTIVE_GRID = "adaptive_grid"  # Adaptive grid refinement
    MULTI_OBJECTIVE = "multi_objective"  # Multi-objective exploration
    HIERARCHICAL = "hierarchical"  # Hierarchical parameter exploration


@dataclass
class ExplorationConfig:
    """Configuration for parameter space exploration."""

    strategy: ExplorationStrategy = ExplorationStrategy.GAUSSIAN_PROCESS

    # Sampling parameters
    num_initial_samples: int = 50
    num_exploration_samples: int = 200
    batch_size: int = 10

    # Gaussian Process parameters
    gp_kernel: str = "matern"
    gp_length_scale: float = 1.0
    gp_noise_level: float = 0.1
    gp_alpha: float = 1e-10

    # Uncertainty sampling
    uncertainty_threshold: float = 0.1
    exploitation_factor: float = 0.3
    exploration_factor: float = 0.7

    # Adaptive grid
    initial_grid_resolution: int = 5
    max_grid_resolution: int = 20
    refinement_threshold: float = 0.05

    # Multi-objective
    pareto_front_size: int = 100
    diversity_weight: float = 0.2

    # Hierarchical exploration
    hierarchy_levels: int = 3
    level_sample_ratios: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])

    # Convergence criteria
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    patience: int = 50


@dataclass
class ParameterSpace:
    """Definition of parameter space for exploration."""

    parameters: Dict[str, Tuple[float, float]]  # parameter: (min, max)
    constraints: List[Callable[[Dict[str, float]], bool]] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    priorities: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate parameter space definition."""
        if not self.parameters:
            raise ChemiAccelError("Parameter space must define at least one parameter")

        for param, (min_val, max_val) in self.parameters.items():
            if min_val >= max_val:
                raise ChemiAccelError(
                    f"Invalid range for parameter {param}: {min_val} >= {max_val}"
                )

        # Set default priorities
        for param in self.parameters:
            if param not in self.priorities:
                self.priorities[param] = 1.0

    @property
    def dimension(self) -> int:
        """Get dimensionality of parameter space."""
        return len(self.parameters)

    @property
    def parameter_names(self) -> List[str]:
        """Get list of parameter names."""
        return list(self.parameters.keys())

    def normalize_parameters(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range."""
        normalized = []
        for param_name in self.parameter_names:
            min_val, max_val = self.parameters[param_name]
            value = params.get(param_name, (min_val + max_val) / 2)
            normalized_value = (value - min_val) / (max_val - min_val)
            normalized.append(np.clip(normalized_value, 0, 1))
        return np.array(normalized)

    def denormalize_parameters(self, normalized_params: np.ndarray) -> Dict[str, float]:
        """Denormalize parameters from [0, 1] range."""
        params = {}
        for i, param_name in enumerate(self.parameter_names):
            min_val, max_val = self.parameters[param_name]
            normalized_value = np.clip(normalized_params[i], 0, 1)
            value = min_val + normalized_value * (max_val - min_val)
            params[param_name] = value
        return params

    def is_valid(self, params: Dict[str, float]) -> bool:
        """Check if parameters satisfy all constraints."""
        # Check bounds
        for param_name, value in params.items():
            if param_name in self.parameters:
                min_val, max_val = self.parameters[param_name]
                if not (min_val <= value <= max_val):
                    return False

        # Check custom constraints
        for constraint in self.constraints:
            if not constraint(params):
                return False

        return True

    def sample_random(self, num_samples: int = 1) -> List[Dict[str, float]]:
        """Generate random samples from parameter space."""
        samples = []
        for _ in range(num_samples):
            sample = {}
            for param_name, (min_val, max_val) in self.parameters.items():
                sample[param_name] = np.random.uniform(min_val, max_val)

            if self.is_valid(sample):
                samples.append(sample)
            elif len(samples) == 0:  # Ensure at least one sample
                samples.append(sample)  # Add even if invalid for robustness

        return samples


class ParameterExplorer(ABC):
    """Abstract base class for parameter space exploration."""

    def __init__(self, parameter_space: ParameterSpace, config: ExplorationConfig):
        self.parameter_space = parameter_space
        self.config = config

        # Exploration history
        self.sampled_parameters = []
        self.observed_responses = []
        self.exploration_history = []

        # Performance tracking
        self.best_parameters = None
        self.best_response = -float("inf")
        self.convergence_history = []

    @abstractmethod
    def suggest_parameters(self, num_suggestions: int = 1) -> List[Dict[str, float]]:
        """Suggest next parameters to evaluate."""
        pass

    @abstractmethod
    def update(self, parameters: Dict[str, float], response: float):
        """Update explorer with new observation."""
        pass

    def explore(
        self,
        evaluation_function: Callable[[Dict[str, float]], float],
        num_iterations: int = None,
    ) -> Dict[str, Any]:
        """Run parameter space exploration."""
        if num_iterations is None:
            num_iterations = self.config.max_iterations

        logger.info(f"Starting parameter exploration with {self.config.strategy.value}")

        # Initial exploration
        if not self.sampled_parameters:
            initial_params = self._get_initial_samples()
            for params in initial_params:
                response = evaluation_function(params)
                self.update(params, response)

        # Iterative exploration
        for iteration in range(num_iterations):
            # Suggest new parameters
            suggestions = self.suggest_parameters(self.config.batch_size)

            # Evaluate suggestions
            for params in suggestions:
                response = evaluation_function(params)
                self.update(params, response)

            # Check convergence
            if self._check_convergence():
                logger.info(f"Converged after {iteration + 1} iterations")
                break

            # Log progress
            if iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}: Best response = {self.best_response:.4f}"
                )

        return self.get_exploration_summary()

    def _get_initial_samples(self) -> List[Dict[str, float]]:
        """Get initial samples for exploration."""
        return self.parameter_space.sample_random(self.config.num_initial_samples)

    def _check_convergence(self) -> bool:
        """Check if exploration has converged."""
        if len(self.convergence_history) < self.config.patience:
            return False

        recent_improvement = max(
            self.convergence_history[-self.config.patience :]
        ) - min(self.convergence_history[-self.config.patience :])

        return recent_improvement < self.config.convergence_tolerance

    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get summary of exploration results."""
        return {
            "strategy": self.config.strategy.value,
            "total_evaluations": len(self.sampled_parameters),
            "best_parameters": self.best_parameters,
            "best_response": self.best_response,
            "convergence_history": self.convergence_history,
            "parameter_importance": self._calculate_parameter_importance(),
            "exploration_efficiency": self._calculate_exploration_efficiency(),
        }

    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate importance of each parameter."""
        if len(self.sampled_parameters) < 10:
            return dict.fromkeys(self.parameter_space.parameter_names, 1.0)

        # Simple variance-based importance
        param_importance = {}

        for param_name in self.parameter_space.parameter_names:
            param_values = [params[param_name] for params in self.sampled_parameters]
            param_variance = np.var(param_values)

            # Correlation with response
            response_correlation = abs(
                np.corrcoef(param_values, self.observed_responses)[0, 1]
            )
            if np.isnan(response_correlation):
                response_correlation = 0.0

            param_importance[param_name] = param_variance * response_correlation

        # Normalize
        total_importance = sum(param_importance.values())
        if total_importance > 0:
            param_importance = {
                k: v / total_importance for k, v in param_importance.items()
            }

        return param_importance

    def _calculate_exploration_efficiency(self) -> float:
        """Calculate exploration efficiency."""
        if len(self.observed_responses) < 2:
            return 0.0

        # Calculate improvement rate
        sorted_responses = sorted(self.observed_responses)
        top_10_percent = int(len(sorted_responses) * 0.1) or 1
        improvement = np.mean(sorted_responses[-top_10_percent:]) - np.mean(
            sorted_responses[:top_10_percent]
        )

        # Normalize by number of evaluations
        efficiency = improvement / len(self.observed_responses)
        return max(0.0, efficiency)


class GaussianProcessExplorer(ParameterExplorer):
    """Gaussian Process-guided parameter exploration."""

    def __init__(self, parameter_space: ParameterSpace, config: ExplorationConfig):
        super().__init__(parameter_space, config)

        # Initialize Gaussian Process
        if config.gp_kernel == "rbf":
            kernel = RBF(length_scale=config.gp_length_scale)
        elif config.gp_kernel == "matern":
            kernel = Matern(length_scale=config.gp_length_scale, nu=2.5)
        else:
            kernel = RBF(length_scale=config.gp_length_scale)

        kernel += WhiteKernel(noise_level=config.gp_noise_level)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=config.gp_alpha,
            n_restarts_optimizer=5,
            normalize_y=True,
        )

        self.scaler = StandardScaler()
        self.is_fitted = False

    def suggest_parameters(self, num_suggestions: int = 1) -> List[Dict[str, float]]:
        """Suggest parameters using acquisition function."""
        if not self.is_fitted:
            return self.parameter_space.sample_random(num_suggestions)

        suggestions = []

        for _ in range(num_suggestions):
            # Optimize acquisition function
            best_params = self._optimize_acquisition()
            suggestions.append(best_params)

        return suggestions

    def update(self, parameters: Dict[str, float], response: float):
        """Update GP with new observation."""
        self.sampled_parameters.append(parameters)
        self.observed_responses.append(response)

        if response > self.best_response:
            self.best_response = response
            self.best_parameters = parameters

        self.convergence_history.append(self.best_response)

        # Refit GP
        if len(self.sampled_parameters) >= 3:
            self._fit_gp()

    def _fit_gp(self):
        """Fit Gaussian Process to current data."""
        # Prepare data
        X = np.array(
            [
                self.parameter_space.normalize_parameters(params)
                for params in self.sampled_parameters
            ]
        )
        y = np.array(self.observed_responses)

        # Fit scaler and GP
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        self.gp.fit(X_scaled, y)
        self.is_fitted = True

    def _optimize_acquisition(self) -> Dict[str, float]:
        """Optimize acquisition function to find next sample point."""

        def acquisition_function(x):
            """Expected Improvement acquisition function."""
            x_scaled = self.scaler.transform(x.reshape(1, -1))
            mean, std = self.gp.predict(x_scaled, return_std=True)

            if std[0] == 0:
                return 0.0

            # Expected Improvement
            z = (mean[0] - self.best_response) / std[0]
            ei = std[0] * (z * self._normal_cdf(z) + self._normal_pdf(z))

            # Add exploration bonus
            exploration_bonus = self.config.exploration_factor * std[0]

            return -(ei + exploration_bonus)  # Negative for minimization

        # Multi-start optimization
        best_x = None
        best_acq = float("inf")

        for _ in range(10):  # Multiple random starts
            x0 = np.random.uniform(0, 1, self.parameter_space.dimension)

            try:
                result = minimize(
                    acquisition_function,
                    x0,
                    bounds=[(0, 1)] * self.parameter_space.dimension,
                    method="L-BFGS-B",
                )

                if result.fun < best_acq:
                    best_acq = result.fun
                    best_x = result.x
            except Exception as e:
                logger.debug(f"Optimization failed: {e}")
                continue

        if best_x is None:
            # Fallback to random sampling
            best_x = np.random.uniform(0, 1, self.parameter_space.dimension)

        return self.parameter_space.denormalize_parameters(best_x)

    def _normal_cdf(self, x):
        """Standard normal CDF."""
        return 0.5 * (1 + np.tanh(x / np.sqrt(2)))

    def _normal_pdf(self, x):
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def predict(self, parameters: Dict[str, float]) -> Tuple[float, float]:
        """Predict response and uncertainty for given parameters."""
        if not self.is_fitted:
            return 0.0, 1.0

        x = self.parameter_space.normalize_parameters(parameters).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        mean, std = self.gp.predict(x_scaled, return_std=True)
        return mean[0], std[0]


class MultiObjectiveExplorer(ParameterExplorer):
    """Multi-objective parameter space exploration."""

    def __init__(
        self,
        parameter_space: ParameterSpace,
        config: ExplorationConfig,
        objectives: List[OptimizationObjective],
    ):
        super().__init__(parameter_space, config)
        self.objectives = objectives
        self.pareto_front = []
        self.objective_responses = {obj: [] for obj in objectives}

    def suggest_parameters(self, num_suggestions: int = 1) -> List[Dict[str, float]]:
        """Suggest parameters for multi-objective optimization."""
        if len(self.pareto_front) < 10:
            # Initial random exploration
            return self.parameter_space.sample_random(num_suggestions)

        suggestions = []

        for _ in range(num_suggestions):
            # Use diversity-based selection around Pareto front
            if np.random.random() < self.config.diversity_weight:
                # Explore around diverse Pareto solutions
                base_params = np.random.choice(self.pareto_front)
                perturbed_params = self._perturb_parameters(base_params)
            else:
                # Exploit around best solutions
                best_solution = self._select_best_pareto_solution()
                perturbed_params = self._perturb_parameters(best_solution)

            suggestions.append(perturbed_params)

        return suggestions

    def update(
        self, parameters: Dict[str, float], response: Union[float, Dict[str, float]]
    ):
        """Update with multi-objective response."""
        self.sampled_parameters.append(parameters)

        if isinstance(response, dict):
            # Multi-objective response
            for obj in self.objectives:
                if obj.value in response:
                    self.objective_responses[obj].append(response[obj.value])
                else:
                    self.objective_responses[obj].append(0.0)

            # Use weighted sum for single response tracking
            weighted_response = sum(
                response.get(obj.value, 0.0) for obj in self.objectives
            ) / len(self.objectives)
        else:
            # Single objective response
            weighted_response = response
            for obj in self.objectives:
                self.objective_responses[obj].append(response)

        self.observed_responses.append(weighted_response)

        if weighted_response > self.best_response:
            self.best_response = weighted_response
            self.best_parameters = parameters

        self.convergence_history.append(self.best_response)

        # Update Pareto front
        self._update_pareto_front(parameters, response)

    def _update_pareto_front(
        self, parameters: Dict[str, float], response: Union[float, Dict[str, float]]
    ):
        """Update Pareto front with new solution."""
        if isinstance(response, dict):
            objective_values = [response.get(obj.value, 0.0) for obj in self.objectives]
        else:
            objective_values = [response] * len(self.objectives)

        solution = {"parameters": parameters, "objectives": objective_values}

        # Check if solution is dominated
        is_dominated = False
        for pareto_solution in self.pareto_front:
            if self._dominates(pareto_solution["objectives"], objective_values):
                is_dominated = True
                break

        if not is_dominated:
            # Remove dominated solutions from Pareto front
            self.pareto_front = [
                sol
                for sol in self.pareto_front
                if not self._dominates(objective_values, sol["objectives"])
            ]

            # Add new solution
            self.pareto_front.append(solution)

            # Limit Pareto front size
            if len(self.pareto_front) > self.config.pareto_front_size:
                self.pareto_front = self._select_diverse_solutions(self.pareto_front)

    def _dominates(self, objectives_a: List[float], objectives_b: List[float]) -> bool:
        """Check if solution A dominates solution B."""
        better_in_one = False
        for a, b in zip(objectives_a, objectives_b):
            if a < b:  # Worse in any objective
                return False
            elif a > b:  # Better in this objective
                better_in_one = True
        return better_in_one

    def _select_diverse_solutions(self, solutions: List[Dict]) -> List[Dict]:
        """Select diverse solutions from Pareto front."""
        if len(solutions) <= self.config.pareto_front_size:
            return solutions

        # Calculate crowding distances
        crowding_distances = []

        for i, solution in enumerate(solutions):
            distance = 0.0
            objectives = solution["objectives"]

            for j, other_solution in enumerate(solutions):
                if i != j:
                    other_objectives = other_solution["objectives"]
                    distance += np.linalg.norm(
                        np.array(objectives) - np.array(other_objectives)
                    )

            crowding_distances.append((distance, i))

        # Select solutions with highest crowding distances
        crowding_distances.sort(reverse=True)
        selected_indices = [
            idx for _, idx in crowding_distances[: self.config.pareto_front_size]
        ]

        return [solutions[i] for i in selected_indices]

    def _select_best_pareto_solution(self) -> Dict[str, float]:
        """Select best solution from Pareto front."""
        if not self.pareto_front:
            return self.parameter_space.sample_random(1)[0]

        # Use weighted sum to select best solution
        best_solution = None
        best_score = -float("inf")

        for solution in self.pareto_front:
            score = sum(solution["objectives"]) / len(solution["objectives"])
            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution["parameters"]

    def _perturb_parameters(
        self, base_parameters: Dict[str, float]
    ) -> Dict[str, float]:
        """Perturb parameters for local exploration."""
        perturbed = {}

        for param_name, value in base_parameters.items():
            min_val, max_val = self.parameter_space.parameters[param_name]
            range_size = max_val - min_val

            # Add small random perturbation (5% of range)
            perturbation = np.random.normal(0, 0.05 * range_size)
            new_value = np.clip(value + perturbation, min_val, max_val)
            perturbed[param_name] = new_value

        return perturbed

    def get_pareto_front(self) -> List[Dict]:
        """Get current Pareto front."""
        return self.pareto_front.copy()


class AdaptiveGridExplorer(ParameterExplorer):
    """Adaptive grid-based parameter exploration."""

    def __init__(self, parameter_space: ParameterSpace, config: ExplorationConfig):
        super().__init__(parameter_space, config)

        # Initialize grid
        self.grid_resolution = config.initial_grid_resolution
        self.grid_points = self._generate_grid()
        self.grid_responses = {}
        self.refined_regions = set()

    def suggest_parameters(self, num_suggestions: int = 1) -> List[Dict[str, float]]:
        """Suggest parameters using adaptive grid."""
        suggestions = []

        # Identify promising regions for refinement
        promising_regions = self._identify_promising_regions()

        for _ in range(num_suggestions):
            if promising_regions and len(suggestions) < len(promising_regions):
                # Refine promising regions
                region = promising_regions[len(suggestions)]
                refined_params = self._refine_region(region)
                suggestions.append(refined_params)
            else:
                # Explore unexplored grid points
                unexplored_points = [
                    point
                    for point in self.grid_points
                    if self._grid_key(point) not in self.grid_responses
                ]

                if unexplored_points:
                    point = np.random.choice(len(unexplored_points))
                    suggestions.append(unexplored_points[point])
                else:
                    # Fallback to random sampling
                    suggestions.append(self.parameter_space.sample_random(1)[0])

        return suggestions

    def update(self, parameters: Dict[str, float], response: float):
        """Update grid with new observation."""
        self.sampled_parameters.append(parameters)
        self.observed_responses.append(response)

        # Store response for nearest grid point
        grid_key = self._find_nearest_grid_point(parameters)
        self.grid_responses[grid_key] = response

        if response > self.best_response:
            self.best_response = response
            self.best_parameters = parameters

        self.convergence_history.append(self.best_response)

    def _generate_grid(self) -> List[Dict[str, float]]:
        """Generate initial grid points."""
        grid_points = []

        # Generate all combinations of grid points
        param_grids = {}
        for param_name, (min_val, max_val) in self.parameter_space.parameters.items():
            param_grids[param_name] = np.linspace(
                min_val, max_val, self.grid_resolution
            )

        # Create all combinations
        param_names = list(param_grids.keys())

        def generate_combinations(param_idx=0, current_combination={}):
            if param_idx == len(param_names):
                grid_points.append(current_combination.copy())
                return

            param_name = param_names[param_idx]
            for value in param_grids[param_name]:
                current_combination[param_name] = value
                generate_combinations(param_idx + 1, current_combination)
                del current_combination[param_name]

        generate_combinations()
        return grid_points

    def _grid_key(self, parameters: Dict[str, float]) -> Tuple:
        """Generate key for grid point."""
        return tuple(
            round(parameters[param], 6)
            for param in self.parameter_space.parameter_names
        )

    def _find_nearest_grid_point(self, parameters: Dict[str, float]) -> Tuple:
        """Find nearest grid point to given parameters."""
        best_distance = float("inf")
        nearest_key = None

        for grid_point in self.grid_points:
            distance = (
                sum(
                    (parameters[param] - grid_point[param]) ** 2
                    for param in self.parameter_space.parameter_names
                )
                ** 0.5
            )

            if distance < best_distance:
                best_distance = distance
                nearest_key = self._grid_key(grid_point)

        return nearest_key

    def _identify_promising_regions(self) -> List[Dict[str, float]]:
        """Identify promising regions for refinement."""
        if len(self.grid_responses) < 5:
            return []

        # Find grid points with high responses
        sorted_responses = sorted(
            self.grid_responses.items(), key=lambda x: x[1], reverse=True
        )

        # Select top regions that haven't been refined recently
        promising_regions = []
        for grid_key, response in sorted_responses[:5]:
            if grid_key not in self.refined_regions:
                # Convert grid key back to parameters
                param_dict = {}
                for i, param_name in enumerate(self.parameter_space.parameter_names):
                    param_dict[param_name] = grid_key[i]
                promising_regions.append(param_dict)

        return promising_regions

    def _refine_region(self, center_parameters: Dict[str, float]) -> Dict[str, float]:
        """Refine region around center parameters."""
        refined_params = {}

        for param_name, center_value in center_parameters.items():
            min_val, max_val = self.parameter_space.parameters[param_name]
            range_size = max_val - min_val

            # Create smaller range around center
            refinement_range = range_size / (self.grid_resolution * 2)
            refined_min = max(min_val, center_value - refinement_range)
            refined_max = min(max_val, center_value + refinement_range)

            # Sample within refined range
            refined_params[param_name] = np.random.uniform(refined_min, refined_max)

        # Mark region as refined
        self.refined_regions.add(self._grid_key(center_parameters))

        return refined_params


class ParameterExplorerFactory:
    """Factory for creating parameter explorers."""

    @staticmethod
    def create_explorer(
        strategy: ExplorationStrategy,
        parameter_space: ParameterSpace,
        config: ExplorationConfig,
        **kwargs,
    ) -> ParameterExplorer:
        """Create parameter explorer based on strategy."""
        if strategy == ExplorationStrategy.GAUSSIAN_PROCESS:
            return GaussianProcessExplorer(parameter_space, config)
        elif strategy == ExplorationStrategy.MULTI_OBJECTIVE:
            objectives = kwargs.get("objectives", [OptimizationObjective.QUALITY])
            return MultiObjectiveExplorer(parameter_space, config, objectives)
        elif strategy == ExplorationStrategy.ADAPTIVE_GRID:
            return AdaptiveGridExplorer(parameter_space, config)
        else:
            raise ChemiAccelError(f"Unsupported exploration strategy: {strategy}")


class RLParameterExplorer:
    """Integration class for RL-guided parameter exploration."""

    def __init__(
        self,
        digital_twin,
        process_type: ProcessType,
        parameter_space: ParameterSpace,
        rl_agent,
        config: ExplorationConfig = None,
    ):
        self.digital_twin = digital_twin
        self.process_type = process_type
        self.parameter_space = parameter_space
        self.rl_agent = rl_agent
        self.config = config or ExplorationConfig()

        # Initialize explorer
        self.explorer = ParameterExplorerFactory.create_explorer(
            self.config.strategy, parameter_space, config
        )

        # Exploration results
        self.exploration_results = []
        self.rl_guidance_history = []

    def explore_with_rl_guidance(
        self, num_iterations: int = 100, rl_guidance_frequency: int = 10
    ) -> Dict[str, Any]:
        """Explore parameter space with RL agent guidance."""
        logger.info("Starting RL-guided parameter exploration")

        exploration_data = []

        for iteration in range(num_iterations):
            # Get parameter suggestions from explorer
            if iteration % rl_guidance_frequency == 0 and iteration > 0:
                # Use RL agent for guidance
                suggested_params = self._get_rl_guided_parameters()
                self.rl_guidance_history.append(
                    {"iteration": iteration, "parameters": suggested_params}
                )
            else:
                # Use standard exploration
                suggestions = self.explorer.suggest_parameters(1)
                suggested_params = suggestions[0]

            # Evaluate parameters
            response = self._evaluate_parameters(suggested_params)

            # Update explorer
            self.explorer.update(suggested_params, response)

            # Store exploration data
            exploration_data.append(
                {
                    "iteration": iteration,
                    "parameters": suggested_params,
                    "response": response,
                    "guided_by_rl": iteration % rl_guidance_frequency == 0
                    and iteration > 0,
                }
            )

            # Log progress
            if iteration % 20 == 0:
                logger.info(
                    f"Exploration iteration {iteration}: Response = {response:.4f}"
                )

        # Compile results
        results = {
            "exploration_strategy": self.config.strategy.value,
            "total_iterations": num_iterations,
            "rl_guidance_frequency": rl_guidance_frequency,
            "exploration_data": exploration_data,
            "best_parameters": self.explorer.best_parameters,
            "best_response": self.explorer.best_response,
            "explorer_summary": self.explorer.get_exploration_summary(),
            "rl_guidance_impact": self._analyze_rl_guidance_impact(exploration_data),
        }

        self.exploration_results.append(results)
        return results

    def _get_rl_guided_parameters(self) -> Dict[str, float]:
        """Get parameter suggestions from RL agent."""
        # Create current process state
        current_conditions = self.digital_twin.process_conditions
        if current_conditions is None:
            # Use default conditions
            from .manufacturing_integration import ProcessConditions

            current_conditions = ProcessConditions(
                temperature=473.15,
                pressure=2e5,
                flow_rate=0.1,
                shear_rate=100.0,
                residence_time=60.0,
            )

        # Convert to ProcessState (simplified)
        from .rl_process_optimization import ProcessState

        current_state = ProcessState(
            temperature=current_conditions.temperature or 473.15,
            pressure=current_conditions.pressure or 2e5,
            flow_rate=current_conditions.flow_rate or 0.1,
            shear_rate=current_conditions.shear_rate or 100.0,
            residence_time=current_conditions.residence_time or 60.0,
            viscosity=1000.0,
            density=1000.0,
            thermal_conductivity=0.2,
            part_quality=0.5,
            dimensional_accuracy=0.5,
            surface_finish=0.5,
        )

        # Get action from RL agent
        action = self.rl_agent.select_action(current_state, training=False)

        # Convert action to parameter suggestions
        suggested_params = {}

        # Apply action deltas to current parameters
        for param_name, (min_val, max_val) in self.parameter_space.parameters.items():
            if param_name == "temperature":
                current_val = current_conditions.temperature or 473.15
                delta = action.temperature_delta
            elif param_name == "pressure":
                current_val = current_conditions.pressure or 2e5
                delta = action.pressure_delta
            elif param_name == "flow_rate":
                current_val = current_conditions.flow_rate or 0.1
                delta = action.flow_rate_delta
            elif param_name == "shear_rate":
                current_val = current_conditions.shear_rate or 100.0
                delta = action.shear_rate_delta
            elif param_name == "residence_time":
                current_val = current_conditions.residence_time or 60.0
                delta = action.residence_time_delta
            else:
                # Use middle of range for unknown parameters
                current_val = (min_val + max_val) / 2
                delta = 0.0

            # Apply delta with bounds
            new_val = current_val + delta * (max_val - min_val) * 0.1
            suggested_params[param_name] = np.clip(new_val, min_val, max_val)

        return suggested_params

    def _evaluate_parameters(self, parameters: Dict[str, float]) -> float:
        """Evaluate parameters using digital twin."""
        try:
            # Convert parameters to ProcessConditions
            from .manufacturing_integration import ProcessConditions

            conditions = ProcessConditions(
                temperature=parameters.get("temperature"),
                pressure=parameters.get("pressure"),
                flow_rate=parameters.get("flow_rate"),
                shear_rate=parameters.get("shear_rate"),
                residence_time=parameters.get("residence_time"),
            )

            # Set conditions and run simulation
            self.digital_twin.set_process_conditions(conditions)
            output = self.digital_twin.run_manufacturing_simulation(self.process_type)

            # Calculate response (weighted quality metric)
            response = (
                0.4 * output.part_quality
                + 0.3 * output.dimensional_accuracy
                + 0.2 * output.surface_finish
                + 0.1 * min(1.0, 100.0 / output.processing_time)  # Time efficiency
            )

            return response

        except Exception as e:
            logger.warning(f"Parameter evaluation failed: {e}")
            return 0.1  # Low response for failed evaluations

    def _analyze_rl_guidance_impact(
        self, exploration_data: List[Dict]
    ) -> Dict[str, float]:
        """Analyze impact of RL guidance on exploration."""
        rl_guided_responses = [
            data["response"]
            for data in exploration_data
            if data.get("guided_by_rl", False)
        ]

        non_rl_responses = [
            data["response"]
            for data in exploration_data
            if not data.get("guided_by_rl", False)
        ]

        if not rl_guided_responses or not non_rl_responses:
            return {"impact": 0.0, "message": "Insufficient data for analysis"}

        rl_mean = np.mean(rl_guided_responses)
        non_rl_mean = np.mean(non_rl_responses)

        improvement = rl_mean - non_rl_mean
        relative_improvement = improvement / non_rl_mean if non_rl_mean > 0 else 0.0

        return {
            "rl_guided_mean": rl_mean,
            "exploration_only_mean": non_rl_mean,
            "absolute_improvement": improvement,
            "relative_improvement": relative_improvement,
            "rl_guidance_effectiveness": relative_improvement
            > 0.05,  # 5% improvement threshold
        }

    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get summary of all exploration runs."""
        if not self.exploration_results:
            return {"message": "No exploration runs completed"}

        latest_result = self.exploration_results[-1]

        return {
            "total_exploration_runs": len(self.exploration_results),
            "latest_strategy": latest_result["exploration_strategy"],
            "best_parameters_overall": latest_result["best_parameters"],
            "best_response_overall": latest_result["best_response"],
            "rl_guidance_frequency": latest_result["rl_guidance_frequency"],
            "rl_guidance_effectiveness": latest_result["rl_guidance_impact"][
                "rl_guidance_effectiveness"
            ],
            "parameter_space_coverage": len(self.explorer.sampled_parameters)
            / self.parameter_space.dimension,
            "exploration_efficiency": latest_result["explorer_summary"][
                "exploration_efficiency"
            ],
        }

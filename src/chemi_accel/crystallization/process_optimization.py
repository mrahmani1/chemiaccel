"""
AI-Driven Process Parameter Optimization for Crystallization

This module implements advanced optimization algorithms for crystallization process
parameters, combining adaptive experimental design with multi-objective optimization
to achieve desired solid-state properties.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.interpolate import RBFInterpolator
    from scipy.optimize import basinhopping, differential_evolution, minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .core import (
    CrystallizationCore,
    OptimizationConstraints,
    OptimizationObjective,
    ProcessConditions,
    SimulationConfig,
)
from .kinetics_modeling import CrystallizationKineticsModel

logger = logging.getLogger(__name__)


@dataclass
class OptimizationParameter:
    """
    Single optimization parameter definition.

    Attributes:
        name: Parameter name
        bounds: Parameter bounds (min, max)
        parameter_type: Type of parameter ('continuous', 'discrete', 'categorical')
        values: Discrete/categorical values (if applicable)
        units: Parameter units
        description: Parameter description
    """

    name: str
    bounds: Tuple[float, float]
    parameter_type: str = "continuous"
    values: Optional[List[Any]] = None
    units: Optional[str] = None
    description: Optional[str] = None

    def validate_value(self, value: Any) -> bool:
        """Validate parameter value."""
        if self.parameter_type == "continuous":
            return self.bounds[0] <= value <= self.bounds[1]
        elif self.parameter_type == "discrete":
            return (
                value in self.values
                if self.values
                else self.bounds[0] <= value <= self.bounds[1]
            )
        elif self.parameter_type == "categorical":
            return value in self.values if self.values else False
        return False


@dataclass
class ExperimentResult:
    """
    Result of a crystallization experiment.

    Attributes:
        parameters: Process parameters used
        objectives: Measured/predicted objective values
        constraints_satisfied: Whether constraints were satisfied
        metadata: Additional experimental metadata
        timestamp: When experiment was conducted
        success: Whether experiment was successful
    """

    parameters: Dict[str, Any]
    objectives: Dict[str, float]
    constraints_satisfied: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    success: bool = True


class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""

    @abstractmethod
    def evaluate(self, x: np.ndarray, model: Any) -> np.ndarray:
        """Evaluate acquisition function."""
        pass


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""

    def __init__(self, best_observed: float, xi: float = 0.01):
        """
        Initialize EI acquisition function.

        Args:
            best_observed: Best observed objective value
            xi: Exploration parameter
        """
        self.best_observed = best_observed
        self.xi = xi

    def evaluate(self, x: np.ndarray, model: Any) -> np.ndarray:
        """
        Evaluate Expected Improvement.

        Args:
            x: Candidate points
            model: Surrogate model

        Returns:
            Acquisition function values
        """
        if not hasattr(model, "predict"):
            raise ValueError("Model must have predict method")

        # Get prediction and uncertainty
        if hasattr(model, "predict") and len(x.shape) == 1:
            x = x.reshape(1, -1)

        mu, sigma = model.predict(x, return_std=True)

        # Handle zero variance
        sigma = np.maximum(sigma, 1e-9)

        # Calculate improvement
        improvement = mu - self.best_observed - self.xi
        z = improvement / sigma

        # Expected improvement
        from scipy.stats import norm

        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

        return ei


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""

    def __init__(self, beta: float = 2.0):
        """
        Initialize UCB acquisition function.

        Args:
            beta: Exploration parameter
        """
        self.beta = beta

    def evaluate(self, x: np.ndarray, model: Any) -> np.ndarray:
        """
        Evaluate Upper Confidence Bound.

        Args:
            x: Candidate points
            model: Surrogate model

        Returns:
            Acquisition function values
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        mu, sigma = model.predict(x, return_std=True)

        # UCB
        ucb = mu + self.beta * sigma

        return ucb


class AdaptiveDesignEngine(CrystallizationCore):
    """
    Adaptive experimental design engine for crystallization optimization.

    Implements Bayesian optimization with Gaussian process surrogates
    for efficient exploration of the process parameter space.
    """

    def __init__(
        self,
        optimization_parameters: List[OptimizationParameter],
        objectives: List[OptimizationObjective],
        constraints: OptimizationConstraints,
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize adaptive design engine.

        Args:
            optimization_parameters: Parameters to optimize
            objectives: Optimization objectives
            constraints: Process constraints
            config: Simulation configuration
        """
        super().__init__(config)
        self.optimization_parameters = optimization_parameters
        self.objectives = objectives
        self.constraints = constraints
        self.experiment_history: List[ExperimentResult] = []
        self.surrogate_models: Dict[str, Any] = {}
        self.acquisition_function = None

    def initialize(self) -> None:
        """Initialize the adaptive design engine."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for adaptive design")

        # Initialize surrogate models for each objective
        for objective in self.objectives:
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel()
            self.surrogate_models[objective.value] = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10
            )

        # Initialize acquisition function (Expected Improvement by default)
        self.acquisition_function = ExpectedImprovement(best_observed=-np.inf)

        self._initialized = True
        logger.info("Adaptive design engine initialized")

    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        return True

    def suggest_next_experiment(self, n_candidates: int = 1000) -> Dict[str, Any]:
        """
        Suggest next experiment using acquisition function optimization.

        Args:
            n_candidates: Number of candidate points to evaluate

        Returns:
            Suggested parameter values
        """
        if not self.is_initialized():
            self.initialize()

        if len(self.experiment_history) < 2:
            # Use random sampling for initial experiments
            return self._generate_random_parameters()

        # Update surrogate models
        self._update_surrogate_models()

        # Generate candidate points
        candidates = self._generate_candidate_points(n_candidates)

        # Evaluate acquisition function
        acquisition_values = self._evaluate_acquisition_function(candidates)

        # Select best candidate
        best_idx = np.argmax(acquisition_values)
        best_candidate = candidates[best_idx]

        # Convert to parameter dictionary
        suggested_params = {}
        for i, param in enumerate(self.optimization_parameters):
            suggested_params[param.name] = best_candidate[i]

        logger.info(f"Suggested next experiment: {suggested_params}")
        return suggested_params

    def add_experiment_result(self, result: ExperimentResult) -> None:
        """Add experiment result to history."""
        self.experiment_history.append(result)

        # Update best observed value for acquisition function
        if self.objectives and len(self.experiment_history) > 0:
            objective_name = self.objectives[0].value
            objective_values = [
                exp.objectives.get(objective_name, -np.inf)
                for exp in self.experiment_history
                if exp.success
            ]
            if objective_values:
                best_value = max(objective_values)
                if isinstance(self.acquisition_function, ExpectedImprovement):
                    self.acquisition_function.best_observed = best_value

        logger.info(
            f"Added experiment result. Total experiments: {len(self.experiment_history)}"
        )

    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameter values for initial exploration."""
        random_params = {}

        for param in self.optimization_parameters:
            if param.parameter_type == "continuous":
                value = np.random.uniform(param.bounds[0], param.bounds[1])
            elif param.parameter_type == "discrete":
                if param.values:
                    value = np.random.choice(param.values)
                else:
                    value = np.random.randint(param.bounds[0], param.bounds[1] + 1)
            elif param.parameter_type == "categorical":
                if param.values:
                    value = np.random.choice(param.values)
                else:
                    raise ValueError(
                        f"Categorical parameter {param.name} must have values"
                    )
            else:
                raise ValueError(f"Unknown parameter type: {param.parameter_type}")

            random_params[param.name] = value

        return random_params

    def _generate_candidate_points(self, n_candidates: int) -> np.ndarray:
        """Generate candidate points for optimization."""
        candidates = np.zeros((n_candidates, len(self.optimization_parameters)))

        for i, param in enumerate(self.optimization_parameters):
            if param.parameter_type == "continuous":
                candidates[:, i] = np.random.uniform(
                    param.bounds[0], param.bounds[1], n_candidates
                )
            elif param.parameter_type == "discrete":
                if param.values:
                    candidates[:, i] = np.random.choice(param.values, n_candidates)
                else:
                    candidates[:, i] = np.random.randint(
                        param.bounds[0], param.bounds[1] + 1, n_candidates
                    )
            # Note: Categorical parameters need special handling

        return candidates

    def _update_surrogate_models(self) -> None:
        """Update surrogate models with experiment history."""
        if len(self.experiment_history) < 2:
            return

        # Prepare training data
        X = []
        y_dict = {obj.value: [] for obj in self.objectives}

        for exp in self.experiment_history:
            if not exp.success:
                continue

            # Convert parameters to feature vector
            x = []
            for param in self.optimization_parameters:
                value = exp.parameters.get(param.name, 0.0)
                if param.parameter_type == "categorical" and param.values:
                    # One-hot encoding for categorical
                    value = param.values.index(value) if value in param.values else 0
                x.append(value)
            X.append(x)

            # Extract objective values
            for obj in self.objectives:
                y_dict[obj.value].append(exp.objectives.get(obj.value, 0.0))

        X = np.array(X)

        # Train models
        for obj in self.objectives:
            y = np.array(y_dict[obj.value])
            if len(y) >= 2:
                self.surrogate_models[obj.value].fit(X, y)

        logger.info("Updated surrogate models")

    def _evaluate_acquisition_function(self, candidates: np.ndarray) -> np.ndarray:
        """Evaluate acquisition function for candidate points."""
        if not self.objectives:
            return np.random.random(len(candidates))

        # Use first objective for single-objective optimization
        primary_objective = self.objectives[0].value
        model = self.surrogate_models.get(primary_objective)

        if model is None or not hasattr(model, "predict"):
            return np.random.random(len(candidates))

        # Check if model is fitted
        try:
            acquisition_values = self.acquisition_function.evaluate(candidates, model)
        except Exception as e:
            logger.warning(f"Acquisition function evaluation failed: {e}")
            return np.random.random(len(candidates))

        return acquisition_values


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for crystallization processes.

    Implements NSGA-II-like algorithms for Pareto-optimal solutions.
    """

    def __init__(
        self,
        objectives: List[OptimizationObjective],
        constraints: OptimizationConstraints,
    ):
        """
        Initialize multi-objective optimizer.

        Args:
            objectives: List of optimization objectives
            constraints: Process constraints
        """
        self.objectives = objectives
        self.constraints = constraints
        self.pareto_front = []

    def optimize(
        self,
        evaluation_function: Callable,
        parameter_space: List[OptimizationParameter],
        n_generations: int = 100,
        population_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Run multi-objective optimization.

        Args:
            evaluation_function: Function to evaluate objectives
            parameter_space: Parameter space definition
            n_generations: Number of generations
            population_size: Population size

        Returns:
            Pareto-optimal solutions
        """
        # Initialize population
        population = self._initialize_population(parameter_space, population_size)

        for generation in range(n_generations):
            # Evaluate population
            objectives_matrix = []
            for individual in population:
                objectives = evaluation_function(individual)
                objectives_matrix.append(
                    [objectives[obj.value] for obj in self.objectives]
                )

            objectives_matrix = np.array(objectives_matrix)

            # Non-dominated sorting
            fronts = self._non_dominated_sorting(objectives_matrix)

            # Crowding distance calculation
            for front in fronts:
                if len(front) > 2:
                    distances = self._calculate_crowding_distance(
                        objectives_matrix[front]
                    )
                    for i, idx in enumerate(front):
                        population[idx]["crowding_distance"] = distances[i]

            # Selection for next generation
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= population_size:
                    new_population.extend([population[i] for i in front])
                else:
                    # Sort by crowding distance and select
                    remaining = population_size - len(new_population)
                    front_sorted = sorted(
                        front,
                        key=lambda i: population[i].get("crowding_distance", 0),
                        reverse=True,
                    )
                    new_population.extend(
                        [population[i] for i in front_sorted[:remaining]]
                    )
                    break

            population = new_population

            # Generate offspring (simplified)
            if generation < n_generations - 1:
                offspring = self._generate_offspring(population, parameter_space)
                population.extend(offspring)

        # Extract Pareto front
        final_objectives = []
        for individual in population:
            objectives = evaluation_function(individual)
            final_objectives.append([objectives[obj.value] for obj in self.objectives])

        final_objectives = np.array(final_objectives)
        pareto_indices = self._non_dominated_sorting(final_objectives)[0]

        self.pareto_front = [population[i] for i in pareto_indices]

        return self.pareto_front

    def _initialize_population(
        self, parameter_space: List[OptimizationParameter], size: int
    ) -> List[Dict[str, Any]]:
        """Initialize random population."""
        population = []

        for _ in range(size):
            individual = {}
            for param in parameter_space:
                if param.parameter_type == "continuous":
                    individual[param.name] = np.random.uniform(
                        param.bounds[0], param.bounds[1]
                    )
                elif param.parameter_type == "discrete":
                    if param.values:
                        individual[param.name] = np.random.choice(param.values)
                    else:
                        individual[param.name] = np.random.randint(
                            param.bounds[0], param.bounds[1] + 1
                        )
                elif param.parameter_type == "categorical":
                    if param.values:
                        individual[param.name] = np.random.choice(param.values)
            population.append(individual)

        return population

    def _non_dominated_sorting(self, objectives: np.ndarray) -> List[List[int]]:
        """Perform non-dominated sorting."""
        n = len(objectives)
        domination_counts = np.zeros(n)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_counts[i] += 1

            if domination_counts[i] == 0:
                fronts[0].append(i)

        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)

            if next_front:
                fronts.append(next_front)
            front_idx += 1

        return fronts[:-1] if not fronts[-1] else fronts

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (assuming maximization)."""
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)

    def _calculate_crowding_distance(self, objectives: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for solutions."""
        n_solutions, n_objectives = objectives.shape
        distances = np.zeros(n_solutions)

        for obj_idx in range(n_objectives):
            sorted_indices = np.argsort(objectives[:, obj_idx])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            obj_range = (
                objectives[sorted_indices[-1], obj_idx]
                - objectives[sorted_indices[0], obj_idx]
            )
            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    idx = sorted_indices[i]
                    distances[idx] += (
                        objectives[sorted_indices[i + 1], obj_idx]
                        - objectives[sorted_indices[i - 1], obj_idx]
                    ) / obj_range

        return distances

    def _generate_offspring(
        self,
        population: List[Dict[str, Any]],
        parameter_space: List[OptimizationParameter],
    ) -> List[Dict[str, Any]]:
        """Generate offspring through crossover and mutation."""
        offspring = []
        n_offspring = len(population) // 2

        for _ in range(n_offspring):
            # Simple tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # Crossover
            child = self._crossover(parent1, parent2, parameter_space)

            # Mutation
            child = self._mutate(child, parameter_space)

            offspring.append(child)

        return offspring

    def _tournament_selection(self, population: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tournament selection."""
        tournament_size = 3
        tournament = np.random.choice(len(population), tournament_size, replace=False)

        # Select best based on crowding distance (simplified)
        best_idx = tournament[0]
        best_distance = population[best_idx].get("crowding_distance", 0)

        for idx in tournament[1:]:
            distance = population[idx].get("crowding_distance", 0)
            if distance > best_distance:
                best_idx = idx
                best_distance = distance

        return population[best_idx].copy()

    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        parameter_space: List[OptimizationParameter],
    ) -> Dict[str, Any]:
        """Simulated binary crossover."""
        child = {}

        for param in parameter_space:
            if param.parameter_type == "continuous":
                # SBX crossover
                if np.random.random() < 0.9:  # Crossover probability
                    eta = 2.0  # Distribution index
                    u = np.random.random()

                    if u <= 0.5:
                        beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

                    val1 = parent1[param.name]
                    val2 = parent2[param.name]

                    child_val = 0.5 * ((val1 + val2) - beta * abs(val2 - val1))
                    child_val = np.clip(child_val, param.bounds[0], param.bounds[1])
                    child[param.name] = child_val
                else:
                    child[param.name] = parent1[param.name]
            else:
                # Random selection for discrete/categorical
                child[param.name] = np.random.choice(
                    [parent1[param.name], parent2[param.name]]
                )

        return child

    def _mutate(
        self, individual: Dict[str, Any], parameter_space: List[OptimizationParameter]
    ) -> Dict[str, Any]:
        """Polynomial mutation."""
        for param in parameter_space:
            if np.random.random() < 0.1:  # Mutation probability
                if param.parameter_type == "continuous":
                    eta = 20.0  # Distribution index
                    u = np.random.random()

                    if u <= 0.5:
                        delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))

                    val = individual[param.name]
                    val_range = param.bounds[1] - param.bounds[0]
                    new_val = val + delta * val_range * 0.1
                    individual[param.name] = np.clip(
                        new_val, param.bounds[0], param.bounds[1]
                    )
                elif param.parameter_type == "discrete" and param.values:
                    individual[param.name] = np.random.choice(param.values)
                elif param.parameter_type == "categorical" and param.values:
                    individual[param.name] = np.random.choice(param.values)

        return individual


class ProcessParameterOptimizer:
    """
    Main process parameter optimizer combining adaptive design and multi-objective optimization.
    """

    def __init__(
        self,
        kinetics_model: CrystallizationKineticsModel,
        optimization_parameters: List[OptimizationParameter],
        objectives: List[OptimizationObjective],
        constraints: OptimizationConstraints,
    ):
        """
        Initialize process parameter optimizer.

        Args:
            kinetics_model: Crystallization kinetics model
            optimization_parameters: Parameters to optimize
            objectives: Optimization objectives
            constraints: Process constraints
        """
        self.kinetics_model = kinetics_model
        self.optimization_parameters = optimization_parameters
        self.objectives = objectives
        self.constraints = constraints

        self.adaptive_engine = AdaptiveDesignEngine(
            optimization_parameters, objectives, constraints
        )
        self.multi_objective_optimizer = MultiObjectiveOptimizer(
            objectives, constraints
        )

    def optimize_process(
        self, n_experiments: int = 50, use_multi_objective: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize crystallization process parameters.

        Args:
            n_experiments: Number of experiments to run
            use_multi_objective: Whether to use multi-objective optimization

        Returns:
            Optimization results
        """
        if use_multi_objective:
            return self._run_multi_objective_optimization()
        else:
            return self._run_adaptive_optimization(n_experiments)

    def _run_adaptive_optimization(self, n_experiments: int) -> Dict[str, Any]:
        """Run adaptive Bayesian optimization."""
        self.adaptive_engine.initialize()

        for i in range(n_experiments):
            # Get next experiment suggestion
            suggested_params = self.adaptive_engine.suggest_next_experiment()

            # Evaluate experiment
            result = self._evaluate_experiment(suggested_params)

            # Add result to history
            self.adaptive_engine.add_experiment_result(result)

            logger.info(f"Completed experiment {i+1}/{n_experiments}")

        # Find best result
        best_result = max(
            self.adaptive_engine.experiment_history,
            key=lambda x: x.objectives.get(self.objectives[0].value, -np.inf),
        )

        return {
            "best_parameters": best_result.parameters,
            "best_objectives": best_result.objectives,
            "optimization_history": self.adaptive_engine.experiment_history,
            "n_experiments": n_experiments,
        }

    def _run_multi_objective_optimization(self) -> Dict[str, Any]:
        """Run multi-objective optimization."""

        def evaluation_function(params):
            result = self._evaluate_experiment(params)
            return result.objectives

        pareto_solutions = self.multi_objective_optimizer.optimize(
            evaluation_function, self.optimization_parameters
        )

        return {"pareto_front": pareto_solutions, "n_solutions": len(pareto_solutions)}

    def _evaluate_experiment(self, parameters: Dict[str, Any]) -> ExperimentResult:
        """Evaluate single experiment."""
        # Convert parameters to ProcessConditions
        conditions = self._parameters_to_conditions(parameters)

        # Run kinetics simulation
        try:
            outcomes = self.kinetics_model.predict_crystallization_outcome(conditions)

            # Map outcomes to objectives
            objectives = {}
            for obj in self.objectives:
                if obj == OptimizationObjective.MAXIMIZE_YIELD:
                    objectives[obj.value] = outcomes.get("final_yield", 0.0)
                elif obj == OptimizationObjective.MINIMIZE_TIME:
                    objectives[obj.value] = -outcomes.get(
                        "crystallization_time", 3600.0
                    )  # Negative for maximization
                elif obj == OptimizationObjective.MAXIMIZE_PURITY:
                    objectives[obj.value] = outcomes.get("quality_metrics", {}).get(
                        "uniformity", 0.0
                    )
                elif obj == OptimizationObjective.CONTROL_PSD:
                    objectives[obj.value] = -outcomes.get(
                        "size_distribution_width", 1e-6
                    )  # Minimize width
                else:
                    objectives[obj.value] = 0.0

            # Check constraints
            constraints_satisfied = self._check_constraints(parameters, outcomes)

            return ExperimentResult(
                parameters=parameters,
                objectives=objectives,
                constraints_satisfied=constraints_satisfied,
                success=True,
            )

        except Exception as e:
            logger.error(f"Experiment evaluation failed: {e}")
            return ExperimentResult(
                parameters=parameters,
                objectives={obj.value: -np.inf for obj in self.objectives},
                constraints_satisfied=False,
                success=False,
            )

    def _parameters_to_conditions(
        self, parameters: Dict[str, Any]
    ) -> ProcessConditions:
        """Convert optimization parameters to ProcessConditions."""
        conditions = ProcessConditions()

        # Map common parameters
        if "temperature" in parameters:
            temp = parameters["temperature"]
            conditions.temperature_profile = [
                (temp, 0.0),
                (temp, 3600.0),
            ]  # Constant temperature

        if "stirring_rate" in parameters:
            conditions.stirring_rate = parameters["stirring_rate"]

        if "supersaturation" in parameters:
            supersat = parameters["supersaturation"]
            conditions.supersaturation_profile = [(supersat, 0.0), (supersat, 3600.0)]

        # Handle solvent composition
        solvent_params = {
            k: v for k, v in parameters.items() if k.startswith("solvent_")
        }
        if solvent_params:
            conditions.solvent_composition = {
                k.replace("solvent_", ""): v for k, v in solvent_params.items()
            }

        return conditions

    def _check_constraints(
        self, parameters: Dict[str, Any], outcomes: Dict[str, Any]
    ) -> bool:
        """Check if constraints are satisfied."""
        # Temperature constraints
        if "temperature" in parameters:
            temp = parameters["temperature"]
            if not (
                self.constraints.temperature_bounds[0]
                <= temp
                <= self.constraints.temperature_bounds[1]
            ):
                return False

        # Quality requirements
        for requirement, min_value in self.constraints.quality_requirements.items():
            if requirement in outcomes and outcomes[requirement] < min_value:
                return False

        return True


class CrystallizationOptimizer(ProcessParameterOptimizer):
    """
    Main crystallization optimizer class.

    Provides high-level interface for crystallization process optimization
    combining all optimization strategies.
    """

    def __init__(
        self,
        kinetics_model: CrystallizationKineticsModel,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize crystallization optimizer.

        Args:
            kinetics_model: Crystallization kinetics model
            config: Configuration dictionary
        """
        # Define default optimization parameters
        default_parameters = [
            OptimizationParameter(
                "temperature", (273.15, 373.15), "continuous", units="K"
            ),
            OptimizationParameter(
                "stirring_rate", (50.0, 500.0), "continuous", units="rpm"
            ),
            OptimizationParameter("supersaturation", (1.1, 5.0), "continuous"),
            OptimizationParameter("solvent_water", (0.0, 1.0), "continuous"),
            OptimizationParameter("solvent_ethanol", (0.0, 1.0), "continuous"),
        ]

        # Define default objectives
        default_objectives = [
            OptimizationObjective.MAXIMIZE_YIELD,
            OptimizationObjective.MAXIMIZE_PURITY,
            OptimizationObjective.CONTROL_PSD,
        ]

        # Define default constraints
        default_constraints = OptimizationConstraints()

        # Override with config if provided
        parameters = (
            config.get("parameters", default_parameters)
            if config
            else default_parameters
        )
        objectives = (
            config.get("objectives", default_objectives)
            if config
            else default_objectives
        )
        constraints = (
            config.get("constraints", default_constraints)
            if config
            else default_constraints
        )

        super().__init__(kinetics_model, parameters, objectives, constraints)

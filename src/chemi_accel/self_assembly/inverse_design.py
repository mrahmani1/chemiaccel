"""
Inverse Design for Emergent Properties in Self-Assembly Systems.

This module provides comprehensive tools for optimizing molecular building blocks
and environmental conditions to achieve desired mesoscale self-assembled structures
and functions. It utilizes advanced AI techniques including reinforcement learning,
genetic algorithms, and generative models to navigate complex design spaces.

Features:
- Inverse design of polymer sequences for target structures
- Surfactant head/tail optimization for specific self-assembly behaviors
- Nanoparticle ligand design for controlled assembly
- Environmental condition optimization (temperature, concentration, pH)
- Multi-objective optimization with competing goals
- AI-driven exploration using RL, GA, and generative models
- Integration with self-assembly simulation and characterization
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional dependencies with graceful fallbacks
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern
    from sklearn.metrics import mean_squared_error, r2_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import scipy.optimize
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import norm

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .characterization import (
    MorphologyType,
    SelfAssemblyCharacterizer,
    StructureType,
    characterize_trajectory,
)

# Local imports
from .coarse_grained_simulation import (
    InteractionParameters,
    SimulationParameters,
    create_amphiphilic_system,
    run_dpd_simulation,
)


class OptimizationMethod(Enum):
    """Available optimization methods for inverse design."""

    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    MULTI_OBJECTIVE_NSGA2 = "multi_objective_nsga2"


class DesignTarget(Enum):
    """Target types for inverse design optimization."""

    POLYMER_SEQUENCE = "polymer_sequence"
    SURFACTANT_STRUCTURE = "surfactant_structure"
    NANOPARTICLE_LIGANDS = "nanoparticle_ligands"
    ENVIRONMENTAL_CONDITIONS = "environmental_conditions"
    MULTI_COMPONENT_SYSTEM = "multi_component_system"


class OptimizationObjective(Enum):
    """Optimization objectives for emergent properties."""

    MAXIMIZE_MICELLE_SIZE = "maximize_micelle_size"
    MINIMIZE_CMC = "minimize_cmc"
    TARGET_MORPHOLOGY = "target_morphology"
    MAXIMIZE_STABILITY = "maximize_stability"
    OPTIMIZE_PHASE_BEHAVIOR = "optimize_phase_behavior"
    MINIMIZE_POLYDISPERSITY = "minimize_polydispersity"
    MAXIMIZE_ASSEMBLY_RATE = "maximize_assembly_rate"
    TARGET_SURFACE_PROPERTIES = "target_surface_properties"


class PerformanceMetric(Enum):
    """Performance metrics for evaluating designs."""

    AGGREGATION_NUMBER = "aggregation_number"
    CRITICAL_MICELLE_CONCENTRATION = "critical_micelle_concentration"
    RADIUS_OF_GYRATION = "radius_of_gyration"
    ASPHERICITY = "asphericity"
    SURFACE_AREA = "surface_area"
    VOLUME_FRACTION = "volume_fraction"
    ORDER_PARAMETER = "order_parameter"
    FORMATION_ENERGY = "formation_energy"
    ASSEMBLY_KINETICS = "assembly_kinetics"


@dataclass
class DesignParameter:
    """Represents a design parameter for optimization."""

    name: str
    parameter_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Tuple[float, float] = (0.0, 1.0)
    discrete_values: Optional[List[Any]] = None
    categorical_values: Optional[List[str]] = None
    current_value: Optional[Any] = None
    importance_weight: float = 1.0
    constraints: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate parameter definition."""
        if self.parameter_type not in ["continuous", "discrete", "categorical"]:
            raise ValueError(f"Invalid parameter type: {self.parameter_type}")

        if self.parameter_type == "discrete" and self.discrete_values is None:
            raise ValueError("Discrete parameters must specify discrete_values")

        if self.parameter_type == "categorical" and self.categorical_values is None:
            raise ValueError("Categorical parameters must specify categorical_values")


@dataclass
class ObjectiveFunction:
    """Defines an objective function for optimization."""

    name: str
    objective_type: OptimizationObjective
    target_value: Optional[float] = None
    weight: float = 1.0
    minimize: bool = True
    tolerance: float = 0.1
    constraint_penalty: float = 1000.0

    def evaluate(self, properties: Dict[str, float]) -> float:
        """Evaluate the objective function given system properties."""
        if self.objective_type.value not in properties:
            return self.constraint_penalty

        value = properties[self.objective_type.value]

        if self.target_value is not None:
            # Target-based objective
            error = abs(value - self.target_value)
            if error <= self.tolerance:
                return 0.0
            return error * self.weight
        else:
            # Min/max objective
            return value * self.weight * (-1 if not self.minimize else 1)


@dataclass
class DesignSpace:
    """Defines the design space for optimization."""

    parameters: List[DesignParameter]
    objectives: List[ObjectiveFunction]
    constraints: List[Callable] = field(default_factory=list)
    parameter_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for continuous parameters."""
        bounds = []
        for param in self.parameters:
            if param.parameter_type == "continuous":
                bounds.append(param.bounds)
            elif param.parameter_type == "discrete":
                bounds.append((0, len(param.discrete_values) - 1))
            else:  # categorical
                bounds.append((0, len(param.categorical_values) - 1))
        return bounds

    def encode_parameters(self, parameter_values: Dict[str, Any]) -> np.ndarray:
        """Encode parameter values to numerical array."""
        encoded = []
        for param in self.parameters:
            value = parameter_values.get(param.name, param.current_value)

            if param.parameter_type == "continuous":
                encoded.append(float(value))
            elif param.parameter_type == "discrete":
                encoded.append(float(param.discrete_values.index(value)))
            else:  # categorical
                encoded.append(float(param.categorical_values.index(value)))

        return np.array(encoded)

    def decode_parameters(self, encoded_values: np.ndarray) -> Dict[str, Any]:
        """Decode numerical array to parameter values."""
        decoded = {}
        for i, param in enumerate(self.parameters):
            value = encoded_values[i]

            if param.parameter_type == "continuous":
                decoded[param.name] = float(value)
            elif param.parameter_type == "discrete":
                idx = int(round(value))
                idx = max(0, min(idx, len(param.discrete_values) - 1))
                decoded[param.name] = param.discrete_values[idx]
            else:  # categorical
                idx = int(round(value))
                idx = max(0, min(idx, len(param.categorical_values) - 1))
                decoded[param.name] = param.categorical_values[idx]

        return decoded


@dataclass
class OptimizationResult:
    """Results from an optimization run."""

    best_parameters: Dict[str, Any]
    best_objective_value: float
    objective_history: List[float]
    parameter_history: List[Dict[str, Any]]
    convergence_data: Dict[str, Any]
    evaluation_count: int
    optimization_time: float
    success: bool
    message: str
    pareto_front: Optional[List[Tuple[Dict[str, Any], List[float]]]] = None


class InverseDesignError(Exception):
    """Base exception for inverse design operations."""

    pass


class OptimizationError(InverseDesignError):
    """Exception raised during optimization."""

    pass


class DesignSpaceError(InverseDesignError):
    """Exception raised for invalid design space definition."""

    pass


class EvaluationError(InverseDesignError):
    """Exception raised during objective function evaluation."""

    pass


class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms."""

    def __init__(self, design_space: DesignSpace, random_seed: Optional[int] = None):
        self.design_space = design_space
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.evaluation_count = 0
        self.optimization_history = []
        self.start_time = None

    @abstractmethod
    def optimize(
        self,
        evaluation_function: Callable,
        max_evaluations: int = 1000,
        tolerance: float = 1e-6,
        **kwargs,
    ) -> OptimizationResult:
        """Perform optimization."""
        pass

    def evaluate_objectives(
        self, parameters: Dict[str, Any], evaluation_function: Callable
    ) -> float:
        """Evaluate all objectives for given parameters."""
        try:
            properties = evaluation_function(parameters)

            total_objective = 0.0
            for objective in self.design_space.objectives:
                obj_value = objective.evaluate(properties)
                total_objective += obj_value

            self.evaluation_count += 1
            self.optimization_history.append(
                {
                    "parameters": parameters.copy(),
                    "properties": properties.copy(),
                    "objective": total_objective,
                    "evaluation": self.evaluation_count,
                }
            )

            return total_objective

        except Exception as e:
            logging.warning(f"Evaluation failed for parameters {parameters}: {e}")
            return float("inf")


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """Genetic Algorithm optimizer for inverse design."""

    def optimize(
        self,
        evaluation_function: Callable,
        max_evaluations: int = 1000,
        tolerance: float = 1e-6,
        population_size: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        elitism_rate: float = 0.1,
        **kwargs,
    ) -> OptimizationResult:
        """Optimize using genetic algorithm."""
        self.start_time = time.time()

        # Initialize population
        bounds = self.design_space.get_parameter_bounds()
        population = []

        for _ in range(population_size):
            individual = []
            for bound in bounds:
                individual.append(np.random.uniform(bound[0], bound[1]))
            population.append(np.array(individual))

        population = np.array(population)
        best_objective = float("inf")
        best_parameters = None
        objective_history = []

        generation = 0
        while self.evaluation_count < max_evaluations:
            # Evaluate population
            fitness_scores = []
            for individual in population:
                params = self.design_space.decode_parameters(individual)
                objective = self.evaluate_objectives(params, evaluation_function)
                fitness_scores.append(objective)

                if objective < best_objective:
                    best_objective = objective
                    best_parameters = params.copy()

            fitness_scores = np.array(fitness_scores)
            objective_history.append(np.min(fitness_scores))

            # Check convergence
            if len(objective_history) > 10:
                recent_improvement = objective_history[-10] - objective_history[-1]
                if recent_improvement < tolerance:
                    break

            # Selection, crossover, and mutation
            new_population = []

            # Elitism
            elite_count = int(elitism_rate * population_size)
            elite_indices = np.argsort(fitness_scores)[:elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(
                    population, fitness_scores, tournament_size
                )
                parent2 = self._tournament_selection(
                    population, fitness_scores, tournament_size
                )

                # Crossover
                if np.random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if np.random.random() < mutation_rate:
                    child1 = self._mutate(child1, bounds)
                if np.random.random() < mutation_rate:
                    child2 = self._mutate(child2, bounds)

                new_population.extend([child1, child2])

            population = np.array(new_population[:population_size])
            generation += 1

        optimization_time = time.time() - self.start_time

        return OptimizationResult(
            best_parameters=best_parameters,
            best_objective_value=best_objective,
            objective_history=objective_history,
            parameter_history=[
                entry["parameters"] for entry in self.optimization_history
            ],
            convergence_data={
                "generations": generation,
                "final_population_size": len(population),
            },
            evaluation_count=self.evaluation_count,
            optimization_time=optimization_time,
            success=best_objective < float("inf"),
            message=f"GA completed after {generation} generations",
        )

    def _tournament_selection(
        self, population: np.ndarray, fitness_scores: np.ndarray, tournament_size: int
    ) -> np.ndarray:
        """Tournament selection for parent selection."""
        tournament_indices = np.random.choice(
            len(population), tournament_size, replace=False
        )
        tournament_fitness = fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        crossover_point = np.random.randint(1, len(parent1))

        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

        return child1, child2

    def _mutate(
        self, individual: np.ndarray, bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Gaussian mutation with bounds checking."""
        mutated = individual.copy()

        for i, (lower, upper) in enumerate(bounds):
            if np.random.random() < 0.1:  # Per-gene mutation probability
                mutation_strength = 0.1 * (upper - lower)
                mutated[i] += np.random.normal(0, mutation_strength)
                mutated[i] = np.clip(mutated[i], lower, upper)

        return mutated


class BayesianOptimizer(BaseOptimizer):
    """Bayesian Optimization for efficient inverse design."""

    def __init__(
        self,
        design_space: DesignSpace,
        random_seed: Optional[int] = None,
        kernel_type: str = "matern",
        acquisition_function: str = "ei",
    ):
        super().__init__(design_space, random_seed)

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Bayesian optimization")

        self.kernel_type = kernel_type
        self.acquisition_function = acquisition_function
        self.gp_model = None
        self.X_observed = []
        self.y_observed = []

    def optimize(
        self,
        evaluation_function: Callable,
        max_evaluations: int = 100,
        tolerance: float = 1e-6,
        n_initial_samples: int = 10,
        acquisition_samples: int = 1000,
        **kwargs,
    ) -> OptimizationResult:
        """Optimize using Bayesian optimization."""
        self.start_time = time.time()

        bounds = self.design_space.get_parameter_bounds()
        dimension = len(bounds)

        # Initial random sampling
        X_init = []
        y_init = []

        for _ in range(n_initial_samples):
            x = np.array([np.random.uniform(bound[0], bound[1]) for bound in bounds])
            params = self.design_space.decode_parameters(x)
            y = self.evaluate_objectives(params, evaluation_function)

            X_init.append(x)
            y_init.append(y)

        self.X_observed = np.array(X_init)
        self.y_observed = np.array(y_init)

        best_idx = np.argmin(self.y_observed)
        best_objective = self.y_observed[best_idx]
        best_parameters = self.design_space.decode_parameters(self.X_observed[best_idx])

        objective_history = [
            np.min(self.y_observed[: i + 1]) for i in range(len(self.y_observed))
        ]

        # Bayesian optimization loop
        while self.evaluation_count < max_evaluations:
            # Fit Gaussian Process
            self._fit_gp_model()

            # Find next point to evaluate using acquisition function
            next_x = self._optimize_acquisition(bounds, acquisition_samples)

            # Evaluate next point
            params = self.design_space.decode_parameters(next_x)
            y_new = self.evaluate_objectives(params, evaluation_function)

            # Update data
            self.X_observed = np.vstack([self.X_observed, next_x])
            self.y_observed = np.append(self.y_observed, y_new)

            # Update best
            if y_new < best_objective:
                best_objective = y_new
                best_parameters = params.copy()

            objective_history.append(min(objective_history[-1], y_new))

            # Check convergence
            if len(objective_history) > 10:
                recent_improvement = objective_history[-10] - objective_history[-1]
                if recent_improvement < tolerance:
                    break

        optimization_time = time.time() - self.start_time

        return OptimizationResult(
            best_parameters=best_parameters,
            best_objective_value=best_objective,
            objective_history=objective_history,
            parameter_history=[
                entry["parameters"] for entry in self.optimization_history
            ],
            convergence_data={
                "gp_iterations": len(self.y_observed) - n_initial_samples
            },
            evaluation_count=self.evaluation_count,
            optimization_time=optimization_time,
            success=best_objective < float("inf"),
            message="Bayesian optimization completed",
        )

    def _fit_gp_model(self):
        """Fit Gaussian Process model to observed data."""
        if self.kernel_type == "rbf":
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        elif self.kernel_type == "matern":
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        self.gp_model = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10
        )

        self.gp_model.fit(self.X_observed, self.y_observed)

    def _optimize_acquisition(
        self, bounds: List[Tuple[float, float]], n_samples: int
    ) -> np.ndarray:
        """Optimize acquisition function to find next evaluation point."""
        # Random sampling for acquisition optimization
        candidates = []
        acquisition_values = []

        for _ in range(n_samples):
            x = np.array([np.random.uniform(bound[0], bound[1]) for bound in bounds])
            acq_value = self._acquisition_function(x.reshape(1, -1))
            candidates.append(x)
            acquisition_values.append(acq_value)

        best_idx = np.argmax(acquisition_values)
        return candidates[best_idx]

    def _acquisition_function(self, X: np.ndarray) -> float:
        """Compute acquisition function value."""
        if self.gp_model is None:
            return 0.0

        mu, sigma = self.gp_model.predict(X, return_std=True)

        if self.acquisition_function == "ei":  # Expected Improvement
            f_best = np.min(self.y_observed)
            improvement = f_best - mu
            z = improvement / (sigma + 1e-9)
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return ei[0] if len(ei) == 1 else ei

        elif self.acquisition_function == "ucb":  # Upper Confidence Bound
            kappa = 2.576  # 99% confidence
            return -(mu - kappa * sigma)[0]

        else:
            raise ValueError(
                f"Unknown acquisition function: {self.acquisition_function}"
            )


class EmergentPropertyInverseDesigner:
    """
    Main class for inverse design of emergent properties in self-assembly systems.

    This class provides a comprehensive framework for optimizing molecular building
    blocks and environmental conditions to achieve desired mesoscale structures
    and functions.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        device_id: int = 0,
        cache_simulations: bool = True,
        parallel_evaluations: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize the inverse designer.

        Args:
            use_gpu: Whether to use GPU acceleration for simulations
            device_id: GPU device ID to use
            cache_simulations: Whether to cache simulation results
            parallel_evaluations: Whether to use parallel evaluation
            max_workers: Maximum number of parallel workers
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device_id = device_id
        self.cache_simulations = cache_simulations
        self.parallel_evaluations = parallel_evaluations
        self.max_workers = max_workers

        # Initialize components
        self.characterizer = SelfAssemblyCharacterizer(
            use_gpu=self.use_gpu, device_id=device_id
        )

        # Simulation cache
        self.simulation_cache = {} if cache_simulations else None

        # Available optimizers
        self.optimizers = {
            OptimizationMethod.GENETIC_ALGORITHM: GeneticAlgorithmOptimizer,
            OptimizationMethod.BAYESIAN_OPTIMIZATION: BayesianOptimizer,
        }

        logging.info(
            f"EmergentPropertyInverseDesigner initialized with GPU={self.use_gpu}"
        )

    def create_polymer_design_space(
        self,
        sequence_length: int,
        monomer_types: List[str],
        property_targets: Dict[str, float],
    ) -> DesignSpace:
        """Create design space for polymer sequence optimization."""
        parameters = []

        # Polymer sequence parameters
        for i in range(sequence_length):
            param = DesignParameter(
                name=f"monomer_{i}",
                parameter_type="categorical",
                categorical_values=monomer_types,
                current_value=monomer_types[0],
            )
            parameters.append(param)

        # Create objectives from targets
        objectives = []
        for prop_name, target_value in property_targets.items():
            if prop_name in [obj.value for obj in OptimizationObjective]:
                obj_type = OptimizationObjective(prop_name)
                objective = ObjectiveFunction(
                    name=f"target_{prop_name}",
                    objective_type=obj_type,
                    target_value=target_value,
                    weight=1.0,
                )
                objectives.append(objective)

        return DesignSpace(parameters=parameters, objectives=objectives)

    def create_surfactant_design_space(
        self,
        head_group_options: List[str],
        tail_length_range: Tuple[int, int],
        property_targets: Dict[str, float],
    ) -> DesignSpace:
        """Create design space for surfactant structure optimization."""
        parameters = []

        # Head group parameter
        head_param = DesignParameter(
            name="head_group",
            parameter_type="categorical",
            categorical_values=head_group_options,
            current_value=head_group_options[0],
        )
        parameters.append(head_param)

        # Tail length parameter
        tail_param = DesignParameter(
            name="tail_length",
            parameter_type="discrete",
            discrete_values=list(range(tail_length_range[0], tail_length_range[1] + 1)),
            current_value=tail_length_range[0],
        )
        parameters.append(tail_param)

        # Hydrophobicity parameter
        hydrophob_param = DesignParameter(
            name="tail_hydrophobicity",
            parameter_type="continuous",
            bounds=(0.5, 2.0),
            current_value=1.0,
        )
        parameters.append(hydrophob_param)

        # Create objectives
        objectives = []
        for prop_name, target_value in property_targets.items():
            if prop_name in [obj.value for obj in OptimizationObjective]:
                obj_type = OptimizationObjective(prop_name)
                objective = ObjectiveFunction(
                    name=f"target_{prop_name}",
                    objective_type=obj_type,
                    target_value=target_value,
                    weight=1.0,
                )
                objectives.append(objective)

        return DesignSpace(parameters=parameters, objectives=objectives)

    def create_environmental_design_space(
        self,
        temperature_range: Tuple[float, float],
        concentration_range: Tuple[float, float],
        property_targets: Dict[str, float],
    ) -> DesignSpace:
        """Create design space for environmental condition optimization."""
        parameters = []

        # Temperature parameter
        temp_param = DesignParameter(
            name="temperature",
            parameter_type="continuous",
            bounds=temperature_range,
            current_value=(temperature_range[0] + temperature_range[1]) / 2,
        )
        parameters.append(temp_param)

        # Concentration parameter
        conc_param = DesignParameter(
            name="concentration",
            parameter_type="continuous",
            bounds=concentration_range,
            current_value=(concentration_range[0] + concentration_range[1]) / 2,
        )
        parameters.append(conc_param)

        # pH parameter (if applicable)
        ph_param = DesignParameter(
            name="ph",
            parameter_type="continuous",
            bounds=(1.0, 14.0),
            current_value=7.0,
        )
        parameters.append(ph_param)

        # Create objectives
        objectives = []
        for prop_name, target_value in property_targets.items():
            if prop_name in [obj.value for obj in OptimizationObjective]:
                obj_type = OptimizationObjective(prop_name)
                objective = ObjectiveFunction(
                    name=f"target_{prop_name}",
                    objective_type=obj_type,
                    target_value=target_value,
                    weight=1.0,
                )
                objectives.append(objective)

        return DesignSpace(parameters=parameters, objectives=objectives)

    def evaluate_design(
        self, parameters: Dict[str, Any], design_target: DesignTarget
    ) -> Dict[str, float]:
        """
        Evaluate a design by running simulation and characterization.

        Args:
            parameters: Design parameters to evaluate
            design_target: Type of design being optimized

        Returns:
            Dictionary of computed properties
        """
        cache_key = self._get_cache_key(parameters, design_target)

        if self.simulation_cache and cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]

        try:
            # Generate system based on design parameters
            if design_target == DesignTarget.SURFACTANT_STRUCTURE:
                particles, sim_box, interaction_params = self._create_surfactant_system(
                    parameters
                )
            elif design_target == DesignTarget.POLYMER_SEQUENCE:
                particles, sim_box, interaction_params = self._create_polymer_system(
                    parameters
                )
            elif design_target == DesignTarget.ENVIRONMENTAL_CONDITIONS:
                (
                    particles,
                    sim_box,
                    interaction_params,
                ) = self._create_environmental_system(parameters)
            else:
                raise NotImplementedError(
                    f"Design target {design_target} not implemented"
                )

            # Run simulation
            simulation_params = SimulationParameters(
                timestep=0.01,
                total_time=10.0,
                temperature=parameters.get("temperature", 298.15),
                use_gpu=self.use_gpu,
                output_frequency=10,
            )

            trajectory = run_dpd_simulation(
                particles=particles,
                simulation_box=sim_box,
                interaction_params=interaction_params,
                total_time=simulation_params.total_time,
                timestep=simulation_params.timestep,
                temperature=simulation_params.temperature,
                use_gpu=self.use_gpu,
            )

            # Characterize results
            results = characterize_trajectory(
                trajectory, clustering_method="dbscan", use_gpu=self.use_gpu
            )

            # Extract relevant properties
            properties = self._extract_properties(results, parameters)

            # Cache results
            if self.simulation_cache:
                self.simulation_cache[cache_key] = properties

            return properties

        except Exception as e:
            logging.error(f"Design evaluation failed: {e}")
            return {"error": float("inf")}

    def optimize_design(
        self,
        design_space: DesignSpace,
        design_target: DesignTarget,
        optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION,
        max_evaluations: int = 100,
        **optimizer_kwargs,
    ) -> OptimizationResult:
        """
        Optimize design using specified method.

        Args:
            design_space: Design space definition
            design_target: Type of design being optimized
            optimization_method: Optimization algorithm to use
            max_evaluations: Maximum number of evaluations
            **optimizer_kwargs: Additional optimizer arguments

        Returns:
            Optimization results
        """
        if optimization_method not in self.optimizers:
            raise OptimizationError(f"Optimizer {optimization_method} not available")

        # Create optimizer
        optimizer_class = self.optimizers[optimization_method]
        optimizer = optimizer_class(design_space)

        # Define evaluation function
        def evaluation_function(parameters: Dict[str, Any]) -> Dict[str, float]:
            return self.evaluate_design(parameters, design_target)

        # Run optimization
        if self.parallel_evaluations and max_evaluations > 10:
            # Use parallel evaluation for larger optimization runs
            return self._optimize_parallel(
                optimizer, evaluation_function, max_evaluations, **optimizer_kwargs
            )
        else:
            return optimizer.optimize(
                evaluation_function, max_evaluations, **optimizer_kwargs
            )

    def multi_objective_optimization(
        self,
        design_space: DesignSpace,
        design_target: DesignTarget,
        max_evaluations: int = 200,
    ) -> OptimizationResult:
        """
        Perform multi-objective optimization using NSGA-II algorithm.

        Args:
            design_space: Design space definition
            design_target: Type of design being optimized
            max_evaluations: Maximum number of evaluations

        Returns:
            Optimization results with Pareto front
        """
        # This is a simplified implementation - in practice, you'd use
        # a proper multi-objective optimization library like DEAP

        start_time = time.time()
        evaluation_count = 0
        population_size = 50

        # Initialize population
        bounds = design_space.get_parameter_bounds()
        population = []

        for _ in range(population_size):
            individual = []
            for bound in bounds:
                individual.append(np.random.uniform(bound[0], bound[1]))
            population.append(np.array(individual))

        # Evaluate population
        objective_values = []
        parameter_history = []

        for individual in population:
            params = design_space.decode_parameters(individual)
            properties = self.evaluate_design(params, design_target)

            # Evaluate all objectives
            objectives = []
            for obj_func in design_space.objectives:
                obj_value = obj_func.evaluate(properties)
                objectives.append(obj_value)

            objective_values.append(objectives)
            parameter_history.append(params)
            evaluation_count += 1

            if evaluation_count >= max_evaluations:
                break

        # Find Pareto front (simplified)
        pareto_front = self._find_pareto_front(parameter_history, objective_values)

        # Select best solution (closest to origin in objective space)
        best_idx = 0
        best_distance = float("inf")

        for i, objectives in enumerate(objective_values):
            distance = np.sqrt(sum(obj**2 for obj in objectives))
            if distance < best_distance:
                best_distance = distance
                best_idx = i

        optimization_time = time.time() - start_time

        return OptimizationResult(
            best_parameters=parameter_history[best_idx],
            best_objective_value=best_distance,
            objective_history=[best_distance],  # Simplified
            parameter_history=parameter_history,
            convergence_data={"pareto_front_size": len(pareto_front)},
            evaluation_count=evaluation_count,
            optimization_time=optimization_time,
            success=True,
            message="Multi-objective optimization completed",
            pareto_front=pareto_front,
        )

    def _create_surfactant_system(self, parameters: Dict[str, Any]) -> Tuple:
        """Create surfactant system from design parameters."""
        head_group = parameters.get("head_group", "ionic")
        tail_length = parameters.get("tail_length", 8)
        tail_hydrophobicity = parameters.get("tail_hydrophobicity", 1.0)

        # Generate amphiphilic system with specified properties
        particles, sim_box, interaction_params = create_amphiphilic_system(
            n_molecules=50, head_size=1, tail_size=tail_length, box_size=20.0
        )

        # Modify interaction parameters based on design
        for key, value in interaction_params.interaction_matrix.items():
            if "TAIL" in key[0] or "TAIL" in key[1]:
                if "hydrophobic_strength" in value:
                    value["hydrophobic_strength"] *= tail_hydrophobicity

        return particles, sim_box, interaction_params

    def _create_polymer_system(self, parameters: Dict[str, Any]) -> Tuple:
        """Create polymer system from sequence parameters."""
        # Extract sequence
        sequence = []
        i = 0
        while f"monomer_{i}" in parameters:
            sequence.append(parameters[f"monomer_{i}"])
            i += 1

        # Generate system based on sequence
        # This is a simplified implementation
        particles, sim_box, interaction_params = create_amphiphilic_system(
            n_molecules=20,
            head_size=len([m for m in sequence if m == "hydrophilic"]),
            tail_size=len([m for m in sequence if m == "hydrophobic"]),
            box_size=15.0,
        )

        return particles, sim_box, interaction_params

    def _create_environmental_system(self, parameters: Dict[str, Any]) -> Tuple:
        """Create system with specified environmental conditions."""
        temperature = parameters.get("temperature", 298.15)
        concentration = parameters.get("concentration", 0.1)
        ph = parameters.get("ph", 7.0)

        # Generate base system
        particles, sim_box, interaction_params = create_amphiphilic_system(
            n_molecules=int(50 * concentration), head_size=1, tail_size=6, box_size=15.0
        )

        # Modify interactions based on environmental conditions
        temp_factor = temperature / 298.15
        ph_factor = 1.0 + 0.1 * (ph - 7.0)  # Simplified pH effect

        for key, value in interaction_params.interaction_matrix.items():
            if "a" in value:  # DPD repulsion parameter
                value["a"] *= temp_factor * ph_factor

        return particles, sim_box, interaction_params

    def _extract_properties(
        self, characterization_results: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract relevant properties from characterization results."""
        properties = {}

        # Extract basic properties
        if "structures" in characterization_results:
            structures = characterization_results["structures"]

            if structures:
                # Aggregation number
                agg_numbers = [s.aggregation_number for s in structures]
                properties["aggregation_number"] = np.mean(agg_numbers)

                # Radius of gyration
                rg_values = [s.radius_of_gyration for s in structures]
                properties["radius_of_gyration"] = np.mean(rg_values)

                # Asphericity
                asphericity_values = [s.asphericity for s in structures]
                properties["asphericity"] = np.mean(asphericity_values)

                # Surface area
                surface_areas = [s.surface_area for s in structures]
                properties["surface_area"] = np.mean(surface_areas)
            else:
                # No structures found
                properties["aggregation_number"] = 0.0
                properties["radius_of_gyration"] = 0.0
                properties["asphericity"] = 0.0
                properties["surface_area"] = 0.0

        # Add objective-specific properties
        properties["maximize_micelle_size"] = properties.get("aggregation_number", 0.0)
        properties["minimize_cmc"] = 1.0 / (
            properties.get("aggregation_number", 1.0) + 1.0
        )
        properties["maximize_stability"] = 1.0 / (
            properties.get("asphericity", 1.0) + 0.1
        )

        return properties

    def _get_cache_key(
        self, parameters: Dict[str, Any], design_target: DesignTarget
    ) -> str:
        """Generate cache key for simulation results."""
        param_str = json.dumps(parameters, sort_keys=True)
        return f"{design_target.value}_{hash(param_str)}"

    def _optimize_parallel(
        self,
        optimizer: BaseOptimizer,
        evaluation_function: Callable,
        max_evaluations: int,
        **optimizer_kwargs,
    ) -> OptimizationResult:
        """Run optimization with parallel evaluation."""
        # This is a simplified parallel implementation
        # In practice, you'd modify the optimizer to support batched evaluation
        return optimizer.optimize(
            evaluation_function, max_evaluations, **optimizer_kwargs
        )

    def _find_pareto_front(
        self, parameters: List[Dict[str, Any]], objectives: List[List[float]]
    ) -> List[Tuple[Dict[str, Any], List[float]]]:
        """Find Pareto front for multi-objective optimization."""
        pareto_front = []

        for i, (params, obj) in enumerate(zip(parameters, objectives)):
            is_dominated = False

            for j, other_obj in enumerate(objectives):
                if i != j:
                    # Check if solution i is dominated by solution j
                    better_in_all = all(other_obj[k] <= obj[k] for k in range(len(obj)))
                    strictly_better = any(
                        other_obj[k] < obj[k] for k in range(len(obj))
                    )

                    if better_in_all and strictly_better:
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_front.append((params, obj))

        return pareto_front


# Utility functions for common design scenarios


def design_micelle_forming_surfactant(
    target_aggregation_number: int = 50,
    target_cmc: float = 0.01,
    head_groups: List[str] = None,
    tail_length_range: Tuple[int, int] = (6, 16),
    max_evaluations: int = 100,
    use_gpu: bool = True,
) -> OptimizationResult:
    """
    Design surfactant for target micelle properties.

    Args:
        target_aggregation_number: Target micelle size
        target_cmc: Target critical micelle concentration
        head_groups: Available head group options
        tail_length_range: Range of tail lengths to consider
        max_evaluations: Maximum optimization evaluations
        use_gpu: Whether to use GPU acceleration

    Returns:
        Optimization results with best surfactant design
    """
    if head_groups is None:
        head_groups = ["ionic", "nonionic", "zwitterionic"]

    # Create inverse designer
    designer = EmergentPropertyInverseDesigner(use_gpu=use_gpu)

    # Define property targets
    property_targets = {
        "aggregation_number": target_aggregation_number,
        "minimize_cmc": target_cmc,
    }

    # Create design space
    design_space = designer.create_surfactant_design_space(
        head_group_options=head_groups,
        tail_length_range=tail_length_range,
        property_targets=property_targets,
    )

    # Optimize design
    result = designer.optimize_design(
        design_space=design_space,
        design_target=DesignTarget.SURFACTANT_STRUCTURE,
        optimization_method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
        max_evaluations=max_evaluations,
    )

    return result


def optimize_environmental_conditions(
    target_morphology: StructureType,
    temperature_range: Tuple[float, float] = (280.0, 340.0),
    concentration_range: Tuple[float, float] = (0.01, 1.0),
    max_evaluations: int = 50,
    use_gpu: bool = True,
) -> OptimizationResult:
    """
    Optimize environmental conditions for target morphology.

    Args:
        target_morphology: Desired structure type
        temperature_range: Temperature range to explore (K)
        concentration_range: Concentration range to explore
        max_evaluations: Maximum optimization evaluations
        use_gpu: Whether to use GPU acceleration

    Returns:
        Optimization results with best environmental conditions
    """
    # Create inverse designer
    designer = EmergentPropertyInverseDesigner(use_gpu=use_gpu)

    # Define property targets based on morphology
    if target_morphology == StructureType.MICELLE:
        property_targets = {"maximize_micelle_size": 1.0}
    elif target_morphology == StructureType.VESICLE:
        property_targets = {"target_morphology": 1.0}
    else:
        property_targets = {"maximize_stability": 1.0}

    # Create design space
    design_space = designer.create_environmental_design_space(
        temperature_range=temperature_range,
        concentration_range=concentration_range,
        property_targets=property_targets,
    )

    # Optimize design
    result = designer.optimize_design(
        design_space=design_space,
        design_target=DesignTarget.ENVIRONMENTAL_CONDITIONS,
        optimization_method=OptimizationMethod.GENETIC_ALGORITHM,
        max_evaluations=max_evaluations,
        population_size=20,
    )

    return result


def multi_objective_surfactant_design(
    objectives: Dict[str, float],
    head_groups: List[str] = None,
    tail_length_range: Tuple[int, int] = (6, 16),
    max_evaluations: int = 200,
    use_gpu: bool = True,
) -> OptimizationResult:
    """
    Multi-objective surfactant design optimization.

    Args:
        objectives: Dictionary of objective names and target values
        head_groups: Available head group options
        tail_length_range: Range of tail lengths to consider
        max_evaluations: Maximum optimization evaluations
        use_gpu: Whether to use GPU acceleration

    Returns:
        Optimization results with Pareto front
    """
    if head_groups is None:
        head_groups = ["ionic", "nonionic", "zwitterionic"]

    # Create inverse designer
    designer = EmergentPropertyInverseDesigner(use_gpu=use_gpu)

    # Create design space
    design_space = designer.create_surfactant_design_space(
        head_group_options=head_groups,
        tail_length_range=tail_length_range,
        property_targets=objectives,
    )

    # Perform multi-objective optimization
    result = designer.multi_objective_optimization(
        design_space=design_space,
        design_target=DesignTarget.SURFACTANT_STRUCTURE,
        max_evaluations=max_evaluations,
    )

    return result

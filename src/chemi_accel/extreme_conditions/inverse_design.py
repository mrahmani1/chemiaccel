"""
Inverse Design for Dynamic/Non-Equilibrium Properties

This module extends inverse design capabilities to optimize molecular compositions,
processing pathways, or microstructural features for desired material properties
that emerge during or after exposure to extreme conditions or non-equilibrium states.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    import optax
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    optax = None
    JAX_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern

    SCIKIT_AVAILABLE = True
except ImportError:
    SCIKIT_AVAILABLE = False

from ..ai.bayesian_optimization import BayesianOptimizer
from ..ai.multi_objective_optimization import MultiObjectiveOptimizer
from ..core.exceptions import ChemiAccelError, OptimizationError
from ..utils.logging import get_logger
from .simulation_engine import (
    ExtremeConditionType,
    NonEquilibriumSimulationEngine,
    SimulationBackend,
    SimulationParameters,
)

logger = get_logger(__name__)


class OptimizationTarget(Enum):
    """Types of optimization targets for extreme conditions."""

    DAMAGE_RESISTANCE = "damage_resistance"
    SHOCK_ABSORPTION = "shock_absorption"
    RADIATION_TOLERANCE = "radiation_tolerance"
    THERMAL_STABILITY = "thermal_stability"
    FRACTURE_TOUGHNESS = "fracture_toughness"
    FATIGUE_RESISTANCE = "fatigue_resistance"
    CREEP_RESISTANCE = "creep_resistance"
    CORROSION_RESISTANCE = "corrosion_resistance"
    RECOVERY_CAPABILITY = "recovery_capability"
    SELF_HEALING = "self_healing"


class OptimizationVariable(Enum):
    """Types of variables that can be optimized."""

    COMPOSITION = "composition"
    MICROSTRUCTURE = "microstructure"
    PROCESSING_PATHWAY = "processing_pathway"
    DEFECT_DENSITY = "defect_density"
    GRAIN_SIZE = "grain_size"
    PHASE_FRACTION = "phase_fraction"
    SURFACE_TREATMENT = "surface_treatment"
    THERMAL_HISTORY = "thermal_history"


@dataclass
class OptimizationConstraints:
    """Constraints for inverse design optimization."""

    # Composition constraints
    min_composition: Dict[str, float] = field(default_factory=dict)
    max_composition: Dict[str, float] = field(default_factory=dict)

    # Processing constraints
    max_temperature: float = 2000.0  # Kelvin
    min_temperature: float = 200.0  # Kelvin
    max_pressure: float = 1e9  # Pa
    min_pressure: float = 1e3  # Pa

    # Economic constraints
    max_cost_per_kg: float = 1000.0  # USD/kg
    min_yield: float = 0.8  # Fraction

    # Performance constraints
    min_strength: float = 0.0  # MPa
    max_density: float = 20000.0  # kg/m³
    min_ductility: float = 0.0  # %

    # Environmental constraints
    max_toxicity_score: float = 5.0
    min_recyclability: float = 0.5
    max_carbon_footprint: float = 100.0  # kg CO2 eq/kg


@dataclass
class OptimizationResult:
    """Results from inverse design optimization."""

    # Optimal design
    optimal_composition: Dict[str, float] = field(default_factory=dict)
    optimal_processing: Dict[str, Any] = field(default_factory=dict)
    optimal_microstructure: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    predicted_properties: Dict[str, float] = field(default_factory=dict)
    objective_values: Dict[str, float] = field(default_factory=dict)

    # Optimization metadata
    n_evaluations: int = 0
    optimization_time: float = 0.0
    convergence_achieved: bool = False

    # Validation results
    simulation_validation: Dict[str, Any] = field(default_factory=dict)
    uncertainty_estimates: Dict[str, float] = field(default_factory=dict)


class BaseInverseDesigner(ABC):
    """Base class for all inverse design methods."""

    def __init__(self, backend: SimulationBackend = SimulationBackend.CUDA):
        self.backend = backend
        self.logger = get_logger(self.__class__.__name__)
        self.simulation_engine = NonEquilibriumSimulationEngine(backend)

    @abstractmethod
    def optimize(
        self,
        target_properties: Dict[str, float],
        optimization_variables: List[OptimizationVariable],
        constraints: OptimizationConstraints,
        **kwargs,
    ) -> OptimizationResult:
        """Perform inverse design optimization."""
        pass

    @abstractmethod
    def evaluate_design(
        self, design_parameters: Dict[str, Any], target_properties: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate a specific design configuration."""
        pass


class DynamicInverseDesigner(BaseInverseDesigner):
    """Inverse designer for dynamic/non-equilibrium properties."""

    def __init__(self, backend: SimulationBackend = SimulationBackend.CUDA):
        super().__init__(backend)
        self.property_predictors = {}
        self._setup_property_predictors()

    def _setup_property_predictors(self):
        """Initialize property prediction models."""
        if SCIKIT_AVAILABLE:
            # Gaussian Process models for different properties
            for target in OptimizationTarget:
                kernel = Matern(length_scale=1.0, nu=2.5)
                self.property_predictors[target.value] = GaussianProcessRegressor(
                    kernel=kernel, alpha=1e-6, n_restarts_optimizer=10
                )

    def optimize(
        self,
        target_properties: Dict[str, float],
        optimization_variables: List[OptimizationVariable],
        constraints: OptimizationConstraints,
        max_evaluations: int = 100,
        **kwargs,
    ) -> OptimizationResult:
        """Optimize design for dynamic properties under extreme conditions."""

        start_time = time.time()
        self.logger.info("Starting dynamic inverse design optimization")

        # Initialize optimization result
        result = OptimizationResult()

        # Setup optimization problem
        bounds = self._setup_optimization_bounds(optimization_variables, constraints)

        # Initialize Bayesian optimizer
        if not SCIKIT_AVAILABLE:
            raise OptimizationError("Scikit-learn required for Bayesian optimization")

        optimizer = BayesianOptimizer(
            bounds=bounds,
            acquisition_function="ei",  # Expected Improvement
            backend=self.backend,
        )

        # Optimization loop
        best_design = None
        best_score = float("inf")

        for iteration in range(max_evaluations):
            # Get next candidate design
            candidate = optimizer.suggest_next_candidate()

            # Convert candidate to design parameters
            design_params = self._candidate_to_design_params(
                candidate, optimization_variables
            )

            # Evaluate design
            try:
                scores = self.evaluate_design(design_params, target_properties)
                total_score = sum(scores.values())

                # Update optimizer
                optimizer.update(candidate, total_score)

                # Track best design
                if total_score < best_score:
                    best_score = total_score
                    best_design = design_params.copy()
                    result.predicted_properties = scores.copy()

                result.n_evaluations += 1

                if iteration % 10 == 0:
                    self.logger.info(
                        f"Iteration {iteration}: Best score = {best_score:.4f}"
                    )

            except Exception as e:
                self.logger.warning(f"Evaluation failed for iteration {iteration}: {e}")
                # Assign penalty score
                optimizer.update(candidate, 1e6)

        # Extract optimal design
        if best_design is not None:
            result.optimal_composition = best_design.get("composition", {})
            result.optimal_processing = best_design.get("processing", {})
            result.optimal_microstructure = best_design.get("microstructure", {})

            # Calculate final objective values
            for prop, target in target_properties.items():
                predicted = result.predicted_properties.get(prop, 0.0)
                result.objective_values[prop] = abs(predicted - target)

        result.optimization_time = time.time() - start_time
        result.convergence_achieved = best_score < 1e-3  # Threshold for convergence

        self.logger.info(
            f"Optimization completed in {result.optimization_time:.2f}s "
            f"with {result.n_evaluations} evaluations"
        )

        return result

    def evaluate_design(
        self, design_parameters: Dict[str, Any], target_properties: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate design through simulation or surrogate models."""

        scores = {}

        # Generate structure from design parameters
        structure = self._generate_structure_from_design(design_parameters)

        # Determine relevant extreme conditions for target properties
        conditions = self._map_properties_to_conditions(target_properties)

        for condition in conditions:
            # Run simulation
            try:
                sim_results = self.simulation_engine.run_extreme_condition_simulation(
                    structure=structure,
                    condition_type=condition,
                    simulator_type="nemd",
                    total_time=1e-13,  # Short simulation for optimization
                    output_frequency=10,
                )

                # Extract relevant properties
                properties = self.simulation_engine.available_simulators[
                    "nemd"
                ].calculate_properties(sim_results)

                # Map simulation properties to target properties
                mapped_scores = self._map_simulation_to_targets(
                    properties, target_properties, condition
                )
                scores.update(mapped_scores)

            except Exception as e:
                self.logger.warning(f"Simulation failed for condition {condition}: {e}")
                # Assign penalty scores
                for prop in target_properties:
                    scores[prop] = 1e6

        # Use surrogate models if simulation fails or as backup
        if not scores:
            scores = self._evaluate_with_surrogate_models(
                design_parameters, target_properties
            )

        return scores

    def _setup_optimization_bounds(
        self,
        variables: List[OptimizationVariable],
        constraints: OptimizationConstraints,
    ) -> List[Tuple[float, float]]:
        """Setup optimization bounds for each variable."""
        bounds = []

        for var in variables:
            if var == OptimizationVariable.COMPOSITION:
                # Bounds for each element (assuming 5 common elements)
                for _ in range(5):
                    bounds.append((0.0, 1.0))

            elif var == OptimizationVariable.MICROSTRUCTURE:
                # Grain size bounds
                bounds.append((1e-9, 1e-3))  # 1nm to 1mm
                # Phase fraction bounds
                bounds.append((0.0, 1.0))

            elif var == OptimizationVariable.PROCESSING_PATHWAY:
                # Temperature bounds
                bounds.append(
                    (constraints.min_temperature, constraints.max_temperature)
                )
                # Pressure bounds
                bounds.append((constraints.min_pressure, constraints.max_pressure))
                # Time bounds
                bounds.append((1.0, 10000.0))  # seconds

            elif var == OptimizationVariable.DEFECT_DENSITY:
                bounds.append((1e10, 1e16))  # defects/m³

            else:
                # Default bounds for other variables
                bounds.append((0.0, 1.0))

        return bounds

    def _candidate_to_design_params(
        self, candidate: np.ndarray, variables: List[OptimizationVariable]
    ) -> Dict[str, Any]:
        """Convert optimization candidate to design parameters."""
        design_params = {"composition": {}, "processing": {}, "microstructure": {}}

        idx = 0
        for var in variables:
            if var == OptimizationVariable.COMPOSITION:
                elements = ["Fe", "Ni", "Cr", "Al", "Ti"]
                for elem in elements:
                    design_params["composition"][elem] = candidate[idx]
                    idx += 1

                # Normalize composition
                total = sum(design_params["composition"].values())
                if total > 0:
                    for elem in elements:
                        design_params["composition"][elem] /= total

            elif var == OptimizationVariable.MICROSTRUCTURE:
                design_params["microstructure"]["grain_size"] = candidate[idx]
                design_params["microstructure"]["phase_fraction"] = candidate[idx + 1]
                idx += 2

            elif var == OptimizationVariable.PROCESSING_PATHWAY:
                design_params["processing"]["temperature"] = candidate[idx]
                design_params["processing"]["pressure"] = candidate[idx + 1]
                design_params["processing"]["time"] = candidate[idx + 2]
                idx += 3

            else:
                design_params[var.value] = candidate[idx]
                idx += 1

        return design_params

    def _generate_structure_from_design(
        self, design_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate atomic structure from design parameters."""

        # Simple cubic lattice generation based on composition
        composition = design_parameters.get("composition", {"Fe": 1.0})
        n_atoms = 100  # Small system for optimization

        # Generate positions
        side_length = int(np.ceil(n_atoms ** (1 / 3)))
        positions = []

        for i in range(side_length):
            for j in range(side_length):
                for k in range(side_length):
                    if len(positions) < n_atoms:
                        positions.append([i * 3.0, j * 3.0, k * 3.0])

        positions = np.array(positions[:n_atoms])

        # Add random displacement based on microstructure
        microstructure = design_parameters.get("microstructure", {})
        grain_size = microstructure.get("grain_size", 1e-6)
        disorder = min(0.5, 1e-6 / grain_size)  # More disorder for smaller grains

        positions += np.random.normal(0, disorder, positions.shape)

        structure = {
            "positions": positions,
            "composition": composition,
            "n_atoms": n_atoms,
        }

        return structure

    def _map_properties_to_conditions(
        self, target_properties: Dict[str, float]
    ) -> List[ExtremeConditionType]:
        """Map target properties to relevant extreme conditions."""

        property_to_condition = {
            "damage_resistance": ExtremeConditionType.HIGH_STRAIN_RATE,
            "shock_absorption": ExtremeConditionType.SHOCK_LOADING,
            "radiation_tolerance": ExtremeConditionType.RADIATION_DAMAGE,
            "thermal_stability": ExtremeConditionType.HIGH_TEMPERATURE,
            "fracture_toughness": ExtremeConditionType.HIGH_STRAIN_RATE,
            "fatigue_resistance": ExtremeConditionType.FATIGUE_LOADING,
            "corrosion_resistance": ExtremeConditionType.CORROSIVE_ENVIRONMENT,
        }

        conditions = []
        for prop in target_properties:
            if prop in property_to_condition:
                conditions.append(property_to_condition[prop])

        # Default to high strain rate if no specific mapping
        if not conditions:
            conditions.append(ExtremeConditionType.HIGH_STRAIN_RATE)

        return list(set(conditions))  # Remove duplicates

    def _map_simulation_to_targets(
        self,
        sim_properties: Dict[str, float],
        target_properties: Dict[str, float],
        condition: ExtremeConditionType,
    ) -> Dict[str, float]:
        """Map simulation properties to target property scores."""

        scores = {}

        # Define mapping from simulation outputs to target properties
        property_mappings = {
            "damage_resistance": "final_damage_fraction",
            "shock_absorption": "avg_energy",
            "radiation_tolerance": "final_defect_count",
            "thermal_stability": "avg_temperature",
            "fracture_toughness": "von_mises_stress",
            "fatigue_resistance": "equivalent_strain",
        }

        for target_prop, target_value in target_properties.items():
            if target_prop in property_mappings:
                sim_prop = property_mappings[target_prop]
                if sim_prop in sim_properties:
                    predicted_value = sim_properties[sim_prop]

                    # Calculate normalized error
                    if target_value != 0:
                        error = abs(predicted_value - target_value) / abs(target_value)
                    else:
                        error = abs(predicted_value)

                    scores[target_prop] = error
                else:
                    scores[target_prop] = 1.0  # Default penalty
            else:
                scores[target_prop] = 1.0  # Unknown property penalty

        return scores

    def _evaluate_with_surrogate_models(
        self, design_parameters: Dict[str, Any], target_properties: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate design using surrogate models."""

        scores = {}

        # Simple heuristic-based surrogate models
        composition = design_parameters.get("composition", {})

        for prop, target in target_properties.items():
            if prop == "damage_resistance":
                # Higher Cr content improves damage resistance
                cr_content = composition.get("Cr", 0.0)
                predicted = 0.8 + 0.2 * cr_content
                scores[prop] = abs(predicted - target) / max(target, 1e-6)

            elif prop == "thermal_stability":
                # Higher Ni content improves thermal stability
                ni_content = composition.get("Ni", 0.0)
                predicted = 0.7 + 0.3 * ni_content
                scores[prop] = abs(predicted - target) / max(target, 1e-6)

            else:
                # Default random prediction with some composition dependence
                fe_content = composition.get("Fe", 0.5)
                predicted = 0.5 + 0.5 * fe_content + np.random.normal(0, 0.1)
                scores[prop] = abs(predicted - target) / max(target, 1e-6)

        return scores


class NonEquilibriumOptimizer(BaseInverseDesigner):
    """Specialized optimizer for non-equilibrium processing conditions."""

    def __init__(self, backend: SimulationBackend = SimulationBackend.CUDA):
        super().__init__(backend)
        self.multi_objective_optimizer = MultiObjectiveOptimizer()

    def optimize(
        self,
        target_properties: Dict[str, float],
        optimization_variables: List[OptimizationVariable],
        constraints: OptimizationConstraints,
        **kwargs,
    ) -> OptimizationResult:
        """Optimize for non-equilibrium processing conditions."""

        self.logger.info("Starting non-equilibrium optimization")

        # Define objectives for multi-objective optimization
        objectives = []
        for prop, target in target_properties.items():
            objectives.append(
                {
                    "name": prop,
                    "target": target,
                    "weight": kwargs.get(f"{prop}_weight", 1.0),
                    "type": "minimize",  # Minimize error from target
                }
            )

        # Setup parameter space
        param_space = self._setup_parameter_space(optimization_variables, constraints)

        # Run multi-objective optimization
        pareto_solutions = self.multi_objective_optimizer.optimize_multiple_objectives(
            objectives=objectives,
            parameter_space=param_space,
            n_generations=kwargs.get("n_generations", 50),
            population_size=kwargs.get("population_size", 100),
        )

        # Select best solution from Pareto front
        best_solution = self._select_best_pareto_solution(
            pareto_solutions, target_properties
        )

        result = OptimizationResult()
        if best_solution:
            result.optimal_composition = best_solution.get("composition", {})
            result.optimal_processing = best_solution.get("processing", {})
            result.predicted_properties = best_solution.get("properties", {})
            result.convergence_achieved = True

        return result

    def evaluate_design(
        self, design_parameters: Dict[str, Any], target_properties: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate design for non-equilibrium conditions."""

        # Focus on rapid processing conditions
        processing = design_parameters.get("processing", {})

        scores = {}
        for prop, target in target_properties.items():
            # Non-equilibrium specific evaluation
            if prop == "rapid_solidification_rate":
                cooling_rate = processing.get("cooling_rate", 1e3)  # K/s
                predicted = min(1.0, cooling_rate / 1e6)  # Normalize
                scores[prop] = abs(predicted - target)

            elif prop == "quench_sensitivity":
                quench_rate = processing.get("quench_rate", 1e2)  # K/s
                predicted = 1.0 - min(1.0, quench_rate / 1e5)
                scores[prop] = abs(predicted - target)

            else:
                # Use base class evaluation
                base_scores = super().evaluate_design(design_parameters, {prop: target})
                scores.update(base_scores)

        return scores

    def _setup_parameter_space(
        self,
        variables: List[OptimizationVariable],
        constraints: OptimizationConstraints,
    ) -> Dict[str, Any]:
        """Setup parameter space for multi-objective optimization."""

        param_space = {}

        for var in variables:
            if var == OptimizationVariable.PROCESSING_PATHWAY:
                param_space["processing"] = {
                    "temperature": (
                        constraints.min_temperature,
                        constraints.max_temperature,
                    ),
                    "pressure": (constraints.min_pressure, constraints.max_pressure),
                    "cooling_rate": (1e2, 1e7),  # K/s
                    "quench_rate": (1e1, 1e6),  # K/s
                    "holding_time": (1.0, 3600.0),  # seconds
                }

            elif var == OptimizationVariable.COMPOSITION:
                param_space["composition"] = {
                    "Fe": (0.0, 1.0),
                    "Ni": (0.0, 0.8),
                    "Cr": (0.0, 0.5),
                    "Al": (0.0, 0.2),
                    "Ti": (0.0, 0.1),
                }

        return param_space

    def _select_best_pareto_solution(
        self,
        pareto_solutions: List[Dict[str, Any]],
        target_properties: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Select best solution from Pareto front."""

        if not pareto_solutions:
            return None

        # Score each solution based on distance to targets
        best_solution = None
        best_score = float("inf")

        for solution in pareto_solutions:
            total_score = 0.0
            properties = solution.get("properties", {})

            for prop, target in target_properties.items():
                if prop in properties:
                    error = abs(properties[prop] - target) / max(abs(target), 1e-6)
                    total_score += error**2

            if total_score < best_score:
                best_score = total_score
                best_solution = solution

        return best_solution


class ProcessingPathwayOptimizer(BaseInverseDesigner):
    """Optimizer specifically for processing pathway design."""

    def __init__(self, backend: SimulationBackend = SimulationBackend.CUDA):
        super().__init__(backend)
        self.thermal_models = {}
        self._setup_thermal_models()

    def _setup_thermal_models(self):
        """Setup thermal processing models."""
        # Simple thermal models for different processes
        self.thermal_models = {
            "annealing": self._model_annealing,
            "quenching": self._model_quenching,
            "tempering": self._model_tempering,
            "solution_treatment": self._model_solution_treatment,
        }

    def optimize(
        self,
        target_properties: Dict[str, float],
        optimization_variables: List[OptimizationVariable],
        constraints: OptimizationConstraints,
        **kwargs,
    ) -> OptimizationResult:
        """Optimize processing pathway for target properties."""

        self.logger.info("Starting processing pathway optimization")

        # Define processing steps
        processing_steps = kwargs.get(
            "processing_steps", ["solution_treatment", "quenching", "tempering"]
        )

        result = OptimizationResult()
        optimal_pathway = {}

        # Optimize each processing step
        for step in processing_steps:
            if step in self.thermal_models:
                step_params = self._optimize_processing_step(
                    step, target_properties, constraints
                )
                optimal_pathway[step] = step_params

        result.optimal_processing = optimal_pathway

        # Evaluate final pathway
        final_scores = self.evaluate_design(
            {"processing": optimal_pathway}, target_properties
        )
        result.predicted_properties = final_scores

        return result

    def evaluate_design(
        self, design_parameters: Dict[str, Any], target_properties: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate processing pathway design."""

        processing = design_parameters.get("processing", {})
        scores = {}

        # Simulate processing pathway effects
        for prop, target in target_properties.items():
            predicted_value = target  # Start with target

            # Apply effects of each processing step
            for step, params in processing.items():
                if step in self.thermal_models:
                    effect = self.thermal_models[step](params, prop)
                    predicted_value *= effect

            # Calculate score
            scores[prop] = abs(predicted_value - target) / max(abs(target), 1e-6)

        return scores

    def _optimize_processing_step(
        self,
        step: str,
        target_properties: Dict[str, float],
        constraints: OptimizationConstraints,
    ) -> Dict[str, float]:
        """Optimize parameters for a single processing step."""

        # Simple optimization for each step
        if step == "solution_treatment":
            return {
                "temperature": min(constraints.max_temperature, 1273.0),  # 1000°C
                "time": 3600.0,  # 1 hour
                "atmosphere": "inert",
            }

        elif step == "quenching":
            return {
                "cooling_rate": 1e5,  # K/s
                "quench_medium": "water",
                "final_temperature": 300.0,  # K
            }

        elif step == "tempering":
            return {
                "temperature": 573.0,  # 300°C
                "time": 7200.0,  # 2 hours
                "cooling_rate": 1e2,  # K/s
            }

        else:
            return {"temperature": 573.0, "time": 3600.0}

    def _model_annealing(self, params: Dict[str, float], property_name: str) -> float:
        """Model annealing effects on properties."""
        temperature = params.get("temperature", 873.0)
        time = params.get("time", 3600.0)

        # Simple empirical model
        if property_name == "ductility":
            return (
                1.0 + 0.1 * np.log(time / 3600.0) + 0.05 * (temperature - 873.0) / 100.0
            )
        elif property_name == "strength":
            return 0.9 - 0.05 * np.log(time / 3600.0)
        else:
            return 1.0

    def _model_quenching(self, params: Dict[str, float], property_name: str) -> float:
        """Model quenching effects on properties."""
        cooling_rate = params.get("cooling_rate", 1e3)

        if property_name == "strength":
            return 1.0 + 0.2 * np.log10(cooling_rate / 1e3)
        elif property_name == "ductility":
            return 0.8 - 0.1 * np.log10(cooling_rate / 1e3)
        else:
            return 1.0

    def _model_tempering(self, params: Dict[str, float], property_name: str) -> float:
        """Model tempering effects on properties."""
        temperature = params.get("temperature", 573.0)
        time = params.get("time", 7200.0)

        if property_name == "toughness":
            return (
                1.0
                + 0.15 * (temperature - 473.0) / 100.0
                + 0.05 * np.log(time / 3600.0)
            )
        elif property_name == "strength":
            return 0.95 + 0.05 * (673.0 - temperature) / 100.0
        else:
            return 1.0

    def _model_solution_treatment(
        self, params: Dict[str, float], property_name: str
    ) -> float:
        """Model solution treatment effects on properties."""
        temperature = params.get("temperature", 1273.0)
        time = params.get("time", 3600.0)

        if property_name == "corrosion_resistance":
            return 1.0 + 0.1 * (temperature - 1073.0) / 100.0
        elif property_name == "strength":
            return 1.05 - 0.02 * np.log(time / 3600.0)
        else:
            return 1.0

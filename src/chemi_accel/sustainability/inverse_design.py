"""
Inverse Design for Sustainable Materials.

This module provides comprehensive inverse material design capabilities with explicit
sustainability metrics optimization including degradation rate, recyclability score,
and carbon footprint as optimization objectives or constraints.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import scipy.optimize
    from scipy.optimize import differential_evolution, minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import from other sustainability modules
from .degradation_modeling import (
    EnvironmentalConditions,
    MultiScaleDegradationPredictor,
    create_standard_material,
)
from .lca_integration import (
    ImpactCategory,
    MatrixLCACalculator,
    create_polymer_manufacturing_process,
)
from .recyclability_assessment import (
    ComprehensiveRecyclabilityAssessor,
    MaterialComposition,
)

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for sustainable material design."""

    MINIMIZE_DEGRADATION_RATE = "minimize_degradation_rate"
    MAXIMIZE_RECYCLABILITY = "maximize_recyclability"
    MINIMIZE_CARBON_FOOTPRINT = "minimize_carbon_footprint"
    MINIMIZE_TOXICITY = "minimize_toxicity"
    MAXIMIZE_DURABILITY = "maximize_durability"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    MINIMIZE_ENVIRONMENTAL_IMPACT = "minimize_environmental_impact"


class ConstraintType(Enum):
    """Types of constraints for material design."""

    INEQUALITY = "inequality"
    EQUALITY = "equality"
    BOUND = "bound"


class OptimizationAlgorithm(Enum):
    """Available optimization algorithms."""

    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRADIENT_DESCENT = "gradient_descent"
    SIMULATED_ANNEALING = "simulated_annealing"
    MULTI_OBJECTIVE_GENETIC = "multi_objective_genetic"


@dataclass
class MaterialParameter:
    """Definition of a material parameter for optimization."""

    name: str
    min_value: float
    max_value: float
    initial_value: Optional[float] = None
    parameter_type: str = "continuous"  # continuous, integer, categorical
    possible_values: Optional[List[Any]] = None  # For categorical parameters
    description: str = ""


@dataclass
class OptimizationConstraint:
    """Constraint definition for material optimization."""

    name: str
    constraint_type: ConstraintType
    function: Callable
    target_value: Optional[float] = None
    tolerance: float = 1e-6
    weight: float = 1.0
    description: str = ""


@dataclass
class SustainabilityTarget:
    """Target definition for sustainability metrics."""

    objective: OptimizationObjective
    target_value: Optional[float] = None
    weight: float = 1.0
    minimize: bool = True
    constraint: bool = False
    tolerance: float = 1e-3


@dataclass
class DesignResult:
    """Result of inverse design optimization."""

    optimal_parameters: Dict[str, float]
    objective_values: Dict[OptimizationObjective, float]
    constraint_violations: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    convergence_metrics: Dict[str, float]
    sustainability_scores: Dict[str, float]
    predicted_properties: Dict[str, Any]
    computation_time: float
    success: bool
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""

    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.MULTI_OBJECTIVE_GENETIC
    max_iterations: int = 1000
    population_size: int = 100
    tolerance: float = 1e-6
    use_gpu: bool = True
    parallel_evaluation: bool = True
    random_seed: Optional[int] = None
    convergence_criteria: Dict[str, float] = field(default_factory=dict)
    algorithm_params: Dict[str, Any] = field(default_factory=dict)


class MaterialPropertyPredictor(ABC):
    """Abstract base class for material property prediction."""

    @abstractmethod
    async def predict_properties(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Predict material properties from design parameters."""
        pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        """Validate parameter values."""
        pass


class SustainabilityPropertyPredictor(MaterialPropertyPredictor):
    """Predictor for sustainability-related material properties."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and TORCH_AVAILABLE

        # Initialize predictors
        self.degradation_predictor = MultiScaleDegradationPredictor(
            use_gpu=self.use_gpu
        )
        self.recyclability_assessor = ComprehensiveRecyclabilityAssessor(
            use_gpu=self.use_gpu
        )
        self.lca_calculator = MatrixLCACalculator(use_gpu=self.use_gpu)

        # Property prediction models
        self.property_models = {}
        self._initialize_property_models()

    def _initialize_property_models(self):
        """Initialize property prediction models."""
        # Simple regression models for demonstration
        # In practice, these would be trained ML models

        self.property_models = {
            "molecular_weight": lambda params: params.get("monomer_ratio", 0.5) * 50000
            + 25000,
            "glass_transition_temp": lambda params: 300
            + params.get("crosslink_density", 0.1) * 100,
            "crystallinity": lambda params: max(
                0, min(1, params.get("isotacticity", 0.5))
            ),
            "tensile_strength": lambda params: 50
            + params.get("filler_content", 0.0) * 100,
            "degradation_temp": lambda params: 250
            + params.get("stabilizer_content", 0.02) * 1000,
        }

    async def predict_properties(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Predict comprehensive material properties."""
        properties = {}

        # Basic material properties
        for prop_name, model in self.property_models.items():
            try:
                properties[prop_name] = model(parameters)
            except Exception as e:
                logger.warning(f"Failed to predict {prop_name}: {e}")
                properties[prop_name] = 0.0

        # Create material for advanced predictions
        material_props = create_standard_material("polymer")
        material_props.molecular_weight = properties.get("molecular_weight", 50000)
        material_props.glass_transition_temp = properties.get(
            "glass_transition_temp", 350
        )
        material_props.crystallinity = properties.get("crystallinity", 0.3)

        # Sustainability predictions
        try:
            # Degradation modeling
            aging_conditions = EnvironmentalConditions(
                stressors={"temperature": 353.15, "humidity": 0.8}, duration=1000.0
            )
            degradation_pred = await self.degradation_predictor.predict_degradation(
                material_props, aging_conditions
            )
            properties["degradation_rate"] = (
                1.0 / degradation_pred.half_life
                if degradation_pred.half_life
                else 0.001
            )
            properties["half_life"] = degradation_pred.half_life or 1000.0

            # Recyclability assessment
            composition = MaterialComposition(
                polymer_content={"polymer": parameters.get("polymer_purity", 0.95)},
                additive_content={
                    "additives": 1.0 - parameters.get("polymer_purity", 0.95)
                },
                filler_content={},
            )
            recyclability_assessment = (
                await self.recyclability_assessor.assess_comprehensive_recyclability(
                    composition
                )
            )
            properties[
                "recyclability_score"
            ] = recyclability_assessment.circularity_metrics.recyclability_score
            properties[
                "circularity_index"
            ] = recyclability_assessment.circularity_metrics.circularity_index

            # LCA assessment
            process = create_polymer_manufacturing_process("custom")
            # Modify process based on parameters
            lca_factors = self.lca_calculator._create_default_characterization_factors()
            lca_results = self.lca_calculator.calculate_impacts([process], lca_factors)
            properties["carbon_footprint"] = lca_results.impact_scores.get(
                ImpactCategory.CLIMATE_CHANGE, 2.0
            )

        except Exception as e:
            logger.warning(f"Failed to predict sustainability properties: {e}")
            properties.update(
                {
                    "degradation_rate": 0.001,
                    "half_life": 1000.0,
                    "recyclability_score": 0.5,
                    "circularity_index": 0.5,
                    "carbon_footprint": 2.0,
                }
            )

        return properties

    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        """Validate parameter values."""
        # Check basic constraints
        for param, value in parameters.items():
            if not isinstance(value, (int, float)):
                return False
            if param.endswith("_ratio") and not (0 <= value <= 1):
                return False
            if param.endswith("_content") and not (0 <= value <= 1):
                return False

        return True


class OptimizationProblem(Problem if PYMOO_AVAILABLE else object):
    """Multi-objective optimization problem for sustainable material design."""

    def __init__(
        self,
        parameters: List[MaterialParameter],
        objectives: List[SustainabilityTarget],
        constraints: List[OptimizationConstraint],
        property_predictor: MaterialPropertyPredictor,
    ):
        self.parameters = parameters
        self.objectives = objectives
        self.constraints = constraints
        self.property_predictor = property_predictor

        # Parameter mapping
        self.param_names = [p.name for p in parameters]
        self.param_bounds = [(p.min_value, p.max_value) for p in parameters]

        n_var = len(parameters)
        n_obj = len(objectives)
        n_constr = len(constraints)

        # Initialize pymoo Problem if available
        if PYMOO_AVAILABLE:
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                n_constr=n_constr,
                xl=np.array([p.min_value for p in parameters]),
                xu=np.array([p.max_value for p in parameters]),
            )

    async def evaluate_async(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Async evaluation of objectives and constraints."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        objectives = []
        constraints = []

        for individual in x:
            # Convert to parameter dict
            params = dict(zip(self.param_names, individual))

            # Predict properties
            try:
                properties = await self.property_predictor.predict_properties(params)
            except Exception as e:
                logger.error(f"Property prediction failed: {e}")
                # Return penalty values
                objectives.append([1e6] * len(self.objectives))
                constraints.append([1e6] * len(self.constraints))
                continue

            # Evaluate objectives
            obj_values = []
            for obj_target in self.objectives:
                value = self._evaluate_objective(obj_target, properties)
                obj_values.append(value)
            objectives.append(obj_values)

            # Evaluate constraints
            constr_values = []
            for constraint in self.constraints:
                value = self._evaluate_constraint(constraint, properties, params)
                constr_values.append(value)
            constraints.append(constr_values)

        return np.array(objectives), np.array(constraints)

    def _evaluate_objective(
        self, target: SustainabilityTarget, properties: Dict[str, Any]
    ) -> float:
        """Evaluate a single objective."""
        obj_map = {
            OptimizationObjective.MINIMIZE_DEGRADATION_RATE: "degradation_rate",
            OptimizationObjective.MAXIMIZE_RECYCLABILITY: "recyclability_score",
            OptimizationObjective.MINIMIZE_CARBON_FOOTPRINT: "carbon_footprint",
            OptimizationObjective.MAXIMIZE_DURABILITY: "half_life",
            OptimizationObjective.MINIMIZE_ENVIRONMENTAL_IMPACT: "carbon_footprint",
        }

        prop_name = obj_map.get(target.objective, "degradation_rate")
        value = properties.get(prop_name, 0.0)

        # Convert maximization to minimization
        if not target.minimize:
            value = -value

        return value * target.weight

    def _evaluate_constraint(
        self,
        constraint: OptimizationConstraint,
        properties: Dict[str, Any],
        parameters: Dict[str, float],
    ) -> float:
        """Evaluate a single constraint."""
        try:
            # Call constraint function
            value = constraint.function(properties, parameters)

            if constraint.constraint_type == ConstraintType.INEQUALITY:
                # g(x) <= 0 format
                return value
            elif constraint.constraint_type == ConstraintType.EQUALITY:
                # h(x) = 0 format
                return (
                    abs(value - (constraint.target_value or 0.0)) - constraint.tolerance
                )
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Constraint evaluation failed: {e}")
            return 1e6  # Large penalty

    def _evaluate(self, x, out, *args, **kwargs):
        """Synchronous evaluation for pymoo compatibility."""
        if PYMOO_AVAILABLE:
            # Run async evaluation in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                objectives, constraints = loop.run_until_complete(
                    self.evaluate_async(x)
                )
                out["F"] = objectives
                if constraints.size > 0:
                    out["G"] = constraints
            finally:
                loop.close()


class SustainableInverseDesigner:
    """Main class for inverse design of sustainable materials."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.property_predictor = SustainabilityPropertyPredictor(
            use_gpu=config.use_gpu
        )

        # Optimization state
        self.current_problem = None
        self.optimization_history = []
        self.best_solutions = []

        # Performance tracking
        self.evaluation_count = 0
        self.start_time = None

    async def design_material(
        self,
        parameters: List[MaterialParameter],
        objectives: List[SustainabilityTarget],
        constraints: Optional[List[OptimizationConstraint]] = None,
    ) -> DesignResult:
        """Design optimal sustainable material."""

        self.start_time = time.time()
        constraints = constraints or []

        # Create optimization problem
        self.current_problem = OptimizationProblem(
            parameters, objectives, constraints, self.property_predictor
        )

        try:
            # Select and run optimization algorithm
            if self.config.algorithm == OptimizationAlgorithm.MULTI_OBJECTIVE_GENETIC:
                result = await self._run_multi_objective_genetic(
                    parameters, objectives, constraints
                )
            elif self.config.algorithm == OptimizationAlgorithm.BAYESIAN_OPTIMIZATION:
                result = await self._run_bayesian_optimization(
                    parameters, objectives, constraints
                )
            elif self.config.algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
                result = await self._run_genetic_algorithm(
                    parameters, objectives, constraints
                )
            else:
                result = await self._run_default_optimization(
                    parameters, objectives, constraints
                )

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return DesignResult(
                optimal_parameters={},
                objective_values={},
                constraint_violations={},
                optimization_history=[],
                convergence_metrics={},
                sustainability_scores={},
                predicted_properties={},
                computation_time=time.time() - self.start_time,
                success=False,
                message=f"Optimization failed: {str(e)}",
            )

    async def _run_multi_objective_genetic(
        self,
        parameters: List[MaterialParameter],
        objectives: List[SustainabilityTarget],
        constraints: List[OptimizationConstraint],
    ) -> DesignResult:
        """Run multi-objective genetic algorithm optimization."""

        if not PYMOO_AVAILABLE:
            logger.warning("pymoo not available, falling back to default optimization")
            return await self._run_default_optimization(
                parameters, objectives, constraints
            )

        # Configure NSGA-II
        algorithm = NSGA2(
            pop_size=self.config.population_size, eliminate_duplicates=True
        )

        # Run optimization
        res = pymoo_minimize(
            self.current_problem,
            algorithm,
            termination=("n_gen", self.config.max_iterations),
            seed=self.config.random_seed,
            save_history=True,
            verbose=True,
        )

        # Extract best solution (first from Pareto front)
        if res.X is not None:
            if res.X.ndim == 1:
                best_x = res.X
            else:
                best_x = res.X[0]  # First solution from Pareto front

            optimal_params = dict(zip([p.name for p in parameters], best_x))

            # Evaluate final solution
            properties = await self.property_predictor.predict_properties(
                optimal_params
            )

            # Calculate objective values
            objective_values = {}
            for i, obj_target in enumerate(objectives):
                if res.F.ndim == 1:
                    objective_values[obj_target.objective] = res.F[i]
                else:
                    objective_values[obj_target.objective] = res.F[0][i]

            # Calculate sustainability scores
            sustainability_scores = {
                "degradation_resistance": 1.0
                / (properties.get("degradation_rate", 0.001) + 1e-6),
                "recyclability": properties.get("recyclability_score", 0.5),
                "environmental_impact": 1.0
                / (properties.get("carbon_footprint", 2.0) + 1e-6),
                "overall_sustainability": self._calculate_overall_sustainability(
                    properties
                ),
            }

            return DesignResult(
                optimal_parameters=optimal_params,
                objective_values=objective_values,
                constraint_violations={},
                optimization_history=self.optimization_history,
                convergence_metrics={"generations": len(res.history)},
                sustainability_scores=sustainability_scores,
                predicted_properties=properties,
                computation_time=time.time() - self.start_time,
                success=True,
                message="Multi-objective optimization completed successfully",
                metadata={
                    "algorithm": "NSGA-II",
                    "pareto_size": len(res.X) if res.X.ndim > 1 else 1,
                },
            )
        else:
            raise RuntimeError("Optimization failed to find valid solutions")

    async def _run_bayesian_optimization(
        self,
        parameters: List[MaterialParameter],
        objectives: List[SustainabilityTarget],
        constraints: List[OptimizationConstraint],
    ) -> DesignResult:
        """Run Bayesian optimization using Optuna."""

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to default optimization")
            return await self._run_default_optimization(
                parameters, objectives, constraints
            )

        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_seed),
        )

        async def objective_function(trial):
            # Sample parameters
            params = {}
            for param in parameters:
                if param.parameter_type == "continuous":
                    params[param.name] = trial.suggest_float(
                        param.name, param.min_value, param.max_value
                    )
                elif param.parameter_type == "integer":
                    params[param.name] = trial.suggest_int(
                        param.name, int(param.min_value), int(param.max_value)
                    )
                elif param.parameter_type == "categorical":
                    params[param.name] = trial.suggest_categorical(
                        param.name,
                        param.possible_values or [param.min_value, param.max_value],
                    )

            # Predict properties
            properties = await self.property_predictor.predict_properties(params)

            # Calculate weighted objective
            total_objective = 0.0
            for obj_target in objectives:
                value = self.current_problem._evaluate_objective(obj_target, properties)
                total_objective += value

            # Add constraint penalties
            for constraint in constraints:
                violation = self.current_problem._evaluate_constraint(
                    constraint, properties, params
                )
                if violation > 0:  # Constraint violated
                    total_objective += violation * 1000  # Large penalty

            return total_objective

        # Run optimization
        for _ in range(self.config.max_iterations):
            trial = study.ask()
            objective_value = await objective_function(trial)
            study.tell(trial, objective_value)

        # Extract best solution
        best_params = study.best_params
        properties = await self.property_predictor.predict_properties(best_params)

        # Calculate objective values
        objective_values = {}
        for obj_target in objectives:
            value = self.current_problem._evaluate_objective(obj_target, properties)
            objective_values[obj_target.objective] = value

        sustainability_scores = {
            "degradation_resistance": 1.0
            / (properties.get("degradation_rate", 0.001) + 1e-6),
            "recyclability": properties.get("recyclability_score", 0.5),
            "environmental_impact": 1.0
            / (properties.get("carbon_footprint", 2.0) + 1e-6),
            "overall_sustainability": self._calculate_overall_sustainability(
                properties
            ),
        }

        return DesignResult(
            optimal_parameters=best_params,
            objective_values=objective_values,
            constraint_violations={},
            optimization_history=[],
            convergence_metrics={
                "best_value": study.best_value,
                "n_trials": len(study.trials),
            },
            sustainability_scores=sustainability_scores,
            predicted_properties=properties,
            computation_time=time.time() - self.start_time,
            success=True,
            message="Bayesian optimization completed successfully",
            metadata={"algorithm": "TPE", "n_trials": len(study.trials)},
        )

    async def _run_genetic_algorithm(
        self,
        parameters: List[MaterialParameter],
        objectives: List[SustainabilityTarget],
        constraints: List[OptimizationConstraint],
    ) -> DesignResult:
        """Run single-objective genetic algorithm using scipy."""

        if not SCIPY_AVAILABLE:
            return await self._run_default_optimization(
                parameters, objectives, constraints
            )

        # Define objective function for scipy
        async def objective_func(x):
            params = dict(zip([p.name for p in parameters], x))
            properties = await self.property_predictor.predict_properties(params)

            # Combine objectives into single value
            total_obj = 0.0
            for obj_target in objectives:
                value = self.current_problem._evaluate_objective(obj_target, properties)
                total_obj += value

            # Add constraint penalties
            for constraint in constraints:
                violation = self.current_problem._evaluate_constraint(
                    constraint, properties, params
                )
                if violation > 0:
                    total_obj += violation * 1000

            return total_obj

        # Wrapper for sync scipy interface
        def sync_objective_func(x):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(objective_func(x))
            finally:
                loop.close()

        # Run differential evolution
        bounds = [(p.min_value, p.max_value) for p in parameters]
        result = differential_evolution(
            sync_objective_func,
            bounds,
            maxiter=self.config.max_iterations,
            popsize=self.config.population_size // len(parameters),
            seed=self.config.random_seed,
            atol=self.config.tolerance,
        )

        if result.success:
            optimal_params = dict(zip([p.name for p in parameters], result.x))
            properties = await self.property_predictor.predict_properties(
                optimal_params
            )

            objective_values = {}
            for obj_target in objectives:
                value = self.current_problem._evaluate_objective(obj_target, properties)
                objective_values[obj_target.objective] = value

            sustainability_scores = {
                "degradation_resistance": 1.0
                / (properties.get("degradation_rate", 0.001) + 1e-6),
                "recyclability": properties.get("recyclability_score", 0.5),
                "environmental_impact": 1.0
                / (properties.get("carbon_footprint", 2.0) + 1e-6),
                "overall_sustainability": self._calculate_overall_sustainability(
                    properties
                ),
            }

            return DesignResult(
                optimal_parameters=optimal_params,
                objective_values=objective_values,
                constraint_violations={},
                optimization_history=[],
                convergence_metrics={
                    "function_evaluations": result.nfev,
                    "success": result.success,
                },
                sustainability_scores=sustainability_scores,
                predicted_properties=properties,
                computation_time=time.time() - self.start_time,
                success=True,
                message=result.message,
                metadata={"algorithm": "Differential Evolution"},
            )
        else:
            raise RuntimeError(f"Genetic algorithm failed: {result.message}")

    async def _run_default_optimization(
        self,
        parameters: List[MaterialParameter],
        objectives: List[SustainabilityTarget],
        constraints: List[OptimizationConstraint],
    ) -> DesignResult:
        """Run default optimization using random search."""

        best_params = None
        best_objective = float("inf")
        best_properties = None

        # Random search optimization
        for iteration in range(self.config.max_iterations):
            # Generate random parameters
            params = {}
            for param in parameters:
                if param.parameter_type == "continuous":
                    params[param.name] = np.random.uniform(
                        param.min_value, param.max_value
                    )
                elif param.parameter_type == "integer":
                    params[param.name] = np.random.randint(
                        int(param.min_value), int(param.max_value) + 1
                    )
                elif param.parameter_type == "categorical":
                    params[param.name] = np.random.choice(
                        param.possible_values or [param.min_value]
                    )

            # Evaluate
            try:
                properties = await self.property_predictor.predict_properties(params)

                # Calculate total objective
                total_obj = 0.0
                for obj_target in objectives:
                    value = self.current_problem._evaluate_objective(
                        obj_target, properties
                    )
                    total_obj += value

                # Check constraints
                feasible = True
                for constraint in constraints:
                    violation = self.current_problem._evaluate_constraint(
                        constraint, properties, params
                    )
                    if violation > constraint.tolerance:
                        feasible = False
                        total_obj += violation * 1000  # Penalty

                # Update best solution
                if total_obj < best_objective:
                    best_objective = total_obj
                    best_params = params.copy()
                    best_properties = properties.copy()

            except Exception as e:
                logger.warning(f"Evaluation failed at iteration {iteration}: {e}")
                continue

        if best_params is None:
            raise RuntimeError("No feasible solutions found")

        # Calculate final results
        objective_values = {}
        for obj_target in objectives:
            value = self.current_problem._evaluate_objective(
                obj_target, best_properties
            )
            objective_values[obj_target.objective] = value

        sustainability_scores = {
            "degradation_resistance": 1.0
            / (best_properties.get("degradation_rate", 0.001) + 1e-6),
            "recyclability": best_properties.get("recyclability_score", 0.5),
            "environmental_impact": 1.0
            / (best_properties.get("carbon_footprint", 2.0) + 1e-6),
            "overall_sustainability": self._calculate_overall_sustainability(
                best_properties
            ),
        }

        return DesignResult(
            optimal_parameters=best_params,
            objective_values=objective_values,
            constraint_violations={},
            optimization_history=[],
            convergence_metrics={"iterations": self.config.max_iterations},
            sustainability_scores=sustainability_scores,
            predicted_properties=best_properties,
            computation_time=time.time() - self.start_time,
            success=True,
            message="Random search optimization completed",
            metadata={"algorithm": "Random Search"},
        )

    def _calculate_overall_sustainability(self, properties: Dict[str, Any]) -> float:
        """Calculate overall sustainability score."""
        scores = []

        # Degradation resistance (higher is better)
        if "degradation_rate" in properties:
            scores.append(1.0 / (properties["degradation_rate"] + 1e-6))

        # Recyclability (higher is better)
        if "recyclability_score" in properties:
            scores.append(properties["recyclability_score"])

        # Environmental impact (lower is better)
        if "carbon_footprint" in properties:
            scores.append(1.0 / (properties["carbon_footprint"] + 1e-6))

        # Circularity (higher is better)
        if "circularity_index" in properties:
            scores.append(properties["circularity_index"])

        return np.mean(scores) if scores else 0.5

    async def multi_objective_design(
        self,
        parameters: List[MaterialParameter],
        objectives: List[SustainabilityTarget],
        constraints: Optional[List[OptimizationConstraint]] = None,
        n_solutions: int = 10,
    ) -> List[DesignResult]:
        """Generate multiple Pareto-optimal solutions."""

        results = []

        # Run multiple optimizations with different weights
        for i in range(n_solutions):
            # Randomize objective weights
            modified_objectives = []
            for obj in objectives:
                new_weight = np.random.uniform(0.1, 2.0)
                modified_obj = SustainabilityTarget(
                    objective=obj.objective,
                    target_value=obj.target_value,
                    weight=new_weight,
                    minimize=obj.minimize,
                    constraint=obj.constraint,
                    tolerance=obj.tolerance,
                )
                modified_objectives.append(modified_obj)

            try:
                result = await self.design_material(
                    parameters, modified_objectives, constraints
                )
                if result.success:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Multi-objective iteration {i} failed: {e}")
                continue

        return results

    def analyze_trade_offs(self, results: List[DesignResult]) -> Dict[str, Any]:
        """Analyze trade-offs between different objectives."""

        if not results:
            return {"error": "No results to analyze"}

        analysis = {
            "pareto_front": [],
            "objective_correlations": {},
            "sensitivity_analysis": {},
            "recommendations": [],
        }

        # Extract objective values
        objectives_data = {}
        for result in results:
            for obj, value in result.objective_values.items():
                if obj not in objectives_data:
                    objectives_data[obj] = []
                objectives_data[obj].append(value)

        # Calculate correlations
        for obj1 in objectives_data:
            for obj2 in objectives_data:
                if obj1 != obj2:
                    corr = np.corrcoef(objectives_data[obj1], objectives_data[obj2])[
                        0, 1
                    ]
                    analysis["objective_correlations"][
                        f"{obj1.value}_vs_{obj2.value}"
                    ] = corr

        # Identify Pareto front
        for result in results:
            is_dominated = False
            for other_result in results:
                if self._dominates(
                    other_result.objective_values, result.objective_values
                ):
                    is_dominated = True
                    break

            if not is_dominated:
                analysis["pareto_front"].append(result.optimal_parameters)

        # Generate recommendations
        if analysis["pareto_front"]:
            analysis["recommendations"].append(
                f"Found {len(analysis['pareto_front'])} Pareto-optimal solutions"
            )

        return analysis

    def _dominates(self, solution1: Dict, solution2: Dict) -> bool:
        """Check if solution1 dominates solution2 (assuming minimization)."""
        better_in_any = False
        for obj in solution1:
            if obj in solution2:
                if solution1[obj] > solution2[obj]:  # Worse in this objective
                    return False
                elif solution1[obj] < solution2[obj]:  # Better in this objective
                    better_in_any = True

        return better_in_any


# Utility functions for common design scenarios


def create_polymer_design_parameters() -> List[MaterialParameter]:
    """Create standard polymer design parameters."""
    return [
        MaterialParameter(
            name="monomer_ratio",
            min_value=0.0,
            max_value=1.0,
            description="Ratio of primary monomer",
        ),
        MaterialParameter(
            name="crosslink_density",
            min_value=0.0,
            max_value=0.3,
            description="Crosslink density",
        ),
        MaterialParameter(
            name="stabilizer_content",
            min_value=0.0,
            max_value=0.1,
            description="Stabilizer content fraction",
        ),
        MaterialParameter(
            name="filler_content",
            min_value=0.0,
            max_value=0.5,
            description="Filler content fraction",
        ),
        MaterialParameter(
            name="polymer_purity",
            min_value=0.8,
            max_value=0.99,
            description="Polymer purity fraction",
        ),
    ]


def create_sustainability_objectives() -> List[SustainabilityTarget]:
    """Create standard sustainability objectives."""
    return [
        SustainabilityTarget(
            objective=OptimizationObjective.MINIMIZE_DEGRADATION_RATE,
            weight=1.0,
            minimize=True,
        ),
        SustainabilityTarget(
            objective=OptimizationObjective.MAXIMIZE_RECYCLABILITY,
            weight=1.0,
            minimize=False,
        ),
        SustainabilityTarget(
            objective=OptimizationObjective.MINIMIZE_CARBON_FOOTPRINT,
            weight=1.0,
            minimize=True,
        ),
    ]


def create_performance_constraints() -> List[OptimizationConstraint]:
    """Create standard performance constraints."""
    return [
        OptimizationConstraint(
            name="minimum_tensile_strength",
            constraint_type=ConstraintType.INEQUALITY,
            function=lambda props, params: 30.0 - props.get("tensile_strength", 0),
            description="Minimum tensile strength of 30 MPa",
        ),
        OptimizationConstraint(
            name="minimum_degradation_temp",
            constraint_type=ConstraintType.INEQUALITY,
            function=lambda props, params: 200.0 - props.get("degradation_temp", 0),
            description="Minimum degradation temperature of 200°C",
        ),
    ]


async def design_sustainable_material(
    material_type: str = "polymer",
    optimization_objectives: Optional[List[OptimizationObjective]] = None,
    constraints: Optional[List[str]] = None,
    use_gpu: bool = True,
    max_iterations: int = 1000,
) -> DesignResult:
    """Utility function for sustainable material design."""

    # Default objectives
    if optimization_objectives is None:
        optimization_objectives = [
            OptimizationObjective.MINIMIZE_DEGRADATION_RATE,
            OptimizationObjective.MAXIMIZE_RECYCLABILITY,
            OptimizationObjective.MINIMIZE_CARBON_FOOTPRINT,
        ]

    # Create design parameters
    parameters = create_polymer_design_parameters()

    # Create objectives
    objectives = []
    for obj in optimization_objectives:
        objectives.append(
            SustainabilityTarget(
                objective=obj, weight=1.0, minimize=obj.value.startswith("minimize")
            )
        )

    # Create constraints
    design_constraints = []
    if constraints:
        design_constraints = create_performance_constraints()

    # Configure optimization
    config = OptimizationConfig(
        algorithm=OptimizationAlgorithm.MULTI_OBJECTIVE_GENETIC,
        max_iterations=max_iterations,
        use_gpu=use_gpu,
    )

    # Run design
    designer = SustainableInverseDesigner(config)
    return await designer.design_material(parameters, objectives, design_constraints)


# Exception classes
class InverseDesignError(Exception):
    """Base exception for inverse design errors."""

    pass


class OptimizationFailedError(InverseDesignError):
    """Raised when optimization fails to converge."""

    pass


class InvalidParameterError(InverseDesignError):
    """Raised when invalid parameters are provided."""

    pass


class PropertyPredictionError(InverseDesignError):
    """Raised when property prediction fails."""

    pass

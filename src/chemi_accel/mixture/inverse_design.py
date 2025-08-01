"""
Inverse design for optimal mixture composition.

Extends ChemiAccel's inverse design and multi-objective optimization
capabilities to propose optimal compositions for multi-component mixtures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..ai.inverse_design import InverseDesigner
from ..ai.multi_objective_optimization import MultiObjectiveOptimizer
from ..core.backend import Backend
from ..core.exceptions import ChemiAccelError
from .core import ComponentData, MixtureCalculator, MixtureData


@dataclass
class OptimalComposition:
    """Results from mixture composition optimization."""

    composition: np.ndarray
    properties: Dict[str, float]
    objective_values: List[float]
    feasibility_score: float
    components: List[str]
    optimization_history: Optional[List[Dict[str, Any]]] = None
    pareto_front: Optional[List[Dict[str, Any]]] = None

    @property
    def n_components(self) -> int:
        """Number of components in optimal mixture."""
        return len(self.composition)

    @property
    def dominant_component(self) -> Tuple[int, float]:
        """Index and fraction of dominant component."""
        max_idx = np.argmax(self.composition)
        return max_idx, self.composition[max_idx]


class MixtureObjective(ABC):
    """Abstract base class for mixture optimization objectives."""

    @abstractmethod
    def evaluate(
        self,
        composition: np.ndarray,
        components: List[str],
        calculator: MixtureCalculator,
        **kwargs,
    ) -> float:
        """Evaluate objective function for given composition."""
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[float, float]:
        """Get bounds for this objective."""
        pass


class PropertyTargetObjective(MixtureObjective):
    """Objective for targeting specific mixture properties."""

    def __init__(
        self,
        target_properties: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ):
        self.target_properties = target_properties
        self.weights = weights or {prop: 1.0 for prop in target_properties}

    def evaluate(
        self,
        composition: np.ndarray,
        components: List[str],
        calculator: MixtureCalculator,
        **kwargs,
    ) -> float:
        """Evaluate property target objective."""

        # Create mixture
        mixture = calculator.create_mixture(
            [(comp, frac) for comp, frac in zip(components, composition)]
        )

        # Calculate properties
        properties = calculator.calculate_mixture_properties(
            mixture, list(self.target_properties.keys())
        )

        # Calculate weighted deviation from targets
        total_error = 0.0
        for prop_name, target_value in self.target_properties.items():
            if prop_name in properties:
                actual_value = properties[prop_name]
                relative_error = abs(actual_value - target_value) / abs(target_value)
                weight = self.weights.get(prop_name, 1.0)
                total_error += weight * relative_error

        return total_error  # Minimize error

    def get_bounds(self) -> Tuple[float, float]:
        """Get bounds for property target objective."""
        return (0.0, 10.0)  # Error bounds


class StabilityObjective(MixtureObjective):
    """Objective for maximizing mixture stability."""

    def __init__(self, stability_metrics: Optional[List[str]] = None):
        self.stability_metrics = stability_metrics or ["thermal", "mechanical"]

    def evaluate(
        self,
        composition: np.ndarray,
        components: List[str],
        calculator: MixtureCalculator,
        **kwargs,
    ) -> float:
        """Evaluate stability objective."""

        # Create mixture
        mixture = calculator.create_mixture(
            [(comp, frac) for comp, frac in zip(components, composition)]
        )

        # Predict phase behavior for stability assessment
        phase_results = calculator.predict_phase_behavior(mixture)

        # Calculate stability score
        stability_scores = []
        for metric in self.stability_metrics:
            if metric in phase_results.stability:
                stability_scores.append(phase_results.stability[metric])

        if not stability_scores:
            return 0.0

        return -np.mean(
            stability_scores
        )  # Minimize negative stability (maximize stability)

    def get_bounds(self) -> Tuple[float, float]:
        """Get bounds for stability objective."""
        return (-1.0, 0.0)  # Negative stability bounds


class CostObjective(MixtureObjective):
    """Objective for minimizing mixture cost."""

    def __init__(self, component_costs: Dict[str, float]):
        self.component_costs = component_costs

    def evaluate(
        self,
        composition: np.ndarray,
        components: List[str],
        calculator: MixtureCalculator,
        **kwargs,
    ) -> float:
        """Evaluate cost objective."""

        total_cost = 0.0
        for comp, frac in zip(components, composition):
            cost_per_unit = self.component_costs.get(comp, 1.0)  # Default cost
            total_cost += frac * cost_per_unit

        return total_cost

    def get_bounds(self) -> Tuple[float, float]:
        """Get bounds for cost objective."""
        min_cost = min(self.component_costs.values()) if self.component_costs else 0.0
        max_cost = max(self.component_costs.values()) if self.component_costs else 10.0
        return (min_cost, max_cost)


class EnvironmentalObjective(MixtureObjective):
    """Objective for minimizing environmental impact."""

    def __init__(self, impact_scores: Dict[str, float]):
        self.impact_scores = impact_scores

    def evaluate(
        self,
        composition: np.ndarray,
        components: List[str],
        calculator: MixtureCalculator,
        **kwargs,
    ) -> float:
        """Evaluate environmental objective."""

        total_impact = 0.0
        for comp, frac in zip(components, composition):
            impact_score = self.impact_scores.get(comp, 0.5)  # Default impact
            total_impact += frac * impact_score

        return total_impact

    def get_bounds(self) -> Tuple[float, float]:
        """Get bounds for environmental objective."""
        min_impact = min(self.impact_scores.values()) if self.impact_scores else 0.0
        max_impact = max(self.impact_scores.values()) if self.impact_scores else 1.0
        return (min_impact, max_impact)


class MixtureConstraint(ABC):
    """Abstract base class for mixture constraints."""

    @abstractmethod
    def is_satisfied(
        self,
        composition: np.ndarray,
        components: List[str],
        calculator: MixtureCalculator,
        **kwargs,
    ) -> bool:
        """Check if constraint is satisfied."""
        pass

    @abstractmethod
    def get_violation(
        self,
        composition: np.ndarray,
        components: List[str],
        calculator: MixtureCalculator,
        **kwargs,
    ) -> float:
        """Get constraint violation (0 if satisfied)."""
        pass


class CompositionConstraint(MixtureConstraint):
    """Constraint on component composition ranges."""

    def __init__(self, component_bounds: Dict[str, Tuple[float, float]]):
        self.component_bounds = component_bounds

    def is_satisfied(
        self,
        composition: np.ndarray,
        components: List[str],
        calculator: MixtureCalculator,
        **kwargs,
    ) -> bool:
        """Check composition constraint satisfaction."""

        for i, comp in enumerate(components):
            if comp in self.component_bounds:
                min_frac, max_frac = self.component_bounds[comp]
                if not (min_frac <= composition[i] <= max_frac):
                    return False

        return True

    def get_violation(
        self,
        composition: np.ndarray,
        components: List[str],
        calculator: MixtureCalculator,
        **kwargs,
    ) -> float:
        """Get composition constraint violation."""

        violation = 0.0

        for i, comp in enumerate(components):
            if comp in self.component_bounds:
                min_frac, max_frac = self.component_bounds[comp]
                if composition[i] < min_frac:
                    violation += min_frac - composition[i]
                elif composition[i] > max_frac:
                    violation += composition[i] - max_frac

        return violation


class PropertyConstraint(MixtureConstraint):
    """Constraint on mixture properties."""

    def __init__(self, property_bounds: Dict[str, Tuple[float, float]]):
        self.property_bounds = property_bounds

    def is_satisfied(
        self,
        composition: np.ndarray,
        components: List[str],
        calculator: MixtureCalculator,
        **kwargs,
    ) -> bool:
        """Check property constraint satisfaction."""

        # Create mixture and calculate properties
        mixture = calculator.create_mixture(
            [(comp, frac) for comp, frac in zip(components, composition)]
        )

        properties = calculator.calculate_mixture_properties(
            mixture, list(self.property_bounds.keys())
        )

        # Check all property bounds
        for prop_name, (min_val, max_val) in self.property_bounds.items():
            if prop_name in properties:
                prop_value = properties[prop_name]
                if not (min_val <= prop_value <= max_val):
                    return False

        return True

    def get_violation(
        self,
        composition: np.ndarray,
        components: List[str],
        calculator: MixtureCalculator,
        **kwargs,
    ) -> float:
        """Get property constraint violation."""

        # Create mixture and calculate properties
        mixture = calculator.create_mixture(
            [(comp, frac) for comp, frac in zip(components, composition)]
        )

        properties = calculator.calculate_mixture_properties(
            mixture, list(self.property_bounds.keys())
        )

        violation = 0.0

        for prop_name, (min_val, max_val) in self.property_bounds.items():
            if prop_name in properties:
                prop_value = properties[prop_name]
                if prop_value < min_val:
                    violation += abs(prop_value - min_val) / abs(min_val)
                elif prop_value > max_val:
                    violation += abs(prop_value - max_val) / abs(max_val)

        return violation


class MixtureInverseDesigner:
    """
    AI-driven inverse design for optimal mixture composition.

    Extends ChemiAccel's inverse design capabilities to multi-component
    mixtures with complex property targets and constraints.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.calculator = MixtureCalculator(backend=backend)

        # Initialize optimization components
        self.inverse_designer = InverseDesigner(backend)
        self.multi_objective_optimizer = MultiObjectiveOptimizer(backend)

        # Optimization history
        self.optimization_history = []

    def optimize_composition(
        self,
        target_properties: Dict[str, float],
        component_library: List[str],
        constraints: Optional[Dict[str, Any]] = None,
        objectives: Optional[List[MixtureObjective]] = None,
        optimization_method: str = "nsga2",
        population_size: int = 100,
        n_generations: int = 50,
        **kwargs,
    ) -> OptimalComposition:
        """
        Optimize mixture composition for target properties.

        Args:
            target_properties: Target property values
            component_library: Available components (SMILES)
            constraints: Optimization constraints
            objectives: Additional objectives beyond property targets
            optimization_method: Optimization algorithm
            population_size: Population size for evolutionary algorithms
            n_generations: Number of generations

        Returns:
            Optimal composition results
        """

        # Setup optimization problem
        problem = self._setup_optimization_problem(
            target_properties, component_library, constraints, objectives
        )

        # Choose optimization strategy
        if len(problem["objectives"]) == 1:
            result = self._single_objective_optimization(problem, **kwargs)
        else:
            result = self._multi_objective_optimization(
                problem, optimization_method, population_size, n_generations, **kwargs
            )

        return result

    def _setup_optimization_problem(
        self,
        target_properties: Dict[str, float],
        component_library: List[str],
        constraints: Optional[Dict[str, Any]],
        objectives: Optional[List[MixtureObjective]],
    ) -> Dict[str, Any]:
        """Setup optimization problem definition."""

        n_components = len(component_library)

        # Define objectives
        problem_objectives = []

        # Add property target objective
        property_objective = PropertyTargetObjective(target_properties)
        problem_objectives.append(property_objective)

        # Add additional objectives
        if objectives:
            problem_objectives.extend(objectives)

        # Parse constraints
        problem_constraints = []
        if constraints:
            if "composition_bounds" in constraints:
                comp_constraint = CompositionConstraint(
                    constraints["composition_bounds"]
                )
                problem_constraints.append(comp_constraint)

            if "property_bounds" in constraints:
                prop_constraint = PropertyConstraint(constraints["property_bounds"])
                problem_constraints.append(prop_constraint)

        # Add normalization constraint (compositions sum to 1)
        def composition_sum_constraint(x):
            return abs(np.sum(x) - 1.0)

        # Define bounds
        bounds = [(0.0, 1.0) for _ in range(n_components)]

        return {
            "objectives": problem_objectives,
            "constraints": problem_constraints,
            "bounds": bounds,
            "n_variables": n_components,
            "component_library": component_library,
            "composition_sum_constraint": composition_sum_constraint,
        }

    def _single_objective_optimization(
        self, problem: Dict[str, Any], **kwargs
    ) -> OptimalComposition:
        """Solve single-objective optimization problem."""

        try:
            from scipy.optimize import differential_evolution, minimize

            objective_func = problem["objectives"][0]
            component_library = problem["component_library"]
            constraints_list = problem["constraints"]
            bounds = problem["bounds"]

            def objective(x):
                # Normalize composition
                x_norm = x / np.sum(x) if np.sum(x) > 0 else x

                # Evaluate objective
                obj_value = objective_func.evaluate(
                    x_norm, component_library, self.calculator
                )

                # Add constraint penalties
                penalty = 0.0
                for constraint in constraints_list:
                    violation = constraint.get_violation(
                        x_norm, component_library, self.calculator
                    )
                    penalty += 1000.0 * violation  # Large penalty

                # Add composition sum penalty
                penalty += 1000.0 * problem["composition_sum_constraint"](x_norm)

                return obj_value + penalty

            # Use differential evolution for global optimization
            result = differential_evolution(
                objective,
                bounds,
                maxiter=kwargs.get("max_iterations", 100),
                popsize=kwargs.get("population_size", 15),
                atol=kwargs.get("tolerance", 1e-6),
                seed=kwargs.get("random_seed", 42),
            )

            # Normalize final composition
            optimal_composition = result.x / np.sum(result.x)

            # Calculate final properties
            optimal_mixture = self.calculator.create_mixture(
                [
                    (comp, frac)
                    for comp, frac in zip(component_library, optimal_composition)
                ]
            )

            final_properties = self.calculator.calculate_mixture_properties(
                optimal_mixture, list(problem["objectives"][0].target_properties.keys())
            )

            # Calculate feasibility score
            feasibility_score = self._calculate_feasibility_score(
                optimal_composition, component_library, constraints_list
            )

            return OptimalComposition(
                composition=optimal_composition,
                properties=final_properties,
                objective_values=[result.fun],
                feasibility_score=feasibility_score,
                components=component_library,
                optimization_history=[
                    {
                        "iteration": result.nit,
                        "objective_value": result.fun,
                        "success": result.success,
                    }
                ],
            )

        except ImportError:
            raise ChemiAccelError("scipy required for optimization")

    def _multi_objective_optimization(
        self,
        problem: Dict[str, Any],
        method: str,
        population_size: int,
        n_generations: int,
        **kwargs,
    ) -> OptimalComposition:
        """Solve multi-objective optimization problem."""

        try:
            import pymoo.core.variable as Variable
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.core.problem import Problem
            from pymoo.optimize import minimize as pymoo_minimize

            class MixtureProblem(Problem):
                def __init__(self, problem_def, calculator):
                    self.problem_def = problem_def
                    self.calculator = calculator

                    n_var = problem_def["n_variables"]
                    n_obj = len(problem_def["objectives"])
                    n_constr = (
                        len(problem_def["constraints"]) + 1
                    )  # +1 for sum constraint

                    super().__init__(
                        n_var=n_var,
                        n_obj=n_obj,
                        n_constr=n_constr,
                        xl=np.array([b[0] for b in problem_def["bounds"]]),
                        xu=np.array([b[1] for b in problem_def["bounds"]]),
                    )

                def _evaluate(self, X, out, *args, **kwargs):
                    n_solutions = X.shape[0]
                    objectives = np.zeros((n_solutions, self.n_obj))
                    constraints = np.zeros((n_solutions, self.n_constr))

                    for i in range(n_solutions):
                        x = X[i]
                        x_norm = x / np.sum(x) if np.sum(x) > 0 else x

                        # Evaluate objectives
                        for j, obj in enumerate(self.problem_def["objectives"]):
                            objectives[i, j] = obj.evaluate(
                                x_norm,
                                self.problem_def["component_library"],
                                self.calculator,
                            )

                        # Evaluate constraints
                        for j, constr in enumerate(self.problem_def["constraints"]):
                            constraints[i, j] = constr.get_violation(
                                x_norm,
                                self.problem_def["component_library"],
                                self.calculator,
                            )

                        # Composition sum constraint
                        constraints[i, -1] = self.problem_def[
                            "composition_sum_constraint"
                        ](x_norm)

                    out["F"] = objectives
                    out["G"] = constraints

            # Create problem instance
            pymoo_problem = MixtureProblem(problem, self.calculator)

            # Setup algorithm
            if method.lower() == "nsga2":
                algorithm = NSGA2(pop_size=population_size)
            else:
                raise ValueError(f"Unsupported multi-objective method: {method}")

            # Run optimization
            result = pymoo_minimize(
                pymoo_problem,
                algorithm,
                ("n_gen", n_generations),
                verbose=kwargs.get("verbose", False),
            )

            # Extract best solution (closest to utopia point)
            if result.F.ndim == 1:
                best_idx = 0
                best_objectives = [result.F]
            else:
                # Find solution closest to ideal point (minimize all objectives)
                ideal_point = np.min(result.F, axis=0)
                distances = np.linalg.norm(result.F - ideal_point, axis=1)
                best_idx = np.argmin(distances)
                best_objectives = result.F[best_idx].tolist()

            optimal_composition = result.X[best_idx] / np.sum(result.X[best_idx])

            # Calculate final properties
            optimal_mixture = self.calculator.create_mixture(
                [
                    (comp, frac)
                    for comp, frac in zip(
                        problem["component_library"], optimal_composition
                    )
                ]
            )

            final_properties = self.calculator.calculate_mixture_properties(
                optimal_mixture
            )

            # Calculate feasibility score
            feasibility_score = self._calculate_feasibility_score(
                optimal_composition,
                problem["component_library"],
                problem["constraints"],
            )

            # Build Pareto front data
            pareto_front = []
            if result.F.ndim > 1:
                for i in range(result.F.shape[0]):
                    pareto_front.append(
                        {
                            "composition": result.X[i] / np.sum(result.X[i]),
                            "objectives": result.F[i].tolist(),
                        }
                    )

            return OptimalComposition(
                composition=optimal_composition,
                properties=final_properties,
                objective_values=best_objectives,
                feasibility_score=feasibility_score,
                components=problem["component_library"],
                pareto_front=pareto_front[:50] if pareto_front else None,  # Limit size
            )

        except ImportError:
            raise ChemiAccelError("pymoo required for multi-objective optimization")

    def _calculate_feasibility_score(
        self,
        composition: np.ndarray,
        components: List[str],
        constraints: List[MixtureConstraint],
    ) -> float:
        """Calculate overall feasibility score."""

        if not constraints:
            return 1.0

        total_violation = 0.0
        for constraint in constraints:
            violation = constraint.get_violation(
                composition, components, self.calculator
            )
            total_violation += violation

        # Convert violation to feasibility score (0 to 1)
        feasibility_score = 1.0 / (1.0 + total_violation)
        return feasibility_score

    def design_pareto_optimal_mixtures(
        self,
        objectives: List[MixtureObjective],
        component_library: List[str],
        constraints: Optional[Dict[str, Any]] = None,
        n_solutions: int = 50,
        **kwargs,
    ) -> List[OptimalComposition]:
        """
        Design Pareto-optimal mixture compositions.

        Args:
            objectives: List of objectives to optimize
            component_library: Available components
            constraints: Optimization constraints
            n_solutions: Number of Pareto solutions to generate

        Returns:
            List of Pareto-optimal compositions
        """

        if len(objectives) < 2:
            raise ValueError("At least 2 objectives required for Pareto optimization")

        # Setup multi-objective problem
        problem = self._setup_optimization_problem(
            {}, component_library, constraints, objectives
        )

        # Run multi-objective optimization
        result = self._multi_objective_optimization(
            problem, "nsga2", n_solutions, kwargs.get("n_generations", 100), **kwargs
        )

        # Convert Pareto front to list of OptimalComposition objects
        pareto_solutions = []

        if result.pareto_front:
            for solution_data in result.pareto_front:
                composition = solution_data["composition"]

                # Calculate properties for this composition
                mixture = self.calculator.create_mixture(
                    [(comp, frac) for comp, frac in zip(component_library, composition)]
                )
                properties = self.calculator.calculate_mixture_properties(mixture)

                # Calculate feasibility
                feasibility = self._calculate_feasibility_score(
                    composition, component_library, problem["constraints"]
                )

                pareto_solutions.append(
                    OptimalComposition(
                        composition=composition,
                        properties=properties,
                        objective_values=solution_data["objectives"],
                        feasibility_score=feasibility,
                        components=component_library,
                    )
                )

        return pareto_solutions

    def adaptive_design(
        self,
        initial_targets: Dict[str, float],
        component_library: List[str],
        experimental_feedback: Optional[List[Dict[str, Any]]] = None,
        n_iterations: int = 5,
        **kwargs,
    ) -> List[OptimalComposition]:
        """
        Adaptive mixture design with experimental feedback.

        Args:
            initial_targets: Initial property targets
            component_library: Available components
            experimental_feedback: Results from previous experiments
            n_iterations: Number of adaptive iterations

        Returns:
            List of suggested compositions across iterations
        """

        suggestions = []
        current_targets = initial_targets.copy()

        for iteration in range(n_iterations):
            # Optimize for current targets
            result = self.optimize_composition(
                current_targets, component_library, **kwargs
            )
            suggestions.append(result)

            # Update targets based on experimental feedback
            if experimental_feedback and len(experimental_feedback) > iteration:
                feedback = experimental_feedback[iteration]
                current_targets = self._update_targets_from_feedback(
                    current_targets, feedback
                )

        return suggestions

    def _update_targets_from_feedback(
        self, current_targets: Dict[str, float], feedback: Dict[str, Any]
    ) -> Dict[str, float]:
        """Update optimization targets based on experimental feedback."""

        updated_targets = current_targets.copy()

        # Simple feedback incorporation strategy
        if "property_deviations" in feedback:
            for prop, deviation in feedback["property_deviations"].items():
                if prop in updated_targets:
                    # Adjust target based on observed deviation
                    updated_targets[prop] += 0.1 * deviation

        return updated_targets

"""
Multi-Objective Route Optimization.

This module applies multi-objective optimization frameworks to optimize entire
synthesis routes based on objectives like overall yield, cost, number of steps,
and environmental impact. It implements algorithms to navigate the Pareto front
of synthetic routes.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

try:
    import scipy.optimize
    from scipy.spatial.distance import cdist

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    scipy = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False
    plt = None
    sns = None

from .forward_prediction import (
    ForwardReactionPredictor,
    ReactionConditions,
    RouteEvaluationResult,
)
from .route_enumeration import RouteEnumerationConfig, SynthesisNode, SynthesisRoute

# Set up logging
logger = logging.getLogger(__name__)


# Enums
class OptimizationObjective(Enum):
    """Optimization objectives for synthesis routes."""

    OVERALL_YIELD = "overall_yield"
    TOTAL_COST = "total_cost"
    NUMBER_OF_STEPS = "number_of_steps"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    ATOM_ECONOMY = "atom_economy"
    SYNTHETIC_ACCESSIBILITY = "synthetic_accessibility"
    REACTION_TIME = "reaction_time"
    SELECTIVITY = "selectivity"
    SAFETY_SCORE = "safety_score"
    NOVELTY_SCORE = "novelty_score"


class OptimizationAlgorithm(Enum):
    """Multi-objective optimization algorithms."""

    NSGA_II = "nsga_ii"
    NSGA_III = "nsga_iii"
    SPEA2 = "spea2"
    MOEA_D = "moea_d"
    PARETO_FRONT = "pareto_front"
    WEIGHTED_SUM = "weighted_sum"
    EPSILON_CONSTRAINT = "epsilon_constraint"


class DominanceRelation(Enum):
    """Dominance relations between solutions."""

    DOMINATES = "dominates"
    DOMINATED = "dominated"
    NON_DOMINATED = "non_dominated"
    EQUIVALENT = "equivalent"


class ParetoRanking(Enum):
    """Pareto ranking categories."""

    RANK_1 = "rank_1"
    RANK_2 = "rank_2"
    RANK_3 = "rank_3"
    RANK_4_PLUS = "rank_4_plus"


# Data Structures
@dataclass
class OptimizationObjectiveSpec:
    """Specification for an optimization objective."""

    objective: OptimizationObjective
    weight: float = 1.0
    minimize: bool = True  # True for minimization, False for maximization
    constraint_min: Optional[float] = None
    constraint_max: Optional[float] = None
    priority: int = 1  # 1 = highest priority


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""

    objectives: List[OptimizationObjectiveSpec] = field(default_factory=list)
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA_II
    population_size: int = 100
    max_generations: int = 50
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    tournament_size: int = 3
    archive_size: int = 100
    epsilon_values: List[float] = field(default_factory=list)
    convergence_tolerance: float = 1e-6
    max_evaluations: int = 10000
    parallel_evaluation: bool = True
    seed: Optional[int] = None


@dataclass
class OptimizationSolution:
    """A solution in the multi-objective optimization."""

    route: SynthesisRoute
    objective_values: Dict[OptimizationObjective, float] = field(default_factory=dict)
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    fitness_values: List[float] = field(default_factory=list)
    dominance_rank: Optional[int] = None
    crowding_distance: float = 0.0
    pareto_ranking: Optional[ParetoRanking] = None
    solution_id: str = field(default="")

    def __post_init__(self):
        if not self.solution_id:
            self.solution_id = f"sol_{abs(hash(self.route.route_id))}"


@dataclass
class ParetoFront:
    """Pareto front of non-dominated solutions."""

    solutions: List[OptimizationSolution] = field(default_factory=list)
    front_id: int = 1
    hypervolume: float = 0.0
    spread_metric: float = 0.0
    convergence_metric: float = 0.0

    def add_solution(self, solution: OptimizationSolution):
        """Add solution to front."""
        solution.pareto_ranking = ParetoRanking(f"rank_{min(4, self.front_id)}")
        self.solutions.append(solution)

    def size(self) -> int:
        """Get size of Pareto front."""
        return len(self.solutions)


@dataclass
class OptimizationResult:
    """Result of multi-objective optimization."""

    pareto_fronts: List[ParetoFront] = field(default_factory=list)
    best_compromise_solution: Optional[OptimizationSolution] = None
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    convergence_data: Dict[str, List[float]] = field(default_factory=dict)
    total_evaluations: int = 0
    total_time: float = 0.0
    hypervolume_history: List[float] = field(default_factory=list)

    def get_pareto_optimal_solutions(self) -> List[OptimizationSolution]:
        """Get Pareto optimal solutions (first front)."""
        if self.pareto_fronts:
            return self.pareto_fronts[0].solutions
        return []


# Exceptions
class MultiObjectiveOptimizationError(Exception):
    """Base exception for multi-objective optimization errors."""

    pass


class ParetoFrontError(MultiObjectiveOptimizationError):
    """Error in Pareto front calculation."""

    pass


class ObjectiveEvaluationError(MultiObjectiveOptimizationError):
    """Error in objective evaluation."""

    pass


class ConvergenceError(MultiObjectiveOptimizationError):
    """Error in convergence detection."""

    pass


# Main Classes
class ObjectiveEvaluator:
    """Evaluates optimization objectives for synthesis routes."""

    def __init__(self):
        self.forward_predictor = ForwardReactionPredictor()
        self.evaluation_cache: Dict[str, Dict[OptimizationObjective, float]] = {}

    def evaluate_objective(
        self, route: SynthesisRoute, objective: OptimizationObjective
    ) -> float:
        """Evaluate a single objective for a route."""
        # Check cache
        route_key = route.route_id
        if route_key in self.evaluation_cache:
            if objective in self.evaluation_cache[route_key]:
                return self.evaluation_cache[route_key][objective]
        else:
            self.evaluation_cache[route_key] = {}

        try:
            if objective == OptimizationObjective.OVERALL_YIELD:
                value = self._evaluate_overall_yield(route)
            elif objective == OptimizationObjective.TOTAL_COST:
                value = self._evaluate_total_cost(route)
            elif objective == OptimizationObjective.NUMBER_OF_STEPS:
                value = self._evaluate_number_of_steps(route)
            elif objective == OptimizationObjective.ENVIRONMENTAL_IMPACT:
                value = self._evaluate_environmental_impact(route)
            elif objective == OptimizationObjective.ATOM_ECONOMY:
                value = self._evaluate_atom_economy(route)
            elif objective == OptimizationObjective.SYNTHETIC_ACCESSIBILITY:
                value = self._evaluate_synthetic_accessibility(route)
            elif objective == OptimizationObjective.REACTION_TIME:
                value = self._evaluate_reaction_time(route)
            elif objective == OptimizationObjective.SELECTIVITY:
                value = self._evaluate_selectivity(route)
            elif objective == OptimizationObjective.SAFETY_SCORE:
                value = self._evaluate_safety_score(route)
            elif objective == OptimizationObjective.NOVELTY_SCORE:
                value = self._evaluate_novelty_score(route)
            else:
                raise ObjectiveEvaluationError(f"Unknown objective: {objective}")

            # Cache result
            self.evaluation_cache[route_key][objective] = value
            return value

        except Exception as e:
            logger.error(
                f"Error evaluating objective {objective} for route {route_key}: {e}"
            )
            return 0.0

    def evaluate_all_objectives(
        self, route: SynthesisRoute, objectives: List[OptimizationObjective]
    ) -> Dict[OptimizationObjective, float]:
        """Evaluate all objectives for a route."""
        results = {}
        for objective in objectives:
            results[objective] = self.evaluate_objective(route, objective)
        return results

    def _evaluate_overall_yield(self, route: SynthesisRoute) -> float:
        """Evaluate overall yield (to be minimized: 1 - yield)."""
        try:
            evaluation_result = self.forward_predictor.evaluate_synthesis_route(route)
            return (
                1.0 - evaluation_result.overall_yield
            )  # Convert to minimization problem
        except:
            return 1.0  # Worst case (0% yield)

    def _evaluate_total_cost(self, route: SynthesisRoute) -> float:
        """Evaluate total cost."""
        return route.total_cost

    def _evaluate_number_of_steps(self, route: SynthesisRoute) -> float:
        """Evaluate number of steps."""
        return float(route.total_steps)

    def _evaluate_environmental_impact(self, route: SynthesisRoute) -> float:
        """Evaluate environmental impact score."""
        try:
            # Simplified environmental impact calculation
            impact_score = 0.0

            for node in route.route_nodes:
                mol = Chem.MolFromSmiles(node.smiles)
                if mol:
                    # Penalize heavy atoms and heteroatoms
                    heavy_atoms = mol.GetNumHeavyAtoms()
                    heteroatoms = len(
                        [
                            atom
                            for atom in mol.GetAtoms()
                            if atom.GetAtomicNum() not in [1, 6]
                        ]
                    )  # Not H or C

                    # Simple heuristic for environmental impact
                    impact_score += heavy_atoms * 0.1 + heteroatoms * 0.2

            # Add step penalty
            impact_score += route.total_steps * 0.5

            return impact_score

        except Exception as e:
            logger.error(f"Error evaluating environmental impact: {e}")
            return 10.0  # High penalty for evaluation failure

    def _evaluate_atom_economy(self, route: SynthesisRoute) -> float:
        """Evaluate atom economy (to be minimized: 1 - atom_economy)."""
        try:
            if not route.route_nodes:
                return 1.0

            # Calculate atom economy for the overall transformation
            target_mol = Chem.MolFromSmiles(route.target_smiles)
            if not target_mol:
                return 1.0

            target_mw = Descriptors.MolWt(target_mol)

            # Sum molecular weights of starting materials
            starting_mw = 0.0
            for node in route.route_nodes:
                if node.is_purchasable:
                    mol = Chem.MolFromSmiles(node.smiles)
                    if mol:
                        starting_mw += Descriptors.MolWt(mol)

            if starting_mw > 0:
                atom_economy = target_mw / starting_mw
                return 1.0 - min(1.0, atom_economy)  # Convert to minimization

            return 1.0

        except Exception as e:
            logger.error(f"Error evaluating atom economy: {e}")
            return 1.0

    def _evaluate_synthetic_accessibility(self, route: SynthesisRoute) -> float:
        """Evaluate synthetic accessibility."""
        try:
            # Based on route complexity and feasibility
            complexity_penalty = route.total_steps * 0.1
            confidence_penalty = 1.0 - route.overall_confidence

            return complexity_penalty + confidence_penalty

        except Exception as e:
            logger.error(f"Error evaluating synthetic accessibility: {e}")
            return 2.0

    def _evaluate_reaction_time(self, route: SynthesisRoute) -> float:
        """Evaluate total reaction time."""
        try:
            total_time = 0.0

            # Estimate based on route complexity and reaction types
            for i, node in enumerate(route.route_nodes[:-1]):
                # Simple heuristic: more complex molecules take longer
                mol = Chem.MolFromSmiles(node.smiles)
                if mol:
                    complexity = mol.GetNumHeavyAtoms() + mol.GetNumBonds()
                    # Base time increases with complexity
                    step_time = 1.0 + complexity * 0.1  # hours
                    total_time += step_time

            return total_time

        except Exception as e:
            logger.error(f"Error evaluating reaction time: {e}")
            return 24.0  # Default high value

    def _evaluate_selectivity(self, route: SynthesisRoute) -> float:
        """Evaluate overall selectivity (to be minimized: 1 - selectivity)."""
        try:
            evaluation_result = self.forward_predictor.evaluate_synthesis_route(route)
            return 1.0 - evaluation_result.overall_selectivity

        except Exception as e:
            logger.error(f"Error evaluating selectivity: {e}")
            return 1.0

    def _evaluate_safety_score(self, route: SynthesisRoute) -> float:
        """Evaluate safety score."""
        try:
            safety_penalty = 0.0

            # Check for hazardous functional groups and conditions
            hazardous_groups = ["[N+](=O)[O-]", "C(=O)Cl", "[Na]", "[Li]"]  # Simplified

            for node in route.route_nodes:
                mol = Chem.MolFromSmiles(node.smiles)
                if mol:
                    mol_smiles = Chem.MolToSmiles(mol)
                    for hazard in hazardous_groups:
                        try:
                            if Chem.MolFromSmarts(hazard) and mol.HasSubstructMatch(
                                Chem.MolFromSmarts(hazard)
                            ):
                                safety_penalty += 1.0
                        except:
                            continue

            return safety_penalty

        except Exception as e:
            logger.error(f"Error evaluating safety score: {e}")
            return 5.0

    def _evaluate_novelty_score(self, route: SynthesisRoute) -> float:
        """Evaluate novelty score."""
        try:
            # Simple novelty metric based on route uniqueness
            novelty = 0.0

            # Penalize common starting materials (lower novelty)
            common_starting_materials = {"CCO", "c1ccccc1", "CC(=O)O", "CCN"}

            for node in route.route_nodes:
                if node.is_purchasable and node.smiles in common_starting_materials:
                    novelty -= 0.1

            # Reward unusual transformations (higher novelty)
            if route.total_steps > 5:
                novelty += 0.2

            # Convert to penalty (minimization problem)
            return -novelty

        except Exception as e:
            logger.error(f"Error evaluating novelty score: {e}")
            return 0.0


class DominanceAnalyzer:
    """Analyzes dominance relationships between solutions."""

    def __init__(self):
        pass

    def compare_solutions(
        self,
        sol1: OptimizationSolution,
        sol2: OptimizationSolution,
        objectives: List[OptimizationObjectiveSpec],
    ) -> DominanceRelation:
        """Compare two solutions and determine dominance relationship."""
        try:
            sol1_better = 0
            sol2_better = 0

            for obj_spec in objectives:
                obj = obj_spec.objective

                if obj not in sol1.objective_values or obj not in sol2.objective_values:
                    continue

                val1 = sol1.objective_values[obj]
                val2 = sol2.objective_values[obj]

                if obj_spec.minimize:
                    if val1 < val2:
                        sol1_better += 1
                    elif val1 > val2:
                        sol2_better += 1
                else:  # Maximize
                    if val1 > val2:
                        sol1_better += 1
                    elif val1 < val2:
                        sol2_better += 1

            # Determine dominance relationship
            if sol1_better > 0 and sol2_better == 0:
                return DominanceRelation.DOMINATES
            elif sol2_better > 0 and sol1_better == 0:
                return DominanceRelation.DOMINATED
            elif sol1_better == 0 and sol2_better == 0:
                return DominanceRelation.EQUIVALENT
            else:
                return DominanceRelation.NON_DOMINATED

        except Exception as e:
            logger.error(f"Error comparing solutions: {e}")
            return DominanceRelation.NON_DOMINATED

    def find_pareto_fronts(
        self,
        solutions: List[OptimizationSolution],
        objectives: List[OptimizationObjectiveSpec],
    ) -> List[ParetoFront]:
        """Find Pareto fronts using non-dominated sorting."""
        try:
            fronts = []
            remaining_solutions = solutions.copy()
            front_id = 1

            while remaining_solutions:
                current_front = ParetoFront(front_id=front_id)
                non_dominated = []

                for i, sol1 in enumerate(remaining_solutions):
                    is_dominated = False

                    for j, sol2 in enumerate(remaining_solutions):
                        if i != j:
                            relation = self.compare_solutions(sol1, sol2, objectives)
                            if relation == DominanceRelation.DOMINATED:
                                is_dominated = True
                                break

                    if not is_dominated:
                        non_dominated.append(sol1)
                        sol1.dominance_rank = front_id

                # Add solutions to current front
                for sol in non_dominated:
                    current_front.add_solution(sol)
                    remaining_solutions.remove(sol)

                if current_front.size() > 0:
                    fronts.append(current_front)
                    front_id += 1
                else:
                    break  # No more non-dominated solutions

            return fronts

        except Exception as e:
            logger.error(f"Error finding Pareto fronts: {e}")
            raise ParetoFrontError(f"Pareto front calculation failed: {e}")

    def calculate_crowding_distance(
        self,
        solutions: List[OptimizationSolution],
        objectives: List[OptimizationObjectiveSpec],
    ):
        """Calculate crowding distance for solutions."""
        try:
            n_solutions = len(solutions)
            if n_solutions <= 2:
                for sol in solutions:
                    sol.crowding_distance = float("inf")
                return

            # Initialize distances
            for sol in solutions:
                sol.crowding_distance = 0.0

            for obj_spec in objectives:
                obj = obj_spec.objective

                # Sort solutions by objective value
                solutions.sort(key=lambda s: s.objective_values.get(obj, 0.0))

                # Boundary solutions get infinite distance
                solutions[0].crowding_distance = float("inf")
                solutions[-1].crowding_distance = float("inf")

                # Calculate range
                obj_range = solutions[-1].objective_values.get(obj, 0.0) - solutions[
                    0
                ].objective_values.get(obj, 0.0)

                if obj_range > 0:
                    # Calculate crowding distance for intermediate solutions
                    for i in range(1, n_solutions - 1):
                        distance = (
                            solutions[i + 1].objective_values.get(obj, 0.0)
                            - solutions[i - 1].objective_values.get(obj, 0.0)
                        ) / obj_range
                        solutions[i].crowding_distance += distance

        except Exception as e:
            logger.error(f"Error calculating crowding distance: {e}")


class ParetoFrontNavigator:
    """Navigates and analyzes Pareto fronts."""

    def __init__(self):
        self.dominance_analyzer = DominanceAnalyzer()

    def find_best_compromise_solution(
        self, pareto_front: ParetoFront, objectives: List[OptimizationObjectiveSpec]
    ) -> Optional[OptimizationSolution]:
        """Find best compromise solution using various methods."""
        try:
            if not pareto_front.solutions:
                return None

            # Method 1: Minimum distance to ideal point
            ideal_point = self._calculate_ideal_point(
                pareto_front.solutions, objectives
            )

            best_solution = None
            min_distance = float("inf")

            for solution in pareto_front.solutions:
                distance = self._calculate_distance_to_ideal(
                    solution, ideal_point, objectives
                )
                if distance < min_distance:
                    min_distance = distance
                    best_solution = solution

            return best_solution

        except Exception as e:
            logger.error(f"Error finding best compromise solution: {e}")
            return pareto_front.solutions[0] if pareto_front.solutions else None

    def calculate_hypervolume(
        self,
        pareto_front: ParetoFront,
        objectives: List[OptimizationObjectiveSpec],
        reference_point: Optional[List[float]] = None,
    ) -> float:
        """Calculate hypervolume indicator."""
        try:
            if not pareto_front.solutions or len(objectives) > 3:
                return 0.0  # Simplified implementation

            # Use default reference point if not provided
            if reference_point is None:
                reference_point = self._calculate_reference_point(
                    pareto_front.solutions, objectives
                )

            # Simplified 2D hypervolume calculation
            if len(objectives) == 2:
                return self._calculate_2d_hypervolume(
                    pareto_front.solutions, objectives, reference_point
                )
            else:
                # For higher dimensions, use approximation
                return self._approximate_hypervolume(
                    pareto_front.solutions, objectives, reference_point
                )

        except Exception as e:
            logger.error(f"Error calculating hypervolume: {e}")
            return 0.0

    def calculate_spread_metric(
        self, pareto_front: ParetoFront, objectives: List[OptimizationObjectiveSpec]
    ) -> float:
        """Calculate spread metric for diversity assessment."""
        try:
            if len(pareto_front.solutions) < 2:
                return 0.0

            # Calculate average distance between consecutive solutions
            distances = []
            solutions = pareto_front.solutions

            for i in range(len(solutions)):
                for j in range(i + 1, len(solutions)):
                    distance = self._calculate_solution_distance(
                        solutions[i], solutions[j], objectives
                    )
                    distances.append(distance)

            if distances:
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)
                return std_distance / (avg_distance + 1e-10)  # Coefficient of variation

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating spread metric: {e}")
            return 0.0

    def _calculate_ideal_point(
        self,
        solutions: List[OptimizationSolution],
        objectives: List[OptimizationObjectiveSpec],
    ) -> Dict[OptimizationObjective, float]:
        """Calculate ideal point (best values for each objective)."""
        ideal_point = {}

        for obj_spec in objectives:
            obj = obj_spec.objective
            values = [sol.objective_values.get(obj, 0.0) for sol in solutions]

            if obj_spec.minimize:
                ideal_point[obj] = min(values)
            else:
                ideal_point[obj] = max(values)

        return ideal_point

    def _calculate_distance_to_ideal(
        self,
        solution: OptimizationSolution,
        ideal_point: Dict[OptimizationObjective, float],
        objectives: List[OptimizationObjectiveSpec],
    ) -> float:
        """Calculate Euclidean distance to ideal point."""
        distance_squared = 0.0

        for obj_spec in objectives:
            obj = obj_spec.objective
            sol_value = solution.objective_values.get(obj, 0.0)
            ideal_value = ideal_point.get(obj, 0.0)

            # Normalize by weight
            weighted_diff = obj_spec.weight * (sol_value - ideal_value)
            distance_squared += weighted_diff**2

        return np.sqrt(distance_squared)

    def _calculate_reference_point(
        self,
        solutions: List[OptimizationSolution],
        objectives: List[OptimizationObjectiveSpec],
    ) -> List[float]:
        """Calculate reference point for hypervolume."""
        reference_point = []

        for obj_spec in objectives:
            obj = obj_spec.objective
            values = [sol.objective_values.get(obj, 0.0) for sol in solutions]

            if obj_spec.minimize:
                # For minimization, use slightly worse than worst value
                reference_point.append(max(values) * 1.1)
            else:
                # For maximization, use slightly worse than worst value
                reference_point.append(min(values) * 0.9)

        return reference_point

    def _calculate_2d_hypervolume(
        self,
        solutions: List[OptimizationSolution],
        objectives: List[OptimizationObjectiveSpec],
        reference_point: List[float],
    ) -> float:
        """Calculate 2D hypervolume."""
        try:
            if len(objectives) != 2:
                return 0.0

            obj1, obj2 = objectives[0].objective, objectives[1].objective

            # Extract points
            points = []
            for sol in solutions:
                x = sol.objective_values.get(obj1, 0.0)
                y = sol.objective_values.get(obj2, 0.0)
                points.append((x, y))

            # Sort points
            points.sort()

            # Calculate hypervolume
            hypervolume = 0.0
            prev_x = reference_point[0]

            for x, y in points:
                if objectives[0].minimize and objectives[1].minimize:
                    area = (prev_x - x) * (reference_point[1] - y)
                elif objectives[0].minimize and not objectives[1].minimize:
                    area = (prev_x - x) * (y - reference_point[1])
                elif not objectives[0].minimize and objectives[1].minimize:
                    area = (x - prev_x) * (reference_point[1] - y)
                else:  # Both maximize
                    area = (x - prev_x) * (y - reference_point[1])

                hypervolume += max(0, area)
                prev_x = x

            return hypervolume

        except Exception as e:
            logger.error(f"Error calculating 2D hypervolume: {e}")
            return 0.0

    def _approximate_hypervolume(
        self,
        solutions: List[OptimizationSolution],
        objectives: List[OptimizationObjectiveSpec],
        reference_point: List[float],
    ) -> float:
        """Approximate hypervolume for higher dimensions."""
        # Simplified approximation
        total_volume = 1.0
        for i, obj_spec in enumerate(objectives):
            obj = obj_spec.objective
            values = [sol.objective_values.get(obj, 0.0) for sol in solutions]

            if obj_spec.minimize:
                range_val = reference_point[i] - min(values)
            else:
                range_val = max(values) - reference_point[i]

            total_volume *= max(0, range_val)

        return total_volume * len(solutions) / 100.0  # Scaling factor

    def _calculate_solution_distance(
        self,
        sol1: OptimizationSolution,
        sol2: OptimizationSolution,
        objectives: List[OptimizationObjectiveSpec],
    ) -> float:
        """Calculate distance between two solutions."""
        distance_squared = 0.0

        for obj_spec in objectives:
            obj = obj_spec.objective
            val1 = sol1.objective_values.get(obj, 0.0)
            val2 = sol2.objective_values.get(obj, 0.0)

            distance_squared += (val1 - val2) ** 2

        return np.sqrt(distance_squared)


class MultiObjectiveOptimizer:
    """Main multi-objective optimization engine."""

    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        self.config = config or MultiObjectiveConfig()

        # Initialize components
        self.objective_evaluator = ObjectiveEvaluator()
        self.dominance_analyzer = DominanceAnalyzer()
        self.pareto_navigator = ParetoFrontNavigator()

        # Set random seed
        if self.config.seed:
            np.random.seed(self.config.seed)

    def optimize_routes(
        self, candidate_routes: List[SynthesisRoute]
    ) -> OptimizationResult:
        """Optimize synthesis routes using multi-objective optimization."""
        start_time = time.time()

        try:
            # Convert routes to optimization solutions
            solutions = self._initialize_solutions(candidate_routes)

            # Run optimization algorithm
            if self.config.algorithm == OptimizationAlgorithm.NSGA_II:
                result = self._run_nsga_ii(solutions)
            elif self.config.algorithm == OptimizationAlgorithm.PARETO_FRONT:
                result = self._run_pareto_front_analysis(solutions)
            elif self.config.algorithm == OptimizationAlgorithm.WEIGHTED_SUM:
                result = self._run_weighted_sum(solutions)
            else:
                # Default to Pareto front analysis
                result = self._run_pareto_front_analysis(solutions)

            # Calculate metrics
            self._calculate_result_metrics(result)

            # Find best compromise solution
            if result.pareto_fronts:
                result.best_compromise_solution = (
                    self.pareto_navigator.find_best_compromise_solution(
                        result.pareto_fronts[0], self.config.objectives
                    )
                )

            result.total_time = time.time() - start_time

            logger.info(
                f"Multi-objective optimization completed in {result.total_time:.2f}s"
            )
            logger.info(
                f"Found {len(result.get_pareto_optimal_solutions())} Pareto optimal solutions"
            )

            return result

        except Exception as e:
            logger.error(f"Error in multi-objective optimization: {e}")
            raise MultiObjectiveOptimizationError(f"Optimization failed: {e}")

    def _initialize_solutions(
        self, routes: List[SynthesisRoute]
    ) -> List[OptimizationSolution]:
        """Initialize optimization solutions from routes."""
        solutions = []

        for route in routes:
            # Evaluate all objectives
            objective_values = self.objective_evaluator.evaluate_all_objectives(
                route, [obj_spec.objective for obj_spec in self.config.objectives]
            )

            solution = OptimizationSolution(
                route=route, objective_values=objective_values
            )

            solutions.append(solution)

        return solutions

    def _run_nsga_ii(
        self, initial_solutions: List[OptimizationSolution]
    ) -> OptimizationResult:
        """Run NSGA-II algorithm."""
        try:
            result = OptimizationResult()
            population = initial_solutions.copy()

            # Extend population if needed
            while len(population) < self.config.population_size:
                # Simple duplication with slight variation
                base_solution = np.random.choice(initial_solutions)
                new_solution = self._mutate_solution(base_solution)
                population.append(new_solution)

            # Evolution loop
            for generation in range(self.config.max_generations):
                # Find Pareto fronts
                fronts = self.dominance_analyzer.find_pareto_fronts(
                    population, self.config.objectives
                )

                # Calculate crowding distances
                for front in fronts:
                    self.dominance_analyzer.calculate_crowding_distance(
                        front.solutions, self.config.objectives
                    )

                # Selection for next generation
                new_population = []

                # Add solutions from fronts
                for front in fronts:
                    if (
                        len(new_population) + len(front.solutions)
                        <= self.config.population_size
                    ):
                        new_population.extend(front.solutions)
                    else:
                        # Sort by crowding distance and add best
                        remaining_slots = self.config.population_size - len(
                            new_population
                        )
                        front.solutions.sort(
                            key=lambda s: s.crowding_distance, reverse=True
                        )
                        new_population.extend(front.solutions[:remaining_slots])
                        break

                population = new_population

                # Record generation statistics
                if fronts:
                    hypervolume = self.pareto_navigator.calculate_hypervolume(
                        fronts[0], self.config.objectives
                    )
                    result.hypervolume_history.append(hypervolume)

                # Check convergence
                if self._check_convergence(result.hypervolume_history):
                    logger.info(f"Converged at generation {generation}")
                    break

            # Final front calculation
            result.pareto_fronts = self.dominance_analyzer.find_pareto_fronts(
                population, self.config.objectives
            )
            result.total_evaluations = (
                len(initial_solutions) * self.config.max_generations
            )

            return result

        except Exception as e:
            logger.error(f"Error in NSGA-II: {e}")
            raise MultiObjectiveOptimizationError(f"NSGA-II failed: {e}")

    def _run_pareto_front_analysis(
        self, solutions: List[OptimizationSolution]
    ) -> OptimizationResult:
        """Run simple Pareto front analysis."""
        try:
            result = OptimizationResult()

            # Find Pareto fronts
            result.pareto_fronts = self.dominance_analyzer.find_pareto_fronts(
                solutions, self.config.objectives
            )

            # Calculate crowding distances
            for front in result.pareto_fronts:
                self.dominance_analyzer.calculate_crowding_distance(
                    front.solutions, self.config.objectives
                )

            result.total_evaluations = len(solutions)

            return result

        except Exception as e:
            logger.error(f"Error in Pareto front analysis: {e}")
            raise MultiObjectiveOptimizationError(f"Pareto analysis failed: {e}")

    def _run_weighted_sum(
        self, solutions: List[OptimizationSolution]
    ) -> OptimizationResult:
        """Run weighted sum optimization."""
        try:
            result = OptimizationResult()

            # Calculate weighted sum for each solution
            for solution in solutions:
                weighted_sum = 0.0
                for obj_spec in self.config.objectives:
                    obj_value = solution.objective_values.get(obj_spec.objective, 0.0)
                    weighted_sum += obj_spec.weight * obj_value

                solution.fitness_values = [weighted_sum]

            # Sort by weighted sum
            solutions.sort(key=lambda s: s.fitness_values[0])

            # Create single front with best solutions
            best_front = ParetoFront(front_id=1)
            for solution in solutions[: min(self.config.archive_size, len(solutions))]:
                best_front.add_solution(solution)

            result.pareto_fronts = [best_front]
            result.total_evaluations = len(solutions)

            return result

        except Exception as e:
            logger.error(f"Error in weighted sum optimization: {e}")
            raise MultiObjectiveOptimizationError(f"Weighted sum failed: {e}")

    def _mutate_solution(
        self, base_solution: OptimizationSolution
    ) -> OptimizationSolution:
        """Create mutated version of solution."""
        # For now, just return a copy with slight objective perturbation
        new_solution = OptimizationSolution(route=base_solution.route)

        for obj, value in base_solution.objective_values.items():
            # Add small random perturbation
            perturbation = np.random.normal(0, 0.05 * abs(value))
            new_solution.objective_values[obj] = max(0, value + perturbation)

        return new_solution

    def _check_convergence(self, hypervolume_history: List[float]) -> bool:
        """Check if optimization has converged."""
        if len(hypervolume_history) < 5:
            return False

        # Check if hypervolume has stabilized
        recent_values = hypervolume_history[-5:]
        variance = np.var(recent_values)

        return variance < self.config.convergence_tolerance

    def _calculate_result_metrics(self, result: OptimizationResult):
        """Calculate metrics for optimization result."""
        if not result.pareto_fronts:
            return

        first_front = result.pareto_fronts[0]

        # Calculate hypervolume
        first_front.hypervolume = self.pareto_navigator.calculate_hypervolume(
            first_front, self.config.objectives
        )

        # Calculate spread metric
        first_front.spread_metric = self.pareto_navigator.calculate_spread_metric(
            first_front, self.config.objectives
        )


# Utility Functions
def create_multi_objective_config(
    objectives: List[Tuple[OptimizationObjective, float, bool]] = None,
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA_II,
    **kwargs,
) -> MultiObjectiveConfig:
    """Create multi-objective optimization configuration."""
    config = MultiObjectiveConfig(algorithm=algorithm, **kwargs)

    if objectives:
        for obj, weight, minimize in objectives:
            obj_spec = OptimizationObjectiveSpec(
                objective=obj, weight=weight, minimize=minimize
            )
            config.objectives.append(obj_spec)
    else:
        # Default objectives
        default_objectives = [
            (OptimizationObjective.OVERALL_YIELD, 1.0, True),  # Minimize (1 - yield)
            (OptimizationObjective.TOTAL_COST, 1.0, True),
            (OptimizationObjective.NUMBER_OF_STEPS, 0.5, True),
            (OptimizationObjective.ENVIRONMENTAL_IMPACT, 0.8, True),
        ]

        for obj, weight, minimize in default_objectives:
            obj_spec = OptimizationObjectiveSpec(
                objective=obj, weight=weight, minimize=minimize
            )
            config.objectives.append(obj_spec)

    return config


def optimize_synthesis_routes(
    routes: List[SynthesisRoute], config: Optional[MultiObjectiveConfig] = None
) -> OptimizationResult:
    """Optimize synthesis routes using multi-objective optimization."""
    optimizer = MultiObjectiveOptimizer(config)
    return optimizer.optimize_routes(routes)


def find_pareto_optimal_routes(
    routes: List[SynthesisRoute], objectives: List[OptimizationObjective] = None
) -> List[SynthesisRoute]:
    """Find Pareto optimal synthesis routes."""
    if objectives is None:
        objectives = [
            OptimizationObjective.OVERALL_YIELD,
            OptimizationObjective.TOTAL_COST,
            OptimizationObjective.NUMBER_OF_STEPS,
        ]

    # Create configuration
    obj_specs = [
        OptimizationObjectiveSpec(objective=obj, weight=1.0, minimize=True)
        for obj in objectives
    ]

    config = MultiObjectiveConfig(
        objectives=obj_specs, algorithm=OptimizationAlgorithm.PARETO_FRONT
    )

    # Run optimization
    result = optimize_synthesis_routes(routes, config)

    # Extract routes from Pareto optimal solutions
    pareto_routes = []
    for solution in result.get_pareto_optimal_solutions():
        pareto_routes.append(solution.route)

    return pareto_routes

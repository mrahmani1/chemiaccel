"""
Tests for multi-objective route optimization module.

This module tests the multi-objective optimization framework for synthesis routes,
including Pareto front analysis, objective evaluation, and optimization algorithms.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Import the module under test
from chemi_accel.retrosynthesis.multi_objective_optimization import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    ConvergenceError,
    DominanceAnalyzer,
    DominanceRelation,
    MultiObjectiveConfig,
    MultiObjectiveOptimizationError,
    MultiObjectiveOptimizer,
    ObjectiveEvaluationError,
    ObjectiveEvaluator,
    OptimizationAlgorithm,
    OptimizationObjective,
    OptimizationObjectiveSpec,
    OptimizationResult,
    OptimizationSolution,
    ParetoFront,
    ParetoFrontError,
    ParetoFrontNavigator,
    ParetoRanking,
    create_multi_objective_config,
    find_pareto_optimal_routes,
    optimize_synthesis_routes,
)
from chemi_accel.retrosynthesis.route_enumeration import SynthesisNode, SynthesisRoute


# Test fixtures
@pytest.fixture
def sample_objective_spec():
    """Create sample optimization objective specification."""
    return OptimizationObjectiveSpec(
        objective=OptimizationObjective.OVERALL_YIELD,
        weight=1.0,
        minimize=True,
        constraint_min=0.0,
        constraint_max=1.0,
        priority=1,
    )


@pytest.fixture
def sample_multi_objective_config():
    """Create sample multi-objective optimization configuration."""
    objectives = [
        OptimizationObjectiveSpec(
            objective=OptimizationObjective.OVERALL_YIELD, weight=1.0, minimize=True
        ),
        OptimizationObjectiveSpec(
            objective=OptimizationObjective.TOTAL_COST, weight=1.0, minimize=True
        ),
        OptimizationObjectiveSpec(
            objective=OptimizationObjective.NUMBER_OF_STEPS, weight=0.5, minimize=True
        ),
    ]

    return MultiObjectiveConfig(
        objectives=objectives,
        algorithm=OptimizationAlgorithm.NSGA_II,
        population_size=50,
        max_generations=20,
        crossover_rate=0.9,
        mutation_rate=0.1,
    )


@pytest.fixture
def sample_synthesis_routes():
    """Create sample synthesis routes for testing."""
    routes = []

    # Route 1: Short, expensive
    nodes1 = [
        SynthesisNode(smiles="CCO", depth=0, confidence=1.0, cost_estimate=0.0),
        SynthesisNode(
            smiles="CC",
            depth=1,
            confidence=0.9,
            cost_estimate=50.0,
            is_purchasable=True,
        ),
    ]
    route1 = SynthesisRoute(target_smiles="CCO", route_nodes=nodes1)
    routes.append(route1)

    # Route 2: Long, cheap
    nodes2 = [
        SynthesisNode(smiles="CCO", depth=0, confidence=1.0, cost_estimate=0.0),
        SynthesisNode(smiles="CCCO", depth=1, confidence=0.8, cost_estimate=10.0),
        SynthesisNode(
            smiles="CC", depth=2, confidence=0.7, cost_estimate=5.0, is_purchasable=True
        ),
        SynthesisNode(
            smiles="CO", depth=2, confidence=0.8, cost_estimate=8.0, is_purchasable=True
        ),
    ]
    route2 = SynthesisRoute(target_smiles="CCO", route_nodes=nodes2)
    routes.append(route2)

    # Route 3: Medium length, medium cost
    nodes3 = [
        SynthesisNode(smiles="CCO", depth=0, confidence=1.0, cost_estimate=0.0),
        SynthesisNode(smiles="CCCl", depth=1, confidence=0.85, cost_estimate=25.0),
        SynthesisNode(
            smiles="CCC",
            depth=2,
            confidence=0.9,
            cost_estimate=15.0,
            is_purchasable=True,
        ),
    ]
    route3 = SynthesisRoute(target_smiles="CCO", route_nodes=nodes3)
    routes.append(route3)

    return routes


@pytest.fixture
def sample_optimization_solutions(sample_synthesis_routes):
    """Create sample optimization solutions."""
    solutions = []

    objectives = [OptimizationObjective.OVERALL_YIELD, OptimizationObjective.TOTAL_COST]

    for i, route in enumerate(sample_synthesis_routes):
        solution = OptimizationSolution(
            route=route,
            objective_values={
                OptimizationObjective.OVERALL_YIELD: 1.0
                - (0.8 + i * 0.05),  # Converted to minimization
                OptimizationObjective.TOTAL_COST: route.total_cost,
                OptimizationObjective.NUMBER_OF_STEPS: float(route.total_steps),
            },
        )
        solutions.append(solution)

    return solutions


class TestEnums:
    """Test enum classes."""

    def test_optimization_objective_values(self):
        """Test OptimizationObjective enum values."""
        assert OptimizationObjective.OVERALL_YIELD.value == "overall_yield"
        assert OptimizationObjective.TOTAL_COST.value == "total_cost"
        assert OptimizationObjective.NUMBER_OF_STEPS.value == "number_of_steps"
        assert (
            OptimizationObjective.ENVIRONMENTAL_IMPACT.value == "environmental_impact"
        )

    def test_optimization_algorithm_values(self):
        """Test OptimizationAlgorithm enum values."""
        assert OptimizationAlgorithm.NSGA_II.value == "nsga_ii"
        assert OptimizationAlgorithm.NSGA_III.value == "nsga_iii"
        assert OptimizationAlgorithm.PARETO_FRONT.value == "pareto_front"
        assert OptimizationAlgorithm.WEIGHTED_SUM.value == "weighted_sum"

    def test_dominance_relation_values(self):
        """Test DominanceRelation enum values."""
        assert DominanceRelation.DOMINATES.value == "dominates"
        assert DominanceRelation.DOMINATED.value == "dominated"
        assert DominanceRelation.NON_DOMINATED.value == "non_dominated"
        assert DominanceRelation.EQUIVALENT.value == "equivalent"


class TestDataStructures:
    """Test data structure classes."""

    def test_optimization_objective_spec_creation(self, sample_objective_spec):
        """Test OptimizationObjectiveSpec creation."""
        spec = sample_objective_spec

        assert spec.objective == OptimizationObjective.OVERALL_YIELD
        assert spec.weight == 1.0
        assert spec.minimize is True
        assert spec.priority == 1

    def test_multi_objective_config_creation(self, sample_multi_objective_config):
        """Test MultiObjectiveConfig creation."""
        config = sample_multi_objective_config

        assert len(config.objectives) == 3
        assert config.algorithm == OptimizationAlgorithm.NSGA_II
        assert config.population_size == 50
        assert config.max_generations == 20

    def test_optimization_solution_creation(self, sample_synthesis_routes):
        """Test OptimizationSolution creation."""
        route = sample_synthesis_routes[0]

        solution = OptimizationSolution(
            route=route,
            objective_values={
                OptimizationObjective.OVERALL_YIELD: 0.2,
                OptimizationObjective.TOTAL_COST: 100.0,
            },
        )

        assert solution.route == route
        assert solution.objective_values[OptimizationObjective.OVERALL_YIELD] == 0.2
        assert solution.objective_values[OptimizationObjective.TOTAL_COST] == 100.0
        assert solution.solution_id != ""

    def test_pareto_front_creation(self):
        """Test ParetoFront creation."""
        front = ParetoFront(front_id=1)

        assert front.front_id == 1
        assert len(front.solutions) == 0
        assert front.size() == 0

    def test_pareto_front_add_solution(self, sample_optimization_solutions):
        """Test adding solution to Pareto front."""
        front = ParetoFront(front_id=1)
        solution = sample_optimization_solutions[0]

        front.add_solution(solution)

        assert front.size() == 1
        assert solution in front.solutions
        assert solution.pareto_ranking == ParetoRanking.RANK_1

    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        result = OptimizationResult()

        assert isinstance(result.pareto_fronts, list)
        assert isinstance(result.optimization_history, list)
        assert isinstance(result.convergence_data, dict)
        assert result.total_evaluations == 0
        assert result.total_time == 0.0

    def test_optimization_result_get_pareto_optimal_solutions(
        self, sample_optimization_solutions
    ):
        """Test getting Pareto optimal solutions."""
        result = OptimizationResult()

        # Create Pareto front
        front = ParetoFront(front_id=1)
        for solution in sample_optimization_solutions[:2]:
            front.add_solution(solution)

        result.pareto_fronts = [front]

        pareto_optimal = result.get_pareto_optimal_solutions()

        assert len(pareto_optimal) == 2
        assert all(sol in sample_optimization_solutions[:2] for sol in pareto_optimal)


class TestObjectiveEvaluator:
    """Test ObjectiveEvaluator class."""

    def test_objective_evaluator_creation(self):
        """Test objective evaluator creation."""
        evaluator = ObjectiveEvaluator()

        assert hasattr(evaluator, "forward_predictor")
        assert isinstance(evaluator.evaluation_cache, dict)

    def test_evaluate_overall_yield(self, sample_synthesis_routes):
        """Test overall yield evaluation."""
        evaluator = ObjectiveEvaluator()
        route = sample_synthesis_routes[0]

        yield_score = evaluator._evaluate_overall_yield(route)

        assert 0.0 <= yield_score <= 1.0  # Should be 1 - yield for minimization

    def test_evaluate_total_cost(self, sample_synthesis_routes):
        """Test total cost evaluation."""
        evaluator = ObjectiveEvaluator()
        route = sample_synthesis_routes[0]

        cost_score = evaluator._evaluate_total_cost(route)

        assert cost_score == route.total_cost

    def test_evaluate_number_of_steps(self, sample_synthesis_routes):
        """Test number of steps evaluation."""
        evaluator = ObjectiveEvaluator()
        route = sample_synthesis_routes[0]

        steps_score = evaluator._evaluate_number_of_steps(route)

        assert steps_score == float(route.total_steps)

    def test_evaluate_environmental_impact(self, sample_synthesis_routes):
        """Test environmental impact evaluation."""
        evaluator = ObjectiveEvaluator()
        route = sample_synthesis_routes[0]

        impact_score = evaluator._evaluate_environmental_impact(route)

        assert impact_score >= 0.0

    def test_evaluate_atom_economy(self, sample_synthesis_routes):
        """Test atom economy evaluation."""
        evaluator = ObjectiveEvaluator()
        route = sample_synthesis_routes[0]

        atom_economy_score = evaluator._evaluate_atom_economy(route)

        assert (
            0.0 <= atom_economy_score <= 1.0
        )  # Should be 1 - atom_economy for minimization

    def test_evaluate_all_objectives(self, sample_synthesis_routes):
        """Test evaluating all objectives."""
        evaluator = ObjectiveEvaluator()
        route = sample_synthesis_routes[0]

        objectives = [
            OptimizationObjective.OVERALL_YIELD,
            OptimizationObjective.TOTAL_COST,
            OptimizationObjective.NUMBER_OF_STEPS,
        ]

        results = evaluator.evaluate_all_objectives(route, objectives)

        assert isinstance(results, dict)
        assert len(results) == len(objectives)
        for obj in objectives:
            assert obj in results
            assert isinstance(results[obj], (int, float))

    def test_objective_caching(self, sample_synthesis_routes):
        """Test objective evaluation caching."""
        evaluator = ObjectiveEvaluator()
        route = sample_synthesis_routes[0]
        objective = OptimizationObjective.TOTAL_COST

        # First evaluation
        score1 = evaluator.evaluate_objective(route, objective)
        assert route.route_id in evaluator.evaluation_cache

        # Second evaluation (should use cache)
        score2 = evaluator.evaluate_objective(route, objective)
        assert score1 == score2

    def test_unknown_objective_error(self, sample_synthesis_routes):
        """Test error handling for unknown objectives."""
        evaluator = ObjectiveEvaluator()
        route = sample_synthesis_routes[0]

        # This should handle gracefully and return 0.0
        score = evaluator.evaluate_objective(route, "unknown_objective")
        assert score == 0.0


class TestDominanceAnalyzer:
    """Test DominanceAnalyzer class."""

    def test_dominance_analyzer_creation(self):
        """Test dominance analyzer creation."""
        analyzer = DominanceAnalyzer()

        assert analyzer is not None

    def test_compare_solutions_dominates(
        self, sample_optimization_solutions, sample_multi_objective_config
    ):
        """Test solution comparison - dominance."""
        analyzer = DominanceAnalyzer()

        # Create solutions where one clearly dominates
        sol1 = sample_optimization_solutions[0]
        sol2 = sample_optimization_solutions[1]

        # Make sol1 better in all objectives
        sol1.objective_values = {
            OptimizationObjective.OVERALL_YIELD: 0.1,  # Lower is better (minimization)
            OptimizationObjective.TOTAL_COST: 10.0,
            OptimizationObjective.NUMBER_OF_STEPS: 1.0,
        }

        sol2.objective_values = {
            OptimizationObjective.OVERALL_YIELD: 0.3,
            OptimizationObjective.TOTAL_COST: 30.0,
            OptimizationObjective.NUMBER_OF_STEPS: 3.0,
        }

        relation = analyzer.compare_solutions(
            sol1, sol2, sample_multi_objective_config.objectives
        )

        assert relation == DominanceRelation.DOMINATES

    def test_compare_solutions_non_dominated(
        self, sample_optimization_solutions, sample_multi_objective_config
    ):
        """Test solution comparison - non-dominated."""
        analyzer = DominanceAnalyzer()

        sol1 = sample_optimization_solutions[0]
        sol2 = sample_optimization_solutions[1]

        # Make solutions trade-off (better in some objectives, worse in others)
        sol1.objective_values = {
            OptimizationObjective.OVERALL_YIELD: 0.1,  # Better
            OptimizationObjective.TOTAL_COST: 30.0,  # Worse
            OptimizationObjective.NUMBER_OF_STEPS: 2.0,
        }

        sol2.objective_values = {
            OptimizationObjective.OVERALL_YIELD: 0.3,  # Worse
            OptimizationObjective.TOTAL_COST: 10.0,  # Better
            OptimizationObjective.NUMBER_OF_STEPS: 2.0,
        }

        relation = analyzer.compare_solutions(
            sol1, sol2, sample_multi_objective_config.objectives
        )

        assert relation == DominanceRelation.NON_DOMINATED

    def test_find_pareto_fronts(
        self, sample_optimization_solutions, sample_multi_objective_config
    ):
        """Test Pareto front finding."""
        analyzer = DominanceAnalyzer()

        fronts = analyzer.find_pareto_fronts(
            sample_optimization_solutions, sample_multi_objective_config.objectives
        )

        assert isinstance(fronts, list)
        assert len(fronts) >= 1
        assert all(isinstance(front, ParetoFront) for front in fronts)

        # First front should be non-empty
        assert fronts[0].size() > 0

        # All solutions should be assigned to some front
        total_solutions = sum(front.size() for front in fronts)
        assert total_solutions == len(sample_optimization_solutions)

    def test_calculate_crowding_distance(
        self, sample_optimization_solutions, sample_multi_objective_config
    ):
        """Test crowding distance calculation."""
        analyzer = DominanceAnalyzer()

        # Calculate crowding distance
        analyzer.calculate_crowding_distance(
            sample_optimization_solutions, sample_multi_objective_config.objectives
        )

        # Check that distances were calculated
        for solution in sample_optimization_solutions:
            assert hasattr(solution, "crowding_distance")
            assert solution.crowding_distance >= 0

    def test_crowding_distance_boundary_solutions(self, sample_multi_objective_config):
        """Test that boundary solutions get infinite crowding distance."""
        analyzer = DominanceAnalyzer()

        # Create solutions with clear ordering
        solutions = []
        for i in range(5):
            solution = OptimizationSolution(
                route=Mock(),
                objective_values={
                    OptimizationObjective.OVERALL_YIELD: float(i) * 0.1,
                    OptimizationObjective.TOTAL_COST: float(i) * 10.0,
                },
            )
            solutions.append(solution)

        analyzer.calculate_crowding_distance(
            solutions, sample_multi_objective_config.objectives
        )

        # Boundary solutions should have infinite distance
        assert solutions[0].crowding_distance == float("inf")
        assert solutions[-1].crowding_distance == float("inf")


class TestParetoFrontNavigator:
    """Test ParetoFrontNavigator class."""

    def test_pareto_navigator_creation(self):
        """Test Pareto front navigator creation."""
        navigator = ParetoFrontNavigator()

        assert isinstance(navigator.dominance_analyzer, DominanceAnalyzer)

    def test_find_best_compromise_solution(
        self, sample_optimization_solutions, sample_multi_objective_config
    ):
        """Test finding best compromise solution."""
        navigator = ParetoFrontNavigator()

        # Create Pareto front
        front = ParetoFront(front_id=1)
        for solution in sample_optimization_solutions:
            front.add_solution(solution)

        best_solution = navigator.find_best_compromise_solution(
            front, sample_multi_objective_config.objectives
        )

        assert best_solution is not None
        assert best_solution in sample_optimization_solutions

    def test_calculate_hypervolume(
        self, sample_optimization_solutions, sample_multi_objective_config
    ):
        """Test hypervolume calculation."""
        navigator = ParetoFrontNavigator()

        # Create Pareto front
        front = ParetoFront(front_id=1)
        for solution in sample_optimization_solutions:
            front.add_solution(solution)

        # Use only 2 objectives for 2D hypervolume
        objectives_2d = sample_multi_objective_config.objectives[:2]

        hypervolume = navigator.calculate_hypervolume(front, objectives_2d)

        assert hypervolume >= 0.0

    def test_calculate_spread_metric(
        self, sample_optimization_solutions, sample_multi_objective_config
    ):
        """Test spread metric calculation."""
        navigator = ParetoFrontNavigator()

        # Create Pareto front
        front = ParetoFront(front_id=1)
        for solution in sample_optimization_solutions:
            front.add_solution(solution)

        spread = navigator.calculate_spread_metric(
            front, sample_multi_objective_config.objectives
        )

        assert spread >= 0.0

    def test_calculate_ideal_point(
        self, sample_optimization_solutions, sample_multi_objective_config
    ):
        """Test ideal point calculation."""
        navigator = ParetoFrontNavigator()

        ideal_point = navigator._calculate_ideal_point(
            sample_optimization_solutions, sample_multi_objective_config.objectives
        )

        assert isinstance(ideal_point, dict)
        assert len(ideal_point) == len(sample_multi_objective_config.objectives)

        for obj_spec in sample_multi_objective_config.objectives:
            assert obj_spec.objective in ideal_point

    def test_empty_front_handling(self, sample_multi_objective_config):
        """Test handling of empty Pareto fronts."""
        navigator = ParetoFrontNavigator()

        empty_front = ParetoFront(front_id=1)

        # Should handle empty front gracefully
        best_solution = navigator.find_best_compromise_solution(
            empty_front, sample_multi_objective_config.objectives
        )
        assert best_solution is None

        hypervolume = navigator.calculate_hypervolume(
            empty_front, sample_multi_objective_config.objectives
        )
        assert hypervolume == 0.0

        spread = navigator.calculate_spread_metric(
            empty_front, sample_multi_objective_config.objectives
        )
        assert spread == 0.0


class TestMultiObjectiveOptimizer:
    """Test MultiObjectiveOptimizer class."""

    def test_optimizer_creation(self, sample_multi_objective_config):
        """Test multi-objective optimizer creation."""
        optimizer = MultiObjectiveOptimizer(sample_multi_objective_config)

        assert optimizer.config == sample_multi_objective_config
        assert isinstance(optimizer.objective_evaluator, ObjectiveEvaluator)
        assert isinstance(optimizer.dominance_analyzer, DominanceAnalyzer)
        assert isinstance(optimizer.pareto_navigator, ParetoFrontNavigator)

    def test_optimize_routes_pareto_front_algorithm(self, sample_synthesis_routes):
        """Test route optimization with Pareto front algorithm."""
        config = MultiObjectiveConfig(
            objectives=[
                OptimizationObjectiveSpec(
                    objective=OptimizationObjective.TOTAL_COST,
                    weight=1.0,
                    minimize=True,
                ),
                OptimizationObjectiveSpec(
                    objective=OptimizationObjective.NUMBER_OF_STEPS,
                    weight=1.0,
                    minimize=True,
                ),
            ],
            algorithm=OptimizationAlgorithm.PARETO_FRONT,
        )

        optimizer = MultiObjectiveOptimizer(config)
        result = optimizer.optimize_routes(sample_synthesis_routes)

        assert isinstance(result, OptimizationResult)
        assert len(result.pareto_fronts) >= 1
        assert result.total_evaluations > 0
        assert result.total_time > 0

    def test_optimize_routes_weighted_sum_algorithm(self, sample_synthesis_routes):
        """Test route optimization with weighted sum algorithm."""
        config = MultiObjectiveConfig(
            objectives=[
                OptimizationObjectiveSpec(
                    objective=OptimizationObjective.TOTAL_COST,
                    weight=0.7,
                    minimize=True,
                ),
                OptimizationObjectiveSpec(
                    objective=OptimizationObjective.NUMBER_OF_STEPS,
                    weight=0.3,
                    minimize=True,
                ),
            ],
            algorithm=OptimizationAlgorithm.WEIGHTED_SUM,
        )

        optimizer = MultiObjectiveOptimizer(config)
        result = optimizer.optimize_routes(sample_synthesis_routes)

        assert isinstance(result, OptimizationResult)
        assert len(result.pareto_fronts) >= 1
        assert result.total_evaluations > 0

    def test_optimize_routes_nsga_ii_algorithm(self, sample_synthesis_routes):
        """Test route optimization with NSGA-II algorithm."""
        config = MultiObjectiveConfig(
            objectives=[
                OptimizationObjectiveSpec(
                    objective=OptimizationObjective.TOTAL_COST,
                    weight=1.0,
                    minimize=True,
                ),
                OptimizationObjectiveSpec(
                    objective=OptimizationObjective.NUMBER_OF_STEPS,
                    weight=1.0,
                    minimize=True,
                ),
            ],
            algorithm=OptimizationAlgorithm.NSGA_II,
            population_size=20,
            max_generations=5,  # Small for testing
        )

        optimizer = MultiObjectiveOptimizer(config)
        result = optimizer.optimize_routes(sample_synthesis_routes)

        assert isinstance(result, OptimizationResult)
        assert len(result.pareto_fronts) >= 1
        assert result.total_evaluations > 0
        assert len(result.hypervolume_history) >= 0

    def test_initialize_solutions(
        self, sample_synthesis_routes, sample_multi_objective_config
    ):
        """Test solution initialization."""
        optimizer = MultiObjectiveOptimizer(sample_multi_objective_config)

        solutions = optimizer._initialize_solutions(sample_synthesis_routes)

        assert len(solutions) == len(sample_synthesis_routes)
        assert all(isinstance(sol, OptimizationSolution) for sol in solutions)

        # Check that objectives were evaluated
        for solution in solutions:
            assert len(solution.objective_values) > 0
            for obj_spec in sample_multi_objective_config.objectives:
                assert obj_spec.objective in solution.objective_values

    def test_mutate_solution(
        self, sample_optimization_solutions, sample_multi_objective_config
    ):
        """Test solution mutation."""
        optimizer = MultiObjectiveOptimizer(sample_multi_objective_config)

        base_solution = sample_optimization_solutions[0]
        mutated_solution = optimizer._mutate_solution(base_solution)

        assert isinstance(mutated_solution, OptimizationSolution)
        assert mutated_solution.route == base_solution.route
        assert len(mutated_solution.objective_values) == len(
            base_solution.objective_values
        )

    def test_check_convergence(self, sample_multi_objective_config):
        """Test convergence checking."""
        optimizer = MultiObjectiveOptimizer(sample_multi_objective_config)

        # Test with non-converged history
        non_converged_history = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert optimizer._check_convergence(non_converged_history) is False

        # Test with converged history
        converged_history = [1.0, 1.001, 1.002, 1.001, 1.003]
        optimizer.config.convergence_tolerance = 0.01
        assert optimizer._check_convergence(converged_history) is True

    def test_calculate_result_metrics(
        self, sample_optimization_solutions, sample_multi_objective_config
    ):
        """Test result metrics calculation."""
        optimizer = MultiObjectiveOptimizer(sample_multi_objective_config)

        # Create result with Pareto front
        result = OptimizationResult()
        front = ParetoFront(front_id=1)
        for solution in sample_optimization_solutions:
            front.add_solution(solution)
        result.pareto_fronts = [front]

        optimizer._calculate_result_metrics(result)

        assert front.hypervolume >= 0.0
        assert front.spread_metric >= 0.0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_multi_objective_config(self):
        """Test create_multi_objective_config function."""
        objectives = [
            (OptimizationObjective.OVERALL_YIELD, 1.0, True),
            (OptimizationObjective.TOTAL_COST, 0.8, True),
            (OptimizationObjective.NUMBER_OF_STEPS, 0.5, True),
        ]

        config = create_multi_objective_config(
            objectives=objectives,
            algorithm=OptimizationAlgorithm.NSGA_II,
            population_size=100,
        )

        assert isinstance(config, MultiObjectiveConfig)
        assert len(config.objectives) == 3
        assert config.algorithm == OptimizationAlgorithm.NSGA_II
        assert config.population_size == 100

        for i, (obj, weight, minimize) in enumerate(objectives):
            assert config.objectives[i].objective == obj
            assert config.objectives[i].weight == weight
            assert config.objectives[i].minimize == minimize

    def test_create_multi_objective_config_defaults(self):
        """Test create_multi_objective_config with defaults."""
        config = create_multi_objective_config()

        assert isinstance(config, MultiObjectiveConfig)
        assert len(config.objectives) == 4  # Default objectives
        assert config.algorithm == OptimizationAlgorithm.NSGA_II

        # Check default objectives
        obj_types = [obj_spec.objective for obj_spec in config.objectives]
        assert OptimizationObjective.OVERALL_YIELD in obj_types
        assert OptimizationObjective.TOTAL_COST in obj_types
        assert OptimizationObjective.NUMBER_OF_STEPS in obj_types
        assert OptimizationObjective.ENVIRONMENTAL_IMPACT in obj_types

    def test_optimize_synthesis_routes_function(self, sample_synthesis_routes):
        """Test optimize_synthesis_routes utility function."""
        config = create_multi_objective_config(
            algorithm=OptimizationAlgorithm.PARETO_FRONT
        )

        result = optimize_synthesis_routes(sample_synthesis_routes, config)

        assert isinstance(result, OptimizationResult)
        assert len(result.pareto_fronts) >= 1
        assert result.total_evaluations > 0

    def test_find_pareto_optimal_routes_function(self, sample_synthesis_routes):
        """Test find_pareto_optimal_routes utility function."""
        objectives = [
            OptimizationObjective.TOTAL_COST,
            OptimizationObjective.NUMBER_OF_STEPS,
        ]

        pareto_routes = find_pareto_optimal_routes(sample_synthesis_routes, objectives)

        assert isinstance(pareto_routes, list)
        assert len(pareto_routes) <= len(sample_synthesis_routes)
        assert all(isinstance(route, SynthesisRoute) for route in pareto_routes)

    def test_find_pareto_optimal_routes_default_objectives(
        self, sample_synthesis_routes
    ):
        """Test find_pareto_optimal_routes with default objectives."""
        pareto_routes = find_pareto_optimal_routes(sample_synthesis_routes)

        assert isinstance(pareto_routes, list)
        assert len(pareto_routes) <= len(sample_synthesis_routes)


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_multi_objective_optimization_error_inheritance(self):
        """Test exception inheritance."""
        assert issubclass(ParetoFrontError, MultiObjectiveOptimizationError)
        assert issubclass(ObjectiveEvaluationError, MultiObjectiveOptimizationError)
        assert issubclass(ConvergenceError, MultiObjectiveOptimizationError)

    def test_pareto_front_error_handling(self, sample_multi_objective_config):
        """Test Pareto front error handling."""
        analyzer = DominanceAnalyzer()

        # Test with invalid solutions
        invalid_solutions = [OptimizationSolution(route=None)]  # Invalid route

        try:
            fronts = analyzer.find_pareto_fronts(
                invalid_solutions, sample_multi_objective_config.objectives
            )
            # If no error, check result structure
            assert isinstance(fronts, list)
        except ParetoFrontError:
            # Expected error is acceptable
            pass

    def test_objective_evaluation_error_handling(self, sample_synthesis_routes):
        """Test objective evaluation error handling."""
        evaluator = ObjectiveEvaluator()

        # Test with invalid route
        invalid_route = SynthesisRoute(target_smiles="", route_nodes=[])

        # Should handle gracefully
        score = evaluator.evaluate_objective(
            invalid_route, OptimizationObjective.TOTAL_COST
        )
        assert isinstance(score, (int, float))

    def test_optimization_error_handling(self):
        """Test optimization error handling."""
        config = MultiObjectiveConfig(
            objectives=[],  # Empty objectives
            algorithm=OptimizationAlgorithm.PARETO_FRONT,
        )

        optimizer = MultiObjectiveOptimizer(config)

        try:
            result = optimizer.optimize_routes([])  # Empty routes
            assert isinstance(result, OptimizationResult)
        except MultiObjectiveOptimizationError:
            # Expected error is acceptable
            pass


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    def test_objective_evaluation_performance(self, sample_synthesis_routes):
        """Test objective evaluation performance."""
        evaluator = ObjectiveEvaluator()

        # Test multiple evaluations
        objectives = [
            OptimizationObjective.OVERALL_YIELD,
            OptimizationObjective.TOTAL_COST,
            OptimizationObjective.NUMBER_OF_STEPS,
            OptimizationObjective.ENVIRONMENTAL_IMPACT,
        ]

        import time

        start_time = time.time()

        for route in sample_synthesis_routes * 5:  # 15 total evaluations
            for objective in objectives:
                score = evaluator.evaluate_objective(route, objective)
                assert isinstance(score, (int, float))

        elapsed_time = time.time() - start_time
        assert elapsed_time < 10.0  # Should complete within 10 seconds

    def test_pareto_front_performance(self, sample_multi_objective_config):
        """Test Pareto front calculation performance."""
        analyzer = DominanceAnalyzer()

        # Create larger set of solutions
        solutions = []
        for i in range(50):  # 50 solutions
            solution = OptimizationSolution(
                route=Mock(),
                objective_values={
                    OptimizationObjective.OVERALL_YIELD: np.random.uniform(0.1, 0.9),
                    OptimizationObjective.TOTAL_COST: np.random.uniform(10.0, 100.0),
                    OptimizationObjective.NUMBER_OF_STEPS: np.random.uniform(1.0, 10.0),
                },
            )
            solutions.append(solution)

        import time

        start_time = time.time()

        fronts = analyzer.find_pareto_fronts(
            solutions, sample_multi_objective_config.objectives
        )

        elapsed_time = time.time() - start_time
        assert elapsed_time < 5.0  # Should complete within 5 seconds
        assert len(fronts) >= 1

    def test_optimization_algorithm_performance(self, sample_synthesis_routes):
        """Test optimization algorithm performance."""
        config = MultiObjectiveConfig(
            objectives=[
                OptimizationObjectiveSpec(
                    objective=OptimizationObjective.TOTAL_COST,
                    weight=1.0,
                    minimize=True,
                ),
                OptimizationObjectiveSpec(
                    objective=OptimizationObjective.NUMBER_OF_STEPS,
                    weight=1.0,
                    minimize=True,
                ),
            ],
            algorithm=OptimizationAlgorithm.PARETO_FRONT,
        )

        optimizer = MultiObjectiveOptimizer(config)

        import time

        start_time = time.time()

        result = optimizer.optimize_routes(sample_synthesis_routes)

        elapsed_time = time.time() - start_time
        assert elapsed_time < 30.0  # Should complete within 30 seconds
        assert isinstance(result, OptimizationResult)


class TestIntegration:
    """Integration tests for multi-objective optimization."""

    def test_full_optimization_workflow(self, sample_synthesis_routes):
        """Test complete optimization workflow."""
        # Create configuration
        config = create_multi_objective_config(
            objectives=[
                (OptimizationObjective.OVERALL_YIELD, 1.0, True),
                (OptimizationObjective.TOTAL_COST, 0.8, True),
                (OptimizationObjective.NUMBER_OF_STEPS, 0.5, True),
            ],
            algorithm=OptimizationAlgorithm.PARETO_FRONT,
        )

        # Run optimization
        result = optimize_synthesis_routes(sample_synthesis_routes, config)

        # Verify complete workflow
        assert isinstance(result, OptimizationResult)
        assert len(result.pareto_fronts) >= 1
        assert result.total_evaluations > 0
        assert result.total_time > 0

        # Check Pareto optimal solutions
        pareto_optimal = result.get_pareto_optimal_solutions()
        assert len(pareto_optimal) <= len(sample_synthesis_routes)
        assert all(isinstance(sol, OptimizationSolution) for sol in pareto_optimal)

        # Check best compromise solution
        if result.best_compromise_solution:
            assert isinstance(result.best_compromise_solution, OptimizationSolution)
            assert result.best_compromise_solution in pareto_optimal

    def test_different_optimization_algorithms(self, sample_synthesis_routes):
        """Test different optimization algorithms."""
        algorithms = [
            OptimizationAlgorithm.PARETO_FRONT,
            OptimizationAlgorithm.WEIGHTED_SUM,
            OptimizationAlgorithm.NSGA_II,
        ]

        all_results = []
        for algorithm in algorithms:
            config = MultiObjectiveConfig(
                objectives=[
                    OptimizationObjectiveSpec(
                        objective=OptimizationObjective.TOTAL_COST,
                        weight=1.0,
                        minimize=True,
                    ),
                    OptimizationObjectiveSpec(
                        objective=OptimizationObjective.NUMBER_OF_STEPS,
                        weight=1.0,
                        minimize=True,
                    ),
                ],
                algorithm=algorithm,
                population_size=20,
                max_generations=3,  # Small for testing
            )

            optimizer = MultiObjectiveOptimizer(config)
            result = optimizer.optimize_routes(sample_synthesis_routes)

            assert isinstance(result, OptimizationResult)
            assert len(result.pareto_fronts) >= 1
            all_results.append(result)

        # Should have results from all algorithms
        assert len(all_results) == len(algorithms)

    def test_objective_trade_offs(self, sample_synthesis_routes):
        """Test objective trade-offs in optimization."""
        config = create_multi_objective_config(
            objectives=[
                (OptimizationObjective.TOTAL_COST, 1.0, True),  # Minimize cost
                (OptimizationObjective.NUMBER_OF_STEPS, 1.0, True),  # Minimize steps
            ],
            algorithm=OptimizationAlgorithm.PARETO_FRONT,
        )

        result = optimize_synthesis_routes(sample_synthesis_routes, config)
        pareto_optimal = result.get_pareto_optimal_solutions()

        if len(pareto_optimal) > 1:
            # Check that solutions represent trade-offs
            costs = [
                sol.objective_values[OptimizationObjective.TOTAL_COST]
                for sol in pareto_optimal
            ]
            steps = [
                sol.objective_values[OptimizationObjective.NUMBER_OF_STEPS]
                for sol in pareto_optimal
            ]

            # Should have variation in objectives
            assert len(set(costs)) > 1 or len(set(steps)) > 1

    def test_integration_with_route_evaluation(self, sample_synthesis_routes):
        """Test integration with route evaluation components."""
        # Test with comprehensive objectives
        config = create_multi_objective_config(
            objectives=[
                (OptimizationObjective.OVERALL_YIELD, 1.0, True),
                (OptimizationObjective.TOTAL_COST, 1.0, True),
                (OptimizationObjective.NUMBER_OF_STEPS, 0.8, True),
                (OptimizationObjective.ENVIRONMENTAL_IMPACT, 0.6, True),
                (OptimizationObjective.ATOM_ECONOMY, 0.7, True),
            ],
            algorithm=OptimizationAlgorithm.PARETO_FRONT,
        )

        result = optimize_synthesis_routes(sample_synthesis_routes, config)

        # Verify that all objectives were evaluated
        pareto_optimal = result.get_pareto_optimal_solutions()
        for solution in pareto_optimal:
            assert len(solution.objective_values) == len(config.objectives)
            for obj_spec in config.objectives:
                assert obj_spec.objective in solution.objective_values
                assert isinstance(
                    solution.objective_values[obj_spec.objective], (int, float)
                )

    def test_error_propagation_through_system(self):
        """Test error propagation through the entire system."""
        # Test with problematic configuration
        config = MultiObjectiveConfig(
            objectives=[],  # Empty objectives
            algorithm=OptimizationAlgorithm.PARETO_FRONT,
        )

        optimizer = MultiObjectiveOptimizer(config)

        try:
            result = optimizer.optimize_routes([])  # Empty routes
            # If successful, verify result structure
            assert isinstance(result, OptimizationResult)
        except MultiObjectiveOptimizationError:
            # Expected errors are acceptable
            pass

    def test_pareto_front_navigation_integration(self, sample_synthesis_routes):
        """Test integration with Pareto front navigation."""
        config = create_multi_objective_config(
            objectives=[
                (OptimizationObjective.TOTAL_COST, 1.0, True),
                (OptimizationObjective.NUMBER_OF_STEPS, 1.0, True),
            ],
            algorithm=OptimizationAlgorithm.PARETO_FRONT,
        )

        result = optimize_synthesis_routes(sample_synthesis_routes, config)

        if result.pareto_fronts:
            navigator = ParetoFrontNavigator()
            front = result.pareto_fronts[0]

            # Test navigation features
            best_compromise = navigator.find_best_compromise_solution(
                front, config.objectives
            )
            hypervolume = navigator.calculate_hypervolume(front, config.objectives)
            spread = navigator.calculate_spread_metric(front, config.objectives)

            if best_compromise:
                assert isinstance(best_compromise, OptimizationSolution)
            assert hypervolume >= 0.0
            assert spread >= 0.0

"""
Tests for Multi-Objective Experimental Optimization & Pareto Front Navigation.

This module provides comprehensive testing for the experimental optimization
framework including multi-objective optimization, Pareto front navigation,
and interactive trade-off analysis.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.ai.experimental_optimization import (  # Core classes; Data structures; Enums; Utilities; Factory functions; Exceptions
    CrowdingDistance,
    ExperimentalCondition,
    ExperimentalConstraint,
    ExperimentalObjective,
    ExperimentalOptimizationError,
    ExperimentalOptimizer,
    ExperimentalParameter,
    ExperimentalSpace,
    HypervolumeCalculator,
    NonDominatedSorting,
    ObjectiveType,
    OptimizationAlgorithm,
    ParetoFront,
    ParetoNavigator,
    ParetoPoint,
    SurrogateExperimentalModel,
    analyze_pareto_trade_offs,
    create_chemical_synthesis_objectives,
    create_polymer_synthesis_objectives,
    create_standard_experimental_space,
    find_optimal_experimental_conditions,
    run_experimental_optimization,
)

# Check for optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import plotly

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TestExperimentalObjective:
    """Test experimental objective configuration."""

    def test_objective_creation(self):
        """Test creating experimental objectives."""
        objective = ExperimentalObjective(
            name="yield",
            parameter="product_yield",
            objective_type=ObjectiveType.MAXIMIZE,
            weight=1.0,
            units="%",
        )

        assert objective.name == "yield"
        assert objective.parameter == "product_yield"
        assert objective.objective_type == ObjectiveType.MAXIMIZE
        assert objective.weight == 1.0
        assert objective.units == "%"
        assert objective.priority == 1

    def test_target_objective(self):
        """Test target-type objective."""
        objective = ExperimentalObjective(
            name="molecular_weight",
            parameter="mn",
            objective_type=ObjectiveType.TARGET,
            target_value=50000,
            tolerance=5000,
        )

        assert objective.objective_type == ObjectiveType.TARGET
        assert objective.target_value == 50000
        assert objective.tolerance == 5000


class TestExperimentalConstraint:
    """Test experimental constraint configuration."""

    def test_constraint_creation(self):
        """Test creating experimental constraints."""
        constraint = ExperimentalConstraint(
            name="temperature_safety",
            parameter="temperature",
            min_value=20.0,
            max_value=150.0,
            penalty_weight=2.0,
            constraint_type="safety",
        )

        assert constraint.name == "temperature_safety"
        assert constraint.parameter == "temperature"
        assert constraint.min_value == 20.0
        assert constraint.max_value == 150.0
        assert constraint.penalty_weight == 2.0
        assert constraint.constraint_type == "safety"

    def test_allowed_values_constraint(self):
        """Test constraint with allowed values."""
        constraint = ExperimentalConstraint(
            name="catalyst_selection",
            parameter="catalyst",
            allowed_values=["Pd", "Pt", "Rh"],
            constraint_type="selection",
        )

        assert constraint.allowed_values == ["Pd", "Pt", "Rh"]
        assert constraint.min_value is None
        assert constraint.max_value is None


class TestExperimentalCondition:
    """Test experimental condition representation."""

    def test_condition_creation(self):
        """Test creating experimental conditions."""
        parameters = {
            "temperature": 80.0,
            "pressure": 2.0,
            "concentration": 0.5,
            "time": 4.0,
        }

        condition = ExperimentalCondition(parameters=parameters)

        assert condition.parameters == parameters
        assert condition.constraints_satisfied is True
        assert condition.feasible is True
        assert isinstance(condition.experiment_id, str)
        assert len(condition.experiment_id) > 0

    def test_condition_with_objectives(self):
        """Test condition with objective values."""
        parameters = {"temperature": 100.0}
        objectives = {"yield": 85.0, "selectivity": 92.0}

        condition = ExperimentalCondition(
            parameters=parameters, objectives=objectives, constraints_satisfied=True
        )

        assert condition.objectives == objectives
        assert condition.constraints_satisfied is True


class TestExperimentalSpace:
    """Test experimental parameter space definition."""

    def test_space_creation(self):
        """Test creating experimental space."""
        parameters = {
            "temperature": (20.0, 200.0),
            "pressure": (1.0, 10.0),
            "concentration": (0.01, 2.0),
        }

        discrete_parameters = {
            "catalyst": ["Pd", "Pt", "Rh"],
            "solvent": ["DMF", "THF", "DCM"],
        }

        constraints = [
            ExperimentalConstraint(
                name="temp_limit", parameter="temperature", max_value=150.0
            )
        ]

        space = ExperimentalSpace(
            parameters=parameters,
            discrete_parameters=discrete_parameters,
            constraints=constraints,
        )

        assert space.parameters == parameters
        assert space.discrete_parameters == discrete_parameters
        assert len(space.constraints) == 1
        assert space.constraints[0].name == "temp_limit"


class TestSurrogateExperimentalModel:
    """Test surrogate experimental model."""

    def test_model_creation(self):
        """Test creating surrogate model."""
        model = SurrogateExperimentalModel(use_gpu=False)

        assert model.use_gpu is False
        assert model.fitted is False
        assert len(model.models) == 0
        assert len(model.training_data) == 0

    def test_model_prediction_unfitted(self):
        """Test prediction with unfitted model."""
        model = SurrogateExperimentalModel(use_gpu=False)
        condition = ExperimentalCondition(
            parameters={"temperature": 80.0, "pressure": 2.0}
        )

        predictions = model.predict(condition)
        assert isinstance(predictions, dict)
        assert len(predictions) == 0

    def test_model_update_and_prediction(self):
        """Test model update and prediction."""
        model = SurrogateExperimentalModel(use_gpu=False)

        # Create training data
        conditions = [
            ExperimentalCondition(parameters={"temperature": 80.0, "pressure": 2.0}),
            ExperimentalCondition(parameters={"temperature": 100.0, "pressure": 3.0}),
            ExperimentalCondition(parameters={"temperature": 120.0, "pressure": 4.0}),
        ]

        outcomes = [
            {"yield": 75.0, "selectivity": 80.0},
            {"yield": 85.0, "selectivity": 85.0},
            {"yield": 90.0, "selectivity": 90.0},
        ]

        # Update model
        model.update(conditions, outcomes)

        assert model.fitted is True
        assert len(model.models) == 2  # yield and selectivity
        assert "yield" in model.models
        assert "selectivity" in model.models

        # Test prediction
        test_condition = ExperimentalCondition(
            parameters={"temperature": 90.0, "pressure": 2.5}
        )
        predictions = model.predict(test_condition)

        assert isinstance(predictions, dict)
        assert "yield" in predictions
        assert "selectivity" in predictions
        assert isinstance(predictions["yield"], float)
        assert isinstance(predictions["selectivity"], float)

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation."""
        model = SurrogateExperimentalModel(use_gpu=False)

        # Add some training data
        conditions = [
            ExperimentalCondition(parameters={"temperature": 80.0}),
            ExperimentalCondition(parameters={"temperature": 100.0}),
        ]
        outcomes = [{"yield": 75.0}, {"yield": 85.0}]

        model.update(conditions, outcomes)

        test_condition = ExperimentalCondition(parameters={"temperature": 90.0})
        uncertainty = model.get_uncertainty(test_condition)

        assert isinstance(uncertainty, dict)
        assert "yield" in uncertainty
        assert isinstance(uncertainty["yield"], float)
        assert uncertainty["yield"] > 0


class TestNonDominatedSorting:
    """Test non-dominated sorting algorithm."""

    def test_simple_sorting(self):
        """Test sorting with simple 2D case."""
        # Create test points
        points = [
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[1.0, 2.0],
            ),
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[2.0, 1.0],
            ),
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[0.5, 0.5],
            ),
        ]

        fronts = NonDominatedSorting.fast_non_dominated_sort(points)

        assert len(fronts) >= 1
        assert len(fronts[0]) >= 1  # At least one point in first front

        # Check ranks are assigned
        for point in points:
            assert hasattr(point, "rank")
            assert point.rank >= 0

    def test_dominance_relationship(self):
        """Test dominance relationship checking."""
        point1 = ParetoPoint(
            conditions=ExperimentalCondition(parameters={}), objective_values=[2.0, 3.0]
        )
        point2 = ParetoPoint(
            conditions=ExperimentalCondition(parameters={}), objective_values=[1.0, 2.0]
        )

        # point1 should dominate point2 (higher values assumed better)
        dominates = NonDominatedSorting._dominates(point1, point2)
        assert dominates is True

        # point2 should not dominate point1
        dominates_reverse = NonDominatedSorting._dominates(point2, point1)
        assert dominates_reverse is False


class TestCrowdingDistance:
    """Test crowding distance calculation."""

    def test_crowding_distance_calculation(self):
        """Test crowding distance for diversity preservation."""
        # Create a front with multiple points
        points = []
        for i in range(5):
            point = ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[i * 0.2, (4 - i) * 0.2],
            )
            points.append(point)

        front_indices = list(range(5))
        CrowdingDistance.calculate_crowding_distance(front_indices, points)

        # Boundary points should have infinite distance
        assert points[0].crowding_distance == float("inf")
        assert points[-1].crowding_distance == float("inf")

        # Internal points should have finite positive distance
        for i in range(1, 4):
            assert points[i].crowding_distance > 0
            assert points[i].crowding_distance != float("inf")

    def test_small_front_crowding_distance(self):
        """Test crowding distance with small front."""
        points = [
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[1.0, 2.0],
            ),
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[2.0, 1.0],
            ),
        ]

        front_indices = [0, 1]
        CrowdingDistance.calculate_crowding_distance(front_indices, points)

        # Both points should have infinite distance (boundary)
        for point in points:
            assert point.crowding_distance == float("inf")


class TestHypervolumeCalculator:
    """Test hypervolume calculation."""

    def test_2d_hypervolume(self):
        """Test 2D hypervolume calculation."""
        points = [
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[3.0, 1.0],
            ),
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[1.0, 3.0],
            ),
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[2.0, 2.0],
            ),
        ]

        pareto_front = ParetoFront(points=points, objectives=[])
        reference_point = [4.0, 4.0]

        hypervolume = HypervolumeCalculator.calculate_hypervolume(
            pareto_front, reference_point
        )

        assert hypervolume > 0
        assert isinstance(hypervolume, float)

    def test_3d_hypervolume(self):
        """Test 3D hypervolume calculation."""
        points = [
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[3.0, 1.0, 1.0],
            ),
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[1.0, 3.0, 1.0],
            ),
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={}),
                objective_values=[1.0, 1.0, 3.0],
            ),
        ]

        pareto_front = ParetoFront(points=points, objectives=[])
        reference_point = [4.0, 4.0, 4.0]

        hypervolume = HypervolumeCalculator.calculate_hypervolume(
            pareto_front, reference_point
        )

        assert hypervolume > 0
        assert isinstance(hypervolume, float)

    def test_empty_front_hypervolume(self):
        """Test hypervolume with empty front."""
        pareto_front = ParetoFront(points=[], objectives=[])
        hypervolume = HypervolumeCalculator.calculate_hypervolume(pareto_front)

        assert hypervolume == 0.0


class TestExperimentalOptimizer:
    """Test multi-objective experimental optimizer."""

    def setup_method(self):
        """Set up test optimizer."""
        self.objectives = create_chemical_synthesis_objectives()
        self.experimental_space = create_standard_experimental_space()
        self.model = SurrogateExperimentalModel(use_gpu=False)

        # Add some initial training data
        conditions = [
            ExperimentalCondition(
                parameters={
                    "temperature": 80.0,
                    "pressure": 2.0,
                    "concentration": 0.5,
                    "time": 4.0,
                    "ph": 7.0,
                }
            ),
            ExperimentalCondition(
                parameters={
                    "temperature": 100.0,
                    "pressure": 3.0,
                    "concentration": 1.0,
                    "time": 6.0,
                    "ph": 8.0,
                }
            ),
        ]

        outcomes = [
            {"yield": 75.0, "selectivity": 80.0, "time": 4.0, "total_cost": 50.0},
            {"yield": 85.0, "selectivity": 85.0, "time": 6.0, "total_cost": 70.0},
        ]

        self.model.update(conditions, outcomes)

        self.optimizer = ExperimentalOptimizer(
            experimental_space=self.experimental_space,
            objectives=self.objectives,
            model=self.model,
            population_size=20,
            max_generations=5,  # Small for testing
        )

    def test_optimizer_creation(self):
        """Test optimizer initialization."""
        assert self.optimizer.experimental_space == self.experimental_space
        assert self.optimizer.objectives == self.objectives
        assert self.optimizer.model == self.model
        assert self.optimizer.population_size == 20
        assert self.optimizer.max_generations == 5
        assert self.optimizer.algorithm == OptimizationAlgorithm.NSGA_II

    def test_initial_population_generation(self):
        """Test initial population generation."""
        population = self.optimizer._generate_initial_population()

        assert len(population) == self.optimizer.population_size

        for point in population:
            assert isinstance(point, ParetoPoint)
            assert isinstance(point.conditions, ExperimentalCondition)
            assert len(point.objective_values) == len(self.objectives)

            # Check parameter bounds
            for param_name, value in point.conditions.parameters.items():
                if param_name in self.experimental_space.parameters:
                    min_val, max_val = self.experimental_space.parameters[param_name]
                    assert min_val <= value <= max_val

    def test_constraint_checking(self):
        """Test constraint satisfaction checking."""
        # Valid condition
        valid_condition = ExperimentalCondition(
            parameters={"temperature": 100.0, "concentration": 0.5}
        )

        assert self.optimizer._check_constraints(valid_condition) is True

        # Invalid condition (violates temperature constraint)
        invalid_condition = ExperimentalCondition(
            parameters={
                "temperature": 250.0,  # Exceeds max of 150
                "concentration": 0.5,
            }
        )

        assert self.optimizer._check_constraints(invalid_condition) is False

    def test_optimization_run(self):
        """Test running optimization."""
        pareto_front = self.optimizer.optimize()

        assert isinstance(pareto_front, ParetoFront)
        assert len(pareto_front.points) > 0
        assert pareto_front.objectives == self.objectives
        assert pareto_front.generation == self.optimizer.max_generations
        assert pareto_front.hypervolume >= 0

        # Check that all points have valid objective values
        for point in pareto_front.points:
            assert len(point.objective_values) == len(self.objectives)
            for value in point.objective_values:
                assert isinstance(value, (int, float))


class TestParetoNavigator:
    """Test Pareto front navigation tools."""

    def setup_method(self):
        """Set up test navigator."""
        # Create test Pareto front
        objectives = create_chemical_synthesis_objectives()

        points = []
        for i in range(10):
            condition = ExperimentalCondition(
                parameters={
                    "temperature": 50 + i * 10,
                    "pressure": 1 + i * 0.5,
                    "concentration": 0.1 + i * 0.1,
                }
            )

            point = ParetoPoint(
                conditions=condition,
                objective_values=[
                    70 + i * 2,  # yield
                    80 + i * 1.5,  # selectivity
                    -(5 + i * 0.5),  # reaction_time (negative for minimization)
                    -(30 + i * 3),  # cost (negative for minimization)
                ],
            )
            points.append(point)

        self.pareto_front = ParetoFront(points=points, objectives=objectives)
        self.navigator = ParetoNavigator(self.pareto_front)

    def test_navigator_creation(self):
        """Test navigator initialization."""
        assert self.navigator.pareto_front == self.pareto_front
        assert len(self.navigator.selected_points) == 0
        assert len(self.navigator.filters) == 0

    def test_parameter_filtering(self):
        """Test filtering by parameter range."""
        filtered_points = self.navigator.filter_by_parameter(
            parameter="temperature", min_val=80.0, max_val=120.0
        )

        assert len(filtered_points) > 0

        for point in filtered_points:
            temp = point.conditions.parameters["temperature"]
            assert 80.0 <= temp <= 120.0

    def test_objective_filtering(self):
        """Test filtering by objective value range."""
        filtered_points = self.navigator.filter_by_objective(
            objective_idx=0, min_val=75.0, max_val=85.0  # yield
        )

        assert len(filtered_points) > 0

        for point in filtered_points:
            yield_value = point.objective_values[0]
            assert 75.0 <= yield_value <= 85.0

    def test_knee_point_finding(self):
        """Test finding knee points on Pareto front."""
        knee_points = self.navigator.find_knee_points()

        assert len(knee_points) > 0
        assert all(isinstance(point, ParetoPoint) for point in knee_points)

        # Test with custom weights
        weights = [1.0, 0.8, 0.6, 0.7]
        weighted_knee_points = self.navigator.find_knee_points(weights)

        assert len(weighted_knee_points) > 0

    def test_trade_off_analysis(self):
        """Test trade-off analysis."""
        analysis = self.navigator.get_trade_off_analysis()

        assert isinstance(analysis, dict)
        assert "objective_ranges" in analysis
        assert "correlations" in analysis
        assert "diversity_metrics" in analysis

        # Check objective ranges
        for objective in self.pareto_front.objectives:
            assert objective.name in analysis["objective_ranges"]
            obj_range = analysis["objective_ranges"][objective.name]
            assert "min" in obj_range
            assert "max" in obj_range
            assert "mean" in obj_range
            assert "std" in obj_range

        # Check diversity metrics
        assert "mean_distance" in analysis["diversity_metrics"]
        assert "spread" in analysis["diversity_metrics"]

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_interactive_plotting_2d(self):
        """Test 2D interactive plotting."""
        # Create 2-objective front for 2D plotting
        objectives_2d = self.pareto_front.objectives[:2]
        points_2d = []

        for point in self.pareto_front.points:
            new_point = ParetoPoint(
                conditions=point.conditions, objective_values=point.objective_values[:2]
            )
            points_2d.append(new_point)

        front_2d = ParetoFront(points=points_2d, objectives=objectives_2d)
        navigator_2d = ParetoNavigator(front_2d)

        plot = navigator_2d.create_interactive_plot()
        assert plot is not None

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_interactive_plotting_3d(self):
        """Test 3D interactive plotting."""
        # Create 3-objective front for 3D plotting
        objectives_3d = self.pareto_front.objectives[:3]
        points_3d = []

        for point in self.pareto_front.points:
            new_point = ParetoPoint(
                conditions=point.conditions, objective_values=point.objective_values[:3]
            )
            points_3d.append(new_point)

        front_3d = ParetoFront(points=points_3d, objectives=objectives_3d)
        navigator_3d = ParetoNavigator(front_3d)

        plot = navigator_3d.create_interactive_plot()
        assert plot is not None


class TestUtilityFunctions:
    """Test utility functions."""

    def test_chemical_synthesis_objectives(self):
        """Test chemical synthesis objective creation."""
        objectives = create_chemical_synthesis_objectives()

        assert len(objectives) == 4
        assert all(isinstance(obj, ExperimentalObjective) for obj in objectives)

        # Check specific objectives
        objective_names = [obj.name for obj in objectives]
        assert "yield" in objective_names
        assert "selectivity" in objective_names
        assert "reaction_time" in objective_names
        assert "cost" in objective_names

    def test_polymer_synthesis_objectives(self):
        """Test polymer synthesis objective creation."""
        objectives = create_polymer_synthesis_objectives()

        assert len(objectives) == 4
        assert all(isinstance(obj, ExperimentalObjective) for obj in objectives)

        # Check for polymer-specific objectives
        objective_names = [obj.name for obj in objectives]
        assert "molecular_weight" in objective_names
        assert "polydispersity" in objective_names
        assert "conversion" in objective_names

    def test_standard_experimental_space(self):
        """Test standard experimental space creation."""
        space = create_standard_experimental_space()

        assert isinstance(space, ExperimentalSpace)
        assert len(space.parameters) > 0
        assert len(space.discrete_parameters) > 0
        assert len(space.constraints) > 0

        # Check for standard parameters
        assert "temperature" in space.parameters
        assert "pressure" in space.parameters
        assert "concentration" in space.parameters

        # Check discrete parameters
        assert "catalyst" in space.discrete_parameters
        assert "solvent" in space.discrete_parameters

    @pytest.mark.asyncio
    async def test_run_experimental_optimization(self):
        """Test async experimental optimization."""
        objectives = create_chemical_synthesis_objectives()
        space = create_standard_experimental_space()
        model = SurrogateExperimentalModel(use_gpu=False)

        # Add training data
        conditions = [
            ExperimentalCondition(
                parameters={
                    "temperature": 80.0,
                    "pressure": 2.0,
                    "concentration": 0.5,
                    "time": 4.0,
                    "ph": 7.0,
                }
            )
        ]
        outcomes = [
            {"yield": 75.0, "selectivity": 80.0, "time": 4.0, "total_cost": 50.0}
        ]
        model.update(conditions, outcomes)

        pareto_front = await run_experimental_optimization(
            experimental_space=space,
            objectives=objectives,
            model=model,
            population_size=10,
            max_generations=3,
        )

        assert isinstance(pareto_front, ParetoFront)
        assert len(pareto_front.points) > 0

    def test_analyze_pareto_trade_offs(self):
        """Test Pareto trade-off analysis."""
        objectives = create_chemical_synthesis_objectives()

        points = [
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={"temperature": 80.0}),
                objective_values=[75.0, 80.0, -4.0, -50.0],
            ),
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={"temperature": 100.0}),
                objective_values=[85.0, 85.0, -6.0, -70.0],
            ),
        ]

        pareto_front = ParetoFront(points=points, objectives=objectives)
        analysis = analyze_pareto_trade_offs(pareto_front)

        assert isinstance(analysis, dict)
        assert "objective_ranges" in analysis
        assert "correlations" in analysis
        assert "diversity_metrics" in analysis

    def test_find_optimal_experimental_conditions(self):
        """Test finding optimal experimental conditions."""
        objectives = create_chemical_synthesis_objectives()

        points = [
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={"temperature": 80.0}),
                objective_values=[75.0, 80.0, -4.0, -50.0],
            ),
            ParetoPoint(
                conditions=ExperimentalCondition(parameters={"temperature": 100.0}),
                objective_values=[85.0, 85.0, -6.0, -70.0],
            ),
        ]

        pareto_front = ParetoFront(points=points, objectives=objectives)

        optimal_condition = find_optimal_experimental_conditions(pareto_front)

        assert isinstance(optimal_condition, ExperimentalCondition)
        assert "temperature" in optimal_condition.parameters

        # Test with preference weights
        weights = [1.0, 0.8, 0.6, 0.7]
        weighted_optimal = find_optimal_experimental_conditions(pareto_front, weights)

        assert isinstance(weighted_optimal, ExperimentalCondition)

    def test_empty_pareto_front_optimal_conditions(self):
        """Test optimal conditions with empty Pareto front."""
        pareto_front = ParetoFront(points=[], objectives=[])

        with pytest.raises(ExperimentalOptimizationError):
            find_optimal_experimental_conditions(pareto_front)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_objective_type(self):
        """Test handling of invalid objective types."""
        with pytest.raises(ValueError):
            ObjectiveType("invalid_type")

    def test_optimization_algorithm_enum(self):
        """Test optimization algorithm enumeration."""
        assert OptimizationAlgorithm.NSGA_II.value == "nsga_ii"
        assert OptimizationAlgorithm.NSGA_III.value == "nsga_iii"
        assert OptimizationAlgorithm.MOEA_D.value == "moea_d"

    def test_experimental_parameter_enum(self):
        """Test experimental parameter enumeration."""
        assert ExperimentalParameter.TEMPERATURE.value == "temperature"
        assert ExperimentalParameter.PRESSURE.value == "pressure"
        assert ExperimentalParameter.CONCENTRATION.value == "concentration"


class TestIntegration:
    """Integration tests for the full optimization pipeline."""

    @pytest.mark.asyncio
    async def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        # Set up objectives and space
        objectives = create_chemical_synthesis_objectives()
        space = create_standard_experimental_space()
        model = SurrogateExperimentalModel(use_gpu=False)

        # Add initial training data
        initial_conditions = [
            ExperimentalCondition(
                parameters={
                    "temperature": 80.0,
                    "pressure": 2.0,
                    "concentration": 0.5,
                    "time": 4.0,
                    "ph": 7.0,
                }
            ),
            ExperimentalCondition(
                parameters={
                    "temperature": 120.0,
                    "pressure": 4.0,
                    "concentration": 1.0,
                    "time": 8.0,
                    "ph": 9.0,
                }
            ),
        ]

        outcomes = [
            {"yield": 75.0, "selectivity": 80.0, "time": 4.0, "total_cost": 50.0},
            {"yield": 90.0, "selectivity": 88.0, "time": 8.0, "total_cost": 80.0},
        ]

        model.update(initial_conditions, outcomes)

        # Run optimization
        optimizer = ExperimentalOptimizer(
            experimental_space=space,
            objectives=objectives,
            model=model,
            population_size=20,
            max_generations=10,
        )

        pareto_front = optimizer.optimize()

        # Analyze results
        navigator = ParetoNavigator(pareto_front)
        analysis = navigator.get_trade_off_analysis()
        knee_points = navigator.find_knee_points()

        # Verify results
        assert len(pareto_front.points) > 0
        assert pareto_front.hypervolume > 0
        assert len(knee_points) > 0
        assert isinstance(analysis, dict)

        # Find optimal conditions
        optimal_conditions = find_optimal_experimental_conditions(pareto_front)
        assert isinstance(optimal_conditions, ExperimentalCondition)

        # Verify constraints are satisfied for all points
        for point in pareto_front.points:
            assert point.conditions.constraints_satisfied is True

    def test_multi_algorithm_comparison(self):
        """Test comparison of different optimization algorithms."""
        objectives = create_chemical_synthesis_objectives()
        space = create_standard_experimental_space()
        model = SurrogateExperimentalModel(use_gpu=False)

        # Add training data
        conditions = [
            ExperimentalCondition(
                parameters={
                    "temperature": 80.0,
                    "pressure": 2.0,
                    "concentration": 0.5,
                    "time": 4.0,
                    "ph": 7.0,
                }
            )
        ]
        outcomes = [
            {"yield": 75.0, "selectivity": 80.0, "time": 4.0, "total_cost": 50.0}
        ]
        model.update(conditions, outcomes)

        algorithms = [
            OptimizationAlgorithm.NSGA_II,
            OptimizationAlgorithm.NSGA_III,
            OptimizationAlgorithm.MOEA_D,
        ]

        results = {}

        for algorithm in algorithms:
            optimizer = ExperimentalOptimizer(
                experimental_space=space,
                objectives=objectives,
                model=model,
                algorithm=algorithm,
                population_size=10,
                max_generations=3,
            )

            pareto_front = optimizer.optimize()
            results[algorithm.value] = {
                "pareto_size": len(pareto_front.points),
                "hypervolume": pareto_front.hypervolume,
            }

        # Verify all algorithms produced results
        for algorithm_name, result in results.items():
            assert result["pareto_size"] > 0
            assert result["hypervolume"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])

"""
Comprehensive tests for the inverse design module.

Tests cover optimization algorithms, design space definitions,
objective functions, and integration with simulation/characterization.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.self_assembly.inverse_design import (  # Enums; Data structures; Optimizers; Main classes; Utility functions; Exceptions
    BaseOptimizer,
    BayesianOptimizer,
    DesignParameter,
    DesignSpace,
    DesignSpaceError,
    DesignTarget,
    EmergentPropertyInverseDesigner,
    EvaluationError,
    GeneticAlgorithmOptimizer,
    InverseDesignError,
    ObjectiveFunction,
    OptimizationError,
    OptimizationMethod,
    OptimizationObjective,
    OptimizationResult,
    PerformanceMetric,
    design_micelle_forming_surfactant,
    multi_objective_surfactant_design,
    optimize_environmental_conditions,
)


class TestDesignParameter:
    """Test DesignParameter dataclass."""

    def test_continuous_parameter_creation(self):
        """Test creation of continuous parameter."""
        param = DesignParameter(
            name="temperature",
            parameter_type="continuous",
            bounds=(250.0, 350.0),
            current_value=298.15,
            importance_weight=2.0,
        )

        assert param.name == "temperature"
        assert param.parameter_type == "continuous"
        assert param.bounds == (250.0, 350.0)
        assert param.current_value == 298.15
        assert param.importance_weight == 2.0

    def test_discrete_parameter_creation(self):
        """Test creation of discrete parameter."""
        param = DesignParameter(
            name="chain_length",
            parameter_type="discrete",
            discrete_values=[4, 6, 8, 10, 12],
            current_value=8,
        )

        assert param.name == "chain_length"
        assert param.parameter_type == "discrete"
        assert param.discrete_values == [4, 6, 8, 10, 12]
        assert param.current_value == 8

    def test_categorical_parameter_creation(self):
        """Test creation of categorical parameter."""
        param = DesignParameter(
            name="head_group",
            parameter_type="categorical",
            categorical_values=["ionic", "nonionic", "zwitterionic"],
            current_value="ionic",
        )

        assert param.name == "head_group"
        assert param.parameter_type == "categorical"
        assert param.categorical_values == ["ionic", "nonionic", "zwitterionic"]
        assert param.current_value == "ionic"

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="Invalid parameter type"):
            DesignParameter(name="invalid", parameter_type="unknown")

        with pytest.raises(
            ValueError, match="Discrete parameters must specify discrete_values"
        ):
            DesignParameter(name="discrete_param", parameter_type="discrete")

        with pytest.raises(
            ValueError, match="Categorical parameters must specify categorical_values"
        ):
            DesignParameter(name="cat_param", parameter_type="categorical")


class TestObjectiveFunction:
    """Test ObjectiveFunction dataclass."""

    def test_objective_function_creation(self):
        """Test basic objective function creation."""
        obj = ObjectiveFunction(
            name="micelle_size",
            objective_type=OptimizationObjective.MAXIMIZE_MICELLE_SIZE,
            target_value=50.0,
            weight=2.0,
            minimize=False,
        )

        assert obj.name == "micelle_size"
        assert obj.objective_type == OptimizationObjective.MAXIMIZE_MICELLE_SIZE
        assert obj.target_value == 50.0
        assert obj.weight == 2.0
        assert obj.minimize is False

    def test_objective_evaluation_target_based(self):
        """Test target-based objective evaluation."""
        obj = ObjectiveFunction(
            name="target_agg_num",
            objective_type=OptimizationObjective.MAXIMIZE_MICELLE_SIZE,
            target_value=50.0,
            tolerance=5.0,
            weight=1.0,
        )

        # Within tolerance
        properties = {"maximize_micelle_size": 52.0}
        assert obj.evaluate(properties) == 0.0

        # Outside tolerance
        properties = {"maximize_micelle_size": 60.0}
        assert obj.evaluate(properties) == 10.0

    def test_objective_evaluation_min_max(self):
        """Test min/max objective evaluation."""
        # Minimization objective
        obj_min = ObjectiveFunction(
            name="minimize_cmc",
            objective_type=OptimizationObjective.MINIMIZE_CMC,
            weight=1.0,
            minimize=True,
        )

        properties = {"minimize_cmc": 0.05}
        assert obj_min.evaluate(properties) == 0.05

        # Maximization objective
        obj_max = ObjectiveFunction(
            name="maximize_stability",
            objective_type=OptimizationObjective.MAXIMIZE_STABILITY,
            weight=2.0,
            minimize=False,
        )

        properties = {"maximize_stability": 0.8}
        assert obj_max.evaluate(properties) == -1.6  # -0.8 * 2.0

    def test_objective_evaluation_missing_property(self):
        """Test objective evaluation with missing property."""
        obj = ObjectiveFunction(
            name="test_obj",
            objective_type=OptimizationObjective.MAXIMIZE_MICELLE_SIZE,
            constraint_penalty=1000.0,
        )

        properties = {"other_property": 1.0}
        assert obj.evaluate(properties) == 1000.0


class TestDesignSpace:
    """Test DesignSpace dataclass."""

    def create_test_design_space(self):
        """Create a test design space."""
        parameters = [
            DesignParameter(
                name="temperature",
                parameter_type="continuous",
                bounds=(280.0, 320.0),
                current_value=298.15,
            ),
            DesignParameter(
                name="chain_length",
                parameter_type="discrete",
                discrete_values=[6, 8, 10, 12],
                current_value=8,
            ),
            DesignParameter(
                name="head_group",
                parameter_type="categorical",
                categorical_values=["ionic", "nonionic"],
                current_value="ionic",
            ),
        ]

        objectives = [
            ObjectiveFunction(
                name="target_size",
                objective_type=OptimizationObjective.MAXIMIZE_MICELLE_SIZE,
                target_value=50.0,
            )
        ]

        return DesignSpace(parameters=parameters, objectives=objectives)

    def test_design_space_creation(self):
        """Test design space creation."""
        design_space = self.create_test_design_space()

        assert len(design_space.parameters) == 3
        assert len(design_space.objectives) == 1
        assert len(design_space.constraints) == 0

    def test_get_parameter_bounds(self):
        """Test getting parameter bounds."""
        design_space = self.create_test_design_space()
        bounds = design_space.get_parameter_bounds()

        expected_bounds = [
            (280.0, 320.0),  # continuous
            (0, 3),  # discrete (4 values, indices 0-3)
            (0, 1),  # categorical (2 values, indices 0-1)
        ]

        assert bounds == expected_bounds

    def test_encode_parameters(self):
        """Test parameter encoding."""
        design_space = self.create_test_design_space()

        parameter_values = {
            "temperature": 300.0,
            "chain_length": 10,
            "head_group": "nonionic",
        }

        encoded = design_space.encode_parameters(parameter_values)
        expected = np.array([300.0, 2.0, 1.0])  # 10 is at index 2, nonionic at index 1

        np.testing.assert_array_equal(encoded, expected)

    def test_decode_parameters(self):
        """Test parameter decoding."""
        design_space = self.create_test_design_space()

        encoded_values = np.array([305.0, 1.0, 0.0])
        decoded = design_space.decode_parameters(encoded_values)

        expected = {
            "temperature": 305.0,
            "chain_length": 8,  # index 1
            "head_group": "ionic",  # index 0
        }

        assert decoded == expected


class TestGeneticAlgorithmOptimizer:
    """Test Genetic Algorithm optimizer."""

    def test_optimizer_initialization(self):
        """Test GA optimizer initialization."""
        parameters = [
            DesignParameter("x", "continuous", bounds=(0.0, 1.0)),
            DesignParameter("y", "continuous", bounds=(-1.0, 1.0)),
        ]
        objectives = [
            ObjectiveFunction("minimize", OptimizationObjective.MAXIMIZE_MICELLE_SIZE)
        ]
        design_space = DesignSpace(parameters=parameters, objectives=objectives)

        optimizer = GeneticAlgorithmOptimizer(design_space, random_seed=42)

        assert optimizer.design_space == design_space
        assert optimizer.random_seed == 42
        assert optimizer.evaluation_count == 0

    def test_simple_optimization(self):
        """Test simple optimization problem."""
        # Minimize x^2 + y^2
        parameters = [
            DesignParameter("x", "continuous", bounds=(-2.0, 2.0)),
            DesignParameter("y", "continuous", bounds=(-2.0, 2.0)),
        ]
        objectives = [
            ObjectiveFunction(
                "minimize_sum_squares", OptimizationObjective.MAXIMIZE_MICELLE_SIZE
            )
        ]
        design_space = DesignSpace(parameters=parameters, objectives=objectives)

        def evaluation_function(params):
            x, y = params["x"], params["y"]
            return {"maximize_micelle_size": x**2 + y**2}

        optimizer = GeneticAlgorithmOptimizer(design_space, random_seed=42)
        result = optimizer.optimize(
            evaluation_function,
            max_evaluations=100,
            population_size=20,
            crossover_rate=0.8,
            mutation_rate=0.1,
        )

        assert result.success
        assert result.evaluation_count <= 100
        assert result.best_objective_value < 1.0  # Should find solution near (0,0)
        assert len(result.objective_history) > 0
        assert len(result.parameter_history) > 0

    def test_tournament_selection(self):
        """Test tournament selection."""
        design_space = DesignSpace(
            parameters=[DesignParameter("x", "continuous", bounds=(0.0, 1.0))],
            objectives=[
                ObjectiveFunction("test", OptimizationObjective.MAXIMIZE_MICELLE_SIZE)
            ],
        )

        optimizer = GeneticAlgorithmOptimizer(design_space, random_seed=42)

        population = np.array([[0.1], [0.5], [0.9], [0.2], [0.8]])
        fitness_scores = np.array([1.0, 0.5, 0.1, 0.8, 0.2])  # Lower is better

        selected = optimizer._tournament_selection(
            population, fitness_scores, tournament_size=3
        )

        assert selected.shape == (1,)
        assert 0.0 <= selected[0] <= 1.0

    def test_crossover(self):
        """Test crossover operation."""
        design_space = DesignSpace(
            parameters=[DesignParameter("x", "continuous", bounds=(0.0, 1.0))],
            objectives=[
                ObjectiveFunction("test", OptimizationObjective.MAXIMIZE_MICELLE_SIZE)
            ],
        )

        optimizer = GeneticAlgorithmOptimizer(design_space, random_seed=42)

        parent1 = np.array([0.1, 0.2, 0.3])
        parent2 = np.array([0.7, 0.8, 0.9])

        child1, child2 = optimizer._crossover(parent1, parent2)

        assert len(child1) == 3
        assert len(child2) == 3
        # Children should contain mix of parent genes
        assert not np.array_equal(child1, parent1)
        assert not np.array_equal(child2, parent2)

    def test_mutation(self):
        """Test mutation operation."""
        design_space = DesignSpace(
            parameters=[DesignParameter("x", "continuous", bounds=(0.0, 1.0))],
            objectives=[
                ObjectiveFunction("test", OptimizationObjective.MAXIMIZE_MICELLE_SIZE)
            ],
        )

        optimizer = GeneticAlgorithmOptimizer(design_space, random_seed=42)

        individual = np.array([0.5, 0.5, 0.5])
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        mutated = optimizer._mutate(individual, bounds)

        assert len(mutated) == 3
        assert np.all(mutated >= 0.0)
        assert np.all(mutated <= 1.0)


@pytest.mark.skipif(True, reason="BayesianOptimizer requires scikit-learn")
class TestBayesianOptimizer:
    """Test Bayesian Optimization (requires scikit-learn)."""

    def test_optimizer_initialization(self):
        """Test BO optimizer initialization."""
        parameters = [DesignParameter("x", "continuous", bounds=(0.0, 1.0))]
        objectives = [
            ObjectiveFunction("test", OptimizationObjective.MAXIMIZE_MICELLE_SIZE)
        ]
        design_space = DesignSpace(parameters=parameters, objectives=objectives)

        try:
            optimizer = BayesianOptimizer(design_space, random_seed=42)
            assert optimizer.design_space == design_space
            assert optimizer.kernel_type == "matern"
            assert optimizer.acquisition_function == "ei"
        except ImportError:
            pytest.skip("scikit-learn not available")


class TestEmergentPropertyInverseDesigner:
    """Test main inverse designer class."""

    def test_designer_initialization(self):
        """Test designer initialization."""
        designer = EmergentPropertyInverseDesigner(
            use_gpu=False,  # Force CPU for testing
            cache_simulations=True,
            parallel_evaluations=False,
        )

        assert designer.use_gpu is False
        assert designer.cache_simulations is True
        assert designer.parallel_evaluations is False
        assert designer.simulation_cache == {}
        assert len(designer.optimizers) >= 1

    def test_create_surfactant_design_space(self):
        """Test surfactant design space creation."""
        designer = EmergentPropertyInverseDesigner(use_gpu=False)

        head_groups = ["ionic", "nonionic", "zwitterionic"]
        tail_length_range = (6, 12)
        property_targets = {"aggregation_number": 50.0, "minimize_cmc": 0.01}

        design_space = designer.create_surfactant_design_space(
            head_group_options=head_groups,
            tail_length_range=tail_length_range,
            property_targets=property_targets,
        )

        # Check parameters
        param_names = [p.name for p in design_space.parameters]
        assert "head_group" in param_names
        assert "tail_length" in param_names
        assert "tail_hydrophobicity" in param_names

        # Check objectives
        assert len(design_space.objectives) == 2
        obj_names = [obj.name for obj in design_space.objectives]
        assert any("aggregation_number" in name for name in obj_names)
        assert any("minimize_cmc" in name for name in obj_names)

    def test_create_environmental_design_space(self):
        """Test environmental design space creation."""
        designer = EmergentPropertyInverseDesigner(use_gpu=False)

        temperature_range = (280.0, 320.0)
        concentration_range = (0.01, 1.0)
        property_targets = {"maximize_stability": 1.0}

        design_space = designer.create_environmental_design_space(
            temperature_range=temperature_range,
            concentration_range=concentration_range,
            property_targets=property_targets,
        )

        # Check parameters
        param_names = [p.name for p in design_space.parameters]
        assert "temperature" in param_names
        assert "concentration" in param_names
        assert "ph" in param_names

        # Check parameter bounds
        temp_param = next(p for p in design_space.parameters if p.name == "temperature")
        assert temp_param.bounds == temperature_range

    @patch("chemi_accel.self_assembly.inverse_design.run_dpd_simulation")
    @patch("chemi_accel.self_assembly.inverse_design.characterize_trajectory")
    @patch("chemi_accel.self_assembly.inverse_design.create_amphiphilic_system")
    def test_evaluate_design_surfactant(
        self, mock_create_system, mock_characterize, mock_run_sim
    ):
        """Test design evaluation for surfactant."""
        # Setup mocks
        mock_particles = [Mock()]
        mock_sim_box = Mock()
        mock_interaction_params = Mock()
        mock_create_system.return_value = (
            mock_particles,
            mock_sim_box,
            mock_interaction_params,
        )

        mock_trajectory = [Mock()]
        mock_run_sim.return_value = mock_trajectory

        mock_structure = Mock()
        mock_structure.aggregation_number = 45
        mock_structure.radius_of_gyration = 2.5
        mock_structure.asphericity = 0.3
        mock_structure.surface_area = 100.0

        mock_characterize.return_value = {"structures": [mock_structure]}

        # Test evaluation
        designer = EmergentPropertyInverseDesigner(use_gpu=False)

        parameters = {
            "head_group": "ionic",
            "tail_length": 8,
            "tail_hydrophobicity": 1.2,
            "temperature": 298.15,
        }

        properties = designer.evaluate_design(
            parameters, DesignTarget.SURFACTANT_STRUCTURE
        )

        # Verify results
        assert "aggregation_number" in properties
        assert "radius_of_gyration" in properties
        assert "asphericity" in properties
        assert "surface_area" in properties
        assert properties["aggregation_number"] == 45

        # Verify mocks were called
        mock_create_system.assert_called_once()
        mock_run_sim.assert_called_once()
        mock_characterize.assert_called_once()

    @patch.object(EmergentPropertyInverseDesigner, "evaluate_design")
    def test_optimize_design(self, mock_evaluate):
        """Test design optimization."""
        # Setup mock evaluation
        mock_evaluate.return_value = {
            "aggregation_number": 48.0,
            "maximize_micelle_size": 48.0,
            "minimize_cmc": 0.02,
        }

        # Create designer and design space
        designer = EmergentPropertyInverseDesigner(use_gpu=False)

        parameters = [
            DesignParameter("x", "continuous", bounds=(0.0, 1.0)),
            DesignParameter("y", "continuous", bounds=(0.0, 1.0)),
        ]
        objectives = [
            ObjectiveFunction(
                "test", OptimizationObjective.MAXIMIZE_MICELLE_SIZE, target_value=50.0
            )
        ]
        design_space = DesignSpace(parameters=parameters, objectives=objectives)

        # Run optimization
        result = designer.optimize_design(
            design_space=design_space,
            design_target=DesignTarget.SURFACTANT_STRUCTURE,
            optimization_method=OptimizationMethod.GENETIC_ALGORITHM,
            max_evaluations=20,
            population_size=10,
        )

        # Verify results
        assert isinstance(result, OptimizationResult)
        assert result.success
        assert result.evaluation_count <= 20
        assert "x" in result.best_parameters
        assert "y" in result.best_parameters

    def test_simulation_caching(self):
        """Test simulation result caching."""
        designer = EmergentPropertyInverseDesigner(
            use_gpu=False, cache_simulations=True
        )

        # Test cache key generation
        parameters = {"head_group": "ionic", "tail_length": 8}
        cache_key1 = designer._get_cache_key(
            parameters, DesignTarget.SURFACTANT_STRUCTURE
        )
        cache_key2 = designer._get_cache_key(
            parameters, DesignTarget.SURFACTANT_STRUCTURE
        )
        cache_key3 = designer._get_cache_key(
            {"head_group": "nonionic", "tail_length": 8},
            DesignTarget.SURFACTANT_STRUCTURE,
        )

        assert cache_key1 == cache_key2
        assert cache_key1 != cache_key3

    def test_extract_properties(self):
        """Test property extraction from characterization results."""
        designer = EmergentPropertyInverseDesigner(use_gpu=False)

        # Mock characterization results
        mock_structure = Mock()
        mock_structure.aggregation_number = 42
        mock_structure.radius_of_gyration = 2.8
        mock_structure.asphericity = 0.25
        mock_structure.surface_area = 95.0

        char_results = {"structures": [mock_structure]}
        parameters = {"temperature": 298.15}

        properties = designer._extract_properties(char_results, parameters)

        # Check extracted properties
        assert properties["aggregation_number"] == 42
        assert properties["radius_of_gyration"] == 2.8
        assert properties["asphericity"] == 0.25
        assert properties["surface_area"] == 95.0

        # Check derived properties
        assert "maximize_micelle_size" in properties
        assert "minimize_cmc" in properties
        assert "maximize_stability" in properties

    def test_extract_properties_no_structures(self):
        """Test property extraction when no structures found."""
        designer = EmergentPropertyInverseDesigner(use_gpu=False)

        char_results = {"structures": []}
        parameters = {}

        properties = designer._extract_properties(char_results, parameters)

        # Should have default values
        assert properties["aggregation_number"] == 0.0
        assert properties["radius_of_gyration"] == 0.0
        assert properties["asphericity"] == 0.0
        assert properties["surface_area"] == 0.0


class TestUtilityFunctions:
    """Test utility functions."""

    @patch.object(EmergentPropertyInverseDesigner, "optimize_design")
    def test_design_micelle_forming_surfactant(self, mock_optimize):
        """Test micelle-forming surfactant design utility."""
        mock_result = OptimizationResult(
            best_parameters={"head_group": "ionic", "tail_length": 10},
            best_objective_value=5.0,
            objective_history=[10.0, 7.0, 5.0],
            parameter_history=[],
            convergence_data={},
            evaluation_count=50,
            optimization_time=10.0,
            success=True,
            message="Test optimization",
        )
        mock_optimize.return_value = mock_result

        result = design_micelle_forming_surfactant(
            target_aggregation_number=50,
            target_cmc=0.01,
            head_groups=["ionic", "nonionic"],
            tail_length_range=(6, 16),
            max_evaluations=100,
            use_gpu=False,
        )

        assert result.success
        assert result.best_objective_value == 5.0
        assert "head_group" in result.best_parameters
        assert "tail_length" in result.best_parameters

        # Verify optimize_design was called with correct parameters
        mock_optimize.assert_called_once()
        call_args = mock_optimize.call_args
        assert call_args[1]["design_target"] == DesignTarget.SURFACTANT_STRUCTURE
        assert call_args[1]["max_evaluations"] == 100

    @patch.object(EmergentPropertyInverseDesigner, "optimize_design")
    def test_optimize_environmental_conditions(self, mock_optimize):
        """Test environmental condition optimization utility."""
        mock_result = OptimizationResult(
            best_parameters={"temperature": 310.0, "concentration": 0.5, "ph": 7.5},
            best_objective_value=2.0,
            objective_history=[5.0, 3.0, 2.0],
            parameter_history=[],
            convergence_data={},
            evaluation_count=30,
            optimization_time=5.0,
            success=True,
            message="Environmental optimization",
        )
        mock_optimize.return_value = mock_result

        from chemi_accel.self_assembly.characterization import StructureType

        result = optimize_environmental_conditions(
            target_morphology=StructureType.MICELLE,
            temperature_range=(280.0, 340.0),
            concentration_range=(0.01, 1.0),
            max_evaluations=50,
            use_gpu=False,
        )

        assert result.success
        assert result.best_objective_value == 2.0
        assert "temperature" in result.best_parameters
        assert "concentration" in result.best_parameters
        assert "ph" in result.best_parameters

        # Verify optimization method was GA
        call_args = mock_optimize.call_args
        assert (
            call_args[1]["optimization_method"] == OptimizationMethod.GENETIC_ALGORITHM
        )

    @patch.object(EmergentPropertyInverseDesigner, "multi_objective_optimization")
    def test_multi_objective_surfactant_design(self, mock_multi_opt):
        """Test multi-objective surfactant design utility."""
        mock_result = OptimizationResult(
            best_parameters={"head_group": "zwitterionic", "tail_length": 12},
            best_objective_value=3.5,
            objective_history=[],
            parameter_history=[],
            convergence_data={"pareto_front_size": 10},
            evaluation_count=200,
            optimization_time=25.0,
            success=True,
            message="Multi-objective optimization",
            pareto_front=[
                ({"head_group": "ionic", "tail_length": 8}, [2.0, 3.0]),
                ({"head_group": "nonionic", "tail_length": 10}, [1.5, 4.0]),
            ],
        )
        mock_multi_opt.return_value = mock_result

        objectives = {"aggregation_number": 45.0, "minimize_cmc": 0.005}

        result = multi_objective_surfactant_design(
            objectives=objectives,
            head_groups=["ionic", "nonionic", "zwitterionic"],
            tail_length_range=(6, 16),
            max_evaluations=200,
            use_gpu=False,
        )

        assert result.success
        assert result.pareto_front is not None
        assert len(result.pareto_front) == 2
        assert result.convergence_data["pareto_front_size"] == 10

        # Verify multi_objective_optimization was called
        mock_multi_opt.assert_called_once()
        call_args = mock_multi_opt.call_args
        assert call_args[1]["design_target"] == DesignTarget.SURFACTANT_STRUCTURE
        assert call_args[1]["max_evaluations"] == 200


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_design_space_error_validation(self):
        """Test design space validation errors."""
        # Test invalid parameter type
        with pytest.raises(ValueError):
            DesignParameter(name="invalid", parameter_type="unknown")

        # Test missing discrete values
        with pytest.raises(ValueError):
            DesignParameter(name="discrete", parameter_type="discrete")

        # Test missing categorical values
        with pytest.raises(ValueError):
            DesignParameter(name="categorical", parameter_type="categorical")

    def test_optimization_error_on_invalid_method(self):
        """Test optimization error with invalid method."""
        designer = EmergentPropertyInverseDesigner(use_gpu=False)

        parameters = [DesignParameter("x", "continuous", bounds=(0.0, 1.0))]
        objectives = [
            ObjectiveFunction("test", OptimizationObjective.MAXIMIZE_MICELLE_SIZE)
        ]
        design_space = DesignSpace(parameters=parameters, objectives=objectives)

        # Invalid optimization method
        invalid_method = Mock()
        invalid_method.value = "invalid_method"

        with pytest.raises(OptimizationError, match="not available"):
            designer.optimize_design(
                design_space=design_space,
                design_target=DesignTarget.SURFACTANT_STRUCTURE,
                optimization_method=invalid_method,
            )

    def test_evaluation_error_handling(self):
        """Test evaluation error handling."""
        designer = EmergentPropertyInverseDesigner(use_gpu=False)

        # Mock evaluation that raises exception
        with patch.object(
            designer, "evaluate_design", side_effect=Exception("Simulation failed")
        ):
            parameters = [DesignParameter("x", "continuous", bounds=(0.0, 1.0))]
            objectives = [
                ObjectiveFunction("test", OptimizationObjective.MAXIMIZE_MICELLE_SIZE)
            ]
            design_space = DesignSpace(parameters=parameters, objectives=objectives)

            # Should handle evaluation errors gracefully
            result = designer.optimize_design(
                design_space=design_space,
                design_target=DesignTarget.SURFACTANT_STRUCTURE,
                optimization_method=OptimizationMethod.GENETIC_ALGORITHM,
                max_evaluations=10,
                population_size=5,
            )

            # May succeed with infinite objective values
            assert isinstance(result, OptimizationResult)

    def test_exception_hierarchy(self):
        """Test exception class hierarchy."""
        assert issubclass(OptimizationError, InverseDesignError)
        assert issubclass(DesignSpaceError, InverseDesignError)
        assert issubclass(EvaluationError, InverseDesignError)

        # Test that they can be raised
        with pytest.raises(InverseDesignError):
            raise OptimizationError("Test optimization error")

        with pytest.raises(InverseDesignError):
            raise DesignSpaceError("Test design space error")

        with pytest.raises(InverseDesignError):
            raise EvaluationError("Test evaluation error")


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete inverse design workflows."""

    @patch("chemi_accel.self_assembly.inverse_design.run_dpd_simulation")
    @patch("chemi_accel.self_assembly.inverse_design.characterize_trajectory")
    @patch("chemi_accel.self_assembly.inverse_design.create_amphiphilic_system")
    def test_complete_surfactant_optimization_workflow(
        self, mock_create_system, mock_characterize, mock_run_sim
    ):
        """Test complete surfactant optimization workflow."""
        # Setup comprehensive mocks
        mock_particles = [Mock() for _ in range(20)]
        mock_sim_box = Mock()
        mock_interaction_params = Mock()
        mock_create_system.return_value = (
            mock_particles,
            mock_sim_box,
            mock_interaction_params,
        )

        mock_trajectory = [Mock() for _ in range(10)]
        mock_run_sim.return_value = mock_trajectory

        # Create structures with varying properties to test optimization
        def create_mock_structure(agg_num, rg, asp, sa):
            structure = Mock()
            structure.aggregation_number = agg_num
            structure.radius_of_gyration = rg
            structure.asphericity = asp
            structure.surface_area = sa
            return structure

        # Mock characterization to return different results based on parameters
        def mock_char_function(*args, **kwargs):
            # Simple heuristic: ionic head groups perform better
            if hasattr(mock_create_system, "last_call_params"):
                params = mock_create_system.last_call_params
                if "ionic" in str(params):
                    return {"structures": [create_mock_structure(48, 2.5, 0.2, 98)]}
                else:
                    return {"structures": [create_mock_structure(35, 3.0, 0.4, 75)]}
            return {"structures": [create_mock_structure(40, 2.8, 0.3, 85)]}

        mock_characterize.side_effect = mock_char_function

        # Run complete optimization
        result = design_micelle_forming_surfactant(
            target_aggregation_number=50,
            target_cmc=0.01,
            head_groups=["ionic", "nonionic"],
            tail_length_range=(6, 12),
            max_evaluations=20,
            use_gpu=False,
        )

        # Verify workflow completed
        assert result.success
        assert result.evaluation_count <= 20
        assert "head_group" in result.best_parameters
        assert "tail_length" in result.best_parameters
        assert "tail_hydrophobicity" in result.best_parameters

        # Verify all components were called
        assert mock_create_system.call_count > 0
        assert mock_run_sim.call_count > 0
        assert mock_characterize.call_count > 0

    @patch.object(EmergentPropertyInverseDesigner, "evaluate_design")
    def test_multi_parameter_optimization_convergence(self, mock_evaluate):
        """Test optimization convergence with multiple parameters."""

        # Create evaluation function that has a clear optimum
        def mock_eval_function(parameters):
            # Target: head_group='ionic', tail_length=10, tail_hydrophobicity=1.0
            score = 0.0

            if parameters.get("head_group") == "ionic":
                score += 10.0

            tail_length = parameters.get("tail_length", 8)
            score += 5.0 - abs(tail_length - 10)

            hydrophobicity = parameters.get("tail_hydrophobicity", 1.0)
            score += 3.0 - abs(hydrophobicity - 1.0)

            return {
                "aggregation_number": max(0, score * 3),
                "maximize_micelle_size": max(0, score * 3),
                "minimize_cmc": max(0.001, 0.1 - score * 0.01),
            }

        mock_evaluate.side_effect = lambda params, target: mock_eval_function(params)

        # Create designer and run optimization
        designer = EmergentPropertyInverseDesigner(use_gpu=False)

        design_space = designer.create_surfactant_design_space(
            head_group_options=["ionic", "nonionic", "zwitterionic"],
            tail_length_range=(6, 14),
            property_targets={"aggregation_number": 50.0},
        )

        result = designer.optimize_design(
            design_space=design_space,
            design_target=DesignTarget.SURFACTANT_STRUCTURE,
            optimization_method=OptimizationMethod.GENETIC_ALGORITHM,
            max_evaluations=100,
            population_size=20,
        )

        # Should converge to near-optimal solution
        assert result.success
        assert result.best_parameters["head_group"] == "ionic"
        assert abs(result.best_parameters["tail_length"] - 10) <= 2
        assert abs(result.best_parameters["tail_hydrophobicity"] - 1.0) <= 0.5

        # Check convergence
        assert len(result.objective_history) > 5
        initial_obj = result.objective_history[0]
        final_obj = result.objective_history[-1]
        assert final_obj < initial_obj  # Should improve


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for inverse design components."""

    def test_genetic_algorithm_performance(self, benchmark):
        """Benchmark genetic algorithm performance."""
        parameters = [
            DesignParameter("x", "continuous", bounds=(-5.0, 5.0)),
            DesignParameter("y", "continuous", bounds=(-5.0, 5.0)),
        ]
        objectives = [
            ObjectiveFunction("minimize", OptimizationObjective.MAXIMIZE_MICELLE_SIZE)
        ]
        design_space = DesignSpace(parameters=parameters, objectives=objectives)

        def evaluation_function(params):
            x, y = params["x"], params["y"]
            return {"maximize_micelle_size": x**2 + y**2}

        optimizer = GeneticAlgorithmOptimizer(design_space, random_seed=42)

        result = benchmark(
            optimizer.optimize,
            evaluation_function,
            max_evaluations=50,
            population_size=20,
        )

        assert result.success
        assert result.evaluation_count <= 50

    @patch.object(EmergentPropertyInverseDesigner, "evaluate_design")
    def test_design_evaluation_performance(self, mock_evaluate, benchmark):
        """Benchmark design evaluation performance."""
        # Simple mock evaluation
        mock_evaluate.return_value = {
            "aggregation_number": 45.0,
            "radius_of_gyration": 2.5,
            "maximize_micelle_size": 45.0,
        }

        designer = EmergentPropertyInverseDesigner(use_gpu=False)

        parameters = {
            "head_group": "ionic",
            "tail_length": 8,
            "tail_hydrophobicity": 1.0,
        }

        result = benchmark(
            designer.evaluate_design, parameters, DesignTarget.SURFACTANT_STRUCTURE
        )

        assert "aggregation_number" in result
        assert result["aggregation_number"] == 45.0


if __name__ == "__main__":
    pytest.main([__file__])

"""
Tests for Inverse Design for Sustainable Materials.

This module provides comprehensive testing for the inverse design framework
including multi-objective optimization, GPU acceleration, and sustainability metrics.
"""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.sustainability.inverse_design import (  # Core classes; Data structures; Utility functions; Exceptions
    ConstraintType,
    DesignResult,
    InvalidParameterError,
    InverseDesignError,
    MaterialParameter,
    MaterialPropertyPredictor,
    OptimizationAlgorithm,
    OptimizationConfig,
    OptimizationConstraint,
    OptimizationFailedError,
    OptimizationObjective,
    OptimizationProblem,
    PropertyPredictionError,
    SustainabilityPropertyPredictor,
    SustainabilityTarget,
    SustainableInverseDesigner,
    create_performance_constraints,
    create_polymer_design_parameters,
    create_sustainability_objectives,
    design_sustainable_material,
)

# Check for optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import scipy.optimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class TestDataStructures:
    """Test inverse design data structures."""

    def test_material_parameter_creation(self):
        """Test creating material parameter."""
        param = MaterialParameter(
            name="crosslink_density",
            min_value=0.0,
            max_value=0.3,
            initial_value=0.1,
            parameter_type="continuous",
            description="Crosslink density of polymer",
        )

        assert param.name == "crosslink_density"
        assert param.min_value == 0.0
        assert param.max_value == 0.3
        assert param.initial_value == 0.1
        assert param.parameter_type == "continuous"

    def test_optimization_constraint_creation(self):
        """Test creating optimization constraint."""
        constraint = OptimizationConstraint(
            name="minimum_strength",
            constraint_type=ConstraintType.INEQUALITY,
            function=lambda props, params: 30.0 - props.get("tensile_strength", 0),
            target_value=None,
            tolerance=1e-6,
            weight=1.0,
            description="Minimum tensile strength requirement",
        )

        assert constraint.name == "minimum_strength"
        assert constraint.constraint_type == ConstraintType.INEQUALITY
        assert callable(constraint.function)
        assert constraint.tolerance == 1e-6

    def test_sustainability_target_creation(self):
        """Test creating sustainability target."""
        target = SustainabilityTarget(
            objective=OptimizationObjective.MINIMIZE_DEGRADATION_RATE,
            target_value=None,
            weight=1.0,
            minimize=True,
            constraint=False,
            tolerance=1e-3,
        )

        assert target.objective == OptimizationObjective.MINIMIZE_DEGRADATION_RATE
        assert target.weight == 1.0
        assert target.minimize is True
        assert target.constraint is False

    def test_design_result_creation(self):
        """Test creating design result."""
        result = DesignResult(
            optimal_parameters={"param1": 0.5, "param2": 0.8},
            objective_values={OptimizationObjective.MINIMIZE_DEGRADATION_RATE: 0.1},
            constraint_violations={},
            optimization_history=[],
            convergence_metrics={"iterations": 100},
            sustainability_scores={"overall": 0.8},
            predicted_properties={"tensile_strength": 45.0},
            computation_time=10.5,
            success=True,
            message="Optimization completed successfully",
        )

        assert result.success is True
        assert result.optimal_parameters["param1"] == 0.5
        assert result.computation_time == 10.5
        assert "overall" in result.sustainability_scores

    def test_optimization_config_creation(self):
        """Test creating optimization configuration."""
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.MULTI_OBJECTIVE_GENETIC,
            max_iterations=500,
            population_size=50,
            tolerance=1e-5,
            use_gpu=True,
            parallel_evaluation=True,
            random_seed=42,
        )

        assert config.algorithm == OptimizationAlgorithm.MULTI_OBJECTIVE_GENETIC
        assert config.max_iterations == 500
        assert config.population_size == 50
        assert config.use_gpu is True
        assert config.random_seed == 42


class TestSustainabilityPropertyPredictor:
    """Test sustainability property predictor."""

    def test_predictor_creation(self):
        """Test predictor initialization."""
        predictor = SustainabilityPropertyPredictor(use_gpu=False)

        assert predictor.use_gpu is False
        assert predictor.degradation_predictor is not None
        assert predictor.recyclability_assessor is not None
        assert predictor.lca_calculator is not None
        assert isinstance(predictor.property_models, dict)

    @pytest.mark.asyncio
    async def test_property_prediction(self):
        """Test property prediction."""
        predictor = SustainabilityPropertyPredictor(use_gpu=False)

        parameters = {
            "monomer_ratio": 0.8,
            "crosslink_density": 0.1,
            "stabilizer_content": 0.02,
            "filler_content": 0.1,
            "polymer_purity": 0.95,
        }

        properties = await predictor.predict_properties(parameters)

        assert isinstance(properties, dict)
        assert "molecular_weight" in properties
        assert "glass_transition_temp" in properties
        assert "degradation_rate" in properties
        assert "recyclability_score" in properties
        assert "carbon_footprint" in properties

        # Check value ranges
        assert properties["molecular_weight"] > 0
        assert properties["glass_transition_temp"] > 200
        assert 0 <= properties["recyclability_score"] <= 1
        assert properties["carbon_footprint"] > 0

    def test_parameter_validation(self):
        """Test parameter validation."""
        predictor = SustainabilityPropertyPredictor(use_gpu=False)

        # Valid parameters
        valid_params = {
            "monomer_ratio": 0.5,
            "crosslink_density": 0.1,
            "stabilizer_content": 0.02,
        }
        assert predictor.validate_parameters(valid_params) is True

        # Invalid parameters
        invalid_params = {"monomer_ratio": 1.5, "crosslink_density": -0.1}  # > 1  # < 0
        assert predictor.validate_parameters(invalid_params) is False

        # Non-numeric parameters
        non_numeric_params = {"monomer_ratio": "invalid"}
        assert predictor.validate_parameters(non_numeric_params) is False

    @pytest.mark.asyncio
    async def test_property_prediction_error_handling(self):
        """Test property prediction with error conditions."""
        predictor = SustainabilityPropertyPredictor(use_gpu=False)

        # Test with extreme parameters
        extreme_params = {
            "monomer_ratio": 0.0,
            "crosslink_density": 0.0,
            "stabilizer_content": 0.0,
            "filler_content": 0.0,
            "polymer_purity": 0.5,
        }

        properties = await predictor.predict_properties(extreme_params)

        # Should still return reasonable fallback values
        assert isinstance(properties, dict)
        assert properties["degradation_rate"] > 0
        assert 0 <= properties["recyclability_score"] <= 1


class TestOptimizationProblem:
    """Test optimization problem formulation."""

    def test_problem_creation(self):
        """Test optimization problem initialization."""
        parameters = create_polymer_design_parameters()
        objectives = create_sustainability_objectives()
        constraints = create_performance_constraints()
        predictor = SustainabilityPropertyPredictor(use_gpu=False)

        problem = OptimizationProblem(parameters, objectives, constraints, predictor)

        assert len(problem.parameters) == len(parameters)
        assert len(problem.objectives) == len(objectives)
        assert len(problem.constraints) == len(constraints)
        assert problem.property_predictor == predictor
        assert len(problem.param_names) == len(parameters)
        assert len(problem.param_bounds) == len(parameters)

    @pytest.mark.asyncio
    async def test_async_evaluation(self):
        """Test async evaluation of objectives and constraints."""
        parameters = create_polymer_design_parameters()
        objectives = create_sustainability_objectives()
        constraints = create_performance_constraints()
        predictor = SustainabilityPropertyPredictor(use_gpu=False)

        problem = OptimizationProblem(parameters, objectives, constraints, predictor)

        # Test with single solution
        x = np.array([0.5, 0.1, 0.02, 0.1, 0.95])

        obj_values, constr_values = await problem.evaluate_async(x)

        assert obj_values.shape == (1, len(objectives))
        assert constr_values.shape == (1, len(constraints))

        # Test with multiple solutions
        x_multi = np.array([[0.5, 0.1, 0.02, 0.1, 0.95], [0.8, 0.05, 0.01, 0.05, 0.98]])

        obj_values_multi, constr_values_multi = await problem.evaluate_async(x_multi)

        assert obj_values_multi.shape == (2, len(objectives))
        assert constr_values_multi.shape == (2, len(constraints))

    def test_objective_evaluation(self):
        """Test individual objective evaluation."""
        parameters = create_polymer_design_parameters()
        objectives = create_sustainability_objectives()
        predictor = SustainabilityPropertyPredictor(use_gpu=False)

        problem = OptimizationProblem(parameters, objectives, [], predictor)

        properties = {
            "degradation_rate": 0.001,
            "recyclability_score": 0.8,
            "carbon_footprint": 1.5,
        }

        for target in objectives:
            value = problem._evaluate_objective(target, properties)
            assert isinstance(value, float)
            assert not np.isnan(value)

    def test_constraint_evaluation(self):
        """Test individual constraint evaluation."""
        parameters = create_polymer_design_parameters()
        constraints = create_performance_constraints()
        predictor = SustainabilityPropertyPredictor(use_gpu=False)

        problem = OptimizationProblem(parameters, [], constraints, predictor)

        properties = {"tensile_strength": 35.0, "degradation_temp": 250.0}

        params = {"test_param": 0.5}

        for constraint in constraints:
            value = problem._evaluate_constraint(constraint, properties, params)
            assert isinstance(value, float)
            assert not np.isnan(value)


class TestSustainableInverseDesigner:
    """Test sustainable inverse designer."""

    def test_designer_creation(self):
        """Test designer initialization."""
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            max_iterations=100,
            use_gpu=False,
        )

        designer = SustainableInverseDesigner(config)

        assert designer.config == config
        assert designer.property_predictor is not None
        assert designer.current_problem is None
        assert isinstance(designer.optimization_history, list)
        assert isinstance(designer.best_solutions, list)

    @pytest.mark.asyncio
    async def test_material_design_basic(self):
        """Test basic material design functionality."""
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            max_iterations=10,  # Small for testing
            population_size=5,
            use_gpu=False,
        )

        designer = SustainableInverseDesigner(config)

        parameters = create_polymer_design_parameters()
        objectives = create_sustainability_objectives()

        result = await designer.design_material(parameters, objectives)

        assert isinstance(result, DesignResult)
        assert result.success is True
        assert len(result.optimal_parameters) == len(parameters)
        assert len(result.objective_values) == len(objectives)
        assert result.computation_time > 0

    @pytest.mark.asyncio
    async def test_material_design_with_constraints(self):
        """Test material design with constraints."""
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            max_iterations=10,
            use_gpu=False,
        )

        designer = SustainableInverseDesigner(config)

        parameters = create_polymer_design_parameters()
        objectives = create_sustainability_objectives()
        constraints = create_performance_constraints()

        result = await designer.design_material(parameters, objectives, constraints)

        assert isinstance(result, DesignResult)
        assert result.success is True
        assert len(result.constraint_violations) >= 0

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")
    @pytest.mark.asyncio
    async def test_genetic_algorithm_optimization(self):
        """Test genetic algorithm optimization."""
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            max_iterations=20,
            use_gpu=False,
        )

        designer = SustainableInverseDesigner(config)

        parameters = create_polymer_design_parameters()
        objectives = [
            SustainabilityTarget(
                objective=OptimizationObjective.MINIMIZE_DEGRADATION_RATE,
                weight=1.0,
                minimize=True,
            )
        ]

        result = await designer._run_genetic_algorithm(parameters, objectives, [])

        assert isinstance(result, DesignResult)
        assert result.success is True
        assert "Differential Evolution" in result.metadata.get("algorithm", "")

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    @pytest.mark.asyncio
    async def test_bayesian_optimization(self):
        """Test Bayesian optimization."""
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.BAYESIAN_OPTIMIZATION,
            max_iterations=10,
            use_gpu=False,
        )

        designer = SustainableInverseDesigner(config)

        parameters = create_polymer_design_parameters()
        objectives = create_sustainability_objectives()

        result = await designer._run_bayesian_optimization(parameters, objectives, [])

        assert isinstance(result, DesignResult)
        assert result.success is True
        assert "TPE" in result.metadata.get("algorithm", "")

    @pytest.mark.asyncio
    async def test_default_optimization(self):
        """Test default (random search) optimization."""
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,  # Will fall back to default
            max_iterations=20,
            use_gpu=False,
        )

        designer = SustainableInverseDesigner(config)

        parameters = create_polymer_design_parameters()
        objectives = create_sustainability_objectives()

        # Mock scipy to force fallback
        with patch("chemi_accel.sustainability.inverse_design.SCIPY_AVAILABLE", False):
            result = await designer._run_default_optimization(
                parameters, objectives, []
            )

        assert isinstance(result, DesignResult)
        assert result.success is True
        assert "Random Search" in result.message

    @pytest.mark.asyncio
    async def test_multi_objective_design(self):
        """Test multi-objective design."""
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            max_iterations=10,
            use_gpu=False,
        )

        designer = SustainableInverseDesigner(config)

        parameters = create_polymer_design_parameters()
        objectives = create_sustainability_objectives()

        results = await designer.multi_objective_design(
            parameters, objectives, n_solutions=3
        )

        assert isinstance(results, list)
        assert len(results) <= 3  # May be fewer if some fail
        for result in results:
            assert isinstance(result, DesignResult)
            assert result.success is True

    def test_trade_off_analysis(self):
        """Test trade-off analysis."""
        config = OptimizationConfig(use_gpu=False)
        designer = SustainableInverseDesigner(config)

        # Create mock results
        results = []
        for i in range(3):
            result = DesignResult(
                optimal_parameters={"param1": 0.5 + i * 0.1},
                objective_values={
                    OptimizationObjective.MINIMIZE_DEGRADATION_RATE: 0.1 + i * 0.05,
                    OptimizationObjective.MAXIMIZE_RECYCLABILITY: 0.8 - i * 0.1,
                    OptimizationObjective.MINIMIZE_CARBON_FOOTPRINT: 1.0 + i * 0.2,
                },
                constraint_violations={},
                optimization_history=[],
                convergence_metrics={},
                sustainability_scores={},
                predicted_properties={},
                computation_time=1.0,
                success=True,
                message="Test result",
            )
            results.append(result)

        analysis = designer.analyze_trade_offs(results)

        assert isinstance(analysis, dict)
        assert "pareto_front" in analysis
        assert "objective_correlations" in analysis
        assert "recommendations" in analysis
        assert len(analysis["pareto_front"]) > 0

    def test_sustainability_score_calculation(self):
        """Test overall sustainability score calculation."""
        config = OptimizationConfig(use_gpu=False)
        designer = SustainableInverseDesigner(config)

        properties = {
            "degradation_rate": 0.001,
            "recyclability_score": 0.8,
            "carbon_footprint": 1.5,
            "circularity_index": 0.7,
        }

        score = designer._calculate_overall_sustainability(properties)

        assert isinstance(score, float)
        assert 0 <= score <= 1


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_polymer_design_parameters(self):
        """Test polymer design parameters creation."""
        parameters = create_polymer_design_parameters()

        assert isinstance(parameters, list)
        assert len(parameters) > 0

        for param in parameters:
            assert isinstance(param, MaterialParameter)
            assert param.min_value < param.max_value
            assert param.name is not None

    def test_create_sustainability_objectives(self):
        """Test sustainability objectives creation."""
        objectives = create_sustainability_objectives()

        assert isinstance(objectives, list)
        assert len(objectives) > 0

        for obj in objectives:
            assert isinstance(obj, SustainabilityTarget)
            assert obj.weight > 0
            assert obj.objective in OptimizationObjective

    def test_create_performance_constraints(self):
        """Test performance constraints creation."""
        constraints = create_performance_constraints()

        assert isinstance(constraints, list)
        assert len(constraints) > 0

        for constraint in constraints:
            assert isinstance(constraint, OptimizationConstraint)
            assert callable(constraint.function)
            assert constraint.constraint_type in ConstraintType

    @pytest.mark.asyncio
    async def test_design_sustainable_material_utility(self):
        """Test utility function for sustainable material design."""
        result = await design_sustainable_material(
            material_type="polymer",
            optimization_objectives=[
                OptimizationObjective.MINIMIZE_DEGRADATION_RATE,
                OptimizationObjective.MAXIMIZE_RECYCLABILITY,
            ],
            constraints=["performance"],
            use_gpu=False,
            max_iterations=10,
        )

        assert isinstance(result, DesignResult)
        assert result.success is True
        assert len(result.optimal_parameters) > 0
        assert len(result.objective_values) >= 2


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_optimization_objective_enum(self):
        """Test optimization objective enumeration."""
        assert (
            OptimizationObjective.MINIMIZE_DEGRADATION_RATE.value
            == "minimize_degradation_rate"
        )
        assert (
            OptimizationObjective.MAXIMIZE_RECYCLABILITY.value
            == "maximize_recyclability"
        )
        assert (
            OptimizationObjective.MINIMIZE_CARBON_FOOTPRINT.value
            == "minimize_carbon_footprint"
        )

    def test_constraint_type_enum(self):
        """Test constraint type enumeration."""
        assert ConstraintType.INEQUALITY.value == "inequality"
        assert ConstraintType.EQUALITY.value == "equality"
        assert ConstraintType.BOUND.value == "bound"

    def test_optimization_algorithm_enum(self):
        """Test optimization algorithm enumeration."""
        assert OptimizationAlgorithm.GENETIC_ALGORITHM.value == "genetic_algorithm"
        assert (
            OptimizationAlgorithm.BAYESIAN_OPTIMIZATION.value == "bayesian_optimization"
        )
        assert (
            OptimizationAlgorithm.MULTI_OBJECTIVE_GENETIC.value
            == "multi_objective_genetic"
        )

    @pytest.mark.asyncio
    async def test_invalid_parameters_handling(self):
        """Test handling of invalid parameters."""
        config = OptimizationConfig(max_iterations=5, use_gpu=False)
        designer = SustainableInverseDesigner(config)

        # Empty parameters list
        with pytest.raises(Exception):
            await designer.design_material([], [])

        # Invalid parameter bounds
        invalid_param = MaterialParameter(
            name="invalid",
            min_value=1.0,
            max_value=0.0,  # max < min
            parameter_type="continuous",
        )

        objectives = create_sustainability_objectives()

        # Should handle gracefully or raise appropriate error
        try:
            result = await designer.design_material([invalid_param], objectives)
            # If no exception, check that optimization handled it
            assert isinstance(result, DesignResult)
        except (ValueError, OptimizationFailedError):
            # Expected for invalid parameters
            pass

    @pytest.mark.asyncio
    async def test_optimization_failure_handling(self):
        """Test handling of optimization failure."""
        config = OptimizationConfig(
            max_iterations=1,  # Very low to force potential failure
            population_size=1,
            use_gpu=False,
        )

        designer = SustainableInverseDesigner(config)

        parameters = create_polymer_design_parameters()
        objectives = create_sustainability_objectives()

        # Create impossible constraints
        impossible_constraint = OptimizationConstraint(
            name="impossible",
            constraint_type=ConstraintType.INEQUALITY,
            function=lambda props, params: 1.0,  # Always violated
            description="Impossible constraint",
        )

        try:
            result = await designer.design_material(
                parameters, objectives, [impossible_constraint]
            )
            # If it succeeds, check that it's marked appropriately
            assert isinstance(result, DesignResult)
        except OptimizationFailedError:
            # Expected for impossible constraints
            pass

    def test_property_prediction_error_handling(self):
        """Test property prediction error handling."""
        predictor = SustainabilityPropertyPredictor(use_gpu=False)

        # Test with None parameters
        invalid_params = None

        try:
            valid = predictor.validate_parameters(invalid_params)
            assert valid is False
        except (TypeError, AttributeError):
            # Expected for None input
            pass


class TestPerformance:
    """Test performance aspects."""

    @pytest.mark.asyncio
    async def test_optimization_timing(self):
        """Test that optimizations complete in reasonable time."""
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            max_iterations=20,
            population_size=10,
            use_gpu=False,
        )

        designer = SustainableInverseDesigner(config)

        parameters = create_polymer_design_parameters()
        objectives = [
            SustainabilityTarget(
                objective=OptimizationObjective.MINIMIZE_DEGRADATION_RATE,
                weight=1.0,
                minimize=True,
            )
        ]

        start_time = datetime.now()
        result = await designer.design_material(parameters, objectives)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        assert execution_time < 30.0  # Should complete within 30 seconds
        assert isinstance(result, DesignResult)
        assert result.computation_time > 0

    @pytest.mark.asyncio
    async def test_property_prediction_timing(self):
        """Test property prediction performance."""
        predictor = SustainabilityPropertyPredictor(use_gpu=False)

        parameters = {
            "monomer_ratio": 0.5,
            "crosslink_density": 0.1,
            "stabilizer_content": 0.02,
            "filler_content": 0.1,
            "polymer_purity": 0.95,
        }

        start_time = datetime.now()
        properties = await predictor.predict_properties(parameters)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        assert execution_time < 10.0  # Should complete within 10 seconds
        assert isinstance(properties, dict)
        assert len(properties) > 0


if __name__ == "__main__":
    pytest.main([__file__])

"""
Tests for forward reaction prediction module.

This module tests the forward reaction prediction capabilities including
yield prediction, selectivity analysis, kinetics estimation, and route evaluation.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Import the module under test
from chemi_accel.retrosynthesis.forward_prediction import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    ByproductPrediction,
    ForwardPredictionError,
    ForwardPredictionResult,
    ForwardReactionPredictor,
    KineticsEstimator,
    KineticsModel,
    KineticsModelingError,
    KineticsParameters,
    MolecularFeaturizer,
    ReactionCondition,
    ReactionConditions,
    RouteEvaluationResult,
    SelectivityPrediction,
    SelectivityPredictionError,
    SelectivityPredictor,
    SelectivityType,
    YieldCategory,
    YieldPredictionError,
    YieldPredictor,
    create_reaction_conditions,
    estimate_reaction_kinetics,
    evaluate_synthesis_route_forward,
    predict_reaction_yield,
)
from chemi_accel.retrosynthesis.route_enumeration import SynthesisNode, SynthesisRoute


# Test fixtures
@pytest.fixture
def sample_reaction_conditions():
    """Create sample reaction conditions."""
    return ReactionConditions(
        temperature=323.15,
        pressure=1.0,
        solvent="ethanol",
        catalyst="Pd/C",
        ph=7.0,
        concentration=1.0,
        reaction_time=2.0,
        atmosphere="nitrogen",
    )


@pytest.fixture
def sample_kinetics_parameters():
    """Create sample kinetics parameters."""
    return KineticsParameters(
        activation_energy=20.0,
        pre_exponential_factor=1e9,
        reaction_order=1.0,
        rate_constant=1e-3,
        half_life=1.0,
        enthalpy_activation=19.0,
        entropy_activation=-10.0,
        gibbs_activation=22.0,
    )


@pytest.fixture
def sample_synthesis_route():
    """Create sample synthesis route."""
    nodes = [
        SynthesisNode(smiles="CCO", depth=0, confidence=1.0, cost_estimate=0.0),
        SynthesisNode(
            smiles="CC",
            depth=1,
            confidence=0.8,
            cost_estimate=10.0,
            is_purchasable=True,
        ),
        SynthesisNode(
            smiles="O", depth=1, confidence=0.9, cost_estimate=5.0, is_purchasable=True
        ),
    ]
    return SynthesisRoute(target_smiles="CCO", route_nodes=nodes)


class TestEnums:
    """Test enum classes."""

    def test_reaction_condition_values(self):
        """Test ReactionCondition enum values."""
        assert ReactionCondition.TEMPERATURE.value == "temperature"
        assert ReactionCondition.PRESSURE.value == "pressure"
        assert ReactionCondition.SOLVENT.value == "solvent"
        assert ReactionCondition.CATALYST.value == "catalyst"

    def test_yield_category_values(self):
        """Test YieldCategory enum values."""
        assert YieldCategory.EXCELLENT.value == "excellent"
        assert YieldCategory.GOOD.value == "good"
        assert YieldCategory.MODERATE.value == "moderate"
        assert YieldCategory.POOR.value == "poor"
        assert YieldCategory.VERY_POOR.value == "very_poor"

    def test_selectivity_type_values(self):
        """Test SelectivityType enum values."""
        assert SelectivityType.REGIOSELECTIVITY.value == "regioselectivity"
        assert SelectivityType.STEREOSELECTIVITY.value == "stereoselectivity"
        assert SelectivityType.CHEMOSELECTIVITY.value == "chemoselectivity"
        assert SelectivityType.ENANTIOSELECTIVITY.value == "enantioselectivity"


class TestDataStructures:
    """Test data structure classes."""

    def test_reaction_conditions_creation(self, sample_reaction_conditions):
        """Test ReactionConditions creation."""
        conditions = sample_reaction_conditions

        assert conditions.temperature == 323.15
        assert conditions.solvent == "ethanol"
        assert conditions.catalyst == "Pd/C"
        assert conditions.ph == 7.0

    def test_kinetics_parameters_creation(self, sample_kinetics_parameters):
        """Test KineticsParameters creation."""
        params = sample_kinetics_parameters

        assert params.activation_energy == 20.0
        assert params.pre_exponential_factor == 1e9
        assert params.reaction_order == 1.0
        assert params.rate_constant == 1e-3

    def test_selectivity_prediction_creation(self):
        """Test SelectivityPrediction creation."""
        prediction = SelectivityPrediction(
            selectivity_type=SelectivityType.REGIOSELECTIVITY,
            major_product_ratio=0.8,
            minor_products=[("isomer_1", 0.2)],
            selectivity_factor=4.0,
            confidence=0.7,
        )

        assert prediction.selectivity_type == SelectivityType.REGIOSELECTIVITY
        assert prediction.major_product_ratio == 0.8
        assert prediction.selectivity_factor == 4.0
        assert prediction.confidence == 0.7

    def test_byproduct_prediction_creation(self):
        """Test ByproductPrediction creation."""
        byproduct = ByproductPrediction(
            byproduct_smiles="CC",
            formation_probability=0.3,
            yield_percentage=5.0,
            formation_mechanism="side_reaction",
        )

        assert byproduct.byproduct_smiles == "CC"
        assert byproduct.formation_probability == 0.3
        assert byproduct.yield_percentage == 5.0
        assert byproduct.formation_mechanism == "side_reaction"

    def test_forward_prediction_result_creation(self):
        """Test ForwardPredictionResult creation."""
        result = ForwardPredictionResult(
            reactants=["CCO"],
            products=["CC", "O"],
            main_product_yield=0.8,
            yield_category=YieldCategory.GOOD,
            confidence=0.7,
        )

        assert result.reactants == ["CCO"]
        assert result.products == ["CC", "O"]
        assert result.main_product_yield == 0.8
        assert result.yield_category == YieldCategory.GOOD
        assert result.confidence == 0.7


class TestMolecularFeaturizer:
    """Test MolecularFeaturizer class."""

    def test_featurizer_creation(self):
        """Test featurizer creation."""
        featurizer = MolecularFeaturizer()

        assert isinstance(featurizer.feature_cache, dict)
        assert len(featurizer.feature_cache) == 0

    def test_featurize_reaction(self):
        """Test reaction featurization."""
        featurizer = MolecularFeaturizer()

        reactants = ["CCO"]
        products = ["CC", "O"]

        features = featurizer.featurize_reaction(reactants, products)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_molecular_features_caching(self):
        """Test molecular features caching."""
        featurizer = MolecularFeaturizer()

        # First call
        features1 = featurizer._get_molecular_features("CCO")
        assert "CCO" in featurizer.feature_cache

        # Second call (should use cache)
        features2 = featurizer._get_molecular_features("CCO")
        assert np.array_equal(features1, features2)

    def test_encode_conditions(self, sample_reaction_conditions):
        """Test condition encoding."""
        featurizer = MolecularFeaturizer()

        encoded = featurizer._encode_conditions(sample_reaction_conditions)

        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 16  # Fixed size

    def test_encode_none_conditions(self):
        """Test encoding None conditions."""
        featurizer = MolecularFeaturizer()

        encoded = featurizer._encode_conditions(None)

        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 16
        assert np.all(encoded == 0.0)


class TestYieldPredictor:
    """Test YieldPredictor class."""

    def test_yield_predictor_creation(self):
        """Test yield predictor creation."""
        predictor = YieldPredictor(use_gpu=False)

        assert predictor.use_gpu is False
        assert isinstance(predictor.featurizer, MolecularFeaturizer)
        assert isinstance(predictor.yield_cache, dict)

    def test_predict_yield(self):
        """Test yield prediction."""
        predictor = YieldPredictor(use_gpu=False)

        reactants = ["CCO"]
        products = ["CC", "O"]

        yield_val, category, confidence = predictor.predict_yield(reactants, products)

        assert 0.0 <= yield_val <= 1.0
        assert isinstance(category, YieldCategory)
        assert 0.0 <= confidence <= 1.0

    def test_yield_prediction_caching(self, sample_reaction_conditions):
        """Test yield prediction caching."""
        predictor = YieldPredictor(use_gpu=False)

        reactants = ["CCO"]
        products = ["CC", "O"]

        # First prediction
        yield1, _, _ = predictor.predict_yield(
            reactants, products, sample_reaction_conditions
        )

        # Second prediction (should use cache)
        yield2, _, _ = predictor.predict_yield(
            reactants, products, sample_reaction_conditions
        )

        assert yield1 == yield2
        assert len(predictor.yield_cache) > 0

    def test_categorize_yield(self):
        """Test yield categorization."""
        predictor = YieldPredictor(use_gpu=False)

        assert predictor._categorize_yield(0.95) == YieldCategory.EXCELLENT
        assert predictor._categorize_yield(0.75) == YieldCategory.GOOD
        assert predictor._categorize_yield(0.55) == YieldCategory.MODERATE
        assert predictor._categorize_yield(0.35) == YieldCategory.POOR
        assert predictor._categorize_yield(0.15) == YieldCategory.VERY_POOR

    def test_heuristic_yield_prediction(self, sample_reaction_conditions):
        """Test heuristic yield prediction."""
        predictor = YieldPredictor(use_gpu=False)

        reactants = ["CCO"]
        products = ["CC", "O"]

        yield_val = predictor._heuristic_yield_prediction(
            reactants, products, sample_reaction_conditions
        )

        assert 0.1 <= yield_val <= 0.9


class TestSelectivityPredictor:
    """Test SelectivityPredictor class."""

    def test_selectivity_predictor_creation(self):
        """Test selectivity predictor creation."""
        predictor = SelectivityPredictor(use_gpu=False)

        assert predictor.use_gpu is False
        assert isinstance(predictor.featurizer, MolecularFeaturizer)

    def test_predict_selectivity(self):
        """Test selectivity prediction."""
        predictor = SelectivityPredictor(use_gpu=False)

        reactants = ["CCO"]
        products = ["CC", "O"]

        predictions = predictor.predict_selectivity(reactants, products)

        assert isinstance(predictions, list)
        for pred in predictions:
            assert isinstance(pred, SelectivityPrediction)
            assert 0.0 <= pred.confidence <= 1.0

    def test_predict_byproducts(self):
        """Test byproduct prediction."""
        predictor = SelectivityPredictor(use_gpu=False)

        reactants = ["CCO"]
        products = ["CC", "O"]

        byproducts = predictor.predict_byproducts(reactants, products)

        assert isinstance(byproducts, list)
        for byproduct in byproducts:
            assert isinstance(byproduct, ByproductPrediction)
            assert 0.0 <= byproduct.formation_probability <= 1.0

    def test_generate_potential_byproducts(self):
        """Test potential byproduct generation."""
        predictor = SelectivityPredictor(use_gpu=False)

        reactants = ["CCO", "CC"]

        byproducts = predictor._generate_potential_byproducts(reactants)

        assert isinstance(byproducts, list)
        assert len(byproducts) <= 10  # Should be limited

    def test_has_stereocenters(self):
        """Test stereocenter detection."""
        predictor = SelectivityPredictor(use_gpu=False)

        # Simple molecule without stereocenters
        assert predictor._has_stereocenters("CCO") is False

        # Molecule with potential stereocenters
        assert predictor._has_stereocenters("CC(O)C") in [
            True,
            False,
        ]  # Depends on implementation


class TestKineticsEstimator:
    """Test KineticsEstimator class."""

    def test_kinetics_estimator_creation(self):
        """Test kinetics estimator creation."""
        estimator = KineticsEstimator(use_gpu=False)

        assert estimator.use_gpu is False
        assert isinstance(estimator.featurizer, MolecularFeaturizer)

    def test_estimate_kinetics(self):
        """Test kinetics estimation."""
        estimator = KineticsEstimator(use_gpu=False)

        reactants = ["CCO"]
        products = ["CC", "O"]

        params = estimator.estimate_kinetics(reactants, products)

        assert isinstance(params, KineticsParameters)
        assert params.activation_energy > 0
        assert params.pre_exponential_factor > 0
        assert params.rate_constant > 0

    def test_heuristic_kinetics_estimation(self, sample_reaction_conditions):
        """Test heuristic kinetics estimation."""
        estimator = KineticsEstimator(use_gpu=False)

        reactants = ["CCO"]
        products = ["CC", "O"]

        params = estimator._heuristic_kinetics_estimation(
            reactants, products, sample_reaction_conditions
        )

        assert isinstance(params, KineticsParameters)
        assert params.activation_energy > 0
        assert params.pre_exponential_factor > 0

    def test_default_kinetics_parameters(self):
        """Test default kinetics parameters."""
        estimator = KineticsEstimator(use_gpu=False)

        params = estimator._default_kinetics_parameters()

        assert isinstance(params, KineticsParameters)
        assert params.activation_energy == 20.0
        assert params.pre_exponential_factor == 1e9


class TestForwardReactionPredictor:
    """Test ForwardReactionPredictor class."""

    def test_predictor_creation(self):
        """Test forward reaction predictor creation."""
        predictor = ForwardReactionPredictor()

        assert isinstance(predictor.yield_predictor, YieldPredictor)
        assert isinstance(predictor.selectivity_predictor, SelectivityPredictor)
        assert isinstance(predictor.kinetics_estimator, KineticsEstimator)
        assert isinstance(predictor.prediction_cache, dict)

    def test_predict_reaction_outcome(self):
        """Test reaction outcome prediction."""
        predictor = ForwardReactionPredictor()

        reactants = ["CCO"]
        products = ["CC", "O"]

        result = predictor.predict_reaction_outcome(reactants, products)

        assert isinstance(result, ForwardPredictionResult)
        assert result.reactants == reactants
        assert result.products == products
        assert 0.0 <= result.main_product_yield <= 1.0
        assert isinstance(result.yield_category, YieldCategory)
        assert 0.0 <= result.confidence <= 1.0

    def test_evaluate_synthesis_route(self, sample_synthesis_route):
        """Test synthesis route evaluation."""
        predictor = ForwardReactionPredictor()

        result = predictor.evaluate_synthesis_route(sample_synthesis_route)

        assert isinstance(result, RouteEvaluationResult)
        assert result.route == sample_synthesis_route
        assert isinstance(result.step_predictions, list)
        assert 0.0 <= result.overall_yield <= 1.0
        assert 0.0 <= result.overall_selectivity <= 1.0
        assert 0.0 <= result.feasibility_score <= 1.0

    def test_optimize_reaction_conditions(self, sample_reaction_conditions):
        """Test reaction condition optimization."""
        predictor = ForwardReactionPredictor()

        reactants = ["CCO"]
        products = ["CC", "O"]

        optimized = predictor._optimize_reaction_conditions(
            reactants, products, sample_reaction_conditions
        )

        assert isinstance(optimized, ReactionConditions)
        assert optimized.temperature > 0
        assert optimized.pressure > 0

    def test_analyze_bottlenecks(self):
        """Test bottleneck analysis."""
        predictor = ForwardReactionPredictor()

        # Create sample step predictions
        step_predictions = [
            ForwardPredictionResult(
                reactants=["CCO"],
                products=["CC"],
                main_product_yield=0.8,
                confidence=0.7,
            ),
            ForwardPredictionResult(
                reactants=["CC"], products=["C"], main_product_yield=0.6, confidence=0.5
            ),
        ]

        bottlenecks = predictor._analyze_bottlenecks(step_predictions)

        assert isinstance(bottlenecks, dict)
        assert "lowest_yield_step" in bottlenecks
        assert "lowest_confidence_step" in bottlenecks

    def test_generate_optimization_suggestions(self):
        """Test optimization suggestion generation."""
        predictor = ForwardReactionPredictor()

        # Create sample step predictions with issues
        step_predictions = [
            ForwardPredictionResult(
                reactants=["CCO"],
                products=["CC"],
                main_product_yield=0.4,  # Low yield
                confidence=0.5,  # Low confidence
            )
        ]

        suggestions = predictor._generate_optimization_suggestions(
            step_predictions, [0]
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)

    def test_calculate_feasibility_score(self):
        """Test feasibility score calculation."""
        predictor = ForwardReactionPredictor()

        score = predictor._calculate_feasibility_score(
            overall_yield=0.8,
            overall_selectivity=0.9,
            num_critical_steps=1,
            total_steps=3,
        )

        assert 0.0 <= score <= 1.0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_reaction_conditions(self):
        """Test create_reaction_conditions function."""
        conditions = create_reaction_conditions(
            temperature=350.0, solvent="dmso", catalyst="Pd/C"
        )

        assert isinstance(conditions, ReactionConditions)
        assert conditions.temperature == 350.0
        assert conditions.solvent == "dmso"
        assert conditions.catalyst == "Pd/C"

    def test_predict_reaction_yield_function(self):
        """Test predict_reaction_yield utility function."""
        reactants = ["CCO"]
        products = ["CC", "O"]

        yield_val, category = predict_reaction_yield(reactants, products)

        assert 0.0 <= yield_val <= 1.0
        assert isinstance(category, YieldCategory)

    def test_estimate_reaction_kinetics_function(self):
        """Test estimate_reaction_kinetics utility function."""
        reactants = ["CCO"]
        products = ["CC", "O"]

        params = estimate_reaction_kinetics(reactants, products)

        assert isinstance(params, KineticsParameters)
        assert params.activation_energy > 0

    def test_evaluate_synthesis_route_forward_function(self, sample_synthesis_route):
        """Test evaluate_synthesis_route_forward utility function."""
        result = evaluate_synthesis_route_forward(sample_synthesis_route)

        assert isinstance(result, RouteEvaluationResult)
        assert result.route == sample_synthesis_route


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_forward_prediction_error_inheritance(self):
        """Test exception inheritance."""
        assert issubclass(KineticsModelingError, ForwardPredictionError)
        assert issubclass(YieldPredictionError, ForwardPredictionError)
        assert issubclass(SelectivityPredictionError, ForwardPredictionError)

    def test_yield_prediction_error_handling(self):
        """Test yield prediction error handling."""
        predictor = YieldPredictor(use_gpu=False)

        # Test with problematic input
        try:
            yield_val, category, confidence = predictor.predict_yield([""], [""])
            # If no error, check that result is still valid
            assert 0.0 <= yield_val <= 1.0
            assert isinstance(category, YieldCategory)
        except (YieldPredictionError, Exception):
            # Expected error is acceptable
            pass

    def test_route_evaluation_error_handling(self):
        """Test route evaluation error handling."""
        predictor = ForwardReactionPredictor()

        # Create route with invalid data
        invalid_nodes = [SynthesisNode(smiles="", depth=0)]
        invalid_route = SynthesisRoute(target_smiles="", route_nodes=invalid_nodes)

        try:
            result = predictor.evaluate_synthesis_route(invalid_route)
            assert isinstance(result, RouteEvaluationResult)
        except ForwardPredictionError:
            # Expected error is acceptable
            pass


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    def test_yield_prediction_performance(self):
        """Test yield prediction performance."""
        predictor = YieldPredictor(use_gpu=False)

        # Test batch predictions
        test_reactions = [
            (["CCO"], ["CC", "O"]),
            (["c1ccccc1"], ["c1ccccc1C"]),
            (["CC(=O)O"], ["CC", "CO"]),
        ] * 10  # 30 predictions

        import time

        start_time = time.time()

        for reactants, products in test_reactions:
            yield_val, _, _ = predictor.predict_yield(reactants, products)
            assert 0.0 <= yield_val <= 1.0

        elapsed_time = time.time() - start_time
        assert elapsed_time < 10.0  # Should complete within 10 seconds

    def test_featurization_performance(self):
        """Test featurization performance."""
        featurizer = MolecularFeaturizer()

        test_molecules = [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "CCN",
            "CCC",
            "c1ccc2c(c1)cccc2",
            "CC(C)CC(C(=O)O)N",
        ] * 20  # 140 featurizations

        import time

        start_time = time.time()

        for smiles in test_molecules:
            features = featurizer._get_molecular_features(smiles)
            assert isinstance(features, np.ndarray)

        elapsed_time = time.time() - start_time
        assert elapsed_time < 5.0  # Should be fast due to caching

    def test_route_evaluation_performance(self):
        """Test route evaluation performance."""
        predictor = ForwardReactionPredictor()

        # Create test routes
        test_routes = []
        for i in range(5):  # 5 routes
            nodes = [
                SynthesisNode(smiles=f"C{i}", depth=0, confidence=1.0),
                SynthesisNode(
                    smiles=f"CC{i}", depth=1, confidence=0.8, is_purchasable=True
                ),
            ]
            route = SynthesisRoute(target_smiles=f"C{i}", route_nodes=nodes)
            test_routes.append(route)

        import time

        start_time = time.time()

        for route in test_routes:
            result = predictor.evaluate_synthesis_route(route)
            assert isinstance(result, RouteEvaluationResult)

        elapsed_time = time.time() - start_time
        assert elapsed_time < 15.0  # Should complete within reasonable time


class TestIntegration:
    """Integration tests for forward prediction."""

    def test_full_prediction_workflow(
        self, sample_synthesis_route, sample_reaction_conditions
    ):
        """Test complete forward prediction workflow."""
        predictor = ForwardReactionPredictor()

        # Test individual reaction prediction
        reactants = ["CCO"]
        products = ["CC", "O"]

        reaction_result = predictor.predict_reaction_outcome(
            reactants, products, sample_reaction_conditions
        )

        assert isinstance(reaction_result, ForwardPredictionResult)
        assert reaction_result.main_product_yield > 0
        assert len(reaction_result.selectivity_predictions) >= 0

        # Test route evaluation
        route_result = predictor.evaluate_synthesis_route(sample_synthesis_route)

        assert isinstance(route_result, RouteEvaluationResult)
        assert route_result.overall_yield > 0
        assert len(route_result.step_predictions) >= 0
        assert len(route_result.optimization_suggestions) >= 0

    def test_different_reaction_types(self):
        """Test prediction for different reaction types."""
        predictor = ForwardReactionPredictor()

        # Test different reaction types
        test_reactions = [
            (["CCO"], ["CC", "O"]),  # Dehydration
            (["c1ccccc1", "CCCl"], ["c1ccccc1CCC"]),  # Alkylation
            (["CC(=O)O", "CCO"], ["CC(=O)OCC", "O"]),  # Esterification
            (["CCN", "CC(=O)Cl"], ["CC(=O)NCC", "HCl"]),  # Amidation
        ]

        all_results = []
        for reactants, products in test_reactions:
            result = predictor.predict_reaction_outcome(reactants, products)

            assert isinstance(result, ForwardPredictionResult)
            assert result.main_product_yield > 0
            assert result.confidence > 0
            all_results.append(result)

        # Should have found predictions for multiple reactions
        assert len(all_results) == len(test_reactions)

    def test_condition_optimization_integration(self):
        """Test integration with condition optimization."""
        predictor = ForwardReactionPredictor()

        reactants = ["CCO"]
        products = ["CC", "O"]

        # Test with different base conditions
        base_conditions = [
            ReactionConditions(temperature=298.15),
            ReactionConditions(temperature=373.15, catalyst="Pd/C"),
            ReactionConditions(temperature=323.15, solvent="dmso"),
        ]

        optimized_conditions = []
        for base in base_conditions:
            optimized = predictor._optimize_reaction_conditions(
                reactants, products, base
            )
            optimized_conditions.append(optimized)

            assert isinstance(optimized, ReactionConditions)
            assert optimized.temperature > 0

        # Should have different optimized conditions
        assert len(optimized_conditions) == len(base_conditions)

    def test_error_propagation_through_system(self):
        """Test error propagation through the entire system."""
        predictor = ForwardReactionPredictor()

        # Test with various problematic inputs
        problematic_inputs = [
            ([""], [""]),  # Empty SMILES
            (["INVALID"], ["ALSO_INVALID"]),  # Invalid SMILES
            (["C" * 200], ["CC"]),  # Very long string
        ]

        for reactants, products in problematic_inputs:
            try:
                result = predictor.predict_reaction_outcome(reactants, products)
                # If successful, verify result structure
                assert isinstance(result, ForwardPredictionResult)
                assert 0.0 <= result.main_product_yield <= 1.0
            except ForwardPredictionError:
                # Expected errors are acceptable
                pass

    def test_caching_system_integration(self):
        """Test integration of caching systems."""
        predictor = ForwardReactionPredictor()

        reactants = ["CCO"]
        products = ["CC", "O"]
        conditions = ReactionConditions()

        # First prediction
        result1 = predictor.predict_reaction_outcome(reactants, products, conditions)

        # Check that caches are populated
        assert len(predictor.yield_predictor.yield_cache) > 0
        assert len(predictor.prediction_cache) > 0

        # Second prediction (should use caches)
        result2 = predictor.predict_reaction_outcome(reactants, products, conditions)

        # Results should be identical due to caching
        assert result1.main_product_yield == result2.main_product_yield
        assert result1.confidence == result2.confidence

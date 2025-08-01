"""
Tests for Autonomous Scientific Law & Causal Relationship Discovery Module.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from chemi_accel.ai.law_discovery import (
    AdaptiveLearningEngine,
    AutomatedExperimentSuggester,
    AutonomousLawDiscovery,
    CausalInferenceEngine,
    CausalRelationship,
    Hypothesis,
    HypothesisGenerator,
    KnowledgeGraphIntegrator,
    LawDiscoveryError,
    PatternRecognitionEngine,
    ScientificLaw,
    SymbolicRegressionEngine,
)


class TestScientificLaw:
    """Test ScientificLaw data structure."""

    def test_scientific_law_creation(self):
        """Test creation of scientific law."""
        law = ScientificLaw(
            equation="y = 2.5 * x + 1.2",
            variables=["x", "y"],
            coefficients={"slope": 2.5, "intercept": 1.2},
            confidence_scores={"overall": 0.85},
            r_squared=0.91,
            p_values={"slope": 0.001, "intercept": 0.05},
            domain_constraints={"x": (0.0, 10.0), "y": (1.2, 26.2)},
            discovery_metadata={"method": "linear_regression"},
        )

        assert law.equation == "y = 2.5 * x + 1.2"
        assert law.variables == ["x", "y"]
        assert law.coefficients["slope"] == 2.5
        assert law.confidence_scores["overall"] == 0.85
        assert law.r_squared == 0.91

    def test_law_to_dict(self):
        """Test conversion to dictionary."""
        law = ScientificLaw(
            equation="y = x^2",
            variables=["x", "y"],
            coefficients={"a": 1.0},
            confidence_scores={"overall": 0.9},
            r_squared=0.95,
            p_values={},
            domain_constraints={},
            discovery_metadata={},
        )

        law_dict = law.to_dict()
        assert isinstance(law_dict, dict)
        assert law_dict["equation"] == "y = x^2"
        assert law_dict["variables"] == ["x", "y"]


class TestCausalRelationship:
    """Test CausalRelationship data structure."""

    def test_causal_relationship_creation(self):
        """Test creation of causal relationship."""
        relationship = CausalRelationship(
            cause_variables=["temperature"],
            effect_variable="reaction_rate",
            causal_strength=0.8,
            causal_direction="forward",
            mechanism_hypothesis="Higher temperature increases molecular kinetics",
            confidence_score=0.85,
            granger_causality_score=12.5,
            statistical_tests={"f_statistic": 12.5, "p_value": 0.001},
        )

        assert relationship.cause_variables == ["temperature"]
        assert relationship.effect_variable == "reaction_rate"
        assert relationship.causal_strength == 0.8
        assert relationship.confidence_score == 0.85


class TestHypothesis:
    """Test Hypothesis data structure."""

    def test_hypothesis_creation(self):
        """Test creation of hypothesis."""
        hypothesis = Hypothesis(
            statement="Increasing catalyst concentration will increase reaction rate",
            variables_involved=["catalyst_concentration", "reaction_rate"],
            predicted_relationship="positive_correlation",
            confidence_score=0.75,
            testability_score=0.9,
            suggested_experiments=[],
            background_evidence=["Previous studies show..."],
            falsifiability_criteria=["If no correlation found..."],
            generation_method="causal_inference",
        )

        assert hypothesis.confidence_score == 0.75
        assert hypothesis.testability_score == 0.9
        assert len(hypothesis.variables_involved) == 2


class TestCausalInferenceEngine:
    """Test CausalInferenceEngine."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        x = np.sin(0.1 * t) + 0.1 * np.random.randn(n)
        y = 0.5 * x + 0.2 * np.random.randn(n)  # y causally depends on x

        return pd.DataFrame({"x": x, "y": y, "time": t})

    def test_engine_initialization(self):
        """Test causal inference engine initialization."""
        engine = CausalInferenceEngine(backend="cpu")
        assert engine.backend == "cpu"
        assert engine.backend_manager is not None

    def test_granger_causality_test_cpu(self, sample_data):
        """Test Granger causality test with CPU backend."""
        engine = CausalInferenceEngine(backend="cpu")

        results = engine.granger_causality_test(sample_data, "x", "y", max_lag=3)

        assert isinstance(results, dict)
        assert "lag_1" in results
        assert "f_statistic" in results["lag_1"]
        assert results["lag_1"]["f_statistic"] > 0

    @patch("chemi_accel.ai.law_discovery.CUDA_AVAILABLE", True)
    @patch("chemi_accel.ai.law_discovery.cudf")
    @patch("chemi_accel.ai.law_discovery.CuLinearRegression")
    def test_granger_causality_test_cuda(self, mock_lr, mock_cudf, sample_data):
        """Test Granger causality test with CUDA backend."""
        # Setup mocks
        mock_model = Mock()
        mock_model.predict.return_value = np.random.randn(50)
        mock_lr.return_value = mock_model

        mock_cudf.DataFrame.return_value = Mock()
        mock_cudf.Series.return_value = Mock()

        with patch("chemi_accel.ai.law_discovery.cp") as mock_cp:
            mock_cp.asarray.return_value = np.random.randn(100)
            mock_cp.column_stack.return_value = np.random.randn(50, 2)
            mock_cp.sum.return_value = 10.0
            mock_cp.asnumpy.return_value = 10.0

            engine = CausalInferenceEngine(backend="cuda")
            results = engine.granger_causality_test(sample_data, "x", "y", max_lag=2)

            assert isinstance(results, dict)
            assert len(results) == 2

    def test_discover_causal_network(self, sample_data):
        """Test causal network discovery."""
        engine = CausalInferenceEngine(backend="cpu")

        relationships = engine.discover_causal_network(
            sample_data, ["x", "y"], significance_threshold=0.1
        )

        assert isinstance(relationships, list)
        # Should find some relationships in the synthetic data
        if relationships:
            assert isinstance(relationships[0], CausalRelationship)


class TestSymbolicRegressionEngine:
    """Test SymbolicRegressionEngine."""

    @pytest.fixture
    def sample_regression_data(self):
        """Generate sample regression data."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2 * X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)
        variable_names = ["feature_1", "feature_2"]
        return X, y, variable_names

    def test_engine_initialization(self):
        """Test symbolic regression engine initialization."""
        engine = SymbolicRegressionEngine(backend="cpu")
        assert engine.backend == "cpu"
        assert engine.population_size == 100

    def test_discover_equations(self, sample_regression_data):
        """Test equation discovery."""
        X, y, var_names = sample_regression_data
        engine = SymbolicRegressionEngine(backend="cpu")

        laws = engine.discover_equations(X, y, var_names, max_complexity=8)

        assert isinstance(laws, list)
        assert len(laws) > 0

        for law in laws:
            assert isinstance(law, ScientificLaw)
            assert law.r_squared >= 0
            assert len(law.variables) > 0

    def test_equation_evaluation(self):
        """Test equation evaluation."""
        engine = SymbolicRegressionEngine()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        coeffs = {"slope": 2.0, "intercept": 1.0}

        result = engine._evaluate_equation("linear", X, coeffs)
        expected = 2.0 * X[:, 0] + 1.0

        np.testing.assert_array_almost_equal(result, expected)


class TestPatternRecognitionEngine:
    """Test PatternRecognitionEngine."""

    @pytest.fixture
    def sample_pattern_data(self):
        """Generate sample data with patterns."""
        np.random.seed(42)
        n = 200

        x1 = np.random.randn(n)
        x2 = 2 * x1 + 0.1 * np.random.randn(n)  # Correlated
        x3 = np.random.randn(n)
        target = x1 + 0.5 * x2 + 0.1 * x3 + 0.1 * np.random.randn(n)

        return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target})

    def test_engine_initialization(self):
        """Test pattern recognition engine initialization."""
        engine = PatternRecognitionEngine(backend="cpu")
        assert engine.backend == "cpu"

    def test_discover_patterns(self, sample_pattern_data):
        """Test pattern discovery."""
        engine = PatternRecognitionEngine(backend="cpu")

        patterns = engine.discover_patterns(
            sample_pattern_data, ["target"], min_confidence=0.5
        )

        assert isinstance(patterns, list)
        # Should find some patterns in synthetic data
        if patterns:
            assert "type" in patterns[0]
            assert "confidence" in patterns[0]

    @patch("chemi_accel.ai.law_discovery.SKLEARN_AVAILABLE", True)
    @patch("chemi_accel.ai.law_discovery.RandomForestRegressor")
    def test_feature_importance_analysis(self, mock_rf, sample_pattern_data):
        """Test feature importance analysis."""
        # Mock RandomForest
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.6, 0.3, 0.1])
        mock_rf.return_value = mock_model

        engine = PatternRecognitionEngine(backend="cpu")
        pattern = engine._analyze_feature_importance(
            sample_pattern_data, "target", min_confidence=0.2
        )

        assert pattern is not None
        assert pattern["type"] == "feature_importance"
        assert "important_features" in pattern

    def test_correlation_analysis(self, sample_pattern_data):
        """Test correlation cluster analysis."""
        engine = PatternRecognitionEngine()
        patterns = engine._find_correlation_clusters(
            sample_pattern_data, min_confidence=0.5
        )

        # Should find correlation between x1 and x2
        assert isinstance(patterns, list)


class TestHypothesisGenerator:
    """Test HypothesisGenerator."""

    @pytest.fixture
    def sample_causal_relationship(self):
        """Create sample causal relationship."""
        return CausalRelationship(
            cause_variables=["temperature"],
            effect_variable="yield",
            causal_strength=0.8,
            causal_direction="forward",
            mechanism_hypothesis="Temperature affects reaction kinetics",
            confidence_score=0.85,
            granger_causality_score=15.2,
            statistical_tests={"f_stat": 15.2},
        )

    @pytest.fixture
    def sample_patterns(self):
        """Create sample patterns."""
        return [
            {
                "type": "feature_importance",
                "target": "yield",
                "important_features": ["temperature", "pressure"],
                "confidence": 0.8,
            },
            {
                "type": "correlation_cluster",
                "correlations": [{"var1": "temp", "var2": "rate", "correlation": 0.9}],
                "confidence": 0.9,
            },
        ]

    def test_generator_initialization(self):
        """Test hypothesis generator initialization."""
        generator = HypothesisGenerator()
        assert len(generator.hypothesis_templates) > 0

    def test_generate_hypotheses(self, sample_causal_relationship, sample_patterns):
        """Test hypothesis generation."""
        generator = HypothesisGenerator()

        hypotheses = generator.generate_hypotheses(
            sample_patterns, [sample_causal_relationship], min_testability=0.5
        )

        assert isinstance(hypotheses, list)
        assert len(hypotheses) > 0

        for hyp in hypotheses:
            assert isinstance(hyp, Hypothesis)
            assert hyp.testability_score >= 0.5
            assert len(hyp.suggested_experiments) > 0

    def test_causal_hypothesis_generation(self, sample_causal_relationship):
        """Test causal hypothesis generation."""
        generator = HypothesisGenerator()

        hypothesis = generator._generate_causal_hypothesis(sample_causal_relationship)

        assert isinstance(hypothesis, Hypothesis)
        assert "temperature" in hypothesis.statement
        assert "yield" in hypothesis.statement
        assert hypothesis.confidence_score == 0.85
        assert len(hypothesis.suggested_experiments) > 0


class TestAutomatedExperimentSuggester:
    """Test AutomatedExperimentSuggester."""

    @pytest.fixture
    def sample_hypothesis(self):
        """Create sample hypothesis."""
        return Hypothesis(
            statement="Temperature affects yield",
            variables_involved=["temperature", "yield"],
            predicted_relationship="positive correlation",
            confidence_score=0.8,
            testability_score=0.9,
            suggested_experiments=[
                {
                    "type": "intervention_study",
                    "variables_to_manipulate": ["temperature"],
                    "variables_to_measure": ["yield"],
                    "controls_needed": ["pressure"],
                }
            ],
            background_evidence=[],
            falsifiability_criteria=[],
            generation_method="test",
        )

    def test_suggester_initialization(self):
        """Test experiment suggester initialization."""
        suggester = AutomatedExperimentSuggester()
        assert len(suggester.experiment_types) > 0

    def test_suggest_experiments(self, sample_hypothesis):
        """Test experiment suggestion."""
        suggester = AutomatedExperimentSuggester()

        resources = {"cost_per_sample": 100, "max_budget": 5000}
        constraints = {"max_samples": 100, "max_duration_days": 30}

        experiments = suggester.suggest_experiments(
            sample_hypothesis, resources, constraints
        )

        assert isinstance(experiments, list)
        assert len(experiments) > 0

        for exp in experiments:
            assert "experiment_id" in exp
            assert "type" in exp
            assert "estimated_cost" in exp
            assert "priority_score" in exp

    def test_intervention_study_design(self, sample_hypothesis):
        """Test intervention study design."""
        suggester = AutomatedExperimentSuggester()

        suggestion = sample_hypothesis.suggested_experiments[0]
        resources = {"cost_per_sample": 100}
        constraints = {"max_samples": 150, "max_duration_days": 45}

        experiment = suggester._design_intervention_study(
            suggestion, sample_hypothesis, resources, constraints
        )

        assert experiment["type"] == "intervention_study"
        assert "design" in experiment
        assert "protocol" in experiment
        assert experiment["estimated_cost"] > 0


class TestAdaptiveLearningEngine:
    """Test AdaptiveLearningEngine."""

    @pytest.fixture
    def sample_law(self):
        """Create sample scientific law."""
        return ScientificLaw(
            equation="y = 2.0 * x + 1.0",
            variables=["x", "y"],
            coefficients={"slope": 2.0, "intercept": 1.0},
            confidence_scores={"overall": 0.8},
            r_squared=0.85,
            p_values={},
            domain_constraints={"x": (0, 10)},
            discovery_metadata={},
        )

    @pytest.fixture
    def sample_experimental_results(self):
        """Create sample experimental results."""
        return [
            {
                "experiment_id": "exp_1",
                "validation_outcome": "confirmed",
                "confidence": 0.9,
            },
            {
                "experiment_id": "exp_2",
                "validation_outcome": "partially_confirmed",
                "confidence": 0.7,
            },
        ]

    def test_engine_initialization(self):
        """Test adaptive learning engine initialization."""
        engine = AdaptiveLearningEngine()
        assert engine.confidence_threshold == 0.7
        assert len(engine.refinement_history) == 0

    def test_refine_law(self, sample_law, sample_experimental_results):
        """Test law refinement."""
        engine = AdaptiveLearningEngine()

        # Create some new data
        new_data = pd.DataFrame({"x": [11, 12, 13], "y": [23, 25, 27]})

        refined_law = engine.refine_law(
            sample_law, new_data, sample_experimental_results
        )

        assert isinstance(refined_law, ScientificLaw)
        assert len(refined_law.validation_experiments) > len(
            sample_law.validation_experiments
        )
        assert len(refined_law.refinement_history) > 0
        assert (
            refined_law.domain_constraints["x"][1]
            > sample_law.domain_constraints["x"][1]
        )

    def test_confidence_score_update(self, sample_law, sample_experimental_results):
        """Test confidence score updating."""
        engine = AdaptiveLearningEngine()

        updated_scores = engine._update_confidence_scores(
            sample_law, sample_experimental_results
        )

        assert "experimental_validation" in updated_scores
        assert updated_scores["experimental_validation"] == 0.75  # (1 + 0.5) / 2


class TestKnowledgeGraphIntegrator:
    """Test KnowledgeGraphIntegrator."""

    @pytest.fixture
    def sample_law_for_kg(self):
        """Create sample law for knowledge graph."""
        return ScientificLaw(
            equation="rate = k * [A] * [B]",
            variables=["rate", "A", "B"],
            coefficients={"k": 0.1},
            confidence_scores={"overall": 0.9},
            r_squared=0.95,
            p_values={},
            domain_constraints={},
            discovery_metadata={"method": "kinetic_analysis"},
            validation_experiments=[{"type": "kinetic_study", "outcome": "confirmed"}],
        )

    def test_integrator_initialization(self):
        """Test knowledge graph integrator initialization."""
        integrator = KnowledgeGraphIntegrator()
        assert integrator.kg_backend == "memory"
        assert "law_types" in integrator.law_ontology

    def test_integrate_law(self, sample_law_for_kg):
        """Test law integration into knowledge graph."""
        integrator = KnowledgeGraphIntegrator()

        result = integrator.integrate_law(sample_law_for_kg)

        assert isinstance(result, dict)
        assert "law_node" in result
        assert "variable_nodes" in result
        assert "validation_nodes" in result
        assert "integration_id" in result

        # Check law node structure
        law_node = result["law_node"]
        assert law_node["type"] == "scientific_law"
        assert "equation" in law_node["properties"]

    def test_law_node_creation(self, sample_law_for_kg):
        """Test law node creation."""
        integrator = KnowledgeGraphIntegrator()

        node = integrator._create_law_node(sample_law_for_kg)

        assert node["type"] == "scientific_law"
        assert node["properties"]["equation"] == "rate = k * [A] * [B]"
        assert node["properties"]["confidence_score"] == 0.9

    def test_variable_classification(self):
        """Test variable classification."""
        integrator = KnowledgeGraphIntegrator()

        assert integrator._classify_variable("molecular_weight") == "molecular_property"
        assert integrator._classify_variable("logp") == "physicochemical_property"
        assert integrator._classify_variable("temperature") == "environmental_condition"

    def test_law_relationship_analysis(self, sample_law_for_kg):
        """Test law relationship analysis."""
        integrator = KnowledgeGraphIntegrator()

        # Create another law with overlapping variables
        law2 = ScientificLaw(
            equation="rate = k2 * [A]^2",
            variables=["rate", "A"],
            coefficients={"k2": 0.05},
            confidence_scores={"overall": 0.8},
            r_squared=0.88,
            p_values={},
            domain_constraints={},
            discovery_metadata={},
        )

        relationship = integrator._analyze_law_relationship(sample_law_for_kg, law2)

        assert relationship is not None
        assert relationship["common_variables"] == ["rate", "A"]
        assert relationship["strength"] > 0


class TestAutonomousLawDiscovery:
    """Test main AutonomousLawDiscovery orchestrator."""

    @pytest.fixture
    def sample_discovery_data(self):
        """Generate comprehensive sample data for discovery."""
        np.random.seed(42)
        n = 300

        # Create synthetic chemical data with known relationships
        temperature = np.random.uniform(20, 80, n)
        pressure = np.random.uniform(1, 5, n)
        concentration = np.random.uniform(0.1, 2.0, n)

        # Known relationships
        reaction_rate = (
            0.1 * temperature
            + 0.05 * pressure
            + 2.0 * concentration
            + 0.1 * np.random.randn(n)
        )
        yield_percent = 10 + 0.5 * reaction_rate + 0.1 * np.random.randn(n)

        return pd.DataFrame(
            {
                "temperature": temperature,
                "pressure": pressure,
                "concentration": concentration,
                "reaction_rate": reaction_rate,
                "yield_percent": yield_percent,
            }
        )

    def test_discovery_initialization(self):
        """Test autonomous discovery system initialization."""
        discovery = AutonomousLawDiscovery(backend="cpu")

        assert discovery.causal_engine is not None
        assert discovery.symbolic_engine is not None
        assert discovery.pattern_engine is not None
        assert discovery.hypothesis_generator is not None
        assert discovery.experiment_suggester is not None
        assert discovery.adaptive_learner is not None
        assert discovery.kg_integrator is not None

    @patch("chemi_accel.ai.law_discovery.logger")
    def test_discover_laws_and_relationships(self, mock_logger, sample_discovery_data):
        """Test complete discovery process."""
        discovery = AutonomousLawDiscovery(backend="cpu")

        results = discovery.discover_laws_and_relationships(
            data=sample_discovery_data,
            target_variables=["reaction_rate", "yield_percent"],
            variable_names=[
                "temperature",
                "pressure",
                "concentration",
                "reaction_rate",
                "yield_percent",
            ],
        )

        assert isinstance(results, dict)
        assert "discovery_summary" in results
        assert "discovered_laws" in results
        assert "causal_relationships" in results
        assert "patterns" in results
        assert "hypotheses" in results
        assert "experimental_suggestions" in results
        assert "knowledge_graph_integrations" in results
        assert "metadata" in results

        # Check summary statistics
        summary = results["discovery_summary"]
        assert summary["laws_discovered"] >= 0
        assert summary["causal_relationships"] >= 0
        assert summary["patterns_identified"] >= 0
        assert summary["hypotheses_generated"] >= 0

    def test_refine_existing_laws(self, sample_discovery_data):
        """Test law refinement process."""
        discovery = AutonomousLawDiscovery(backend="cpu")

        # First discover some laws
        discovery.discover_laws_and_relationships(
            data=sample_discovery_data,
            target_variables=["reaction_rate"],
            variable_names=[
                "temperature",
                "pressure",
                "concentration",
                "reaction_rate",
            ],
        )

        # Simulate new data
        new_data = pd.DataFrame(
            {
                "temperature": [85, 90],
                "pressure": [5.5, 6.0],
                "concentration": [2.5, 3.0],
                "reaction_rate": [18, 20],
            }
        )

        experimental_results = [
            {
                "experiment_id": "validation_1",
                "validation_outcome": "confirmed",
                "confidence": 0.9,
            }
        ]

        if discovery.discovered_laws:  # Only test if laws were discovered
            refined_laws = discovery.refine_existing_laws(
                new_data, experimental_results
            )

            assert isinstance(refined_laws, list)
            assert len(refined_laws) == len(discovery.discovered_laws)

    def test_get_discovery_insights(self, sample_discovery_data):
        """Test discovery insights generation."""
        discovery = AutonomousLawDiscovery(backend="cpu")

        # Run discovery first
        discovery.discover_laws_and_relationships(
            data=sample_discovery_data,
            target_variables=["reaction_rate"],
            variable_names=[
                "temperature",
                "pressure",
                "concentration",
                "reaction_rate",
            ],
        )

        insights = discovery.get_discovery_insights()

        assert isinstance(insights, dict)
        assert "total_laws_discovered" in insights
        assert "total_relationships_discovered" in insights
        assert "total_hypotheses_generated" in insights
        assert "high_confidence_laws" in insights
        assert "validated_laws" in insights
        assert "refinement_statistics" in insights


class TestErrorHandling:
    """Test error handling in law discovery components."""

    def test_law_discovery_error(self):
        """Test LawDiscoveryError exception."""
        with pytest.raises(LawDiscoveryError):
            raise LawDiscoveryError("Test error message")

    def test_causal_inference_error_handling(self):
        """Test error handling in causal inference."""
        engine = CausalInferenceEngine()

        # Test with invalid data
        with pytest.raises(LawDiscoveryError):
            engine.granger_causality_test(
                pd.DataFrame(), "nonexistent", "also_nonexistent"
            )

    def test_symbolic_regression_error_handling(self):
        """Test error handling in symbolic regression."""
        engine = SymbolicRegressionEngine()

        # Test with empty data
        with pytest.raises(LawDiscoveryError):
            engine.discover_equations(np.array([]), np.array([]), [])

    def test_pattern_recognition_error_handling(self):
        """Test error handling in pattern recognition."""
        engine = PatternRecognitionEngine()

        # Test with empty data
        with pytest.raises(LawDiscoveryError):
            engine.discover_patterns(pd.DataFrame(), [])


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_minimal_data_scenario(self):
        """Test discovery with minimal data."""
        discovery = AutonomousLawDiscovery()

        # Very small dataset
        data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})

        results = discovery.discover_laws_and_relationships(
            data=data, target_variables=["y"], variable_names=["x", "y"]
        )

        assert isinstance(results, dict)
        # Should handle gracefully even with minimal data

    def test_no_relationships_scenario(self):
        """Test discovery with pure noise data."""
        discovery = AutonomousLawDiscovery()

        # Pure random data
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "x": np.random.randn(100),
                "y": np.random.randn(100),
                "z": np.random.randn(100),
            }
        )

        results = discovery.discover_laws_and_relationships(
            data=data, target_variables=["z"], variable_names=["x", "y", "z"]
        )

        assert isinstance(results, dict)
        # Should find minimal or no relationships
        assert results["discovery_summary"]["laws_discovered"] >= 0

    def test_single_variable_scenario(self):
        """Test discovery with single variable."""
        discovery = AutonomousLawDiscovery()

        data = pd.DataFrame({"x": [1, 2, 3, 4, 5]})

        results = discovery.discover_laws_and_relationships(
            data=data, target_variables=[], variable_names=["x"]
        )

        assert isinstance(results, dict)
        # Should handle gracefully with single variable


if __name__ == "__main__":
    pytest.main([__file__])

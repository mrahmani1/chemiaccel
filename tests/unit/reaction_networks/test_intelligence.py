"""
Unit tests for reaction intelligence module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.chemi_accel.core.exceptions import ChemiAccelError
from src.chemi_accel.reaction_networks.intelligence import (
    CatalystSuggestion,
    CatalystSuggestionEngine,
    ExperimentalFeedbackProcessor,
    ExperimentalResult,
    ReactionConditions,
    ReactionIntelligenceEngine,
    ReactionScenario,
    WhatIfSimulator,
)


class TestReactionConditions:
    """Test ReactionConditions dataclass."""

    def test_default_initialization(self):
        """Test default initialization of ReactionConditions."""
        conditions = ReactionConditions()

        assert conditions.temperature is None
        assert conditions.pressure is None
        assert conditions.solvent is None
        assert conditions.catalyst is None
        assert conditions.additives == []

    def test_full_initialization(self):
        """Test full initialization with all parameters."""
        conditions = ReactionConditions(
            temperature=353.15,
            pressure=2.0,
            solvent="DMF",
            catalyst="Pd(PPh3)4",
            catalyst_loading=5.0,
            ph=7.0,
            reaction_time=12.0,
            atmosphere="N2",
            concentration=0.1,
            additives=["NaOAc"],
        )

        assert conditions.temperature == 353.15
        assert conditions.pressure == 2.0
        assert conditions.solvent == "DMF"
        assert conditions.catalyst == "Pd(PPh3)4"
        assert conditions.catalyst_loading == 5.0
        assert conditions.ph == 7.0
        assert conditions.reaction_time == 12.0
        assert conditions.atmosphere == "N2"
        assert conditions.concentration == 0.1
        assert conditions.additives == ["NaOAc"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        conditions = ReactionConditions(
            temperature=353.15, solvent="DMF", catalyst="Pd(PPh3)4"
        )

        result = conditions.to_dict()

        assert isinstance(result, dict)
        assert result["temperature"] == 353.15
        assert result["solvent"] == "DMF"
        assert result["catalyst"] == "Pd(PPh3)4"
        assert result["pressure"] is None


class TestCatalystSuggestion:
    """Test CatalystSuggestion dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        suggestion = CatalystSuggestion(catalyst_name="Pd(PPh3)4")

        assert suggestion.catalyst_name == "Pd(PPh3)4"
        assert suggestion.catalyst_smiles is None
        assert suggestion.confidence_score == 0.0
        assert suggestion.literature_precedents == 0
        assert suggestion.compatible_solvents == []

    def test_full_initialization(self):
        """Test full initialization."""
        suggestion = CatalystSuggestion(
            catalyst_name="Pd(PPh3)4",
            catalyst_smiles="[Pd]",
            predicted_yield=0.85,
            predicted_selectivity=0.90,
            confidence_score=0.75,
            literature_precedents=150,
            recommended_loading=5.0,
            compatible_solvents=["DMF", "THF"],
            notes="Excellent for cross-coupling",
        )

        assert suggestion.catalyst_name == "Pd(PPh3)4"
        assert suggestion.catalyst_smiles == "[Pd]"
        assert suggestion.predicted_yield == 0.85
        assert suggestion.predicted_selectivity == 0.90
        assert suggestion.confidence_score == 0.75
        assert suggestion.literature_precedents == 150
        assert suggestion.recommended_loading == 5.0
        assert suggestion.compatible_solvents == ["DMF", "THF"]
        assert suggestion.notes == "Excellent for cross-coupling"


class TestReactionIntelligenceEngine:
    """Test ReactionIntelligenceEngine class."""

    @pytest.fixture
    def intelligence_engine(self):
        """Create ReactionIntelligenceEngine instance."""
        return ReactionIntelligenceEngine()

    def test_initialization(self, intelligence_engine):
        """Test proper initialization."""
        assert intelligence_engine.kg_engine is None
        assert intelligence_engine.mechanism_predictor is not None
        assert intelligence_engine.pathway_generator is not None
        assert intelligence_engine.experimental_data == []

    def test_analyze_reaction_requirements_valid_smiles(self, intelligence_engine):
        """Test reaction analysis with valid SMILES."""
        reactants = ["CC(=O)O", "CCO"]  # Acetic acid + Ethanol
        products = ["CC(=O)OCC"]  # Ethyl acetate

        result = intelligence_engine.analyze_reaction_requirements(reactants, products)

        assert isinstance(result, dict)
        assert "bond_changes" in result
        assert "reaction_type" in result
        assert "thermodynamics" in result
        assert "complexity_score" in result
        assert isinstance(result["complexity_score"], int)

    def test_analyze_reaction_requirements_invalid_smiles(self, intelligence_engine):
        """Test reaction analysis with invalid SMILES."""
        reactants = ["INVALID_SMILES"]
        products = ["CC(=O)OCC"]

        with pytest.raises(ChemiAccelError):
            intelligence_engine.analyze_reaction_requirements(reactants, products)

    def test_classify_reaction_type(self, intelligence_engine):
        """Test reaction type classification."""
        # Test coupling reaction
        coupling_changes = [{"bonds_formed": [1, 2], "bonds_broken": []}]
        assert (
            intelligence_engine._classify_reaction_type(coupling_changes) == "coupling"
        )

        # Test cleavage reaction
        cleavage_changes = [{"bonds_formed": [], "bonds_broken": [1, 2]}]
        assert (
            intelligence_engine._classify_reaction_type(cleavage_changes) == "cleavage"
        )

        # Test rearrangement
        rearrangement_changes = [{"bonds_formed": [1], "bonds_broken": [1]}]
        assert (
            intelligence_engine._classify_reaction_type(rearrangement_changes)
            == "rearrangement"
        )

        # Test unknown
        assert intelligence_engine._classify_reaction_type([]) == "unknown"


class TestCatalystSuggestionEngine:
    """Test CatalystSuggestionEngine class."""

    @pytest.fixture
    def suggestion_engine(self):
        """Create CatalystSuggestionEngine instance."""
        intelligence_engine = ReactionIntelligenceEngine()
        return CatalystSuggestionEngine(intelligence_engine)

    def test_initialization(self, suggestion_engine):
        """Test proper initialization."""
        assert suggestion_engine.intelligence_engine is not None
        assert len(suggestion_engine.catalyst_database) > 0

    def test_suggest_catalysts(self, suggestion_engine):
        """Test catalyst suggestion."""
        reactants = ["CC(=O)O", "CCBr"]  # Coupling reaction
        products = ["CC(=O)CC"]

        suggestions = suggestion_engine.suggest_catalysts(
            reactants, products, max_suggestions=3
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3

        for suggestion in suggestions:
            assert isinstance(suggestion, CatalystSuggestion)
            assert suggestion.catalyst_name
            assert 0 <= suggestion.confidence_score <= 1

    def test_suggest_conditions(self, suggestion_engine):
        """Test condition suggestion."""
        reactants = ["CC(=O)O", "CCBr"]
        products = ["CC(=O)CC"]

        conditions = suggestion_engine.suggest_conditions(reactants, products)

        assert isinstance(conditions, ReactionConditions)
        assert conditions.temperature is not None
        assert conditions.temperature > 0
        assert conditions.solvent is not None
        assert conditions.pressure is not None
        assert conditions.reaction_time is not None
        assert conditions.concentration is not None

    def test_suggest_conditions_with_catalyst(self, suggestion_engine):
        """Test condition suggestion with specific catalyst."""
        reactants = ["CC(=O)O", "CCBr"]
        products = ["CC(=O)CC"]
        catalyst = "Pd(PPh3)4"

        conditions = suggestion_engine.suggest_conditions(reactants, products, catalyst)

        assert conditions.catalyst == catalyst
        assert conditions.catalyst_loading is not None
        assert conditions.catalyst_loading > 0

    def test_filter_catalysts_by_type(self, suggestion_engine):
        """Test catalyst filtering by reaction type."""
        coupling_catalysts = suggestion_engine._filter_catalysts_by_type("coupling")

        assert isinstance(coupling_catalysts, list)
        assert len(coupling_catalysts) > 0

        for catalyst in coupling_catalysts:
            assert "coupling" in catalyst["types"]

    def test_score_catalyst(self, suggestion_engine):
        """Test catalyst scoring."""
        catalyst_info = {"name": "Pd(PPh3)4", "precedents": 500, "types": ["coupling"]}
        analysis = {"reaction_type": "coupling", "complexity_score": 2}

        score = suggestion_engine._score_catalyst(catalyst_info, analysis)

        assert isinstance(score, float)
        assert 0 <= score <= 1


class TestWhatIfSimulator:
    """Test WhatIfSimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create WhatIfSimulator instance."""
        intelligence_engine = ReactionIntelligenceEngine()
        return WhatIfSimulator(intelligence_engine)

    @pytest.fixture
    def base_scenario(self):
        """Create base reaction scenario."""
        conditions = ReactionConditions(
            temperature=298.15, solvent="DMF", catalyst="Pd(PPh3)4"
        )
        return ReactionScenario(
            reactants=["CC(=O)O", "CCBr"], products=["CC(=O)CC"], conditions=conditions
        )

    def test_set_base_scenario(self, simulator, base_scenario):
        """Test setting base scenario."""
        simulator.set_base_scenario(base_scenario)

        assert simulator.base_scenario == base_scenario

    def test_simulate_condition_change(self, simulator, base_scenario):
        """Test simulating condition changes."""
        simulator.set_base_scenario(base_scenario)

        modified_scenario = simulator.simulate_condition_change("temperature", 353.15)

        assert modified_scenario.conditions.temperature == 353.15
        assert modified_scenario.conditions.solvent == base_scenario.conditions.solvent
        assert modified_scenario.predicted_yield is not None
        assert modified_scenario.predicted_selectivity is not None
        assert modified_scenario.reaction_rate is not None

    def test_simulate_condition_change_no_base_scenario(self, simulator):
        """Test simulating condition change without base scenario."""
        with pytest.raises(ChemiAccelError):
            simulator.simulate_condition_change("temperature", 353.15)

    def test_simulate_condition_change_invalid_parameter(
        self, simulator, base_scenario
    ):
        """Test simulating change with invalid parameter."""
        simulator.set_base_scenario(base_scenario)

        with pytest.raises(ChemiAccelError):
            simulator.simulate_condition_change("invalid_parameter", 100)

    def test_simulate_catalyst_change(self, simulator, base_scenario):
        """Test simulating catalyst change."""
        simulator.set_base_scenario(base_scenario)

        modified_scenario = simulator.simulate_catalyst_change("CuI")

        assert modified_scenario.conditions.catalyst == "CuI"

    def test_simulate_temperature_sweep(self, simulator, base_scenario):
        """Test temperature sweep simulation."""
        simulator.set_base_scenario(base_scenario)

        scenarios = simulator.simulate_temperature_sweep((298.15, 373.15), num_points=5)

        assert len(scenarios) == 5

        temperatures = [s.conditions.temperature for s in scenarios]
        assert temperatures == sorted(temperatures)  # Should be in ascending order
        assert min(temperatures) >= 298.15
        assert max(temperatures) <= 373.15

    def test_compare_scenarios(self, simulator, base_scenario):
        """Test scenario comparison."""
        simulator.set_base_scenario(base_scenario)

        scenarios = [
            simulator.simulate_condition_change("temperature", 298.15),
            simulator.simulate_condition_change("temperature", 353.15),
            simulator.simulate_condition_change("temperature", 373.15),
        ]

        comparison_df = simulator.compare_scenarios(scenarios)

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 3
        assert "scenario_id" in comparison_df.columns
        assert "temperature" in comparison_df.columns
        assert "predicted_yield" in comparison_df.columns


class TestExperimentalFeedbackProcessor:
    """Test ExperimentalFeedbackProcessor class."""

    @pytest.fixture
    def feedback_processor(self):
        """Create ExperimentalFeedbackProcessor instance."""
        intelligence_engine = ReactionIntelligenceEngine()
        return ExperimentalFeedbackProcessor(intelligence_engine)

    @pytest.fixture
    def experimental_result(self):
        """Create experimental result."""
        conditions = ReactionConditions(temperature=353.15, catalyst="Pd(PPh3)4")
        scenario = ReactionScenario(
            reactants=["CC(=O)O", "CCBr"],
            products=["CC(=O)CC"],
            conditions=conditions,
            predicted_yield=0.80,
            predicted_selectivity=0.85,
        )
        return ExperimentalResult(
            scenario=scenario, actual_yield=0.75, actual_selectivity=0.90
        )

    def test_ingest_experimental_result(self, feedback_processor, experimental_result):
        """Test ingesting experimental results."""
        initial_count = len(feedback_processor.intelligence_engine.experimental_data)

        feedback_processor.ingest_experimental_result(experimental_result)

        assert (
            len(feedback_processor.intelligence_engine.experimental_data)
            == initial_count + 1
        )
        assert len(feedback_processor.prediction_errors) == 1

        error = feedback_processor.prediction_errors[0]
        assert "yield_error" in error
        assert "selectivity_error" in error
        assert error["yield_error"] == abs(0.75 - 0.80)
        assert error["selectivity_error"] == abs(0.90 - 0.85)

    def test_analyze_prediction_accuracy_empty(self, feedback_processor):
        """Test accuracy analysis with no data."""
        analysis = feedback_processor.analyze_prediction_accuracy()

        assert analysis == {}

    def test_analyze_prediction_accuracy_with_data(
        self, feedback_processor, experimental_result
    ):
        """Test accuracy analysis with data."""
        feedback_processor.ingest_experimental_result(experimental_result)

        analysis = feedback_processor.analyze_prediction_accuracy()

        assert "mean_yield_error" in analysis
        assert "std_yield_error" in analysis
        assert "num_experiments" in analysis
        assert "mean_selectivity_error" in analysis
        assert "std_selectivity_error" in analysis

        assert analysis["num_experiments"] == 1
        assert analysis["mean_yield_error"] == abs(0.75 - 0.80)

    def test_refine_model_weights_insufficient_data(self, feedback_processor):
        """Test weight refinement with insufficient data."""
        initial_weights = feedback_processor.model_weights.copy()

        feedback_processor.refine_model_weights()

        # Weights should be unchanged with insufficient data
        assert feedback_processor.model_weights == initial_weights

    def test_suggest_next_experiments(self, feedback_processor):
        """Test next experiment suggestions."""
        reactants = ["CC(=O)O", "CCBr"]
        products = ["CC(=O)CC"]

        suggestions = feedback_processor.suggest_next_experiments(
            (reactants, products), num_suggestions=3
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3

        for suggestion in suggestions:
            assert isinstance(suggestion, ReactionScenario)
            assert suggestion.reactants == reactants
            assert suggestion.products == products
            assert suggestion.conditions is not None


class TestIntegration:
    """Integration tests for reaction intelligence components."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from analysis to suggestion."""
        # Initialize components
        intelligence_engine = ReactionIntelligenceEngine()
        suggestion_engine = CatalystSuggestionEngine(intelligence_engine)
        simulator = WhatIfSimulator(intelligence_engine)
        feedback_processor = ExperimentalFeedbackProcessor(intelligence_engine)

        # Define reaction
        reactants = ["CC(=O)O", "CCBr"]
        products = ["CC(=O)CC"]

        # Analyze reaction
        analysis = intelligence_engine.analyze_reaction_requirements(
            reactants, products
        )
        assert "reaction_type" in analysis

        # Get catalyst suggestions
        catalyst_suggestions = suggestion_engine.suggest_catalysts(reactants, products)
        assert len(catalyst_suggestions) > 0

        # Get condition suggestions
        conditions = suggestion_engine.suggest_conditions(reactants, products)
        assert conditions.temperature is not None

        # Set up what-if simulation
        scenario = ReactionScenario(
            reactants=reactants,
            products=products,
            conditions=conditions,
            predicted_yield=0.80,
        )
        simulator.set_base_scenario(scenario)

        # Simulate temperature change
        modified_scenario = simulator.simulate_condition_change("temperature", 373.15)
        assert modified_scenario.conditions.temperature == 373.15

        # Create experimental result
        experimental_result = ExperimentalResult(
            scenario=modified_scenario, actual_yield=0.75
        )

        # Ingest feedback
        feedback_processor.ingest_experimental_result(experimental_result)
        assert len(feedback_processor.prediction_errors) == 1

        # Analyze accuracy
        accuracy = feedback_processor.analyze_prediction_accuracy()
        assert "mean_yield_error" in accuracy

    @patch("src.chemi_accel.reaction_networks.intelligence.np.random.random")
    def test_prediction_reproducibility(self, mock_random):
        """Test that predictions are reproducible with fixed random seed."""
        mock_random.return_value = 0.5

        intelligence_engine = ReactionIntelligenceEngine()
        simulator = WhatIfSimulator(intelligence_engine)

        conditions = ReactionConditions(temperature=298.15, catalyst="Pd(PPh3)4")
        scenario = ReactionScenario(
            reactants=["CC(=O)O", "CCBr"], products=["CC(=O)CC"], conditions=conditions
        )

        simulator.set_base_scenario(scenario)
        scenario1 = simulator.simulate_condition_change("temperature", 353.15)
        scenario2 = simulator.simulate_condition_change("temperature", 353.15)

        # With fixed random seed, predictions should be identical
        assert scenario1.predicted_yield == scenario2.predicted_yield
        assert scenario1.reaction_rate == scenario2.reaction_rate


if __name__ == "__main__":
    pytest.main([__file__])

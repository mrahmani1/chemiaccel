"""
Unit tests for knowledge graph integration module.
"""

from unittest.mock import Mock, patch

import pytest

from src.chemi_accel.reaction_networks.intelligence import (
    CatalystSuggestion,
    ReactionConditions,
)
from src.chemi_accel.reaction_networks.kg_integration import (
    CatalystKnowledgeRetriever,
    KnowledgeGraphEnhancedSuggestionEngine,
    LiteratureBasedOptimizer,
)


class TestCatalystKnowledgeRetriever:
    """Test CatalystKnowledgeRetriever class."""

    @pytest.fixture
    def mock_kg_engine(self):
        """Create mock knowledge graph engine."""
        mock_engine = Mock()
        return mock_engine

    @pytest.fixture
    def retriever(self, mock_kg_engine):
        """Create CatalystKnowledgeRetriever instance."""
        return CatalystKnowledgeRetriever(mock_kg_engine)

    def test_initialization(self, retriever, mock_kg_engine):
        """Test proper initialization."""
        assert retriever.kg_engine == mock_kg_engine

    def test_query_catalyst_performance_success(self, retriever, mock_kg_engine):
        """Test successful catalyst performance query."""
        # Mock SPARQL results
        mock_results = [
            {
                "catalystName": "Pd(PPh3)4",
                "catalyst": "http://example.org/Pd",
                "yield": "0.85",
                "selectivity": "0.90",
                "conditions": "DMF, 80C",
                "reference": "DOI:10.1021/example",
            },
            {
                "catalystName": "CuI",
                "catalyst": "http://example.org/Cu",
                "yield": "0.70",
                "selectivity": None,
                "conditions": "acetonitrile, RT",
                "reference": "DOI:10.1021/example2",
            },
        ]
        mock_kg_engine.execute_sparql.return_value = mock_results

        results = retriever.query_catalyst_performance("coupling")

        assert len(results) == 2
        assert results[0]["catalyst_name"] == "Pd(PPh3)4"
        assert results[0]["yield"] == 0.85
        assert results[0]["selectivity"] == 0.90
        assert results[1]["catalyst_name"] == "CuI"
        assert results[1]["yield"] == 0.70
        assert results[1]["selectivity"] is None

    def test_query_catalyst_performance_empty_results(self, retriever, mock_kg_engine):
        """Test catalyst performance query with empty results."""
        mock_kg_engine.execute_sparql.return_value = []

        results = retriever.query_catalyst_performance("unknown_reaction")

        assert results == []

    def test_query_catalyst_performance_error(self, retriever, mock_kg_engine):
        """Test catalyst performance query with error."""
        mock_kg_engine.execute_sparql.side_effect = Exception("SPARQL error")

        results = retriever.query_catalyst_performance("coupling")

        assert results == []

    def test_query_reaction_conditions(self, retriever, mock_kg_engine):
        """Test reaction conditions query."""
        mock_results = [
            {
                "reaction": "http://example.org/reaction1",
                "temperature": "353.15",
                "solvent": "DMF",
                "catalyst": "Pd(PPh3)4",
                "yield": "0.80",
                "reference": "DOI:10.1021/example",
            }
        ]
        mock_kg_engine.execute_sparql.return_value = mock_results

        reactants = ["CC(=O)O"]
        products = ["CC(=O)OCC"]

        results = retriever.query_reaction_conditions(reactants, products)

        assert len(results) == 1
        assert results[0]["temperature"] == 353.15
        assert results[0]["solvent"] == "DMF"
        assert results[0]["catalyst"] == "Pd(PPh3)4"
        assert results[0]["yield"] == 0.80

    def test_query_catalyst_literature_precedents(self, retriever, mock_kg_engine):
        """Test catalyst literature precedents query."""
        mock_results = [{"count": "25"}]
        mock_kg_engine.execute_sparql.return_value = mock_results

        precedents = retriever.query_catalyst_literature_precedents(
            "Pd(PPh3)4", "coupling"
        )

        assert precedents == 25

    def test_query_catalyst_literature_precedents_empty(
        self, retriever, mock_kg_engine
    ):
        """Test catalyst precedents query with empty results."""
        mock_kg_engine.execute_sparql.return_value = []

        precedents = retriever.query_catalyst_literature_precedents("Unknown_Catalyst")

        assert precedents == 0

    def test_query_solvent_compatibility(self, retriever, mock_kg_engine):
        """Test solvent compatibility query."""
        mock_results = [
            {"solvent": "DMF"},
            {"solvent": "THF"},
            {"solvent": "acetonitrile"},
        ]
        mock_kg_engine.execute_sparql.return_value = mock_results

        solvents = retriever.query_solvent_compatibility("Pd(PPh3)4")

        assert solvents == ["DMF", "THF", "acetonitrile"]

    def test_build_substrate_filter(self, retriever):
        """Test substrate filter building."""
        substrate_classes = ["aryl_halide", "alkene"]

        filter_str = retriever._build_substrate_filter(substrate_classes)

        assert "FILTER" in filter_str
        assert "aryl_halide" in filter_str
        assert "alkene" in filter_str
        assert "||" in filter_str

    def test_build_substrate_filter_empty(self, retriever):
        """Test substrate filter with empty list."""
        filter_str = retriever._build_substrate_filter([])

        assert filter_str == ""

    def test_process_catalyst_results_invalid_data(self, retriever):
        """Test processing results with invalid data."""
        invalid_results = [
            {
                "catalystName": "Pd(PPh3)4",
                "yield": "invalid_yield",  # Should cause ValueError
                "selectivity": "0.90",
            },
            {
                "catalystName": "CuI",
                "yield": "0.75",
                "selectivity": "invalid_selectivity",  # Should cause ValueError
            },
        ]

        processed = retriever._process_catalyst_results(invalid_results)

        # Should skip invalid entries and return valid ones
        assert len(processed) == 1
        assert processed[0]["catalyst_name"] == "CuI"
        assert processed[0]["yield"] == 0.75


class TestLiteratureBasedOptimizer:
    """Test LiteratureBasedOptimizer class."""

    @pytest.fixture
    def mock_retriever(self):
        """Create mock CatalystKnowledgeRetriever."""
        return Mock(spec=CatalystKnowledgeRetriever)

    @pytest.fixture
    def optimizer(self, mock_retriever):
        """Create LiteratureBasedOptimizer instance."""
        return LiteratureBasedOptimizer(mock_retriever)

    def test_initialization(self, optimizer, mock_retriever):
        """Test proper initialization."""
        assert optimizer.kg_retriever == mock_retriever

    def test_optimize_catalyst_selection_success(self, optimizer, mock_retriever):
        """Test successful catalyst selection optimization."""
        # Mock performance data
        mock_performance_data = [
            {
                "catalyst_name": "Pd(PPh3)4",
                "yield": 0.85,
                "selectivity": 0.90,
                "reference": "ref1",
            },
            {
                "catalyst_name": "Pd(PPh3)4",
                "yield": 0.80,
                "selectivity": 0.85,
                "reference": "ref2",
            },
            {
                "catalyst_name": "CuI",
                "yield": 0.70,
                "selectivity": 0.75,
                "reference": "ref3",
            },
        ]
        mock_retriever.query_catalyst_performance.return_value = mock_performance_data
        mock_retriever.query_solvent_compatibility.return_value = ["DMF", "THF"]

        suggestions = optimizer.optimize_catalyst_selection(
            "coupling", max_suggestions=2
        )

        assert len(suggestions) <= 2
        assert all(isinstance(s, CatalystSuggestion) for s in suggestions)

        # Should be sorted by confidence score
        if len(suggestions) > 1:
            assert suggestions[0].confidence_score >= suggestions[1].confidence_score

        # Check Pd catalyst (should have higher score due to multiple references)
        pd_suggestion = next(
            (s for s in suggestions if s.catalyst_name == "Pd(PPh3)4"), None
        )
        assert pd_suggestion is not None
        assert pd_suggestion.predicted_yield == 0.825  # Average of 0.85 and 0.80
        assert pd_suggestion.predicted_selectivity == 0.875  # Average of 0.90 and 0.85
        assert pd_suggestion.literature_precedents == 2

    def test_optimize_catalyst_selection_no_data(self, optimizer, mock_retriever):
        """Test catalyst selection with no literature data."""
        mock_retriever.query_catalyst_performance.return_value = []

        suggestions = optimizer.optimize_catalyst_selection("unknown_reaction")

        assert suggestions == []

    def test_optimize_reaction_conditions_success(self, optimizer, mock_retriever):
        """Test successful reaction condition optimization."""
        mock_similar_reactions = [
            {
                "temperature": 353.15,
                "solvent": "DMF",
                "catalyst": "Pd(PPh3)4",
                "yield": 0.85,
            },
            {
                "temperature": 373.15,
                "solvent": "DMF",
                "catalyst": "Pd(PPh3)4",
                "yield": 0.80,
            },
            {"temperature": 353.15, "solvent": "THF", "catalyst": "CuI", "yield": 0.70},
        ]
        mock_retriever.query_reaction_conditions.return_value = mock_similar_reactions

        reactants = ["CC(=O)O"]
        products = ["CC(=O)OCC"]

        conditions = optimizer.optimize_reaction_conditions(reactants, products)

        assert isinstance(conditions, ReactionConditions)
        assert conditions.temperature == 363.15  # Average of 353.15 and 373.15
        assert conditions.solvent == "DMF"  # Most common
        assert conditions.catalyst == "Pd(PPh3)4"  # Best performing

    def test_optimize_reaction_conditions_no_data(self, optimizer, mock_retriever):
        """Test condition optimization with no data."""
        mock_retriever.query_reaction_conditions.return_value = []

        reactants = ["CC(=O)O"]
        products = ["CC(=O)OCC"]

        conditions = optimizer.optimize_reaction_conditions(reactants, products)

        assert isinstance(conditions, ReactionConditions)
        # Should return default/empty conditions
        assert conditions.temperature is None
        assert conditions.solvent is None
        assert conditions.catalyst is None

    def test_calculate_confidence_score(self, optimizer):
        """Test confidence score calculation."""
        # High yield, high consistency, many precedents
        stats_high = {"yields": [0.85, 0.83, 0.87]}
        score_high = optimizer._calculate_confidence_score(stats_high, 100)

        # Low yield, low consistency, few precedents
        stats_low = {"yields": [0.30, 0.60, 0.20]}
        score_low = optimizer._calculate_confidence_score(stats_low, 5)

        assert score_high > score_low
        assert 0 <= score_high <= 1
        assert 0 <= score_low <= 1

    def test_calculate_confidence_score_empty_yields(self, optimizer):
        """Test confidence score with empty yields."""
        stats_empty = {"yields": []}
        score = optimizer._calculate_confidence_score(stats_empty, 10)

        assert score == 0.0


class TestKnowledgeGraphEnhancedSuggestionEngine:
    """Test KnowledgeGraphEnhancedSuggestionEngine class."""

    @pytest.fixture
    def mock_kg_engine(self):
        """Create mock knowledge graph engine."""
        return Mock()

    @pytest.fixture
    def enhanced_engine(self, mock_kg_engine):
        """Create KnowledgeGraphEnhancedSuggestionEngine instance."""
        return KnowledgeGraphEnhancedSuggestionEngine(mock_kg_engine)

    def test_initialization(self, enhanced_engine):
        """Test proper initialization."""
        assert enhanced_engine.kg_retriever is not None
        assert enhanced_engine.literature_optimizer is not None

    @patch.object(LiteratureBasedOptimizer, "optimize_catalyst_selection")
    def test_suggest_catalysts_with_literature(self, mock_optimize, enhanced_engine):
        """Test catalyst suggestion with literature integration."""
        mock_suggestions = [
            CatalystSuggestion(catalyst_name="Pd(PPh3)4", confidence_score=0.85),
            CatalystSuggestion(catalyst_name="CuI", confidence_score=0.70),
        ]
        mock_optimize.return_value = mock_suggestions

        suggestions = enhanced_engine.suggest_catalysts_with_literature(
            "coupling", ["aryl_halide"], max_suggestions=2
        )

        assert suggestions == mock_suggestions
        mock_optimize.assert_called_once_with("coupling", ["aryl_halide"], 2)

    @patch.object(LiteratureBasedOptimizer, "optimize_reaction_conditions")
    def test_suggest_conditions_with_literature(self, mock_optimize, enhanced_engine):
        """Test condition suggestion with literature integration."""
        mock_conditions = ReactionConditions(temperature=353.15, solvent="DMF")
        mock_optimize.return_value = mock_conditions

        reactants = ["CC(=O)O"]
        products = ["CC(=O)OCC"]

        conditions = enhanced_engine.suggest_conditions_with_literature(
            reactants, products
        )

        assert conditions == mock_conditions
        mock_optimize.assert_called_once_with(reactants, products)

    def test_validate_catalyst_choice(self, enhanced_engine):
        """Test catalyst choice validation."""
        # Mock the retriever methods
        enhanced_engine.kg_retriever.query_catalyst_literature_precedents = Mock(
            return_value=25
        )
        enhanced_engine.kg_retriever.query_solvent_compatibility = Mock(
            return_value=["DMF", "THF"]
        )
        enhanced_engine.kg_retriever.query_catalyst_performance = Mock(
            return_value=[
                {"catalyst_name": "Pd(PPh3)4", "yield": 0.85},
                {"catalyst_name": "Pd(PPh3)4", "yield": 0.80},
            ]
        )

        validation = enhanced_engine.validate_catalyst_choice("Pd(PPh3)4", "coupling")

        assert isinstance(validation, dict)
        assert validation["literature_precedents"] == 25
        assert validation["compatible_solvents"] == ["DMF", "THF"]
        assert len(validation["performance_data"]) == 2
        assert "validation_score" in validation
        assert 0 <= validation["validation_score"] <= 1

    def test_calculate_validation_score_no_precedents(self, enhanced_engine):
        """Test validation score with no precedents."""
        score = enhanced_engine._calculate_validation_score(0, [])

        assert score == 0.0

    def test_calculate_validation_score_with_data(self, enhanced_engine):
        """Test validation score with precedents and performance data."""
        performance_data = [{"yield": 0.85}, {"yield": 0.80}]

        score = enhanced_engine._calculate_validation_score(50, performance_data)

        assert 0 < score <= 1
        # Should be sum of precedent score (0.6) and performance score (0.825 * 0.4 = 0.33)
        expected_score = 0.6 + 0.33
        assert abs(score - expected_score) < 0.01


class TestIntegration:
    """Integration tests for knowledge graph components."""

    def test_end_to_end_kg_workflow(self):
        """Test complete knowledge graph integration workflow."""
        # Mock knowledge graph engine
        mock_kg_engine = Mock()

        # Mock SPARQL results for catalyst performance
        mock_kg_engine.execute_sparql.side_effect = [
            # First call: catalyst performance
            [
                {
                    "catalystName": "Pd(PPh3)4",
                    "catalyst": "http://example.org/Pd",
                    "yield": "0.85",
                    "selectivity": "0.90",
                    "conditions": "DMF, 80C",
                    "reference": "ref1",
                }
            ],
            # Second call: solvent compatibility
            [{"solvent": "DMF"}, {"solvent": "THF"}],
            # Third call: reaction conditions
            [
                {
                    "reaction": "http://example.org/reaction1",
                    "temperature": "353.15",
                    "solvent": "DMF",
                    "catalyst": "Pd(PPh3)4",
                    "yield": "0.80",
                    "reference": "ref1",
                }
            ],
        ]

        # Create enhanced suggestion engine
        enhanced_engine = KnowledgeGraphEnhancedSuggestionEngine(mock_kg_engine)

        # Test catalyst suggestions
        catalyst_suggestions = enhanced_engine.suggest_catalysts_with_literature(
            "coupling", ["aryl_halide"], max_suggestions=3
        )

        assert len(catalyst_suggestions) <= 3
        assert all(isinstance(s, CatalystSuggestion) for s in catalyst_suggestions)

        # Test condition suggestions
        reactants = ["CC(=O)O", "CCBr"]
        products = ["CC(=O)CC"]

        conditions = enhanced_engine.suggest_conditions_with_literature(
            reactants, products
        )

        assert isinstance(conditions, ReactionConditions)

        # Test catalyst validation
        validation = enhanced_engine.validate_catalyst_choice("Pd(PPh3)4", "coupling")

        assert isinstance(validation, dict)
        assert "validation_score" in validation


if __name__ == "__main__":
    pytest.main([__file__])

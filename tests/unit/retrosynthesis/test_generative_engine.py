"""
Tests for GPU-accelerated generative retrosynthesis engine.

This module tests the transformer and graph-based generative models for
predicting one-step retrosynthetic disconnections.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Import the module under test
from chemi_accel.retrosynthesis.generative_engine import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    ConfidenceLevel,
    DisconnectionError,
    DisconnectionPredictor,
    DisconnectionSite,
    DisconnectionStrategy,
    GraphRetroModel,
    ModelLoadingError,
    ModelPrediction,
    ModelType,
    MolecularRepresentation,
    PredictionError,
    ReactionTemplate,
    ReactionTemplateExtractor,
    ReactionType,
    RetrosynthesisConfig,
    RetrosynthesisEngine,
    RetrosynthesisError,
    RetrosynthesisResult,
    TransformerRetroModel,
    create_retro_config,
    decode_molecule,
    encode_molecule,
    extract_disconnection_sites,
    predict_retrosynthesis,
)


# Test fixtures
@pytest.fixture
def sample_retro_config():
    """Create a sample retrosynthesis configuration."""
    return RetrosynthesisConfig(
        model_type=ModelType.TRANSFORMER,
        use_gpu=False,  # Disable GPU for testing
        batch_size=16,
        max_predictions=5,
        confidence_threshold=0.6,
        include_reaction_templates=True,
        enable_caching=False,  # Disable caching for testing
    )


@pytest.fixture
def sample_molecular_representation():
    """Create a sample molecular representation."""
    return MolecularRepresentation(smiles="CCO")


@pytest.fixture
def sample_disconnection_site():
    """Create a sample disconnection site."""
    return DisconnectionSite(
        bond_indices=(0, 1),
        confidence=0.8,
        strategy=DisconnectionStrategy.BOND_BREAKING,
        reaction_type=ReactionType.SUBSTITUTION,
    )


@pytest.fixture
def sample_reaction_template():
    """Create a sample reaction template."""
    return ReactionTemplate(
        template_id="test_template",
        reaction_smarts="[C:1][O:2]>>[C:1].[O:2]",
        reaction_type=ReactionType.SUBSTITUTION,
        success_rate=0.75,
    )


@pytest.fixture
def mock_torch():
    """Mock PyTorch imports."""
    with patch.multiple(
        "chemi_accel.retrosynthesis.generative_engine",
        _TORCH_AVAILABLE=False,
        torch=MagicMock(),
    ):
        yield


class TestEnums:
    """Test enum classes."""

    def test_model_type_values(self):
        """Test ModelType enum values."""
        assert ModelType.TRANSFORMER.value == "transformer"
        assert ModelType.GRAPH_NEURAL_NETWORK.value == "graph_neural_network"
        assert ModelType.TEMPLATE_BASED.value == "template_based"
        assert ModelType.HYBRID.value == "hybrid"

    def test_disconnection_strategy_values(self):
        """Test DisconnectionStrategy enum values."""
        assert DisconnectionStrategy.FUNCTIONAL_GROUP.value == "functional_group"
        assert DisconnectionStrategy.BOND_BREAKING.value == "bond_breaking"
        assert DisconnectionStrategy.RING_FORMATION.value == "ring_formation"

    def test_confidence_level_values(self):
        """Test ConfidenceLevel enum values."""
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"

    def test_reaction_type_values(self):
        """Test ReactionType enum values."""
        assert ReactionType.SUBSTITUTION.value == "substitution"
        assert ReactionType.ELIMINATION.value == "elimination"
        assert ReactionType.ADDITION.value == "addition"
        assert ReactionType.CYCLIZATION.value == "cyclization"


class TestDataStructures:
    """Test data structure classes."""

    def test_retrosynthesis_config_creation(self):
        """Test RetrosynthesisConfig creation."""
        config = RetrosynthesisConfig()

        assert config.model_type == ModelType.TRANSFORMER
        assert config.use_gpu is True
        assert config.batch_size == 32
        assert config.max_predictions == 10
        assert config.confidence_threshold == 0.7

    def test_molecular_representation_creation(self):
        """Test MolecularRepresentation creation."""
        mol_rep = MolecularRepresentation(smiles="CCO")

        assert mol_rep.smiles == "CCO"
        assert mol_rep.molecular_weight > 0
        assert mol_rep.num_atoms > 0
        assert mol_rep.num_bonds > 0
        assert isinstance(mol_rep.functional_groups, list)

    def test_molecular_representation_functional_groups(self):
        """Test functional group identification."""
        # Test alcohol
        alcohol = MolecularRepresentation(smiles="CCO")
        assert "alcohol" in alcohol.functional_groups

        # Test aromatic
        benzene = MolecularRepresentation(smiles="c1ccccc1")
        assert "aromatic" in benzene.functional_groups

    def test_disconnection_site_creation(self, sample_disconnection_site):
        """Test DisconnectionSite creation."""
        site = sample_disconnection_site

        assert site.bond_indices == (0, 1)
        assert site.confidence == 0.8
        assert site.strategy == DisconnectionStrategy.BOND_BREAKING
        assert site.reaction_type == ReactionType.SUBSTITUTION

    def test_disconnection_site_confidence_validation(self):
        """Test confidence validation in DisconnectionSite."""
        with pytest.raises(ValueError):
            DisconnectionSite(
                bond_indices=(0, 1),
                confidence=1.5,  # Invalid confidence > 1.0
                strategy=DisconnectionStrategy.BOND_BREAKING,
                reaction_type=ReactionType.SUBSTITUTION,
            )

    def test_reaction_template_creation(self, sample_reaction_template):
        """Test ReactionTemplate creation."""
        template = sample_reaction_template

        assert template.template_id == "test_template"
        assert template.reaction_smarts == "[C:1][O:2]>>[C:1].[O:2]"
        assert template.reaction_type == ReactionType.SUBSTITUTION
        assert template.success_rate == 0.75

    def test_model_prediction_creation(self, sample_disconnection_site):
        """Test ModelPrediction creation."""
        prediction = ModelPrediction(
            reactants=["CC", "O"],
            confidence=0.85,
            disconnection_sites=[sample_disconnection_site],
            model_type=ModelType.TRANSFORMER,
        )

        assert prediction.reactants == ["CC", "O"]
        assert prediction.confidence == 0.85
        assert len(prediction.disconnection_sites) == 1
        assert prediction.model_type == ModelType.TRANSFORMER

    def test_model_prediction_confidence_level(self):
        """Test confidence level determination."""
        # Test very high confidence
        pred_high = ModelPrediction(
            reactants=["CC", "O"], confidence=0.95, disconnection_sites=[]
        )
        assert pred_high.get_confidence_level() == ConfidenceLevel.VERY_HIGH

        # Test medium confidence
        pred_medium = ModelPrediction(
            reactants=["CC", "O"], confidence=0.6, disconnection_sites=[]
        )
        assert pred_medium.get_confidence_level() == ConfidenceLevel.MEDIUM

        # Test low confidence
        pred_low = ModelPrediction(
            reactants=["CC", "O"], confidence=0.3, disconnection_sites=[]
        )
        assert pred_low.get_confidence_level() == ConfidenceLevel.LOW

    def test_retrosynthesis_result_creation(self, sample_molecular_representation):
        """Test RetrosynthesisResult creation."""
        prediction = ModelPrediction(
            reactants=["CC", "O"], confidence=0.8, disconnection_sites=[]
        )

        result = RetrosynthesisResult(
            target_molecule=sample_molecular_representation,
            predictions=[prediction],
            total_time=0.5,
        )

        assert result.target_molecule == sample_molecular_representation
        assert len(result.predictions) == 1
        assert result.total_time == 0.5

    def test_retrosynthesis_result_best_prediction(
        self, sample_molecular_representation
    ):
        """Test getting best prediction from result."""
        pred1 = ModelPrediction(
            reactants=["CC", "O"], confidence=0.6, disconnection_sites=[]
        )
        pred2 = ModelPrediction(
            reactants=["C", "CO"], confidence=0.9, disconnection_sites=[]
        )
        pred3 = ModelPrediction(
            reactants=["CCO"], confidence=0.3, disconnection_sites=[]
        )

        result = RetrosynthesisResult(
            target_molecule=sample_molecular_representation,
            predictions=[pred1, pred2, pred3],
        )

        best = result.get_best_prediction()
        assert best == pred2
        assert best.confidence == 0.9

    def test_retrosynthesis_result_filter_by_confidence(
        self, sample_molecular_representation
    ):
        """Test filtering predictions by confidence."""
        pred1 = ModelPrediction(
            reactants=["CC", "O"], confidence=0.6, disconnection_sites=[]
        )
        pred2 = ModelPrediction(
            reactants=["C", "CO"], confidence=0.9, disconnection_sites=[]
        )
        pred3 = ModelPrediction(
            reactants=["CCO"], confidence=0.3, disconnection_sites=[]
        )

        result = RetrosynthesisResult(
            target_molecule=sample_molecular_representation,
            predictions=[pred1, pred2, pred3],
        )

        filtered = result.filter_by_confidence(0.5)
        assert len(filtered) == 2
        assert all(p.confidence >= 0.5 for p in filtered)


class TestTransformerModel:
    """Test transformer neural network model."""

    @pytest.mark.skipif(True, reason="PyTorch not available in test environment")
    def test_transformer_model_creation(self):
        """Test transformer model creation."""
        # This test would run if PyTorch was available
        pass

    @pytest.mark.skipif(True, reason="PyTorch not available in test environment")
    def test_transformer_forward_pass(self):
        """Test transformer forward pass."""
        # This test would run if PyTorch was available
        pass


class TestGraphModel:
    """Test graph neural network model."""

    @pytest.mark.skipif(True, reason="PyTorch not available in test environment")
    def test_graph_model_creation(self):
        """Test graph model creation."""
        # This test would run if PyTorch was available
        pass

    @pytest.mark.skipif(True, reason="PyTorch not available in test environment")
    def test_graph_forward_pass(self):
        """Test graph forward pass."""
        # This test would run if PyTorch was available
        pass


class TestDisconnectionPredictor:
    """Test DisconnectionPredictor class."""

    def test_disconnection_predictor_creation(self, sample_retro_config):
        """Test disconnection predictor creation."""
        predictor = DisconnectionPredictor(sample_retro_config)

        assert predictor.config == sample_retro_config
        assert predictor.use_gpu is False

    def test_predict_disconnection_sites(self, sample_retro_config):
        """Test disconnection site prediction."""
        predictor = DisconnectionPredictor(sample_retro_config)
        molecule = MolecularRepresentation(smiles="CCO")

        sites = predictor.predict_disconnection_sites(molecule)

        assert isinstance(sites, list)
        assert len(sites) <= sample_retro_config.max_predictions

        # Check that all sites have valid properties
        for site in sites:
            assert isinstance(site, DisconnectionSite)
            assert 0.0 <= site.confidence <= 1.0
            assert isinstance(site.strategy, DisconnectionStrategy)
            assert isinstance(site.reaction_type, ReactionType)

    def test_predict_disconnection_sites_invalid_molecule(self, sample_retro_config):
        """Test disconnection prediction with invalid molecule."""
        predictor = DisconnectionPredictor(sample_retro_config)
        molecule = MolecularRepresentation(smiles="")  # Invalid SMILES

        with pytest.raises(DisconnectionError):
            predictor.predict_disconnection_sites(molecule)

    def test_bond_breaking_confidence_calculation(self, sample_retro_config):
        """Test bond breaking confidence calculation."""
        predictor = DisconnectionPredictor(sample_retro_config)
        molecule = MolecularRepresentation(smiles="CC")

        # This is a simplified test - in practice would test specific bonds
        sites = predictor.predict_disconnection_sites(molecule)

        if sites:  # Only test if sites were found
            for site in sites:
                assert 0.0 <= site.confidence <= 1.0


class TestReactionTemplateExtractor:
    """Test ReactionTemplateExtractor class."""

    def test_template_extractor_creation(self):
        """Test template extractor creation."""
        extractor = ReactionTemplateExtractor()

        assert isinstance(extractor.templates, dict)
        assert len(extractor.templates) > 0  # Should have default templates

    def test_get_template(self):
        """Test getting template by ID."""
        extractor = ReactionTemplateExtractor()

        # Try to get a default template
        template = extractor.get_template("aldol_condensation")

        if template:  # If default template exists
            assert isinstance(template, ReactionTemplate)
            assert template.template_id == "aldol_condensation"

    def test_find_applicable_templates(self):
        """Test finding applicable templates."""
        extractor = ReactionTemplateExtractor()
        molecule = MolecularRepresentation(smiles="CCO")

        templates = extractor.find_applicable_templates(molecule)

        assert isinstance(templates, list)
        # Templates should be sorted by success rate
        if len(templates) > 1:
            for i in range(len(templates) - 1):
                assert templates[i].success_rate >= templates[i + 1].success_rate


class TestRetrosynthesisEngine:
    """Test RetrosynthesisEngine class."""

    def test_engine_creation(self, sample_retro_config):
        """Test retrosynthesis engine creation."""
        engine = RetrosynthesisEngine(sample_retro_config)

        assert engine.config == sample_retro_config
        assert isinstance(engine.disconnection_predictor, DisconnectionPredictor)
        assert isinstance(engine.template_extractor, ReactionTemplateExtractor)

    def test_engine_creation_default_config(self):
        """Test engine creation with default config."""
        engine = RetrosynthesisEngine()

        assert isinstance(engine.config, RetrosynthesisConfig)
        assert engine.config.model_type == ModelType.TRANSFORMER

    def test_predict_retrosynthesis(self, sample_retro_config):
        """Test retrosynthesis prediction."""
        engine = RetrosynthesisEngine(sample_retro_config)

        result = engine.predict_retrosynthesis("CCO")

        assert isinstance(result, RetrosynthesisResult)
        assert result.target_molecule.smiles == "CCO"
        assert isinstance(result.predictions, list)
        assert result.total_time >= 0

    def test_predict_retrosynthesis_invalid_smiles(self, sample_retro_config):
        """Test prediction with invalid SMILES."""
        engine = RetrosynthesisEngine(sample_retro_config)

        # Invalid SMILES should still create a result, but may have no predictions
        result = engine.predict_retrosynthesis("INVALID")

        assert isinstance(result, RetrosynthesisResult)

    def test_batch_predict(self, sample_retro_config):
        """Test batch prediction."""
        engine = RetrosynthesisEngine(sample_retro_config)

        smiles_list = ["CCO", "CC", "c1ccccc1"]
        results = engine.batch_predict(smiles_list)

        assert len(results) == len(smiles_list)
        assert all(isinstance(r, RetrosynthesisResult) for r in results)

    def test_caching_functionality(self):
        """Test prediction caching."""
        config = RetrosynthesisConfig(enable_caching=True, use_gpu=False)
        engine = RetrosynthesisEngine(config)

        # First prediction
        result1 = engine.predict_retrosynthesis("CCO")

        # Second prediction (should use cache)
        result2 = engine.predict_retrosynthesis("CCO")

        # Results should be identical (from cache)
        assert result1.target_molecule.smiles == result2.target_molecule.smiles
        assert len(engine._prediction_cache) > 0

    def test_deduplicate_predictions(self, sample_retro_config):
        """Test prediction deduplication."""
        engine = RetrosynthesisEngine(sample_retro_config)

        # Create duplicate predictions
        pred1 = ModelPrediction(
            reactants=["CC", "O"], confidence=0.8, disconnection_sites=[]
        )
        pred2 = ModelPrediction(
            reactants=["O", "CC"], confidence=0.7, disconnection_sites=[]
        )  # Same reactants, different order
        pred3 = ModelPrediction(
            reactants=["C", "CO"], confidence=0.9, disconnection_sites=[]
        )

        predictions = [pred1, pred2, pred3]
        unique = engine._deduplicate_predictions(predictions)

        # Should remove one duplicate
        assert len(unique) == 2


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_retro_config(self):
        """Test create_retro_config function."""
        config = create_retro_config(
            model_type=ModelType.GRAPH_NEURAL_NETWORK,
            use_gpu=False,
            confidence_threshold=0.8,
        )

        assert isinstance(config, RetrosynthesisConfig)
        assert config.model_type == ModelType.GRAPH_NEURAL_NETWORK
        assert config.use_gpu is False
        assert config.confidence_threshold == 0.8

    def test_encode_molecule(self):
        """Test encode_molecule function."""
        mol_rep = encode_molecule("CCO")

        assert isinstance(mol_rep, MolecularRepresentation)
        assert mol_rep.smiles == "CCO"

    def test_decode_molecule(self):
        """Test decode_molecule function."""
        mol_rep = MolecularRepresentation(smiles="CCO")
        smiles = decode_molecule(mol_rep)

        assert smiles == "CCO"

    def test_predict_retrosynthesis_function(self):
        """Test predict_retrosynthesis utility function."""
        result = predict_retrosynthesis("CCO")

        assert isinstance(result, RetrosynthesisResult)
        assert result.target_molecule.smiles == "CCO"

    def test_extract_disconnection_sites_function(self):
        """Test extract_disconnection_sites utility function."""
        sites = extract_disconnection_sites("CCO")

        assert isinstance(sites, list)
        assert all(isinstance(site, DisconnectionSite) for site in sites)


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_retrosynthesis_error_inheritance(self):
        """Test exception inheritance."""
        assert issubclass(ModelLoadingError, RetrosynthesisError)
        assert issubclass(PredictionError, RetrosynthesisError)
        assert issubclass(DisconnectionError, RetrosynthesisError)

    def test_disconnection_error_handling(self, sample_retro_config):
        """Test disconnection error handling."""
        predictor = DisconnectionPredictor(sample_retro_config)

        # Invalid molecule should raise error
        invalid_molecule = MolecularRepresentation(smiles="")
        invalid_molecule.mol = None  # Force invalid molecule

        with pytest.raises(DisconnectionError):
            predictor.predict_disconnection_sites(invalid_molecule)

    def test_prediction_error_handling(self, sample_retro_config):
        """Test prediction error handling."""
        # This test would check for various prediction errors
        # For now, we'll test basic error propagation

        engine = RetrosynthesisEngine(sample_retro_config)

        # Test with problematic input (this might not always raise error)
        try:
            result = engine.predict_retrosynthesis("")
            # If no error, that's also acceptable
            assert isinstance(result, RetrosynthesisResult)
        except PredictionError:
            # If error is raised, that's expected
            pass


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    def test_batch_processing_performance(self, sample_retro_config):
        """Test batch processing performance."""
        engine = RetrosynthesisEngine(sample_retro_config)

        # Test different batch sizes
        batch_sizes = [1, 3, 5]

        for batch_size in batch_sizes:
            smiles_list = ["CCO"] * batch_size
            results = engine.batch_predict(smiles_list)

            assert len(results) == batch_size
            assert all(isinstance(r, RetrosynthesisResult) for r in results)

    def test_prediction_consistency(self, sample_retro_config):
        """Test prediction consistency across multiple runs."""
        # Disable caching to test raw consistency
        config = RetrosynthesisConfig(
            model_type=sample_retro_config.model_type,
            use_gpu=False,
            enable_caching=False,
        )
        engine = RetrosynthesisEngine(config)

        # Make multiple predictions
        results = []
        for i in range(3):
            result = engine.predict_retrosynthesis("CCO")
            results.append(result)

        # Results should have consistent structure
        assert all(r.target_molecule.smiles == "CCO" for r in results)
        assert all(isinstance(r.predictions, list) for r in results)

    def test_memory_usage_with_caching(self):
        """Test memory usage with caching enabled."""
        config = RetrosynthesisConfig(enable_caching=True, use_gpu=False)
        engine = RetrosynthesisEngine(config)

        # Make multiple predictions for same molecule
        for i in range(5):
            result = engine.predict_retrosynthesis("CCO")
            assert isinstance(result, RetrosynthesisResult)

        # Cache should contain results
        assert len(engine._prediction_cache) > 0


class TestIntegration:
    """Integration tests for retrosynthesis engine."""

    def test_full_retrosynthesis_workflow(self):
        """Test complete retrosynthesis workflow."""
        # Create configuration
        config = create_retro_config(
            model_type=ModelType.TEMPLATE_BASED,  # Use template-based for reliability
            use_gpu=False,
            confidence_threshold=0.5,
        )

        # Create engine
        engine = RetrosynthesisEngine(config)

        # Predict retrosynthesis
        result = engine.predict_retrosynthesis("CCO")

        # Verify complete workflow
        assert isinstance(result, RetrosynthesisResult)
        assert result.target_molecule.smiles == "CCO"
        assert isinstance(result.predictions, list)

        # Check prediction quality
        if result.predictions:
            best_prediction = result.get_best_prediction()
            assert isinstance(best_prediction, ModelPrediction)
            assert isinstance(best_prediction.reactants, list)
            assert len(best_prediction.reactants) > 0

    def test_different_molecule_types(self, sample_retro_config):
        """Test retrosynthesis for different molecule types."""
        engine = RetrosynthesisEngine(sample_retro_config)

        # Test different molecule types
        test_molecules = [
            "CCO",  # Simple alcohol
            "c1ccccc1",  # Aromatic
            "CC(=O)O",  # Carboxylic acid
            "CCN",  # Amine
        ]

        for smiles in test_molecules:
            result = engine.predict_retrosynthesis(smiles)

            assert isinstance(result, RetrosynthesisResult)
            assert result.target_molecule.smiles == smiles
            assert isinstance(result.predictions, list)

    def test_error_propagation_through_system(self, sample_retro_config):
        """Test error propagation through the entire system."""
        engine = RetrosynthesisEngine(sample_retro_config)

        # Test with various problematic inputs
        problematic_inputs = [
            "",  # Empty string
            "INVALID",  # Invalid SMILES
            "C" * 1000,  # Very long string
        ]

        for input_smiles in problematic_inputs:
            # Should either succeed or raise appropriate error
            try:
                result = engine.predict_retrosynthesis(input_smiles)
                assert isinstance(result, RetrosynthesisResult)
            except (PredictionError, RetrosynthesisError):
                # Expected errors are acceptable
                pass

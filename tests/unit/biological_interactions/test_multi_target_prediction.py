"""
Unit tests for multi-target interaction prediction module.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from chemi_accel.biological_interactions.multi_target_prediction import (
    BiologicalTarget,
    BiologicalTargetEncoder,
    CompoundTargetInteraction,
    CompoundTargetTransformer,
    ConfidenceLevel,
    GPUBioInteractionPredictor,
    InteractionPrediction,
    InteractionScorer,
    InteractionType,
    ModelArchitecture,
    ModelLoadingError,
    MultiTargetInteractionPredictor,
    MultiTargetPredictionError,
    MultiTaskGNN,
    PredictionResult,
    TargetEncodingError,
    TargetPanel,
    TargetType,
    evaluate_prediction_accuracy,
    load_target_panel,
    predict_interactions_batch,
)


@pytest.fixture
def sample_target():
    """Create a sample biological target."""
    return BiologicalTarget(
        target_id="TEST_001",
        name="Test Kinase",
        target_type=TargetType.KINASE,
        sequence="MKTAYEKQAHSKQGQALKEFLQLDQKVLVNAQQQASDDKDGMKQKYLSLLSRKLNREEIDIRKTLKQMGMMEAQRAALFKIVNCL",
        uniprot_id="P12345",
        organism="Homo sapiens",
        gene_name="TEST1",
        description="Test kinase for unit testing",
    )


@pytest.fixture
def sample_target_panel():
    """Create a sample target panel."""
    targets = []
    for i in range(10):
        target = BiologicalTarget(
            target_id=f"TARGET_{i:03d}",
            name=f"Target {i}",
            target_type=list(TargetType)[i % len(TargetType)],
            uniprot_id=f"P{i:05d}",
        )
        targets.append(target)

    return TargetPanel(
        panel_id="TEST_PANEL",
        name="Test Panel",
        targets=targets,
        description="Test panel for unit testing",
    )


@pytest.fixture
def sample_compound_smiles():
    """Sample compound SMILES string."""
    return "CCO"


@pytest.fixture
def sample_interaction():
    """Create a sample compound-target interaction."""
    return CompoundTargetInteraction(
        compound_smiles="CCO",
        target_id="TEST_001",
        interaction_type=InteractionType.INHIBITION,
        affinity_value=100.0,
        affinity_unit="nM",
        confidence_score=0.85,
        experimental_evidence=True,
    )


class TestBiologicalTarget:
    """Test BiologicalTarget class."""

    def test_target_creation(self, sample_target):
        """Test target creation and basic properties."""
        assert sample_target.target_id == "TEST_001"
        assert sample_target.name == "Test Kinase"
        assert sample_target.target_type == TargetType.KINASE
        assert sample_target.uniprot_id == "P12345"
        assert sample_target.organism == "Homo sapiens"

    def test_target_post_init(self):
        """Test target __post_init__ method."""
        target = BiologicalTarget(
            target_id="TEST_002", name="Test Target", target_type=TargetType.ENZYME
        )

        assert isinstance(target.structural_features, dict)
        assert isinstance(target.binding_site_info, dict)
        assert isinstance(target.known_ligands, list)


class TestCompoundTargetInteraction:
    """Test CompoundTargetInteraction class."""

    def test_interaction_creation(self, sample_interaction):
        """Test interaction creation."""
        assert sample_interaction.compound_smiles == "CCO"
        assert sample_interaction.target_id == "TEST_001"
        assert sample_interaction.interaction_type == InteractionType.INHIBITION
        assert sample_interaction.affinity_value == 100.0
        assert sample_interaction.experimental_evidence is True

    def test_interaction_post_init(self):
        """Test interaction __post_init__ method."""
        interaction = CompoundTargetInteraction(
            compound_smiles="CCO",
            target_id="TEST_001",
            interaction_type=InteractionType.BINDING,
        )

        assert isinstance(interaction.publication_info, dict)


class TestInteractionPrediction:
    """Test InteractionPrediction class."""

    def test_prediction_creation(self):
        """Test prediction creation."""
        prediction = InteractionPrediction(
            compound_smiles="CCO",
            target_id="TEST_001",
            predicted_interaction_type=InteractionType.INHIBITION,
            probability_score=0.85,
        )

        assert prediction.compound_smiles == "CCO"
        assert prediction.target_id == "TEST_001"
        assert prediction.probability_score == 0.85
        assert prediction.confidence_level == ConfidenceLevel.HIGH

    def test_confidence_level_assignment(self):
        """Test confidence level assignment based on probability."""
        # Very high confidence
        pred1 = InteractionPrediction(
            compound_smiles="CCO",
            target_id="T1",
            predicted_interaction_type=InteractionType.BINDING,
            probability_score=0.95,
        )
        assert pred1.confidence_level == ConfidenceLevel.VERY_HIGH

        # High confidence
        pred2 = InteractionPrediction(
            compound_smiles="CCO",
            target_id="T1",
            predicted_interaction_type=InteractionType.BINDING,
            probability_score=0.75,
        )
        assert pred2.confidence_level == ConfidenceLevel.HIGH

        # Medium confidence
        pred3 = InteractionPrediction(
            compound_smiles="CCO",
            target_id="T1",
            predicted_interaction_type=InteractionType.BINDING,
            probability_score=0.55,
        )
        assert pred3.confidence_level == ConfidenceLevel.MEDIUM

        # Low confidence
        pred4 = InteractionPrediction(
            compound_smiles="CCO",
            target_id="T1",
            predicted_interaction_type=InteractionType.BINDING,
            probability_score=0.35,
        )
        assert pred4.confidence_level == ConfidenceLevel.LOW

        # Very low confidence
        pred5 = InteractionPrediction(
            compound_smiles="CCO",
            target_id="T1",
            predicted_interaction_type=InteractionType.BINDING,
            probability_score=0.15,
        )
        assert pred5.confidence_level == ConfidenceLevel.VERY_LOW


class TestPredictionResult:
    """Test PredictionResult class."""

    def test_result_creation(self, sample_compound_smiles):
        """Test result creation."""
        predictions = [
            InteractionPrediction(
                compound_smiles=sample_compound_smiles,
                target_id="T1",
                predicted_interaction_type=InteractionType.BINDING,
                probability_score=0.9,
            ),
            InteractionPrediction(
                compound_smiles=sample_compound_smiles,
                target_id="T2",
                predicted_interaction_type=InteractionType.INHIBITION,
                probability_score=0.7,
            ),
        ]

        result = PredictionResult(
            compound_smiles=sample_compound_smiles,
            predictions=predictions,
            total_targets_screened=10,
            active_predictions=2,
            processing_time=1.5,
            model_version="1.0",
        )

        assert result.compound_smiles == sample_compound_smiles
        assert len(result.predictions) == 2
        assert result.total_targets_screened == 10
        assert result.active_predictions == 2

    def test_get_top_predictions(self, sample_compound_smiles):
        """Test getting top predictions."""
        predictions = [
            InteractionPrediction(
                compound_smiles=sample_compound_smiles,
                target_id="T1",
                predicted_interaction_type=InteractionType.BINDING,
                probability_score=0.9,
            ),
            InteractionPrediction(
                compound_smiles=sample_compound_smiles,
                target_id="T2",
                predicted_interaction_type=InteractionType.INHIBITION,
                probability_score=0.7,
            ),
            InteractionPrediction(
                compound_smiles=sample_compound_smiles,
                target_id="T3",
                predicted_interaction_type=InteractionType.ACTIVATION,
                probability_score=0.8,
            ),
        ]

        result = PredictionResult(
            compound_smiles=sample_compound_smiles,
            predictions=predictions,
            total_targets_screened=10,
            active_predictions=3,
            processing_time=1.5,
            model_version="1.0",
        )

        top_2 = result.get_top_predictions(2)
        assert len(top_2) == 2
        assert top_2[0].probability_score == 0.9
        assert top_2[1].probability_score == 0.8

    def test_filter_by_confidence(self, sample_compound_smiles):
        """Test filtering predictions by confidence level."""
        predictions = [
            InteractionPrediction(
                compound_smiles=sample_compound_smiles,
                target_id="T1",
                predicted_interaction_type=InteractionType.BINDING,
                probability_score=0.95,  # Very high
            ),
            InteractionPrediction(
                compound_smiles=sample_compound_smiles,
                target_id="T2",
                predicted_interaction_type=InteractionType.INHIBITION,
                probability_score=0.75,  # High
            ),
            InteractionPrediction(
                compound_smiles=sample_compound_smiles,
                target_id="T3",
                predicted_interaction_type=InteractionType.ACTIVATION,
                probability_score=0.35,  # Low
            ),
        ]

        result = PredictionResult(
            compound_smiles=sample_compound_smiles,
            predictions=predictions,
            total_targets_screened=10,
            active_predictions=3,
            processing_time=1.5,
            model_version="1.0",
        )

        high_confidence = result.filter_by_confidence(ConfidenceLevel.HIGH)
        assert len(high_confidence) == 2  # Very high and high confidence predictions


class TestTargetPanel:
    """Test TargetPanel class."""

    def test_panel_creation(self, sample_target_panel):
        """Test panel creation."""
        assert sample_target_panel.panel_id == "TEST_PANEL"
        assert sample_target_panel.name == "Test Panel"
        assert len(sample_target_panel.targets) == 10

    def test_get_targets_by_type(self, sample_target_panel):
        """Test getting targets by type."""
        kinase_targets = sample_target_panel.get_targets_by_type(TargetType.KINASE)
        assert len(kinase_targets) >= 0

        for target in kinase_targets:
            assert target.target_type == TargetType.KINASE

    def test_get_target_by_id(self, sample_target_panel):
        """Test getting target by ID."""
        target = sample_target_panel.get_target_by_id("TARGET_001")
        assert target is not None
        assert target.target_id == "TARGET_001"

        # Test non-existent target
        non_existent = sample_target_panel.get_target_by_id("NON_EXISTENT")
        assert non_existent is None


class TestMultiTaskGNN:
    """Test MultiTaskGNN model."""

    def test_model_creation(self):
        """Test model creation."""
        model = MultiTaskGNN(
            input_dim=128, hidden_dim=256, output_dim=128, num_layers=3, num_targets=10
        )

        assert model.num_targets == 10
        assert len(model.conv_layers) == 3
        assert len(model.target_heads) == 10

    @patch("torch.cuda.is_available", return_value=False)
    def test_model_forward(self, mock_cuda):
        """Test model forward pass."""
        model = MultiTaskGNN(input_dim=64, hidden_dim=128, num_targets=5)

        # Create mock graph data
        from torch_geometric.data import Data

        x = torch.randn(10, 64)  # 10 nodes, 64 features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        batch = torch.zeros(10, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, batch=batch)

        # Forward pass
        results = model(data, target_ids=[0, 1, 2])

        assert "target_predictions" in results
        assert "interaction_types" in results
        assert "binding_affinities" in results

        assert results["target_predictions"].shape[1] == 3  # 3 targets
        assert results["interaction_types"].shape[1] == len(InteractionType)


class TestCompoundTargetTransformer:
    """Test CompoundTargetTransformer model."""

    def test_model_creation(self):
        """Test model creation."""
        model = CompoundTargetTransformer(
            compound_dim=2048, target_dim=1280, hidden_dim=512
        )

        assert model.compound_projection.in_features == 2048
        assert model.target_projection.in_features == 1280

    @patch("torch.cuda.is_available", return_value=False)
    def test_model_forward(self, mock_cuda):
        """Test model forward pass."""
        model = CompoundTargetTransformer(
            compound_dim=100, target_dim=80, hidden_dim=64
        )

        # Create mock input tensors
        compound_features = torch.randn(2, 100)  # Batch size 2
        target_features = torch.randn(2, 80)

        # Forward pass
        results = model(compound_features, target_features)

        assert "interaction_probability" in results
        assert "binding_affinity" in results

        assert results["interaction_probability"].shape == (2, 1)
        assert results["binding_affinity"].shape == (2, 1)


class TestBiologicalTargetEncoder:
    """Test BiologicalTargetEncoder."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_encoder_creation(self, mock_cuda):
        """Test encoder creation."""
        encoder = BiologicalTargetEncoder(use_gpu=False)
        assert encoder.device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=False)
    def test_encode_target(self, mock_cuda, sample_target):
        """Test target encoding."""
        encoder = BiologicalTargetEncoder(use_gpu=False)
        target_repr = encoder.encode_target(sample_target)

        assert isinstance(target_repr, torch.Tensor)
        assert target_repr.device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=False)
    def test_encode_target_type(self, mock_cuda):
        """Test target type encoding."""
        encoder = BiologicalTargetEncoder(use_gpu=False)
        type_features = encoder._encode_target_type(TargetType.KINASE)

        assert isinstance(type_features, torch.Tensor)
        assert type_features.shape[0] == len(TargetType)
        assert type_features.sum().item() == 1.0  # One-hot encoding


class TestInteractionScorer:
    """Test InteractionScorer."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_scorer_creation(self, mock_cuda):
        """Test scorer creation."""
        scorer = InteractionScorer(use_gpu=False)
        assert scorer.device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=False)
    def test_score_interaction(self, mock_cuda, sample_target):
        """Test interaction scoring."""
        scorer = InteractionScorer(use_gpu=False)
        scores = scorer.score_interaction(
            compound_smiles="CCO",
            target=sample_target,
            interaction_prob=0.8,
            binding_affinity=100.0,
        )

        assert isinstance(scores, dict)
        assert "interaction_probability" in scores
        assert "affinity_score" in scores
        assert "molecular_properties" in scores
        assert "target_druggability" in scores
        assert "selectivity_score" in scores
        assert "combined_score" in scores

        assert 0 <= scores["combined_score"] <= 1

    @patch("torch.cuda.is_available", return_value=False)
    def test_normalize_affinity(self, mock_cuda):
        """Test affinity normalization."""
        scorer = InteractionScorer(use_gpu=False)

        # Test various affinity values
        normalized_high = scorer._normalize_affinity(1.0)  # 1 nM (high affinity)
        normalized_low = scorer._normalize_affinity(10000.0)  # 10 μM (low affinity)

        assert 0 <= normalized_high <= 1
        assert 0 <= normalized_low <= 1
        assert normalized_high > normalized_low


class TestMultiTargetInteractionPredictor:
    """Test MultiTargetInteractionPredictor."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_predictor_creation(self, mock_cuda):
        """Test predictor creation."""
        predictor = MultiTargetInteractionPredictor(use_gpu=False)
        assert predictor.device.type == "cpu"
        assert predictor.model_architecture == ModelArchitecture.MULTI_TASK_GNN

    @patch("torch.cuda.is_available", return_value=False)
    def test_predict_interactions(
        self, mock_cuda, sample_target_panel, sample_compound_smiles
    ):
        """Test interaction prediction."""
        predictor = MultiTargetInteractionPredictor(use_gpu=False, batch_size=5)

        result = predictor.predict_interactions(
            compound_smiles=sample_compound_smiles,
            target_panel=sample_target_panel,
            confidence_threshold=0.1,  # Low threshold to get some predictions
        )

        assert isinstance(result, PredictionResult)
        assert result.compound_smiles == sample_compound_smiles
        assert result.total_targets_screened == len(sample_target_panel.targets)
        assert isinstance(result.predictions, list)
        assert result.processing_time > 0

    @patch("torch.cuda.is_available", return_value=False)
    def test_encode_compound(self, mock_cuda, sample_compound_smiles):
        """Test compound encoding."""
        predictor = MultiTargetInteractionPredictor(use_gpu=False)
        compound_features = predictor._encode_compound(sample_compound_smiles)

        assert isinstance(compound_features, torch.Tensor)
        assert compound_features.device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=False)
    def test_predict_single_interaction(self, mock_cuda):
        """Test single interaction prediction."""
        predictor = MultiTargetInteractionPredictor(use_gpu=False)

        compound_features = torch.randn(100)
        target_features = torch.randn(100)

        probability = predictor._predict_single_interaction(
            compound_features, target_features
        )

        assert isinstance(probability, float)
        assert 0 <= probability <= 1

    @patch("torch.cuda.is_available", return_value=False)
    def test_model_save_load(self, mock_cuda, tmp_path):
        """Test model saving and loading."""
        predictor = MultiTargetInteractionPredictor(use_gpu=False)
        model_path = tmp_path / "test_model.pt"

        # Save model
        predictor.save_model(str(model_path))
        assert model_path.exists()

        # Load model
        new_predictor = MultiTargetInteractionPredictor(use_gpu=False)
        new_predictor.load_model(str(model_path))


class TestGPUBioInteractionPredictor:
    """Test GPUBioInteractionPredictor."""

    @patch("torch.cuda.is_available", return_value=False)
    @patch(
        "chemi_accel.biological_interactions.multi_target_prediction.CUPY_AVAILABLE",
        False,
    )
    def test_gpu_predictor_creation(self, mock_cuda):
        """Test GPU predictor creation without CUDA."""
        predictor = GPUBioInteractionPredictor()
        # Should fall back to CPU
        assert predictor.device.type == "cpu"


class TestUtilityFunctions:
    """Test utility functions."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_predict_interactions_batch(self, mock_cuda, sample_target_panel):
        """Test batch interaction prediction."""
        compound_smiles_list = ["CCO", "CCC", "CCCO"]

        results = predict_interactions_batch(
            compound_smiles_list,
            sample_target_panel,
            use_gpu=False,
            confidence_threshold=0.1,
        )

        assert len(results) == 3
        for result in results:
            assert isinstance(result, PredictionResult)

    def test_load_target_panel(self):
        """Test loading target panel."""
        panel = load_target_panel("dummy_file.json")

        assert isinstance(panel, TargetPanel)
        assert panel.panel_id == "SAMPLE_PANEL"
        assert len(panel.targets) == 100

    def test_evaluate_prediction_accuracy(self):
        """Test prediction accuracy evaluation."""
        # Create mock data
        predictions = [
            PredictionResult(
                compound_smiles="CCO",
                predictions=[],
                total_targets_screened=10,
                active_predictions=0,
                processing_time=1.0,
                model_version="1.0",
            )
        ]

        ground_truth = [
            CompoundTargetInteraction(
                compound_smiles="CCO",
                target_id="T1",
                interaction_type=InteractionType.BINDING,
            )
        ]

        metrics = evaluate_prediction_accuracy(predictions, ground_truth)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics


class TestExceptions:
    """Test exception classes."""

    def test_multi_target_prediction_error(self):
        """Test MultiTargetPredictionError."""
        with pytest.raises(MultiTargetPredictionError):
            raise MultiTargetPredictionError("Test error")

    def test_model_loading_error(self):
        """Test ModelLoadingError."""
        with pytest.raises(ModelLoadingError):
            raise ModelLoadingError("Model loading failed")

    def test_target_encoding_error(self):
        """Test TargetEncodingError."""
        with pytest.raises(TargetEncodingError):
            raise TargetEncodingError("Target encoding failed")

    @patch("torch.cuda.is_available", return_value=False)
    def test_model_loading_error_in_predictor(self, mock_cuda):
        """Test model loading error in predictor."""
        predictor = MultiTargetInteractionPredictor(use_gpu=False)

        with pytest.raises(ModelLoadingError):
            predictor.load_model("non_existent_model.pt")


class TestIntegration:
    """Integration tests."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_end_to_end_prediction(
        self, mock_cuda, sample_target_panel, sample_compound_smiles
    ):
        """Test end-to-end prediction workflow."""
        # Create predictor
        predictor = MultiTargetInteractionPredictor(
            model_architecture=ModelArchitecture.MULTI_TASK_GNN,
            use_gpu=False,
            batch_size=5,
        )

        # Make predictions
        result = predictor.predict_interactions(
            compound_smiles=sample_compound_smiles,
            target_panel=sample_target_panel,
            confidence_threshold=0.1,
        )

        # Verify results
        assert isinstance(result, PredictionResult)
        assert result.compound_smiles == sample_compound_smiles
        assert result.total_targets_screened == len(sample_target_panel.targets)

        # Test filtering and top predictions
        if result.predictions:
            top_preds = result.get_top_predictions(3)
            assert len(top_preds) <= 3

            high_conf_preds = result.filter_by_confidence(ConfidenceLevel.MEDIUM)
            assert isinstance(high_conf_preds, list)

    @patch("torch.cuda.is_available", return_value=False)
    def test_different_model_architectures(self, mock_cuda, sample_target_panel):
        """Test different model architectures."""
        architectures = [
            ModelArchitecture.MULTI_TASK_GNN,
            ModelArchitecture.COMPOUND_TARGET_TRANSFORMER,
        ]

        for arch in architectures:
            predictor = MultiTargetInteractionPredictor(
                model_architecture=arch, use_gpu=False
            )

            # Verify model was created
            assert predictor.model is not None
            assert predictor.model_architecture == arch

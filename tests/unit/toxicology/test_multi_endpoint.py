"""
Test cases for multi-endpoint toxicology prediction.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from chemi_accel.core.backend import get_backend
from chemi_accel.toxicology.core import (
    ConfidenceLevel,
    SpeciesType,
    ToxicityEndpointType,
    ToxicityPrediction,
)
from chemi_accel.toxicology.multi_endpoint import (
    EndpointResult,
    FeatureType,
    GraphNeuralNetworkModel,
    ModelArchitecture,
    MultiEndpointToxicologyPredictor,
    MultiTaskDNNModel,
)


class TestMultiTaskDNNModel:
    """Test MultiTaskDNNModel class."""

    def test_model_initialization(self):
        """Test model initialization."""
        backend = get_backend("cpu")
        model = MultiTaskDNNModel(backend)

        assert model.backend == backend
        assert model.model is None
        assert model.is_trained is False
        assert isinstance(model.feature_scalers, dict)
        assert isinstance(model.endpoint_encoders, dict)

    @patch("chemi_accel.toxicology.multi_endpoint.torch")
    @patch("chemi_accel.toxicology.multi_endpoint.StandardScaler")
    def test_model_training_with_pytorch(self, mock_scaler, mock_torch):
        """Test model training with PyTorch available."""
        # Mock PyTorch components
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        mock_torch.FloatTensor.return_value = MagicMock()
        mock_torch.BoolTensor.return_value = MagicMock()
        mock_torch.utils.data.random_split.return_value = (MagicMock(), MagicMock())

        # Mock DataLoader
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(
            return_value=iter([(MagicMock(), MagicMock(), MagicMock())])  # batch data
        )
        mock_dataloader.__len__ = MagicMock(return_value=1)

        with patch(
            "chemi_accel.toxicology.multi_endpoint.DataLoader",
            return_value=mock_dataloader,
        ):
            backend = get_backend("cpu")
            model = MultiTaskDNNModel(backend)

            # Create mock training data
            training_data = pd.DataFrame(
                {
                    "acute_oral_rat": [100, 200, 300],
                    "carcinogenicity_rat": [0.1, 0.8, 0.3],
                }
            )
            molecular_features = np.random.rand(3, 10)
            endpoints = [
                ToxicityEndpointType.ACUTE_ORAL,
                ToxicityEndpointType.CARCINOGENICITY,
            ]
            species_list = [SpeciesType.RAT]

            # Should not raise exception
            try:
                model.train(
                    training_data, molecular_features, endpoints, species_list, epochs=1
                )
            except ImportError:
                pytest.skip("PyTorch not available")

    def test_model_training_without_pytorch(self):
        """Test model training without PyTorch."""
        backend = get_backend("cpu")
        model = MultiTaskDNNModel(backend)

        training_data = pd.DataFrame({"test": [1, 2, 3]})
        molecular_features = np.random.rand(3, 10)
        endpoints = [ToxicityEndpointType.ACUTE_ORAL]
        species_list = [SpeciesType.RAT]

        # Should raise ChemiAccelError due to missing PyTorch
        with patch(
            "chemi_accel.toxicology.multi_endpoint.torch", side_effect=ImportError
        ):
            from chemi_accel.core.exceptions import ChemiAccelError

            with pytest.raises(ChemiAccelError, match="PyTorch required"):
                model.train(training_data, molecular_features, endpoints, species_list)

    def test_prediction_without_training(self):
        """Test prediction on untrained model."""
        backend = get_backend("cpu")
        model = MultiTaskDNNModel(backend)

        features = np.random.rand(10)
        pred_value, uncertainty = model.predict(
            features, ToxicityEndpointType.ACUTE_ORAL, SpeciesType.RAT
        )

        # Should return dummy values
        assert isinstance(pred_value, float)
        assert isinstance(uncertainty, float)
        assert 0 <= pred_value <= 100
        assert 0 <= uncertainty <= 1

    def test_feature_importance_without_training(self):
        """Test feature importance on untrained model."""
        backend = get_backend("cpu")
        model = MultiTaskDNNModel(backend)

        features = np.random.rand(10)
        importance = model.get_feature_importance(
            features, ToxicityEndpointType.ACUTE_ORAL
        )

        assert isinstance(importance, dict)
        assert len(importance) <= 10  # Should return up to 10 features
        assert all(isinstance(v, float) for v in importance.values())


class TestGraphNeuralNetworkModel:
    """Test GraphNeuralNetworkModel class."""

    def test_model_initialization(self):
        """Test GNN model initialization."""
        backend = get_backend("cpu")
        model = GraphNeuralNetworkModel(backend)

        assert model.backend == backend
        assert model.model is None
        assert model.is_trained is False

    def test_prediction_fallback(self):
        """Test GNN prediction fallback."""
        backend = get_backend("cpu")
        model = GraphNeuralNetworkModel(backend)

        features = np.random.rand(10)
        pred_value, uncertainty = model.predict(
            features, ToxicityEndpointType.ACUTE_ORAL, SpeciesType.RAT
        )

        # Should return random values for demo
        assert isinstance(pred_value, float)
        assert isinstance(uncertainty, float)

    def test_feature_importance_fallback(self):
        """Test GNN feature importance fallback."""
        backend = get_backend("cpu")
        model = GraphNeuralNetworkModel(backend)

        features = np.random.rand(10)
        importance = model.get_feature_importance(
            features, ToxicityEndpointType.ACUTE_ORAL
        )

        assert isinstance(importance, dict)
        assert len(importance) == 10  # Should return 10 atom features


class TestMultiEndpointToxicologyPredictor:
    """Test MultiEndpointToxicologyPredictor class."""

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        backend = get_backend("cpu")
        predictor = MultiEndpointToxicologyPredictor(backend)

        assert predictor.backend == backend
        assert "multi_task_dnn" in predictor.models
        assert "gnn" in predictor.models
        assert predictor.default_model == "multi_task_dnn"
        assert isinstance(predictor.feature_extractors, dict)

    @patch("chemi_accel.toxicology.multi_endpoint.Chem")
    @patch("chemi_accel.toxicology.multi_endpoint.Descriptors")
    @patch("chemi_accel.toxicology.multi_endpoint.rdMolDescriptors")
    def test_feature_extraction_with_rdkit(self, mock_rdmol, mock_desc, mock_chem):
        """Test feature extraction with RDKit."""
        # Mock RDKit components
        mock_mol = MagicMock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_desc.MolWt.return_value = 46.07
        mock_desc.MolLogP.return_value = -0.77
        mock_desc.TPSA.return_value = 20.23
        mock_desc.NumHDonors.return_value = 1
        mock_desc.NumHAcceptors.return_value = 1
        mock_desc.NumRotatableBonds.return_value = 0
        mock_desc.NumAromaticRings.return_value = 0
        mock_desc.FractionCsp3.return_value = 1.0
        mock_desc.BalabanJ.return_value = 0.0
        mock_desc.BertzCT.return_value = 0.0

        # Mock fingerprint
        mock_fp = MagicMock()
        mock_fp.__iter__ = MagicMock(return_value=iter([0] * 128))
        mock_rdmol.GetMorganFingerprintAsBitVect.return_value = mock_fp

        # Mock atom counts
        mock_atom = MagicMock()
        mock_atom.GetIsAromatic.return_value = False
        mock_atom.GetHybridization.return_value = MagicMock()
        mock_atom.GetAtomicNum.side_effect = [6, 6, 8]  # C, C, O
        mock_mol.GetAtoms.return_value = [mock_atom, mock_atom, mock_atom]
        mock_mol.GetRings.return_value = []

        backend = get_backend("cpu")
        predictor = MultiEndpointToxicologyPredictor(backend)

        features = predictor._extract_features(["CCO"])

        assert features.shape == (1, 200)  # 1 compound, 200 features
        assert features[0, 0] == 46.07  # Molecular weight
        assert features[0, 1] == -0.77  # LogP

    def test_feature_extraction_fallback(self):
        """Test feature extraction fallback without RDKit."""
        backend = get_backend("cpu")
        predictor = MultiEndpointToxicologyPredictor(backend)

        # Mock ImportError for RDKit
        with patch(
            "chemi_accel.toxicology.multi_endpoint.Chem", side_effect=ImportError
        ):
            features = predictor._extract_features(["CCO", "CCC"])

            assert features.shape == (2, 200)  # 2 compounds, 200 random features
            assert np.all((features >= 0) & (features <= 1))  # Random values 0-1

    def test_predict_endpoints_single_smiles(self):
        """Test endpoint prediction for single SMILES."""
        backend = get_backend("cpu")
        predictor = MultiEndpointToxicologyPredictor(backend)

        # Mock model prediction
        with patch.object(
            predictor.models["multi_task_dnn"], "predict", return_value=(50.0, 0.2)
        ):
            with patch.object(
                predictor.models["multi_task_dnn"],
                "get_feature_importance",
                return_value={"feature_0": 0.5, "feature_1": 0.3},
            ):
                predictions = predictor.predict_endpoints(
                    "CCO",
                    [ToxicityEndpointType.ACUTE_ORAL],
                    [SpeciesType.RAT],
                    include_uncertainty=True,
                )

        assert len(predictions) == 1
        assert isinstance(predictions[0], ToxicityPrediction)
        assert predictions[0].predicted_value == 50.0
        assert predictions[0].uncertainty == 0.2
        assert predictions[0].model_type == "multi_task_dnn"

    def test_predict_endpoints_multiple_smiles(self):
        """Test endpoint prediction for multiple SMILES."""
        backend = get_backend("cpu")
        predictor = MultiEndpointToxicologyPredictor(backend)

        # Mock model prediction
        with patch.object(
            predictor.models["multi_task_dnn"], "predict", return_value=(50.0, 0.2)
        ):
            with patch.object(
                predictor.models["multi_task_dnn"],
                "get_feature_importance",
                return_value={"feature_0": 0.5},
            ):
                predictions = predictor.predict_endpoints(
                    ["CCO", "CCC"], [ToxicityEndpointType.ACUTE_ORAL], [SpeciesType.RAT]
                )

        assert len(predictions) == 2  # 2 SMILES * 1 endpoint * 1 species

    def test_predict_endpoints_multiple_endpoints_species(self):
        """Test prediction for multiple endpoints and species."""
        backend = get_backend("cpu")
        predictor = MultiEndpointToxicologyPredictor(backend)

        # Mock model prediction
        with patch.object(
            predictor.models["multi_task_dnn"], "predict", return_value=(50.0, 0.2)
        ):
            with patch.object(
                predictor.models["multi_task_dnn"],
                "get_feature_importance",
                return_value={},
            ):
                predictions = predictor.predict_endpoints(
                    "CCO",
                    [
                        ToxicityEndpointType.ACUTE_ORAL,
                        ToxicityEndpointType.CARCINOGENICITY,
                    ],
                    [SpeciesType.RAT, SpeciesType.MOUSE],
                )

        assert len(predictions) == 4  # 1 SMILES * 2 endpoints * 2 species

    def test_uncertainty_to_confidence_conversion(self):
        """Test uncertainty to confidence level conversion."""
        backend = get_backend("cpu")
        predictor = MultiEndpointToxicologyPredictor(backend)

        # Test different uncertainty levels
        assert predictor._uncertainty_to_confidence(0.05) == ConfidenceLevel.VERY_HIGH
        assert predictor._uncertainty_to_confidence(0.15) == ConfidenceLevel.HIGH
        assert predictor._uncertainty_to_confidence(0.3) == ConfidenceLevel.MODERATE
        assert predictor._uncertainty_to_confidence(0.5) == ConfidenceLevel.LOW
        assert predictor._uncertainty_to_confidence(0.8) == ConfidenceLevel.VERY_LOW

    def test_model_performance_evaluation_untrained(self):
        """Test model performance evaluation on untrained model."""
        backend = get_backend("cpu")
        predictor = MultiEndpointToxicologyPredictor(backend)

        test_data = pd.DataFrame({"test": [1, 2, 3]})
        features = np.random.rand(3, 10)
        endpoints = [ToxicityEndpointType.ACUTE_ORAL]

        performance = predictor.evaluate_model_performance(
            test_data, features, endpoints
        )

        assert "error" in performance
        assert performance["error"] == "Model not trained"

    def test_get_model_explanation(self):
        """Test model explanation generation."""
        backend = get_backend("cpu")
        predictor = MultiEndpointToxicologyPredictor(backend)

        # Mock feature importance
        with patch.object(
            predictor.models["multi_task_dnn"],
            "get_feature_importance",
            return_value={"feature_0": 0.8, "feature_1": 0.6, "feature_2": 0.4},
        ):
            explanation = predictor.get_model_explanation(
                "CCO", ToxicityEndpointType.ACUTE_ORAL
            )

        assert explanation["compound"] == "CCO"
        assert explanation["endpoint"] == "acute_oral"
        assert explanation["model"] == "multi_task_dnn"
        assert "feature_importance" in explanation
        assert "top_features" in explanation
        assert len(explanation["top_features"]) <= 10

    def test_train_models(self):
        """Test training multiple models."""
        backend = get_backend("cpu")
        predictor = MultiEndpointToxicologyPredictor(backend)

        training_data = pd.DataFrame({"acute_oral_rat": [100, 200, 300]})
        smiles_list = ["CCO", "CCC", "CCCO"]
        endpoints = [ToxicityEndpointType.ACUTE_ORAL]
        species_list = [SpeciesType.RAT]

        # Mock model training
        with patch.object(predictor.models["multi_task_dnn"], "train") as mock_train:
            predictor.train_models(
                training_data,
                smiles_list,
                endpoints,
                species_list,
                model_types=["multi_task_dnn"],
            )

            mock_train.assert_called_once()

    def test_prediction_error_handling(self):
        """Test error handling during prediction."""
        backend = get_backend("cpu")
        predictor = MultiEndpointToxicologyPredictor(backend)

        # Mock model to raise exception
        with patch.object(
            predictor.models["multi_task_dnn"],
            "predict",
            side_effect=Exception("Test error"),
        ):
            with patch.object(
                predictor.models["multi_task_dnn"],
                "get_feature_importance",
                return_value={},
            ):
                predictions = predictor.predict_endpoints(
                    "CCO", [ToxicityEndpointType.ACUTE_ORAL], [SpeciesType.RAT]
                )

        # Should handle exception gracefully and return empty list
        assert len(predictions) == 0


if __name__ == "__main__":
    pytest.main([__file__])

"""
Unit tests for predictive spectroscopy models.

This module tests the GPU-accelerated predictive models for various
spectroscopy techniques including NMR, IR, MS, and UV-Vis.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.spectroscopy.predictive_models import (
    GPUSpectroscopyPredictor,
    IRPredictor,
    ModelArchitecture,
    ModelConfiguration,
    ModelError,
    MSPredictor,
    NMRPredictor,
    PredictionError,
    PredictionResult,
    QMLSurrogateModel,
    QuantumMethod,
    SpectralPrediction,
    SpectroscopyType,
    TrainingData,
    TrainingError,
    UVVisPredictor,
    evaluate_model_accuracy,
    predict_spectrum_batch,
    train_spectral_model,
)


class TestSpectralPrediction:
    """Test SpectralPrediction data structure."""

    def test_basic_prediction(self):
        """Test basic spectral prediction creation."""
        frequencies = np.linspace(0, 4000, 100)
        intensities = np.random.rand(100)

        prediction = SpectralPrediction(
            frequencies=frequencies,
            intensities=intensities,
            spectroscopy_type=SpectroscopyType.IR,
        )

        assert len(prediction.frequencies) == 100
        assert len(prediction.intensities) == 100
        assert prediction.spectroscopy_type == SpectroscopyType.IR
        assert prediction.confidence_intervals is None
        assert prediction.computation_time == 0.0

    def test_prediction_with_confidence(self):
        """Test prediction with confidence intervals."""
        frequencies = np.linspace(0, 12, 50)
        intensities = np.random.rand(50)
        confidence = np.random.rand(50, 2)

        prediction = SpectralPrediction(
            frequencies=frequencies,
            intensities=intensities,
            spectroscopy_type=SpectroscopyType.NMR_1H,
            confidence_intervals=confidence,
        )

        assert prediction.confidence_intervals.shape == (50, 2)

    def test_prediction_validation(self):
        """Test prediction data validation."""
        frequencies = np.linspace(0, 4000, 100)
        intensities = np.random.rand(50)  # Wrong length

        with pytest.raises(
            ValueError, match="Frequencies and intensities must have same length"
        ):
            SpectralPrediction(
                frequencies=frequencies,
                intensities=intensities,
                spectroscopy_type=SpectroscopyType.IR,
            )

    def test_prediction_with_peak_assignments(self):
        """Test prediction with peak assignments."""
        frequencies = np.array([1000, 1500, 3000])
        intensities = np.array([0.5, 0.8, 1.0])
        assignments = {1000.0: "C-O stretch", 3000.0: "C-H stretch"}

        prediction = SpectralPrediction(
            frequencies=frequencies,
            intensities=intensities,
            spectroscopy_type=SpectroscopyType.IR,
            peak_assignments=assignments,
        )

        assert prediction.peak_assignments[1000.0] == "C-O stretch"
        assert prediction.peak_assignments[3000.0] == "C-H stretch"


class TestModelConfiguration:
    """Test ModelConfiguration data structure."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
        )

        assert config.architecture == ModelArchitecture.TRANSFORMER
        assert config.spectroscopy_type == SpectroscopyType.IR
        assert config.hidden_dims == [512, 256, 128]
        assert config.dropout_rate == 0.1
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
        assert config.use_gpu is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfiguration(
            architecture=ModelArchitecture.CONVOLUTIONAL,
            spectroscopy_type=SpectroscopyType.NMR_1H,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.2,
            learning_rate=1e-3,
            batch_size=64,
            frequency_range=(0.0, 12.0),
        )

        assert config.hidden_dims == [256, 128, 64]
        assert config.dropout_rate == 0.2
        assert config.learning_rate == 1e-3
        assert config.batch_size == 64
        assert config.frequency_range == (0.0, 12.0)

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="Dropout rate must be between 0 and 1"):
            ModelConfiguration(
                architecture=ModelArchitecture.TRANSFORMER,
                spectroscopy_type=SpectroscopyType.IR,
                dropout_rate=1.5,
            )

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            ModelConfiguration(
                architecture=ModelArchitecture.TRANSFORMER,
                spectroscopy_type=SpectroscopyType.IR,
                learning_rate=-0.001,
            )


class TestTrainingData:
    """Test TrainingData structure."""

    def test_basic_training_data(self):
        """Test basic training data creation."""
        features = np.random.rand(100, 256)
        spectra = np.random.rand(100, 1000)
        frequencies = np.linspace(0, 4000, 1000)

        data = TrainingData(
            molecular_features=features, spectra=spectra, frequencies=frequencies
        )

        assert data.molecular_features.shape == (100, 256)
        assert data.spectra.shape == (100, 1000)
        assert len(data.frequencies) == 1000
        assert data.molecule_ids is None
        assert data.weights is None

    def test_training_data_with_metadata(self):
        """Test training data with additional metadata."""
        features = np.random.rand(50, 128)
        spectra = np.random.rand(50, 500)
        frequencies = np.linspace(0, 4000, 500)
        molecule_ids = [f"mol_{i}" for i in range(50)]
        weights = np.random.rand(50)
        metadata = {"dataset": "test", "version": "1.0"}

        data = TrainingData(
            molecular_features=features,
            spectra=spectra,
            frequencies=frequencies,
            molecule_ids=molecule_ids,
            weights=weights,
            metadata=metadata,
        )

        assert len(data.molecule_ids) == 50
        assert len(data.weights) == 50
        assert data.metadata["dataset"] == "test"

    def test_training_data_validation(self):
        """Test training data validation."""
        features = np.random.rand(100, 256)
        spectra = np.random.rand(50, 1000)  # Wrong number of samples
        frequencies = np.linspace(0, 4000, 1000)

        with pytest.raises(
            ValueError, match="Features and spectra must have same number of samples"
        ):
            TrainingData(
                molecular_features=features, spectra=spectra, frequencies=frequencies
            )


class TestPredictionResult:
    """Test PredictionResult structure."""

    def test_basic_result(self):
        """Test basic prediction result."""
        predictions = [
            SpectralPrediction(
                frequencies=np.linspace(0, 4000, 100),
                intensities=np.random.rand(100),
                spectroscopy_type=SpectroscopyType.IR,
            )
            for _ in range(5)
        ]

        result = PredictionResult(predictions=predictions, batch_computation_time=1.5)

        assert len(result.predictions) == 5
        assert result.batch_computation_time == 1.5
        assert result.gpu_memory_used is None

    def test_average_confidence_no_intervals(self):
        """Test average confidence calculation without confidence intervals."""
        predictions = [
            SpectralPrediction(
                frequencies=np.linspace(0, 4000, 100),
                intensities=np.random.rand(100),
                spectroscopy_type=SpectroscopyType.IR,
            )
            for _ in range(3)
        ]

        result = PredictionResult(predictions=predictions, batch_computation_time=1.0)

        confidence = result.get_average_confidence()
        assert confidence == 0.0

    def test_average_confidence_with_intervals(self):
        """Test average confidence calculation with confidence intervals."""
        predictions = []
        for _ in range(3):
            frequencies = np.linspace(0, 4000, 100)
            intensities = np.random.rand(100)
            confidence = np.random.rand(100, 2)
            confidence[:, 1] += confidence[:, 0]  # Ensure upper > lower

            pred = SpectralPrediction(
                frequencies=frequencies,
                intensities=intensities,
                spectroscopy_type=SpectroscopyType.IR,
                confidence_intervals=confidence,
            )
            predictions.append(pred)

        result = PredictionResult(predictions=predictions, batch_computation_time=1.0)

        confidence = result.get_average_confidence()
        assert 0.0 <= confidence <= 1.0


@patch("chemi_accel.spectroscopy.predictive_models.TORCH_AVAILABLE", True)
class TestGPUSpectroscopyPredictor:
    """Test GPU spectroscopy predictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
        )

        with patch("torch.cuda.is_available", return_value=False):
            predictor = GPUSpectroscopyPredictor(config)
            assert predictor.config == config
            assert predictor.device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.device")
    def test_gpu_initialization(self, mock_device):
        """Test GPU initialization."""
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
            use_gpu=True,
        )

        mock_device.return_value = Mock()
        mock_device.return_value.type = "cuda"

        predictor = GPUSpectroscopyPredictor(config)
        mock_device.assert_called_with("cuda")

    def test_model_creation_transformer(self):
        """Test transformer model creation."""
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
        )

        with patch("torch.cuda.is_available", return_value=False):
            predictor = GPUSpectroscopyPredictor(config)
            model = predictor._create_model(256, 1000)

            # Check if model is created (mocked)
            assert model is not None

    @patch("torch.utils.data.DataLoader")
    @patch("torch.utils.data.TensorDataset")
    @patch("torch.optim.Adam")
    @patch("torch.nn.MSELoss")
    def test_training(self, mock_loss, mock_optimizer, mock_dataset, mock_dataloader):
        """Test model training."""
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
            num_epochs=2,
        )

        # Mock training data
        features = np.random.rand(100, 256)
        spectra = np.random.rand(100, 1000)
        frequencies = np.linspace(0, 4000, 1000)

        training_data = TrainingData(
            molecular_features=features, spectra=spectra, frequencies=frequencies
        )

        # Mock components
        mock_model = Mock()
        mock_optimizer_instance = Mock()
        mock_loss_instance = Mock()
        mock_dataloader_instance = Mock()

        mock_optimizer.return_value = mock_optimizer_instance
        mock_loss.return_value = mock_loss_instance
        mock_dataloader.return_value = mock_dataloader_instance

        # Mock training loop
        mock_dataloader_instance.__iter__.return_value = [
            (Mock(), Mock()) for _ in range(5)  # 5 batches
        ]
        mock_loss_instance.return_value.item.return_value = 0.1

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.tensor"):
                predictor = GPUSpectroscopyPredictor(config)
                predictor.model = mock_model

                metrics = predictor.train(training_data)

                assert "final_loss" in metrics
                assert "min_loss" in metrics
                assert "epochs_trained" in metrics

    def test_prediction_without_training(self):
        """Test prediction without training."""
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
        )

        with patch("torch.cuda.is_available", return_value=False):
            predictor = GPUSpectroscopyPredictor(config)
            features = np.random.rand(10, 256)

            with pytest.raises(
                RuntimeError, match="Model must be trained before prediction"
            ):
                predictor.predict(features)

    @patch("torch.no_grad")
    @patch("torch.tensor")
    def test_prediction_with_training(self, mock_tensor, mock_no_grad):
        """Test prediction with trained model."""
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
        )

        # Mock model and tensors
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tensor_instance = Mock()
        mock_tensor.return_value = mock_tensor_instance
        mock_tensor_instance.to.return_value = mock_tensor_instance

        # Mock prediction output
        pred_output = Mock()
        pred_output.cpu.return_value.numpy.return_value = np.random.rand(5, 1000)
        mock_model.return_value = pred_output

        with patch("torch.cuda.is_available", return_value=False):
            predictor = GPUSpectroscopyPredictor(config)
            predictor.model = mock_model
            predictor.is_trained = True

            features = np.random.rand(5, 256)
            predictions = predictor.predict(features)

            assert len(predictions) == 5
            assert all(isinstance(p, SpectralPrediction) for p in predictions)


class TestSpecializedPredictors:
    """Test specialized predictors for different spectroscopy types."""

    def test_nmr_predictor_1h(self):
        """Test 1H NMR predictor initialization."""
        predictor = NMRPredictor("1H")

        assert predictor.config.spectroscopy_type == SpectroscopyType.NMR_1H
        assert predictor.config.frequency_range == (0.0, 12.0)
        assert predictor.config.frequency_resolution == 0.01

    def test_nmr_predictor_13c(self):
        """Test 13C NMR predictor initialization."""
        predictor = NMRPredictor("13C")

        assert predictor.config.spectroscopy_type == SpectroscopyType.NMR_13C
        assert predictor.config.frequency_range == (0.0, 220.0)

    def test_ir_predictor(self):
        """Test IR predictor initialization."""
        predictor = IRPredictor()

        assert predictor.config.spectroscopy_type == SpectroscopyType.IR
        assert predictor.config.architecture == ModelArchitecture.CONVOLUTIONAL
        assert predictor.config.frequency_range == (400.0, 4000.0)

    def test_ms_predictor(self):
        """Test MS predictor initialization."""
        predictor = MSPredictor()

        assert predictor.config.spectroscopy_type == SpectroscopyType.MS
        assert predictor.config.architecture == ModelArchitecture.GRAPH_NEURAL
        assert predictor.config.frequency_range == (0.0, 1000.0)

    def test_uv_vis_predictor(self):
        """Test UV-Vis predictor initialization."""
        predictor = UVVisPredictor()

        assert predictor.config.spectroscopy_type == SpectroscopyType.UV_VIS
        assert predictor.config.architecture == ModelArchitecture.TRANSFORMER
        assert predictor.config.frequency_range == (200.0, 800.0)


class TestQMLSurrogateModel:
    """Test QML surrogate model."""

    def test_initialization(self):
        """Test QML surrogate initialization."""
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
        )

        model = QMLSurrogateModel(config)
        assert model.config == config
        assert model.classical_model is None

    def test_quantum_feature_extraction(self):
        """Test quantum feature extraction."""
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
        )

        model = QMLSurrogateModel(config)
        features = np.random.rand(10, 256)

        enhanced_features = model._extract_quantum_features(features)

        # Should add quantum features
        assert enhanced_features.shape[1] > features.shape[1]
        assert enhanced_features.shape[0] == features.shape[0]

    @patch("chemi_accel.spectroscopy.predictive_models.GPUSpectroscopyPredictor")
    def test_training_with_classical_fallback(self, mock_predictor_class):
        """Test training with classical model fallback."""
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
        )

        # Mock classical predictor
        mock_predictor = Mock()
        mock_predictor.train.return_value = {"final_loss": 0.1}
        mock_predictor_class.return_value = mock_predictor

        model = QMLSurrogateModel(config)

        # Training data
        features = np.random.rand(50, 128)
        spectra = np.random.rand(50, 1000)
        frequencies = np.linspace(0, 4000, 1000)

        training_data = TrainingData(
            molecular_features=features, spectra=spectra, frequencies=frequencies
        )

        metrics = model.train(training_data)

        assert "final_loss" in metrics
        assert model.classical_model is not None
        mock_predictor.train.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    @patch("chemi_accel.spectroscopy.predictive_models.GPUSpectroscopyPredictor")
    def test_predict_spectrum_batch(self, mock_predictor_class):
        """Test batch spectrum prediction."""
        # Mock predictor
        mock_predictor = Mock()
        mock_predictions = [
            SpectralPrediction(
                frequencies=np.linspace(0, 4000, 100),
                intensities=np.random.rand(100),
                spectroscopy_type=SpectroscopyType.IR,
            )
            for _ in range(10)
        ]
        mock_predictor.predict.return_value = mock_predictions[:5]

        features = np.random.rand(10, 256)

        with patch("time.time", side_effect=[0, 1.5]):  # Mock timing
            result = predict_spectrum_batch(mock_predictor, features, batch_size=5)

        assert len(result.predictions) == 10
        assert result.batch_computation_time == 1.5
        assert "samples_per_second" in result.model_performance_metrics
        assert result.model_performance_metrics["samples_per_second"] == 10 / 1.5

    @patch("chemi_accel.spectroscopy.predictive_models.IRPredictor")
    def test_train_spectral_model_ir(self, mock_predictor_class):
        """Test training IR spectral model."""
        mock_predictor = Mock()
        mock_predictor.train.return_value = {"final_loss": 0.05}
        mock_predictor_class.return_value = mock_predictor

        # Training data
        features = np.random.rand(50, 256)
        spectra = np.random.rand(50, 1000)
        frequencies = np.linspace(400, 4000, 1000)

        training_data = TrainingData(
            molecular_features=features, spectra=spectra, frequencies=frequencies
        )

        with patch("builtins.print"):  # Mock print function
            model = train_spectral_model("ir", training_data)

        assert model is not None
        mock_predictor_class.assert_called_once()
        mock_predictor.train.assert_called_once_with(training_data)

    def test_evaluate_model_accuracy(self):
        """Test model accuracy evaluation."""
        # Mock predictor
        mock_predictor = Mock()

        # Create test predictions
        test_spectra = np.random.rand(5, 100)
        mock_predictions = []

        for i in range(5):
            # Add some noise to create realistic predictions
            pred_spectrum = test_spectra[i] + np.random.normal(0, 0.1, 100)
            pred = SpectralPrediction(
                frequencies=np.linspace(0, 4000, 100),
                intensities=pred_spectrum,
                spectroscopy_type=SpectroscopyType.IR,
            )
            mock_predictions.append(pred)

        mock_predictor.predict.return_value = mock_predictions

        # Test data
        features = np.random.rand(5, 256)
        frequencies = np.linspace(0, 4000, 100)

        test_data = TrainingData(
            molecular_features=features, spectra=test_spectra, frequencies=frequencies
        )

        metrics = evaluate_model_accuracy(mock_predictor, test_data)

        assert "mean_squared_error" in metrics
        assert "mean_absolute_error" in metrics
        assert "r_squared" in metrics
        assert "samples_evaluated" in metrics
        assert metrics["samples_evaluated"] == 5


class TestExceptionHandling:
    """Test exception handling in predictive models."""

    def test_prediction_error(self):
        """Test PredictionError."""
        with pytest.raises(PredictionError, match="Test prediction error"):
            raise PredictionError("Test prediction error")

    def test_model_error(self):
        """Test ModelError."""
        with pytest.raises(ModelError, match="Test model error"):
            raise ModelError("Test model error")

    def test_training_error(self):
        """Test TrainingError."""
        with pytest.raises(TrainingError, match="Test training error"):
            raise TrainingError("Test training error")


class TestIntegration:
    """Integration tests for predictive models."""

    @patch("chemi_accel.spectroscopy.predictive_models.TORCH_AVAILABLE", True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_end_to_end_workflow(self):
        """Test complete workflow from training to prediction."""
        # Configuration
        config = ModelConfiguration(
            architecture=ModelArchitecture.TRANSFORMER,
            spectroscopy_type=SpectroscopyType.IR,
            num_epochs=1,  # Quick training for test
            batch_size=16,
        )

        # Mock training data
        features = np.random.rand(32, 256)
        spectra = np.random.rand(32, 100)
        frequencies = np.linspace(400, 4000, 100)

        training_data = TrainingData(
            molecular_features=features, spectra=spectra, frequencies=frequencies
        )

        # Mock PyTorch components
        with patch("torch.tensor") as mock_tensor, patch(
            "torch.utils.data.DataLoader"
        ) as mock_dataloader, patch("torch.optim.Adam") as mock_optimizer, patch(
            "torch.nn.MSELoss"
        ) as mock_loss:
            # Setup mocks
            mock_tensor.return_value = Mock()
            mock_dataloader.return_value = [
                (Mock(), Mock()) for _ in range(2)  # 2 batches
            ]
            mock_optimizer.return_value = Mock()
            mock_loss.return_value = Mock()
            mock_loss.return_value.return_value.item.return_value = 0.1

            predictor = GPUSpectroscopyPredictor(config)

            # Mock model
            predictor.model = Mock()
            predictor.model.train = Mock()
            predictor.model.eval = Mock()
            predictor.model.to = Mock(return_value=predictor.model)

            # Train model
            with patch("torch.cuda.amp.GradScaler"):
                metrics = predictor.train(training_data)

            assert predictor.is_trained
            assert "final_loss" in metrics

            # Test prediction
            test_features = np.random.rand(5, 256)

            # Mock prediction
            mock_output = Mock()
            mock_output.cpu.return_value.numpy.return_value = np.random.rand(5, 100)
            predictor.model.return_value = mock_output

            with patch("torch.no_grad"), patch("torch.tensor") as mock_tensor_pred:
                mock_tensor_pred.return_value.to.return_value = Mock()
                predictions = predictor.predict(test_features)

            assert len(predictions) == 5
            assert all(isinstance(p, SpectralPrediction) for p in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

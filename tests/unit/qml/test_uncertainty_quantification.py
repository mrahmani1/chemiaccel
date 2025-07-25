"""
Tests for QML uncertainty quantification module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.chemi_accel.ai.uncertainty_quantification import UncertaintyResult
from src.chemi_accel.qml.base import QMLPrediction, QuantumProperty
from src.chemi_accel.qml.uncertainty_quantification import (
    EnsembleUncertaintyExtractor,
    MonteCarloDropoutExtractor,
    QMLUncertaintyConfig,
    QMLUncertaintyError,
    QMLUncertaintyQuantifier,
    QMLUncertaintyResult,
    QMLUQIntegration,
    UncertaintyType,
)


class TestQMLUncertaintyConfig:
    """Test QML uncertainty configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QMLUncertaintyConfig()

        assert config.uncertainty_methods == ["ensemble", "mc_dropout"]
        assert config.ensemble_size == 10
        assert config.mc_samples == 100
        assert config.confidence_levels == [0.68, 0.95, 0.99]
        assert config.calibration_method == "temperature_scaling"
        assert config.enable_reliability_checking is True
        assert config.uncertainty_threshold == 0.15
        assert config.min_confidence == 0.7

    def test_custom_config(self):
        """Test custom configuration."""
        config = QMLUncertaintyConfig(
            uncertainty_methods=["ensemble"],
            ensemble_size=5,
            confidence_levels=[0.9],
            uncertainty_threshold=0.2,
        )

        assert config.uncertainty_methods == ["ensemble"]
        assert config.ensemble_size == 5
        assert config.confidence_levels == [0.9]
        assert config.uncertainty_threshold == 0.2


class TestQMLUncertaintyResult:
    """Test QML uncertainty result class."""

    def test_get_reliable_predictions(self):
        """Test reliable predictions filtering."""
        # Create mock result
        result = QMLUncertaintyResult(
            predictions={QuantumProperty.HOMO_ENERGY: np.array([1.0, 2.0, 3.0])},
            epistemic_uncertainty={
                QuantumProperty.HOMO_ENERGY: np.array([0.1, 0.2, 0.05])
            },
            aleatoric_uncertainty={
                QuantumProperty.HOMO_ENERGY: np.array([0.05, 0.1, 0.02])
            },
            total_uncertainty={
                QuantumProperty.HOMO_ENERGY: np.array([0.11, 0.22, 0.054])
            },
            prediction_intervals={},
            reliability_scores={
                QuantumProperty.HOMO_ENERGY: np.array([0.85, 0.65, 0.95])
            },
            calibration_metrics={},
            method_info={},
        )

        # Test with default threshold (0.8)
        reliable_indices = result.get_reliable_predictions(QuantumProperty.HOMO_ENERGY)
        expected = np.array([0, 2])  # Indices with reliability >= 0.8
        np.testing.assert_array_equal(reliable_indices, expected)

        # Test with custom threshold
        reliable_indices = result.get_reliable_predictions(
            QuantumProperty.HOMO_ENERGY, reliability_threshold=0.9
        )
        expected = np.array([2])  # Only index 2 has reliability >= 0.9
        np.testing.assert_array_equal(reliable_indices, expected)

        # Test with non-existent property
        reliable_indices = result.get_reliable_predictions(QuantumProperty.LUMO_ENERGY)
        assert len(reliable_indices) == 0


class TestEnsembleUncertaintyExtractor:
    """Test ensemble uncertainty extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = QMLUncertaintyConfig(ensemble_size=3)
        self.extractor = EnsembleUncertaintyExtractor(self.config)

        # Mock QML model with ensemble prediction capability
        self.mock_model = Mock()

        # Create mock ensemble predictions
        self.mock_predictions = [
            QMLPrediction(
                properties={QuantumProperty.HOMO_ENERGY: -5.0 + i * 0.1},
                uncertainty={QuantumProperty.HOMO_ENERGY: 0.05},
            )
            for i in range(3)
        ]

        self.mock_model.predict_ensemble.return_value = self.mock_predictions

    def test_extract_uncertainty_success(self):
        """Test successful uncertainty extraction from ensemble."""
        molecules = ["CCO", "CCC"]
        properties = [QuantumProperty.HOMO_ENERGY]

        result = self.extractor.extract_uncertainty(
            self.mock_model, molecules, properties
        )

        # Verify structure
        assert isinstance(result, QMLUncertaintyResult)
        assert QuantumProperty.HOMO_ENERGY in result.predictions
        assert QuantumProperty.HOMO_ENERGY in result.epistemic_uncertainty
        assert QuantumProperty.HOMO_ENERGY in result.aleatoric_uncertainty
        assert QuantumProperty.HOMO_ENERGY in result.total_uncertainty

        # Verify method info
        assert result.method_info["method_type"] == "ensemble"
        assert result.method_info["ensemble_size"] == 3

        # Verify model was called correctly
        self.mock_model.predict_ensemble.assert_called_once_with(
            molecules, properties, n_models=3
        )

    def test_extract_uncertainty_no_ensemble_support(self):
        """Test error when model doesn't support ensemble predictions."""
        # Create model without ensemble support
        mock_model = Mock()
        del mock_model.predict_ensemble

        molecules = ["CCO"]
        properties = [QuantumProperty.HOMO_ENERGY]

        with pytest.raises(QMLUncertaintyError, match="does not support ensemble"):
            self.extractor.extract_uncertainty(mock_model, molecules, properties)

    @patch("src.chemi_accel.qml.uncertainty_quantification.logger")
    def test_calibrate_uncertainty(self, mock_logger):
        """Test uncertainty calibration."""
        validation_data = [
            ("CCO", {QuantumProperty.HOMO_ENERGY: -5.1}),
            ("CCC", {QuantumProperty.HOMO_ENERGY: -4.9}),
        ]

        self.extractor.calibrate_uncertainty(validation_data)

        # Verify calibration parameters are set
        assert "method" in self.extractor.calibration_params
        assert self.extractor.calibration_params["method"] == "temperature_scaling"
        assert self.extractor.calibration_params["n_validation_points"] == 2

        # Verify logging
        mock_logger.info.assert_called_once()

    def test_get_confidence_z_score(self):
        """Test z-score calculation for confidence levels."""
        # Test common confidence levels
        assert abs(self.extractor._get_confidence_z_score(0.68) - 1.0) < 0.05
        assert abs(self.extractor._get_confidence_z_score(0.95) - 1.96) < 0.05
        assert abs(self.extractor._get_confidence_z_score(0.99) - 2.58) < 0.05


class TestMonteCarloDropoutExtractor:
    """Test Monte Carlo Dropout uncertainty extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = QMLUncertaintyConfig(mc_samples=5)
        self.extractor = MonteCarloDropoutExtractor(self.config)

        # Mock QML model with MC dropout capability
        self.mock_model = Mock()

        # Create mock MC dropout predictions
        self.mock_predictions = [
            QMLPrediction(
                properties={
                    QuantumProperty.HOMO_ENERGY: -5.0 + np.random.normal(0, 0.1)
                }
            )
            for _ in range(5)
        ]

        self.mock_model.predict_mc_dropout.return_value = self.mock_predictions

    def test_extract_uncertainty_success(self):
        """Test successful MC dropout uncertainty extraction."""
        molecules = ["CCO", "CCC"]
        properties = [QuantumProperty.HOMO_ENERGY]

        result = self.extractor.extract_uncertainty(
            self.mock_model, molecules, properties
        )

        # Verify structure
        assert isinstance(result, QMLUncertaintyResult)
        assert QuantumProperty.HOMO_ENERGY in result.predictions
        assert QuantumProperty.HOMO_ENERGY in result.epistemic_uncertainty

        # MC dropout should have zero aleatoric uncertainty
        assert QuantumProperty.HOMO_ENERGY in result.aleatoric_uncertainty
        aleatoric = result.aleatoric_uncertainty[QuantumProperty.HOMO_ENERGY]
        np.testing.assert_array_equal(aleatoric, np.zeros_like(aleatoric))

        # Verify method info
        assert result.method_info["method_type"] == "mc_dropout"
        assert result.method_info["n_samples"] == 5

        # Verify model was called correctly
        self.mock_model.predict_mc_dropout.assert_called_once_with(
            molecules, properties, n_samples=5
        )

    def test_extract_uncertainty_no_mc_dropout_support(self):
        """Test error when model doesn't support MC dropout."""
        mock_model = Mock()
        del mock_model.predict_mc_dropout

        molecules = ["CCO"]
        properties = [QuantumProperty.HOMO_ENERGY]

        with pytest.raises(QMLUncertaintyError, match="does not support MC Dropout"):
            self.extractor.extract_uncertainty(mock_model, molecules, properties)


class TestQMLUncertaintyQuantifier:
    """Test main QML uncertainty quantifier."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = QMLUncertaintyConfig(uncertainty_methods=["ensemble"])
        self.quantifier = QMLUncertaintyQuantifier(self.config)

        # Mock model and result
        self.mock_model = Mock()
        self.mock_result = QMLUncertaintyResult(
            predictions={QuantumProperty.HOMO_ENERGY: np.array([1.0, 2.0])},
            epistemic_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.1, 0.2])},
            aleatoric_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.05, 0.1])},
            total_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.11, 0.22])},
            prediction_intervals={},
            reliability_scores={QuantumProperty.HOMO_ENERGY: np.array([0.9, 0.8])},
            calibration_metrics={},
            method_info={"method_type": "ensemble"},
        )

    def test_initialization(self):
        """Test quantifier initialization."""
        assert "ensemble" in self.quantifier.extractors
        assert isinstance(
            self.quantifier.extractors["ensemble"], EnsembleUncertaintyExtractor
        )

        # Test with MC dropout
        config = QMLUncertaintyConfig(uncertainty_methods=["mc_dropout"])
        quantifier = QMLUncertaintyQuantifier(config)
        assert "mc_dropout" in quantifier.extractors
        assert isinstance(
            quantifier.extractors["mc_dropout"], MonteCarloDropoutExtractor
        )

    @patch.object(EnsembleUncertaintyExtractor, "extract_uncertainty")
    def test_quantify_uncertainty_success(self, mock_extract):
        """Test successful uncertainty quantification."""
        mock_extract.return_value = self.mock_result

        molecules = ["CCO"]
        properties = [QuantumProperty.HOMO_ENERGY]

        result = self.quantifier.quantify_uncertainty(
            self.mock_model, molecules, properties
        )

        assert result == self.mock_result
        mock_extract.assert_called_once_with(self.mock_model, molecules, properties)

    def test_quantify_uncertainty_invalid_method(self):
        """Test error with invalid uncertainty method."""
        molecules = ["CCO"]
        properties = [QuantumProperty.HOMO_ENERGY]

        with pytest.raises(QMLUncertaintyError, match="not available"):
            self.quantifier.quantify_uncertainty(
                self.mock_model, molecules, properties, method="invalid_method"
            )

    def test_combine_uncertainties_average(self):
        """Test combining uncertainties using average method."""
        # Create two mock results
        result1 = QMLUncertaintyResult(
            predictions={QuantumProperty.HOMO_ENERGY: np.array([1.0, 2.0])},
            epistemic_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.1, 0.2])},
            aleatoric_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.05, 0.1])},
            total_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.11, 0.22])},
            prediction_intervals={},
            reliability_scores={QuantumProperty.HOMO_ENERGY: np.array([0.9, 0.8])},
            calibration_metrics={"mae": 0.1},
            method_info={"method_type": "ensemble"},
        )

        result2 = QMLUncertaintyResult(
            predictions={QuantumProperty.HOMO_ENERGY: np.array([1.1, 1.9])},
            epistemic_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.12, 0.18])},
            aleatoric_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.06, 0.09])},
            total_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.13, 0.20])},
            prediction_intervals={},
            reliability_scores={QuantumProperty.HOMO_ENERGY: np.array([0.85, 0.85])},
            calibration_metrics={"mae": 0.12},
            method_info={"method_type": "mc_dropout"},
        )

        combined = self.quantifier.combine_uncertainties([result1, result2])

        # Verify averaging
        expected_pred = np.array([1.05, 1.95])
        np.testing.assert_array_almost_equal(
            combined.predictions[QuantumProperty.HOMO_ENERGY], expected_pred
        )

        # Verify method info
        assert combined.method_info["method_type"] == "combined"
        assert combined.method_info["combination_method"] == "average"
        assert combined.method_info["n_methods"] == 2

        # Verify calibration metrics averaging
        assert combined.calibration_metrics["mae"] == 0.11

    def test_combine_uncertainties_empty_list(self):
        """Test error when combining empty results list."""
        with pytest.raises(QMLUncertaintyError, match="No results to combine"):
            self.quantifier.combine_uncertainties([])

    @patch.object(EnsembleUncertaintyExtractor, "calibrate_uncertainty")
    @patch("src.chemi_accel.qml.uncertainty_quantification.logger")
    def test_calibrate_all_methods(self, mock_logger, mock_calibrate):
        """Test calibrating all uncertainty methods."""
        validation_data = [("CCO", {QuantumProperty.HOMO_ENERGY: -5.0})]

        self.quantifier.calibrate_all_methods(validation_data)

        # Verify calibration was called
        mock_calibrate.assert_called_once_with(validation_data)
        mock_logger.info.assert_called_once()

    def test_filter_unreliable_predictions(self):
        """Test filtering unreliable predictions."""
        # Create result with mixed reliability
        result = QMLUncertaintyResult(
            predictions={QuantumProperty.HOMO_ENERGY: np.array([1.0, 2.0, 3.0])},
            epistemic_uncertainty={
                QuantumProperty.HOMO_ENERGY: np.array([0.1, 0.2, 0.05])
            },
            aleatoric_uncertainty={
                QuantumProperty.HOMO_ENERGY: np.array([0.05, 0.1, 0.02])
            },
            total_uncertainty={
                QuantumProperty.HOMO_ENERGY: np.array([0.11, 0.22, 0.054])
            },
            prediction_intervals={
                QuantumProperty.HOMO_ENERGY: {
                    0.95: (np.array([0.8, 1.6, 2.9]), np.array([1.2, 2.4, 3.1]))
                }
            },
            reliability_scores={
                QuantumProperty.HOMO_ENERGY: np.array([0.85, 0.65, 0.95])
            },
            calibration_metrics={},
            method_info={},
        )

        # Set low confidence threshold to filter out middle prediction
        config = QMLUncertaintyConfig(min_confidence=0.7)
        quantifier = QMLUncertaintyQuantifier(config)

        filtered_result = quantifier._filter_unreliable_predictions(result)

        # Should keep only indices 0 and 2
        expected_predictions = np.array([1.0, 3.0])
        np.testing.assert_array_equal(
            filtered_result.predictions[QuantumProperty.HOMO_ENERGY],
            expected_predictions,
        )

        # Check prediction intervals are also filtered
        lower, upper = filtered_result.prediction_intervals[
            QuantumProperty.HOMO_ENERGY
        ][0.95]
        np.testing.assert_array_equal(lower, np.array([0.8, 2.9]))
        np.testing.assert_array_equal(upper, np.array([1.2, 3.1]))


class TestQMLUQIntegration:
    """Test integration with standard UQ framework."""

    def test_qml_to_standard_uq(self):
        """Test conversion from QML to standard UQ format."""
        qml_result = QMLUncertaintyResult(
            predictions={QuantumProperty.HOMO_ENERGY: np.array([1.0, 2.0])},
            epistemic_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.1, 0.2])},
            aleatoric_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.05, 0.1])},
            total_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.11, 0.22])},
            prediction_intervals={
                QuantumProperty.HOMO_ENERGY: {
                    0.95: (np.array([0.8, 1.6]), np.array([1.2, 2.4]))
                }
            },
            reliability_scores={QuantumProperty.HOMO_ENERGY: np.array([0.9, 0.8])},
            calibration_metrics={"calibration_error": 0.05},
            method_info={"method_type": "ensemble"},
        )

        standard_result = QMLUQIntegration.qml_to_standard_uq(
            qml_result, QuantumProperty.HOMO_ENERGY
        )

        assert isinstance(standard_result, UncertaintyResult)
        np.testing.assert_array_equal(standard_result.predictions, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(
            standard_result.uncertainties, np.array([0.11, 0.22])
        )
        assert standard_result.method == "ensemble"
        assert standard_result.calibration_score == 0.05

        # Check confidence intervals
        expected_intervals = np.array([[0.8, 1.2], [1.6, 2.4]])
        np.testing.assert_array_equal(
            standard_result.confidence_intervals, expected_intervals
        )

    def test_qml_to_standard_uq_missing_property(self):
        """Test error when property is missing from QML result."""
        qml_result = QMLUncertaintyResult(
            predictions={},
            epistemic_uncertainty={},
            aleatoric_uncertainty={},
            total_uncertainty={},
            prediction_intervals={},
            reliability_scores={},
            calibration_metrics={},
            method_info={},
        )

        with pytest.raises(QMLUncertaintyError, match="No data for property"):
            QMLUQIntegration.qml_to_standard_uq(qml_result, QuantumProperty.HOMO_ENERGY)

    def test_enhance_qml_prediction(self):
        """Test enhancing QML prediction with uncertainty info."""
        # Create original prediction
        prediction = QMLPrediction(
            properties={QuantumProperty.HOMO_ENERGY: -5.0}, model_name="test_model"
        )

        # Create uncertainty result
        uncertainty_result = QMLUncertaintyResult(
            predictions={QuantumProperty.HOMO_ENERGY: np.array([-5.0])},
            epistemic_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.1])},
            aleatoric_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.05])},
            total_uncertainty={QuantumProperty.HOMO_ENERGY: np.array([0.11])},
            prediction_intervals={
                QuantumProperty.HOMO_ENERGY: {
                    0.95: (np.array([-5.2]), np.array([-4.8]))
                }
            },
            reliability_scores={QuantumProperty.HOMO_ENERGY: np.array([0.9])},
            calibration_metrics={"calibration_error": 0.05},
            method_info={"method_type": "ensemble"},
        )

        enhanced = QMLUQIntegration.enhance_qml_prediction(
            prediction, uncertainty_result
        )

        # Verify uncertainty info was added
        assert enhanced.uncertainty[QuantumProperty.HOMO_ENERGY] == 0.11
        assert enhanced.epistemic_uncertainty[QuantumProperty.HOMO_ENERGY] == 0.1
        assert enhanced.aleatoric_uncertainty[QuantumProperty.HOMO_ENERGY] == 0.05
        assert enhanced.prediction_intervals[QuantumProperty.HOMO_ENERGY] == (
            -5.2,
            -4.8,
        )
        assert enhanced.uncertainty_method == "ensemble"
        assert enhanced.calibration_score == 0.05


@pytest.mark.parametrize(
    "uncertainty_type",
    [
        UncertaintyType.EPISTEMIC,
        UncertaintyType.ALEATORIC,
        UncertaintyType.TOTAL,
        UncertaintyType.ENSEMBLE,
    ],
)
def test_uncertainty_type_enum(uncertainty_type):
    """Test uncertainty type enumeration."""
    assert isinstance(uncertainty_type.value, str)
    assert len(uncertainty_type.value) > 0


class TestQMLPredictionUncertaintyMethods:
    """Test uncertainty methods added to QMLPrediction class."""

    def setup_method(self):
        """Set up test prediction with uncertainty info."""
        self.prediction = QMLPrediction(
            properties={QuantumProperty.HOMO_ENERGY: -5.0},
            uncertainty={QuantumProperty.HOMO_ENERGY: 0.15},
            epistemic_uncertainty={QuantumProperty.HOMO_ENERGY: 0.1},
            aleatoric_uncertainty={QuantumProperty.HOMO_ENERGY: 0.05},
            prediction_intervals={QuantumProperty.HOMO_ENERGY: (-5.3, -4.7)},
        )

    def test_get_uncertainty_total(self):
        """Test getting total uncertainty."""
        uncertainty = self.prediction.get_uncertainty(
            QuantumProperty.HOMO_ENERGY, "total"
        )
        assert uncertainty == 0.15

    def test_get_uncertainty_epistemic(self):
        """Test getting epistemic uncertainty."""
        uncertainty = self.prediction.get_uncertainty(
            QuantumProperty.HOMO_ENERGY, "epistemic"
        )
        assert uncertainty == 0.1

    def test_get_uncertainty_aleatoric(self):
        """Test getting aleatoric uncertainty."""
        uncertainty = self.prediction.get_uncertainty(
            QuantumProperty.HOMO_ENERGY, "aleatoric"
        )
        assert uncertainty == 0.05

    def test_get_uncertainty_missing(self):
        """Test getting uncertainty for missing property."""
        uncertainty = self.prediction.get_uncertainty(
            QuantumProperty.LUMO_ENERGY, "total"
        )
        assert uncertainty is None

    def test_get_prediction_interval(self):
        """Test getting prediction interval."""
        interval = self.prediction.get_prediction_interval(QuantumProperty.HOMO_ENERGY)
        assert interval == (-5.3, -4.7)

    def test_get_prediction_interval_missing(self):
        """Test getting prediction interval for missing property."""
        interval = self.prediction.get_prediction_interval(QuantumProperty.LUMO_ENERGY)
        assert interval is None

    def test_is_reliable_true(self):
        """Test reliability check - should be reliable."""
        # With property value -5.0 and uncertainty 0.15, relative uncertainty is 0.03 < 0.1
        reliable = self.prediction.is_reliable(
            QuantumProperty.HOMO_ENERGY, uncertainty_threshold=0.1
        )
        assert reliable is True

    def test_is_reliable_false(self):
        """Test reliability check - should be unreliable."""
        # With relative uncertainty 0.03, should fail threshold of 0.01
        reliable = self.prediction.is_reliable(
            QuantumProperty.HOMO_ENERGY, uncertainty_threshold=0.01
        )
        assert reliable is False

    def test_is_reliable_no_uncertainty(self):
        """Test reliability check when no uncertainty info available."""
        prediction = QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -5.0})
        reliable = prediction.is_reliable(QuantumProperty.HOMO_ENERGY)
        assert reliable is True  # Should assume reliable if no uncertainty info


if __name__ == "__main__":
    pytest.main([__file__])

"""
Test cases for toxicology core functionality.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.core.backend import get_backend
from chemi_accel.toxicology.core import (
    ConfidenceLevel,
    SpeciesType,
    ToxicityData,
    ToxicityEndpoint,
    ToxicityEndpointType,
    ToxicityPrediction,
    ToxicologyCalculator,
)


class TestToxicityEndpoint:
    """Test ToxicityEndpoint class."""

    def test_endpoint_creation(self):
        """Test creating toxicity endpoint."""
        endpoint = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.ACUTE_ORAL,
            species=SpeciesType.RAT,
            value=150.0,
            unit="mg/kg",
        )

        assert endpoint.endpoint_type == ToxicityEndpointType.ACUTE_ORAL
        assert endpoint.species == SpeciesType.RAT
        assert endpoint.value == 150.0
        assert endpoint.unit == "mg/kg"

    def test_negative_value_validation(self):
        """Test validation of negative toxicity values."""
        with pytest.raises(ValueError, match="Toxicity value cannot be negative"):
            ToxicityEndpoint(
                endpoint_type=ToxicityEndpointType.ACUTE_ORAL,
                species=SpeciesType.RAT,
                value=-10.0,
            )


class TestToxicityPrediction:
    """Test ToxicityPrediction class."""

    def test_prediction_creation(self):
        """Test creating toxicity prediction."""
        endpoint = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
        )

        prediction = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=250.0,
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.15,
            model_type="multi_task_dnn",
        )

        assert prediction.predicted_value == 250.0
        assert prediction.confidence == ConfidenceLevel.HIGH
        assert prediction.uncertainty == 0.15
        assert prediction.model_type == "multi_task_dnn"

    def test_is_toxic_acute_oral(self):
        """Test toxicity classification for acute oral."""
        endpoint = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
        )

        # Toxic prediction (< 2000 mg/kg)
        toxic_pred = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=500.0,
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        assert toxic_pred.is_toxic is True

        # Non-toxic prediction (>= 2000 mg/kg)
        non_toxic_pred = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=3000.0,
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        assert non_toxic_pred.is_toxic is False

    def test_is_toxic_carcinogenicity(self):
        """Test toxicity classification for carcinogenicity."""
        endpoint = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.CARCINOGENICITY, species=SpeciesType.RAT
        )

        # Carcinogenic prediction (> 0.5 probability)
        carcinogenic_pred = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=0.8,
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        assert carcinogenic_pred.is_toxic is True

        # Non-carcinogenic prediction (<= 0.5 probability)
        non_carcinogenic_pred = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=0.3,
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        assert non_carcinogenic_pred.is_toxic is False


class TestToxicityData:
    """Test ToxicityData class."""

    def test_toxicity_data_creation(self):
        """Test creating toxicity data."""
        data = ToxicityData(smiles="CCO")
        assert data.smiles == "CCO"
        assert len(data.experimental_endpoints) == 0
        assert len(data.predicted_endpoints) == 0

    def test_get_endpoint_predictions(self):
        """Test getting predictions for specific endpoint."""
        # Create test predictions
        endpoint1 = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
        )
        endpoint2 = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.CARCINOGENICITY, species=SpeciesType.RAT
        )

        pred1 = ToxicityPrediction(
            endpoint=endpoint1,
            predicted_value=500.0,
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        pred2 = ToxicityPrediction(
            endpoint=endpoint2,
            predicted_value=0.7,
            confidence=ConfidenceLevel.MODERATE,
            uncertainty=0.2,
            model_type="test",
        )

        data = ToxicityData(smiles="CCO", predicted_endpoints=[pred1, pred2])

        # Get acute oral predictions
        acute_preds = data.get_endpoint_predictions(ToxicityEndpointType.ACUTE_ORAL)
        assert len(acute_preds) == 1
        assert acute_preds[0] == pred1

        # Get carcinogenicity predictions
        carc_preds = data.get_endpoint_predictions(ToxicityEndpointType.CARCINOGENICITY)
        assert len(carc_preds) == 1
        assert carc_preds[0] == pred2

        # Get non-existent endpoint
        geno_preds = data.get_endpoint_predictions(ToxicityEndpointType.GENOTOXICITY)
        assert len(geno_preds) == 0

    def test_get_highest_risk_endpoints(self):
        """Test getting highest risk endpoints."""
        # Create test predictions - mix of toxic and non-toxic
        endpoints = [
            ToxicityEndpoint(
                endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
            ),
            ToxicityEndpoint(
                endpoint_type=ToxicityEndpointType.CARCINOGENICITY,
                species=SpeciesType.RAT,
            ),
            ToxicityEndpoint(
                endpoint_type=ToxicityEndpointType.GENOTOXICITY, species=SpeciesType.RAT
            ),
        ]

        predictions = [
            ToxicityPrediction(
                endpoint=endpoints[0],
                predicted_value=500.0,
                confidence=ConfidenceLevel.HIGH,
                uncertainty=0.1,
                model_type="test",
            ),  # Toxic
            ToxicityPrediction(
                endpoint=endpoints[1],
                predicted_value=0.8,
                confidence=ConfidenceLevel.HIGH,
                uncertainty=0.1,
                model_type="test",
            ),  # Toxic
            ToxicityPrediction(
                endpoint=endpoints[2],
                predicted_value=3000.0,
                confidence=ConfidenceLevel.HIGH,
                uncertainty=0.1,
                model_type="test",
            ),  # Non-toxic
        ]

        data = ToxicityData(smiles="CCO", predicted_endpoints=predictions)

        high_risk = data.get_highest_risk_endpoints(n=2)
        assert len(high_risk) == 2
        # Should return the two toxic predictions, ordered by value (descending)
        assert (
            high_risk[0].endpoint.endpoint_type == ToxicityEndpointType.CARCINOGENICITY
        )
        assert high_risk[1].endpoint.endpoint_type == ToxicityEndpointType.ACUTE_ORAL

    def test_overall_risk_score(self):
        """Test overall risk score calculation."""
        # Create predictions with different endpoint types and weights
        endpoints = [
            ToxicityEndpoint(
                endpoint_type=ToxicityEndpointType.CARCINOGENICITY,
                species=SpeciesType.RAT,
            ),
            ToxicityEndpoint(
                endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
            ),
        ]

        predictions = [
            ToxicityPrediction(
                endpoint=endpoints[0],
                predicted_value=0.8,
                confidence=ConfidenceLevel.HIGH,
                uncertainty=0.1,
                model_type="test",
            ),
            ToxicityPrediction(
                endpoint=endpoints[1],
                predicted_value=500.0,
                confidence=ConfidenceLevel.HIGH,
                uncertainty=0.1,
                model_type="test",
            ),
        ]

        data = ToxicityData(smiles="CCO", predicted_endpoints=predictions)
        risk_score = data.overall_risk_score

        assert 0.0 <= risk_score <= 1.0
        assert risk_score > 0  # Should have some risk due to toxic predictions

        # Test with no predictions
        empty_data = ToxicityData(smiles="CCO")
        assert empty_data.overall_risk_score == 0.0


class TestToxicologyCalculator:
    """Test ToxicologyCalculator class."""

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = ToxicologyCalculator(backend="cpu", use_gpu=False)
        assert calc.backend.name == "cpu"
        assert calc.uncertainty_quantification is True

    def test_setup_backend(self):
        """Test backend setup."""
        # Test with string backend
        calc = ToxicologyCalculator(backend="cpu")
        assert calc.backend.name == "cpu"

        # Test with Backend object
        backend = get_backend("cpu")
        calc = ToxicologyCalculator(backend=backend)
        assert calc.backend == backend

        # Test with None (should default)
        calc = ToxicologyCalculator(backend=None, use_gpu=False)
        assert calc.backend.name == "cpu"

    @patch("chemi_accel.toxicology.core.ToxicologyCalculator.multi_endpoint_predictor")
    def test_predict_toxicity_single_compound(self, mock_predictor):
        """Test toxicity prediction for single compound."""
        # Mock the predictor
        mock_prediction = ToxicityPrediction(
            endpoint=ToxicityEndpoint(
                endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
            ),
            predicted_value=500.0,
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        mock_predictor.predict_endpoints.return_value = [mock_prediction]

        calc = ToxicologyCalculator(backend="cpu")
        result = calc.predict_toxicity("CCO")

        assert isinstance(result, ToxicityData)
        assert result.smiles == "CCO"
        assert len(result.predicted_endpoints) == 1
        assert result.predicted_endpoints[0] == mock_prediction

    @patch("chemi_accel.toxicology.core.ToxicologyCalculator.multi_endpoint_predictor")
    def test_predict_toxicity_multiple_compounds(self, mock_predictor):
        """Test toxicity prediction for multiple compounds."""
        # Mock the predictor
        mock_prediction = ToxicityPrediction(
            endpoint=ToxicityEndpoint(
                endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
            ),
            predicted_value=500.0,
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        mock_predictor.predict_endpoints.return_value = [mock_prediction]

        calc = ToxicologyCalculator(backend="cpu")
        results = calc.predict_toxicity(["CCO", "CCC"])

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, ToxicityData) for r in results)
        assert results[0].smiles == "CCO"
        assert results[1].smiles == "CCC"

    def test_property_lazy_loading(self):
        """Test lazy loading of predictor properties."""
        calc = ToxicologyCalculator(backend="cpu")

        # Properties should be None initially
        assert calc._multi_endpoint_predictor is None
        assert calc._environmental_fate_predictor is None
        assert calc._metabolic_predictor is None
        assert calc._risk_assessor is None
        assert calc._safer_designer is None

        # Accessing properties should create instances
        multi_endpoint = calc.multi_endpoint_predictor
        assert multi_endpoint is not None
        assert calc._multi_endpoint_predictor is multi_endpoint

        # Subsequent access should return same instance
        assert calc.multi_endpoint_predictor is multi_endpoint

    def test_create_toxicity_report(self):
        """Test toxicity report generation."""
        # Create test toxicity data
        endpoint = ToxicityEndpoint(
            endpoint_type=ToxicityEndpointType.ACUTE_ORAL, species=SpeciesType.RAT
        )
        prediction = ToxicityPrediction(
            endpoint=endpoint,
            predicted_value=500.0,
            confidence=ConfidenceLevel.HIGH,
            uncertainty=0.1,
            model_type="test",
        )
        toxicity_data = ToxicityData(smiles="CCO", predicted_endpoints=[prediction])

        calc = ToxicologyCalculator(backend="cpu")

        # Test HTML report
        html_report = calc.create_toxicity_report(
            toxicity_data,
            output_format="html",
            include_environmental_fate=False,
            include_metabolic_pathways=False,
            include_risk_assessment=False,
        )
        assert "<html>" in html_report
        assert "CCO" in html_report
        assert "Acute Oral" in html_report

        # Test JSON report
        json_report = calc.create_toxicity_report(
            toxicity_data,
            output_format="json",
            include_environmental_fate=False,
            include_metabolic_pathways=False,
            include_risk_assessment=False,
        )
        assert '"compound": "CCO"' in json_report

        # Test text report
        text_report = calc.create_toxicity_report(
            toxicity_data,
            output_format="text",
            include_environmental_fate=False,
            include_metabolic_pathways=False,
            include_risk_assessment=False,
        )
        assert "CCO" in text_report
        assert "Acute Oral" in text_report

    def test_unsupported_report_format(self):
        """Test error handling for unsupported report format."""
        toxicity_data = ToxicityData(smiles="CCO")
        calc = ToxicologyCalculator(backend="cpu")

        with pytest.raises(ValueError, match="Unsupported report format"):
            calc.create_toxicity_report(toxicity_data, output_format="pdf")


if __name__ == "__main__":
    pytest.main([__file__])

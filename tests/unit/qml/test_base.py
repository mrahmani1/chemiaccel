"""
Tests for QML base classes and functionality.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from chemi_accel.qml.base import (
    BaseQMLModel,
    QMLModelError,
    QMLModelRegistry,
    QMLPrediction,
    QuantumProperty,
    qml_registry,
)


class TestQuantumProperty:
    """Test QuantumProperty enum."""

    def test_property_values(self):
        """Test that all properties have correct string values."""
        assert QuantumProperty.HOMO_ENERGY.value == "homo_energy"
        assert QuantumProperty.LUMO_ENERGY.value == "lumo_energy"
        assert QuantumProperty.TOTAL_ENERGY.value == "total_energy"
        assert QuantumProperty.DIPOLE_MOMENT.value == "dipole_moment"

    def test_property_count(self):
        """Test that we have expected number of properties."""
        properties = list(QuantumProperty)
        assert len(properties) >= 20  # Should have many quantum properties


class TestQMLPrediction:
    """Test QMLPrediction class."""

    def test_initialization(self):
        """Test QMLPrediction initialization."""
        properties = {
            QuantumProperty.HOMO_ENERGY: -8.5,
            QuantumProperty.LUMO_ENERGY: -2.3,
        }

        prediction = QMLPrediction(
            properties=properties,
            uncertainty={QuantumProperty.HOMO_ENERGY: 0.1},
            confidence=0.95,
            model_name="TestModel",
        )

        assert prediction.properties == properties
        assert prediction.uncertainty[QuantumProperty.HOMO_ENERGY] == 0.1
        assert prediction.confidence == 0.95
        assert prediction.model_name == "TestModel"

    def test_get_property(self):
        """Test property getter."""
        properties = {QuantumProperty.HOMO_ENERGY: -8.5}
        prediction = QMLPrediction(properties=properties)

        assert prediction.get_property(QuantumProperty.HOMO_ENERGY) == -8.5

        with pytest.raises(ValueError):
            prediction.get_property(QuantumProperty.LUMO_ENERGY)

    def test_has_property(self):
        """Test property existence check."""
        properties = {QuantumProperty.HOMO_ENERGY: -8.5}
        prediction = QMLPrediction(properties=properties)

        assert prediction.has_property(QuantumProperty.HOMO_ENERGY)
        assert not prediction.has_property(QuantumProperty.LUMO_ENERGY)

    def test_array_properties(self):
        """Test handling of array properties."""
        charges = np.array([0.1, -0.2, 0.15, -0.05])
        properties = {QuantumProperty.PARTIAL_CHARGES: charges}
        prediction = QMLPrediction(properties=properties)

        retrieved_charges = prediction.get_property(QuantumProperty.PARTIAL_CHARGES)
        np.testing.assert_array_equal(retrieved_charges, charges)


class MockQMLModel(BaseQMLModel):
    """Mock QML model for testing."""

    def load_model(self):
        self._model = Mock()
        self._is_loaded = True

    def predict(self, molecules, properties=None, **kwargs):
        if isinstance(molecules, list):
            return [
                QMLPrediction(
                    properties={QuantumProperty.HOMO_ENERGY: -8.0},
                    model_name="MockModel",
                )
                for _ in molecules
            ]
        else:
            return QMLPrediction(
                properties={QuantumProperty.HOMO_ENERGY: -8.0}, model_name="MockModel"
            )

    def get_supported_properties(self):
        return [QuantumProperty.HOMO_ENERGY, QuantumProperty.LUMO_ENERGY]

    def validate_molecule(self, molecule):
        return True


class TestBaseQMLModel:
    """Test BaseQMLModel abstract class."""

    def test_initialization(self):
        """Test model initialization."""
        model = MockQMLModel(device="cpu", precision="float32", cache_predictions=True)

        assert model.device == "cpu"
        assert model.precision == "float32"
        assert model.cache_predictions == True
        assert not model._is_loaded

    def test_device_selection(self):
        """Test automatic device selection."""
        model = MockQMLModel(device="auto")

        # Device should be either 'cpu' or 'cuda'
        assert model.device in ["cpu", "cuda"]

    def test_model_loading(self):
        """Test model loading."""
        model = MockQMLModel()
        assert not model._is_loaded

        model.load_model()
        assert model._is_loaded

    def test_single_prediction(self):
        """Test single molecule prediction."""
        model = MockQMLModel()
        model.load_model()

        prediction = model.predict("CCO")  # Ethanol

        assert isinstance(prediction, QMLPrediction)
        assert prediction.has_property(QuantumProperty.HOMO_ENERGY)
        assert prediction.model_name == "MockModel"

    def test_batch_prediction(self):
        """Test batch prediction."""
        model = MockQMLModel()
        model.load_model()

        molecules = ["CCO", "CC", "CCC"]
        predictions = model.predict(molecules)

        assert isinstance(predictions, list)
        assert len(predictions) == 3
        assert all(isinstance(p, QMLPrediction) for p in predictions)

    def test_batch_predict_method(self):
        """Test dedicated batch prediction method."""
        model = MockQMLModel()
        model.load_model()

        molecules = ["CCO", "CC", "CCC", "CCCC", "CCCCC"]
        predictions = model.batch_predict(molecules, batch_size=2)

        assert len(predictions) == 5
        assert all(isinstance(p, QMLPrediction) for p in predictions)

    def test_get_model_info(self):
        """Test model information retrieval."""
        model = MockQMLModel(device="cpu", precision="float64")
        model.load_model()

        info = model.get_model_info()

        assert info["model_type"] == "MockQMLModel"
        assert info["device"] == "cpu"
        assert info["precision"] == "float64"
        assert info["is_loaded"] == True
        assert len(info["supported_properties"]) == 2


class TestQMLModelRegistry:
    """Test QMLModelRegistry class."""

    def test_model_registration(self):
        """Test model registration."""
        registry = QMLModelRegistry()
        model = MockQMLModel()

        registry.register_model("test_model", model)

        assert "test_model" in registry.list_models()
        retrieved_model = registry.get_model("test_model")
        assert retrieved_model is model

    def test_default_model(self):
        """Test default model handling."""
        registry = QMLModelRegistry()
        model1 = MockQMLModel()
        model2 = MockQMLModel()

        registry.register_model("model1", model1, set_as_default=True)
        registry.register_model("model2", model2)

        # Should get model1 as default
        default_model = registry.get_model()
        assert default_model is model1

    def test_model_not_found(self):
        """Test handling of non-existent models."""
        registry = QMLModelRegistry()

        with pytest.raises(QMLModelError):
            registry.get_model("nonexistent")

    def test_no_default_model(self):
        """Test error when no default model is set."""
        registry = QMLModelRegistry()

        with pytest.raises(QMLModelError):
            registry.get_model()

    def test_predict_with_best_model(self):
        """Test prediction with best available model."""
        registry = QMLModelRegistry()

        # Model with limited properties
        model1 = MockQMLModel()
        registry.register_model("limited", model1)

        # Model with more properties (mock)
        model2 = MockQMLModel()
        model2.get_supported_properties = lambda: [
            QuantumProperty.HOMO_ENERGY,
            QuantumProperty.LUMO_ENERGY,
            QuantumProperty.DIPOLE_MOMENT,
        ]
        registry.register_model("comprehensive", model2)

        # Should use model2 for multiple properties
        properties = [QuantumProperty.HOMO_ENERGY, QuantumProperty.DIPOLE_MOMENT]
        prediction = registry.predict_with_best_model("CCO", properties)

        assert isinstance(prediction, QMLPrediction)

    def test_list_models(self):
        """Test listing registered models."""
        registry = QMLModelRegistry()

        assert registry.list_models() == []

        registry.register_model("model1", MockQMLModel())
        registry.register_model("model2", MockQMLModel())

        models = registry.list_models()
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models


class TestGlobalRegistry:
    """Test global QML registry."""

    def test_global_registry_exists(self):
        """Test that global registry is available."""
        assert qml_registry is not None
        assert isinstance(qml_registry, QMLModelRegistry)

    def test_global_registry_operations(self):
        """Test operations on global registry."""
        # Clear any existing models
        initial_models = qml_registry.list_models()

        # Register a test model
        model = MockQMLModel()
        qml_registry.register_model("test_global", model)

        # Verify registration
        assert "test_global" in qml_registry.list_models()
        retrieved = qml_registry.get_model("test_global")
        assert retrieved is model


class TestQMLModelError:
    """Test QMLModelError exception."""

    def test_error_creation(self):
        """Test error creation and inheritance."""
        error = QMLModelError("Test error message")

        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_error_raising(self):
        """Test that error can be raised and caught."""
        with pytest.raises(QMLModelError) as exc_info:
            raise QMLModelError("Custom error")

        assert str(exc_info.value) == "Custom error"

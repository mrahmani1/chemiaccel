"""
Tests for mixture core module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.core.exceptions import ChemiAccelError
from chemi_accel.mixture.core import ComponentData, MixtureCalculator, MixtureData


class TestComponentData:
    """Tests for ComponentData class."""

    def test_component_data_creation(self):
        """Test basic component data creation."""
        comp = ComponentData(smiles="CCO", mol_fraction=0.5, molecular_weight=46.07)

        assert comp.smiles == "CCO"
        assert comp.mol_fraction == 0.5
        assert comp.molecular_weight == 46.07

    def test_molecular_weight_calculation(self):
        """Test automatic molecular weight calculation."""
        comp = ComponentData(smiles="CCO", mol_fraction=0.5)

        # Should calculate MW automatically
        assert comp.molecular_weight is not None
        assert comp.molecular_weight > 0

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        comp = ComponentData(smiles="INVALID", mol_fraction=0.5)
        # Should not crash, but MW might be None
        assert comp.smiles == "INVALID"


class TestMixtureData:
    """Tests for MixtureData class."""

    def test_mixture_data_creation(self):
        """Test basic mixture data creation."""
        components = [
            ComponentData(smiles="CCO", mol_fraction=0.6),
            ComponentData(smiles="O", mol_fraction=0.4),
        ]

        mixture = MixtureData(
            components=components, temperature=298.15, pressure=101325.0
        )

        assert mixture.n_components == 2
        assert len(mixture.smiles_list) == 2
        assert mixture.temperature == 298.15
        assert mixture.pressure == 101325.0

    def test_composition_normalization(self):
        """Test automatic composition normalization."""
        components = [
            ComponentData(smiles="CCO", mol_fraction=0.6),
            ComponentData(smiles="O", mol_fraction=0.6),  # Sum > 1
        ]

        mixture = MixtureData(components=components)

        # Should be normalized to sum to 1
        total_moles = sum(comp.mol_fraction for comp in mixture.components)
        assert abs(total_moles - 1.0) < 1e-6

    def test_composition_vector_property(self):
        """Test composition vector property."""
        components = [
            ComponentData(smiles="CCO", mol_fraction=0.3),
            ComponentData(smiles="O", mol_fraction=0.7),
        ]

        mixture = MixtureData(components=components)
        comp_vector = mixture.composition_vector

        assert isinstance(comp_vector, np.ndarray)
        assert len(comp_vector) == 2
        assert abs(np.sum(comp_vector) - 1.0) < 1e-6

    def test_empty_mixture_error(self):
        """Test error for empty mixture."""
        with pytest.raises(
            ValueError, match="Mixture must contain at least one component"
        ):
            MixtureData(components=[])


class TestMixtureCalculator:
    """Tests for MixtureCalculator class."""

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = MixtureCalculator()

        assert calc is not None
        assert calc.default_temperature == 298.15
        assert calc.default_pressure == 101325.0

    def test_create_mixture_from_smiles_list(self):
        """Test mixture creation from SMILES list."""
        calc = MixtureCalculator()

        components = ["CCO", "O"]
        mixture = calc.create_mixture(components)

        assert mixture.n_components == 2
        assert mixture.components[0].smiles == "CCO"
        assert mixture.components[1].smiles == "O"

        # Should have equal fractions
        for comp in mixture.components:
            assert abs(comp.mol_fraction - 0.5) < 1e-6

    def test_create_mixture_with_fractions(self):
        """Test mixture creation with specified fractions."""
        calc = MixtureCalculator()

        components = [("CCO", 0.3), ("O", 0.7)]
        mixture = calc.create_mixture(components)

        assert mixture.components[0].mol_fraction == 0.3
        assert mixture.components[1].mol_fraction == 0.7

    def test_calculate_mixture_properties(self):
        """Test basic mixture property calculations."""
        calc = MixtureCalculator()

        components = [("CCO", 0.5), ("O", 0.5)]
        mixture = calc.create_mixture(components)

        properties = calc.calculate_mixture_properties(mixture)

        assert isinstance(properties, dict)
        assert "molecular_weight_avg" in properties
        assert properties["molecular_weight_avg"] > 0

    @patch("chemi_accel.mixture.phase_behavior.PhaseBehaviorPredictor")
    def test_phase_predictor_property(self, mock_predictor_class):
        """Test phase predictor property access."""
        calc = MixtureCalculator()

        # First access should create the predictor
        predictor = calc.phase_predictor
        assert predictor is not None
        mock_predictor_class.assert_called_once()

        # Second access should return the same instance
        predictor2 = calc.phase_predictor
        assert predictor is predictor2

    @patch("chemi_accel.mixture.interactions.InteractionModeler")
    def test_interaction_modeler_property(self, mock_modeler_class):
        """Test interaction modeler property access."""
        calc = MixtureCalculator()

        modeler = calc.interaction_modeler
        assert modeler is not None
        mock_modeler_class.assert_called_once()

    @patch("chemi_accel.mixture.inverse_design.MixtureInverseDesigner")
    def test_inverse_designer_property(self, mock_designer_class):
        """Test inverse designer property access."""
        calc = MixtureCalculator()

        designer = calc.inverse_designer
        assert designer is not None
        mock_designer_class.assert_called_once()

    def test_invalid_component_format(self):
        """Test error handling for invalid component format."""
        calc = MixtureCalculator()

        with pytest.raises(ValueError, match="Invalid component format"):
            calc.create_mixture([("CCO", 0.5, "extra")])  # Too many elements

    def test_backend_setup(self):
        """Test backend setup options."""
        # Test with string backend
        calc1 = MixtureCalculator(backend="cpu")
        assert calc1.backend is not None

        # Test with use_gpu=False
        calc2 = MixtureCalculator(use_gpu=False)
        assert calc2.backend is not None

    def test_custom_temperature_pressure(self):
        """Test custom temperature and pressure settings."""
        calc = MixtureCalculator(temperature=350.0, pressure=200000.0)

        assert calc.default_temperature == 350.0
        assert calc.default_pressure == 200000.0

        # Should use these defaults when creating mixtures
        mixture = calc.create_mixture(["CCO", "O"])
        assert mixture.temperature == 350.0
        assert mixture.pressure == 200000.0

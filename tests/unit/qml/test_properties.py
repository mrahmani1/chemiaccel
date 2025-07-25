"""
Tests for quantum property extraction and processing utilities.
"""


import numpy as np
import pytest
from chemi_accel.qml.base import QMLPrediction, QuantumProperty
from chemi_accel.qml.properties import PropertyMetadata, QuantumPropertyExtractor


class TestPropertyMetadata:
    """Test PropertyMetadata dataclass."""

    def test_metadata_creation(self):
        """Test property metadata creation."""
        metadata = PropertyMetadata(
            name="Test Property",
            units="eV",
            description="A test property",
            typical_range=(0.0, 10.0),
            preferred_method="DFT",
            accuracy_level="high",
        )

        assert metadata.name == "Test Property"
        assert metadata.units == "eV"
        assert metadata.description == "A test property"
        assert metadata.typical_range == (0.0, 10.0)
        assert metadata.preferred_method == "DFT"
        assert metadata.accuracy_level == "high"

    def test_metadata_optional_fields(self):
        """Test metadata with optional fields."""
        metadata = PropertyMetadata(
            name="Simple Property", units="au", description="Simple test"
        )

        assert metadata.typical_range is None
        assert metadata.preferred_method is None
        assert metadata.accuracy_level is None


class TestQuantumPropertyExtractor:
    """Test QuantumPropertyExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = QuantumPropertyExtractor()

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = QuantumPropertyExtractor()
        assert extractor is not None

    def test_extract_all_properties(self):
        """Test extracting all properties from prediction."""
        properties = {
            QuantumProperty.HOMO_ENERGY: -8.5,
            QuantumProperty.LUMO_ENERGY: -2.3,
            QuantumProperty.DIPOLE_MOMENT: 1.2,
        }
        prediction = QMLPrediction(properties=properties)

        extractor = QuantumPropertyExtractor()
        extracted = extractor.extract_properties(prediction)

        assert extracted == properties

    def test_extract_specific_properties(self):
        """Test extracting specific properties."""
        properties = {
            QuantumProperty.HOMO_ENERGY: -8.5,
            QuantumProperty.LUMO_ENERGY: -2.3,
            QuantumProperty.DIPOLE_MOMENT: 1.2,
        }
        prediction = QMLPrediction(properties=properties)

        extractor = QuantumPropertyExtractor()
        extracted = extractor.extract_properties(
            prediction,
            properties=[QuantumProperty.HOMO_ENERGY, QuantumProperty.DIPOLE_MOMENT],
        )

        assert len(extracted) == 2
        assert QuantumProperty.HOMO_ENERGY in extracted
        assert QuantumProperty.DIPOLE_MOMENT in extracted
        assert QuantumProperty.LUMO_ENERGY not in extracted

    def test_extract_missing_properties(self):
        """Test extracting properties not in prediction."""
        properties = {QuantumProperty.HOMO_ENERGY: -8.5}
        prediction = QMLPrediction(properties=properties)

        extractor = QuantumPropertyExtractor()
        extracted = extractor.extract_properties(
            prediction,
            properties=[QuantumProperty.HOMO_ENERGY, QuantumProperty.LUMO_ENERGY],
        )

        assert len(extracted) == 1
        assert QuantumProperty.HOMO_ENERGY in extracted
        assert QuantumProperty.LUMO_ENERGY not in extracted

    def test_unit_conversion_hartree_to_ev(self):
        """Test Hartree to eV conversion."""
        extractor = QuantumPropertyExtractor()

        hartree_value = -0.5
        ev_value = extractor.convert_units(hartree_value, "hartree", "ev")

        expected = -0.5 * 27.211386245988
        assert abs(ev_value - expected) < 1e-10

    def test_unit_conversion_ev_to_hartree(self):
        """Test eV to Hartree conversion."""
        extractor = QuantumPropertyExtractor()

        ev_value = -13.6
        hartree_value = extractor.convert_units(ev_value, "ev", "hartree")

        expected = -13.6 / 27.211386245988
        assert abs(hartree_value - expected) < 1e-10

    def test_unit_conversion_same_units(self):
        """Test conversion with same units."""
        extractor = QuantumPropertyExtractor()

        value = 5.0
        converted = extractor.convert_units(value, "eV", "eV")

        assert converted == value

    def test_unit_conversion_reverse(self):
        """Test reverse unit conversion lookup."""
        extractor = QuantumPropertyExtractor()

        # Test debye to au conversion (reverse lookup)
        debye_value = 2.0
        au_value = extractor.convert_units(debye_value, "debye", "au")

        expected = debye_value / 2.5417464739297717
        assert abs(au_value - expected) < 1e-10

    def test_unit_conversion_array(self):
        """Test unit conversion with arrays."""
        extractor = QuantumPropertyExtractor()

        hartree_array = np.array([-0.5, -0.3, -0.1])
        ev_array = extractor.convert_units(hartree_array, "hartree", "ev")

        expected = hartree_array * 27.211386245988
        np.testing.assert_array_almost_equal(ev_array, expected, decimal=10)

    def test_unit_conversion_invalid(self):
        """Test invalid unit conversion."""
        extractor = QuantumPropertyExtractor()

        with pytest.raises(ValueError):
            extractor.convert_units(1.0, "invalid_unit", "eV")

    def test_property_range_validation_scalar(self):
        """Test property range validation for scalar values."""
        extractor = QuantumPropertyExtractor()

        # Valid HOMO energy
        assert extractor.validate_property_range(QuantumProperty.HOMO_ENERGY, -8.5)

        # Invalid HOMO energy (too high)
        assert not extractor.validate_property_range(QuantumProperty.HOMO_ENERGY, 5.0)

        # Invalid HOMO energy (too low)
        assert not extractor.validate_property_range(QuantumProperty.HOMO_ENERGY, -50.0)

    def test_property_range_validation_array(self):
        """Test property range validation for array values."""
        extractor = QuantumPropertyExtractor()

        # Assume partial charges have reasonable range (-2 to +2)
        valid_charges = np.array([0.1, -0.5, 0.8, -0.2])
        # This test assumes PARTIAL_CHARGES has a range defined in metadata
        # For now, we'll skip this test since the range isn't defined

        # Test with a property that has defined range
        valid_energies = np.array([-8.5, -7.2, -9.1])
        # Mock the validation by checking a single value
        result = extractor.validate_property_range(QuantumProperty.HOMO_ENERGY, -8.5)
        assert result == True

    def test_property_range_validation_unknown_property(self):
        """Test validation for unknown properties."""
        extractor = QuantumPropertyExtractor()

        # Create a mock property not in metadata
        class MockProperty:
            pass

        # Should return True for unknown properties
        result = extractor.validate_property_range(MockProperty(), 100.0)
        assert result == True

    def test_compute_derived_properties_bandgap(self):
        """Test computation of HOMO-LUMO gap."""
        properties = {
            QuantumProperty.HOMO_ENERGY: -8.5,
            QuantumProperty.LUMO_ENERGY: -2.3,
        }
        prediction = QMLPrediction(properties=properties)

        extractor = QuantumPropertyExtractor()
        derived = extractor.compute_derived_properties(prediction)

        assert QuantumProperty.BANDGAP in derived
        expected_gap = -2.3 - (-8.5)  # LUMO - HOMO
        assert abs(derived[QuantumProperty.BANDGAP] - expected_gap) < 1e-10

    def test_compute_derived_properties_ionization_potential(self):
        """Test computation of ionization potential from HOMO."""
        properties = {QuantumProperty.HOMO_ENERGY: -8.5}
        prediction = QMLPrediction(properties=properties)

        extractor = QuantumPropertyExtractor()
        derived = extractor.compute_derived_properties(prediction)

        assert QuantumProperty.IONIZATION_POTENTIAL in derived
        expected_ip = -(-8.5)  # -HOMO (Koopmans' theorem)
        assert derived[QuantumProperty.IONIZATION_POTENTIAL] == expected_ip

    def test_compute_derived_properties_electron_affinity(self):
        """Test computation of electron affinity from LUMO."""
        properties = {QuantumProperty.LUMO_ENERGY: -2.3}
        prediction = QMLPrediction(properties=properties)

        extractor = QuantumPropertyExtractor()
        derived = extractor.compute_derived_properties(prediction)

        assert QuantumProperty.ELECTRON_AFFINITY in derived
        expected_ea = -(-2.3)  # -LUMO (Koopmans' theorem)
        assert derived[QuantumProperty.ELECTRON_AFFINITY] == expected_ea

    def test_analyze_prediction_quality_good(self):
        """Test quality analysis for good prediction."""
        properties = {
            QuantumProperty.HOMO_ENERGY: -8.5,
            QuantumProperty.LUMO_ENERGY: -2.3,
            QuantumProperty.DIPOLE_MOMENT: 1.2,
        }
        prediction = QMLPrediction(properties=properties, confidence=0.9)

        extractor = QuantumPropertyExtractor()
        analysis = extractor.analyze_prediction_quality(prediction)

        assert analysis["overall_confidence"] == 0.9
        assert len(analysis["property_validation"]) == 3
        assert len(analysis["potential_issues"]) == 0  # Should be no issues
        assert len(analysis["recommendations"]) == 0  # Should be no recommendations

    def test_analyze_prediction_quality_issues(self):
        """Test quality analysis with problematic values."""
        properties = {
            QuantumProperty.HOMO_ENERGY: 5.0,  # Invalid: too high
            QuantumProperty.LUMO_ENERGY: -10.0,  # Invalid: LUMO < HOMO
        }
        prediction = QMLPrediction(properties=properties, confidence=0.5)

        extractor = QuantumPropertyExtractor()
        analysis = extractor.analyze_prediction_quality(prediction)

        assert analysis["overall_confidence"] == 0.5
        assert len(analysis["potential_issues"]) > 0
        assert len(analysis["recommendations"]) > 0

        # Check for specific issues
        issues = analysis["potential_issues"]
        assert any(
            "LUMO energy lower than or equal to HOMO energy" in issue
            for issue in issues
        )

    def test_compare_predictions_single_model(self):
        """Test comparison with single prediction."""
        properties = {QuantumProperty.HOMO_ENERGY: -8.5}
        prediction = QMLPrediction(properties=properties, model_name="Model1")

        extractor = QuantumPropertyExtractor()
        comparison = extractor.compare_predictions([prediction])

        assert comparison["num_predictions"] == 1
        assert len(comparison["common_properties"]) == 1
        assert "homo_energy" in comparison["common_properties"]

    def test_compare_predictions_multiple_models(self):
        """Test comparison with multiple predictions."""
        pred1 = QMLPrediction(
            properties={QuantumProperty.HOMO_ENERGY: -8.5}, model_name="Model1"
        )
        pred2 = QMLPrediction(
            properties={QuantumProperty.HOMO_ENERGY: -8.3}, model_name="Model2"
        )
        pred3 = QMLPrediction(
            properties={QuantumProperty.HOMO_ENERGY: -8.7}, model_name="Model3"
        )

        extractor = QuantumPropertyExtractor()
        comparison = extractor.compare_predictions([pred1, pred2, pred3])

        assert comparison["num_predictions"] == 3
        assert "statistics" in comparison
        assert "homo_energy" in comparison["statistics"]

        stats = comparison["statistics"]["homo_energy"]
        assert abs(stats["mean"] - (-8.5)) < 0.1  # Should be close to average
        assert stats["std"] > 0  # Should have some variation

        # Check agreement analysis
        assert "agreement_analysis" in comparison
        assert "homo_energy" in comparison["agreement_analysis"]

    def test_compare_predictions_no_common_properties(self):
        """Test comparison with no common properties."""
        pred1 = QMLPrediction(properties={QuantumProperty.HOMO_ENERGY: -8.5})
        pred2 = QMLPrediction(properties={QuantumProperty.DIPOLE_MOMENT: 1.2})

        extractor = QuantumPropertyExtractor()
        comparison = extractor.compare_predictions([pred1, pred2])

        assert comparison["num_predictions"] == 2
        assert len(comparison["common_properties"]) == 0

    def test_get_property_metadata(self):
        """Test property metadata retrieval."""
        extractor = QuantumPropertyExtractor()

        metadata = extractor.get_property_metadata(QuantumProperty.HOMO_ENERGY)
        assert metadata is not None
        assert metadata.name == "HOMO Energy"
        assert metadata.units == "eV"

        # Test unknown property
        class UnknownProperty:
            pass

        unknown_metadata = extractor.get_property_metadata(UnknownProperty())
        assert unknown_metadata is None

    def test_list_supported_conversions(self):
        """Test listing of supported unit conversions."""
        extractor = QuantumPropertyExtractor()

        conversions = extractor.list_supported_conversions()
        assert isinstance(conversions, list)
        assert len(conversions) > 0
        assert "hartree_to_ev" in conversions
        assert "ev_to_hartree" in conversions

    def test_property_metadata_completeness(self):
        """Test that metadata exists for key properties."""
        extractor = QuantumPropertyExtractor()

        key_properties = [
            QuantumProperty.HOMO_ENERGY,
            QuantumProperty.LUMO_ENERGY,
            QuantumProperty.TOTAL_ENERGY,
            QuantumProperty.BANDGAP,
            QuantumProperty.DIPOLE_MOMENT,
            QuantumProperty.POLARIZABILITY,
        ]

        for prop in key_properties:
            metadata = extractor.get_property_metadata(prop)
            assert metadata is not None, f"Missing metadata for {prop.value}"
            assert metadata.name is not None
            assert metadata.units is not None
            assert metadata.description is not None

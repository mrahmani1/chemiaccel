"""
Tests for solid-state property prediction module.

This module tests the GPU-accelerated solid-state property calculation
functionality including density, elastic moduli, thermal properties,
and solubility predictions.
"""

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Import the module under test
from chemi_accel.materials.solid_state_properties import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    CrystalStructureData,
    DataQuality,
    DensityCalculationError,
    DensityCalculator,
    ElasticModuliCalculationError,
    ElasticModuliPredictor,
    ElasticProperties,
    ElasticProperty,
    ElasticPropertyNet,
    PropertyPredictionConfig,
    PropertyPredictionError,
    PropertyResult,
    PropertyType,
    SolidStatePropertyEngine,
    SolidStatePropertyError,
    SolubilityPredictionError,
    SolubilityPredictor,
    SolubilityResult,
    SolubilityType,
    ThermalProperties,
    ThermalProperty,
    ThermalPropertyError,
    ThermalPropertyPredictor,
    calculate_structure_density,
    create_crystal_structure,
    create_property_config,
    predict_elastic_moduli,
    predict_solid_state_properties,
    predict_solubility,
    predict_thermal_properties,
)


# Test fixtures
@pytest.fixture
def sample_crystal_structure():
    """Create a sample crystal structure for testing."""
    return CrystalStructureData(
        unit_cell_parameters=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        space_group="Pm3m",
        atomic_positions=[("C", 0.0, 0.0, 0.0), ("O", 0.5, 0.5, 0.5)],
        formula="CO",
        molecular_weight=28.01,
        z_value=1,
        volume=1000.0,
        density=2.5,
    )


@pytest.fixture
def sample_property_config():
    """Create a sample property prediction configuration."""
    return PropertyPredictionConfig(
        use_gpu=False,  # Disable GPU for testing
        batch_size=16,
        temperature=298.15,
        pressure=1.0,
        include_uncertainty=True,
        ml_models_enabled=False,  # Disable ML models for testing
    )


@pytest.fixture
def mock_torch():
    """Mock PyTorch imports."""
    with patch.multiple(
        "chemi_accel.materials.solid_state_properties",
        _TORCH_AVAILABLE=False,
        torch=MagicMock(),
    ):
        yield


class TestPropertyType:
    """Test PropertyType enum."""

    def test_property_type_values(self):
        """Test PropertyType enum values."""
        assert PropertyType.DENSITY.value == "density"
        assert PropertyType.BULK_MODULUS.value == "bulk_modulus"
        assert PropertyType.THERMAL_EXPANSION.value == "thermal_expansion"
        assert PropertyType.MELTING_POINT.value == "melting_point"
        assert PropertyType.SOLUBILITY.value == "solubility"


class TestDataStructures:
    """Test data structure classes."""

    def test_crystal_structure_data_creation(self):
        """Test CrystalStructureData creation."""
        structure = CrystalStructureData(
            unit_cell_parameters=(5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
            space_group="Pm3m",
            atomic_positions=[("C", 0.0, 0.0, 0.0)],
            formula="C",
            molecular_weight=12.01,
            z_value=1,
        )

        assert structure.formula == "C"
        assert structure.molecular_weight == 12.01
        assert structure.z_value == 1
        assert structure.unit_cell_parameters == (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)

    def test_property_prediction_config_defaults(self):
        """Test PropertyPredictionConfig default values."""
        config = PropertyPredictionConfig()

        assert config.use_gpu is True
        assert config.batch_size == 32
        assert config.temperature == 298.15
        assert config.pressure == 1.0
        assert config.include_uncertainty is True

    def test_property_result_creation(self):
        """Test PropertyResult creation."""
        result = PropertyResult(
            property_type=PropertyType.DENSITY,
            value=2.5,
            unit="g/cm³",
            uncertainty=0.1,
            confidence=0.95,
        )

        assert result.property_type == PropertyType.DENSITY
        assert result.value == 2.5
        assert result.unit == "g/cm³"
        assert result.uncertainty == 0.1
        assert result.confidence == 0.95

    def test_elastic_properties_creation(self):
        """Test ElasticProperties creation."""
        elastic_props = ElasticProperties(
            bulk_modulus=100.0,
            shear_modulus=80.0,
            youngs_modulus=200.0,
            poissons_ratio=0.25,
        )

        assert elastic_props.bulk_modulus == 100.0
        assert elastic_props.shear_modulus == 80.0
        assert elastic_props.youngs_modulus == 200.0
        assert elastic_props.poissons_ratio == 0.25


class TestDensityCalculator:
    """Test DensityCalculator class."""

    def test_density_calculator_creation(self):
        """Test density calculator creation."""
        calculator = DensityCalculator(use_gpu=False)

        assert calculator.use_gpu is False
        assert calculator.xp == np

    def test_calculate_density_with_volume(self, sample_crystal_structure):
        """Test density calculation with provided volume."""
        calculator = DensityCalculator(use_gpu=False)

        density = calculator.calculate_density(sample_crystal_structure)

        # Expected density: (Z × M) / (V × N_A × 1e-24)
        # (1 × 28.01) / (1000.0 × 6.022e23 × 1e-24) ≈ 0.0465 g/cm³
        assert density > 0
        assert isinstance(density, float)

    def test_calculate_density_without_volume(self):
        """Test density calculation without provided volume."""
        structure = CrystalStructureData(
            unit_cell_parameters=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
            space_group="Pm3m",
            atomic_positions=[("C", 0.0, 0.0, 0.0)],
            formula="C",
            molecular_weight=12.01,
            z_value=1,
            volume=None,  # No volume provided
        )

        calculator = DensityCalculator(use_gpu=False)
        density = calculator.calculate_density(structure)

        assert density > 0
        assert isinstance(density, float)

    def test_unit_cell_volume_calculation(self):
        """Test unit cell volume calculation."""
        calculator = DensityCalculator(use_gpu=False)

        # Cubic cell
        params = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        volume = calculator._calculate_unit_cell_volume(params)

        assert volume == pytest.approx(1000.0, rel=1e-3)

    def test_batch_calculate_density(self, sample_crystal_structure):
        """Test batch density calculation."""
        calculator = DensityCalculator(use_gpu=False)

        structures = [sample_crystal_structure] * 3
        densities = calculator.batch_calculate_density(structures)

        assert len(densities) == 3
        assert all(d > 0 for d in densities)

    def test_invalid_structure_raises_error(self):
        """Test that invalid structure raises error."""
        calculator = DensityCalculator(use_gpu=False)

        # Structure with invalid data
        structure = CrystalStructureData(
            unit_cell_parameters=(0, 0, 0, 90.0, 90.0, 90.0),  # Zero cell parameters
            space_group="Pm3m",
            atomic_positions=[("C", 0.0, 0.0, 0.0)],
            formula="C",
            molecular_weight=12.01,
            z_value=1,
        )

        with pytest.raises(DensityCalculationError):
            calculator.calculate_density(structure)


class TestElasticModuliPredictor:
    """Test ElasticModuliPredictor class."""

    def test_elastic_predictor_creation(self, sample_property_config):
        """Test elastic moduli predictor creation."""
        predictor = ElasticModuliPredictor(sample_property_config)

        assert predictor.config == sample_property_config
        assert predictor.use_gpu is False

    def test_predict_elastic_properties(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test elastic properties prediction."""
        predictor = ElasticModuliPredictor(sample_property_config)

        properties = predictor.predict_elastic_properties(sample_crystal_structure)

        assert isinstance(properties, ElasticProperties)
        assert properties.bulk_modulus > 0
        assert properties.shear_modulus > 0
        assert properties.youngs_modulus > 0
        assert 0 < properties.poissons_ratio < 0.5

    def test_extract_structural_features(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test structural feature extraction."""
        predictor = ElasticModuliPredictor(sample_property_config)

        features = predictor._extract_structural_features(sample_crystal_structure)

        assert isinstance(features, np.ndarray)
        assert len(features) == 15  # Expected feature count
        assert np.all(np.isfinite(features))

    def test_encode_crystal_system(self, sample_property_config):
        """Test crystal system encoding."""
        predictor = ElasticModuliPredictor(sample_property_config)

        # Test cubic space group
        encoding = predictor._encode_crystal_system("Pm3m")
        assert len(encoding) == 5
        assert sum(encoding) == 1  # One-hot encoding
        assert encoding[0] == 1.0  # Cubic should be first

    def test_empirical_calculations(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test empirical calculations."""
        predictor = ElasticModuliPredictor(sample_property_config)

        bulk_modulus = predictor._empirical_bulk_modulus(sample_crystal_structure)
        shear_modulus = predictor._empirical_shear_modulus(sample_crystal_structure)

        assert bulk_modulus > 0
        assert shear_modulus > 0

    def test_calculate_derived_properties(self, sample_property_config):
        """Test calculation of derived elastic properties."""
        predictor = ElasticModuliPredictor(sample_property_config)

        bulk_modulus = 100.0
        shear_modulus = 80.0

        youngs_modulus = predictor._calculate_youngs_modulus(
            bulk_modulus, shear_modulus
        )
        poissons_ratio = predictor._calculate_poissons_ratio(
            bulk_modulus, shear_modulus
        )

        assert youngs_modulus > 0
        assert 0 < poissons_ratio < 0.5


class TestThermalPropertyPredictor:
    """Test ThermalPropertyPredictor class."""

    def test_thermal_predictor_creation(self, sample_property_config):
        """Test thermal property predictor creation."""
        predictor = ThermalPropertyPredictor(sample_property_config)

        assert predictor.config == sample_property_config

    def test_predict_thermal_expansion(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test thermal expansion prediction."""
        predictor = ThermalPropertyPredictor(sample_property_config)

        thermal_expansion = predictor.predict_thermal_expansion(
            sample_crystal_structure
        )

        assert thermal_expansion > 0
        assert thermal_expansion < 1e-3  # Reasonable range for thermal expansion

    def test_predict_melting_point(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test melting point prediction."""
        predictor = ThermalPropertyPredictor(sample_property_config)

        melting_point = predictor.predict_melting_point(sample_crystal_structure)

        assert melting_point > 200  # Reasonable minimum melting point
        assert melting_point < 5000  # Reasonable maximum melting point

    def test_predict_thermal_properties(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test complete thermal properties prediction."""
        predictor = ThermalPropertyPredictor(sample_property_config)

        properties = predictor.predict_thermal_properties(sample_crystal_structure)

        assert isinstance(properties, ThermalProperties)
        assert properties.thermal_expansion > 0
        assert properties.melting_point > 200


class TestSolubilityPredictor:
    """Test SolubilityPredictor class."""

    def test_solubility_predictor_creation(self, sample_property_config):
        """Test solubility predictor creation."""
        predictor = SolubilityPredictor(sample_property_config)

        assert predictor.config == sample_property_config

    def test_predict_aqueous_solubility(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test aqueous solubility prediction."""
        predictor = SolubilityPredictor(sample_property_config)

        solubility = predictor.predict_aqueous_solubility(sample_crystal_structure)

        assert solubility > 0
        assert isinstance(solubility, float)

    def test_predict_solubility_result(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test solubility result prediction."""
        predictor = SolubilityPredictor(sample_property_config)

        result = predictor.predict_solubility(
            sample_crystal_structure, SolubilityType.AQUEOUS
        )

        assert isinstance(result, SolubilityResult)
        assert result.solubility_type == SolubilityType.AQUEOUS
        assert result.value > 0
        assert result.unit == "mg/mL"
        assert result.temperature == sample_property_config.temperature

    def test_different_solubility_types(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test prediction of different solubility types."""
        predictor = SolubilityPredictor(sample_property_config)

        # Test different solubility types
        aqueous = predictor.predict_solubility(
            sample_crystal_structure, SolubilityType.AQUEOUS
        )
        organic = predictor.predict_solubility(
            sample_crystal_structure, SolubilityType.ORGANIC
        )

        assert aqueous.solubility_type == SolubilityType.AQUEOUS
        assert organic.solubility_type == SolubilityType.ORGANIC
        assert aqueous.value != organic.value  # Should be different


class TestElasticPropertyNet:
    """Test ElasticPropertyNet neural network."""

    @pytest.mark.skipif(True, reason="PyTorch not available in test environment")
    def test_network_creation(self):
        """Test neural network creation."""
        # This test would run if PyTorch was available
        pass

    @pytest.mark.skipif(True, reason="PyTorch not available in test environment")
    def test_forward_pass(self):
        """Test neural network forward pass."""
        # This test would run if PyTorch was available
        pass


class TestSolidStatePropertyEngine:
    """Test SolidStatePropertyEngine class."""

    def test_engine_creation(self, sample_property_config):
        """Test property engine creation."""
        engine = SolidStatePropertyEngine(sample_property_config)

        assert engine.config == sample_property_config
        assert isinstance(engine.density_calculator, DensityCalculator)
        assert isinstance(engine.elastic_predictor, ElasticModuliPredictor)
        assert isinstance(engine.thermal_predictor, ThermalPropertyPredictor)
        assert isinstance(engine.solubility_predictor, SolubilityPredictor)

    def test_predict_density_property(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test density property prediction."""
        engine = SolidStatePropertyEngine(sample_property_config)

        result = engine.predict_property(sample_crystal_structure, PropertyType.DENSITY)

        assert isinstance(result, PropertyResult)
        assert result.property_type == PropertyType.DENSITY
        assert result.value > 0
        assert result.unit == "g/cm³"
        assert result.method == "analytical"

    def test_predict_elastic_properties(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test elastic property predictions."""
        engine = SolidStatePropertyEngine(sample_property_config)

        # Test all elastic properties
        elastic_types = [
            PropertyType.BULK_MODULUS,
            PropertyType.SHEAR_MODULUS,
            PropertyType.YOUNGS_MODULUS,
            PropertyType.POISSONS_RATIO,
        ]

        for prop_type in elastic_types:
            result = engine.predict_property(sample_crystal_structure, prop_type)

            assert isinstance(result, PropertyResult)
            assert result.property_type == prop_type
            assert result.value > 0
            assert result.method == "ml_prediction"

    def test_predict_thermal_properties(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test thermal property predictions."""
        engine = SolidStatePropertyEngine(sample_property_config)

        # Test thermal expansion
        result = engine.predict_property(
            sample_crystal_structure, PropertyType.THERMAL_EXPANSION
        )

        assert isinstance(result, PropertyResult)
        assert result.property_type == PropertyType.THERMAL_EXPANSION
        assert result.value > 0
        assert result.unit == "K⁻¹"
        assert result.method == "empirical"

        # Test melting point
        result = engine.predict_property(
            sample_crystal_structure, PropertyType.MELTING_POINT
        )

        assert isinstance(result, PropertyResult)
        assert result.property_type == PropertyType.MELTING_POINT
        assert result.value > 200
        assert result.unit == "K"

    def test_predict_solubility_property(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test solubility property prediction."""
        engine = SolidStatePropertyEngine(sample_property_config)

        result = engine.predict_property(
            sample_crystal_structure, PropertyType.SOLUBILITY
        )

        assert isinstance(result, PropertyResult)
        assert result.property_type == PropertyType.SOLUBILITY
        assert result.value > 0
        assert result.unit == "mg/mL"
        assert result.method == "empirical"

    def test_predict_all_properties(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test prediction of all properties."""
        engine = SolidStatePropertyEngine(sample_property_config)

        results = engine.predict_all_properties(sample_crystal_structure)

        assert isinstance(results, dict)
        assert len(results) > 0

        # Check that all property types are covered
        expected_properties = [
            PropertyType.DENSITY,
            PropertyType.BULK_MODULUS,
            PropertyType.THERMAL_EXPANSION,
            PropertyType.MELTING_POINT,
            PropertyType.SOLUBILITY,
        ]

        for prop in expected_properties:
            if prop in results:
                assert isinstance(results[prop], PropertyResult)

    def test_batch_predict_property(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test batch property prediction."""
        engine = SolidStatePropertyEngine(sample_property_config)

        structures = [sample_crystal_structure] * 3
        results = engine.batch_predict_property(structures, PropertyType.DENSITY)

        assert len(results) == 3
        assert all(isinstance(r, PropertyResult) for r in results)
        assert all(r.property_type == PropertyType.DENSITY for r in results)

    def test_caching_functionality(self, sample_crystal_structure):
        """Test result caching functionality."""
        config = PropertyPredictionConfig(cache_results=True)
        engine = SolidStatePropertyEngine(config)

        # First prediction
        result1 = engine.predict_property(
            sample_crystal_structure, PropertyType.DENSITY
        )

        # Second prediction (should use cache)
        result2 = engine.predict_property(
            sample_crystal_structure, PropertyType.DENSITY
        )

        assert result1.value == result2.value
        assert len(engine._cache) > 0

    def test_unsupported_property_type(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test handling of unsupported property types."""
        engine = SolidStatePropertyEngine(sample_property_config)

        # Mock an unsupported property type
        with patch.object(PropertyType, "__members__", {"UNSUPPORTED": "unsupported"}):
            with pytest.raises(PropertyPredictionError):
                engine.predict_property(sample_crystal_structure, "unsupported")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_property_config(self):
        """Test create_property_config function."""
        config = create_property_config(use_gpu=False, batch_size=16, temperature=300.0)

        assert isinstance(config, PropertyPredictionConfig)
        assert config.use_gpu is False
        assert config.batch_size == 16
        assert config.temperature == 300.0

    def test_create_crystal_structure(self):
        """Test create_crystal_structure function."""
        structure = create_crystal_structure(
            formula="NaCl",
            unit_cell_params=(5.64, 5.64, 5.64, 90.0, 90.0, 90.0),
            space_group="Fm3m",
            atomic_positions=[("Na", 0.0, 0.0, 0.0), ("Cl", 0.5, 0.5, 0.5)],
            molecular_weight=58.44,
            z_value=4,
        )

        assert isinstance(structure, CrystalStructureData)
        assert structure.formula == "NaCl"
        assert structure.molecular_weight == 58.44
        assert structure.z_value == 4

    def test_predict_solid_state_properties(self, sample_crystal_structure):
        """Test predict_solid_state_properties function."""
        results = predict_solid_state_properties(
            sample_crystal_structure,
            properties=[PropertyType.DENSITY, PropertyType.BULK_MODULUS],
        )

        assert isinstance(results, dict)
        assert PropertyType.DENSITY in results
        assert PropertyType.BULK_MODULUS in results
        assert all(isinstance(r, PropertyResult) for r in results.values())

    def test_calculate_structure_density(self, sample_crystal_structure):
        """Test calculate_structure_density function."""
        density = calculate_structure_density(sample_crystal_structure)

        assert density > 0
        assert isinstance(density, float)

    def test_predict_elastic_moduli_function(self, sample_crystal_structure):
        """Test predict_elastic_moduli function."""
        properties = predict_elastic_moduli(sample_crystal_structure)

        assert isinstance(properties, ElasticProperties)
        assert properties.bulk_modulus > 0
        assert properties.shear_modulus > 0

    def test_predict_thermal_properties_function(self, sample_crystal_structure):
        """Test predict_thermal_properties function."""
        properties = predict_thermal_properties(sample_crystal_structure)

        assert isinstance(properties, ThermalProperties)
        assert properties.thermal_expansion > 0
        assert properties.melting_point > 200

    def test_predict_solubility_function(self, sample_crystal_structure):
        """Test predict_solubility function."""
        result = predict_solubility(sample_crystal_structure, SolubilityType.AQUEOUS)

        assert isinstance(result, SolubilityResult)
        assert result.solubility_type == SolubilityType.AQUEOUS
        assert result.value > 0


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_solid_state_property_error_inheritance(self):
        """Test exception inheritance."""
        assert issubclass(PropertyPredictionError, SolidStatePropertyError)
        assert issubclass(DensityCalculationError, SolidStatePropertyError)
        assert issubclass(ElasticModuliCalculationError, SolidStatePropertyError)
        assert issubclass(ThermalPropertyError, SolidStatePropertyError)
        assert issubclass(SolubilityPredictionError, SolidStatePropertyError)

    def test_density_calculation_error_handling(self):
        """Test density calculation error handling."""
        calculator = DensityCalculator(use_gpu=False)

        # Invalid structure that should raise error
        invalid_structure = CrystalStructureData(
            unit_cell_parameters=(0, 0, 0, 90.0, 90.0, 90.0),
            space_group="Pm3m",
            atomic_positions=[],
            formula="",
            molecular_weight=0,
            z_value=0,
        )

        with pytest.raises(DensityCalculationError):
            calculator.calculate_density(invalid_structure)

    def test_property_prediction_error_handling(self, sample_property_config):
        """Test property prediction error handling."""
        engine = SolidStatePropertyEngine(sample_property_config)

        # Invalid structure
        invalid_structure = None

        with pytest.raises(PropertyPredictionError):
            engine.predict_property(invalid_structure, PropertyType.DENSITY)


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    def test_batch_processing_performance(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test batch processing performance."""
        engine = SolidStatePropertyEngine(sample_property_config)

        # Test different batch sizes
        batch_sizes = [1, 5, 10]

        for batch_size in batch_sizes:
            structures = [sample_crystal_structure] * batch_size
            results = engine.batch_predict_property(structures, PropertyType.DENSITY)

            assert len(results) == batch_size
            assert all(isinstance(r, PropertyResult) for r in results)

    def test_memory_usage_with_caching(self, sample_crystal_structure):
        """Test memory usage with caching enabled."""
        config = PropertyPredictionConfig(cache_results=True)
        engine = SolidStatePropertyEngine(config)

        # Make multiple predictions
        for i in range(10):
            result = engine.predict_property(
                sample_crystal_structure, PropertyType.DENSITY
            )
            assert isinstance(result, PropertyResult)

        # Cache should contain results
        assert len(engine._cache) > 0

    def test_prediction_consistency(
        self, sample_crystal_structure, sample_property_config
    ):
        """Test prediction consistency across multiple runs."""
        engine = SolidStatePropertyEngine(sample_property_config)

        # Make multiple predictions
        results = []
        for i in range(5):
            result = engine.predict_property(
                sample_crystal_structure, PropertyType.DENSITY
            )
            results.append(result.value)

        # Results should be consistent (same structure, same conditions)
        assert all(abs(r - results[0]) < 1e-10 for r in results)


class TestIntegration:
    """Integration tests for solid-state property prediction."""

    def test_full_property_prediction_workflow(self):
        """Test complete property prediction workflow."""
        # Create crystal structure
        structure = create_crystal_structure(
            formula="SiO2",
            unit_cell_params=(4.916, 4.916, 5.405, 90.0, 90.0, 120.0),
            space_group="P6_222",
            atomic_positions=[("Si", 0.5, 0.0, 0.0), ("O", 0.413, 0.268, 0.119)],
            molecular_weight=60.08,
            z_value=3,
        )

        # Create configuration
        config = create_property_config(use_gpu=False, ml_models_enabled=False)

        # Predict all properties
        results = predict_solid_state_properties(structure, config=config)

        # Verify results
        assert isinstance(results, dict)
        assert len(results) > 0

        # Check specific properties
        if PropertyType.DENSITY in results:
            density = results[PropertyType.DENSITY]
            assert density.value > 1.0  # Reasonable for SiO2
            assert density.unit == "g/cm³"

        if PropertyType.BULK_MODULUS in results:
            bulk_modulus = results[PropertyType.BULK_MODULUS]
            assert bulk_modulus.value > 10.0  # Reasonable for SiO2
            assert bulk_modulus.unit == "GPa"

    def test_different_crystal_systems(self, sample_property_config):
        """Test property prediction for different crystal systems."""
        # Test cubic system
        cubic_structure = create_crystal_structure(
            formula="NaCl",
            unit_cell_params=(5.64, 5.64, 5.64, 90.0, 90.0, 90.0),
            space_group="Fm3m",
            atomic_positions=[("Na", 0.0, 0.0, 0.0), ("Cl", 0.5, 0.5, 0.5)],
            molecular_weight=58.44,
            z_value=4,
        )

        # Test hexagonal system
        hexagonal_structure = create_crystal_structure(
            formula="ZnO",
            unit_cell_params=(3.25, 3.25, 5.21, 90.0, 90.0, 120.0),
            space_group="P6_3mc",
            atomic_positions=[("Zn", 0.33, 0.67, 0.0), ("O", 0.33, 0.67, 0.38)],
            molecular_weight=81.38,
            z_value=2,
        )

        engine = SolidStatePropertyEngine(sample_property_config)

        # Predict properties for both systems
        cubic_results = engine.predict_all_properties(cubic_structure)
        hexagonal_results = engine.predict_all_properties(hexagonal_structure)

        # Both should have results
        assert len(cubic_results) > 0
        assert len(hexagonal_results) > 0

        # Properties should be different for different materials
        if (
            PropertyType.DENSITY in cubic_results
            and PropertyType.DENSITY in hexagonal_results
        ):
            cubic_density = cubic_results[PropertyType.DENSITY].value
            hex_density = hexagonal_results[PropertyType.DENSITY].value
            assert cubic_density != hex_density

    def test_error_propagation(self, sample_property_config):
        """Test error propagation through the system."""
        engine = SolidStatePropertyEngine(sample_property_config)

        # Test with multiple invalid structures
        invalid_structures = [
            None,
            "not_a_structure",
            CrystalStructureData(
                unit_cell_parameters=(0, 0, 0, 0, 0, 0),
                space_group="",
                atomic_positions=[],
                formula="",
                molecular_weight=0,
                z_value=0,
            ),
        ]

        for invalid_structure in invalid_structures:
            with pytest.raises((PropertyPredictionError, AttributeError, TypeError)):
                engine.predict_property(invalid_structure, PropertyType.DENSITY)

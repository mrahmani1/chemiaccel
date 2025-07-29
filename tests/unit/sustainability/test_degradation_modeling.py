"""
Tests for GPU-Accelerated Degradation Modeling.

This module provides comprehensive testing for the degradation modeling
framework including multi-scale models, environmental stressors, and
AI surrogate models.
"""

import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.sustainability.degradation_modeling import (  # Core models; Data structures; Utility functions; Exceptions
    AtomisticMDModel,
    DegradationMechanism,
    DegradationModelingError,
    DegradationPrediction,
    DegradationState,
    EnvironmentalConditions,
    EnvironmentalStressor,
    InvalidEnvironmentalConditionsError,
    KineticMonteCarloModel,
    MaterialProperties,
    ModelNotFittedError,
    MultiScaleDegradationPredictor,
    ReactiveForceFieldModel,
    SimulationScale,
    SurrogateAIModel,
    create_accelerated_aging_conditions,
    create_outdoor_weathering_conditions,
    create_standard_material,
    run_degradation_prediction,
)

# Check for optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class TestMaterialProperties:
    """Test material properties data structure."""

    def test_material_properties_creation(self):
        """Test creating material properties."""
        properties = MaterialProperties(
            molecular_weight=100000.0,
            glass_transition_temp=373.15,
            melting_point=473.15,
            crystallinity=0.3,
            crosslink_density=0.05,
            functional_groups=["ester", "aromatic"],
            chemical_composition={"C": 0.6, "H": 0.3, "O": 0.1},
        )

        assert properties.molecular_weight == 100000.0
        assert properties.glass_transition_temp == 373.15
        assert properties.crystallinity == 0.3
        assert "ester" in properties.functional_groups
        assert properties.chemical_composition["C"] == 0.6

    def test_standard_material_creation(self):
        """Test standard material creation utility."""
        polymer = create_standard_material("polymer")
        assert isinstance(polymer, MaterialProperties)
        assert polymer.molecular_weight > 0
        assert len(polymer.functional_groups) > 0

        composite = create_standard_material("composite")
        assert isinstance(composite, MaterialProperties)
        assert composite.molecular_weight > polymer.molecular_weight


class TestEnvironmentalConditions:
    """Test environmental conditions and stressors."""

    def test_environmental_conditions_creation(self):
        """Test creating environmental conditions."""
        conditions = EnvironmentalConditions(
            stressors={
                EnvironmentalStressor.TEMPERATURE: 353.15,
                EnvironmentalStressor.HUMIDITY: 0.85,
                EnvironmentalStressor.UV_RADIATION: 0.8,
            },
            duration=1000.0,
        )

        assert conditions.duration == 1000.0
        assert conditions.stressors[EnvironmentalStressor.TEMPERATURE] == 353.15
        assert conditions.stressors[EnvironmentalStressor.HUMIDITY] == 0.85
        assert conditions.stressors[EnvironmentalStressor.UV_RADIATION] == 0.8

    def test_accelerated_aging_conditions(self):
        """Test accelerated aging conditions utility."""
        conditions = create_accelerated_aging_conditions(
            temperature=363.15, humidity=0.9, uv_intensity=0.9, duration=500.0
        )

        assert isinstance(conditions, EnvironmentalConditions)
        assert conditions.stressors[EnvironmentalStressor.TEMPERATURE] == 363.15
        assert conditions.duration == 500.0

    def test_outdoor_weathering_conditions(self):
        """Test outdoor weathering conditions."""
        temperate = create_outdoor_weathering_conditions("temperate")
        tropical = create_outdoor_weathering_conditions("tropical")
        arid = create_outdoor_weathering_conditions("arid")

        assert isinstance(temperate, EnvironmentalConditions)
        assert isinstance(tropical, EnvironmentalConditions)
        assert isinstance(arid, EnvironmentalConditions)

        # Tropical should have higher temperature and humidity
        assert (
            tropical.stressors[EnvironmentalStressor.TEMPERATURE]
            > temperate.stressors[EnvironmentalStressor.TEMPERATURE]
        )
        assert (
            tropical.stressors[EnvironmentalStressor.HUMIDITY]
            > temperate.stressors[EnvironmentalStressor.HUMIDITY]
        )


class TestDegradationState:
    """Test degradation state representation."""

    def test_degradation_state_creation(self):
        """Test creating degradation state."""
        state = DegradationState(
            time=100.0,
            molecular_weight_retention=0.85,
            mechanical_retention={"tensile_strength": 0.8, "modulus": 0.9},
            chemical_changes={"crosslink_density": 0.7},
        )

        assert state.time == 100.0
        assert state.molecular_weight_retention == 0.85
        assert state.mechanical_retention["tensile_strength"] == 0.8
        assert state.chemical_changes["crosslink_density"] == 0.7


class TestAtomisticMDModel:
    """Test atomistic molecular dynamics model."""

    def test_model_creation(self):
        """Test MD model initialization."""
        model = AtomisticMDModel(use_gpu=False)

        assert model.use_gpu is False
        assert model.fitted is False
        assert isinstance(model.simulation_params, dict)

    def test_degradation_prediction(self):
        """Test degradation prediction."""
        model = AtomisticMDModel(use_gpu=False)

        material = create_standard_material("polymer")
        conditions = create_accelerated_aging_conditions(duration=100.0)

        prediction = model.predict_degradation(material, conditions)

        assert isinstance(prediction, DegradationPrediction)
        assert len(prediction.degradation_states) > 0
        assert len(prediction.dominant_mechanisms) > 0
        assert prediction.half_life is None or prediction.half_life > 0

    def test_bond_breaking_rate_calculation(self):
        """Test bond breaking rate calculation."""
        model = AtomisticMDModel(use_gpu=False)

        material = create_standard_material("polymer")
        conditions = create_accelerated_aging_conditions()

        rate = model._calculate_bond_breaking_rate(material, conditions, 10.0)

        assert isinstance(rate, float)
        assert rate > 0

    def test_mechanism_identification(self):
        """Test identification of degradation mechanisms."""
        model = AtomisticMDModel(use_gpu=False)

        # High temperature conditions
        hot_conditions = EnvironmentalConditions(
            stressors={EnvironmentalStressor.TEMPERATURE: 400.0}, duration=100.0
        )

        mechanisms = model._identify_dominant_mechanisms(hot_conditions)
        assert DegradationMechanism.THERMAL_DEGRADATION in mechanisms

        # UV conditions
        uv_conditions = EnvironmentalConditions(
            stressors={EnvironmentalStressor.UV_RADIATION: 0.8}, duration=100.0
        )

        uv_mechanisms = model._identify_dominant_mechanisms(uv_conditions)
        assert DegradationMechanism.UV_DEGRADATION in uv_mechanisms

    def test_model_fitting(self):
        """Test model fitting process."""
        model = AtomisticMDModel(use_gpu=False)

        # Create dummy training data
        training_data = []

        model.fit(training_data)
        assert model.fitted is True


class TestReactiveForceFieldModel:
    """Test reactive force field model."""

    def test_model_creation(self):
        """Test ReaxFF model initialization."""
        model = ReactiveForceFieldModel(use_gpu=False)

        assert model.use_gpu is False
        assert model.fitted is False
        assert isinstance(model.reaction_parameters, dict)

    def test_degradation_prediction(self):
        """Test degradation prediction with ReaxFF."""
        model = ReactiveForceFieldModel(use_gpu=False)

        material = create_standard_material("polymer")
        conditions = create_accelerated_aging_conditions(duration=50.0)

        prediction = model.predict_degradation(material, conditions)

        assert isinstance(prediction, DegradationPrediction)
        assert len(prediction.degradation_states) > 0
        assert (
            DegradationMechanism.OXIDATIVE_DEGRADATION in prediction.dominant_mechanisms
        )

    def test_reaction_rate_calculation(self):
        """Test reaction rate calculation."""
        model = ReactiveForceFieldModel(use_gpu=False)

        material = create_standard_material("polymer")
        conditions = create_accelerated_aging_conditions()

        rate = model._calculate_reaction_rate(material, conditions, 5.0)

        assert isinstance(rate, float)
        assert rate > 0


class TestKineticMonteCarloModel:
    """Test kinetic Monte Carlo model."""

    def test_model_creation(self):
        """Test KMC model initialization."""
        model = KineticMonteCarloModel(use_gpu=False)

        assert model.use_gpu is False
        assert model.fitted is False
        assert isinstance(model.reaction_network, dict)

    def test_degradation_prediction(self):
        """Test degradation prediction with KMC."""
        model = KineticMonteCarloModel(use_gpu=False)

        material = create_standard_material("polymer")
        conditions = EnvironmentalConditions(
            stressors={EnvironmentalStressor.MECHANICAL_STRESS: 0.5}, duration=100.0
        )

        prediction = model.predict_degradation(material, conditions)

        assert isinstance(prediction, DegradationPrediction)
        assert len(prediction.degradation_states) > 0
        assert DegradationMechanism.MECHANICAL_FATIGUE in prediction.dominant_mechanisms

    def test_kmc_rate_calculation(self):
        """Test KMC rate calculation."""
        model = KineticMonteCarloModel(use_gpu=False)

        material = create_standard_material("polymer")
        conditions = EnvironmentalConditions(
            stressors={EnvironmentalStressor.MECHANICAL_STRESS: 0.3}, duration=100.0
        )

        rate = model._calculate_kmc_rate(material, conditions)

        assert isinstance(rate, float)
        assert rate > 0


class TestSurrogateAIModel:
    """Test AI surrogate model."""

    def test_model_creation_cpu(self):
        """Test AI model initialization for CPU."""
        model = SurrogateAIModel(use_gpu=False, model_type="random_forest")

        assert model.use_gpu is False
        assert model.model_type == "random_forest"
        assert model.fitted is False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_model_creation_gpu(self):
        """Test AI model initialization for GPU."""
        model = SurrogateAIModel(use_gpu=True, model_type="neural_network")

        assert model.model_type == "neural_network"
        assert model.device is not None

    def test_feature_extraction(self):
        """Test feature extraction for AI model."""
        model = SurrogateAIModel(use_gpu=False)

        material = create_standard_material("polymer")
        conditions = create_accelerated_aging_conditions()

        features = model._extract_features(material, conditions)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_degradation_prediction_unfitted(self):
        """Test prediction with unfitted model."""
        model = SurrogateAIModel(use_gpu=False)

        material = create_standard_material("polymer")
        conditions = create_accelerated_aging_conditions(duration=100.0)

        prediction = model.predict_degradation(material, conditions)

        assert isinstance(prediction, DegradationPrediction)
        assert len(prediction.degradation_states) > 0

    def test_model_fitting_sklearn(self):
        """Test model fitting with scikit-learn."""
        model = SurrogateAIModel(use_gpu=False, model_type="random_forest")

        # Create dummy training data
        training_data = []
        for i in range(10):
            material = create_standard_material("polymer")
            conditions = create_accelerated_aging_conditions(duration=100.0)

            # Create dummy degradation states
            states = [
                DegradationState(
                    time=10.0 * j, molecular_weight_retention=1.0 - 0.1 * j
                )
                for j in range(3)
            ]

            training_data.append((material, conditions, states))

        model.fit(training_data)
        assert model.fitted is True
        assert "sklearn_model" in model.models

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_model_fitting_neural_network(self):
        """Test model fitting with neural network."""
        model = SurrogateAIModel(use_gpu=False, model_type="neural_network")

        # Create training data
        training_data = []
        for i in range(5):
            material = create_standard_material("polymer")
            conditions = create_accelerated_aging_conditions(duration=50.0)

            states = [
                DegradationState(
                    time=10.0 * j, molecular_weight_retention=1.0 - 0.05 * j
                )
                for j in range(2)
            ]

            training_data.append((material, conditions, states))

        model.fit(training_data)
        assert model.fitted is True
        assert "nn_model" in model.models


class TestMultiScaleDegradationPredictor:
    """Test multi-scale degradation predictor."""

    def test_predictor_creation(self):
        """Test multi-scale predictor initialization."""
        predictor = MultiScaleDegradationPredictor(use_gpu=False)

        assert len(predictor.models) == 4
        assert SimulationScale.ATOMISTIC in predictor.models
        assert SimulationScale.MOLECULAR in predictor.models
        assert SimulationScale.MESOSCALE in predictor.models
        assert SimulationScale.MACROSCALE in predictor.models

        assert len(predictor.ensemble_weights) == 4
        assert sum(predictor.ensemble_weights.values()) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_multi_scale_prediction(self):
        """Test multi-scale degradation prediction."""
        predictor = MultiScaleDegradationPredictor(use_gpu=False)

        material = create_standard_material("polymer")
        conditions = create_accelerated_aging_conditions(duration=50.0)

        prediction = await predictor.predict_degradation(material, conditions)

        assert isinstance(prediction, DegradationPrediction)
        assert len(prediction.degradation_states) > 0
        assert len(prediction.dominant_mechanisms) > 0
        assert "ensemble" in prediction.material_id

    @pytest.mark.asyncio
    async def test_selective_scale_prediction(self):
        """Test prediction with selected scales."""
        predictor = MultiScaleDegradationPredictor(use_gpu=False)

        material = create_standard_material("polymer")
        conditions = create_accelerated_aging_conditions(duration=30.0)

        scales = [SimulationScale.ATOMISTIC, SimulationScale.MACROSCALE]
        prediction = await predictor.predict_degradation(material, conditions, scales)

        assert isinstance(prediction, DegradationPrediction)
        assert len(prediction.degradation_states) > 0

    def test_ensemble_weight_setting(self):
        """Test setting custom ensemble weights."""
        predictor = MultiScaleDegradationPredictor(use_gpu=False)

        custom_weights = {
            SimulationScale.ATOMISTIC: 0.5,
            SimulationScale.MOLECULAR: 0.3,
            SimulationScale.MESOSCALE: 0.1,
            SimulationScale.MACROSCALE: 0.1,
        }

        predictor.set_ensemble_weights(custom_weights)

        assert predictor.ensemble_weights[SimulationScale.ATOMISTIC] == 0.5
        assert predictor.ensemble_weights[SimulationScale.MOLECULAR] == 0.3
        assert sum(predictor.ensemble_weights.values()) == pytest.approx(1.0)

    def test_model_fitting(self):
        """Test fitting all scale models."""
        predictor = MultiScaleDegradationPredictor(use_gpu=False)

        # Create dummy training data
        training_data = []

        predictor.fit(training_data)
        assert predictor.fitted is True


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.mark.asyncio
    async def test_run_degradation_prediction(self):
        """Test utility function for running predictions."""
        material = create_standard_material("polymer")
        conditions = create_accelerated_aging_conditions(duration=100.0)

        prediction = await run_degradation_prediction(
            material, conditions, use_gpu=False
        )

        assert isinstance(prediction, DegradationPrediction)
        assert len(prediction.degradation_states) > 0

    @pytest.mark.asyncio
    async def test_run_degradation_prediction_selective_scales(self):
        """Test prediction with selective scales."""
        material = create_standard_material("composite")
        conditions = create_outdoor_weathering_conditions("tropical")

        scales = [SimulationScale.MOLECULAR, SimulationScale.MACROSCALE]
        prediction = await run_degradation_prediction(
            material, conditions, scales, use_gpu=False
        )

        assert isinstance(prediction, DegradationPrediction)
        assert len(prediction.degradation_states) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_material_type(self):
        """Test handling of invalid material type."""
        with pytest.raises(ValueError):
            create_standard_material("invalid_type")

    def test_invalid_climate(self):
        """Test handling of invalid climate."""
        with pytest.raises(ValueError):
            create_outdoor_weathering_conditions("invalid_climate")

    def test_degradation_mechanism_enum(self):
        """Test degradation mechanism enumeration."""
        assert DegradationMechanism.UV_DEGRADATION.value == "uv_degradation"
        assert DegradationMechanism.THERMAL_DEGRADATION.value == "thermal_degradation"
        assert (
            DegradationMechanism.OXIDATIVE_DEGRADATION.value == "oxidative_degradation"
        )

    def test_simulation_scale_enum(self):
        """Test simulation scale enumeration."""
        assert SimulationScale.ATOMISTIC.value == "atomistic"
        assert SimulationScale.MOLECULAR.value == "molecular"
        assert SimulationScale.MESOSCALE.value == "mesoscale"
        assert SimulationScale.MACROSCALE.value == "macroscale"

    def test_environmental_stressor_enum(self):
        """Test environmental stressor enumeration."""
        assert EnvironmentalStressor.UV_RADIATION.value == "uv_radiation"
        assert EnvironmentalStressor.TEMPERATURE.value == "temperature"
        assert EnvironmentalStressor.HUMIDITY.value == "humidity"


class TestPerformance:
    """Test performance and optimization aspects."""

    def test_prediction_timing(self):
        """Test that predictions complete in reasonable time."""
        model = AtomisticMDModel(use_gpu=False)

        material = create_standard_material("polymer")
        conditions = create_accelerated_aging_conditions(duration=10.0)

        start_time = datetime.now()
        prediction = model.predict_degradation(material, conditions)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        assert execution_time < 5.0  # Should complete within 5 seconds
        assert isinstance(prediction, DegradationPrediction)

    @pytest.mark.asyncio
    async def test_async_prediction_timing(self):
        """Test async prediction timing."""
        start_time = datetime.now()

        material = create_standard_material("polymer")
        conditions = create_accelerated_aging_conditions(duration=20.0)

        prediction = await run_degradation_prediction(
            material, conditions, use_gpu=False
        )

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        assert execution_time < 10.0  # Should complete within 10 seconds
        assert isinstance(prediction, DegradationPrediction)


if __name__ == "__main__":
    pytest.main([__file__])

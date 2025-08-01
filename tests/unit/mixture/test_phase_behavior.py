"""
Tests for phase behavior prediction module.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.core.exceptions import ChemiAccelError
from chemi_accel.mixture.core import ComponentData, MixtureData
from chemi_accel.mixture.phase_behavior import (
    AIPhasePredictor,
    NRTLModel,
    PengRobinsonEOS,
    PhaseBehaviorPredictor,
    PhaseEquilibrium,
)


class TestPhaseEquilibrium:
    """Tests for PhaseEquilibrium class."""

    def test_phase_equilibrium_creation(self):
        """Test basic phase equilibrium creation."""
        phases = [
            {"type": "liquid", "composition": [0.5, 0.5]},
            {"type": "vapor", "composition": [0.3, 0.7]},
        ]
        stability = {"overall": 0.8, "thermal": 0.9}

        pe = PhaseEquilibrium(
            temperature=298.15, pressure=101325.0, phases=phases, stability=stability
        )

        assert pe.temperature == 298.15
        assert pe.pressure == 101325.0
        assert pe.n_phases == 2
        assert pe.is_stable  # All stability values > 0

    def test_unstable_system(self):
        """Test unstable system detection."""
        phases = [{"type": "liquid", "composition": [1.0]}]
        stability = {"overall": -0.1}  # Negative stability

        pe = PhaseEquilibrium(
            temperature=298.15, pressure=101325.0, phases=phases, stability=stability
        )

        assert not pe.is_stable


class TestNRTLModel:
    """Tests for NRTL thermodynamic model."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = Mock()
        self.backend.name = "cpu"
        self.model = NRTLModel(self.backend)

    def test_model_initialization(self):
        """Test NRTL model initialization."""
        backend = Mock()
        model = NRTLModel(backend)

        assert model.backend is backend
        assert model.binary_parameters == {}

    def test_single_component_activity_coefficients(self):
        """Test activity coefficients for single component."""
        backend = Mock()
        backend.name = "cpu"
        model = NRTLModel(backend)

        components = [ComponentData(smiles="CCO", mol_fraction=1.0)]
        mixture = MixtureData(components=components)

        gamma = model.calculate_activity_coefficients(mixture, 298.15, 101325.0)

        assert len(gamma) == 1
        assert gamma[0] == 1.0  # Pure component has unit activity coefficient

    def test_binary_activity_coefficients(self):
        """Test activity coefficients for binary mixture."""
        backend = Mock()
        backend.name = "cpu"
        model = NRTLModel(backend)

        components = [
            ComponentData(smiles="CCO", mol_fraction=0.5),
            ComponentData(smiles="O", mol_fraction=0.5),
        ]
        mixture = MixtureData(components=components)

        gamma = model.calculate_activity_coefficients(mixture, 298.15, 101325.0)

        assert len(gamma) == 2
        assert all(g > 0 for g in gamma)  # Activity coefficients should be positive

    @patch("cupy.asarray")
    @patch("cupy.ones")
    def test_cuda_implementation(self, mock_ones, mock_asarray):
        """Test CUDA implementation path."""
        backend = Mock()
        backend.name = "cuda"
        model = NRTLModel(backend)

        components = [
            ComponentData(smiles="CCO", mol_fraction=0.5),
            ComponentData(smiles="O", mol_fraction=0.5),
        ]
        mixture = MixtureData(components=components)

        # Mock cupy arrays
        mock_asarray.return_value = Mock()
        mock_ones.return_value = Mock()

        # Should not raise an error
        gamma = model.calculate_activity_coefficients(mixture, 298.15, 101325.0)
        assert len(gamma) == 2


class TestPengRobinsonEOS:
    """Tests for Peng-Robinson equation of state."""

    def test_model_initialization(self):
        """Test PR EOS initialization."""
        backend = Mock()
        model = PengRobinsonEOS(backend)

        assert model.backend is backend
        assert model.critical_constants == {}
        assert model.acentric_factors == {}

    def test_fugacity_coefficients(self):
        """Test fugacity coefficient calculations."""
        backend = Mock()
        model = PengRobinsonEOS(backend)

        components = [
            ComponentData(smiles="CCO", mol_fraction=0.6),
            ComponentData(smiles="O", mol_fraction=0.4),
        ]
        mixture = MixtureData(components=components)

        phi = model.calculate_fugacity_coefficients(mixture, 298.15, 101325.0)

        assert len(phi) == 2
        assert all(p > 0 for p in phi)  # Fugacity coefficients should be positive

    def test_activity_coefficients_not_applicable(self):
        """Test that activity coefficients return unity for EOS."""
        backend = Mock()
        model = PengRobinsonEOS(backend)

        components = [ComponentData(smiles="CCO", mol_fraction=1.0)]
        mixture = MixtureData(components=components)

        gamma = model.calculate_activity_coefficients(mixture, 298.15, 101325.0)

        assert len(gamma) == 1
        assert gamma[0] == 1.0


class TestAIPhasePredictor:
    """Tests for AI phase behavior predictor."""

    def test_model_initialization(self):
        """Test AI predictor initialization."""
        backend = Mock()
        predictor = AIPhasePredictor(backend)

        assert predictor.backend is backend
        assert predictor.model is None
        assert not predictor.is_trained

    @patch("chemi_accel.mixture.phase_behavior.RandomForestRegressor")
    @patch("chemi_accel.mixture.phase_behavior.StandardScaler")
    def test_model_training(self, mock_scaler_class, mock_rf_class):
        """Test AI model training."""
        backend = Mock()
        predictor = AIPhasePredictor(backend)

        # Mock sklearn components
        mock_scaler = Mock()
        mock_scaler_class.return_value = mock_scaler
        mock_rf = Mock()
        mock_rf_class.return_value = mock_rf

        # Create mock training data
        training_data = Mock()
        training_data.__getitem__ = Mock(
            return_value=np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
        )

        predictor.train_model(training_data)

        assert predictor.is_trained
        mock_scaler_class.assert_called_once()
        mock_rf_class.assert_called_once()

    def test_prediction_without_training(self):
        """Test error when predicting without training."""
        backend = Mock()
        predictor = AIPhasePredictor(backend)

        components = [ComponentData(smiles="CCO", mol_fraction=1.0)]
        mixture = MixtureData(components=components)

        with pytest.raises(ChemiAccelError, match="AI model not trained"):
            predictor.predict_phase_behavior(mixture, 298.15, 101325.0)


class TestPhaseBehaviorPredictor:
    """Tests for main phase behavior predictor."""

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        backend = Mock()
        predictor = PhaseBehaviorPredictor(backend)

        assert predictor.backend is backend
        assert "nrtl" in predictor.models
        assert "peng_robinson" in predictor.models
        assert "ai_surrogate" in predictor.models
        assert predictor.default_model == "nrtl"

    def test_predict_phase_equilibrium_basic(self):
        """Test basic phase equilibrium prediction."""
        backend = Mock()
        predictor = PhaseBehaviorPredictor(backend)

        components = [
            ComponentData(smiles="CCO", mol_fraction=0.5),
            ComponentData(smiles="O", mol_fraction=0.5),
        ]
        mixture = MixtureData(
            components=components, temperature=298.15, pressure=101325.0
        )

        # Mock the NRTL model
        mock_nrtl = Mock()
        mock_nrtl.calculate_activity_coefficients.return_value = np.array([1.1, 1.2])
        mock_nrtl.calculate_fugacity_coefficients.return_value = np.array([0.9, 0.8])
        predictor.models["nrtl"] = mock_nrtl

        result = predictor.predict_phase_equilibrium(mixture)

        assert isinstance(result, PhaseEquilibrium)
        assert result.temperature == 298.15
        assert result.pressure == 101325.0
        assert len(result.phases) > 0

    def test_predict_with_ai_surrogate(self):
        """Test prediction using AI surrogate model."""
        backend = Mock()
        predictor = PhaseBehaviorPredictor(backend)

        components = [ComponentData(smiles="CCO", mol_fraction=0.7)]
        mixture = MixtureData(components=components)

        # Mock AI predictor
        mock_ai = Mock()
        mock_ai.predict_phase_behavior.return_value = {
            "phase_split_probability": 0.2,
            "stability_score": 0.8,
        }
        predictor.models["ai_surrogate"] = mock_ai

        result = predictor.predict_phase_equilibrium(mixture, model="ai_surrogate")

        assert isinstance(result, PhaseEquilibrium)
        assert len(result.phases) > 0
        mock_ai.predict_phase_behavior.assert_called_once()

    def test_generate_phase_diagram(self):
        """Test phase diagram generation."""
        backend = Mock()
        predictor = PhaseBehaviorPredictor(backend)

        components = [ComponentData(smiles="CCO", mol_fraction=0.5)]
        mixture = MixtureData(components=components)

        # Mock the thermodynamic model
        mock_model = Mock()
        mock_model.calculate_activity_coefficients.return_value = np.array([1.0])
        mock_model.calculate_fugacity_coefficients.return_value = np.array([1.0])
        predictor.models["nrtl"] = mock_model

        result = predictor.predict_phase_equilibrium(
            mixture, temperature_range=(250, 350), pressure_range=(1e5, 5e5), n_points=5
        )

        assert result.phase_diagram_data is not None
        assert "data_points" in result.phase_diagram_data
        assert "temperature_range" in result.phase_diagram_data
        assert "pressure_range" in result.phase_diagram_data

    @patch("scipy.optimize.brentq")
    def test_rachford_rice_solution(self, mock_brentq):
        """Test Rachford-Rice equation solver."""
        backend = Mock()
        predictor = PhaseBehaviorPredictor(backend)

        mock_brentq.return_value = 0.3  # Mock vapor fraction

        z = np.array([0.5, 0.5])
        K = np.array([2.0, 0.5])

        vapor_frac = predictor._solve_rachford_rice(z, K)

        assert vapor_frac == 0.3
        mock_brentq.assert_called_once()

    def test_stability_calculation(self):
        """Test stability calculation."""
        backend = Mock()
        predictor = PhaseBehaviorPredictor(backend)

        components = [ComponentData(smiles="CCO", mol_fraction=1.0)]
        mixture = MixtureData(components=components)

        gamma = np.array([1.0])
        phi = np.array([1.0])

        stability = predictor._calculate_stability(
            mixture, 298.15, 101325.0, gamma, phi
        )

        assert isinstance(stability, dict)
        assert "overall" in stability
        assert stability["overall"] > 0

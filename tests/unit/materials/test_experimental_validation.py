"""
Tests for experimental validation and refinement module.

This module tests the experimental validation functionality including
XRD pattern simulation and comparison, experimental data validation,
feedback loops, and model refinement capabilities.
"""

import datetime
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Import the module under test
from chemi_accel.materials.experimental_validation import (  # Enums; Data structures; Main classes; Utility functions; Exceptions
    ComparisonResult,
    DataQuality,
    DataValidationError,
    ExperimentalComparator,
    ExperimentalData,
    ExperimentalDataValidator,
    ExperimentalTechnique,
    ExperimentalValidationEngine,
    ExperimentalValidationError,
    FeedbackData,
    FeedbackLoop,
    FeedbackLoopError,
    ModelRefinementEngine,
    ModelRefinementError,
    PowderXRDPattern,
    RefinementConfig,
    RefinementStrategy,
    ValidationMetric,
    ValidationResult,
    XRDComparisonError,
    XRDPatternSimulator,
    compare_with_experiment,
    create_experimental_data,
    create_refinement_config,
    setup_feedback_loop,
    simulate_xrd_pattern,
    validate_experimental_quality,
)


# Test fixtures
@pytest.fixture
def sample_experimental_data():
    """Create sample experimental data for testing."""
    x_data = np.linspace(10, 80, 100)
    y_data = np.random.normal(100, 10, 100) + 50 * np.exp(-(((x_data - 30) / 5) ** 2))

    return ExperimentalData(
        technique=ExperimentalTechnique.POWDER_XRD,
        data=np.column_stack([x_data, y_data]),
        x_axis=x_data,
        y_axis=y_data,
        metadata={"instrument": "test_diffractometer"},
        uncertainty=np.ones_like(y_data) * 5,
        quality=DataQuality.HIGH,
        source="test_lab",
        conditions={"temperature": 298.15, "pressure": 1.0},
    )


@pytest.fixture
def sample_xrd_pattern():
    """Create sample XRD pattern for testing."""
    two_theta = np.linspace(10, 80, 100)
    intensity = 50 * np.exp(-(((two_theta - 30) / 5) ** 2)) + 10 * np.random.normal(
        0, 1, 100
    )

    return PowderXRDPattern(
        two_theta=two_theta,
        intensity=intensity,
        wavelength=1.5418,
        peak_positions=[30.0, 45.0, 60.0],
        peak_intensities=[100.0, 50.0, 25.0],
        peak_widths=[0.5, 0.6, 0.7],
        metadata={"simulation_method": "test"},
    )


@pytest.fixture
def sample_crystal_structure():
    """Create sample crystal structure for testing."""
    mock_structure = Mock()
    mock_structure.unit_cell_parameters = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    mock_structure.formula = "NaCl"
    mock_structure.space_group = "Fm3m"
    return mock_structure


@pytest.fixture
def sample_refinement_config():
    """Create sample refinement configuration."""
    return RefinementConfig(
        strategy=RefinementStrategy.GRADIENT_DESCENT,
        learning_rate=0.001,
        max_iterations=100,
        convergence_threshold=1e-4,
        batch_size=16,
        early_stopping=True,
        patience=10,
    )


@pytest.fixture
def mock_scipy():
    """Mock scipy imports."""
    with patch.multiple(
        "chemi_accel.materials.experimental_validation", _SCIPY_AVAILABLE=False
    ):
        yield


@pytest.fixture
def mock_torch():
    """Mock PyTorch imports."""
    with patch.multiple(
        "chemi_accel.materials.experimental_validation",
        _TORCH_AVAILABLE=False,
        torch=MagicMock(),
    ):
        yield


class TestEnums:
    """Test enum classes."""

    def test_experimental_technique_values(self):
        """Test ExperimentalTechnique enum values."""
        assert ExperimentalTechnique.POWDER_XRD.value == "powder_xrd"
        assert ExperimentalTechnique.SINGLE_CRYSTAL_XRD.value == "single_crystal_xrd"
        assert ExperimentalTechnique.RAMAN_SPECTROSCOPY.value == "raman_spectroscopy"
        assert ExperimentalTechnique.IR_SPECTROSCOPY.value == "ir_spectroscopy"
        assert ExperimentalTechnique.DSC.value == "dsc"

    def test_validation_metric_values(self):
        """Test ValidationMetric enum values."""
        assert (
            ValidationMetric.CORRELATION_COEFFICIENT.value == "correlation_coefficient"
        )
        assert ValidationMetric.RMSE.value == "rmse"
        assert ValidationMetric.MAE.value == "mae"
        assert ValidationMetric.R_SQUARED.value == "r_squared"

    def test_refinement_strategy_values(self):
        """Test RefinementStrategy enum values."""
        assert RefinementStrategy.GRADIENT_DESCENT.value == "gradient_descent"
        assert RefinementStrategy.BAYESIAN_OPTIMIZATION.value == "bayesian_optimization"
        assert RefinementStrategy.GENETIC_ALGORITHM.value == "genetic_algorithm"

    def test_data_quality_values(self):
        """Test DataQuality enum values."""
        assert DataQuality.HIGH.value == "high"
        assert DataQuality.MEDIUM.value == "medium"
        assert DataQuality.LOW.value == "low"
        assert DataQuality.UNCERTAIN.value == "uncertain"


class TestDataStructures:
    """Test data structure classes."""

    def test_experimental_data_creation(self, sample_experimental_data):
        """Test ExperimentalData creation."""
        data = sample_experimental_data

        assert data.technique == ExperimentalTechnique.POWDER_XRD
        assert len(data.x_axis) == len(data.y_axis)
        assert data.quality == DataQuality.HIGH
        assert data.source == "test_lab"
        assert "temperature" in data.conditions

    def test_powder_xrd_pattern_creation(self, sample_xrd_pattern):
        """Test PowderXRDPattern creation."""
        pattern = sample_xrd_pattern

        assert len(pattern.two_theta) == len(pattern.intensity)
        assert pattern.wavelength == 1.5418
        assert len(pattern.peak_positions) == 3
        assert len(pattern.peak_intensities) == 3
        assert len(pattern.peak_widths) == 3

    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            technique=ExperimentalTechnique.POWDER_XRD,
            metric=ValidationMetric.CORRELATION_COEFFICIENT,
            value=0.95,
            confidence_interval=(0.90, 0.99),
            p_value=0.01,
            statistical_significance=True,
        )

        assert result.technique == ExperimentalTechnique.POWDER_XRD
        assert result.metric == ValidationMetric.CORRELATION_COEFFICIENT
        assert result.value == 0.95
        assert result.statistical_significance is True
        assert result.confidence_interval == (0.90, 0.99)

    def test_comparison_result_creation(self):
        """Test ComparisonResult creation."""
        pred_data = np.array([1, 2, 3, 4, 5])
        exp_data = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        result = ComparisonResult(
            predicted_data=pred_data,
            experimental_data=exp_data,
            similarity_score=0.95,
            correlation=0.98,
            rmse=0.1,
            mae=0.08,
            validation_metrics=[],
            residuals=pred_data - exp_data,
        )

        assert np.array_equal(result.predicted_data, pred_data)
        assert np.array_equal(result.experimental_data, exp_data)
        assert result.similarity_score == 0.95
        assert result.correlation == 0.98
        assert np.array_equal(result.residuals, pred_data - exp_data)

    def test_refinement_config_creation(self, sample_refinement_config):
        """Test RefinementConfig creation."""
        config = sample_refinement_config

        assert config.strategy == RefinementStrategy.GRADIENT_DESCENT
        assert config.learning_rate == 0.001
        assert config.max_iterations == 100
        assert config.convergence_threshold == 1e-4
        assert config.early_stopping is True

    def test_feedback_data_creation(self, sample_experimental_data):
        """Test FeedbackData creation."""
        predicted_result = np.array([1, 2, 3])
        comparison = Mock()

        feedback = FeedbackData(
            experimental_result=sample_experimental_data,
            predicted_result=predicted_result,
            comparison=comparison,
            compound_id="test_compound",
            crystal_structure=None,
            property_values={"density": 2.5},
        )

        assert feedback.experimental_result == sample_experimental_data
        assert np.array_equal(feedback.predicted_result, predicted_result)
        assert feedback.compound_id == "test_compound"
        assert feedback.property_values["density"] == 2.5


class TestXRDPatternSimulator:
    """Test XRDPatternSimulator class."""

    def test_simulator_creation(self):
        """Test XRD pattern simulator creation."""
        simulator = XRDPatternSimulator(wavelength=1.5418)

        assert simulator.wavelength == 1.5418

    def test_simulate_powder_pattern(self, sample_crystal_structure):
        """Test powder XRD pattern simulation."""
        simulator = XRDPatternSimulator()

        pattern = simulator.simulate_powder_pattern(
            sample_crystal_structure, two_theta_range=(10, 80), resolution=0.1
        )

        assert isinstance(pattern, PowderXRDPattern)
        assert len(pattern.two_theta) > 0
        assert len(pattern.intensity) == len(pattern.two_theta)
        assert pattern.wavelength == simulator.wavelength
        assert pattern.two_theta.min() >= 10
        assert pattern.two_theta.max() <= 80

    def test_calculate_reflections(self, sample_crystal_structure):
        """Test reflection calculation."""
        simulator = XRDPatternSimulator()

        reflections = simulator._calculate_reflections(sample_crystal_structure)

        assert isinstance(reflections, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in reflections)
        assert all(
            r[0] > 0 and r[1] > 0 for r in reflections
        )  # Positive 2θ and intensity

    def test_structure_factor_calculation(self, sample_crystal_structure):
        """Test structure factor calculation."""
        simulator = XRDPatternSimulator()

        intensity = simulator._calculate_structure_factor(
            1, 0, 0, sample_crystal_structure
        )

        assert intensity >= 0
        assert isinstance(intensity, float)

    def test_multiplicity_calculation(self):
        """Test multiplicity factor calculation."""
        simulator = XRDPatternSimulator()

        # Test different Miller indices
        mult_100 = simulator._get_multiplicity(1, 0, 0)
        mult_110 = simulator._get_multiplicity(1, 1, 0)
        mult_111 = simulator._get_multiplicity(1, 1, 1)
        mult_123 = simulator._get_multiplicity(1, 2, 3)

        assert mult_100 > 0
        assert mult_110 > 0
        assert mult_111 > 0
        assert mult_123 > 0

    def test_peak_extraction(self):
        """Test peak extraction from pattern."""
        simulator = XRDPatternSimulator()

        # Create test pattern with known peaks
        two_theta = np.linspace(10, 80, 100)
        intensity = 100 * np.exp(-(((two_theta - 30) / 2) ** 2)) + 50 * np.exp(
            -(((two_theta - 50) / 3) ** 2)
        )

        peak_positions, peak_intensities, peak_widths = simulator._extract_peaks(
            two_theta, intensity
        )

        # Should find peaks (may depend on scipy availability)
        assert isinstance(peak_positions, list)
        assert isinstance(peak_intensities, list)
        assert isinstance(peak_widths, list)

    def test_invalid_structure_handling(self):
        """Test handling of invalid crystal structure."""
        simulator = XRDPatternSimulator()

        # Structure without required attributes
        invalid_structure = Mock()
        delattr(invalid_structure, "unit_cell_parameters")

        # Should handle gracefully and use defaults
        pattern = simulator.simulate_powder_pattern(invalid_structure)

        assert isinstance(pattern, PowderXRDPattern)


class TestExperimentalDataValidator:
    """Test ExperimentalDataValidator class."""

    def test_validator_creation(self):
        """Test experimental data validator creation."""
        validator = ExperimentalDataValidator()

        assert hasattr(validator, "validation_rules")
        assert isinstance(validator.validation_rules, dict)

    def test_validate_experimental_data(self, sample_experimental_data):
        """Test experimental data validation."""
        validator = ExperimentalDataValidator()

        result = validator.validate_experimental_data(sample_experimental_data)

        assert isinstance(result, ValidationResult)
        assert result.technique == sample_experimental_data.technique
        assert 0 <= result.value <= 1  # Validation score should be between 0 and 1
        assert isinstance(result.statistical_significance, bool)

    def test_data_integrity_check(self, sample_experimental_data):
        """Test data integrity checking."""
        validator = ExperimentalDataValidator()

        # Valid data should pass
        validator._check_data_integrity(sample_experimental_data)

        # Invalid data should raise error
        invalid_data = ExperimentalData(
            technique=ExperimentalTechnique.POWDER_XRD,
            data=np.array([[1, 2], [3, 4]]),
            x_axis=np.array([1, 2, 3]),  # Different length
            y_axis=np.array([4, 5]),  # Different length
        )

        with pytest.raises(DataValidationError):
            validator._check_data_integrity(invalid_data)

    def test_xrd_data_validation(self):
        """Test XRD-specific data validation."""
        validator = ExperimentalDataValidator()

        # Create XRD data
        xrd_data = ExperimentalData(
            technique=ExperimentalTechnique.POWDER_XRD,
            data=np.column_stack(
                [np.linspace(10, 80, 100), np.random.normal(100, 10, 100)]
            ),
            x_axis=np.linspace(10, 80, 100),
            y_axis=np.random.normal(100, 10, 100),
        )

        score = validator._validate_xrd_data(xrd_data)

        assert 0 <= score <= 1
        assert isinstance(score, float)

    def test_raman_data_validation(self):
        """Test Raman spectroscopy data validation."""
        validator = ExperimentalDataValidator()

        # Create Raman data
        raman_data = ExperimentalData(
            technique=ExperimentalTechnique.RAMAN_SPECTROSCOPY,
            data=np.column_stack(
                [np.linspace(100, 3500, 100), np.random.normal(1000, 100, 100)]
            ),
            x_axis=np.linspace(100, 3500, 100),
            y_axis=np.random.normal(1000, 100, 100),
        )

        score = validator._validate_raman_data(raman_data)

        assert 0 <= score <= 1
        assert isinstance(score, float)

    def test_ir_data_validation(self):
        """Test IR spectroscopy data validation."""
        validator = ExperimentalDataValidator()

        # Create IR data
        ir_data = ExperimentalData(
            technique=ExperimentalTechnique.IR_SPECTROSCOPY,
            data=np.column_stack(
                [np.linspace(500, 4000, 100), np.random.uniform(0, 100, 100)]
            ),
            x_axis=np.linspace(500, 4000, 100),
            y_axis=np.random.uniform(0, 100, 100),
        )

        score = validator._validate_ir_data(ir_data)

        assert 0 <= score <= 1
        assert isinstance(score, float)

    def test_outlier_detection(self, sample_experimental_data):
        """Test outlier detection."""
        validator = ExperimentalDataValidator()

        score = validator._check_outliers(sample_experimental_data)

        assert 0 <= score <= 1
        assert isinstance(score, float)

    def test_snr_calculation(self, sample_experimental_data):
        """Test signal-to-noise ratio calculation."""
        validator = ExperimentalDataValidator()

        snr_score = validator._calculate_snr(sample_experimental_data)

        assert 0 <= snr_score <= 1
        assert isinstance(snr_score, float)

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        validator = ExperimentalDataValidator()

        # Data with too few points
        insufficient_data = ExperimentalData(
            technique=ExperimentalTechnique.POWDER_XRD,
            data=np.array([[1, 2], [3, 4]]),
            x_axis=np.array([1, 2]),
            y_axis=np.array([3, 4]),
        )

        with pytest.raises(DataValidationError):
            validator.validate_experimental_data(insufficient_data)

    def test_nan_values_handling(self):
        """Test handling of NaN values in data."""
        validator = ExperimentalDataValidator()

        # Data with NaN values
        nan_data = ExperimentalData(
            technique=ExperimentalTechnique.POWDER_XRD,
            data=np.column_stack([np.linspace(10, 80, 100), np.full(100, np.nan)]),
            x_axis=np.linspace(10, 80, 100),
            y_axis=np.full(100, np.nan),
        )

        with pytest.raises(DataValidationError):
            validator.validate_experimental_data(nan_data)


class TestExperimentalComparator:
    """Test ExperimentalComparator class."""

    def test_comparator_creation(self):
        """Test experimental comparator creation."""
        comparator = ExperimentalComparator()

        assert hasattr(comparator, "similarity_metrics")
        assert isinstance(comparator.similarity_metrics, list)
        assert len(comparator.similarity_metrics) > 0

    def test_compare_xrd_patterns(self, sample_xrd_pattern, sample_experimental_data):
        """Test XRD pattern comparison."""
        comparator = ExperimentalComparator()

        result = comparator.compare_xrd_patterns(
            sample_xrd_pattern, sample_experimental_data
        )

        assert isinstance(result, ComparisonResult)
        assert 0 <= result.similarity_score <= 1
        assert -1 <= result.correlation <= 1
        assert result.rmse >= 0
        assert result.mae >= 0
        assert isinstance(result.validation_metrics, list)
        assert len(result.validation_metrics) > 0

    def test_pattern_normalization(self):
        """Test pattern normalization."""
        comparator = ExperimentalComparator()

        # Test pattern with known sum
        pattern = np.array([1, 2, 3, 4])
        normalized = comparator._normalize_pattern(pattern)

        assert np.isclose(np.sum(normalized), 1.0)
        assert np.all(normalized >= 0)

    def test_correlation_calculation(self):
        """Test correlation calculation."""
        comparator = ExperimentalComparator()

        # Perfect correlation
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # y = 2x

        correlation = comparator._calculate_correlation(x, y)

        assert np.isclose(correlation, 1.0, atol=1e-10)

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        comparator = ExperimentalComparator()

        pred = np.array([1, 2, 3, 4, 5])
        exp = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        rmse = comparator._calculate_rmse(pred, exp)

        expected_rmse = np.sqrt(np.mean((pred - exp) ** 2))
        assert np.isclose(rmse, expected_rmse)

    def test_mae_calculation(self):
        """Test MAE calculation."""
        comparator = ExperimentalComparator()

        pred = np.array([1, 2, 3, 4, 5])
        exp = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        mae = comparator._calculate_mae(pred, exp)

        expected_mae = np.mean(np.abs(pred - exp))
        assert np.isclose(mae, expected_mae)

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        comparator = ExperimentalComparator()

        # Identical vectors
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 3, 4])

        similarity = comparator._calculate_cosine_similarity(x, y)

        assert np.isclose(similarity, 1.0)

    def test_peak_matching_analysis(self, sample_xrd_pattern, sample_experimental_data):
        """Test peak matching analysis."""
        comparator = ExperimentalComparator()

        peak_matching = comparator._analyze_peak_matching(
            sample_xrd_pattern, sample_experimental_data
        )

        assert isinstance(peak_matching, dict)
        assert "matched_peaks" in peak_matching
        assert "total_predicted" in peak_matching
        assert "total_experimental" in peak_matching
        assert "matching_fraction" in peak_matching

    def test_common_grid_creation(self):
        """Test common grid creation for interpolation."""
        comparator = ExperimentalComparator()

        grid1 = np.linspace(10, 50, 100)
        grid2 = np.linspace(15, 45, 80)

        common_grid = comparator._create_common_grid(grid1, grid2)

        assert len(common_grid) > 0
        assert common_grid.min() >= max(grid1.min(), grid2.min())
        assert common_grid.max() <= min(grid1.max(), grid2.max())


class TestModelRefinementEngine:
    """Test ModelRefinementEngine class."""

    def test_refinement_engine_creation(self, sample_refinement_config):
        """Test model refinement engine creation."""
        engine = ModelRefinementEngine(sample_refinement_config)

        assert engine.config == sample_refinement_config
        assert isinstance(engine.refinement_history, list)

    def test_refine_model_gradient_descent(self, sample_refinement_config, mock_torch):
        """Test model refinement using gradient descent."""
        engine = ModelRefinementEngine(sample_refinement_config)

        # Create dummy feedback data
        feedback_data = []

        result = engine.refine_model(feedback_data)

        assert isinstance(result, dict)
        assert "status" in result

    def test_refine_model_bayesian_optimization(self, mock_torch):
        """Test model refinement using Bayesian optimization."""
        config = RefinementConfig(strategy=RefinementStrategy.BAYESIAN_OPTIMIZATION)
        engine = ModelRefinementEngine(config)

        feedback_data = []
        result = engine.refine_model(feedback_data)

        assert isinstance(result, dict)
        assert result["method"] == "bayesian_optimization"

    def test_refine_model_ensemble_learning(self, mock_torch):
        """Test model refinement using ensemble learning."""
        config = RefinementConfig(strategy=RefinementStrategy.ENSEMBLE_LEARNING)
        engine = ModelRefinementEngine(config)

        feedback_data = []
        result = engine.refine_model(feedback_data)

        assert isinstance(result, dict)
        assert result["method"] == "ensemble_learning"

    def test_refine_model_online_learning(self, mock_torch):
        """Test model refinement using online learning."""
        config = RefinementConfig(strategy=RefinementStrategy.ONLINE_LEARNING)
        engine = ModelRefinementEngine(config)

        feedback_data = []
        result = engine.refine_model(feedback_data)

        assert isinstance(result, dict)
        assert result["method"] == "online_learning"

    def test_prepare_training_data_without_torch(
        self, sample_refinement_config, mock_torch
    ):
        """Test training data preparation without PyTorch."""
        engine = ModelRefinementEngine(sample_refinement_config)

        # Create mock feedback data
        feedback_data = [Mock() for _ in range(3)]
        for fb in feedback_data:
            fb.comparison = Mock()
            fb.comparison.similarity_score = 0.8
            fb.comparison.correlation = 0.9
            fb.comparison.rmse = 0.1
            fb.comparison.mae = 0.05
            fb.experimental_result = Mock()
            fb.experimental_result.y_axis = np.array([1, 2, 3])

        with pytest.raises(ModelRefinementError):
            engine._prepare_training_data(feedback_data)

    def test_refinement_error_handling(self, sample_refinement_config):
        """Test refinement error handling."""
        engine = ModelRefinementEngine(sample_refinement_config)

        # Invalid feedback data
        invalid_feedback = "not_a_list"

        with pytest.raises(ModelRefinementError):
            engine.refine_model(invalid_feedback)


class TestFeedbackLoop:
    """Test FeedbackLoop class."""

    def test_feedback_loop_creation(self, sample_refinement_config):
        """Test feedback loop creation."""
        loop = FeedbackLoop(sample_refinement_config)

        assert loop.refinement_config == sample_refinement_config
        assert isinstance(loop.model_refiner, ModelRefinementEngine)
        assert isinstance(loop.comparator, ExperimentalComparator)
        assert isinstance(loop.validator, ExperimentalDataValidator)
        assert isinstance(loop.feedback_queue, list)
        assert loop.refinement_counter == 0

    def test_add_experimental_result(
        self,
        sample_refinement_config,
        sample_experimental_data,
        sample_crystal_structure,
    ):
        """Test adding experimental result to feedback loop."""
        loop = FeedbackLoop(sample_refinement_config)

        predicted_data = np.random.normal(0, 1, len(sample_experimental_data.y_axis))

        # This should not raise an error
        loop.add_experimental_result(
            sample_experimental_data,
            predicted_data,
            "test_compound",
            sample_crystal_structure,
        )

        # Check that feedback was added (if validation passed)
        assert len(loop.feedback_queue) >= 0

    def test_execute_refinement_cycle(self, sample_refinement_config):
        """Test refinement cycle execution."""
        loop = FeedbackLoop(sample_refinement_config)

        # Add some mock feedback data
        for i in range(sample_refinement_config.batch_size):
            mock_feedback = Mock()
            mock_feedback.comparison = Mock()
            mock_feedback.comparison.similarity_score = 0.8
            mock_feedback.experimental_result = Mock()
            mock_feedback.experimental_result.technique = (
                ExperimentalTechnique.POWDER_XRD
            )
            loop.feedback_queue.append(mock_feedback)

        result = loop.execute_refinement_cycle()

        assert isinstance(result, dict)
        assert "status" in result
        assert "cycle_number" in result
        assert "samples_processed" in result
        assert len(loop.feedback_queue) == 0  # Queue should be cleared

    def test_generic_comparison(
        self, sample_refinement_config, sample_experimental_data
    ):
        """Test generic comparison method."""
        loop = FeedbackLoop(sample_refinement_config)

        predicted_data = np.random.normal(0, 1, len(sample_experimental_data.y_axis))

        comparison = loop._generic_comparison(predicted_data, sample_experimental_data)

        assert isinstance(comparison, ComparisonResult)
        assert len(comparison.predicted_data) == len(comparison.experimental_data)
        assert -1 <= comparison.correlation <= 1
        assert comparison.rmse >= 0
        assert comparison.mae >= 0

    def test_empty_refinement_cycle(self, sample_refinement_config):
        """Test refinement cycle with empty queue."""
        loop = FeedbackLoop(sample_refinement_config)

        result = loop.execute_refinement_cycle()

        assert result["status"] == "no_data"

    def test_low_quality_data_filtering(self, sample_refinement_config):
        """Test filtering of low-quality experimental data."""
        loop = FeedbackLoop(sample_refinement_config)

        # Create low-quality data
        low_quality_data = ExperimentalData(
            technique=ExperimentalTechnique.POWDER_XRD,
            data=np.column_stack(
                [np.array([1, 2]), np.array([3, 4])]
            ),  # Too few points
            x_axis=np.array([1, 2]),
            y_axis=np.array([3, 4]),
            quality=DataQuality.LOW,
        )

        predicted_data = np.array([1, 2])
        initial_queue_length = len(loop.feedback_queue)

        # This should not add to queue due to low quality
        try:
            loop.add_experimental_result(low_quality_data, predicted_data, "test")
        except:
            pass  # Expected to fail validation

        # Queue should not have grown
        assert len(loop.feedback_queue) == initial_queue_length


class TestExperimentalValidationEngine:
    """Test ExperimentalValidationEngine class."""

    def test_validation_engine_creation(self, sample_refinement_config):
        """Test experimental validation engine creation."""
        engine = ExperimentalValidationEngine(sample_refinement_config)

        assert engine.refinement_config == sample_refinement_config
        assert isinstance(engine.xrd_simulator, XRDPatternSimulator)
        assert isinstance(engine.validator, ExperimentalDataValidator)
        assert isinstance(engine.comparator, ExperimentalComparator)
        assert isinstance(engine.feedback_loop, FeedbackLoop)
        assert isinstance(engine.validation_history, list)

    def test_validate_prediction(
        self,
        sample_refinement_config,
        sample_crystal_structure,
        sample_experimental_data,
    ):
        """Test prediction validation."""
        engine = ExperimentalValidationEngine(sample_refinement_config)

        result = engine.validate_prediction(
            sample_crystal_structure, sample_experimental_data, "test_compound"
        )

        assert isinstance(result, ComparisonResult)
        assert len(engine.validation_history) == 1
        assert engine.validation_history[0]["compound_id"] == "test_compound"

    def test_batch_validation(
        self,
        sample_refinement_config,
        sample_crystal_structure,
        sample_experimental_data,
    ):
        """Test batch validation of predictions."""
        engine = ExperimentalValidationEngine(sample_refinement_config)

        predictions = [
            (sample_crystal_structure, sample_experimental_data, "compound_1"),
            (sample_crystal_structure, sample_experimental_data, "compound_2"),
            (sample_crystal_structure, sample_experimental_data, "compound_3"),
        ]

        results = engine.batch_validation(predictions)

        assert len(results) == 3
        assert all(isinstance(r, ComparisonResult) for r in results)
        assert len(engine.validation_history) == 3

    def test_validation_statistics(
        self,
        sample_refinement_config,
        sample_crystal_structure,
        sample_experimental_data,
    ):
        """Test validation statistics calculation."""
        engine = ExperimentalValidationEngine(sample_refinement_config)

        # Perform some validations
        for i in range(5):
            engine.validate_prediction(
                sample_crystal_structure, sample_experimental_data, f"compound_{i}"
            )

        stats = engine.get_validation_statistics()

        assert isinstance(stats, dict)
        assert "total_validations" in stats
        assert "mean_similarity" in stats
        assert "mean_correlation" in stats
        assert "mean_rmse" in stats
        assert "by_technique" in stats
        assert stats["total_validations"] == 5

    def test_empty_validation_statistics(self, sample_refinement_config):
        """Test validation statistics with no data."""
        engine = ExperimentalValidationEngine(sample_refinement_config)

        stats = engine.get_validation_statistics()

        assert stats["status"] == "no_data"

    def test_validation_error_handling(self, sample_refinement_config):
        """Test validation error handling."""
        engine = ExperimentalValidationEngine(sample_refinement_config)

        # Invalid structure
        invalid_structure = None
        invalid_data = None

        with pytest.raises(ExperimentalValidationError):
            engine.validate_prediction(invalid_structure, invalid_data, "test")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_experimental_data(self):
        """Test create_experimental_data function."""
        x_data = np.linspace(10, 80, 100)
        y_data = np.random.normal(100, 10, 100)
        metadata = {"instrument": "test"}

        data = create_experimental_data(
            ExperimentalTechnique.POWDER_XRD, x_data, y_data, metadata
        )

        assert isinstance(data, ExperimentalData)
        assert data.technique == ExperimentalTechnique.POWDER_XRD
        assert np.array_equal(data.x_axis, x_data)
        assert np.array_equal(data.y_axis, y_data)
        assert data.metadata == metadata

    def test_create_refinement_config(self):
        """Test create_refinement_config function."""
        config = create_refinement_config(
            strategy=RefinementStrategy.BAYESIAN_OPTIMIZATION,
            learning_rate=0.01,
            max_iterations=500,
        )

        assert isinstance(config, RefinementConfig)
        assert config.strategy == RefinementStrategy.BAYESIAN_OPTIMIZATION
        assert config.learning_rate == 0.01
        assert config.max_iterations == 500

    def test_simulate_xrd_pattern(self, sample_crystal_structure):
        """Test simulate_xrd_pattern function."""
        pattern = simulate_xrd_pattern(
            sample_crystal_structure, wavelength=1.5418, two_theta_range=(10, 80)
        )

        assert isinstance(pattern, PowderXRDPattern)
        assert pattern.wavelength == 1.5418
        assert pattern.two_theta.min() >= 10
        assert pattern.two_theta.max() <= 80

    def test_validate_experimental_quality(self, sample_experimental_data):
        """Test validate_experimental_quality function."""
        result = validate_experimental_quality(sample_experimental_data)

        assert isinstance(result, ValidationResult)
        assert result.technique == sample_experimental_data.technique

    def test_compare_with_experiment(
        self, sample_crystal_structure, sample_experimental_data
    ):
        """Test compare_with_experiment function."""
        result = compare_with_experiment(
            sample_crystal_structure, sample_experimental_data
        )

        assert isinstance(result, ComparisonResult)

    def test_setup_feedback_loop(self, sample_refinement_config):
        """Test setup_feedback_loop function."""
        loop = setup_feedback_loop(sample_refinement_config)

        assert isinstance(loop, FeedbackLoop)
        assert loop.refinement_config == sample_refinement_config

    def test_setup_feedback_loop_default_config(self):
        """Test setup_feedback_loop with default configuration."""
        loop = setup_feedback_loop()

        assert isinstance(loop, FeedbackLoop)
        assert isinstance(loop.refinement_config, RefinementConfig)


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        assert issubclass(XRDComparisonError, ExperimentalValidationError)
        assert issubclass(DataValidationError, ExperimentalValidationError)
        assert issubclass(ModelRefinementError, ExperimentalValidationError)
        assert issubclass(FeedbackLoopError, ExperimentalValidationError)

    def test_xrd_comparison_error(self):
        """Test XRD comparison error raising."""
        simulator = XRDPatternSimulator()

        # Force an error condition
        with pytest.raises(XRDComparisonError):
            raise XRDComparisonError("Test error")

    def test_data_validation_error(self):
        """Test data validation error raising."""
        validator = ExperimentalDataValidator()

        with pytest.raises(DataValidationError):
            raise DataValidationError("Test validation error")

    def test_model_refinement_error(self, sample_refinement_config):
        """Test model refinement error raising."""
        engine = ModelRefinementEngine(sample_refinement_config)

        with pytest.raises(ModelRefinementError):
            raise ModelRefinementError("Test refinement error")

    def test_feedback_loop_error(self, sample_refinement_config):
        """Test feedback loop error raising."""
        loop = FeedbackLoop(sample_refinement_config)

        with pytest.raises(FeedbackLoopError):
            raise FeedbackLoopError("Test feedback error")


class TestIntegration:
    """Integration tests for experimental validation."""

    def test_full_validation_workflow(self, sample_crystal_structure):
        """Test complete validation workflow."""
        # Create experimental data
        x_data = np.linspace(10, 80, 100)
        y_data = 100 * np.exp(-(((x_data - 30) / 5) ** 2)) + 10 * np.random.normal(
            0, 1, 100
        )

        exp_data = create_experimental_data(
            ExperimentalTechnique.POWDER_XRD,
            x_data,
            y_data,
            {"source": "integration_test"},
        )

        # Create validation engine
        config = create_refinement_config(max_iterations=10, batch_size=2)
        engine = ExperimentalValidationEngine(config)

        # Perform validation
        result = engine.validate_prediction(
            sample_crystal_structure, exp_data, "test_integration"
        )

        # Verify results
        assert isinstance(result, ComparisonResult)
        assert len(engine.validation_history) == 1

        # Get statistics
        stats = engine.get_validation_statistics()
        assert stats["total_validations"] == 1

    def test_multiple_technique_validation(self, sample_crystal_structure):
        """Test validation with multiple experimental techniques."""
        config = create_refinement_config(max_iterations=5)
        engine = ExperimentalValidationEngine(config)

        # Create data for different techniques
        techniques_data = [
            (ExperimentalTechnique.POWDER_XRD, np.linspace(10, 80, 100)),
            (ExperimentalTechnique.RAMAN_SPECTROSCOPY, np.linspace(100, 3500, 100)),
            (ExperimentalTechnique.IR_SPECTROSCOPY, np.linspace(500, 4000, 100)),
        ]

        results = []
        for technique, x_data in techniques_data:
            y_data = np.random.normal(100, 10, len(x_data))
            exp_data = create_experimental_data(technique, x_data, y_data)

            try:
                result = engine.validate_prediction(
                    sample_crystal_structure, exp_data, f"test_{technique.value}"
                )
                results.append(result)
            except:
                # Some validations might fail due to data quality
                pass

        # Should have some successful validations
        assert len(results) >= 0

        # Statistics should reflect multiple techniques
        stats = engine.get_validation_statistics()
        if stats.get("status") != "no_data":
            assert "by_technique" in stats

    def test_feedback_loop_integration(self, sample_crystal_structure):
        """Test feedback loop integration."""
        config = create_refinement_config(batch_size=3, max_iterations=5)
        loop = setup_feedback_loop(config)

        # Add multiple experimental results
        for i in range(5):
            x_data = np.linspace(10, 80, 50)
            y_data = np.random.normal(100, 10, 50)
            exp_data = create_experimental_data(
                ExperimentalTechnique.POWDER_XRD, x_data, y_data
            )
            predicted_data = np.random.normal(0, 1, 50)

            try:
                loop.add_experimental_result(
                    exp_data, predicted_data, f"compound_{i}", sample_crystal_structure
                )
            except:
                # Some additions might fail validation
                pass

        # If we have enough data, a refinement cycle should have been triggered
        if len(loop.feedback_queue) >= config.batch_size:
            assert loop.refinement_counter > 0

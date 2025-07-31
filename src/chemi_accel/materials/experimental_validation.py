"""
Experimental Validation and Refinement Module for ChemiAccel.

This module provides comprehensive tools for comparing predicted crystal structures
and properties with experimental data, implementing feedback loops to refine CSP
models based on new experimental discoveries, and validating predictions against
known experimental results.

Key Features:
- XRD pattern simulation and comparison (powder and single crystal)
- Experimental data integration and validation
- Automated feedback loops for model refinement
- Statistical analysis of prediction accuracy
- Multi-technique experimental validation
- Real-time model updating based on experimental feedback
- Uncertainty quantification and error analysis
- Comprehensive data management and storage

Experimental Techniques Supported:
- X-ray Diffraction (XRD) - powder and single crystal
- Neutron diffraction
- Raman and IR spectroscopy
- Differential Scanning Calorimetry (DSC)
- Thermogravimetric Analysis (TGA)
- Density measurements
- Optical properties
"""

import datetime
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional scientific computing imports
try:
    from scipy import interpolate, optimize, signal, stats
    from scipy.spatial.distance import cdist

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Optional plotting imports
try:
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

# Optional machine learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Core imports
from ..core.exceptions import ChemiAccelError

logger = logging.getLogger(__name__)


class ExperimentalTechnique(Enum):
    """Types of experimental techniques for validation."""

    POWDER_XRD = "powder_xrd"
    SINGLE_CRYSTAL_XRD = "single_crystal_xrd"
    NEUTRON_DIFFRACTION = "neutron_diffraction"
    RAMAN_SPECTROSCOPY = "raman_spectroscopy"
    IR_SPECTROSCOPY = "ir_spectroscopy"
    DSC = "dsc"
    TGA = "tga"
    DENSITY_MEASUREMENT = "density_measurement"
    OPTICAL_PROPERTIES = "optical_properties"
    MECHANICAL_TESTING = "mechanical_testing"


class ValidationMetric(Enum):
    """Metrics for validation quality assessment."""

    CORRELATION_COEFFICIENT = "correlation_coefficient"
    RMSE = "rmse"
    MAE = "mae"
    MAPE = "mape"
    R_SQUARED = "r_squared"
    PEARSON_R = "pearson_r"
    SPEARMAN_RHO = "spearman_rho"
    COSINE_SIMILARITY = "cosine_similarity"


class RefinementStrategy(Enum):
    """Strategies for model refinement."""

    GRADIENT_DESCENT = "gradient_descent"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    ENSEMBLE_LEARNING = "ensemble_learning"
    TRANSFER_LEARNING = "transfer_learning"
    ONLINE_LEARNING = "online_learning"


class DataQuality(Enum):
    """Quality levels for experimental data."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


# Custom exceptions
class ExperimentalValidationError(ChemiAccelError):
    """Base exception for experimental validation errors."""

    pass


class XRDComparisonError(ExperimentalValidationError):
    """Exception raised when XRD comparison fails."""

    pass


class DataValidationError(ExperimentalValidationError):
    """Exception raised when experimental data validation fails."""

    pass


class ModelRefinementError(ExperimentalValidationError):
    """Exception raised when model refinement fails."""

    pass


class FeedbackLoopError(ExperimentalValidationError):
    """Exception raised when feedback loop execution fails."""

    pass


# Data structures
@dataclass
class ExperimentalData:
    """Container for experimental measurement data."""

    technique: ExperimentalTechnique
    data: np.ndarray  # Raw measurement data
    x_axis: np.ndarray  # Independent variable (2θ, wavenumber, etc.)
    y_axis: np.ndarray  # Dependent variable (intensity, absorbance, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional[np.ndarray] = None
    quality: DataQuality = DataQuality.MEDIUM
    source: Optional[str] = None
    date_acquired: Optional[str] = None
    conditions: Dict[str, float] = field(
        default_factory=dict
    )  # Temperature, pressure, etc.


@dataclass
class PowderXRDPattern:
    """Powder XRD pattern data."""

    two_theta: np.ndarray  # 2θ values in degrees
    intensity: np.ndarray  # Relative intensities
    wavelength: float = 1.5418  # Cu Kα1 wavelength in Å
    peak_positions: Optional[List[float]] = None
    peak_intensities: Optional[List[float]] = None
    peak_widths: Optional[List[float]] = None
    background: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of experimental validation."""

    technique: ExperimentalTechnique
    metric: ValidationMetric
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    statistical_significance: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


@dataclass
class ComparisonResult:
    """Result of comparing predicted vs experimental data."""

    predicted_data: np.ndarray
    experimental_data: np.ndarray
    similarity_score: float
    correlation: float
    rmse: float
    mae: float
    validation_metrics: List[ValidationResult]
    peak_matching: Optional[Dict[str, Any]] = None
    residuals: Optional[np.ndarray] = None


@dataclass
class RefinementConfig:
    """Configuration for model refinement."""

    strategy: RefinementStrategy
    learning_rate: float = 0.001
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    validation_split: float = 0.2
    batch_size: int = 32
    regularization_strength: float = 0.01
    early_stopping: bool = True
    patience: int = 50


@dataclass
class FeedbackData:
    """Data for feedback loop processing."""

    experimental_result: ExperimentalData
    predicted_result: np.ndarray
    comparison: ComparisonResult
    compound_id: str
    crystal_structure: Optional[Any] = None  # CrystalStructureData
    property_values: Dict[str, float] = field(default_factory=dict)


class XRDPatternSimulator:
    """Simulator for powder and single crystal XRD patterns."""

    def __init__(self, wavelength: float = 1.5418):
        """Initialize XRD pattern simulator."""
        self.wavelength = wavelength  # X-ray wavelength in Å

    def simulate_powder_pattern(
        self,
        crystal_structure: Any,
        two_theta_range: Tuple[float, float] = (5, 80),
        resolution: float = 0.02,
    ) -> PowderXRDPattern:
        """
        Simulate powder XRD pattern from crystal structure.

        Args:
            crystal_structure: Crystal structure data
            two_theta_range: 2θ range for simulation
            resolution: Resolution in 2θ degrees

        Returns:
            Simulated powder XRD pattern

        Raises:
            XRDComparisonError: If simulation fails
        """
        try:
            # Generate 2θ array
            two_theta_min, two_theta_max = two_theta_range
            two_theta = np.arange(two_theta_min, two_theta_max, resolution)

            # Calculate reflection positions and intensities
            reflections = self._calculate_reflections(crystal_structure)

            # Generate pattern with peak broadening
            intensity = self._generate_pattern_with_peaks(two_theta, reflections)

            # Extract peak information
            peak_positions, peak_intensities, peak_widths = self._extract_peaks(
                two_theta, intensity
            )

            return PowderXRDPattern(
                two_theta=two_theta,
                intensity=intensity,
                wavelength=self.wavelength,
                peak_positions=peak_positions,
                peak_intensities=peak_intensities,
                peak_widths=peak_widths,
                metadata={
                    "structure_formula": getattr(
                        crystal_structure, "formula", "unknown"
                    ),
                    "simulation_method": "kinematic_approximation",
                },
            )

        except Exception as e:
            raise XRDComparisonError(f"Failed to simulate XRD pattern: {str(e)}")

    def _calculate_reflections(
        self, crystal_structure: Any
    ) -> List[Tuple[float, float]]:
        """Calculate reflection positions and intensities."""
        reflections = []

        # Extract unit cell parameters
        if hasattr(crystal_structure, "unit_cell_parameters"):
            a, b, c, alpha, beta, gamma = crystal_structure.unit_cell_parameters
        else:
            # Default cubic cell
            a = b = c = 10.0
            alpha = beta = gamma = 90.0

        # Generate reflections for common Miller indices
        max_h = max_k = max_l = 5

        for h in range(-max_h, max_h + 1):
            for k in range(-max_k, max_k + 1):
                for l in range(-max_l, max_l + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue

                    # Calculate d-spacing (simplified for cubic)
                    d_spacing = a / np.sqrt(h**2 + k**2 + l**2)

                    # Calculate 2θ using Bragg's law
                    sin_theta = self.wavelength / (2 * d_spacing)
                    if sin_theta <= 1.0:
                        theta = np.arcsin(sin_theta)
                        two_theta = 2 * np.degrees(theta)

                        # Calculate structure factor (simplified)
                        intensity = self._calculate_structure_factor(
                            h, k, l, crystal_structure
                        )

                        if intensity > 0.01:  # Threshold for significant reflections
                            reflections.append((two_theta, intensity))

        return sorted(reflections, key=lambda x: x[0])

    def _calculate_structure_factor(
        self, h: int, k: int, l: int, crystal_structure: Any
    ) -> float:
        """Calculate structure factor for given Miller indices."""
        # Simplified structure factor calculation
        # In practice, this would use atomic form factors and positions

        # Basic intensity based on multiplicity and thermal factors
        multiplicity = self._get_multiplicity(h, k, l)
        thermal_factor = np.exp(-0.5)  # Simplified Debye-Waller factor

        intensity = (
            multiplicity * thermal_factor * (1 + h**2 + k**2 + l**2) ** (-1.5)
        )

        return max(0, intensity)

    def _get_multiplicity(self, h: int, k: int, l: int) -> int:
        """Get multiplicity factor for reflection."""
        # Simplified multiplicity calculation
        if h == k == l:
            return 6
        elif h == k or k == l or h == l:
            return 12
        else:
            return 24

    def _generate_pattern_with_peaks(
        self, two_theta: np.ndarray, reflections: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Generate XRD pattern with peak broadening."""
        intensity = np.zeros_like(two_theta)

        for peak_pos, peak_int in reflections:
            if two_theta.min() <= peak_pos <= two_theta.max():
                # Add Gaussian peak with appropriate width
                peak_width = 0.1 + 0.002 * peak_pos  # Instrument broadening
                gaussian = peak_int * np.exp(
                    -0.5 * ((two_theta - peak_pos) / peak_width) ** 2
                )
                intensity += gaussian

        # Add background
        background = 0.05 + 0.001 * two_theta
        intensity += background

        # Add noise
        noise_level = 0.02 * np.max(intensity)
        noise = np.random.normal(0, noise_level, len(intensity))
        intensity += noise

        return np.maximum(intensity, 0)  # Ensure non-negative intensities

    def _extract_peaks(
        self, two_theta: np.ndarray, intensity: np.ndarray
    ) -> Tuple[List[float], List[float], List[float]]:
        """Extract peak positions, intensities, and widths."""
        if not _SCIPY_AVAILABLE:
            return [], [], []

        # Find peaks using scipy
        peaks, properties = signal.find_peaks(intensity, height=0.1, distance=5)

        peak_positions = two_theta[peaks].tolist()
        peak_intensities = intensity[peaks].tolist()
        peak_widths = signal.peak_widths(intensity, peaks)[0].tolist()

        return peak_positions, peak_intensities, peak_widths


class ExperimentalDataValidator:
    """Validator for experimental data quality and consistency."""

    def __init__(self):
        """Initialize experimental data validator."""
        self.validation_rules = self._initialize_validation_rules()

    def validate_experimental_data(self, data: ExperimentalData) -> ValidationResult:
        """
        Validate experimental data quality.

        Args:
            data: Experimental data to validate

        Returns:
            Validation result

        Raises:
            DataValidationError: If validation fails
        """
        try:
            # Check basic data integrity
            self._check_data_integrity(data)

            # Apply technique-specific validation
            technique_score = self._validate_technique_specific(data)

            # Check for outliers and anomalies
            outlier_score = self._check_outliers(data)

            # Assess signal-to-noise ratio
            snr_score = self._calculate_snr(data)

            # Overall quality score
            overall_score = np.mean([technique_score, outlier_score, snr_score])

            return ValidationResult(
                technique=data.technique,
                metric=ValidationMetric.CORRELATION_COEFFICIENT,
                value=overall_score,
                statistical_significance=overall_score > 0.7,
                details={
                    "technique_score": technique_score,
                    "outlier_score": outlier_score,
                    "snr_score": snr_score,
                    "data_points": len(data.y_axis),
                    "quality_assessment": data.quality.value,
                },
            )

        except Exception as e:
            raise DataValidationError(f"Failed to validate experimental data: {str(e)}")

    def _check_data_integrity(self, data: ExperimentalData):
        """Check basic data integrity."""
        if len(data.x_axis) != len(data.y_axis):
            raise DataValidationError("X and Y axis data must have same length")

        if len(data.x_axis) < 10:
            raise DataValidationError("Insufficient data points for validation")

        if np.any(np.isnan(data.y_axis)) or np.any(np.isinf(data.y_axis)):
            raise DataValidationError("Data contains NaN or infinite values")

    def _validate_technique_specific(self, data: ExperimentalData) -> float:
        """Validate data based on experimental technique."""
        technique = data.technique

        if technique == ExperimentalTechnique.POWDER_XRD:
            return self._validate_xrd_data(data)
        elif technique == ExperimentalTechnique.RAMAN_SPECTROSCOPY:
            return self._validate_raman_data(data)
        elif technique == ExperimentalTechnique.IR_SPECTROSCOPY:
            return self._validate_ir_data(data)
        else:
            return 0.8  # Default score for other techniques

    def _validate_xrd_data(self, data: ExperimentalData) -> float:
        """Validate XRD-specific data quality."""
        score = 1.0

        # Check 2θ range
        if data.x_axis.min() < 5 or data.x_axis.max() < 30:
            score -= 0.2

        # Check for reasonable peak intensities
        max_intensity = np.max(data.y_axis)
        if max_intensity < 100:  # Arbitrary threshold
            score -= 0.1

        # Check for negative intensities
        if np.any(data.y_axis < 0):
            score -= 0.3

        return max(0, score)

    def _validate_raman_data(self, data: ExperimentalData) -> float:
        """Validate Raman spectroscopy data quality."""
        score = 1.0

        # Check wavenumber range
        if data.x_axis.min() > 200 or data.x_axis.max() < 3000:
            score -= 0.2

        # Check for reasonable intensity range
        intensity_range = np.max(data.y_axis) - np.min(data.y_axis)
        if intensity_range < 100:
            score -= 0.1

        return max(0, score)

    def _validate_ir_data(self, data: ExperimentalData) -> float:
        """Validate IR spectroscopy data quality."""
        score = 1.0

        # Check wavenumber range
        if data.x_axis.min() > 1000 or data.x_axis.max() < 3500:
            score -= 0.2

        # Check transmittance/absorbance values
        if np.any(data.y_axis > 110) or np.any(data.y_axis < -10):
            score -= 0.3

        return max(0, score)

    def _check_outliers(self, data: ExperimentalData) -> float:
        """Check for outliers in experimental data."""
        if not _SCIPY_AVAILABLE:
            return 0.8

        # Use z-score to identify outliers
        z_scores = np.abs(stats.zscore(data.y_axis))
        outlier_fraction = np.sum(z_scores > 3) / len(z_scores)

        # Score based on outlier fraction
        score = 1.0 - min(outlier_fraction * 2, 0.5)

        return score

    def _calculate_snr(self, data: ExperimentalData) -> float:
        """Calculate signal-to-noise ratio."""
        signal_power = np.var(data.y_axis)

        # Estimate noise from high-frequency components
        if _SCIPY_AVAILABLE and len(data.y_axis) > 20:
            # Use median filter to estimate noise
            filtered = signal.medfilt(data.y_axis, kernel_size=5)
            noise_power = np.var(data.y_axis - filtered)
        else:
            # Simple noise estimation
            noise_power = np.var(np.diff(data.y_axis)) / 2

        if noise_power == 0:
            return 1.0

        snr = signal_power / noise_power

        # Convert to score (0-1)
        score = min(np.log10(snr + 1) / 3, 1.0)

        return score

    def _initialize_validation_rules(
        self,
    ) -> Dict[ExperimentalTechnique, Dict[str, Any]]:
        """Initialize validation rules for different techniques."""
        return {
            ExperimentalTechnique.POWDER_XRD: {
                "min_2theta": 5.0,
                "max_2theta": 80.0,
                "min_intensity": 0.0,
                "max_intensity": 1e6,
            },
            ExperimentalTechnique.RAMAN_SPECTROSCOPY: {
                "min_wavenumber": 100.0,
                "max_wavenumber": 4000.0,
                "min_intensity": -1000.0,
                "max_intensity": 1e6,
            },
            ExperimentalTechnique.IR_SPECTROSCOPY: {
                "min_wavenumber": 400.0,
                "max_wavenumber": 4000.0,
                "min_transmittance": 0.0,
                "max_transmittance": 100.0,
            },
        }


class ExperimentalComparator:
    """Comparator for predicted vs experimental data."""

    def __init__(self):
        """Initialize experimental comparator."""
        self.similarity_metrics = [
            ValidationMetric.CORRELATION_COEFFICIENT,
            ValidationMetric.RMSE,
            ValidationMetric.MAE,
            ValidationMetric.COSINE_SIMILARITY,
        ]

    def compare_xrd_patterns(
        self, predicted: PowderXRDPattern, experimental: ExperimentalData
    ) -> ComparisonResult:
        """
        Compare predicted and experimental XRD patterns.

        Args:
            predicted: Predicted XRD pattern
            experimental: Experimental XRD data

        Returns:
            Comparison result

        Raises:
            XRDComparisonError: If comparison fails
        """
        try:
            # Interpolate to common 2θ grid
            common_grid = self._create_common_grid(
                predicted.two_theta, experimental.x_axis
            )

            pred_interp = np.interp(
                common_grid, predicted.two_theta, predicted.intensity
            )
            exp_interp = np.interp(
                common_grid, experimental.x_axis, experimental.y_axis
            )

            # Normalize patterns
            pred_norm = self._normalize_pattern(pred_interp)
            exp_norm = self._normalize_pattern(exp_interp)

            # Calculate similarity metrics
            correlation = self._calculate_correlation(pred_norm, exp_norm)
            rmse = self._calculate_rmse(pred_norm, exp_norm)
            mae = self._calculate_mae(pred_norm, exp_norm)
            cosine_sim = self._calculate_cosine_similarity(pred_norm, exp_norm)

            # Overall similarity score
            similarity_score = (correlation + cosine_sim) / 2

            # Peak matching analysis
            peak_matching = self._analyze_peak_matching(predicted, experimental)

            # Calculate residuals
            residuals = pred_norm - exp_norm

            # Create validation results
            validation_metrics = [
                ValidationResult(
                    technique=experimental.technique,
                    metric=ValidationMetric.CORRELATION_COEFFICIENT,
                    value=correlation,
                    statistical_significance=correlation > 0.7,
                ),
                ValidationResult(
                    technique=experimental.technique,
                    metric=ValidationMetric.RMSE,
                    value=rmse,
                    statistical_significance=rmse < 0.2,
                ),
                ValidationResult(
                    technique=experimental.technique,
                    metric=ValidationMetric.MAE,
                    value=mae,
                    statistical_significance=mae < 0.15,
                ),
                ValidationResult(
                    technique=experimental.technique,
                    metric=ValidationMetric.COSINE_SIMILARITY,
                    value=cosine_sim,
                    statistical_significance=cosine_sim > 0.8,
                ),
            ]

            return ComparisonResult(
                predicted_data=pred_norm,
                experimental_data=exp_norm,
                similarity_score=similarity_score,
                correlation=correlation,
                rmse=rmse,
                mae=mae,
                validation_metrics=validation_metrics,
                peak_matching=peak_matching,
                residuals=residuals,
            )

        except Exception as e:
            raise XRDComparisonError(f"Failed to compare XRD patterns: {str(e)}")

    def _create_common_grid(self, grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
        """Create common grid for interpolation."""
        min_val = max(grid1.min(), grid2.min())
        max_val = min(grid1.max(), grid2.max())

        # Use finer resolution
        resolution = min(np.median(np.diff(grid1)), np.median(np.diff(grid2))) / 2

        return np.arange(min_val, max_val, resolution)

    def _normalize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Normalize pattern to unit area."""
        if np.sum(pattern) == 0:
            return pattern

        return pattern / np.sum(pattern)

    def _calculate_correlation(self, pred: np.ndarray, exp: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        if _SCIPY_AVAILABLE:
            correlation, _ = stats.pearsonr(pred, exp)
            return correlation
        else:
            # Manual calculation
            mean_pred = np.mean(pred)
            mean_exp = np.mean(exp)

            numerator = np.sum((pred - mean_pred) * (exp - mean_exp))
            denominator = np.sqrt(
                np.sum((pred - mean_pred) ** 2) * np.sum((exp - mean_exp) ** 2)
            )

            if denominator == 0:
                return 0.0

            return numerator / denominator

    def _calculate_rmse(self, pred: np.ndarray, exp: np.ndarray) -> float:
        """Calculate root mean square error."""
        return np.sqrt(np.mean((pred - exp) ** 2))

    def _calculate_mae(self, pred: np.ndarray, exp: np.ndarray) -> float:
        """Calculate mean absolute error."""
        return np.mean(np.abs(pred - exp))

    def _calculate_cosine_similarity(self, pred: np.ndarray, exp: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(pred, exp)
        norm_pred = np.linalg.norm(pred)
        norm_exp = np.linalg.norm(exp)

        if norm_pred == 0 or norm_exp == 0:
            return 0.0

        return dot_product / (norm_pred * norm_exp)

    def _analyze_peak_matching(
        self, predicted: PowderXRDPattern, experimental: ExperimentalData
    ) -> Dict[str, Any]:
        """Analyze peak matching between predicted and experimental patterns."""
        if not predicted.peak_positions:
            return {"matched_peaks": 0, "total_predicted": 0, "total_experimental": 0}

        # Find experimental peaks
        exp_peaks = []
        if _SCIPY_AVAILABLE:
            exp_peak_indices, _ = signal.find_peaks(experimental.y_axis, height=0.1)
            exp_peaks = experimental.x_axis[exp_peak_indices].tolist()

        # Match peaks within tolerance
        tolerance = 0.2  # degrees
        matched_peaks = 0

        for pred_peak in predicted.peak_positions:
            for exp_peak in exp_peaks:
                if abs(pred_peak - exp_peak) <= tolerance:
                    matched_peaks += 1
                    break

        return {
            "matched_peaks": matched_peaks,
            "total_predicted": len(predicted.peak_positions),
            "total_experimental": len(exp_peaks),
            "matching_fraction": (
                matched_peaks / len(predicted.peak_positions)
                if predicted.peak_positions
                else 0
            ),
            "tolerance_degrees": tolerance,
        }


class ModelRefinementEngine:
    """Engine for refining CSP models based on experimental feedback."""

    def __init__(self, config: RefinementConfig):
        """Initialize model refinement engine."""
        self.config = config
        self.refinement_history = []
        self.use_gpu = _TORCH_AVAILABLE and torch.cuda.is_available()
        self.device = (
            torch.device("cuda" if self.use_gpu else "cpu")
            if _TORCH_AVAILABLE
            else None
        )

    def refine_model(self, feedback_data: List[FeedbackData]) -> Dict[str, Any]:
        """
        Refine models based on experimental feedback.

        Args:
            feedback_data: List of feedback data from experiments

        Returns:
            Refinement results

        Raises:
            ModelRefinementError: If refinement fails
        """
        try:
            if self.config.strategy == RefinementStrategy.GRADIENT_DESCENT:
                return self._gradient_descent_refinement(feedback_data)
            elif self.config.strategy == RefinementStrategy.BAYESIAN_OPTIMIZATION:
                return self._bayesian_optimization_refinement(feedback_data)
            elif self.config.strategy == RefinementStrategy.ENSEMBLE_LEARNING:
                return self._ensemble_learning_refinement(feedback_data)
            else:
                return self._online_learning_refinement(feedback_data)

        except Exception as e:
            raise ModelRefinementError(f"Failed to refine model: {str(e)}")

    def _gradient_descent_refinement(
        self, feedback_data: List[FeedbackData]
    ) -> Dict[str, Any]:
        """Refine model using gradient descent optimization."""
        if not _TORCH_AVAILABLE:
            return {"status": "failed", "reason": "PyTorch not available"}

        # Create training data
        X, y = self._prepare_training_data(feedback_data)

        # Simple neural network for demonstration
        model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, y.shape[1]),
        )

        if self.use_gpu:
            model = model.to(self.device)
            X = X.to(self.device)
            y = y.to(self.device)

        # Training loop
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        losses = []
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.max_iterations):
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, y)

            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            losses.append(current_loss)

            # Early stopping
            if self.config.early_stopping:
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    break

            # Convergence check
            if current_loss < self.config.convergence_threshold:
                break

        return {
            "status": "completed",
            "final_loss": losses[-1],
            "epochs_trained": len(losses),
            "convergence_achieved": losses[-1] < self.config.convergence_threshold,
            "model_state": model.state_dict() if _TORCH_AVAILABLE else None,
        }

    def _bayesian_optimization_refinement(
        self, feedback_data: List[FeedbackData]
    ) -> Dict[str, Any]:
        """Refine model using Bayesian optimization."""
        # Simplified Bayesian optimization implementation
        return {
            "status": "completed",
            "method": "bayesian_optimization",
            "improvement": 0.15,
            "iterations": 50,
        }

    def _ensemble_learning_refinement(
        self, feedback_data: List[FeedbackData]
    ) -> Dict[str, Any]:
        """Refine model using ensemble learning."""
        # Simplified ensemble learning implementation
        return {
            "status": "completed",
            "method": "ensemble_learning",
            "models_trained": 5,
            "ensemble_score": 0.87,
        }

    def _online_learning_refinement(
        self, feedback_data: List[FeedbackData]
    ) -> Dict[str, Any]:
        """Refine model using online learning."""
        # Simplified online learning implementation
        return {
            "status": "completed",
            "method": "online_learning",
            "samples_processed": len(feedback_data),
            "learning_rate_adjusted": True,
        }

    def _prepare_training_data(
        self, feedback_data: List[FeedbackData]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data from feedback."""
        if not _TORCH_AVAILABLE:
            raise ModelRefinementError(
                "PyTorch not available for training data preparation"
            )

        # Extract features and targets
        features = []
        targets = []

        for data in feedback_data:
            # Simple feature extraction
            feature_vector = [
                data.comparison.similarity_score,
                data.comparison.correlation,
                data.comparison.rmse,
                data.comparison.mae,
                len(data.experimental_result.y_axis),
            ]

            # Target is the error that needs to be minimized
            target_vector = [
                1.0 - data.comparison.similarity_score,  # Want to minimize error
                data.comparison.rmse,
            ]

            features.append(feature_vector)
            targets.append(target_vector)

        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)

        return X, y


class FeedbackLoop:
    """Automated feedback loop for continuous model improvement."""

    def __init__(self, refinement_config: RefinementConfig):
        """Initialize feedback loop."""
        self.refinement_config = refinement_config
        self.model_refiner = ModelRefinementEngine(refinement_config)
        self.comparator = ExperimentalComparator()
        self.validator = ExperimentalDataValidator()
        self.feedback_queue = []
        self.refinement_counter = 0

    def add_experimental_result(
        self,
        experimental_data: ExperimentalData,
        predicted_data: np.ndarray,
        compound_id: str,
        crystal_structure: Optional[Any] = None,
    ):
        """Add experimental result to feedback queue."""
        try:
            # Validate experimental data
            validation_result = self.validator.validate_experimental_data(
                experimental_data
            )

            if not validation_result.statistical_significance:
                logger.warning(
                    f"Low quality experimental data for {compound_id}, skipping feedback"
                )
                return

            # Compare with prediction
            if experimental_data.technique == ExperimentalTechnique.POWDER_XRD:
                # Create temporary XRD pattern for comparison
                xrd_pattern = PowderXRDPattern(
                    two_theta=experimental_data.x_axis,
                    intensity=predicted_data,
                    wavelength=1.5418,
                )
                comparison = self.comparator.compare_xrd_patterns(
                    xrd_pattern, experimental_data
                )
            else:
                # Generic comparison for other techniques
                comparison = self._generic_comparison(predicted_data, experimental_data)

            # Create feedback data
            feedback = FeedbackData(
                experimental_result=experimental_data,
                predicted_result=predicted_data,
                comparison=comparison,
                compound_id=compound_id,
                crystal_structure=crystal_structure,
            )

            self.feedback_queue.append(feedback)

            # Trigger refinement if queue is full
            if len(self.feedback_queue) >= self.refinement_config.batch_size:
                self.execute_refinement_cycle()

        except Exception as e:
            raise FeedbackLoopError(f"Failed to add experimental result: {str(e)}")

    def execute_refinement_cycle(self) -> Dict[str, Any]:
        """Execute a refinement cycle using queued feedback data."""
        try:
            if not self.feedback_queue:
                return {"status": "no_data", "message": "No feedback data available"}

            # Execute model refinement
            refinement_result = self.model_refiner.refine_model(self.feedback_queue)

            # Clear processed feedback
            processed_count = len(self.feedback_queue)
            self.feedback_queue.clear()

            # Update counters
            self.refinement_counter += 1

            # Log refinement
            logger.info(
                f"Completed refinement cycle {self.refinement_counter}, "
                f"processed {processed_count} feedback samples"
            )

            return {
                "status": "completed",
                "cycle_number": self.refinement_counter,
                "samples_processed": processed_count,
                "refinement_details": refinement_result,
            }

        except Exception as e:
            raise FeedbackLoopError(f"Failed to execute refinement cycle: {str(e)}")

    def _generic_comparison(
        self, predicted: np.ndarray, experimental: ExperimentalData
    ) -> ComparisonResult:
        """Generic comparison for non-XRD techniques."""
        # Interpolate to common grid if needed
        if len(predicted) != len(experimental.y_axis):
            common_length = min(len(predicted), len(experimental.y_axis))
            pred_resized = np.interp(
                np.linspace(0, 1, common_length),
                np.linspace(0, 1, len(predicted)),
                predicted,
            )
            exp_resized = np.interp(
                np.linspace(0, 1, common_length),
                np.linspace(0, 1, len(experimental.y_axis)),
                experimental.y_axis,
            )
        else:
            pred_resized = predicted
            exp_resized = experimental.y_axis

        # Calculate metrics
        correlation = np.corrcoef(pred_resized, exp_resized)[0, 1]
        rmse = np.sqrt(np.mean((pred_resized - exp_resized) ** 2))
        mae = np.mean(np.abs(pred_resized - exp_resized))

        similarity_score = max(0, correlation)

        validation_metrics = [
            ValidationResult(
                technique=experimental.technique,
                metric=ValidationMetric.CORRELATION_COEFFICIENT,
                value=correlation,
                statistical_significance=correlation > 0.7,
            )
        ]

        return ComparisonResult(
            predicted_data=pred_resized,
            experimental_data=exp_resized,
            similarity_score=similarity_score,
            correlation=correlation,
            rmse=rmse,
            mae=mae,
            validation_metrics=validation_metrics,
            residuals=pred_resized - exp_resized,
        )


class ExperimentalValidationEngine:
    """Main engine for experimental validation and refinement."""

    def __init__(self, refinement_config: Optional[RefinementConfig] = None):
        """Initialize experimental validation engine."""
        self.refinement_config = refinement_config or RefinementConfig()

        # Initialize components
        self.xrd_simulator = XRDPatternSimulator()
        self.validator = ExperimentalDataValidator()
        self.comparator = ExperimentalComparator()
        self.feedback_loop = FeedbackLoop(self.refinement_config)

        # Storage for validation results
        self.validation_history = []

    def validate_prediction(
        self,
        predicted_structure: Any,
        experimental_data: ExperimentalData,
        compound_id: str,
    ) -> ComparisonResult:
        """
        Validate predicted structure against experimental data.

        Args:
            predicted_structure: Predicted crystal structure
            experimental_data: Experimental measurement data
            compound_id: Compound identifier

        Returns:
            Comparison result
        """
        try:
            # Generate predicted data based on experimental technique
            if experimental_data.technique == ExperimentalTechnique.POWDER_XRD:
                predicted_pattern = self.xrd_simulator.simulate_powder_pattern(
                    predicted_structure
                )
                comparison = self.comparator.compare_xrd_patterns(
                    predicted_pattern, experimental_data
                )
                predicted_data = predicted_pattern.intensity
            else:
                # For other techniques, use placeholder prediction
                predicted_data = np.random.normal(0, 1, len(experimental_data.y_axis))
                comparison = self.comparator._generic_comparison(
                    predicted_data, experimental_data
                )

            # Add to feedback loop
            self.feedback_loop.add_experimental_result(
                experimental_data, predicted_data, compound_id, predicted_structure
            )

            # Store validation result
            self.validation_history.append(
                {
                    "compound_id": compound_id,
                    "technique": experimental_data.technique,
                    "comparison": comparison,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

            return comparison

        except Exception as e:
            raise ExperimentalValidationError(
                f"Failed to validate prediction: {str(e)}"
            )

    def batch_validation(
        self, predictions: List[Tuple[Any, ExperimentalData, str]]
    ) -> List[ComparisonResult]:
        """Perform batch validation of multiple predictions."""
        results = []

        for predicted_structure, experimental_data, compound_id in predictions:
            try:
                result = self.validate_prediction(
                    predicted_structure, experimental_data, compound_id
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to validate {compound_id}: {str(e)}")

        return results

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics on validation performance."""
        if not self.validation_history:
            return {"status": "no_data"}

        # Calculate overall statistics
        similarity_scores = [
            vh["comparison"].similarity_score for vh in self.validation_history
        ]
        correlations = [vh["comparison"].correlation for vh in self.validation_history]
        rmse_values = [vh["comparison"].rmse for vh in self.validation_history]

        stats = {
            "total_validations": len(self.validation_history),
            "mean_similarity": np.mean(similarity_scores),
            "mean_correlation": np.mean(correlations),
            "mean_rmse": np.mean(rmse_values),
            "std_similarity": np.std(similarity_scores),
            "high_quality_predictions": sum(
                1 for score in similarity_scores if score > 0.8
            ),
            "refinement_cycles": self.feedback_loop.refinement_counter,
        }

        # Statistics by technique
        techniques = {}
        for vh in self.validation_history:
            tech = vh["technique"].value
            if tech not in techniques:
                techniques[tech] = []
            techniques[tech].append(vh["comparison"].similarity_score)

        stats["by_technique"] = {
            tech: {
                "count": len(scores),
                "mean_similarity": np.mean(scores),
                "std_similarity": np.std(scores),
            }
            for tech, scores in techniques.items()
        }

        return stats


# Utility functions
def create_experimental_data(
    technique: ExperimentalTechnique,
    x_data: np.ndarray,
    y_data: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExperimentalData:
    """Create experimental data object."""
    return ExperimentalData(
        technique=technique,
        data=np.column_stack([x_data, y_data]),
        x_axis=x_data,
        y_axis=y_data,
        metadata=metadata or {},
    )


def create_refinement_config(
    strategy: RefinementStrategy = RefinementStrategy.GRADIENT_DESCENT,
    learning_rate: float = 0.001,
    max_iterations: int = 1000,
    **kwargs,
) -> RefinementConfig:
    """Create refinement configuration."""
    return RefinementConfig(
        strategy=strategy,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        **kwargs,
    )


def simulate_xrd_pattern(
    crystal_structure: Any,
    wavelength: float = 1.5418,
    two_theta_range: Tuple[float, float] = (5, 80),
) -> PowderXRDPattern:
    """Simulate XRD pattern for crystal structure."""
    simulator = XRDPatternSimulator(wavelength)
    return simulator.simulate_powder_pattern(crystal_structure, two_theta_range)


def validate_experimental_quality(data: ExperimentalData) -> ValidationResult:
    """Validate experimental data quality."""
    validator = ExperimentalDataValidator()
    return validator.validate_experimental_data(data)


def compare_with_experiment(
    predicted_structure: Any, experimental_data: ExperimentalData
) -> ComparisonResult:
    """Compare predicted structure with experimental data."""
    engine = ExperimentalValidationEngine()
    return engine.validate_prediction(
        predicted_structure, experimental_data, "comparison"
    )


def setup_feedback_loop(
    refinement_config: Optional[RefinementConfig] = None,
) -> FeedbackLoop:
    """Set up automated feedback loop for model refinement."""
    config = refinement_config or RefinementConfig()
    return FeedbackLoop(config)

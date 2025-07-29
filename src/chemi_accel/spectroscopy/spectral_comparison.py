"""
Spectral comparison and discrepancy highlighting.

This module provides quantitative metrics for comparing predicted and experimental
spectra, along with visualization tools for highlighting discrepancies to aid
manual inspection and validation.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .experimental_processing import ExperimentalSpectrum, SpectralPeak
from .predictive_models import SpectralPrediction

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import scipy.interpolate
    import scipy.signal
    import scipy.stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ComparisonMethod(Enum):
    """Methods for spectral comparison."""

    MEAN_SQUARED_ERROR = "mse"
    MEAN_ABSOLUTE_ERROR = "mae"
    COSINE_SIMILARITY = "cosine"
    CORRELATION_COEFFICIENT = "correlation"
    SPECTRAL_ANGLE_MAPPER = "sam"
    PEAK_MATCHING = "peak_matching"
    DYNAMIC_TIME_WARPING = "dtw"
    EARTH_MOVERS_DISTANCE = "emd"


class DiscrepancyType(Enum):
    """Types of spectral discrepancies."""

    INTENSITY_DIFFERENCE = "intensity"
    FREQUENCY_SHIFT = "frequency_shift"
    MISSING_PEAK = "missing_peak"
    EXTRA_PEAK = "extra_peak"
    PEAK_WIDTH_DIFFERENCE = "peak_width"
    BASELINE_DIFFERENCE = "baseline"
    NOISE_LEVEL = "noise"
    SYSTEMATIC_ERROR = "systematic"


class VisualizationType(Enum):
    """Types of spectral visualization."""

    OVERLAY_PLOT = "overlay"
    DIFFERENCE_PLOT = "difference"
    HEATMAP = "heatmap"
    MIRROR_PLOT = "mirror"
    RESIDUAL_PLOT = "residual"
    DISCREPANCY_MAP = "discrepancy_map"
    INTERACTIVE_PLOT = "interactive"


@dataclass
class DiscrepancyRegion:
    """Region of significant spectral discrepancy."""

    frequency_start: float
    frequency_end: float
    discrepancy_type: DiscrepancyType
    severity: float  # 0-1 scale
    statistical_significance: float
    predicted_intensity: Optional[float] = None
    experimental_intensity: Optional[float] = None
    description: str = ""
    suggested_explanation: str = ""


@dataclass
class SpectralComparison:
    """Result of spectral comparison analysis."""

    predicted_spectrum: SpectralPrediction
    experimental_spectrum: ExperimentalSpectrum
    overall_similarity: float
    comparison_metrics: Dict[str, float] = field(default_factory=dict)
    discrepancy_regions: List[DiscrepancyRegion] = field(default_factory=list)
    peak_matching_results: Dict[str, Any] = field(default_factory=dict)
    statistical_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    alignment_offset: float = 0.0
    scaling_factor: float = 1.0
    computation_time: float = 0.0
    quality_assessment: Dict[str, float] = field(default_factory=dict)


@dataclass
class VisualizationConfig:
    """Configuration for spectral visualization."""

    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 150
    style: str = "seaborn-v0_8"
    color_predicted: str = "#2E86AB"
    color_experimental: str = "#A23B72"
    color_difference: str = "#F18F01"
    color_discrepancy: str = "#C73E1D"
    alpha_fill: float = 0.3
    line_width: float = 2.0
    font_size: int = 12
    title_size: int = 14
    legend_size: int = 10
    highlight_threshold: float = 0.1
    show_statistics: bool = True
    show_peak_labels: bool = True
    interactive_mode: bool = False


class SpectralAligner:
    """Alignment utilities for spectral comparison."""

    def __init__(self, use_gpu: bool = True):
        """Initialize spectral aligner."""
        self.use_gpu = use_gpu and self._check_gpu_availability()
        logger.info(f"SpectralAligner initialized with GPU: {self.use_gpu}")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import cupy as cp

            return cp.cuda.is_available()
        except ImportError:
            return False

    def align_spectra(
        self,
        predicted: SpectralPrediction,
        experimental: ExperimentalSpectrum,
        method: str = "cross_correlation",
    ) -> Tuple[SpectralPrediction, ExperimentalSpectrum, float]:
        """
        Align predicted and experimental spectra.

        Args:
            predicted: Predicted spectrum
            experimental: Experimental spectrum
            method: Alignment method

        Returns:
            Tuple of aligned spectra and offset
        """
        if method == "cross_correlation":
            return self._align_cross_correlation(predicted, experimental)
        elif method == "peak_matching":
            return self._align_peak_matching(predicted, experimental)
        elif method == "dtw":
            return self._align_dtw(predicted, experimental)
        else:
            raise ValueError(f"Unknown alignment method: {method}")

    def _align_cross_correlation(
        self, predicted: SpectralPrediction, experimental: ExperimentalSpectrum
    ) -> Tuple[SpectralPrediction, ExperimentalSpectrum, float]:
        """Align spectra using cross-correlation."""
        # Interpolate to common frequency grid
        common_freq = self._create_common_frequency_grid(predicted, experimental)

        # Interpolate both spectra
        pred_interp = np.interp(
            common_freq, predicted.frequencies, predicted.intensities
        )
        exp_interp = np.interp(
            common_freq, experimental.frequencies, experimental.intensities
        )

        # Cross-correlation
        if SCIPY_AVAILABLE:
            correlation = scipy.signal.correlate(pred_interp, exp_interp, mode="full")
            lags = scipy.signal.correlation_lags(
                len(pred_interp), len(exp_interp), mode="full"
            )

            # Find best alignment
            best_lag_idx = np.argmax(correlation)
            best_lag = lags[best_lag_idx]

            # Convert lag to frequency offset
            freq_resolution = common_freq[1] - common_freq[0]
            offset = best_lag * freq_resolution
        else:
            offset = 0.0

        # Apply offset
        aligned_predicted = SpectralPrediction(
            frequencies=predicted.frequencies + offset,
            intensities=predicted.intensities,
            spectroscopy_type=predicted.spectroscopy_type,
            confidence_intervals=predicted.confidence_intervals,
            peak_assignments=predicted.peak_assignments,
            computation_time=predicted.computation_time,
        )

        return aligned_predicted, experimental, offset

    def _align_peak_matching(
        self, predicted: SpectralPrediction, experimental: ExperimentalSpectrum
    ) -> Tuple[SpectralPrediction, ExperimentalSpectrum, float]:
        """Align spectra using peak matching."""
        # Find major peaks in both spectra
        pred_peaks = self._find_major_peaks(
            predicted.frequencies, predicted.intensities
        )
        exp_peaks = self._find_major_peaks(
            experimental.frequencies, experimental.intensities
        )

        if len(pred_peaks) == 0 or len(exp_peaks) == 0:
            return predicted, experimental, 0.0

        # Find best peak matching
        best_offset = 0.0
        best_score = -np.inf

        for pred_peak in pred_peaks[:3]:  # Use top 3 peaks
            for exp_peak in exp_peaks[:3]:
                offset = exp_peak - pred_peak
                score = self._score_peak_alignment(predicted, experimental, offset)

                if score > best_score:
                    best_score = score
                    best_offset = offset

        # Apply offset
        aligned_predicted = SpectralPrediction(
            frequencies=predicted.frequencies + best_offset,
            intensities=predicted.intensities,
            spectroscopy_type=predicted.spectroscopy_type,
            confidence_intervals=predicted.confidence_intervals,
            peak_assignments=predicted.peak_assignments,
            computation_time=predicted.computation_time,
        )

        return aligned_predicted, experimental, best_offset

    def _align_dtw(
        self, predicted: SpectralPrediction, experimental: ExperimentalSpectrum
    ) -> Tuple[SpectralPrediction, ExperimentalSpectrum, float]:
        """Align spectra using dynamic time warping."""
        # Simplified DTW implementation
        # In practice, you'd use a specialized DTW library

        # Create common frequency grid
        common_freq = self._create_common_frequency_grid(predicted, experimental)

        # Interpolate both spectra
        pred_interp = np.interp(
            common_freq, predicted.frequencies, predicted.intensities
        )
        exp_interp = np.interp(
            common_freq, experimental.frequencies, experimental.intensities
        )

        # Simple warping using correlation
        if SCIPY_AVAILABLE:
            correlation = scipy.signal.correlate(pred_interp, exp_interp, mode="same")
            offset_idx = np.argmax(correlation) - len(correlation) // 2
            freq_resolution = common_freq[1] - common_freq[0]
            offset = offset_idx * freq_resolution
        else:
            offset = 0.0

        # Apply offset
        aligned_predicted = SpectralPrediction(
            frequencies=predicted.frequencies + offset,
            intensities=predicted.intensities,
            spectroscopy_type=predicted.spectroscopy_type,
            confidence_intervals=predicted.confidence_intervals,
            peak_assignments=predicted.peak_assignments,
            computation_time=predicted.computation_time,
        )

        return aligned_predicted, experimental, offset

    def _create_common_frequency_grid(
        self, predicted: SpectralPrediction, experimental: ExperimentalSpectrum
    ) -> np.ndarray:
        """Create common frequency grid for both spectra."""
        min_freq = max(np.min(predicted.frequencies), np.min(experimental.frequencies))
        max_freq = min(np.max(predicted.frequencies), np.max(experimental.frequencies))

        # Use higher resolution
        pred_res = np.median(np.diff(predicted.frequencies))
        exp_res = np.median(np.diff(experimental.frequencies))
        resolution = min(pred_res, exp_res)

        return np.arange(min_freq, max_freq, resolution)

    def _find_major_peaks(
        self, frequencies: np.ndarray, intensities: np.ndarray
    ) -> List[float]:
        """Find major peaks in spectrum."""
        if not SCIPY_AVAILABLE:
            return []

        # Find peaks
        peaks, properties = scipy.signal.find_peaks(
            intensities,
            height=np.max(intensities) * 0.1,  # At least 10% of max intensity
            prominence=np.std(intensities) * 2,  # Significant prominence
        )

        # Sort by intensity
        peak_intensities = intensities[peaks]
        sorted_indices = np.argsort(peak_intensities)[::-1]

        # Return frequencies of major peaks
        return [frequencies[peaks[i]] for i in sorted_indices[:5]]

    def _score_peak_alignment(
        self,
        predicted: SpectralPrediction,
        experimental: ExperimentalSpectrum,
        offset: float,
    ) -> float:
        """Score peak alignment quality."""
        # Apply offset to predicted spectrum
        shifted_freq = predicted.frequencies + offset

        # Find overlap region
        min_freq = max(np.min(shifted_freq), np.min(experimental.frequencies))
        max_freq = min(np.max(shifted_freq), np.max(experimental.frequencies))

        if min_freq >= max_freq:
            return -np.inf

        # Create common grid and interpolate
        common_freq = np.linspace(min_freq, max_freq, 1000)
        pred_interp = np.interp(common_freq, shifted_freq, predicted.intensities)
        exp_interp = np.interp(
            common_freq, experimental.frequencies, experimental.intensities
        )

        # Calculate correlation
        if np.std(pred_interp) == 0 or np.std(exp_interp) == 0:
            return -np.inf

        correlation = np.corrcoef(pred_interp, exp_interp)[0, 1]
        return correlation if not np.isnan(correlation) else -np.inf


class SpectralComparator:
    """Main class for spectral comparison and analysis."""

    def __init__(self, use_gpu: bool = True):
        """Initialize spectral comparator."""
        self.use_gpu = use_gpu and self._check_gpu_availability()
        self.aligner = SpectralAligner(use_gpu=use_gpu)
        logger.info(f"SpectralComparator initialized with GPU: {self.use_gpu}")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import cupy as cp

            return cp.cuda.is_available()
        except ImportError:
            return False

    def compare_spectra(
        self,
        predicted: SpectralPrediction,
        experimental: ExperimentalSpectrum,
        methods: List[ComparisonMethod] = None,
        align_spectra: bool = True,
        detect_discrepancies: bool = True,
    ) -> SpectralComparison:
        """
        Compare predicted and experimental spectra.

        Args:
            predicted: Predicted spectrum
            experimental: Experimental spectrum
            methods: Comparison methods to use
            align_spectra: Whether to align spectra first
            detect_discrepancies: Whether to detect discrepancy regions

        Returns:
            Spectral comparison results
        """
        start_time = time.time()

        if methods is None:
            methods = [
                ComparisonMethod.MEAN_SQUARED_ERROR,
                ComparisonMethod.CORRELATION_COEFFICIENT,
                ComparisonMethod.COSINE_SIMILARITY,
                ComparisonMethod.PEAK_MATCHING,
            ]

        # Align spectra if requested
        alignment_offset = 0.0
        if align_spectra:
            predicted, experimental, alignment_offset = self.aligner.align_spectra(
                predicted, experimental
            )

        # Calculate comparison metrics
        comparison_metrics = {}
        for method in methods:
            try:
                metric_value = self._calculate_metric(predicted, experimental, method)
                comparison_metrics[method.value] = metric_value
            except Exception as e:
                logger.warning(f"Failed to calculate {method.value}: {e}")
                comparison_metrics[method.value] = np.nan

        # Calculate overall similarity
        overall_similarity = self._calculate_overall_similarity(comparison_metrics)

        # Detect discrepancy regions
        discrepancy_regions = []
        if detect_discrepancies:
            discrepancy_regions = self._detect_discrepancies(predicted, experimental)

        # Peak matching analysis
        peak_matching_results = self._analyze_peak_matching(predicted, experimental)

        # Statistical tests
        statistical_tests = self._perform_statistical_tests(predicted, experimental)

        # Quality assessment
        quality_assessment = self._assess_comparison_quality(predicted, experimental)

        computation_time = time.time() - start_time

        return SpectralComparison(
            predicted_spectrum=predicted,
            experimental_spectrum=experimental,
            overall_similarity=overall_similarity,
            comparison_metrics=comparison_metrics,
            discrepancy_regions=discrepancy_regions,
            peak_matching_results=peak_matching_results,
            statistical_tests=statistical_tests,
            alignment_offset=alignment_offset,
            computation_time=computation_time,
            quality_assessment=quality_assessment,
        )

    def _calculate_metric(
        self,
        predicted: SpectralPrediction,
        experimental: ExperimentalSpectrum,
        method: ComparisonMethod,
    ) -> float:
        """Calculate specific comparison metric."""
        # Create common frequency grid
        common_freq = self._create_common_frequency_grid(predicted, experimental)

        # Interpolate both spectra
        pred_interp = np.interp(
            common_freq, predicted.frequencies, predicted.intensities
        )
        exp_interp = np.interp(
            common_freq, experimental.frequencies, experimental.intensities
        )

        if method == ComparisonMethod.MEAN_SQUARED_ERROR:
            return float(np.mean((pred_interp - exp_interp) ** 2))

        elif method == ComparisonMethod.MEAN_ABSOLUTE_ERROR:
            return float(np.mean(np.abs(pred_interp - exp_interp)))

        elif method == ComparisonMethod.COSINE_SIMILARITY:
            norm_pred = np.linalg.norm(pred_interp)
            norm_exp = np.linalg.norm(exp_interp)
            if norm_pred == 0 or norm_exp == 0:
                return 0.0
            return float(np.dot(pred_interp, exp_interp) / (norm_pred * norm_exp))

        elif method == ComparisonMethod.CORRELATION_COEFFICIENT:
            if np.std(pred_interp) == 0 or np.std(exp_interp) == 0:
                return 0.0
            correlation = np.corrcoef(pred_interp, exp_interp)[0, 1]
            return float(correlation if not np.isnan(correlation) else 0.0)

        elif method == ComparisonMethod.SPECTRAL_ANGLE_MAPPER:
            # Spectral Angle Mapper (SAM)
            norm_pred = np.linalg.norm(pred_interp)
            norm_exp = np.linalg.norm(exp_interp)
            if norm_pred == 0 or norm_exp == 0:
                return np.pi / 2
            cos_angle = np.dot(pred_interp, exp_interp) / (norm_pred * norm_exp)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return float(np.arccos(cos_angle))

        elif method == ComparisonMethod.PEAK_MATCHING:
            return self._calculate_peak_matching_score(predicted, experimental)

        elif method == ComparisonMethod.DYNAMIC_TIME_WARPING:
            return self._calculate_dtw_distance(pred_interp, exp_interp)

        elif method == ComparisonMethod.EARTH_MOVERS_DISTANCE:
            return self._calculate_emd_distance(pred_interp, exp_interp)

        else:
            raise ValueError(f"Unknown comparison method: {method}")

    def _calculate_overall_similarity(self, metrics: Dict[str, float]) -> float:
        """Calculate overall similarity score from individual metrics."""
        if not metrics:
            return 0.0

        # Weighted combination of metrics
        weights = {
            ComparisonMethod.CORRELATION_COEFFICIENT.value: 0.3,
            ComparisonMethod.COSINE_SIMILARITY.value: 0.25,
            ComparisonMethod.MEAN_SQUARED_ERROR.value: -0.2,  # Lower is better
            ComparisonMethod.MEAN_ABSOLUTE_ERROR.value: -0.15,  # Lower is better
            ComparisonMethod.PEAK_MATCHING.value: 0.2,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for metric_name, value in metrics.items():
            if metric_name in weights and not np.isnan(value):
                weight = weights[metric_name]

                # Normalize MSE and MAE (invert since lower is better)
                if metric_name in [
                    ComparisonMethod.MEAN_SQUARED_ERROR.value,
                    ComparisonMethod.MEAN_ABSOLUTE_ERROR.value,
                ]:
                    # Convert to similarity (0-1 scale)
                    normalized_value = 1.0 / (1.0 + value)
                    weighted_sum += abs(weight) * normalized_value
                else:
                    weighted_sum += weight * value

                total_weight += abs(weight)

        if total_weight == 0:
            return 0.0

        # Normalize to 0-1 range
        similarity = weighted_sum / total_weight
        return float(np.clip(similarity, 0.0, 1.0))

    def _detect_discrepancies(
        self,
        predicted: SpectralPrediction,
        experimental: ExperimentalSpectrum,
        threshold: float = 0.1,
    ) -> List[DiscrepancyRegion]:
        """Detect regions of significant discrepancy."""
        discrepancies = []

        # Create common frequency grid
        common_freq = self._create_common_frequency_grid(predicted, experimental)

        # Interpolate both spectra
        pred_interp = np.interp(
            common_freq, predicted.frequencies, predicted.intensities
        )
        exp_interp = np.interp(
            common_freq, experimental.frequencies, experimental.intensities
        )

        # Calculate point-wise differences
        intensity_diff = np.abs(pred_interp - exp_interp)

        # Find regions above threshold
        above_threshold = intensity_diff > threshold

        # Group consecutive regions
        regions = self._group_consecutive_regions(above_threshold)

        for start_idx, end_idx in regions:
            if end_idx - start_idx < 5:  # Skip very small regions
                continue

            region_freq_start = common_freq[start_idx]
            region_freq_end = common_freq[end_idx]
            region_diff = intensity_diff[start_idx : end_idx + 1]

            # Determine discrepancy type
            pred_region = pred_interp[start_idx : end_idx + 1]
            exp_region = exp_interp[start_idx : end_idx + 1]

            discrepancy_type = self._classify_discrepancy(pred_region, exp_region)

            # Calculate severity and significance
            severity = float(
                np.mean(region_diff) / np.max([np.max(pred_interp), np.max(exp_interp)])
            )
            significance = self._calculate_statistical_significance(
                pred_region, exp_region
            )

            # Generate description and explanation
            description = f"{discrepancy_type.value} from {region_freq_start:.1f} to {region_freq_end:.1f}"
            explanation = self._generate_discrepancy_explanation(
                discrepancy_type, severity
            )

            discrepancy = DiscrepancyRegion(
                frequency_start=region_freq_start,
                frequency_end=region_freq_end,
                discrepancy_type=discrepancy_type,
                severity=severity,
                statistical_significance=significance,
                predicted_intensity=float(np.mean(pred_region)),
                experimental_intensity=float(np.mean(exp_region)),
                description=description,
                suggested_explanation=explanation,
            )

            discrepancies.append(discrepancy)

        return discrepancies

    def _create_common_frequency_grid(
        self, predicted: SpectralPrediction, experimental: ExperimentalSpectrum
    ) -> np.ndarray:
        """Create common frequency grid for comparison."""
        min_freq = max(np.min(predicted.frequencies), np.min(experimental.frequencies))
        max_freq = min(np.max(predicted.frequencies), np.max(experimental.frequencies))

        # Use higher resolution
        pred_res = np.median(np.diff(predicted.frequencies))
        exp_res = np.median(np.diff(experimental.frequencies))
        resolution = min(pred_res, exp_res)

        return np.arange(min_freq, max_freq, resolution)

    def _calculate_peak_matching_score(
        self, predicted: SpectralPrediction, experimental: ExperimentalSpectrum
    ) -> float:
        """Calculate peak matching score."""
        # Find peaks in both spectra
        pred_peaks = self._find_spectrum_peaks(
            predicted.frequencies, predicted.intensities
        )
        exp_peaks = self._find_spectrum_peaks(
            experimental.frequencies, experimental.intensities
        )

        if len(pred_peaks) == 0 and len(exp_peaks) == 0:
            return 1.0

        if len(pred_peaks) == 0 or len(exp_peaks) == 0:
            return 0.0

        # Match peaks
        matched_peaks = 0
        tolerance = (
            np.median(np.diff(predicted.frequencies)) * 5
        )  # 5 data points tolerance

        for pred_freq, pred_int in pred_peaks:
            for exp_freq, exp_int in exp_peaks:
                if abs(pred_freq - exp_freq) <= tolerance:
                    # Weight by intensity similarity
                    intensity_similarity = 1.0 - abs(pred_int - exp_int) / max(
                        pred_int, exp_int
                    )
                    matched_peaks += intensity_similarity
                    break

        # Calculate score
        max_peaks = max(len(pred_peaks), len(exp_peaks))
        return float(matched_peaks / max_peaks)

    def _find_spectrum_peaks(
        self, frequencies: np.ndarray, intensities: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Find peaks in spectrum."""
        if not SCIPY_AVAILABLE:
            # Simple peak detection without scipy
            peaks = []
            for i in range(1, len(intensities) - 1):
                if (
                    intensities[i] > intensities[i - 1]
                    and intensities[i] > intensities[i + 1]
                    and intensities[i] > np.mean(intensities) + np.std(intensities)
                ):
                    peaks.append((frequencies[i], intensities[i]))
            return peaks

        # Use scipy for more sophisticated peak detection
        peak_indices, properties = scipy.signal.find_peaks(
            intensities,
            height=np.max(intensities) * 0.05,  # At least 5% of max intensity
            prominence=np.std(intensities),  # Some prominence
        )

        peaks = [(frequencies[i], intensities[i]) for i in peak_indices]

        # Sort by intensity and return top peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:10]  # Top 10 peaks

    def _calculate_dtw_distance(
        self, series1: np.ndarray, series2: np.ndarray
    ) -> float:
        """Calculate dynamic time warping distance."""
        # Simplified DTW implementation
        n, m = len(series1), len(series2)

        # Create distance matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(series1[i - 1] - series2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],  # insertion
                    dtw_matrix[i, j - 1],  # deletion
                    dtw_matrix[i - 1, j - 1],  # match
                )

        return float(dtw_matrix[n, m])

    def _calculate_emd_distance(
        self, series1: np.ndarray, series2: np.ndarray
    ) -> float:
        """Calculate Earth Mover's Distance (Wasserstein distance)."""
        if not SCIPY_AVAILABLE:
            # Fallback to simple L1 distance
            return float(np.sum(np.abs(series1 - series2)))

        # Normalize to probability distributions
        series1_norm = series1 / np.sum(series1) if np.sum(series1) > 0 else series1
        series2_norm = series2 / np.sum(series2) if np.sum(series2) > 0 else series2

        # Calculate cumulative distributions
        cum1 = np.cumsum(series1_norm)
        cum2 = np.cumsum(series2_norm)

        # EMD is the L1 distance between cumulative distributions
        return float(np.sum(np.abs(cum1 - cum2)))

    def _group_consecutive_regions(
        self, boolean_array: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Group consecutive True regions in boolean array."""
        regions = []
        in_region = False
        start_idx = 0

        for i, value in enumerate(boolean_array):
            if value and not in_region:
                # Start of new region
                start_idx = i
                in_region = True
            elif not value and in_region:
                # End of region
                regions.append((start_idx, i - 1))
                in_region = False

        # Handle case where array ends in True region
        if in_region:
            regions.append((start_idx, len(boolean_array) - 1))

        return regions

    def _classify_discrepancy(
        self, predicted_region: np.ndarray, experimental_region: np.ndarray
    ) -> DiscrepancyType:
        """Classify type of discrepancy in region."""
        pred_mean = np.mean(predicted_region)
        exp_mean = np.mean(experimental_region)

        if pred_mean > exp_mean * 1.5:
            return DiscrepancyType.EXTRA_PEAK
        elif exp_mean > pred_mean * 1.5:
            return DiscrepancyType.MISSING_PEAK
        elif abs(pred_mean - exp_mean) > 0.2:
            return DiscrepancyType.INTENSITY_DIFFERENCE
        else:
            return DiscrepancyType.SYSTEMATIC_ERROR

    def _calculate_statistical_significance(
        self, predicted_region: np.ndarray, experimental_region: np.ndarray
    ) -> float:
        """Calculate statistical significance of discrepancy."""
        if not SCIPY_AVAILABLE:
            return 0.5  # Default moderate significance

        try:
            # Perform t-test
            statistic, p_value = scipy.stats.ttest_ind(
                predicted_region, experimental_region
            )
            return float(1.0 - p_value)  # Convert p-value to significance
        except:
            return 0.5

    def _generate_discrepancy_explanation(
        self, discrepancy_type: DiscrepancyType, severity: float
    ) -> str:
        """Generate explanation for discrepancy."""
        explanations = {
            DiscrepancyType.INTENSITY_DIFFERENCE: [
                "Possible calibration error or different experimental conditions",
                "Model may not capture all intensity-affecting factors",
                "Different sample concentration or path length",
            ],
            DiscrepancyType.MISSING_PEAK: [
                "Model may be missing a chemical component",
                "Peak may be below detection threshold in experiment",
                "Possible overlapping peaks in experimental spectrum",
            ],
            DiscrepancyType.EXTRA_PEAK: [
                "Model may predict spurious peak",
                "Experimental artifact or contamination",
                "Model overfitting to training data",
            ],
            DiscrepancyType.FREQUENCY_SHIFT: [
                "Calibration offset between predicted and experimental",
                "Different reference standards used",
                "Temperature or pressure effects",
            ],
            DiscrepancyType.SYSTEMATIC_ERROR: [
                "Systematic bias in model predictions",
                "Instrumental drift or bias",
                "Model limitation in this spectral region",
            ],
        }

        # Select explanation based on severity
        type_explanations = explanations.get(discrepancy_type, ["Unknown discrepancy"])

        if severity > 0.5:
            return f"High severity: {type_explanations[0]}"
        elif severity > 0.2:
            return f"Moderate severity: {type_explanations[1] if len(type_explanations) > 1 else type_explanations[0]}"
        else:
            return f"Low severity: {type_explanations[-1]}"

    def _analyze_peak_matching(
        self, predicted: SpectralPrediction, experimental: ExperimentalSpectrum
    ) -> Dict[str, Any]:
        """Analyze peak matching between spectra."""
        pred_peaks = self._find_spectrum_peaks(
            predicted.frequencies, predicted.intensities
        )
        exp_peaks = self._find_spectrum_peaks(
            experimental.frequencies, experimental.intensities
        )

        results = {
            "predicted_peaks": len(pred_peaks),
            "experimental_peaks": len(exp_peaks),
            "matched_peaks": 0,
            "missing_peaks": [],
            "extra_peaks": [],
            "peak_matches": [],
        }

        if not pred_peaks or not exp_peaks:
            return results

        tolerance = np.median(np.diff(predicted.frequencies)) * 5
        matched_pred = set()
        matched_exp = set()

        # Find matches
        for i, (pred_freq, pred_int) in enumerate(pred_peaks):
            best_match = None
            best_distance = float("inf")

            for j, (exp_freq, exp_int) in enumerate(exp_peaks):
                if j in matched_exp:
                    continue

                distance = abs(pred_freq - exp_freq)
                if distance <= tolerance and distance < best_distance:
                    best_distance = distance
                    best_match = j

            if best_match is not None:
                matched_pred.add(i)
                matched_exp.add(best_match)
                results["matched_peaks"] += 1

                exp_freq, exp_int = exp_peaks[best_match]
                results["peak_matches"].append(
                    {
                        "predicted_frequency": pred_freq,
                        "experimental_frequency": exp_freq,
                        "frequency_difference": pred_freq - exp_freq,
                        "predicted_intensity": pred_int,
                        "experimental_intensity": exp_int,
                        "intensity_ratio": pred_int / exp_int
                        if exp_int > 0
                        else np.inf,
                    }
                )

        # Find unmatched peaks
        for i, (freq, intensity) in enumerate(pred_peaks):
            if i not in matched_pred:
                results["extra_peaks"].append(
                    {"frequency": freq, "intensity": intensity}
                )

        for j, (freq, intensity) in enumerate(exp_peaks):
            if j not in matched_exp:
                results["missing_peaks"].append(
                    {"frequency": freq, "intensity": intensity}
                )

        return results

    def _perform_statistical_tests(
        self, predicted: SpectralPrediction, experimental: ExperimentalSpectrum
    ) -> Dict[str, Dict[str, float]]:
        """Perform statistical tests on spectral comparison."""
        tests = {}

        if not SCIPY_AVAILABLE:
            return tests

        # Create common frequency grid
        common_freq = self._create_common_frequency_grid(predicted, experimental)
        pred_interp = np.interp(
            common_freq, predicted.frequencies, predicted.intensities
        )
        exp_interp = np.interp(
            common_freq, experimental.frequencies, experimental.intensities
        )

        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = scipy.stats.ks_2samp(pred_interp, exp_interp)
            tests["kolmogorov_smirnov"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "interpretation": "similar" if ks_p > 0.05 else "different",
            }
        except Exception as e:
            logger.warning(f"KS test failed: {e}")

        try:
            # Mann-Whitney U test
            mw_stat, mw_p = scipy.stats.mannwhitneyu(
                pred_interp, exp_interp, alternative="two-sided"
            )
            tests["mann_whitney"] = {
                "statistic": float(mw_stat),
                "p_value": float(mw_p),
                "interpretation": "similar" if mw_p > 0.05 else "different",
            }
        except Exception as e:
            logger.warning(f"Mann-Whitney test failed: {e}")

        try:
            # Pearson correlation test
            corr_stat, corr_p = scipy.stats.pearsonr(pred_interp, exp_interp)
            tests["pearson_correlation"] = {
                "statistic": float(corr_stat),
                "p_value": float(corr_p),
                "interpretation": "correlated"
                if corr_p < 0.05 and abs(corr_stat) > 0.5
                else "not_correlated",
            }
        except Exception as e:
            logger.warning(f"Pearson correlation test failed: {e}")

        return tests

    def _assess_comparison_quality(
        self, predicted: SpectralPrediction, experimental: ExperimentalSpectrum
    ) -> Dict[str, float]:
        """Assess quality of spectral comparison."""
        quality = {}

        # Frequency range overlap
        pred_min, pred_max = np.min(predicted.frequencies), np.max(
            predicted.frequencies
        )
        exp_min, exp_max = np.min(experimental.frequencies), np.max(
            experimental.frequencies
        )

        overlap_min = max(pred_min, exp_min)
        overlap_max = min(pred_max, exp_max)

        if overlap_max > overlap_min:
            pred_range = pred_max - pred_min
            exp_range = exp_max - exp_min
            overlap_range = overlap_max - overlap_min

            quality["frequency_overlap"] = float(
                overlap_range / min(pred_range, exp_range)
            )
        else:
            quality["frequency_overlap"] = 0.0

        # Resolution compatibility
        pred_resolution = np.median(np.diff(predicted.frequencies))
        exp_resolution = np.median(np.diff(experimental.frequencies))
        quality["resolution_ratio"] = float(
            min(pred_resolution, exp_resolution) / max(pred_resolution, exp_resolution)
        )

        # Signal-to-noise estimates
        pred_snr = np.max(predicted.intensities) / np.std(predicted.intensities)
        exp_snr = np.max(experimental.intensities) / np.std(experimental.intensities)
        quality["predicted_snr"] = float(pred_snr)
        quality["experimental_snr"] = float(exp_snr)

        # Data completeness
        quality["predicted_data_points"] = len(predicted.frequencies)
        quality["experimental_data_points"] = len(experimental.frequencies)

        return quality


class SpectralVisualizer:
    """Visualization tools for spectral comparison."""

    def __init__(self, config: VisualizationConfig = None):
        """Initialize spectral visualizer."""
        self.config = config or VisualizationConfig()

        if not MATPLOTLIB_AVAILABLE:
            logger.warning(
                "Matplotlib not available. Visualization functionality limited."
            )

        # Set matplotlib style
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(self.config.style)
            except:
                logger.warning(f"Could not set style {self.config.style}")

    def create_comparison_plot(
        self,
        comparison: SpectralComparison,
        plot_type: VisualizationType = VisualizationType.OVERLAY_PLOT,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> Optional[Any]:
        """
        Create comparison visualization plot.

        Args:
            comparison: Spectral comparison results
            plot_type: Type of visualization
            save_path: Path to save plot
            show_plot: Whether to display plot

        Returns:
            Figure object if matplotlib available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for plotting")
            return None

        if plot_type == VisualizationType.OVERLAY_PLOT:
            return self._create_overlay_plot(comparison, save_path, show_plot)
        elif plot_type == VisualizationType.DIFFERENCE_PLOT:
            return self._create_difference_plot(comparison, save_path, show_plot)
        elif plot_type == VisualizationType.MIRROR_PLOT:
            return self._create_mirror_plot(comparison, save_path, show_plot)
        elif plot_type == VisualizationType.RESIDUAL_PLOT:
            return self._create_residual_plot(comparison, save_path, show_plot)
        elif plot_type == VisualizationType.DISCREPANCY_MAP:
            return self._create_discrepancy_map(comparison, save_path, show_plot)
        elif plot_type == VisualizationType.HEATMAP:
            return self._create_heatmap(comparison, save_path, show_plot)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

    def _create_overlay_plot(
        self, comparison: SpectralComparison, save_path: Optional[str], show_plot: bool
    ) -> Any:
        """Create overlay plot of predicted vs experimental spectra."""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        predicted = comparison.predicted_spectrum
        experimental = comparison.experimental_spectrum

        # Plot spectra
        ax.plot(
            predicted.frequencies,
            predicted.intensities,
            color=self.config.color_predicted,
            linewidth=self.config.line_width,
            label="Predicted",
            alpha=0.8,
        )

        ax.plot(
            experimental.frequencies,
            experimental.intensities,
            color=self.config.color_experimental,
            linewidth=self.config.line_width,
            label="Experimental",
            alpha=0.8,
        )

        # Highlight discrepancy regions
        if comparison.discrepancy_regions:
            for region in comparison.discrepancy_regions:
                if region.severity > self.config.highlight_threshold:
                    ax.axvspan(
                        region.frequency_start,
                        region.frequency_end,
                        color=self.config.color_discrepancy,
                        alpha=self.config.alpha_fill,
                        label="Discrepancy"
                        if region == comparison.discrepancy_regions[0]
                        else "",
                    )

        # Formatting
        ax.set_xlabel("Frequency", fontsize=self.config.font_size)
        ax.set_ylabel("Intensity", fontsize=self.config.font_size)
        ax.set_title(
            f"Spectral Comparison (Similarity: {comparison.overall_similarity:.3f})",
            fontsize=self.config.title_size,
        )
        ax.legend(fontsize=self.config.legend_size)
        ax.grid(True, alpha=0.3)

        # Add statistics if requested
        if self.config.show_statistics:
            stats_text = self._format_statistics_text(comparison)
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=self.config.font_size - 2,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        if show_plot:
            plt.show()

        return fig

    def _create_difference_plot(
        self, comparison: SpectralComparison, save_path: Optional[str], show_plot: bool
    ) -> Any:
        """Create difference plot showing pred - experimental."""
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=self.config.figure_size, dpi=self.config.dpi, sharex=True
        )

        predicted = comparison.predicted_spectrum
        experimental = comparison.experimental_spectrum

        # Top plot: original spectra
        ax1.plot(
            predicted.frequencies,
            predicted.intensities,
            color=self.config.color_predicted,
            linewidth=self.config.line_width,
            label="Predicted",
            alpha=0.8,
        )

        ax1.plot(
            experimental.frequencies,
            experimental.intensities,
            color=self.config.color_experimental,
            linewidth=self.config.line_width,
            label="Experimental",
            alpha=0.8,
        )

        ax1.set_ylabel("Intensity", fontsize=self.config.font_size)
        ax1.legend(fontsize=self.config.legend_size)
        ax1.grid(True, alpha=0.3)

        # Bottom plot: difference
        # Create common frequency grid
        common_freq = self._create_common_frequency_grid(predicted, experimental)
        pred_interp = np.interp(
            common_freq, predicted.frequencies, predicted.intensities
        )
        exp_interp = np.interp(
            common_freq, experimental.frequencies, experimental.intensities
        )

        difference = pred_interp - exp_interp

        ax2.plot(
            common_freq,
            difference,
            color=self.config.color_difference,
            linewidth=self.config.line_width,
            label="Difference",
        )
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        # Highlight significant discrepancies
        threshold = np.std(difference) * 2
        significant = np.abs(difference) > threshold
        ax2.fill_between(
            common_freq,
            difference,
            0,
            where=significant,
            color=self.config.color_discrepancy,
            alpha=self.config.alpha_fill,
            label="Significant discrepancy",
        )

        ax2.set_xlabel("Frequency", fontsize=self.config.font_size)
        ax2.set_ylabel("Difference", fontsize=self.config.font_size)
        ax2.legend(fontsize=self.config.legend_size)
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            f"Spectral Difference Analysis (Similarity: {comparison.overall_similarity:.3f})",
            fontsize=self.config.title_size,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        if show_plot:
            plt.show()

        return fig

    def _create_mirror_plot(
        self, comparison: SpectralComparison, save_path: Optional[str], show_plot: bool
    ) -> Any:
        """Create mirror plot with predicted on top, experimental on bottom."""
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=self.config.figure_size, dpi=self.config.dpi, sharex=True
        )

        predicted = comparison.predicted_spectrum
        experimental = comparison.experimental_spectrum

        # Top plot: predicted (normal)
        ax1.plot(
            predicted.frequencies,
            predicted.intensities,
            color=self.config.color_predicted,
            linewidth=self.config.line_width,
        )
        ax1.fill_between(
            predicted.frequencies,
            predicted.intensities,
            0,
            color=self.config.color_predicted,
            alpha=self.config.alpha_fill,
        )
        ax1.set_ylabel("Predicted\nIntensity", fontsize=self.config.font_size)
        ax1.grid(True, alpha=0.3)

        # Bottom plot: experimental (inverted)
        ax2.plot(
            experimental.frequencies,
            -experimental.intensities,
            color=self.config.color_experimental,
            linewidth=self.config.line_width,
        )
        ax2.fill_between(
            experimental.frequencies,
            -experimental.intensities,
            0,
            color=self.config.color_experimental,
            alpha=self.config.alpha_fill,
        )
        ax2.set_ylabel("Experimental\nIntensity", fontsize=self.config.font_size)
        ax2.set_xlabel("Frequency", fontsize=self.config.font_size)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()

        plt.suptitle(
            f"Mirror Plot Comparison (Similarity: {comparison.overall_similarity:.3f})",
            fontsize=self.config.title_size,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        if show_plot:
            plt.show()

        return fig

    def _create_residual_plot(
        self, comparison: SpectralComparison, save_path: Optional[str], show_plot: bool
    ) -> Any:
        """Create residual analysis plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(15, 10), dpi=self.config.dpi
        )

        predicted = comparison.predicted_spectrum
        experimental = comparison.experimental_spectrum

        # Create common frequency grid
        common_freq = self._create_common_frequency_grid(predicted, experimental)
        pred_interp = np.interp(
            common_freq, predicted.frequencies, predicted.intensities
        )
        exp_interp = np.interp(
            common_freq, experimental.frequencies, experimental.intensities
        )

        residuals = pred_interp - exp_interp

        # Plot 1: Residuals vs frequency
        ax1.plot(common_freq, residuals, color=self.config.color_difference, alpha=0.7)
        ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Frequency")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residuals vs predicted values
        ax2.scatter(
            pred_interp, residuals, color=self.config.color_difference, alpha=0.7
        )
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals vs Predicted")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Histogram of residuals
        ax3.hist(
            residuals,
            bins=50,
            color=self.config.color_difference,
            alpha=0.7,
            edgecolor="black",
        )
        ax3.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Residuals")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Residual Distribution")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Q-Q plot (if scipy available)
        if SCIPY_AVAILABLE:
            scipy.stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title("Q-Q Plot (Normal)")
        else:
            ax4.text(
                0.5,
                0.5,
                "Q-Q Plot\n(scipy required)",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Q-Q Plot (Not Available)")

        plt.suptitle(
            f"Residual Analysis (RMSE: {np.sqrt(np.mean(residuals**2)):.4f})",
            fontsize=self.config.title_size,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        if show_plot:
            plt.show()

        return fig

    def _create_discrepancy_map(
        self, comparison: SpectralComparison, save_path: Optional[str], show_plot: bool
    ) -> Any:
        """Create discrepancy highlighting map."""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        predicted = comparison.predicted_spectrum
        experimental = comparison.experimental_spectrum

        # Plot base spectra
        ax.plot(
            predicted.frequencies,
            predicted.intensities,
            color=self.config.color_predicted,
            linewidth=self.config.line_width,
            label="Predicted",
            alpha=0.6,
        )

        ax.plot(
            experimental.frequencies,
            experimental.intensities,
            color=self.config.color_experimental,
            linewidth=self.config.line_width,
            label="Experimental",
            alpha=0.6,
        )

        # Create discrepancy regions with different colors/intensities
        severity_colormap = LinearSegmentedColormap.from_list(
            "severity", ["yellow", "orange", "red"]
        )

        for i, region in enumerate(comparison.discrepancy_regions):
            color = severity_colormap(region.severity)

            # Highlight region
            ax.axvspan(
                region.frequency_start,
                region.frequency_end,
                color=color,
                alpha=0.4,
                label=f"Discrepancy {i+1}" if i < 5 else "",
            )

            # Add annotation
            mid_freq = (region.frequency_start + region.frequency_end) / 2
            max_intensity = max(
                np.max(predicted.intensities), np.max(experimental.intensities)
            )

            ax.annotate(
                f"{region.discrepancy_type.value}\n(severity: {region.severity:.2f})",
                xy=(mid_freq, max_intensity * 0.9),
                xytext=(mid_freq, max_intensity * 1.05),
                ha="center",
                va="bottom",
                fontsize=self.config.font_size - 2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.7),
            )

        ax.set_xlabel("Frequency", fontsize=self.config.font_size)
        ax.set_ylabel("Intensity", fontsize=self.config.font_size)
        ax.set_title(
            f"Discrepancy Map ({len(comparison.discrepancy_regions)} regions)",
            fontsize=self.config.title_size,
        )
        ax.legend(fontsize=self.config.legend_size)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        if show_plot:
            plt.show()

        return fig

    def _create_heatmap(
        self, comparison: SpectralComparison, save_path: Optional[str], show_plot: bool
    ) -> Any:
        """Create 2D heatmap comparison."""
        if not SEABORN_AVAILABLE:
            logger.warning("Seaborn not available. Creating simple heatmap.")

        predicted = comparison.predicted_spectrum
        experimental = comparison.experimental_spectrum

        # Create common frequency grid
        common_freq = self._create_common_frequency_grid(predicted, experimental)
        pred_interp = np.interp(
            common_freq, predicted.frequencies, predicted.intensities
        )
        exp_interp = np.interp(
            common_freq, experimental.frequencies, experimental.intensities
        )

        # Create 2D array for heatmap
        data = np.array([pred_interp, exp_interp])

        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        if SEABORN_AVAILABLE:
            sns.heatmap(
                data,
                xticklabels=False,
                yticklabels=["Predicted", "Experimental"],
                cmap="viridis",
                ax=ax,
                cbar_kws={"label": "Intensity"},
            )
        else:
            im = ax.imshow(data, aspect="auto", cmap="viridis")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Predicted", "Experimental"])
            plt.colorbar(im, ax=ax, label="Intensity")

        ax.set_xlabel("Frequency Index", fontsize=self.config.font_size)
        ax.set_title("Spectral Intensity Heatmap", fontsize=self.config.title_size)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        if show_plot:
            plt.show()

        return fig

    def _create_common_frequency_grid(
        self, predicted: SpectralPrediction, experimental: ExperimentalSpectrum
    ) -> np.ndarray:
        """Create common frequency grid for visualization."""
        min_freq = max(np.min(predicted.frequencies), np.min(experimental.frequencies))
        max_freq = min(np.max(predicted.frequencies), np.max(experimental.frequencies))

        # Use reasonable number of points for visualization
        n_points = min(1000, len(predicted.frequencies), len(experimental.frequencies))

        return np.linspace(min_freq, max_freq, n_points)

    def _format_statistics_text(self, comparison: SpectralComparison) -> str:
        """Format statistics text for plot annotation."""
        stats = []

        # Overall similarity
        stats.append(f"Similarity: {comparison.overall_similarity:.3f}")

        # Key metrics
        metrics = comparison.comparison_metrics
        if "correlation" in metrics:
            stats.append(f"Correlation: {metrics['correlation']:.3f}")
        if "mse" in metrics:
            stats.append(f"MSE: {metrics['mse']:.4f}")

        # Discrepancies
        if comparison.discrepancy_regions:
            stats.append(f"Discrepancies: {len(comparison.discrepancy_regions)}")

        return "\n".join(stats)


# Utility functions for spectral comparison
def compare_spectra_batch(
    predicted_spectra: List[SpectralPrediction],
    experimental_spectra: List[ExperimentalSpectrum],
    methods: List[ComparisonMethod] = None,
    use_gpu: bool = True,
) -> List[SpectralComparison]:
    """
    Compare multiple spectra in batch.

    Args:
        predicted_spectra: List of predicted spectra
        experimental_spectra: List of experimental spectra
        methods: Comparison methods to use
        use_gpu: Whether to use GPU acceleration

    Returns:
        List of spectral comparison results
    """
    if len(predicted_spectra) != len(experimental_spectra):
        raise ValueError("Number of predicted and experimental spectra must match")

    comparator = SpectralComparator(use_gpu=use_gpu)
    comparisons = []

    for pred, exp in zip(predicted_spectra, experimental_spectra):
        try:
            comparison = comparator.compare_spectra(pred, exp, methods=methods)
            comparisons.append(comparison)
        except Exception as e:
            logger.error(f"Failed to compare spectra: {e}")
            # Create empty comparison for failed cases
            empty_comparison = SpectralComparison(
                predicted_spectrum=pred,
                experimental_spectrum=exp,
                overall_similarity=0.0,
            )
            comparisons.append(empty_comparison)

    return comparisons


def generate_comparison_report(
    comparison: SpectralComparison, include_plots: bool = True, output_dir: str = "."
) -> Dict[str, Any]:
    """
    Generate comprehensive comparison report.

    Args:
        comparison: Spectral comparison results
        include_plots: Whether to generate plots
        output_dir: Output directory for files

    Returns:
        Report dictionary
    """
    report = {
        "summary": {
            "overall_similarity": comparison.overall_similarity,
            "alignment_offset": comparison.alignment_offset,
            "computation_time": comparison.computation_time,
            "discrepancy_count": len(comparison.discrepancy_regions),
        },
        "metrics": comparison.comparison_metrics,
        "discrepancies": [],
        "peak_analysis": comparison.peak_matching_results,
        "statistical_tests": comparison.statistical_tests,
        "quality_assessment": comparison.quality_assessment,
    }

    # Format discrepancy information
    for region in comparison.discrepancy_regions:
        report["discrepancies"].append(
            {
                "frequency_range": [region.frequency_start, region.frequency_end],
                "type": region.discrepancy_type.value,
                "severity": region.severity,
                "significance": region.statistical_significance,
                "description": region.description,
                "explanation": region.suggested_explanation,
            }
        )

    # Generate plots if requested
    if include_plots and MATPLOTLIB_AVAILABLE:
        visualizer = SpectralVisualizer()
        plot_files = []

        try:
            # Overlay plot
            overlay_path = f"{output_dir}/spectral_overlay.png"
            visualizer.create_comparison_plot(
                comparison,
                VisualizationType.OVERLAY_PLOT,
                save_path=overlay_path,
                show_plot=False,
            )
            plot_files.append(overlay_path)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to create overlay plot: {e}")

        try:
            # Difference plot
            diff_path = f"{output_dir}/spectral_difference.png"
            visualizer.create_comparison_plot(
                comparison,
                VisualizationType.DIFFERENCE_PLOT,
                save_path=diff_path,
                show_plot=False,
            )
            plot_files.append(diff_path)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to create difference plot: {e}")

        try:
            # Discrepancy map
            discrepancy_path = f"{output_dir}/discrepancy_map.png"
            visualizer.create_comparison_plot(
                comparison,
                VisualizationType.DISCREPANCY_MAP,
                save_path=discrepancy_path,
                show_plot=False,
            )
            plot_files.append(discrepancy_path)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to create discrepancy map: {e}")

        report["plot_files"] = plot_files

    return report


# Exception classes
class SpectralComparisonError(Exception):
    """Base exception for spectral comparison errors."""

    pass


class AlignmentError(SpectralComparisonError):
    """Error in spectral alignment."""

    pass


class VisualizationError(SpectralComparisonError):
    """Error in spectral visualization."""

    pass

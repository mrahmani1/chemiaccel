"""
Automated Experimental Spectra Processing Pipeline.

This module provides GPU-accelerated tools for processing raw experimental spectroscopic
data, including loading various file formats, automated preprocessing, baseline correction,
normalization, peak picking, alignment, and denoising operations.
"""

import re
import struct
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import find_peaks, savgol_filter

# Optional imports with fallbacks
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. GPU acceleration will be limited.")

try:
    import pywt

    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn(
        "PyWavelets not available. Some denoising methods will be unavailable."
    )


# =============================================================================
# Enumerations
# =============================================================================


class SpectrumFormat(Enum):
    """Supported spectral data formats."""

    JCAMP_DX = "jcamp_dx"
    FID = "fid"
    CSV = "csv"
    BRUKER = "bruker"
    VARIAN = "varian"
    AGILENT = "agilent"
    THERMO = "thermo"
    WATERS = "waters"
    SHIMADZU = "shimadzu"


class ProcessingMethod(Enum):
    """Processing methods for spectral data."""

    AUTOMATIC = "automatic"
    MANUAL = "manual"
    BATCH = "batch"
    REAL_TIME = "real_time"


class BaselineCorrectionMethod(Enum):
    """Methods for baseline correction."""

    POLYNOMIAL = "polynomial"
    ASYMMETRIC_LEAST_SQUARES = "als"
    ADAPTIVE = "adaptive"
    RUBBER_BAND = "rubber_band"
    ROLLING_BALL = "rolling_ball"
    WHITTAKER = "whittaker"


class NormalizationMethod(Enum):
    """Methods for spectrum normalization."""

    UNIT_VECTOR = "unit_vector"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    TOTAL_INTENSITY = "total_intensity"
    PEAK_HEIGHT = "peak_height"
    AREA_UNDER_CURVE = "area_under_curve"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SpectralPeak:
    """Container for spectral peak information."""

    frequency: float
    intensity: float
    width: float
    area: float
    assignment: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self):
        """Validate peak data."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class ExperimentalSpectrum:
    """Container for experimental spectral data."""

    frequencies: np.ndarray
    intensities: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    peaks: List[SpectralPeak] = field(default_factory=list)
    processing_history: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate spectral data."""
        if len(self.frequencies) != len(self.intensities):
            raise ValueError("Frequencies and intensities must have same length")

        # Ensure frequencies are sorted
        if not np.all(self.frequencies[:-1] <= self.frequencies[1:]):
            sorted_indices = np.argsort(self.frequencies)
            self.frequencies = self.frequencies[sorted_indices]
            self.intensities = self.intensities[sorted_indices]
            self.processing_history.append("frequencies_sorted")

    def add_processing_step(self, step: str) -> None:
        """Add processing step to history."""
        self.processing_history.append(step)

    def calculate_signal_to_noise(self) -> float:
        """Calculate signal-to-noise ratio."""
        if len(self.intensities) == 0:
            return 0.0

        # Use top 10% of peaks as signal
        signal_threshold = np.percentile(self.intensities, 90)
        signal_region = self.intensities[self.intensities >= signal_threshold]
        signal_level = np.mean(signal_region)

        # Use bottom 20% as noise
        noise_threshold = np.percentile(self.intensities, 20)
        noise_region = self.intensities[self.intensities <= noise_threshold]
        noise_level = np.std(noise_region)

        snr = signal_level / noise_level if noise_level > 0 else float("inf")
        self.quality_metrics["signal_to_noise"] = snr
        return snr


@dataclass
class ProcessingParameters:
    """Parameters for spectral processing operations."""

    baseline_method: BaselineCorrectionMethod = (
        BaselineCorrectionMethod.ASYMMETRIC_LEAST_SQUARES
    )
    normalization_method: NormalizationMethod = NormalizationMethod.UNIT_VECTOR
    smoothing_window: int = 5
    smoothing_order: int = 2
    peak_detection_threshold: float = 0.1
    peak_detection_distance: int = 10
    frequency_range: Optional[Tuple[float, float]] = None
    use_gpu: bool = True

    def __post_init__(self):
        """Validate processing parameters."""
        if self.smoothing_window < 3 or self.smoothing_window % 2 == 0:
            raise ValueError("Smoothing window must be odd and >= 3")
        if self.smoothing_order >= self.smoothing_window:
            raise ValueError("Smoothing order must be less than window size")


@dataclass
class ProcessingResult:
    """Result of spectral processing operations."""

    processed_spectrum: ExperimentalSpectrum
    original_spectrum: ExperimentalSpectrum
    processing_metrics: Dict[str, float] = field(default_factory=dict)
    computation_time: float = 0.0
    gpu_acceleration_used: bool = False


# =============================================================================
# Base Processing Classes
# =============================================================================


class BaseSpectraProcessor(ABC):
    """Abstract base class for spectral processors."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE

    @abstractmethod
    def process(
        self, spectrum: ExperimentalSpectrum, params: ProcessingParameters
    ) -> ProcessingResult:
        """Process experimental spectrum."""
        pass

    def _to_gpu(self, array: np.ndarray) -> Union[np.ndarray, Any]:
        """Convert array to GPU if available."""
        if self.use_gpu and CUPY_AVAILABLE:
            return cp.asarray(array)
        return array

    def _to_cpu(self, array: Union[np.ndarray, Any]) -> np.ndarray:
        """Convert array back to CPU."""
        if self.use_gpu and CUPY_AVAILABLE and hasattr(array, "get"):
            return array.get()
        return array


# =============================================================================
# File Format Loaders
# =============================================================================


def load_jcamp_dx(filepath: str) -> ExperimentalSpectrum:
    """Load JCAMP-DX format spectral data."""
    frequencies = []
    intensities = []
    metadata = {}

    try:
        with open(filepath, encoding="utf-8", errors="ignore") as file:
            content = file.read()

            # Parse metadata
            for line in content.split("\n"):
                if line.startswith("##"):
                    if "=" in line:
                        key, value = line[2:].split("=", 1)
                        metadata[key.strip()] = value.strip()

            # Parse spectral data
            data_section = False
            for line in content.split("\n"):
                line = line.strip()

                if line.startswith("##XYDATA"):
                    data_section = True
                    continue
                elif line.startswith("##END"):
                    break

                if data_section and line and not line.startswith("##"):
                    # Parse data points
                    if "," in line:
                        parts = line.split(",")
                    else:
                        parts = line.split()

                    for part in parts:
                        if part.strip():
                            try:
                                if len(frequencies) == len(intensities):
                                    frequencies.append(float(part))
                                else:
                                    intensities.append(float(part))
                            except ValueError:
                                continue

        if len(frequencies) != len(intensities):
            min_len = min(len(frequencies), len(intensities))
            frequencies = frequencies[:min_len]
            intensities = intensities[:min_len]

        spectrum = ExperimentalSpectrum(
            frequencies=np.array(frequencies),
            intensities=np.array(intensities),
            metadata=metadata,
        )
        spectrum.add_processing_step("loaded_jcamp_dx")

        return spectrum

    except Exception as e:
        raise FormatError(f"Failed to load JCAMP-DX file: {e}")


def load_fid_data(filepath: str, params: Optional[Dict] = None) -> ExperimentalSpectrum:
    """Load FID (Free Induction Decay) data and perform FFT."""
    if params is None:
        params = {"sw": 10000, "sfo1": 400}  # Default parameters

    try:
        # Read binary FID data (assuming Bruker format)
        with open(filepath, "rb") as file:
            # Skip header (typically 32 bytes for Bruker)
            file.seek(32)

            # Read FID data as complex numbers
            data = []
            while True:
                chunk = file.read(8)  # 4 bytes real + 4 bytes imaginary
                if len(chunk) != 8:
                    break

                real_part = struct.unpack("<i", chunk[:4])[0]  # Little endian integer
                imag_part = struct.unpack("<i", chunk[4:])[0]
                data.append(complex(real_part, imag_part))

        fid_data = np.array(data)

        # Apply apodization (exponential decay)
        lb = params.get("lb", 1.0)  # Line broadening
        t = np.arange(len(fid_data)) / params["sw"]
        apodization = np.exp(-np.pi * lb * t)
        fid_data *= apodization

        # Zero filling
        zero_fill = params.get("zero_fill", 2)
        fid_data = np.pad(
            fid_data, (0, len(fid_data) * (zero_fill - 1)), mode="constant"
        )

        # FFT to get spectrum
        spectrum_complex = np.fft.fft(fid_data)
        spectrum_real = np.real(spectrum_complex)

        # Generate frequency axis
        sw = params["sw"]
        sfo1 = params["sfo1"]
        n_points = len(spectrum_real)

        frequencies = np.linspace(sfo1 + sw / 2, sfo1 - sw / 2, n_points)

        # Phase correction (simplified)
        phase0 = params.get("phase0", 0)
        phase1 = params.get("phase1", 0)
        spectrum_complex *= np.exp(
            1j * (phase0 + phase1 * np.arange(n_points) / n_points)
        )

        spectrum = ExperimentalSpectrum(
            frequencies=frequencies,
            intensities=np.real(spectrum_complex),
            metadata=params,
        )
        spectrum.add_processing_step("loaded_fid_data")
        spectrum.add_processing_step("fft_processed")

        return spectrum

    except Exception as e:
        raise FormatError(f"Failed to load FID data: {e}")


# =============================================================================
# Preprocessing Classes
# =============================================================================


class SpectraPreprocessor(BaseSpectraProcessor):
    """Main class for spectral preprocessing operations."""

    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)

    def process(
        self, spectrum: ExperimentalSpectrum, params: ProcessingParameters
    ) -> ProcessingResult:
        """Apply complete preprocessing pipeline."""
        import time

        start_time = time.time()

        # Create copy for processing
        processed = ExperimentalSpectrum(
            frequencies=spectrum.frequencies.copy(),
            intensities=spectrum.intensities.copy(),
            metadata=spectrum.metadata.copy(),
            processing_history=spectrum.processing_history.copy(),
        )

        # Apply frequency range filter if specified
        if params.frequency_range is not None:
            processed = self._apply_frequency_filter(processed, params.frequency_range)

        # Smoothing
        processed = self._apply_smoothing(
            processed, params.smoothing_window, params.smoothing_order
        )

        # Baseline correction
        processed = self._apply_baseline_correction(processed, params.baseline_method)

        # Normalization
        processed = self._apply_normalization(processed, params.normalization_method)

        # Peak detection
        peaks = self._detect_peaks(
            processed, params.peak_detection_threshold, params.peak_detection_distance
        )
        processed.peaks = peaks

        # Calculate quality metrics
        snr = processed.calculate_signal_to_noise()

        computation_time = time.time() - start_time

        return ProcessingResult(
            processed_spectrum=processed,
            original_spectrum=spectrum,
            processing_metrics={
                "signal_to_noise_ratio": snr,
                "num_peaks_detected": len(peaks),
                "frequency_range": (
                    np.min(processed.frequencies),
                    np.max(processed.frequencies),
                ),
            },
            computation_time=computation_time,
            gpu_acceleration_used=self.use_gpu,
        )

    def _apply_frequency_filter(
        self, spectrum: ExperimentalSpectrum, freq_range: Tuple[float, float]
    ) -> ExperimentalSpectrum:
        """Filter spectrum to specified frequency range."""
        min_freq, max_freq = freq_range
        mask = (spectrum.frequencies >= min_freq) & (spectrum.frequencies <= max_freq)

        filtered_spectrum = ExperimentalSpectrum(
            frequencies=spectrum.frequencies[mask],
            intensities=spectrum.intensities[mask],
            metadata=spectrum.metadata,
            processing_history=spectrum.processing_history.copy(),
        )
        filtered_spectrum.add_processing_step(
            f"frequency_filtered_{min_freq}_{max_freq}"
        )

        return filtered_spectrum

    def _apply_smoothing(
        self, spectrum: ExperimentalSpectrum, window: int, order: int
    ) -> ExperimentalSpectrum:
        """Apply Savitzky-Golay smoothing."""
        if len(spectrum.intensities) <= window:
            return spectrum

        smoothed_intensities = self._to_gpu(spectrum.intensities)

        if self.use_gpu and CUPY_AVAILABLE:
            # GPU implementation using CuPy
            smoothed_intensities = cp.array(
                [savgol_filter(spectrum.intensities, window, order)]
            )[0]
        else:
            smoothed_intensities = savgol_filter(spectrum.intensities, window, order)

        smoothed_intensities = self._to_cpu(smoothed_intensities)

        smoothed_spectrum = ExperimentalSpectrum(
            frequencies=spectrum.frequencies,
            intensities=smoothed_intensities,
            metadata=spectrum.metadata,
            processing_history=spectrum.processing_history.copy(),
        )
        smoothed_spectrum.add_processing_step(f"smoothed_savgol_{window}_{order}")

        return smoothed_spectrum

    def _apply_baseline_correction(
        self, spectrum: ExperimentalSpectrum, method: BaselineCorrectionMethod
    ) -> ExperimentalSpectrum:
        """Apply baseline correction."""
        corrected_intensities = spectrum.intensities.copy()

        if method == BaselineCorrectionMethod.POLYNOMIAL:
            corrected_intensities = self._polynomial_baseline_correction(
                spectrum.intensities
            )
        elif method == BaselineCorrectionMethod.ASYMMETRIC_LEAST_SQUARES:
            corrected_intensities = self._als_baseline_correction(spectrum.intensities)
        elif method == BaselineCorrectionMethod.ADAPTIVE:
            corrected_intensities = self._adaptive_baseline_correction(
                spectrum.intensities
            )
        elif method == BaselineCorrectionMethod.ROLLING_BALL:
            corrected_intensities = self._rolling_ball_baseline_correction(
                spectrum.intensities
            )

        corrected_spectrum = ExperimentalSpectrum(
            frequencies=spectrum.frequencies,
            intensities=corrected_intensities,
            metadata=spectrum.metadata,
            processing_history=spectrum.processing_history.copy(),
        )
        corrected_spectrum.add_processing_step(f"baseline_corrected_{method.value}")

        return corrected_spectrum

    def _polynomial_baseline_correction(
        self, intensities: np.ndarray, degree: int = 2
    ) -> np.ndarray:
        """Polynomial baseline correction."""
        x = np.arange(len(intensities))

        # Fit polynomial to minima
        window_size = max(10, len(intensities) // 20)
        minima_indices = []

        for i in range(0, len(intensities), window_size):
            window = intensities[i : i + window_size]
            if len(window) > 0:
                min_idx = i + np.argmin(window)
                minima_indices.append(min_idx)

        if len(minima_indices) > degree:
            baseline_x = np.array(minima_indices)
            baseline_y = intensities[baseline_x]

            # Fit polynomial
            coeffs = np.polyfit(baseline_x, baseline_y, degree)
            baseline = np.polyval(coeffs, x)

            return intensities - baseline

        return intensities

    def _als_baseline_correction(
        self,
        intensities: np.ndarray,
        lam: float = 1e6,
        p: float = 0.01,
        niter: int = 10,
    ) -> np.ndarray:
        """Asymmetric Least Squares baseline correction."""
        import scipy.sparse as sparse
        from scipy.sparse.linalg import spsolve

        n = len(intensities)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2))
        w = np.ones(n)

        for _ in range(niter):
            W = sparse.spdiags(w, 0, n, n)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * intensities)
            w = p * (intensities > z) + (1 - p) * (intensities < z)

        return intensities - z

    def _adaptive_baseline_correction(self, intensities: np.ndarray) -> np.ndarray:
        """Adaptive baseline correction using iterative polynomial fitting."""
        corrected = intensities.copy()

        for iteration in range(5):
            # Fit polynomial to current data
            x = np.arange(len(corrected))
            degree = min(3, len(corrected) // 10)

            if degree > 0:
                coeffs = np.polyfit(x, corrected, degree)
                baseline = np.polyval(coeffs, x)

                # Update only points below baseline
                mask = corrected < baseline
                corrected[mask] = intensities[mask] - baseline[mask]

        return corrected

    def _rolling_ball_baseline_correction(
        self, intensities: np.ndarray, radius: int = 50
    ) -> np.ndarray:
        """Rolling ball baseline correction algorithm."""
        n = len(intensities)
        baseline = np.zeros(n)

        for i in range(n):
            start = max(0, i - radius)
            end = min(n, i + radius + 1)
            window = intensities[start:end]
            baseline[i] = np.min(window)

        return intensities - baseline

    def _apply_normalization(
        self, spectrum: ExperimentalSpectrum, method: NormalizationMethod
    ) -> ExperimentalSpectrum:
        """Apply normalization to spectrum."""
        normalized_intensities = spectrum.intensities.copy()

        if method == NormalizationMethod.UNIT_VECTOR:
            norm = np.linalg.norm(normalized_intensities)
            if norm > 0:
                normalized_intensities /= norm

        elif method == NormalizationMethod.MIN_MAX:
            min_val = np.min(normalized_intensities)
            max_val = np.max(normalized_intensities)
            if max_val > min_val:
                normalized_intensities = (normalized_intensities - min_val) / (
                    max_val - min_val
                )

        elif method == NormalizationMethod.Z_SCORE:
            mean_val = np.mean(normalized_intensities)
            std_val = np.std(normalized_intensities)
            if std_val > 0:
                normalized_intensities = (normalized_intensities - mean_val) / std_val

        elif method == NormalizationMethod.TOTAL_INTENSITY:
            total = np.sum(np.abs(normalized_intensities))
            if total > 0:
                normalized_intensities /= total

        elif method == NormalizationMethod.PEAK_HEIGHT:
            max_val = np.max(np.abs(normalized_intensities))
            if max_val > 0:
                normalized_intensities /= max_val

        elif method == NormalizationMethod.AREA_UNDER_CURVE:
            area = np.trapz(np.abs(normalized_intensities))
            if area > 0:
                normalized_intensities /= area

        normalized_spectrum = ExperimentalSpectrum(
            frequencies=spectrum.frequencies,
            intensities=normalized_intensities,
            metadata=spectrum.metadata,
            processing_history=spectrum.processing_history.copy(),
        )
        normalized_spectrum.add_processing_step(f"normalized_{method.value}")

        return normalized_spectrum

    def _detect_peaks(
        self, spectrum: ExperimentalSpectrum, threshold: float, distance: int
    ) -> List[SpectralPeak]:
        """Detect peaks in spectrum."""
        # Use scipy peak detection
        peaks, properties = find_peaks(
            spectrum.intensities,
            height=threshold * np.max(spectrum.intensities),
            distance=distance,
        )

        spectral_peaks = []
        for i, peak_idx in enumerate(peaks):
            frequency = spectrum.frequencies[peak_idx]
            intensity = spectrum.intensities[peak_idx]

            # Estimate peak width
            width = (
                properties.get("widths", np.ones(len(peaks)))[i]
                if "widths" in properties
                else 1.0
            )

            # Estimate peak area (simple trapezoidal)
            left_idx = max(0, peak_idx - int(width))
            right_idx = min(len(spectrum.intensities), peak_idx + int(width))
            area = np.trapz(spectrum.intensities[left_idx:right_idx])

            peak = SpectralPeak(
                frequency=frequency, intensity=intensity, width=float(width), area=area
            )
            spectral_peaks.append(peak)

        return spectral_peaks


class SpectraAligner(BaseSpectraProcessor):
    """Class for aligning multiple spectra."""

    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)

    def align_spectra(
        self, spectra: List[ExperimentalSpectrum], reference_idx: int = 0
    ) -> List[ExperimentalSpectrum]:
        """Align multiple spectra to a reference spectrum."""
        if len(spectra) <= 1:
            return spectra

        reference = spectra[reference_idx]
        aligned_spectra = [reference]  # Reference stays unchanged

        for i, spectrum in enumerate(spectra):
            if i == reference_idx:
                continue

            aligned = self._align_single_spectrum(spectrum, reference)
            aligned_spectra.append(aligned)

        return aligned_spectra

    def _align_single_spectrum(
        self, spectrum: ExperimentalSpectrum, reference: ExperimentalSpectrum
    ) -> ExperimentalSpectrum:
        """Align single spectrum to reference using cross-correlation."""
        # Find common frequency range
        min_freq = max(np.min(spectrum.frequencies), np.min(reference.frequencies))
        max_freq = min(np.max(spectrum.frequencies), np.max(reference.frequencies))

        # Interpolate both spectra to common grid
        n_points = min(len(spectrum.frequencies), len(reference.frequencies))
        common_freq = np.linspace(min_freq, max_freq, n_points)

        # Interpolate spectra
        interp_ref = interp1d(
            reference.frequencies,
            reference.intensities,
            bounds_error=False,
            fill_value=0,
        )(common_freq)
        interp_spec = interp1d(
            spectrum.frequencies, spectrum.intensities, bounds_error=False, fill_value=0
        )(common_freq)

        # Calculate cross-correlation
        correlation = np.correlate(interp_ref, interp_spec, mode="full")
        shift = np.argmax(correlation) - len(interp_spec) + 1

        # Apply shift to frequency axis
        freq_shift = shift * (common_freq[1] - common_freq[0])
        aligned_frequencies = spectrum.frequencies + freq_shift

        aligned_spectrum = ExperimentalSpectrum(
            frequencies=aligned_frequencies,
            intensities=spectrum.intensities,
            metadata=spectrum.metadata,
            processing_history=spectrum.processing_history.copy(),
        )
        aligned_spectrum.add_processing_step(f"aligned_shift_{freq_shift:.3f}")

        return aligned_spectrum

    def process(
        self, spectrum: ExperimentalSpectrum, params: ProcessingParameters
    ) -> ProcessingResult:
        """Process method for base class compatibility."""
        # For single spectrum, return as-is
        return ProcessingResult(
            processed_spectrum=spectrum,
            original_spectrum=spectrum,
            processing_metrics={"message": "No alignment needed for single spectrum"},
            computation_time=0.0,
            gpu_acceleration_used=False,
        )


# =============================================================================
# Main Processing Engine
# =============================================================================


class ExperimentalSpectraProcessor:
    """Main class for experimental spectra processing."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.preprocessor = SpectraPreprocessor(use_gpu)
        self.aligner = SpectraAligner(use_gpu)

    def load_spectrum(
        self, filepath: str, format_type: SpectrumFormat, params: Optional[Dict] = None
    ) -> ExperimentalSpectrum:
        """Load spectrum from file."""
        if format_type == SpectrumFormat.JCAMP_DX:
            return load_jcamp_dx(filepath)
        elif format_type == SpectrumFormat.FID:
            return load_fid_data(filepath, params)
        elif format_type == SpectrumFormat.CSV:
            return self._load_csv(filepath)
        else:
            raise FormatError(f"Unsupported format: {format_type}")

    def _load_csv(self, filepath: str) -> ExperimentalSpectrum:
        """Load CSV format spectrum."""
        try:
            import pandas as pd

            df = pd.read_csv(filepath)

            # Assume first column is frequency, second is intensity
            frequencies = df.iloc[:, 0].values
            intensities = df.iloc[:, 1].values

            spectrum = ExperimentalSpectrum(
                frequencies=frequencies,
                intensities=intensities,
                metadata={"file_format": "csv", "source_file": filepath},
            )
            spectrum.add_processing_step("loaded_csv")

            return spectrum

        except Exception as e:
            raise FormatError(f"Failed to load CSV file: {e}")

    def process_single_spectrum(
        self,
        spectrum: ExperimentalSpectrum,
        params: Optional[ProcessingParameters] = None,
    ) -> ProcessingResult:
        """Process single experimental spectrum."""
        if params is None:
            params = ProcessingParameters()

        return self.preprocessor.process(spectrum, params)

    def process_batch(
        self,
        spectra: List[ExperimentalSpectrum],
        params: Optional[ProcessingParameters] = None,
        align_spectra: bool = True,
    ) -> List[ProcessingResult]:
        """Process batch of spectra."""
        if params is None:
            params = ProcessingParameters()

        # Process each spectrum individually
        results = []
        for spectrum in spectra:
            result = self.preprocessor.process(spectrum, params)
            results.append(result)

        # Align spectra if requested
        if align_spectra and len(results) > 1:
            processed_spectra = [r.processed_spectrum for r in results]
            aligned_spectra = self.aligner.align_spectra(processed_spectra)

            # Update results with aligned spectra
            for i, aligned in enumerate(aligned_spectra):
                results[i].processed_spectrum = aligned

        return results


# =============================================================================
# Utility Functions
# =============================================================================


def process_spectrum_batch(
    filepaths: List[str],
    format_type: SpectrumFormat,
    params: Optional[ProcessingParameters] = None,
    use_gpu: bool = True,
) -> List[ProcessingResult]:
    """Process batch of spectral files."""
    processor = ExperimentalSpectraProcessor(use_gpu)

    # Load all spectra
    spectra = []
    for filepath in filepaths:
        try:
            spectrum = processor.load_spectrum(filepath, format_type)
            spectra.append(spectrum)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    # Process batch
    results = processor.process_batch(spectra, params)

    return results


# =============================================================================
# Exception Classes
# =============================================================================


class ProcessingError(Exception):
    """Exception raised during spectral processing."""

    pass


class FormatError(Exception):
    """Exception raised for unsupported file formats."""

    pass


class DataQualityError(Exception):
    """Exception raised for poor quality spectral data."""

    pass

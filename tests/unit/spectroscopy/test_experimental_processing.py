"""
Unit tests for experimental spectra processing.

This module tests the GPU-accelerated experimental data processing pipeline,
including file loading, preprocessing, baseline correction, normalization,
and spectral alignment.
"""

import os
import tempfile
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest
from chemi_accel.spectroscopy.experimental_processing import (
    BaselineCorrectionMethod,
    DataQualityError,
    ExperimentalSpectraProcessor,
    ExperimentalSpectrum,
    FormatError,
    NormalizationMethod,
    ProcessingError,
    ProcessingMethod,
    ProcessingParameters,
    ProcessingResult,
    SpectraAligner,
    SpectralPeak,
    SpectraPreprocessor,
    SpectrumFormat,
    load_fid_data,
    load_jcamp_dx,
    process_spectrum_batch,
)


class TestSpectralPeak:
    """Test SpectralPeak data structure."""

    def test_basic_peak(self):
        """Test basic spectral peak creation."""
        peak = SpectralPeak(frequency=1500.0, intensity=0.8, width=10.0, area=8.0)

        assert peak.frequency == 1500.0
        assert peak.intensity == 0.8
        assert peak.width == 10.0
        assert peak.area == 8.0
        assert peak.assignment is None
        assert peak.confidence == 1.0

    def test_peak_with_assignment(self):
        """Test peak with assignment information."""
        peak = SpectralPeak(
            frequency=3000.0,
            intensity=1.0,
            width=20.0,
            area=20.0,
            assignment="C-H stretch",
            confidence=0.9,
        )

        assert peak.assignment == "C-H stretch"
        assert peak.confidence == 0.9

    def test_peak_confidence_validation(self):
        """Test peak confidence validation."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            SpectralPeak(
                frequency=1500.0, intensity=0.8, width=10.0, area=8.0, confidence=1.5
            )


class TestExperimentalSpectrum:
    """Test ExperimentalSpectrum data structure."""

    def test_basic_spectrum(self):
        """Test basic experimental spectrum creation."""
        frequencies = np.linspace(400, 4000, 1000)
        intensities = np.random.rand(1000)

        spectrum = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        assert len(spectrum.frequencies) == 1000
        assert len(spectrum.intensities) == 1000
        assert len(spectrum.metadata) == 0
        assert len(spectrum.peaks) == 0
        assert len(spectrum.processing_history) == 0

    def test_spectrum_with_metadata(self):
        """Test spectrum with metadata."""
        frequencies = np.linspace(0, 12, 500)
        intensities = np.random.rand(500)
        metadata = {
            "instrument": "Bruker AVANCE",
            "temperature": 298.15,
            "solvent": "CDCl3",
        }

        spectrum = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities, metadata=metadata
        )

        assert spectrum.metadata["instrument"] == "Bruker AVANCE"
        assert spectrum.metadata["temperature"] == 298.15
        assert spectrum.metadata["solvent"] == "CDCl3"

    def test_spectrum_validation(self):
        """Test spectrum data validation."""
        frequencies = np.linspace(400, 4000, 1000)
        intensities = np.random.rand(500)  # Wrong length

        with pytest.raises(
            ValueError, match="Frequencies and intensities must have same length"
        ):
            ExperimentalSpectrum(frequencies=frequencies, intensities=intensities)

    def test_frequency_sorting(self):
        """Test automatic frequency sorting."""
        frequencies = np.array([4000, 1000, 2000, 3000])
        intensities = np.array([0.1, 0.8, 0.5, 0.3])

        spectrum = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        # Should be automatically sorted
        assert np.array_equal(spectrum.frequencies, [1000, 2000, 3000, 4000])
        assert np.array_equal(spectrum.intensities, [0.8, 0.5, 0.3, 0.1])
        assert "frequencies_sorted" in spectrum.processing_history

    def test_add_processing_step(self):
        """Test adding processing steps."""
        spectrum = ExperimentalSpectrum(
            frequencies=np.linspace(0, 4000, 100), intensities=np.random.rand(100)
        )

        spectrum.add_processing_step("baseline_corrected")
        spectrum.add_processing_step("normalized")

        assert len(spectrum.processing_history) == 2
        assert "baseline_corrected" in spectrum.processing_history
        assert "normalized" in spectrum.processing_history

    def test_signal_to_noise_calculation(self):
        """Test signal-to-noise ratio calculation."""
        # Create spectrum with known noise characteristics
        frequencies = np.linspace(0, 4000, 1000)

        # Base noise level
        intensities = np.random.normal(0.1, 0.05, 1000)

        # Add some peaks
        peak_indices = [200, 400, 600, 800]
        for idx in peak_indices:
            intensities[idx] += 0.5  # Add signal

        spectrum = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        snr = spectrum.calculate_signal_to_noise()

        assert snr > 0
        assert "signal_to_noise" in spectrum.quality_metrics
        assert spectrum.quality_metrics["signal_to_noise"] == snr


class TestProcessingParameters:
    """Test ProcessingParameters structure."""

    def test_default_parameters(self):
        """Test default processing parameters."""
        params = ProcessingParameters()

        assert (
            params.baseline_method == BaselineCorrectionMethod.ASYMMETRIC_LEAST_SQUARES
        )
        assert params.normalization_method == NormalizationMethod.UNIT_VECTOR
        assert params.smoothing_window == 5
        assert params.smoothing_order == 2
        assert params.peak_detection_threshold == 0.1
        assert params.use_gpu is True

    def test_custom_parameters(self):
        """Test custom parameters."""
        params = ProcessingParameters(
            baseline_method=BaselineCorrectionMethod.POLYNOMIAL,
            normalization_method=NormalizationMethod.MIN_MAX,
            smoothing_window=7,
            smoothing_order=3,
            frequency_range=(500.0, 3500.0),
        )

        assert params.baseline_method == BaselineCorrectionMethod.POLYNOMIAL
        assert params.normalization_method == NormalizationMethod.MIN_MAX
        assert params.smoothing_window == 7
        assert params.smoothing_order == 3
        assert params.frequency_range == (500.0, 3500.0)

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="Smoothing window must be odd"):
            ProcessingParameters(smoothing_window=6)

        with pytest.raises(
            ValueError, match="Smoothing order must be less than window size"
        ):
            ProcessingParameters(smoothing_window=5, smoothing_order=5)


class TestProcessingResult:
    """Test ProcessingResult structure."""

    def test_basic_result(self):
        """Test basic processing result."""
        original = ExperimentalSpectrum(
            frequencies=np.linspace(0, 4000, 100), intensities=np.random.rand(100)
        )

        processed = ExperimentalSpectrum(
            frequencies=original.frequencies,
            intensities=original.intensities * 0.9,  # Some processing
        )

        result = ProcessingResult(
            processed_spectrum=processed,
            original_spectrum=original,
            computation_time=1.5,
            gpu_acceleration_used=True,
        )

        assert result.processed_spectrum == processed
        assert result.original_spectrum == original
        assert result.computation_time == 1.5
        assert result.gpu_acceleration_used is True


class TestFileLoaders:
    """Test file format loaders."""

    def test_load_jcamp_dx_basic(self):
        """Test basic JCAMP-DX file loading."""
        jcamp_content = """##TITLE=Test Spectrum
##JCAMP-DX=4.24
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=400
##LASTX=4000
##NPOINTS=5
##XYDATA=(X++(Y..Y))
400 0.1
1000 0.5
2000 0.8
3000 0.3
4000 0.1
##END=
"""

        with patch("builtins.open", mock_open(read_data=jcamp_content)):
            spectrum = load_jcamp_dx("test.jdx")

            assert len(spectrum.frequencies) == 5
            assert len(spectrum.intensities) == 5
            assert spectrum.frequencies[0] == 400
            assert spectrum.frequencies[-1] == 4000
            assert "loaded_jcamp_dx" in spectrum.processing_history

    def test_load_jcamp_dx_with_metadata(self):
        """Test JCAMP-DX loading with metadata."""
        jcamp_content = """##TITLE=Ethanol IR Spectrum
##JCAMP-DX=4.24
##ORIGIN=Test Lab
##OWNER=Test User
##SPECTROMETER/DATA SYSTEM=Bruker ALPHA
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##XYDATA=(X++(Y..Y))
1000 0.5
2000 0.8, 3000 0.3
##END=
"""

        with patch("builtins.open", mock_open(read_data=jcamp_content)):
            spectrum = load_jcamp_dx("test.jdx")

            assert spectrum.metadata["TITLE"] == "Ethanol IR Spectrum"
            assert spectrum.metadata["ORIGIN"] == "Test Lab"
            assert spectrum.metadata["SPECTROMETER/DATA SYSTEM"] == "Bruker ALPHA"

    def test_load_fid_data_basic(self):
        """Test basic FID data loading."""
        # Create mock FID data (simplified)
        with patch("builtins.open", mock_open()):
            with patch("struct.unpack") as mock_unpack:
                # Mock complex FID data
                mock_unpack.side_effect = [
                    (1000,),
                    (500,),  # First complex point
                    (800,),
                    (300,),  # Second complex point
                    (600,),
                    (100,),  # Third complex point
                ]

                with patch("numpy.fft.fft") as mock_fft:
                    mock_fft.return_value = np.array([1 + 0j, 0.5 + 0.3j, 0.2 + 0.1j])

                    params = {"sw": 10000, "sfo1": 400, "lb": 1.0}
                    spectrum = load_fid_data("test.fid", params)

                    assert spectrum is not None
                    assert "loaded_fid_data" in spectrum.processing_history
                    assert "fft_processed" in spectrum.processing_history

    def test_load_jcamp_dx_format_error(self):
        """Test JCAMP-DX format error handling."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FormatError, match="Failed to load JCAMP-DX file"):
                load_jcamp_dx("nonexistent.jdx")


class TestSpectraPreprocessor:
    """Test spectral preprocessing operations."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = SpectraPreprocessor(use_gpu=False)
        assert preprocessor.use_gpu is False

    def test_complete_processing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create test spectrum with known characteristics
        frequencies = np.linspace(400, 4000, 1000)

        # Create spectrum with baseline drift and noise
        baseline = 0.1 + 0.05 * np.sin(frequencies / 1000)
        signal = np.zeros_like(frequencies)

        # Add some peaks
        peak_positions = [1000, 1500, 2800, 3000]
        for pos in peak_positions:
            idx = np.argmin(np.abs(frequencies - pos))
            signal[idx] = 0.8

        # Add noise
        noise = np.random.normal(0, 0.02, len(frequencies))
        intensities = baseline + signal + noise

        spectrum = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        params = ProcessingParameters(
            frequency_range=(500.0, 3500.0),
            smoothing_window=5,
            baseline_method=BaselineCorrectionMethod.POLYNOMIAL,
            normalization_method=NormalizationMethod.MIN_MAX,
        )

        preprocessor = SpectraPreprocessor(use_gpu=False)
        result = preprocessor.process(spectrum, params)

        assert isinstance(result, ProcessingResult)
        assert result.processed_spectrum is not None
        assert result.computation_time > 0
        assert len(result.processed_spectrum.peaks) > 0

        # Check processing history
        history = result.processed_spectrum.processing_history
        assert any("frequency_filtered" in step for step in history)
        assert any("smoothed_savgol" in step for step in history)
        assert any("baseline_corrected" in step for step in history)
        assert any("normalized" in step for step in history)

    def test_frequency_filtering(self):
        """Test frequency range filtering."""
        frequencies = np.linspace(0, 5000, 1000)
        intensities = np.random.rand(1000)

        spectrum = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        preprocessor = SpectraPreprocessor(use_gpu=False)
        filtered = preprocessor._apply_frequency_filter(spectrum, (1000.0, 4000.0))

        assert np.min(filtered.frequencies) >= 1000.0
        assert np.max(filtered.frequencies) <= 4000.0
        assert len(filtered.frequencies) < len(spectrum.frequencies)

    @patch("scipy.signal.savgol_filter")
    def test_smoothing(self, mock_savgol):
        """Test spectral smoothing."""
        frequencies = np.linspace(400, 4000, 100)
        intensities = np.random.rand(100)

        spectrum = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        # Mock smoothing result
        mock_savgol.return_value = intensities * 0.9

        preprocessor = SpectraPreprocessor(use_gpu=False)
        smoothed = preprocessor._apply_smoothing(spectrum, 5, 2)

        mock_savgol.assert_called_once()
        assert "smoothed_savgol_5_2" in smoothed.processing_history

    def test_baseline_correction_polynomial(self):
        """Test polynomial baseline correction."""
        frequencies = np.linspace(400, 4000, 1000)

        # Create spectrum with polynomial baseline
        baseline = 0.1 + 0.0001 * frequencies + 0.000001 * frequencies**2
        signal = np.zeros_like(frequencies)
        signal[200] = 0.5  # Add a peak
        signal[600] = 0.8  # Add another peak

        intensities = baseline + signal

        preprocessor = SpectraPreprocessor(use_gpu=False)
        corrected = preprocessor._polynomial_baseline_correction(intensities)

        # Baseline should be reduced
        assert np.mean(corrected) < np.mean(intensities)

    def test_als_baseline_correction(self):
        """Test asymmetric least squares baseline correction."""
        frequencies = np.linspace(400, 4000, 100)

        # Create spectrum with baseline drift
        baseline = 0.2 + 0.1 * np.sin(frequencies / 1000)
        signal = np.zeros_like(frequencies)
        signal[30] = 0.8  # Add a peak

        intensities = baseline + signal

        preprocessor = SpectraPreprocessor(use_gpu=False)
        corrected = preprocessor._als_baseline_correction(intensities)

        # Should return corrected spectrum
        assert len(corrected) == len(intensities)
        assert not np.array_equal(corrected, intensities)

    def test_normalization_methods(self):
        """Test different normalization methods."""
        frequencies = np.linspace(400, 4000, 100)
        intensities = np.random.rand(100) * 2 + 1  # Values between 1 and 3

        spectrum = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        preprocessor = SpectraPreprocessor(use_gpu=False)

        # Test unit vector normalization
        unit_norm = preprocessor._apply_normalization(
            spectrum, NormalizationMethod.UNIT_VECTOR
        )
        assert abs(np.linalg.norm(unit_norm.intensities) - 1.0) < 1e-10

        # Test min-max normalization
        minmax_norm = preprocessor._apply_normalization(
            spectrum, NormalizationMethod.MIN_MAX
        )
        assert np.min(minmax_norm.intensities) >= 0.0
        assert np.max(minmax_norm.intensities) <= 1.0

        # Test z-score normalization
        zscore_norm = preprocessor._apply_normalization(
            spectrum, NormalizationMethod.Z_SCORE
        )
        assert abs(np.mean(zscore_norm.intensities)) < 1e-10
        assert abs(np.std(zscore_norm.intensities) - 1.0) < 1e-10

    @patch("scipy.signal.find_peaks")
    def test_peak_detection(self, mock_find_peaks):
        """Test peak detection."""
        frequencies = np.linspace(400, 4000, 100)
        intensities = np.random.rand(100)

        spectrum = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        # Mock peak detection results
        mock_find_peaks.return_value = (
            np.array([20, 40, 60]),
            {"widths": np.array([2, 3, 4])},
        )

        preprocessor = SpectraPreprocessor(use_gpu=False)
        peaks = preprocessor._detect_peaks(spectrum, 0.1, 10)

        assert len(peaks) == 3
        assert all(isinstance(peak, SpectralPeak) for peak in peaks)
        mock_find_peaks.assert_called_once()


class TestSpectraAligner:
    """Test spectral alignment operations."""

    def test_aligner_initialization(self):
        """Test aligner initialization."""
        aligner = SpectraAligner(use_gpu=False)
        assert aligner.use_gpu is False

    def test_align_single_spectrum(self):
        """Test alignment of single spectrum."""
        # Create reference spectrum
        ref_freq = np.linspace(400, 4000, 100)
        ref_int = np.random.rand(100)
        reference = ExperimentalSpectrum(frequencies=ref_freq, intensities=ref_int)

        # Create shifted spectrum
        shift = 10.0  # 10 cm-1 shift
        spec_freq = ref_freq + shift
        spec_int = ref_int + np.random.normal(0, 0.1, 100)  # Add some noise
        spectrum = ExperimentalSpectrum(frequencies=spec_freq, intensities=spec_int)

        aligner = SpectraAligner(use_gpu=False)
        aligned = aligner._align_single_spectrum(spectrum, reference)

        assert "aligned_shift" in aligned.processing_history[0]
        # Frequencies should be shifted back toward reference
        assert not np.array_equal(aligned.frequencies, spectrum.frequencies)

    def test_align_multiple_spectra(self):
        """Test alignment of multiple spectra."""
        # Create reference spectrum
        ref_freq = np.linspace(400, 4000, 100)
        ref_int = np.random.rand(100)
        reference = ExperimentalSpectrum(frequencies=ref_freq, intensities=ref_int)

        # Create additional spectra with different shifts
        spectra = [reference]  # Reference is first

        for i in range(3):
            shift = (i + 1) * 5.0  # Different shifts
            spec_freq = ref_freq + shift
            spec_int = ref_int + np.random.normal(0, 0.1, 100)
            spectrum = ExperimentalSpectrum(frequencies=spec_freq, intensities=spec_int)
            spectra.append(spectrum)

        aligner = SpectraAligner(use_gpu=False)
        aligned_spectra = aligner.align_spectra(spectra, reference_idx=0)

        assert len(aligned_spectra) == 4
        # Reference should remain unchanged
        assert np.array_equal(aligned_spectra[0].frequencies, reference.frequencies)
        # Others should be aligned
        for i in range(1, 4):
            assert "aligned_shift" in aligned_spectra[i].processing_history[-1]

    def test_align_empty_list(self):
        """Test alignment with empty spectrum list."""
        aligner = SpectraAligner(use_gpu=False)
        result = aligner.align_spectra([])
        assert result == []

    def test_align_single_spectrum_list(self):
        """Test alignment with single spectrum."""
        spectrum = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )

        aligner = SpectraAligner(use_gpu=False)
        result = aligner.align_spectra([spectrum])

        assert len(result) == 1
        assert result[0] == spectrum


class TestExperimentalSpectraProcessor:
    """Test main experimental spectra processor."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = ExperimentalSpectraProcessor(use_gpu=False)
        assert processor.use_gpu is False
        assert processor.preprocessor is not None
        assert processor.aligner is not None

    def test_load_csv_spectrum(self):
        """Test CSV spectrum loading."""
        csv_data = "frequency,intensity\n1000,0.5\n2000,0.8\n3000,0.3\n"

        with patch("pandas.read_csv") as mock_read_csv:
            # Mock pandas DataFrame
            mock_df = Mock()
            mock_df.iloc = Mock()
            mock_df.iloc.__getitem__ = Mock()
            mock_df.iloc.__getitem__.side_effect = [
                Mock(values=np.array([1000, 2000, 3000])),  # frequencies
                Mock(values=np.array([0.5, 0.8, 0.3])),  # intensities
            ]
            mock_read_csv.return_value = mock_df

            processor = ExperimentalSpectraProcessor(use_gpu=False)
            spectrum = processor._load_csv("test.csv")

            assert len(spectrum.frequencies) == 3
            assert len(spectrum.intensities) == 3
            assert "loaded_csv" in spectrum.processing_history

    def test_load_spectrum_unsupported_format(self):
        """Test loading with unsupported format."""
        processor = ExperimentalSpectraProcessor(use_gpu=False)

        with pytest.raises(FormatError, match="Unsupported format"):
            processor.load_spectrum("test.xyz", SpectrumFormat.BRUKER)

    def test_process_single_spectrum(self):
        """Test processing single spectrum."""
        spectrum = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )

        processor = ExperimentalSpectraProcessor(use_gpu=False)
        result = processor.process_single_spectrum(spectrum)

        assert isinstance(result, ProcessingResult)
        assert result.processed_spectrum is not None
        assert result.original_spectrum == spectrum

    def test_process_batch_without_alignment(self):
        """Test batch processing without alignment."""
        spectra = []
        for i in range(3):
            spectrum = ExperimentalSpectrum(
                frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
            )
            spectra.append(spectrum)

        processor = ExperimentalSpectraProcessor(use_gpu=False)
        results = processor.process_batch(spectra, align_spectra=False)

        assert len(results) == 3
        assert all(isinstance(r, ProcessingResult) for r in results)

    def test_process_batch_with_alignment(self):
        """Test batch processing with alignment."""
        # Create spectra with different frequency shifts
        base_freq = np.linspace(400, 4000, 100)
        spectra = []

        for i in range(3):
            freq_shift = i * 5.0  # Different shifts
            frequencies = base_freq + freq_shift
            intensities = np.random.rand(100)

            spectrum = ExperimentalSpectrum(
                frequencies=frequencies, intensities=intensities
            )
            spectra.append(spectrum)

        processor = ExperimentalSpectraProcessor(use_gpu=False)
        results = processor.process_batch(spectra, align_spectra=True)

        assert len(results) == 3
        # Check that alignment was applied (except for reference)
        for i in range(1, 3):
            history = results[i].processed_spectrum.processing_history
            assert any("aligned_shift" in step for step in history)


class TestUtilityFunctions:
    """Test utility functions."""

    @patch(
        "chemi_accel.spectroscopy.experimental_processing.ExperimentalSpectraProcessor"
    )
    def test_process_spectrum_batch(self, mock_processor_class):
        """Test batch spectrum processing utility."""
        # Mock processor
        mock_processor = Mock()
        mock_spectrum = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )
        mock_processor.load_spectrum.return_value = mock_spectrum

        mock_result = ProcessingResult(
            processed_spectrum=mock_spectrum,
            original_spectrum=mock_spectrum,
            computation_time=0.5,
        )
        mock_processor.process_batch.return_value = [mock_result, mock_result]

        mock_processor_class.return_value = mock_processor

        filepaths = ["file1.jdx", "file2.jdx"]
        results = process_spectrum_batch(
            filepaths, SpectrumFormat.JCAMP_DX, use_gpu=False
        )

        assert len(results) == 2
        mock_processor.load_spectrum.assert_called()
        mock_processor.process_batch.assert_called_once()

    @patch(
        "chemi_accel.spectroscopy.experimental_processing.ExperimentalSpectraProcessor"
    )
    def test_process_spectrum_batch_with_errors(self, mock_processor_class):
        """Test batch processing with file loading errors."""
        mock_processor = Mock()

        # First file loads successfully, second fails
        def load_side_effect(filepath, format_type):
            if "file1" in filepath:
                return ExperimentalSpectrum(
                    frequencies=np.linspace(400, 4000, 100),
                    intensities=np.random.rand(100),
                )
            else:
                raise FormatError("Failed to load file")

        mock_processor.load_spectrum.side_effect = load_side_effect
        mock_processor.process_batch.return_value = [Mock()]

        mock_processor_class.return_value = mock_processor

        filepaths = ["file1.jdx", "file2.jdx"]

        with patch("builtins.print"):  # Mock print for warning messages
            results = process_spectrum_batch(filepaths, SpectrumFormat.JCAMP_DX)

        # Should only process the successfully loaded spectrum
        assert len(results) == 1


class TestExceptionHandling:
    """Test exception handling in experimental processing."""

    def test_processing_error(self):
        """Test ProcessingError."""
        with pytest.raises(ProcessingError, match="Test processing error"):
            raise ProcessingError("Test processing error")

    def test_format_error(self):
        """Test FormatError."""
        with pytest.raises(FormatError, match="Test format error"):
            raise FormatError("Test format error")

    def test_data_quality_error(self):
        """Test DataQualityError."""
        with pytest.raises(DataQualityError, match="Test data quality error"):
            raise DataQualityError("Test data quality error")


class TestIntegration:
    """Integration tests for experimental processing."""

    def test_complete_workflow(self):
        """Test complete processing workflow."""
        # Create realistic test spectrum
        frequencies = np.linspace(400, 4000, 1000)

        # Create baseline + peaks + noise
        baseline = 0.1 + 0.02 * np.sin(frequencies / 1000)

        # Add characteristic IR peaks
        peaks_data = [
            (1000, 0.3),  # C-O stretch
            (1650, 0.8),  # C=O stretch
            (2900, 0.4),  # C-H stretch
            (3300, 0.6),  # O-H stretch
        ]

        signal = np.zeros_like(frequencies)
        for peak_freq, peak_int in peaks_data:
            idx = np.argmin(np.abs(frequencies - peak_freq))
            # Add Gaussian peak
            peak = peak_int * np.exp(-(((frequencies - peak_freq) / 20) ** 2))
            signal += peak

        # Add noise
        noise = np.random.normal(0, 0.02, len(frequencies))
        intensities = baseline + signal + noise

        # Create spectrum
        spectrum = ExperimentalSpectrum(
            frequencies=frequencies,
            intensities=intensities,
            metadata={
                "sample": "test_compound",
                "temperature": 298.15,
                "resolution": 4.0,
            },
        )

        # Process spectrum
        params = ProcessingParameters(
            frequency_range=(500.0, 3500.0),
            baseline_method=BaselineCorrectionMethod.ASYMMETRIC_LEAST_SQUARES,
            normalization_method=NormalizationMethod.MIN_MAX,
            peak_detection_threshold=0.1,
        )

        processor = ExperimentalSpectraProcessor(use_gpu=False)
        result = processor.process_single_spectrum(spectrum, params)

        # Verify processing results
        assert result.processed_spectrum is not None
        assert result.original_spectrum == spectrum
        assert result.computation_time > 0

        processed = result.processed_spectrum

        # Check frequency range filtering
        assert np.min(processed.frequencies) >= 500.0
        assert np.max(processed.frequencies) <= 3500.0

        # Check normalization (min-max should give range [0,1])
        assert np.min(processed.intensities) >= 0.0
        assert np.max(processed.intensities) <= 1.0

        # Check peak detection
        assert len(processed.peaks) > 0

        # Check processing history
        history = processed.processing_history
        assert any("frequency_filtered" in step for step in history)
        assert any("smoothed_savgol" in step for step in history)
        assert any("baseline_corrected" in step for step in history)
        assert any("normalized" in step for step in history)

        # Check quality metrics
        snr = processed.calculate_signal_to_noise()
        assert snr > 0
        assert "signal_to_noise" in processed.quality_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

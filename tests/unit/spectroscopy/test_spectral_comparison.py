"""
Unit tests for spectral comparison and discrepancy highlighting.

This module tests the quantitative metrics for comparing predicted and experimental
spectra, along with visualization tools for highlighting discrepancies.
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.spectroscopy.experimental_processing import ExperimentalSpectrum
from chemi_accel.spectroscopy.predictive_models import (
    SpectralPrediction,
    SpectroscopyType,
)
from chemi_accel.spectroscopy.spectral_comparison import (
    AlignmentError,
    ComparisonMethod,
    DiscrepancyRegion,
    DiscrepancyType,
    SpectralAligner,
    SpectralComparator,
    SpectralComparison,
    SpectralComparisonError,
    SpectralVisualizer,
    VisualizationConfig,
    VisualizationError,
    VisualizationType,
    compare_spectra_batch,
    generate_comparison_report,
)


class TestDiscrepancyRegion:
    """Test DiscrepancyRegion data structure."""

    def test_basic_discrepancy_region(self):
        """Test basic discrepancy region creation."""
        region = DiscrepancyRegion(
            frequency_start=1000.0,
            frequency_end=1200.0,
            discrepancy_type=DiscrepancyType.INTENSITY_DIFFERENCE,
            severity=0.5,
            statistical_significance=0.8,
        )

        assert region.frequency_start == 1000.0
        assert region.frequency_end == 1200.0
        assert region.discrepancy_type == DiscrepancyType.INTENSITY_DIFFERENCE
        assert region.severity == 0.5
        assert region.statistical_significance == 0.8
        assert region.predicted_intensity is None
        assert region.experimental_intensity is None
        assert region.description == ""
        assert region.suggested_explanation == ""

    def test_complete_discrepancy_region(self):
        """Test discrepancy region with all fields."""
        region = DiscrepancyRegion(
            frequency_start=1500.0,
            frequency_end=1600.0,
            discrepancy_type=DiscrepancyType.MISSING_PEAK,
            severity=0.9,
            statistical_significance=0.95,
            predicted_intensity=0.1,
            experimental_intensity=0.8,
            description="Missing C=O stretch",
            suggested_explanation="Model may not predict carbonyl peak correctly",
        )

        assert region.predicted_intensity == 0.1
        assert region.experimental_intensity == 0.8
        assert region.description == "Missing C=O stretch"
        assert "Model may not predict" in region.suggested_explanation


class TestSpectralComparison:
    """Test SpectralComparison data structure."""

    def test_basic_spectral_comparison(self):
        """Test basic spectral comparison creation."""
        predicted = SpectralPrediction(
            frequencies=np.linspace(400, 4000, 100),
            intensities=np.random.rand(100),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )

        comparison = SpectralComparison(
            predicted_spectrum=predicted,
            experimental_spectrum=experimental,
            overall_similarity=0.75,
        )

        assert comparison.predicted_spectrum == predicted
        assert comparison.experimental_spectrum == experimental
        assert comparison.overall_similarity == 0.75
        assert len(comparison.comparison_metrics) == 0
        assert len(comparison.discrepancy_regions) == 0
        assert comparison.alignment_offset == 0.0
        assert comparison.scaling_factor == 1.0


class TestVisualizationConfig:
    """Test VisualizationConfig data structure."""

    def test_default_config(self):
        """Test default visualization configuration."""
        config = VisualizationConfig()

        assert config.figure_size == (12, 8)
        assert config.dpi == 150
        assert config.style == "seaborn-v0_8"
        assert config.line_width == 2.0
        assert config.font_size == 12
        assert config.highlight_threshold == 0.1
        assert config.show_statistics is True
        assert config.show_peak_labels is True
        assert config.interactive_mode is False

    def test_custom_config(self):
        """Test custom visualization configuration."""
        config = VisualizationConfig(
            figure_size=(15, 10),
            dpi=300,
            style="ggplot",
            line_width=3.0,
            font_size=14,
            highlight_threshold=0.2,
            show_statistics=False,
        )

        assert config.figure_size == (15, 10)
        assert config.dpi == 300
        assert config.style == "ggplot"
        assert config.line_width == 3.0
        assert config.font_size == 14
        assert config.highlight_threshold == 0.2
        assert config.show_statistics is False


class TestSpectralAligner:
    """Test spectral alignment functionality."""

    def test_aligner_initialization(self):
        """Test aligner initialization."""
        aligner = SpectralAligner(use_gpu=False)
        assert aligner.use_gpu is False

    def test_align_cross_correlation(self):
        """Test cross-correlation alignment."""
        # Create test spectra with known offset
        base_freq = np.linspace(400, 4000, 100)
        base_intensities = np.random.rand(100)

        predicted = SpectralPrediction(
            frequencies=base_freq,
            intensities=base_intensities,
            spectroscopy_type=SpectroscopyType.IR,
        )

        # Create experimental spectrum with 50 cm-1 shift
        exp_freq = base_freq + 50.0
        experimental = ExperimentalSpectrum(
            frequencies=exp_freq,
            intensities=base_intensities + np.random.normal(0, 0.1, 100),
        )

        aligner = SpectralAligner(use_gpu=False)
        aligned_pred, aligned_exp, offset = aligner.align_spectra(
            predicted, experimental, method="cross_correlation"
        )

        assert aligned_pred is not None
        assert aligned_exp == experimental  # Experimental shouldn't change
        assert isinstance(offset, float)

    def test_align_peak_matching(self):
        """Test peak matching alignment."""
        # Create spectra with identifiable peaks
        freq = np.linspace(400, 4000, 1000)

        # Create predicted spectrum with peaks at specific positions
        pred_intensities = np.zeros_like(freq)
        peak_positions = [1000, 1500, 2000, 3000]
        for pos in peak_positions:
            idx = np.argmin(np.abs(freq - pos))
            pred_intensities[idx] = 0.8

        predicted = SpectralPrediction(
            frequencies=freq,
            intensities=pred_intensities,
            spectroscopy_type=SpectroscopyType.IR,
        )

        # Create experimental spectrum with shifted peaks
        exp_intensities = np.zeros_like(freq)
        for pos in peak_positions:
            shifted_pos = pos + 20  # 20 cm-1 shift
            idx = np.argmin(np.abs(freq - shifted_pos))
            exp_intensities[idx] = 0.7

        experimental = ExperimentalSpectrum(
            frequencies=freq, intensities=exp_intensities
        )

        aligner = SpectralAligner(use_gpu=False)
        aligned_pred, aligned_exp, offset = aligner.align_spectra(
            predicted, experimental, method="peak_matching"
        )

        assert aligned_pred is not None
        assert offset != 0.0  # Should detect some offset

    def test_align_empty_spectra(self):
        """Test alignment with empty or minimal spectra."""
        predicted = SpectralPrediction(
            frequencies=np.array([1000]),
            intensities=np.array([0.5]),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.array([1000]), intensities=np.array([0.5])
        )

        aligner = SpectralAligner(use_gpu=False)
        aligned_pred, aligned_exp, offset = aligner.align_spectra(
            predicted, experimental
        )

        assert aligned_pred is not None
        assert aligned_exp == experimental
        assert offset == 0.0

    def test_find_major_peaks(self):
        """Test major peak detection."""
        frequencies = np.linspace(400, 4000, 1000)

        # Create spectrum with specific peaks
        intensities = np.random.normal(0.1, 0.02, 1000)  # Noise
        peak_positions = [800, 1200, 1600, 2400, 3200]
        peak_heights = [0.9, 0.7, 0.8, 0.6, 0.5]

        for pos, height in zip(peak_positions, peak_heights):
            idx = np.argmin(np.abs(frequencies - pos))
            intensities[idx] = height

        aligner = SpectralAligner(use_gpu=False)
        major_peaks = aligner._find_major_peaks(frequencies, intensities)

        assert len(major_peaks) > 0
        assert len(major_peaks) <= 5  # Should return top 5

        # Check that peaks are sorted by intensity
        peak_intensities = [
            intensities[np.argmin(np.abs(frequencies - freq))] for freq in major_peaks
        ]
        assert peak_intensities == sorted(peak_intensities, reverse=True)


class TestSpectralComparator:
    """Test spectral comparison functionality."""

    def test_comparator_initialization(self):
        """Test comparator initialization."""
        comparator = SpectralComparator(use_gpu=False)
        assert comparator.use_gpu is False
        assert comparator.aligner is not None

    def test_compare_identical_spectra(self):
        """Test comparison of identical spectra."""
        frequencies = np.linspace(400, 4000, 100)
        intensities = np.random.rand(100)

        predicted = SpectralPrediction(
            frequencies=frequencies,
            intensities=intensities,
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities
        )

        comparator = SpectralComparator(use_gpu=False)
        comparison = comparator.compare_spectra(
            predicted, experimental, align_spectra=False
        )

        assert isinstance(comparison, SpectralComparison)
        assert comparison.overall_similarity > 0.9  # Should be very similar
        assert comparison.computation_time > 0
        assert len(comparison.comparison_metrics) > 0

    def test_compare_different_spectra(self):
        """Test comparison of different spectra."""
        frequencies = np.linspace(400, 4000, 100)

        predicted = SpectralPrediction(
            frequencies=frequencies,
            intensities=np.random.rand(100),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=frequencies, intensities=np.random.rand(100)
        )

        comparator = SpectralComparator(use_gpu=False)
        comparison = comparator.compare_spectra(predicted, experimental)

        assert isinstance(comparison, SpectralComparison)
        assert 0.0 <= comparison.overall_similarity <= 1.0
        assert len(comparison.comparison_metrics) > 0

    def test_specific_comparison_methods(self):
        """Test specific comparison methods."""
        frequencies = np.linspace(400, 4000, 100)
        intensities1 = np.random.rand(100)
        intensities2 = intensities1 + np.random.normal(0, 0.1, 100)

        predicted = SpectralPrediction(
            frequencies=frequencies,
            intensities=intensities1,
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=frequencies, intensities=intensities2
        )

        comparator = SpectralComparator(use_gpu=False)

        methods = [
            ComparisonMethod.MEAN_SQUARED_ERROR,
            ComparisonMethod.CORRELATION_COEFFICIENT,
            ComparisonMethod.COSINE_SIMILARITY,
        ]

        comparison = comparator.compare_spectra(
            predicted, experimental, methods=methods, align_spectra=False
        )

        assert "mse" in comparison.comparison_metrics
        assert "correlation" in comparison.comparison_metrics
        assert "cosine" in comparison.comparison_metrics

        # MSE should be small for similar spectra
        assert comparison.comparison_metrics["mse"] < 1.0

        # Correlation should be positive for similar spectra
        assert comparison.comparison_metrics["correlation"] > 0.0

    def test_discrepancy_detection(self):
        """Test discrepancy region detection."""
        frequencies = np.linspace(400, 4000, 1000)

        # Create predicted spectrum
        pred_intensities = np.random.normal(0.1, 0.02, 1000)

        # Create experimental spectrum with specific discrepancies
        exp_intensities = pred_intensities.copy()

        # Add missing peak (high in predicted, low in experimental)
        missing_peak_idx = 300
        pred_intensities[missing_peak_idx : missing_peak_idx + 10] = 0.8
        exp_intensities[missing_peak_idx : missing_peak_idx + 10] = 0.1

        # Add extra peak (low in predicted, high in experimental)
        extra_peak_idx = 700
        pred_intensities[extra_peak_idx : extra_peak_idx + 10] = 0.1
        exp_intensities[extra_peak_idx : extra_peak_idx + 10] = 0.7

        predicted = SpectralPrediction(
            frequencies=frequencies,
            intensities=pred_intensities,
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=frequencies, intensities=exp_intensities
        )

        comparator = SpectralComparator(use_gpu=False)
        comparison = comparator.compare_spectra(
            predicted, experimental, detect_discrepancies=True, align_spectra=False
        )

        assert len(comparison.discrepancy_regions) > 0

        # Check that discrepancies were classified
        discrepancy_types = [
            region.discrepancy_type for region in comparison.discrepancy_regions
        ]
        assert any(
            dtype
            in [
                DiscrepancyType.MISSING_PEAK,
                DiscrepancyType.EXTRA_PEAK,
                DiscrepancyType.INTENSITY_DIFFERENCE,
            ]
            for dtype in discrepancy_types
        )

    def test_peak_matching_analysis(self):
        """Test peak matching analysis."""
        frequencies = np.linspace(400, 4000, 1000)

        # Create spectra with specific peaks
        pred_intensities = np.zeros_like(frequencies)
        exp_intensities = np.zeros_like(frequencies)

        # Matching peaks
        matching_positions = [800, 1200, 2000]
        for pos in matching_positions:
            idx = np.argmin(np.abs(frequencies - pos))
            pred_intensities[idx] = 0.8
            exp_intensities[idx] = 0.7

        # Extra peak in predicted
        extra_pred_idx = np.argmin(np.abs(frequencies - 1600))
        pred_intensities[extra_pred_idx] = 0.6

        # Extra peak in experimental
        extra_exp_idx = np.argmin(np.abs(frequencies - 2400))
        exp_intensities[extra_exp_idx] = 0.5

        predicted = SpectralPrediction(
            frequencies=frequencies,
            intensities=pred_intensities,
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=frequencies, intensities=exp_intensities
        )

        comparator = SpectralComparator(use_gpu=False)
        comparison = comparator.compare_spectra(
            predicted, experimental, align_spectra=False
        )

        peak_results = comparison.peak_matching_results

        assert "predicted_peaks" in peak_results
        assert "experimental_peaks" in peak_results
        assert "matched_peaks" in peak_results
        assert "missing_peaks" in peak_results
        assert "extra_peaks" in peak_results

        assert peak_results["matched_peaks"] >= 3  # Should find matching peaks
        assert len(peak_results["missing_peaks"]) >= 0
        assert len(peak_results["extra_peaks"]) >= 0

    def test_statistical_tests(self):
        """Test statistical tests (mocked)."""
        frequencies = np.linspace(400, 4000, 100)

        predicted = SpectralPrediction(
            frequencies=frequencies,
            intensities=np.random.normal(0.5, 0.1, 100),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=frequencies, intensities=np.random.normal(0.5, 0.1, 100)
        )

        comparator = SpectralComparator(use_gpu=False)

        with patch(
            "chemi_accel.spectroscopy.spectral_comparison.SCIPY_AVAILABLE", True
        ):
            with patch("scipy.stats.ks_2samp", return_value=(0.1, 0.8)):
                with patch("scipy.stats.mannwhitneyu", return_value=(100, 0.6)):
                    with patch("scipy.stats.pearsonr", return_value=(0.7, 0.01)):
                        comparison = comparator.compare_spectra(
                            predicted, experimental, align_spectra=False
                        )

        assert len(comparison.statistical_tests) > 0

    def test_quality_assessment(self):
        """Test comparison quality assessment."""
        # Test with overlapping frequency ranges
        pred_freq = np.linspace(400, 4000, 100)
        exp_freq = np.linspace(500, 3500, 80)  # Smaller range

        predicted = SpectralPrediction(
            frequencies=pred_freq,
            intensities=np.random.rand(100),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=exp_freq, intensities=np.random.rand(80)
        )

        comparator = SpectralComparator(use_gpu=False)
        comparison = comparator.compare_spectra(
            predicted, experimental, align_spectra=False
        )

        quality = comparison.quality_assessment

        assert "frequency_overlap" in quality
        assert "resolution_ratio" in quality
        assert "predicted_snr" in quality
        assert "experimental_snr" in quality
        assert "predicted_data_points" in quality
        assert "experimental_data_points" in quality

        assert 0.0 <= quality["frequency_overlap"] <= 1.0
        assert 0.0 <= quality["resolution_ratio"] <= 1.0
        assert quality["predicted_data_points"] == 100
        assert quality["experimental_data_points"] == 80


@patch("chemi_accel.spectroscopy.spectral_comparison.MATPLOTLIB_AVAILABLE", True)
class TestSpectralVisualizer:
    """Test spectral visualization functionality."""

    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        config = VisualizationConfig(figure_size=(10, 6))
        visualizer = SpectralVisualizer(config)
        assert visualizer.config.figure_size == (10, 6)

    def test_visualizer_default_config(self):
        """Test visualizer with default configuration."""
        visualizer = SpectralVisualizer()
        assert visualizer.config is not None
        assert isinstance(visualizer.config, VisualizationConfig)

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_create_overlay_plot(
        self, mock_close, mock_show, mock_savefig, mock_tight_layout, mock_subplots
    ):
        """Test overlay plot creation."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Create test data
        predicted = SpectralPrediction(
            frequencies=np.linspace(400, 4000, 100),
            intensities=np.random.rand(100),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )

        comparison = SpectralComparison(
            predicted_spectrum=predicted,
            experimental_spectrum=experimental,
            overall_similarity=0.8,
        )

        visualizer = SpectralVisualizer()

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot.png")

            fig = visualizer.create_comparison_plot(
                comparison,
                VisualizationType.OVERLAY_PLOT,
                save_path=save_path,
                show_plot=False,
            )

            assert fig is not None
            mock_subplots.assert_called_once()
            mock_ax.plot.assert_called()
            mock_savefig.assert_called_with(save_path, dpi=150, bbox_inches="tight")
            mock_show.assert_not_called()  # show_plot=False

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_create_difference_plot(self, mock_tight_layout, mock_subplots):
        """Test difference plot creation."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        # Create test data
        predicted = SpectralPrediction(
            frequencies=np.linspace(400, 4000, 100),
            intensities=np.random.rand(100),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )

        comparison = SpectralComparison(
            predicted_spectrum=predicted,
            experimental_spectrum=experimental,
            overall_similarity=0.8,
        )

        visualizer = SpectralVisualizer()
        fig = visualizer.create_comparison_plot(
            comparison, VisualizationType.DIFFERENCE_PLOT, show_plot=False
        )

        assert fig is not None
        mock_subplots.assert_called_once()
        assert mock_ax1.plot.called
        assert mock_ax2.plot.called

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_create_mirror_plot(self, mock_tight_layout, mock_subplots):
        """Test mirror plot creation."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        # Create test data
        predicted = SpectralPrediction(
            frequencies=np.linspace(400, 4000, 100),
            intensities=np.random.rand(100),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )

        comparison = SpectralComparison(
            predicted_spectrum=predicted,
            experimental_spectrum=experimental,
            overall_similarity=0.8,
        )

        visualizer = SpectralVisualizer()
        fig = visualizer.create_comparison_plot(
            comparison, VisualizationType.MIRROR_PLOT, show_plot=False
        )

        assert fig is not None
        mock_subplots.assert_called_once()
        assert mock_ax1.plot.called
        assert mock_ax2.plot.called
        assert mock_ax2.invert_yaxis.called

    @patch("matplotlib.pyplot.subplots")
    def test_create_discrepancy_map(self, mock_subplots):
        """Test discrepancy map creation."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Create test data with discrepancies
        predicted = SpectralPrediction(
            frequencies=np.linspace(400, 4000, 100),
            intensities=np.random.rand(100),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )

        # Add discrepancy regions
        discrepancy_regions = [
            DiscrepancyRegion(
                frequency_start=1000.0,
                frequency_end=1200.0,
                discrepancy_type=DiscrepancyType.INTENSITY_DIFFERENCE,
                severity=0.7,
                statistical_significance=0.9,
            ),
            DiscrepancyRegion(
                frequency_start=2500.0,
                frequency_end=2600.0,
                discrepancy_type=DiscrepancyType.MISSING_PEAK,
                severity=0.5,
                statistical_significance=0.8,
            ),
        ]

        comparison = SpectralComparison(
            predicted_spectrum=predicted,
            experimental_spectrum=experimental,
            overall_similarity=0.6,
            discrepancy_regions=discrepancy_regions,
        )

        visualizer = SpectralVisualizer()
        fig = visualizer.create_comparison_plot(
            comparison, VisualizationType.DISCREPANCY_MAP, show_plot=False
        )

        assert fig is not None
        mock_subplots.assert_called_once()
        assert mock_ax.plot.called
        assert mock_ax.axvspan.called  # Should highlight discrepancy regions
        assert mock_ax.annotate.called  # Should annotate discrepancies

    def test_unsupported_plot_type(self):
        """Test error handling for unsupported plot types."""
        predicted = SpectralPrediction(
            frequencies=np.linspace(400, 4000, 100),
            intensities=np.random.rand(100),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )

        comparison = SpectralComparison(
            predicted_spectrum=predicted,
            experimental_spectrum=experimental,
            overall_similarity=0.8,
        )

        visualizer = SpectralVisualizer()

        with pytest.raises(ValueError, match="Unsupported plot type"):
            # Use a string that's not a valid VisualizationType
            visualizer.create_comparison_plot(comparison, "invalid_type")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_compare_spectra_batch(self):
        """Test batch spectral comparison."""
        # Create test data
        predicted_spectra = []
        experimental_spectra = []

        for i in range(3):
            predicted = SpectralPrediction(
                frequencies=np.linspace(400, 4000, 100),
                intensities=np.random.rand(100),
                spectroscopy_type=SpectroscopyType.IR,
            )
            predicted_spectra.append(predicted)

            experimental = ExperimentalSpectrum(
                frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
            )
            experimental_spectra.append(experimental)

        comparisons = compare_spectra_batch(
            predicted_spectra, experimental_spectra, use_gpu=False
        )

        assert len(comparisons) == 3
        assert all(isinstance(comp, SpectralComparison) for comp in comparisons)

    def test_compare_spectra_batch_length_mismatch(self):
        """Test batch comparison with mismatched lengths."""
        predicted_spectra = [Mock()]
        experimental_spectra = [Mock(), Mock()]

        with pytest.raises(
            ValueError, match="Number of predicted and experimental spectra must match"
        ):
            compare_spectra_batch(predicted_spectra, experimental_spectra)

    @patch("chemi_accel.spectroscopy.spectral_comparison.MATPLOTLIB_AVAILABLE", True)
    @patch("matplotlib.pyplot.close")
    def test_generate_comparison_report(self, mock_close):
        """Test comparison report generation."""
        # Create test comparison
        predicted = SpectralPrediction(
            frequencies=np.linspace(400, 4000, 100),
            intensities=np.random.rand(100),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )

        discrepancy_regions = [
            DiscrepancyRegion(
                frequency_start=1000.0,
                frequency_end=1200.0,
                discrepancy_type=DiscrepancyType.INTENSITY_DIFFERENCE,
                severity=0.5,
                statistical_significance=0.8,
                description="Test discrepancy",
                suggested_explanation="Test explanation",
            )
        ]

        comparison = SpectralComparison(
            predicted_spectrum=predicted,
            experimental_spectrum=experimental,
            overall_similarity=0.75,
            comparison_metrics={"correlation": 0.8, "mse": 0.1},
            discrepancy_regions=discrepancy_regions,
            peak_matching_results={"matched_peaks": 5},
            statistical_tests={"ks_test": {"p_value": 0.05}},
            quality_assessment={"frequency_overlap": 0.9},
            computation_time=1.2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "chemi_accel.spectroscopy.spectral_comparison.SpectralVisualizer"
            ) as mock_visualizer_class:
                mock_visualizer = Mock()
                mock_visualizer_class.return_value = mock_visualizer
                mock_visualizer.create_comparison_plot.return_value = Mock()

                report = generate_comparison_report(
                    comparison, include_plots=True, output_dir=temp_dir
                )

        # Check report structure
        assert "summary" in report
        assert "metrics" in report
        assert "discrepancies" in report
        assert "peak_analysis" in report
        assert "statistical_tests" in report
        assert "quality_assessment" in report

        # Check summary content
        summary = report["summary"]
        assert summary["overall_similarity"] == 0.75
        assert summary["computation_time"] == 1.2
        assert summary["discrepancy_count"] == 1

        # Check discrepancy formatting
        discrepancies = report["discrepancies"]
        assert len(discrepancies) == 1
        assert discrepancies[0]["frequency_range"] == [1000.0, 1200.0]
        assert discrepancies[0]["type"] == "intensity"
        assert discrepancies[0]["severity"] == 0.5

    def test_generate_comparison_report_no_plots(self):
        """Test report generation without plots."""
        predicted = SpectralPrediction(
            frequencies=np.linspace(400, 4000, 100),
            intensities=np.random.rand(100),
            spectroscopy_type=SpectroscopyType.IR,
        )

        experimental = ExperimentalSpectrum(
            frequencies=np.linspace(400, 4000, 100), intensities=np.random.rand(100)
        )

        comparison = SpectralComparison(
            predicted_spectrum=predicted,
            experimental_spectrum=experimental,
            overall_similarity=0.75,
        )

        report = generate_comparison_report(comparison, include_plots=False)

        assert "plot_files" not in report
        assert "summary" in report
        assert "metrics" in report


class TestExceptionHandling:
    """Test exception handling."""

    def test_spectral_comparison_error(self):
        """Test SpectralComparisonError."""
        with pytest.raises(SpectralComparisonError, match="Test comparison error"):
            raise SpectralComparisonError("Test comparison error")

    def test_alignment_error(self):
        """Test AlignmentError."""
        with pytest.raises(AlignmentError, match="Test alignment error"):
            raise AlignmentError("Test alignment error")

    def test_visualization_error(self):
        """Test VisualizationError."""
        with pytest.raises(VisualizationError, match="Test visualization error"):
            raise VisualizationError("Test visualization error")


class TestIntegration:
    """Integration tests for spectral comparison."""

    def test_complete_comparison_workflow(self):
        """Test complete spectral comparison workflow."""
        # Create realistic test spectra
        frequencies = np.linspace(400, 4000, 1000)

        # Create predicted spectrum with specific peaks
        pred_intensities = np.random.normal(0.05, 0.01, 1000)  # Baseline noise

        # Add characteristic IR peaks
        ir_peaks = [
            (1000, 0.6),  # C-O stretch
            (1650, 0.9),  # C=O stretch
            (2900, 0.4),  # C-H stretch
            (3300, 0.7),  # O-H stretch
        ]

        for peak_freq, peak_int in ir_peaks:
            idx = np.argmin(np.abs(frequencies - peak_freq))
            # Add Gaussian peak
            sigma = 20  # Peak width
            peak_profile = peak_int * np.exp(
                -(((frequencies - peak_freq) / sigma) ** 2)
            )
            pred_intensities += peak_profile

        predicted = SpectralPrediction(
            frequencies=frequencies,
            intensities=pred_intensities,
            spectroscopy_type=SpectroscopyType.IR,
        )

        # Create experimental spectrum with modifications
        exp_intensities = pred_intensities.copy()

        # Add noise
        exp_intensities += np.random.normal(0, 0.02, 1000)

        # Modify some peaks (simulate experimental differences)
        carbonyl_idx = np.argmin(np.abs(frequencies - 1650))
        exp_intensities[
            carbonyl_idx - 10 : carbonyl_idx + 10
        ] *= 0.7  # Reduce carbonyl intensity

        # Add new peak (impurity or different compound)
        new_peak_idx = np.argmin(np.abs(frequencies - 1200))
        exp_intensities[new_peak_idx - 5 : new_peak_idx + 5] += 0.3

        experimental = ExperimentalSpectrum(
            frequencies=frequencies, intensities=exp_intensities
        )

        # Perform comprehensive comparison
        comparator = SpectralComparator(use_gpu=False)
        comparison = comparator.compare_spectra(
            predicted,
            experimental,
            methods=[
                ComparisonMethod.CORRELATION_COEFFICIENT,
                ComparisonMethod.MEAN_SQUARED_ERROR,
                ComparisonMethod.COSINE_SIMILARITY,
                ComparisonMethod.PEAK_MATCHING,
            ],
            align_spectra=True,
            detect_discrepancies=True,
        )

        # Verify comprehensive results
        assert isinstance(comparison, SpectralComparison)
        assert 0.0 <= comparison.overall_similarity <= 1.0
        assert comparison.computation_time > 0

        # Should have multiple metrics
        assert len(comparison.comparison_metrics) >= 3
        assert "correlation" in comparison.comparison_metrics
        assert "mse" in comparison.comparison_metrics
        assert "cosine" in comparison.comparison_metrics

        # Should detect discrepancies
        assert len(comparison.discrepancy_regions) > 0

        # Should have peak matching results
        peak_results = comparison.peak_matching_results
        assert "predicted_peaks" in peak_results
        assert "experimental_peaks" in peak_results
        assert "matched_peaks" in peak_results

        # Should have quality assessment
        quality = comparison.quality_assessment
        assert "frequency_overlap" in quality
        assert "predicted_snr" in quality
        assert "experimental_snr" in quality

        # Test visualization (if matplotlib available)
        if hasattr(comparison, "predicted_spectrum"):
            try:
                visualizer = SpectralVisualizer()

                with tempfile.TemporaryDirectory() as temp_dir:
                    # Test multiple plot types
                    plot_types = [
                        VisualizationType.OVERLAY_PLOT,
                        VisualizationType.DIFFERENCE_PLOT,
                        VisualizationType.DISCREPANCY_MAP,
                    ]

                    for plot_type in plot_types:
                        save_path = os.path.join(temp_dir, f"{plot_type.value}.png")

                        with patch("matplotlib.pyplot.show"):
                            with patch("matplotlib.pyplot.savefig"):
                                fig = visualizer.create_comparison_plot(
                                    comparison,
                                    plot_type,
                                    save_path=save_path,
                                    show_plot=False,
                                )

                                if fig is not None:  # Only if matplotlib available
                                    assert fig is not None

            except ImportError:
                # Matplotlib not available - skip visualization tests
                pass

        # Generate comprehensive report
        with tempfile.TemporaryDirectory() as temp_dir:
            report = generate_comparison_report(
                comparison,
                include_plots=False,  # Skip plots to avoid matplotlib dependency
                output_dir=temp_dir,
            )

            assert isinstance(report, dict)
            assert "summary" in report
            assert "discrepancies" in report
            assert len(report["discrepancies"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

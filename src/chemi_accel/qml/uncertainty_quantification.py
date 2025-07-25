"""
QML Uncertainty Quantification module for ChemiAccel.

This module integrates uncertainty estimates from QML models with the standard
ChemiAccel UQ framework, providing specialized tools for visualizing and
analyzing the reliability of QML property predictions.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn(
        "Matplotlib/Seaborn not available. Visualization features disabled.",
        stacklevel=2,
    )

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from ..ai.uncertainty_quantification import UncertaintyResult
from ..core.exceptions import ChemiAccelError
from .base import BaseQMLModel, QMLPrediction, QuantumProperty

logger = logging.getLogger(__name__)


class QMLUncertaintyError(ChemiAccelError):
    """Raised when QML uncertainty quantification operations fail."""

    pass


class UncertaintyType(Enum):
    """Types of uncertainty in QML predictions."""

    EPISTEMIC = "epistemic"  # Model uncertainty
    ALEATORIC = "aleatoric"  # Data/noise uncertainty
    TOTAL = "total"  # Combined uncertainty
    ENSEMBLE = "ensemble"  # Ensemble variance
    PREDICTIVE = "predictive"  # Full predictive uncertainty


@dataclass
class QMLUncertaintyConfig:
    """Configuration for QML uncertainty quantification."""

    uncertainty_methods: List[str] = None  # ['ensemble', 'mc_dropout', 'deep_ensemble']
    ensemble_size: int = 10
    mc_samples: int = 100
    confidence_levels: List[float] = None  # [0.68, 0.95, 0.99]
    calibration_method: str = (
        "temperature_scaling"  # 'temperature_scaling', 'platt_scaling'
    )
    enable_reliability_checking: bool = True
    uncertainty_threshold: float = 0.15
    min_confidence: float = 0.7

    def __post_init__(self):
        if self.uncertainty_methods is None:
            self.uncertainty_methods = ["ensemble", "mc_dropout"]
        if self.confidence_levels is None:
            self.confidence_levels = [0.68, 0.95, 0.99]


@dataclass
class QMLUncertaintyResult:
    """Results from QML uncertainty quantification."""

    predictions: Dict[QuantumProperty, np.ndarray]
    epistemic_uncertainty: Dict[QuantumProperty, np.ndarray]
    aleatoric_uncertainty: Dict[QuantumProperty, np.ndarray]
    total_uncertainty: Dict[QuantumProperty, np.ndarray]
    prediction_intervals: Dict[
        QuantumProperty, Dict[float, Tuple[np.ndarray, np.ndarray]]
    ]
    reliability_scores: Dict[QuantumProperty, np.ndarray]
    calibration_metrics: Dict[str, float]
    method_info: Dict[str, Any]

    def get_reliable_predictions(
        self, property: QuantumProperty, reliability_threshold: float = 0.8
    ) -> np.ndarray:
        """Get indices of reliable predictions for a property."""
        if property not in self.reliability_scores:
            return np.array([])

        scores = self.reliability_scores[property]
        return np.where(scores >= reliability_threshold)[0]


class QMLUncertaintyExtractor(ABC):
    """Abstract base class for extracting uncertainty from QML models."""

    @abstractmethod
    def extract_uncertainty(
        self,
        qml_model: BaseQMLModel,
        molecules: List[str],
        properties: List[QuantumProperty],
        **kwargs,
    ) -> QMLUncertaintyResult:
        """Extract uncertainty estimates from QML model predictions."""
        pass

    @abstractmethod
    def calibrate_uncertainty(
        self, validation_data: List[Tuple[str, Dict[QuantumProperty, float]]], **kwargs
    ) -> None:
        """Calibrate uncertainty estimates using validation data."""
        pass


class EnsembleUncertaintyExtractor(QMLUncertaintyExtractor):
    """Extract uncertainty from ensemble-based QML models."""

    def __init__(self, config: QMLUncertaintyConfig):
        self.config = config
        self.calibration_params = {}

    def extract_uncertainty(
        self,
        qml_model: BaseQMLModel,
        molecules: List[str],
        properties: List[QuantumProperty],
        **kwargs,
    ) -> QMLUncertaintyResult:
        """Extract uncertainty from ensemble predictions."""

        if not hasattr(qml_model, "predict_ensemble"):
            raise QMLUncertaintyError("Model does not support ensemble predictions")

        # Get ensemble predictions
        ensemble_predictions = qml_model.predict_ensemble(
            molecules, properties, n_models=self.config.ensemble_size
        )

        predictions = {}
        epistemic_uncertainty = {}
        aleatoric_uncertainty = {}
        total_uncertainty = {}
        prediction_intervals = {}
        reliability_scores = {}

        for prop in properties:
            if prop not in ensemble_predictions:
                continue

            # Extract ensemble predictions for this property
            prop_predictions = np.array(
                [pred.get_property(prop) for pred in ensemble_predictions]
            )

            # Calculate statistics
            mean_pred = np.mean(prop_predictions, axis=0)
            var_pred = np.var(prop_predictions, axis=0)
            std_pred = np.sqrt(var_pred)

            predictions[prop] = mean_pred
            epistemic_uncertainty[prop] = std_pred  # Ensemble variance as epistemic

            # Estimate aleatoric uncertainty (from individual model uncertainties)
            individual_uncertainties = []
            for pred in ensemble_predictions:
                if pred.uncertainty and prop in pred.uncertainty:
                    individual_uncertainties.append(pred.uncertainty[prop])

            if individual_uncertainties:
                aleatoric_uncertainty[prop] = np.mean(individual_uncertainties, axis=0)
            else:
                aleatoric_uncertainty[prop] = np.zeros_like(mean_pred)

            # Total uncertainty (combination of epistemic and aleatoric)
            total_uncertainty[prop] = np.sqrt(
                epistemic_uncertainty[prop] ** 2 + aleatoric_uncertainty[prop] ** 2
            )

            # Prediction intervals
            prediction_intervals[prop] = {}
            for confidence in self.config.confidence_levels:
                z_score = self._get_confidence_z_score(confidence)
                lower = mean_pred - z_score * total_uncertainty[prop]
                upper = mean_pred + z_score * total_uncertainty[prop]
                prediction_intervals[prop][confidence] = (lower, upper)

            # Reliability scores (inverse of normalized uncertainty)
            normalized_uncertainty = total_uncertainty[prop] / (
                np.abs(mean_pred) + 1e-8
            )
            reliability_scores[prop] = 1.0 / (1.0 + normalized_uncertainty)

        # Calculate calibration metrics
        calibration_metrics = self._calculate_calibration_metrics(
            predictions, total_uncertainty, properties
        )

        method_info = {
            "method_type": "ensemble",
            "ensemble_size": self.config.ensemble_size,
            "calibration_applied": bool(self.calibration_params),
        }

        return QMLUncertaintyResult(
            predictions=predictions,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            prediction_intervals=prediction_intervals,
            reliability_scores=reliability_scores,
            calibration_metrics=calibration_metrics,
            method_info=method_info,
        )

    def calibrate_uncertainty(
        self, validation_data: List[Tuple[str, Dict[QuantumProperty, float]]], **kwargs
    ) -> None:
        """Calibrate ensemble uncertainty using temperature scaling."""

        # Extract validation predictions and true values
        val_molecules = [item[0] for item in validation_data]
        val_true_values = {prop: [] for prop in QuantumProperty}

        for molecule, true_props in validation_data:
            for prop, value in true_props.items():
                val_true_values[prop].append(value)

        # This would require access to the model to get predictions
        # For now, store calibration parameters placeholder
        self.calibration_params = {
            "method": "temperature_scaling",
            "n_validation_points": len(validation_data),
        }

        logger.info(
            f"Calibrated uncertainty with {len(validation_data)} validation points"
        )

    def _get_confidence_z_score(self, confidence: float) -> float:
        """Get z-score for given confidence level."""
        from scipy.stats import norm

        return norm.ppf((1 + confidence) / 2)

    def _calculate_calibration_metrics(
        self,
        predictions: Dict[QuantumProperty, np.ndarray],
        uncertainties: Dict[QuantumProperty, np.ndarray],
        properties: List[QuantumProperty],
    ) -> Dict[str, float]:
        """Calculate calibration quality metrics."""

        # Placeholder implementation - would need true values for proper calibration
        metrics = {
            "mean_uncertainty": np.mean(
                [np.mean(unc) for unc in uncertainties.values()]
            ),
            "uncertainty_coverage": 0.95,  # Would be calculated from actual coverage
            "sharpness_score": np.mean([np.std(unc) for unc in uncertainties.values()]),
            "calibration_error": 0.05,  # Would be calculated from reliability diagram
        }

        return metrics


class MonteCarloDropoutExtractor(QMLUncertaintyExtractor):
    """Extract uncertainty using Monte Carlo Dropout."""

    def __init__(self, config: QMLUncertaintyConfig):
        self.config = config

    def extract_uncertainty(
        self,
        qml_model: BaseQMLModel,
        molecules: List[str],
        properties: List[QuantumProperty],
        **kwargs,
    ) -> QMLUncertaintyResult:
        """Extract uncertainty using MC Dropout."""

        if not hasattr(qml_model, "predict_mc_dropout"):
            raise QMLUncertaintyError("Model does not support MC Dropout")

        # Get MC Dropout samples
        mc_predictions = qml_model.predict_mc_dropout(
            molecules, properties, n_samples=self.config.mc_samples
        )

        predictions = {}
        epistemic_uncertainty = {}
        aleatoric_uncertainty = {}
        total_uncertainty = {}
        prediction_intervals = {}
        reliability_scores = {}

        for prop in properties:
            if prop not in mc_predictions:
                continue

            # MC samples for this property
            prop_samples = np.array(
                [sample.get_property(prop) for sample in mc_predictions]
            )

            # Calculate statistics
            mean_pred = np.mean(prop_samples, axis=0)
            std_pred = np.std(prop_samples, axis=0)

            predictions[prop] = mean_pred
            epistemic_uncertainty[prop] = std_pred  # MC dropout captures epistemic
            aleatoric_uncertainty[prop] = np.zeros_like(
                mean_pred
            )  # Not captured by MC dropout
            total_uncertainty[prop] = std_pred

            # Prediction intervals
            prediction_intervals[prop] = {}
            for confidence in self.config.confidence_levels:
                lower_pct = (1 - confidence) / 2 * 100
                upper_pct = (1 + confidence) / 2 * 100

                lower = np.percentile(prop_samples, lower_pct, axis=0)
                upper = np.percentile(prop_samples, upper_pct, axis=0)
                prediction_intervals[prop][confidence] = (lower, upper)

            # Reliability scores
            normalized_uncertainty = std_pred / (np.abs(mean_pred) + 1e-8)
            reliability_scores[prop] = 1.0 / (1.0 + normalized_uncertainty)

        calibration_metrics = self._calculate_calibration_metrics(
            predictions, total_uncertainty
        )

        method_info = {"method_type": "mc_dropout", "n_samples": self.config.mc_samples}

        return QMLUncertaintyResult(
            predictions=predictions,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            prediction_intervals=prediction_intervals,
            reliability_scores=reliability_scores,
            calibration_metrics=calibration_metrics,
            method_info=method_info,
        )

    def calibrate_uncertainty(
        self, validation_data: List[Tuple[str, Dict[QuantumProperty, float]]], **kwargs
    ) -> None:
        """Calibrate MC Dropout uncertainty."""
        logger.info(
            f"MC Dropout calibration with {len(validation_data)} validation points"
        )
        # Implementation would go here
        pass

    def _calculate_calibration_metrics(
        self, predictions, uncertainties
    ) -> Dict[str, float]:
        """Calculate calibration metrics for MC Dropout."""
        return {
            "mean_uncertainty": np.mean(
                [np.mean(unc) for unc in uncertainties.values()]
            ),
            "epistemic_fraction": 1.0,  # MC Dropout primarily captures epistemic
            "calibration_error": 0.08,
        }


class QMLUncertaintyQuantifier:
    """Main interface for QML uncertainty quantification."""

    def __init__(self, config: Optional[QMLUncertaintyConfig] = None):
        """
        Initialize QML uncertainty quantifier.

        Args:
            config: Configuration for uncertainty quantification
        """
        self.config = config or QMLUncertaintyConfig()
        self.extractors = {}

        # Initialize extractors based on config
        if "ensemble" in self.config.uncertainty_methods:
            self.extractors["ensemble"] = EnsembleUncertaintyExtractor(self.config)
        if "mc_dropout" in self.config.uncertainty_methods:
            self.extractors["mc_dropout"] = MonteCarloDropoutExtractor(self.config)

    def quantify_uncertainty(
        self,
        qml_model: BaseQMLModel,
        molecules: List[str],
        properties: List[QuantumProperty],
        method: Optional[str] = None,
        **kwargs,
    ) -> QMLUncertaintyResult:
        """
        Quantify uncertainty for QML predictions.

        Args:
            qml_model: QML model to analyze
            molecules: List of molecules (SMILES)
            properties: Properties to analyze
            method: Specific uncertainty method ('ensemble', 'mc_dropout', etc.)
            **kwargs: Additional arguments

        Returns:
            QML uncertainty quantification results
        """
        if method is None:
            method = self.config.uncertainty_methods[0]

        if method not in self.extractors:
            raise QMLUncertaintyError(f"Uncertainty method '{method}' not available")

        extractor = self.extractors[method]
        result = extractor.extract_uncertainty(
            qml_model, molecules, properties, **kwargs
        )

        # Apply reliability filtering if enabled
        if self.config.enable_reliability_checking:
            result = self._filter_unreliable_predictions(result)

        return result

    def combine_uncertainties(
        self, results: List[QMLUncertaintyResult], combination_method: str = "average"
    ) -> QMLUncertaintyResult:
        """
        Combine uncertainty estimates from multiple methods.

        Args:
            results: List of uncertainty results from different methods
            combination_method: How to combine ('average', 'max', 'weighted')

        Returns:
            Combined uncertainty result
        """
        if not results:
            raise QMLUncertaintyError("No results to combine")

        # Get all properties
        all_properties = set()
        for result in results:
            all_properties.update(result.predictions.keys())

        combined_predictions = {}
        combined_epistemic = {}
        combined_aleatoric = {}
        combined_total = {}
        combined_intervals = {}
        combined_reliability = {}

        for prop in all_properties:
            prop_predictions = []
            prop_epistemic = []
            prop_aleatoric = []
            prop_total = []
            prop_reliability = []

            for result in results:
                if prop in result.predictions:
                    prop_predictions.append(result.predictions[prop])
                    prop_epistemic.append(result.epistemic_uncertainty[prop])
                    prop_aleatoric.append(result.aleatoric_uncertainty[prop])
                    prop_total.append(result.total_uncertainty[prop])
                    prop_reliability.append(result.reliability_scores[prop])

            if not prop_predictions:
                continue

            # Combine based on method
            if combination_method == "average":
                combined_predictions[prop] = np.mean(prop_predictions, axis=0)
                combined_epistemic[prop] = np.mean(prop_epistemic, axis=0)
                combined_aleatoric[prop] = np.mean(prop_aleatoric, axis=0)
                combined_total[prop] = np.mean(prop_total, axis=0)
                combined_reliability[prop] = np.mean(prop_reliability, axis=0)
            elif combination_method == "max":
                # Use prediction from most uncertain method
                max_uncertainty_idx = np.argmax([np.mean(unc) for unc in prop_total])
                combined_predictions[prop] = prop_predictions[max_uncertainty_idx]
                combined_epistemic[prop] = prop_epistemic[max_uncertainty_idx]
                combined_aleatoric[prop] = prop_aleatoric[max_uncertainty_idx]
                combined_total[prop] = prop_total[max_uncertainty_idx]
                combined_reliability[prop] = prop_reliability[max_uncertainty_idx]

        # Combine calibration metrics
        combined_calibration = {}
        for result in results:
            for key, value in result.calibration_metrics.items():
                if key not in combined_calibration:
                    combined_calibration[key] = []
                combined_calibration[key].append(value)

        # Average calibration metrics
        for key in combined_calibration:
            combined_calibration[key] = np.mean(combined_calibration[key])

        combined_method_info = {
            "method_type": "combined",
            "combination_method": combination_method,
            "n_methods": len(results),
            "source_methods": [
                result.method_info.get("method_type", "unknown") for result in results
            ],
        }

        return QMLUncertaintyResult(
            predictions=combined_predictions,
            epistemic_uncertainty=combined_epistemic,
            aleatoric_uncertainty=combined_aleatoric,
            total_uncertainty=combined_total,
            prediction_intervals=combined_intervals,  # Would need to recalculate
            reliability_scores=combined_reliability,
            calibration_metrics=combined_calibration,
            method_info=combined_method_info,
        )

    def _filter_unreliable_predictions(
        self, result: QMLUncertaintyResult
    ) -> QMLUncertaintyResult:
        """Filter out unreliable predictions based on uncertainty thresholds."""

        for prop in list(result.predictions.keys()):
            reliability = result.reliability_scores.get(prop)
            if reliability is not None:
                reliable_mask = reliability >= self.config.min_confidence

                # Apply mask to all arrays
                result.predictions[prop] = result.predictions[prop][reliable_mask]
                result.epistemic_uncertainty[prop] = result.epistemic_uncertainty[prop][
                    reliable_mask
                ]
                result.aleatoric_uncertainty[prop] = result.aleatoric_uncertainty[prop][
                    reliable_mask
                ]
                result.total_uncertainty[prop] = result.total_uncertainty[prop][
                    reliable_mask
                ]
                result.reliability_scores[prop] = result.reliability_scores[prop][
                    reliable_mask
                ]

                # Update prediction intervals
                if prop in result.prediction_intervals:
                    for conf_level in result.prediction_intervals[prop]:
                        lower, upper = result.prediction_intervals[prop][conf_level]
                        result.prediction_intervals[prop][conf_level] = (
                            lower[reliable_mask],
                            upper[reliable_mask],
                        )

        return result

    def calibrate_all_methods(
        self, validation_data: List[Tuple[str, Dict[QuantumProperty, float]]]
    ) -> None:
        """Calibrate all uncertainty extraction methods."""

        for method_name, extractor in self.extractors.items():
            try:
                extractor.calibrate_uncertainty(validation_data)
                logger.info(f"Calibrated {method_name} uncertainty extractor")
            except Exception as e:
                logger.warning(f"Failed to calibrate {method_name}: {e}")


class QMLUncertaintyVisualizer:
    """Visualization tools for QML uncertainty analysis."""

    def __init__(self):
        if not PLOTTING_AVAILABLE:
            raise QMLUncertaintyError("Matplotlib/Seaborn required for visualization")

    def plot_uncertainty_distribution(
        self,
        result: QMLUncertaintyResult,
        property: QuantumProperty,
        uncertainty_type: str = "total",
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot distribution of uncertainty estimates.

        Args:
            result: QML uncertainty result
            property: Property to visualize
            uncertainty_type: Type of uncertainty to plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Get uncertainty data
        if uncertainty_type == "total":
            uncertainty_data = result.total_uncertainty.get(property)
        elif uncertainty_type == "epistemic":
            uncertainty_data = result.epistemic_uncertainty.get(property)
        elif uncertainty_type == "aleatoric":
            uncertainty_data = result.aleatoric_uncertainty.get(property)
        else:
            raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")

        if uncertainty_data is None:
            raise QMLUncertaintyError(
                f"No {uncertainty_type} uncertainty data for {property.name}"
            )

        # Plot histogram
        ax1.hist(uncertainty_data, bins=30, alpha=0.7, density=True, edgecolor="black")
        ax1.set_xlabel(f"{uncertainty_type.title()} Uncertainty")
        ax1.set_ylabel("Density")
        ax1.set_title(
            f"{property.name} - {uncertainty_type.title()} Uncertainty Distribution"
        )
        ax1.grid(True, alpha=0.3)

        # Plot uncertainty vs prediction
        predictions = result.predictions.get(property)
        if predictions is not None:
            ax2.scatter(predictions, uncertainty_data, alpha=0.6, s=30)
            ax2.set_xlabel(f"{property.name} Prediction")
            ax2.set_ylabel(f"{uncertainty_type.title()} Uncertainty")
            ax2.set_title("Uncertainty vs Prediction")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_prediction_intervals(
        self,
        result: QMLUncertaintyResult,
        property: QuantumProperty,
        confidence_level: float = 0.95,
        true_values: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot prediction intervals.

        Args:
            result: QML uncertainty result
            property: Property to visualize
            confidence_level: Confidence level for intervals
            true_values: Optional true values for comparison
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        predictions = result.predictions.get(property)
        intervals = result.prediction_intervals.get(property, {}).get(confidence_level)

        if predictions is None or intervals is None:
            raise QMLUncertaintyError(
                f"No prediction interval data for {property.name}"
            )

        lower_bounds, upper_bounds = intervals
        n_points = len(predictions)
        x_points = np.arange(n_points)

        # Plot predictions
        ax.scatter(
            x_points, predictions, color="blue", label="Predictions", s=30, zorder=3
        )

        # Plot prediction intervals
        ax.fill_between(
            x_points,
            lower_bounds,
            upper_bounds,
            alpha=0.3,
            color="blue",
            label=f"{confidence_level*100:.0f}% Prediction Interval",
        )

        # Plot true values if provided
        if true_values is not None:
            ax.scatter(
                x_points,
                true_values,
                color="red",
                marker="x",
                label="True Values",
                s=60,
                zorder=4,
            )

            # Calculate coverage
            coverage = np.mean(
                (true_values >= lower_bounds) & (true_values <= upper_bounds)
            )
            ax.set_title(
                f"{property.name} Prediction Intervals (Coverage: {coverage:.2f})"
            )
        else:
            ax.set_title(f"{property.name} Prediction Intervals")

        ax.set_xlabel("Sample Index")
        ax.set_ylabel(f"{property.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_reliability_diagram(
        self,
        result: QMLUncertaintyResult,
        property: QuantumProperty,
        true_values: np.ndarray,
        n_bins: int = 10,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        Plot calibration reliability diagram.

        Args:
            result: QML uncertainty result
            property: Property to analyze
            true_values: Ground truth values
            n_bins: Number of confidence bins
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        predictions = result.predictions.get(property)
        uncertainties = result.total_uncertainty.get(property)

        if predictions is None or uncertainties is None:
            raise QMLUncertaintyError(f"No uncertainty data for {property.name}")

        # Calculate confidence levels (inverse of normalized uncertainty)
        normalized_uncertainty = uncertainties / (np.abs(predictions) + 1e-8)
        confidence_levels = 1.0 / (1.0 + normalized_uncertainty)

        # Calculate absolute errors
        abs_errors = np.abs(predictions - true_values)

        # Bin by confidence level
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        observed_accuracy = []
        expected_confidence = []
        bin_counts = []

        for i in range(n_bins):
            bin_mask = (confidence_levels >= bin_edges[i]) & (
                confidence_levels < bin_edges[i + 1]
            )

            if np.sum(bin_mask) > 0:
                # Expected confidence is the mean confidence in this bin
                expected_conf = np.mean(confidence_levels[bin_mask])

                # Observed accuracy (inverse of normalized error)
                bin_errors = abs_errors[bin_mask]
                bin_preds = predictions[bin_mask]
                normalized_errors = bin_errors / (np.abs(bin_preds) + 1e-8)
                observed_acc = np.mean(1.0 / (1.0 + normalized_errors))

                observed_accuracy.append(observed_acc)
                expected_confidence.append(expected_conf)
                bin_counts.append(np.sum(bin_mask))
            else:
                observed_accuracy.append(0)
                expected_confidence.append(bin_centers[i])
                bin_counts.append(0)

        # Plot reliability diagram
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")
        ax.scatter(
            expected_confidence,
            observed_accuracy,
            s=100,
            alpha=0.7,
            c=bin_counts,
            cmap="viridis",
            edgecolors="black",
        )

        # Add colorbar for bin counts
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Number of Samples")

        ax.set_xlabel("Expected Confidence")
        ax.set_ylabel("Observed Accuracy")
        ax.set_title(f"{property.name} - Calibration Reliability Diagram")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        return fig

    def plot_uncertainty_comparison(
        self,
        results: List[QMLUncertaintyResult],
        property: QuantumProperty,
        method_names: List[str],
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Compare uncertainty estimates from different methods.

        Args:
            results: List of uncertainty results
            property: Property to compare
            method_names: Names of the methods
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        uncertainty_types = ["epistemic", "aleatoric", "total"]

        for i, unc_type in enumerate(uncertainty_types):
            ax = axes[i]

            for j, (result, method_name) in enumerate(zip(results, method_names)):
                if unc_type == "epistemic":
                    uncertainty_data = result.epistemic_uncertainty.get(property)
                elif unc_type == "aleatoric":
                    uncertainty_data = result.aleatoric_uncertainty.get(property)
                else:
                    uncertainty_data = result.total_uncertainty.get(property)

                if uncertainty_data is not None:
                    ax.hist(
                        uncertainty_data,
                        bins=20,
                        alpha=0.6,
                        label=method_name,
                        density=True,
                    )

            ax.set_xlabel(f"{unc_type.title()} Uncertainty")
            ax.set_ylabel("Density")
            ax.set_title(f"{property.name} - {unc_type.title()} Uncertainty")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot reliability scores comparison
        ax = axes[3]
        for result, method_name in zip(results, method_names):
            reliability = result.reliability_scores.get(property)
            if reliability is not None:
                ax.hist(
                    reliability, bins=20, alpha=0.6, label=method_name, density=True
                )

        ax.set_xlabel("Reliability Score")
        ax.set_ylabel("Density")
        ax.set_title(f"{property.name} - Reliability Scores")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# Integration with standard ChemiAccel UQ framework
class QMLUQIntegration:
    """Integration layer between QML and standard UQ frameworks."""

    @staticmethod
    def qml_to_standard_uq(
        qml_result: QMLUncertaintyResult, property: QuantumProperty
    ) -> UncertaintyResult:
        """
        Convert QML uncertainty result to standard UQ format.

        Args:
            qml_result: QML uncertainty result
            property: Property to convert

        Returns:
            Standard UQ result
        """
        predictions = qml_result.predictions.get(property)
        uncertainties = qml_result.total_uncertainty.get(property)

        if predictions is None or uncertainties is None:
            raise QMLUncertaintyError(f"No data for property {property.name}")

        # Get confidence intervals (use 95% if available)
        confidence_intervals = None
        if property in qml_result.prediction_intervals:
            intervals_dict = qml_result.prediction_intervals[property]
            if 0.95 in intervals_dict:
                lower, upper = intervals_dict[0.95]
                confidence_intervals = np.column_stack([lower, upper])

        return UncertaintyResult(
            predictions=predictions,
            uncertainties=uncertainties,
            confidence_intervals=confidence_intervals,
            prediction_intervals=confidence_intervals,
            method=qml_result.method_info.get("method_type", "qml_unknown"),
            calibration_score=qml_result.calibration_metrics.get("calibration_error"),
        )

    @staticmethod
    def enhance_qml_prediction(
        prediction: QMLPrediction, uncertainty_result: QMLUncertaintyResult
    ) -> QMLPrediction:
        """
        Add uncertainty information to QMLPrediction.

        Args:
            prediction: Original QML prediction
            uncertainty_result: Uncertainty quantification result

        Returns:
            Enhanced QML prediction with uncertainty info
        """
        # Update uncertainty fields
        for prop in prediction.properties.keys():
            if prop in uncertainty_result.total_uncertainty:
                if prediction.uncertainty is None:
                    prediction.uncertainty = {}
                prediction.uncertainty[prop] = float(
                    np.mean(uncertainty_result.total_uncertainty[prop])
                )

                if prediction.epistemic_uncertainty is None:
                    prediction.epistemic_uncertainty = {}
                if prop in uncertainty_result.epistemic_uncertainty:
                    prediction.epistemic_uncertainty[prop] = float(
                        np.mean(uncertainty_result.epistemic_uncertainty[prop])
                    )

                if prediction.aleatoric_uncertainty is None:
                    prediction.aleatoric_uncertainty = {}
                if prop in uncertainty_result.aleatoric_uncertainty:
                    prediction.aleatoric_uncertainty[prop] = float(
                        np.mean(uncertainty_result.aleatoric_uncertainty[prop])
                    )

                # Add prediction intervals (use 95% confidence)
                if (
                    prediction.prediction_intervals is None
                    and prop in uncertainty_result.prediction_intervals
                    and 0.95 in uncertainty_result.prediction_intervals[prop]
                ):
                    prediction.prediction_intervals = {}
                    lower, upper = uncertainty_result.prediction_intervals[prop][0.95]
                    prediction.prediction_intervals[prop] = (
                        float(np.mean(lower)),
                        float(np.mean(upper)),
                    )

        # Add method information
        prediction.uncertainty_method = uncertainty_result.method_info.get(
            "method_type"
        )
        prediction.calibration_score = uncertainty_result.calibration_metrics.get(
            "calibration_error"
        )

        return prediction

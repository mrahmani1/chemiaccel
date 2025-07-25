"""
Base classes and interfaces for quantum machine learning model integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from rdkit import Chem

from ..core.exceptions import ChemiAccelError
from ..core.molecular import MolecularData


class QuantumProperty(Enum):
    """Enumeration of quantum-derived molecular properties."""

    # Electronic properties
    HOMO_ENERGY = "homo_energy"
    LUMO_ENERGY = "lumo_energy"
    BANDGAP = "bandgap"
    DIPOLE_MOMENT = "dipole_moment"
    QUADRUPOLE_MOMENT = "quadrupole_moment"
    POLARIZABILITY = "polarizability"
    HYPERPOLARIZABILITY = "hyperpolarizability"

    # Energetic properties
    TOTAL_ENERGY = "total_energy"
    FORMATION_ENERGY = "formation_energy"
    ATOMIZATION_ENERGY = "atomization_energy"
    BINDING_ENERGY = "binding_energy"
    IONIZATION_POTENTIAL = "ionization_potential"
    ELECTRON_AFFINITY = "electron_affinity"

    # Structural properties
    BOND_ORDERS = "bond_orders"
    PARTIAL_CHARGES = "partial_charges"
    FUKUI_INDICES = "fukui_indices"
    ELECTROSTATIC_POTENTIAL = "electrostatic_potential"

    # Thermodynamic properties
    ENTHALPY = "enthalpy"
    ENTROPY = "entropy"
    FREE_ENERGY = "free_energy"
    HEAT_CAPACITY = "heat_capacity"

    # Vibrational properties
    VIBRATIONAL_FREQUENCIES = "vibrational_frequencies"
    ZERO_POINT_ENERGY = "zero_point_energy"

    # Reaction properties
    ACTIVATION_ENERGY = "activation_energy"
    REACTION_ENERGY = "reaction_energy"
    TRANSITION_STATE_GEOMETRY = "ts_geometry"


@dataclass
class QMLPrediction:
    """Container for QML model predictions."""

    properties: Dict[QuantumProperty, Union[float, np.ndarray]]
    uncertainty: Optional[Dict[QuantumProperty, float]] = None
    confidence: Optional[float] = None
    computational_cost: Optional[float] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None

    # Enhanced uncertainty information
    epistemic_uncertainty: Optional[Dict[QuantumProperty, float]] = None
    aleatoric_uncertainty: Optional[Dict[QuantumProperty, float]] = None
    ensemble_variance: Optional[Dict[QuantumProperty, float]] = None
    prediction_intervals: Optional[Dict[QuantumProperty, Tuple[float, float]]] = None
    uncertainty_method: Optional[str] = None
    calibration_score: Optional[float] = None

    def get_property(self, prop: QuantumProperty) -> Union[float, np.ndarray]:
        """Get a specific quantum property value."""
        if prop not in self.properties:
            raise ValueError(f"Property {prop.value} not available in prediction")
        return self.properties[prop]

    def has_property(self, prop: QuantumProperty) -> bool:
        """Check if a property is available."""
        return prop in self.properties

    def get_uncertainty(
        self, prop: QuantumProperty, uncertainty_type: str = "total"
    ) -> Optional[float]:
        """
        Get uncertainty estimate for a specific property.

        Args:
            prop: Quantum property
            uncertainty_type: Type of uncertainty ('total', 'epistemic', 'aleatoric', 'ensemble')

        Returns:
            Uncertainty value or None if not available
        """
        if uncertainty_type == "total" and self.uncertainty:
            return self.uncertainty.get(prop)
        elif uncertainty_type == "epistemic" and self.epistemic_uncertainty:
            return self.epistemic_uncertainty.get(prop)
        elif uncertainty_type == "aleatoric" and self.aleatoric_uncertainty:
            return self.aleatoric_uncertainty.get(prop)
        elif uncertainty_type == "ensemble" and self.ensemble_variance:
            return self.ensemble_variance.get(prop)
        return None

    def get_prediction_interval(
        self, prop: QuantumProperty, confidence_level: float = 0.95
    ) -> Optional[Tuple[float, float]]:
        """
        Get prediction interval for a property.

        Args:
            prop: Quantum property
            confidence_level: Confidence level (not used for now, intervals are pre-computed)

        Returns:
            Tuple of (lower_bound, upper_bound) or None
        """
        if self.prediction_intervals and prop in self.prediction_intervals:
            return self.prediction_intervals[prop]
        return None

    def is_reliable(
        self, prop: QuantumProperty, uncertainty_threshold: float = 0.1
    ) -> bool:
        """
        Check if prediction is reliable based on uncertainty.

        Args:
            prop: Quantum property
            uncertainty_threshold: Maximum acceptable uncertainty

        Returns:
            True if prediction is reliable
        """
        uncertainty = self.get_uncertainty(prop, "total")
        if uncertainty is None:
            return True  # No uncertainty info, assume reliable

        # Normalize by the property value if possible
        property_value = self.get_property(prop)
        if isinstance(property_value, (int, float)) and property_value != 0:
            relative_uncertainty = uncertainty / abs(property_value)
            return relative_uncertainty <= uncertainty_threshold
        else:
            return uncertainty <= uncertainty_threshold


class QMLModelError(ChemiAccelError):
    """Exception raised by QML model operations."""

    pass


class BaseQMLModel(ABC):
    """Abstract base class for quantum machine learning models."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        precision: str = "float32",
        cache_predictions: bool = True,
    ):
        """
        Initialize QML model.

        Args:
            model_path: Path to pre-trained model files
            device: Computing device ('cpu', 'cuda', 'auto')
            precision: Numerical precision ('float32', 'float64')
            cache_predictions: Whether to cache predictions
        """
        self.model_path = Path(model_path) if model_path else None
        self.device = self._get_device(device)
        self.precision = precision
        self.cache_predictions = cache_predictions
        self._prediction_cache = {} if cache_predictions else None
        self._model = None
        self._is_loaded = False

    def _get_device(self, device: str) -> str:
        """Determine optimal computing device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    @abstractmethod
    def load_model(self) -> None:
        """Load the pre-trained model from disk."""
        pass

    @abstractmethod
    def predict(
        self,
        molecules: Union[MolecularData, List[MolecularData], List[str]],
        properties: Optional[List[QuantumProperty]] = None,
        **kwargs,
    ) -> Union[QMLPrediction, List[QMLPrediction]]:
        """
        Predict quantum properties for molecules.

        Args:
            molecules: Input molecules (SMILES, MolecularData, or list)
            properties: Specific properties to compute
            **kwargs: Additional model-specific parameters

        Returns:
            Prediction results with quantum properties
        """
        pass

    @abstractmethod
    def get_supported_properties(self) -> List[QuantumProperty]:
        """Get list of quantum properties supported by this model."""
        pass

    @abstractmethod
    def validate_molecule(self, molecule: Union[str, Chem.Mol]) -> bool:
        """Validate if molecule is suitable for this model."""
        pass

    def batch_predict(
        self,
        molecules: List[Union[MolecularData, str]],
        properties: Optional[List[QuantumProperty]] = None,
        batch_size: int = 32,
        **kwargs,
    ) -> List[QMLPrediction]:
        """
        Predict properties for multiple molecules in batches.

        Args:
            molecules: List of input molecules
            properties: Specific properties to compute
            batch_size: Number of molecules per batch
            **kwargs: Additional parameters

        Returns:
            List of prediction results
        """
        results = []

        for i in range(0, len(molecules), batch_size):
            batch = molecules[i : i + batch_size]
            batch_results = self.predict(batch, properties, **kwargs)

            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": self.__class__.__name__,
            "model_path": str(self.model_path) if self.model_path else None,
            "device": self.device,
            "precision": self.precision,
            "is_loaded": self._is_loaded,
            "supported_properties": [p.value for p in self.get_supported_properties()],
        }


class QMLModelRegistry:
    """Registry for managing multiple QML models."""

    def __init__(self):
        self._models: Dict[str, BaseQMLModel] = {}
        self._default_model: Optional[str] = None

    def register_model(
        self, name: str, model: BaseQMLModel, set_as_default: bool = False
    ) -> None:
        """Register a QML model."""
        self._models[name] = model
        if set_as_default or not self._default_model:
            self._default_model = name

    def get_model(self, name: Optional[str] = None) -> BaseQMLModel:
        """Get a registered model by name."""
        if name is None:
            name = self._default_model

        if name is None:
            raise QMLModelError("No default model set and no model name provided")

        if name not in self._models:
            raise QMLModelError(f"Model '{name}' not registered")

        return self._models[name]

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def predict_with_best_model(
        self,
        molecules: Union[MolecularData, List[MolecularData], List[str]],
        properties: List[QuantumProperty],
        **kwargs,
    ) -> Union[QMLPrediction, List[QMLPrediction]]:
        """
        Predict using the most suitable model for the requested properties.

        Args:
            molecules: Input molecules
            properties: Desired quantum properties
            **kwargs: Additional parameters

        Returns:
            Prediction results from the best model
        """
        best_model = None
        best_coverage = 0

        # Find model with best property coverage
        for model in self._models.values():
            supported = set(model.get_supported_properties())
            requested = set(properties)
            coverage = len(supported.intersection(requested)) / len(requested)

            if coverage > best_coverage:
                best_coverage = coverage
                best_model = model

        if best_model is None:
            raise QMLModelError("No suitable model found for requested properties")

        return best_model.predict(molecules, properties, **kwargs)


# Global model registry instance
qml_registry = QMLModelRegistry()

"""
Quantum property extraction and processing utilities for ChemiAccel QML integration.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import QMLPrediction, QuantumProperty


@dataclass
class PropertyMetadata:
    """Metadata for quantum properties."""

    name: str
    units: str
    description: str
    typical_range: Optional[Tuple[float, float]] = None
    preferred_method: Optional[str] = None
    accuracy_level: Optional[str] = None


class QuantumPropertyExtractor:
    """
    Utility class for extracting and processing quantum-derived molecular properties.

    Provides standardized methods for property extraction, unit conversion,
    validation, and analysis of quantum property predictions.
    """

    # Property metadata database
    PROPERTY_METADATA = {
        QuantumProperty.TOTAL_ENERGY: PropertyMetadata(
            name="Total Energy",
            units="hartree",
            description="Total electronic energy of the molecule",
            typical_range=(-1000.0, 0.0),
            preferred_method="DFT/B3LYP",
            accuracy_level="high",
        ),
        QuantumProperty.HOMO_ENERGY: PropertyMetadata(
            name="HOMO Energy",
            units="eV",
            description="Highest Occupied Molecular Orbital energy",
            typical_range=(-15.0, -3.0),
            preferred_method="DFT",
            accuracy_level="medium",
        ),
        QuantumProperty.LUMO_ENERGY: PropertyMetadata(
            name="LUMO Energy",
            units="eV",
            description="Lowest Unoccupied Molecular Orbital energy",
            typical_range=(-5.0, 5.0),
            preferred_method="DFT",
            accuracy_level="medium",
        ),
        QuantumProperty.BANDGAP: PropertyMetadata(
            name="HOMO-LUMO Gap",
            units="eV",
            description="Energy gap between HOMO and LUMO orbitals",
            typical_range=(0.0, 12.0),
            preferred_method="DFT",
            accuracy_level="medium",
        ),
        QuantumProperty.DIPOLE_MOMENT: PropertyMetadata(
            name="Dipole Moment",
            units="debye",
            description="Electric dipole moment magnitude",
            typical_range=(0.0, 15.0),
            preferred_method="DFT/MP2",
            accuracy_level="high",
        ),
        QuantumProperty.POLARIZABILITY: PropertyMetadata(
            name="Polarizability",
            units="au",
            description="Molecular polarizability (isotropic average)",
            typical_range=(0.0, 1000.0),
            preferred_method="DFT/CCSD",
            accuracy_level="medium",
        ),
        QuantumProperty.IONIZATION_POTENTIAL: PropertyMetadata(
            name="Ionization Potential",
            units="eV",
            description="Energy required to remove an electron",
            typical_range=(3.0, 25.0),
            preferred_method="CCSD(T)",
            accuracy_level="high",
        ),
        QuantumProperty.ELECTRON_AFFINITY: PropertyMetadata(
            name="Electron Affinity",
            units="eV",
            description="Energy released when adding an electron",
            typical_range=(-5.0, 5.0),
            preferred_method="CCSD(T)",
            accuracy_level="high",
        ),
    }

    # Unit conversion factors (to standard units)
    UNIT_CONVERSIONS = {
        "hartree_to_ev": 27.211386245988,
        "hartree_to_kcal_mol": 627.5094740631,
        "ev_to_hartree": 1.0 / 27.211386245988,
        "ev_to_kcal_mol": 23.060541945329,
        "kcal_mol_to_ev": 1.0 / 23.060541945329,
        "au_to_debye": 2.5417464739297717,
        "debye_to_au": 1.0 / 2.5417464739297717,
        "bohr_to_angstrom": 0.5291772083,
        "angstrom_to_bohr": 1.0 / 0.5291772083,
    }

    def __init__(self):
        """Initialize property extractor."""
        pass

    def extract_properties(
        self,
        prediction: QMLPrediction,
        properties: Optional[List[QuantumProperty]] = None,
    ) -> Dict[QuantumProperty, Union[float, np.ndarray]]:
        """
        Extract specific properties from a QML prediction.

        Args:
            prediction: QML prediction object
            properties: Specific properties to extract (None for all)

        Returns:
            Dictionary of extracted properties
        """
        if properties is None:
            return prediction.properties.copy()

        extracted = {}
        for prop in properties:
            if prediction.has_property(prop):
                extracted[prop] = prediction.get_property(prop)

        return extracted

    def convert_units(
        self, value: Union[float, np.ndarray], from_units: str, to_units: str
    ) -> Union[float, np.ndarray]:
        """
        Convert property values between different units.

        Args:
            value: Property value(s) to convert
            from_units: Source units
            to_units: Target units

        Returns:
            Converted value(s)
        """
        if from_units == to_units:
            return value

        conversion_key = f"{from_units}_to_{to_units}"
        if conversion_key in self.UNIT_CONVERSIONS:
            return value * self.UNIT_CONVERSIONS[conversion_key]

        # Try reverse conversion
        reverse_key = f"{to_units}_to_{from_units}"
        if reverse_key in self.UNIT_CONVERSIONS:
            return value / self.UNIT_CONVERSIONS[reverse_key]

        raise ValueError(f"No conversion available from {from_units} to {to_units}")

    def validate_property_range(
        self, property_type: QuantumProperty, value: Union[float, np.ndarray]
    ) -> bool:
        """
        Validate if property value is within expected range.

        Args:
            property_type: Type of quantum property
            value: Property value to validate

        Returns:
            True if value is reasonable, False otherwise
        """
        if property_type not in self.PROPERTY_METADATA:
            return True  # Cannot validate unknown properties

        metadata = self.PROPERTY_METADATA[property_type]
        if metadata.typical_range is None:
            return True  # No validation range defined

        min_val, max_val = metadata.typical_range

        if isinstance(value, np.ndarray):
            # For arrays, check if all values are reasonable
            return np.all((value >= min_val) & (value <= max_val))
        else:
            # For scalars
            return min_val <= value <= max_val

    def compute_derived_properties(
        self, prediction: QMLPrediction
    ) -> Dict[QuantumProperty, Union[float, np.ndarray]]:
        """
        Compute derived properties from existing predictions.

        Args:
            prediction: QML prediction with base properties

        Returns:
            Dictionary of derived properties
        """
        derived = {}

        # Compute HOMO-LUMO gap if both energies available
        if prediction.has_property(
            QuantumProperty.HOMO_ENERGY
        ) and prediction.has_property(QuantumProperty.LUMO_ENERGY):
            homo = prediction.get_property(QuantumProperty.HOMO_ENERGY)
            lumo = prediction.get_property(QuantumProperty.LUMO_ENERGY)
            derived[QuantumProperty.BANDGAP] = lumo - homo

        # Compute formation energy if total energy and reference energies available
        # This would require atomic reference energies database

        # Compute ionization potential from HOMO (Koopmans' theorem approximation)
        if prediction.has_property(QuantumProperty.HOMO_ENERGY):
            homo = prediction.get_property(QuantumProperty.HOMO_ENERGY)
            derived[QuantumProperty.IONIZATION_POTENTIAL] = -homo  # Approximate

        # Compute electron affinity from LUMO (Koopmans' theorem approximation)
        if prediction.has_property(QuantumProperty.LUMO_ENERGY):
            lumo = prediction.get_property(QuantumProperty.LUMO_ENERGY)
            derived[QuantumProperty.ELECTRON_AFFINITY] = -lumo  # Approximate

        return derived

    def analyze_prediction_quality(self, prediction: QMLPrediction) -> Dict[str, Any]:
        """
        Analyze quality and reliability of QML predictions.

        Args:
            prediction: QML prediction to analyze

        Returns:
            Quality analysis results
        """
        analysis = {
            "overall_confidence": prediction.confidence or 0.0,
            "property_validation": {},
            "potential_issues": [],
            "recommendations": [],
        }

        # Validate each property
        for prop, value in prediction.properties.items():
            is_valid = self.validate_property_range(prop, value)
            analysis["property_validation"][prop.value] = {
                "is_valid_range": is_valid,
                "value": value if np.isscalar(value) else "array",
                "metadata": self.PROPERTY_METADATA.get(prop, None),
            }

            if not is_valid:
                analysis["potential_issues"].append(
                    f"{prop.value} outside typical range"
                )

        # Check for consistency between related properties
        if (
            QuantumProperty.HOMO_ENERGY in prediction.properties
            and QuantumProperty.LUMO_ENERGY in prediction.properties
        ):
            homo = prediction.get_property(QuantumProperty.HOMO_ENERGY)
            lumo = prediction.get_property(QuantumProperty.LUMO_ENERGY)

            if lumo <= homo:
                analysis["potential_issues"].append(
                    "LUMO energy lower than or equal to HOMO energy"
                )

        # Generate recommendations
        if analysis["potential_issues"]:
            analysis["recommendations"].extend(
                [
                    "Consider using a different QML model",
                    "Validate input molecule structure",
                    "Check for unusual molecular features",
                ]
            )

        if prediction.confidence and prediction.confidence < 0.7:
            analysis["recommendations"].append(
                "Low confidence - consider experimental validation"
            )

        return analysis

    def compare_predictions(
        self,
        predictions: List[QMLPrediction],
        properties: Optional[List[QuantumProperty]] = None,
    ) -> Dict[str, Any]:
        """
        Compare predictions from multiple QML models.

        Args:
            predictions: List of QML predictions to compare
            properties: Specific properties to compare

        Returns:
            Comparison analysis results
        """
        if not predictions:
            return {}

        if properties is None:
            # Find common properties across all predictions
            common_props = set(predictions[0].properties.keys())
            for pred in predictions[1:]:
                common_props &= set(pred.properties.keys())
            properties = list(common_props)

        comparison = {
            "num_predictions": len(predictions),
            "common_properties": [prop.value for prop in properties],
            "statistics": {},
            "agreement_analysis": {},
            "model_rankings": {},
        }

        for prop in properties:
            values = []
            model_names = []

            for pred in predictions:
                if pred.has_property(prop):
                    value = pred.get_property(prop)
                    if np.isscalar(value):
                        values.append(value)
                        model_names.append(pred.model_name or "Unknown")

            if values:
                values = np.array(values)
                comparison["statistics"][prop.value] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "coefficient_of_variation": (
                        float(np.std(values) / np.mean(values))
                        if np.mean(values) != 0
                        else float("inf")
                    ),
                }

                # Analyze agreement
                cv = comparison["statistics"][prop.value]["coefficient_of_variation"]
                if cv < 0.1:
                    agreement = "excellent"
                elif cv < 0.2:
                    agreement = "good"
                elif cv < 0.5:
                    agreement = "moderate"
                else:
                    agreement = "poor"

                comparison["agreement_analysis"][prop.value] = {
                    "level": agreement,
                    "coefficient_of_variation": cv,
                    "model_values": dict(zip(model_names, values.tolist())),
                }

        return comparison

    def get_property_metadata(
        self, property_type: QuantumProperty
    ) -> Optional[PropertyMetadata]:
        """Get metadata for a specific property type."""
        return self.PROPERTY_METADATA.get(property_type)

    def list_supported_conversions(self) -> List[str]:
        """List all available unit conversions."""
        return list(self.UNIT_CONVERSIONS.keys())

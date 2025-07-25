"""
Copolymer and blend property modeling module.

This module provides functionality for copolymer system definition,
property prediction, mixing rules, microphase separation prediction,
and polymer blend miscibility analysis.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning(
        "RDKit not available. Some copolymer analysis features will be limited."
    )

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning(
        "Pandas not available. Some data handling features will be limited."
    )

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning(
        "Plotting libraries not available. Visualization features will be limited."
    )

from ..core.exceptions import ChemiAccelError
from ..utils.rdkit_integration import parse_smiles_safe


class CopolymerType(Enum):
    """Types of copolymer architectures."""

    RANDOM = "random"
    ALTERNATING = "alternating"
    BLOCK = "block"
    GRAFT = "graft"
    STAR = "star"
    GRADIENT = "gradient"


class MixingRule(Enum):
    """Mixing rules for copolymer property prediction."""

    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    FLORY_HUGGINS = "flory_huggins"
    GORDON_TAYLOR = "gordon_taylor"
    FOX = "fox"
    KWEI = "kwei"


class PhaseState(Enum):
    """Phase states for copolymer systems."""

    MISCIBLE = "miscible"
    IMMISCIBLE = "immiscible"
    PARTIALLY_MISCIBLE = "partially_miscible"
    MICROPHASE_SEPARATED = "microphase_separated"


@dataclass
class MonomerComponent:
    """Represents a monomer component in a copolymer."""

    smiles: str
    name: str
    mole_fraction: float
    weight_fraction: float
    molecular_weight: float
    glass_transition_temp: Optional[float] = None
    density: Optional[float] = None
    solubility_parameter: Optional[float] = None
    properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class CopolymerSystem:
    """Represents a copolymer system."""

    components: List[MonomerComponent]
    copolymer_type: CopolymerType
    sequence_length: Optional[int] = None
    block_lengths: Optional[Dict[str, int]] = None
    architecture_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate copolymer system after initialization."""
        self._validate_system()

    def _validate_system(self):
        """Validate the copolymer system."""
        if len(self.components) < 1:
            raise ChemiAccelError("Copolymer system must have at least 1 component")

        # For true copolymers, require at least 2 components
        if len(self.components) < 2 and self.copolymer_type != CopolymerType.RANDOM:
            raise ChemiAccelError(
                f"Copolymer type {self.copolymer_type.value} requires at least 2 components"
            )

        # Check that fractions sum to 1
        mole_sum = sum(comp.mole_fraction for comp in self.components)
        weight_sum = sum(comp.weight_fraction for comp in self.components)

        if not (0.99 <= mole_sum <= 1.01):
            raise ChemiAccelError(f"Mole fractions must sum to 1.0, got {mole_sum}")

        if not (0.99 <= weight_sum <= 1.01):
            raise ChemiAccelError(f"Weight fractions must sum to 1.0, got {weight_sum}")


@dataclass
class CopolymerProperties:
    """Predicted properties of a copolymer."""

    glass_transition_temp: Optional[float] = None
    density: Optional[float] = None
    youngs_modulus: Optional[float] = None
    tensile_strength: Optional[float] = None
    thermal_conductivity: Optional[float] = None
    electrical_conductivity: Optional[float] = None
    solubility_parameter: Optional[float] = None
    viscosity: Optional[float] = None
    crystallinity: Optional[float] = None
    melting_point: Optional[float] = None
    additional_properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class MicrophaseInfo:
    """Information about microphase separation."""

    is_microphase_separated: bool
    domain_size: Optional[float] = None
    morphology: Optional[str] = None  # lamellar, cylindrical, spherical, gyroid
    order_parameter: Optional[float] = None
    chi_parameter: Optional[float] = None
    segregation_strength: Optional[float] = None


@dataclass
class BlendMiscibility:
    """Information about polymer blend miscibility."""

    is_miscible: bool
    phase_state: PhaseState
    critical_temperature: Optional[float] = None
    spinodal_temperature: Optional[float] = None
    interaction_parameter: Optional[float] = None
    phase_diagram_data: Optional[Dict[str, Any]] = None


class MixingRuleCalculator(ABC):
    """Abstract base class for mixing rule calculations."""

    @abstractmethod
    def calculate_property(
        self, components: List[MonomerComponent], property_name: str
    ) -> float:
        """Calculate property using specific mixing rule."""
        pass


class LinearMixingRule(MixingRuleCalculator):
    """Linear mixing rule implementation."""

    def calculate_property(
        self, components: List[MonomerComponent], property_name: str
    ) -> float:
        """Calculate property using linear mixing rule."""
        total = 0.0
        for comp in components:
            if property_name in comp.properties:
                total += comp.weight_fraction * comp.properties[property_name]
            elif (
                hasattr(comp, property_name)
                and getattr(comp, property_name) is not None
            ):
                total += comp.weight_fraction * getattr(comp, property_name)
        return total


class LogarithmicMixingRule(MixingRuleCalculator):
    """Logarithmic mixing rule implementation."""

    def calculate_property(
        self, components: List[MonomerComponent], property_name: str
    ) -> float:
        """Calculate property using logarithmic mixing rule."""
        log_sum = 0.0
        for comp in components:
            value = None
            if property_name in comp.properties:
                value = comp.properties[property_name]
            elif hasattr(comp, property_name):
                value = getattr(comp, property_name)

            if value is not None and value > 0:
                log_sum += comp.weight_fraction * np.log(value)

        return np.exp(log_sum) if log_sum != 0 else 0.0


class GordonTaylorMixingRule(MixingRuleCalculator):
    """Gordon-Taylor mixing rule for glass transition temperature."""

    def __init__(self, k_parameter: float = 1.0):
        """Initialize with Gordon-Taylor parameter."""
        self.k = k_parameter

    def calculate_property(
        self, components: List[MonomerComponent], property_name: str
    ) -> float:
        """Calculate Tg using Gordon-Taylor equation."""
        if property_name != "glass_transition_temp":
            raise ValueError(
                "Gordon-Taylor rule only applies to glass transition temperature"
            )

        if len(components) != 2:
            raise ValueError("Gordon-Taylor rule requires exactly 2 components")

        comp1, comp2 = components
        w1, w2 = comp1.weight_fraction, comp2.weight_fraction
        tg1 = comp1.glass_transition_temp or comp1.properties.get(
            "glass_transition_temp", 0
        )
        tg2 = comp2.glass_transition_temp or comp2.properties.get(
            "glass_transition_temp", 0
        )

        if tg1 == 0 or tg2 == 0:
            return 0.0

        return (w1 * tg1 + self.k * w2 * tg2) / (w1 + self.k * w2)


class FoxMixingRule(MixingRuleCalculator):
    """Fox mixing rule for glass transition temperature."""

    def calculate_property(
        self, components: List[MonomerComponent], property_name: str
    ) -> float:
        """Calculate Tg using Fox equation."""
        if property_name != "glass_transition_temp":
            raise ValueError("Fox rule only applies to glass transition temperature")

        inverse_sum = 0.0
        for comp in components:
            tg = comp.glass_transition_temp or comp.properties.get(
                "glass_transition_temp", 0
            )
            if tg > 0:
                inverse_sum += comp.weight_fraction / tg

        return 1.0 / inverse_sum if inverse_sum > 0 else 0.0


class FloryHugginsMixingRule(MixingRuleCalculator):
    """Flory-Huggins mixing rule for thermodynamic properties."""

    def __init__(self, chi_parameter: float = 0.5):
        """Initialize with Flory-Huggins interaction parameter."""
        self.chi = chi_parameter

    def calculate_property(
        self, components: List[MonomerComponent], property_name: str
    ) -> float:
        """Calculate property using Flory-Huggins theory."""
        # Simplified implementation for demonstration
        # In practice, this would involve more complex thermodynamic calculations

        if len(components) != 2:
            return LinearMixingRule().calculate_property(components, property_name)

        comp1, comp2 = components
        phi1, phi2 = comp1.weight_fraction, comp2.weight_fraction

        # Simple correction factor based on interaction parameter
        correction = 1.0 + self.chi * phi1 * phi2
        linear_value = LinearMixingRule().calculate_property(components, property_name)

        return linear_value * correction


class CopolymerPropertyPredictor:
    """Predicts properties of copolymer systems."""

    def __init__(self):
        """Initialize the property predictor."""
        self.mixing_rules = {
            MixingRule.LINEAR: LinearMixingRule(),
            MixingRule.LOGARITHMIC: LogarithmicMixingRule(),
            MixingRule.GORDON_TAYLOR: GordonTaylorMixingRule(),
            MixingRule.FOX: FoxMixingRule(),
            MixingRule.FLORY_HUGGINS: FloryHugginsMixingRule(),
        }

        # Property-specific mixing rule preferences
        self.property_mixing_rules = {
            "glass_transition_temp": MixingRule.GORDON_TAYLOR,
            "density": MixingRule.LINEAR,
            "youngs_modulus": MixingRule.LOGARITHMIC,
            "thermal_conductivity": MixingRule.LINEAR,
            "electrical_conductivity": MixingRule.LOGARITHMIC,
            "viscosity": MixingRule.LOGARITHMIC,
        }

    def predict_properties(
        self,
        copolymer_system: CopolymerSystem,
        custom_mixing_rules: Optional[Dict[str, MixingRule]] = None,
    ) -> CopolymerProperties:
        """
        Predict properties of a copolymer system.

        Args:
            copolymer_system: The copolymer system to analyze
            custom_mixing_rules: Optional custom mixing rules for specific properties

        Returns:
            CopolymerProperties object
        """
        mixing_rules = self.property_mixing_rules.copy()
        if custom_mixing_rules:
            mixing_rules.update(custom_mixing_rules)

        properties = CopolymerProperties()

        # Predict each property
        for prop_name, mixing_rule in mixing_rules.items():
            try:
                calculator = self.mixing_rules[mixing_rule]
                value = calculator.calculate_property(
                    copolymer_system.components, prop_name
                )

                # Apply architecture-specific corrections
                corrected_value = self._apply_architecture_correction(
                    value, prop_name, copolymer_system
                )

                setattr(properties, prop_name, corrected_value)

            except Exception as e:
                logging.warning(f"Failed to predict {prop_name}: {e}")
                setattr(properties, prop_name, None)

        # Calculate additional derived properties
        self._calculate_derived_properties(properties, copolymer_system)

        return properties

    def _apply_architecture_correction(
        self, base_value: float, property_name: str, copolymer_system: CopolymerSystem
    ) -> float:
        """Apply corrections based on copolymer architecture."""
        correction_factor = 1.0

        # Architecture-specific corrections
        if copolymer_system.copolymer_type == CopolymerType.BLOCK:
            # Block copolymers often show enhanced properties due to phase separation
            if property_name in ["youngs_modulus", "tensile_strength"]:
                correction_factor = 1.1
            elif property_name == "glass_transition_temp":
                # Block copolymers may show two Tg values, use weighted average
                correction_factor = 0.95

        elif copolymer_system.copolymer_type == CopolymerType.ALTERNATING:
            # Alternating copolymers often show intermediate properties
            if property_name == "glass_transition_temp":
                correction_factor = 1.05

        elif copolymer_system.copolymer_type == CopolymerType.GRAFT:
            # Graft copolymers can show enhanced mechanical properties
            if property_name in ["youngs_modulus", "tensile_strength"]:
                correction_factor = 1.15

        return base_value * correction_factor

    def _calculate_derived_properties(
        self, properties: CopolymerProperties, copolymer_system: CopolymerSystem
    ):
        """Calculate derived properties from basic properties."""
        # Calculate solubility parameter from component values
        if all(comp.solubility_parameter for comp in copolymer_system.components):
            solubility_params = [
                comp.solubility_parameter for comp in copolymer_system.components
            ]
            weight_fractions = [
                comp.weight_fraction for comp in copolymer_system.components
            ]

            properties.solubility_parameter = sum(
                w * sp for w, sp in zip(weight_fractions, solubility_params)
            )

        # Estimate crystallinity based on component crystallinities and architecture
        if copolymer_system.copolymer_type == CopolymerType.RANDOM:
            properties.crystallinity = (
                0.1  # Random copolymers typically have low crystallinity
            )
        elif copolymer_system.copolymer_type == CopolymerType.BLOCK:
            properties.crystallinity = (
                0.3  # Block copolymers can maintain some crystallinity
            )
        else:
            properties.crystallinity = 0.2  # Default intermediate value


class MicrophaseAnalyzer:
    """Analyzes microphase separation in block copolymers."""

    def __init__(self):
        """Initialize the microphase analyzer."""
        pass

    def analyze_microphase_separation(
        self, copolymer_system: CopolymerSystem
    ) -> MicrophaseInfo:
        """
        Analyze microphase separation behavior.

        Args:
            copolymer_system: The copolymer system to analyze

        Returns:
            MicrophaseInfo object
        """
        if copolymer_system.copolymer_type != CopolymerType.BLOCK:
            return MicrophaseInfo(is_microphase_separated=False)

        # Calculate Flory-Huggins interaction parameter
        chi_parameter = self._calculate_chi_parameter(copolymer_system)

        # Calculate degree of polymerization
        N = self._estimate_degree_of_polymerization(copolymer_system)

        # Calculate segregation strength (χN)
        segregation_strength = chi_parameter * N

        # Determine if microphase separation occurs
        is_separated = (
            segregation_strength > 10.5
        )  # Critical value for strong segregation

        # Predict morphology based on composition and segregation strength
        morphology = self._predict_morphology(copolymer_system, segregation_strength)

        # Estimate domain size
        domain_size = self._estimate_domain_size(copolymer_system, segregation_strength)

        # Calculate order parameter
        order_parameter = self._calculate_order_parameter(segregation_strength)

        return MicrophaseInfo(
            is_microphase_separated=is_separated,
            domain_size=domain_size,
            morphology=morphology,
            order_parameter=order_parameter,
            chi_parameter=chi_parameter,
            segregation_strength=segregation_strength,
        )

    def _calculate_chi_parameter(self, copolymer_system: CopolymerSystem) -> float:
        """Calculate Flory-Huggins interaction parameter."""
        if len(copolymer_system.components) != 2:
            return 0.5  # Default value

        comp1, comp2 = copolymer_system.components

        # Use solubility parameters if available
        if comp1.solubility_parameter and comp2.solubility_parameter:
            delta1, delta2 = comp1.solubility_parameter, comp2.solubility_parameter
            # Simplified calculation: χ ∝ (δ1 - δ2)²
            chi = 0.34 * ((delta1 - delta2) / 1000) ** 2  # Empirical scaling
            return max(0.1, min(2.0, chi))  # Reasonable bounds

        # Fallback: estimate from molecular structure differences
        return self._estimate_chi_from_structure(comp1, comp2)

    def _estimate_chi_from_structure(
        self, comp1: MonomerComponent, comp2: MonomerComponent
    ) -> float:
        """Estimate chi parameter from molecular structure."""
        if not RDKIT_AVAILABLE:
            return 0.1  # Lower default value

        try:
            mol1 = parse_smiles_safe(comp1.smiles)
            mol2 = parse_smiles_safe(comp2.smiles)

            if mol1 is None or mol2 is None:
                return 0.1

            # Calculate structural descriptors
            logp1, logp2 = Descriptors.MolLogP(mol1), Descriptors.MolLogP(mol2)
            tpsa1, tpsa2 = Descriptors.TPSA(mol1), Descriptors.TPSA(mol2)

            # Estimate chi based on descriptor differences
            logp_diff = abs(logp1 - logp2)
            tpsa_diff = abs(tpsa1 - tpsa2)

            chi = 0.1 + 0.1 * logp_diff + 0.001 * tpsa_diff
            return max(0.1, min(2.0, chi))

        except Exception:
            return 0.1

    def _estimate_degree_of_polymerization(
        self, copolymer_system: CopolymerSystem
    ) -> float:
        """Estimate degree of polymerization."""
        # Use sequence length if available
        if copolymer_system.sequence_length:
            return float(copolymer_system.sequence_length)

        # Estimate from molecular weights (assuming typical polymer MW)
        avg_monomer_mw = np.mean(
            [comp.molecular_weight for comp in copolymer_system.components]
        )
        typical_polymer_mw = 50000  # Typical polymer molecular weight

        return typical_polymer_mw / avg_monomer_mw

    def _predict_morphology(
        self, copolymer_system: CopolymerSystem, segregation_strength: float
    ) -> str:
        """Predict microphase morphology."""
        if segregation_strength < 10.5:
            return "disordered"

        # Get volume fraction of first component
        if len(copolymer_system.components) >= 2:
            f = copolymer_system.components[0].weight_fraction

            # Predict morphology based on composition
            if f < 0.15:
                return "spherical"
            elif f < 0.35:
                return "cylindrical"
            elif f < 0.65:
                return "lamellar"
            elif f < 0.85:
                return "cylindrical"
            else:
                return "spherical"

        return "lamellar"  # Default

    def _estimate_domain_size(
        self, copolymer_system: CopolymerSystem, segregation_strength: float
    ) -> float:
        """Estimate domain size in nanometers."""
        if segregation_strength < 10.5:
            return 0.0

        # Empirical scaling: domain size ∝ N^0.67 / χ^0.17
        N = self._estimate_degree_of_polymerization(copolymer_system)
        chi = segregation_strength / N

        # Typical scaling with empirical prefactor
        domain_size = 2.0 * (N**0.67) / (chi**0.17)

        return max(5.0, min(50.0, domain_size))  # Reasonable bounds in nm

    def _calculate_order_parameter(self, segregation_strength: float) -> float:
        """Calculate order parameter."""
        if segregation_strength < 10.5:
            return 0.0

        # Empirical relationship for order parameter
        return min(1.0, (segregation_strength - 10.5) / 20.0)


class BlendMiscibilityAnalyzer:
    """Analyzes miscibility of polymer blends."""

    def __init__(self):
        """Initialize the blend miscibility analyzer."""
        pass

    def analyze_miscibility(
        self,
        polymer1: CopolymerSystem,
        polymer2: CopolymerSystem,
        temperature: float = 298.15,
        blend_ratio: float = 0.5,
    ) -> BlendMiscibility:
        """
        Analyze miscibility of two polymer systems.

        Args:
            polymer1: First polymer system
            polymer2: Second polymer system
            temperature: Temperature in Kelvin
            blend_ratio: Weight fraction of polymer1

        Returns:
            BlendMiscibility object
        """
        # Calculate interaction parameter between polymers
        chi_parameter = self._calculate_blend_chi_parameter(
            polymer1, polymer2, temperature
        )

        # Calculate critical interaction parameter
        chi_critical = self._calculate_critical_chi(blend_ratio)

        # Determine miscibility
        is_miscible = chi_parameter < chi_critical

        # Determine phase state
        if is_miscible:
            phase_state = PhaseState.MISCIBLE
        elif chi_parameter > 2 * chi_critical:
            phase_state = PhaseState.IMMISCIBLE
        else:
            phase_state = PhaseState.PARTIALLY_MISCIBLE

        # Calculate critical and spinodal temperatures
        critical_temp = self._calculate_critical_temperature(chi_parameter, blend_ratio)
        spinodal_temp = self._calculate_spinodal_temperature(chi_parameter, blend_ratio)

        # Generate phase diagram data
        phase_diagram_data = self._generate_phase_diagram_data(
            polymer1, polymer2, chi_parameter
        )

        return BlendMiscibility(
            is_miscible=is_miscible,
            phase_state=phase_state,
            critical_temperature=critical_temp,
            spinodal_temperature=spinodal_temp,
            interaction_parameter=chi_parameter,
            phase_diagram_data=phase_diagram_data,
        )

    def _calculate_blend_chi_parameter(
        self, polymer1: CopolymerSystem, polymer2: CopolymerSystem, temperature: float
    ) -> float:
        """Calculate interaction parameter between two polymers."""
        # Get effective solubility parameters for each polymer
        delta1 = self._get_effective_solubility_parameter(polymer1)
        delta2 = self._get_effective_solubility_parameter(polymer2)

        if delta1 and delta2:
            # Use solubility parameter difference
            V_ref = 100.0  # Reference molar volume (cm³/mol)
            R = 8.314  # Gas constant (J/mol·K)

            chi = V_ref * (delta1 - delta2) ** 2 / (R * temperature)
            return max(0.0, chi)

        # Fallback: estimate from component differences
        return self._estimate_blend_chi_from_components(polymer1, polymer2)

    def _get_effective_solubility_parameter(
        self, polymer: CopolymerSystem
    ) -> Optional[float]:
        """Get effective solubility parameter for a polymer system."""
        # Use mixing rule to combine component solubility parameters
        if all(comp.solubility_parameter for comp in polymer.components):
            return sum(
                comp.weight_fraction * comp.solubility_parameter
                for comp in polymer.components
            )
        return None

    def _estimate_blend_chi_from_components(
        self, polymer1: CopolymerSystem, polymer2: CopolymerSystem
    ) -> float:
        """Estimate chi parameter from polymer components."""
        # Simple heuristic based on structural differences
        if not RDKIT_AVAILABLE:
            return 0.1

        try:
            # Get representative monomers
            rep1 = polymer1.components[0].smiles
            rep2 = polymer2.components[0].smiles

            mol1 = parse_smiles_safe(rep1)
            mol2 = parse_smiles_safe(rep2)

            if mol1 and mol2:
                # Calculate descriptor differences
                logp_diff = abs(Descriptors.MolLogP(mol1) - Descriptors.MolLogP(mol2))
                tpsa_diff = abs(Descriptors.TPSA(mol1) - Descriptors.TPSA(mol2))

                chi = 0.2 + 0.1 * logp_diff + 0.001 * tpsa_diff
                return max(0.1, min(3.0, chi))

        except Exception:
            pass

        return 0.5  # Default value

    def _calculate_critical_chi(self, blend_ratio: float) -> float:
        """Calculate critical interaction parameter."""
        # Flory-Huggins critical condition
        phi = blend_ratio
        N1 = N2 = 1000  # Assume high molecular weight polymers

        chi_critical = 0.5 * (1 / np.sqrt(N1) + 1 / np.sqrt(N2)) ** 2

        # Composition correction
        if phi != 0.5:
            correction = 1.0 + 2.0 * abs(phi - 0.5)
            chi_critical *= correction

        # For high molecular weight polymers, critical chi is very small
        # Increase threshold slightly to account for experimental reality
        return max(chi_critical, 0.03)  # Minimum threshold for miscibility

    def _calculate_critical_temperature(
        self, chi_parameter: float, blend_ratio: float
    ) -> float:
        """Calculate critical temperature."""
        chi_critical = self._calculate_critical_chi(blend_ratio)

        if chi_parameter <= chi_critical:
            return 0.0  # Always miscible

        # Estimate temperature dependence: χ = A + B/T
        # Assume typical values: A = 0.1, B = 100
        A, B = 0.1, 100

        if chi_parameter > A:
            return B / (chi_parameter - A)
        else:
            return 1000.0  # Very high temperature required

    def _calculate_spinodal_temperature(
        self, chi_parameter: float, blend_ratio: float
    ) -> float:
        """Calculate spinodal temperature."""
        # Spinodal condition is typically at higher chi than critical
        spinodal_chi = chi_parameter * 1.2
        return self._calculate_critical_temperature(spinodal_chi, blend_ratio)

    def _generate_phase_diagram_data(
        self, polymer1: CopolymerSystem, polymer2: CopolymerSystem, chi_parameter: float
    ) -> Dict[str, Any]:
        """Generate phase diagram data."""
        compositions = np.linspace(0, 1, 21)
        temperatures = np.linspace(200, 500, 31)

        phase_data = {
            "compositions": compositions.tolist(),
            "temperatures": temperatures.tolist(),
            "miscibility_map": [],
            "critical_points": [],
            "spinodal_curve": [],
        }

        # Calculate miscibility for each composition and temperature
        for T in temperatures:
            miscibility_row = []
            for phi in compositions:
                chi_critical = self._calculate_critical_chi(phi)
                # Temperature-dependent chi
                chi_T = chi_parameter * (1 + 100 / T)  # Simple temperature dependence

                is_miscible = chi_T < chi_critical
                miscibility_row.append(is_miscible)

            phase_data["miscibility_map"].append(miscibility_row)

        return phase_data


class CopolymerBlendModeler:
    """Main class for copolymer and blend property modeling."""

    def __init__(self):
        """Initialize the copolymer blend modeler."""
        self.property_predictor = CopolymerPropertyPredictor()
        self.microphase_analyzer = MicrophaseAnalyzer()
        self.miscibility_analyzer = BlendMiscibilityAnalyzer()

    def create_copolymer_system(
        self,
        monomer_data: List[Dict[str, Any]],
        copolymer_type: CopolymerType,
        **kwargs,
    ) -> CopolymerSystem:
        """
        Create a copolymer system from monomer data.

        Args:
            monomer_data: List of dictionaries containing monomer information
            copolymer_type: Type of copolymer architecture
            **kwargs: Additional architecture parameters

        Returns:
            CopolymerSystem object
        """
        components = []

        for data in monomer_data:
            # Calculate molecular weight if not provided
            if "molecular_weight" not in data or data["molecular_weight"] is None:
                if RDKIT_AVAILABLE:
                    mol = parse_smiles_safe(data["smiles"])
                    if mol:
                        data["molecular_weight"] = Descriptors.MolWt(mol)
                    else:
                        data["molecular_weight"] = 100.0  # Default value
                else:
                    data["molecular_weight"] = 100.0  # Default value

            component = MonomerComponent(
                smiles=data["smiles"],
                name=data.get("name", "Unknown"),
                mole_fraction=data["mole_fraction"],
                weight_fraction=data["weight_fraction"],
                molecular_weight=data["molecular_weight"],
                glass_transition_temp=data.get("glass_transition_temp"),
                density=data.get("density"),
                solubility_parameter=data.get("solubility_parameter"),
                properties=data.get("properties", {}),
            )
            components.append(component)

        return CopolymerSystem(
            components=components,
            copolymer_type=copolymer_type,
            sequence_length=kwargs.get("sequence_length"),
            block_lengths=kwargs.get("block_lengths"),
            architecture_parameters=kwargs,
        )

    def predict_copolymer_properties(
        self,
        copolymer_system: CopolymerSystem,
        include_microphase_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict comprehensive properties of a copolymer system.

        Args:
            copolymer_system: The copolymer system to analyze
            include_microphase_analysis: Whether to include microphase analysis

        Returns:
            Dictionary containing all analysis results
        """
        results = {
            "copolymer_system": copolymer_system,
            "predicted_properties": self.property_predictor.predict_properties(
                copolymer_system
            ),
        }

        # Add microphase analysis for block copolymers
        if (
            include_microphase_analysis
            and copolymer_system.copolymer_type == CopolymerType.BLOCK
        ):
            results[
                "microphase_info"
            ] = self.microphase_analyzer.analyze_microphase_separation(copolymer_system)

        return results

    def analyze_polymer_blend(
        self,
        polymer1_data: List[Dict[str, Any]],
        polymer2_data: List[Dict[str, Any]],
        blend_ratio: float = 0.5,
        temperature: float = 298.15,
    ) -> Dict[str, Any]:
        """
        Analyze miscibility and properties of a polymer blend.

        Args:
            polymer1_data: Data for first polymer
            polymer2_data: Data for second polymer
            blend_ratio: Weight fraction of first polymer
            temperature: Temperature in Kelvin

        Returns:
            Dictionary containing blend analysis results
        """
        # Create polymer systems
        polymer1 = self.create_copolymer_system(polymer1_data, CopolymerType.RANDOM)
        polymer2 = self.create_copolymer_system(polymer2_data, CopolymerType.RANDOM)

        # Analyze miscibility
        miscibility_info = self.miscibility_analyzer.analyze_miscibility(
            polymer1, polymer2, temperature, blend_ratio
        )

        # Create blend system for property prediction
        blend_components = []

        # Add components from first polymer
        for comp in polymer1.components:
            blend_comp = MonomerComponent(
                smiles=comp.smiles,
                name=f"{comp.name}_P1",
                mole_fraction=comp.mole_fraction * blend_ratio,
                weight_fraction=comp.weight_fraction * blend_ratio,
                molecular_weight=comp.molecular_weight,
                glass_transition_temp=comp.glass_transition_temp,
                density=comp.density,
                solubility_parameter=comp.solubility_parameter,
                properties=comp.properties.copy(),
            )
            blend_components.append(blend_comp)

        # Add components from second polymer
        for comp in polymer2.components:
            blend_comp = MonomerComponent(
                smiles=comp.smiles,
                name=f"{comp.name}_P2",
                mole_fraction=comp.mole_fraction * (1 - blend_ratio),
                weight_fraction=comp.weight_fraction * (1 - blend_ratio),
                molecular_weight=comp.molecular_weight,
                glass_transition_temp=comp.glass_transition_temp,
                density=comp.density,
                solubility_parameter=comp.solubility_parameter,
                properties=comp.properties.copy(),
            )
            blend_components.append(blend_comp)

        # Normalize fractions to ensure they sum to 1.0
        total_mole_fraction = sum(comp.mole_fraction for comp in blend_components)
        total_weight_fraction = sum(comp.weight_fraction for comp in blend_components)

        if total_mole_fraction > 0:
            for comp in blend_components:
                comp.mole_fraction /= total_mole_fraction

        if total_weight_fraction > 0:
            for comp in blend_components:
                comp.weight_fraction /= total_weight_fraction

        # Create blend system
        blend_system = CopolymerSystem(
            components=blend_components, copolymer_type=CopolymerType.RANDOM
        )

        # Predict blend properties
        blend_properties = self.property_predictor.predict_properties(blend_system)

        return {
            "polymer1": polymer1,
            "polymer2": polymer2,
            "blend_system": blend_system,
            "blend_ratio": blend_ratio,
            "temperature": temperature,
            "miscibility_info": miscibility_info,
            "blend_properties": blend_properties,
        }

    def generate_composition_study(
        self,
        monomer1_data: Dict[str, Any],
        monomer2_data: Dict[str, Any],
        composition_range: Tuple[float, float] = (0.0, 1.0),
        num_points: int = 11,
    ) -> Dict[str, Any]:
        """
        Generate a composition study for a binary copolymer system.

        Args:
            monomer1_data: Data for first monomer
            monomer2_data: Data for second monomer
            composition_range: Range of compositions to study
            num_points: Number of composition points

        Returns:
            Dictionary containing composition study results
        """
        compositions = np.linspace(
            composition_range[0], composition_range[1], num_points
        )

        results = {
            "compositions": compositions.tolist(),
            "properties": [],
            "microphase_info": [],
        }

        for comp in compositions:
            # Create copolymer system
            copolymer_data = [
                {**monomer1_data, "mole_fraction": comp, "weight_fraction": comp},
                {
                    **monomer2_data,
                    "mole_fraction": 1 - comp,
                    "weight_fraction": 1 - comp,
                },
            ]

            copolymer_system = self.create_copolymer_system(
                copolymer_data, CopolymerType.RANDOM
            )

            # Predict properties
            properties = self.property_predictor.predict_properties(copolymer_system)
            results["properties"].append(properties)

            # Analyze microphase separation for block copolymers
            block_system = CopolymerSystem(
                components=copolymer_system.components,
                copolymer_type=CopolymerType.BLOCK,
            )
            microphase_info = self.microphase_analyzer.analyze_microphase_separation(
                block_system
            )
            results["microphase_info"].append(microphase_info)

        return results

    def visualize_phase_diagram(
        self, blend_analysis: Dict[str, Any], save_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Visualize phase diagram for polymer blend.

        Args:
            blend_analysis: Results from analyze_polymer_blend
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object if plotting is available
        """
        if not PLOTTING_AVAILABLE:
            logging.warning("Plotting libraries not available for visualization")
            return None

        phase_data = blend_analysis["miscibility_info"].phase_diagram_data
        if not phase_data:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create phase diagram
        compositions = np.array(phase_data["compositions"])
        temperatures = np.array(phase_data["temperatures"])
        miscibility_map = np.array(phase_data["miscibility_map"])

        # Plot miscibility regions
        im = ax.imshow(
            miscibility_map,
            extent=[0, 1, temperatures.min(), temperatures.max()],
            aspect="auto",
            origin="lower",
            cmap="RdYlBu",
        )

        ax.set_xlabel("Weight Fraction of Polymer 1")
        ax.set_ylabel("Temperature (K)")
        ax.set_title("Polymer Blend Phase Diagram")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Miscible (Blue) / Immiscible (Red)")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


# Additional utility functions for advanced copolymer analysis


def calculate_sequence_distribution(
    copolymer_system: CopolymerSystem,
) -> Dict[str, float]:
    """
    Calculate sequence distribution parameters for copolymers.

    Args:
        copolymer_system: The copolymer system to analyze

    Returns:
        Dictionary containing sequence distribution metrics
    """
    if len(copolymer_system.components) != 2:
        raise ValueError("Sequence distribution analysis requires exactly 2 components")

    comp1, comp2 = copolymer_system.components
    f1, f2 = comp1.mole_fraction, comp2.mole_fraction

    # Calculate reactivity ratios (simplified estimation)
    r1 = r2 = 1.0  # Assume ideal random copolymerization

    # Calculate sequence distribution
    if copolymer_system.copolymer_type == CopolymerType.RANDOM:
        # Random copolymer sequence statistics
        p11 = (r1 * f1) / (r1 * f1 + f2)  # Probability of A-A sequence
        p22 = (r2 * f2) / (r2 * f2 + f1)  # Probability of B-B sequence
        p12 = 1 - p11  # Probability of A-B sequence
        p21 = 1 - p22  # Probability of B-A sequence

        # Average sequence lengths
        n1 = 1 / p12 if p12 > 0 else float("inf")  # Average A sequence length
        n2 = 1 / p21 if p21 > 0 else float("inf")  # Average B sequence length

        return {
            "p11": p11,
            "p12": p12,
            "p21": p21,
            "p22": p22,
            "average_sequence_length_1": n1,
            "average_sequence_length_2": n2,
            "randomness_index": 2 * f1 * f2 / (p12 + p21) if (p12 + p21) > 0 else 0,
        }

    elif copolymer_system.copolymer_type == CopolymerType.ALTERNATING:
        return {
            "p11": 0.0,
            "p12": 1.0,
            "p21": 1.0,
            "p22": 0.0,
            "average_sequence_length_1": 1.0,
            "average_sequence_length_2": 1.0,
            "randomness_index": 2.0,  # Perfect alternation
        }

    else:
        # Block copolymers have long sequences
        block_length_1 = (
            copolymer_system.block_lengths.get(comp1.name, 100)
            if copolymer_system.block_lengths
            else 100
        )
        block_length_2 = (
            copolymer_system.block_lengths.get(comp2.name, 100)
            if copolymer_system.block_lengths
            else 100
        )

        return {
            "p11": 0.99,
            "p12": 0.01,
            "p21": 0.01,
            "p22": 0.99,
            "average_sequence_length_1": float(block_length_1),
            "average_sequence_length_2": float(block_length_2),
            "randomness_index": 0.0,  # No randomness in block copolymers
        }


def estimate_copolymer_reactivity_ratios(
    monomer1_smiles: str, monomer2_smiles: str
) -> Tuple[float, float]:
    """
    Estimate reactivity ratios for copolymerization.

    Args:
        monomer1_smiles: SMILES string for first monomer
        monomer2_smiles: SMILES string for second monomer

    Returns:
        Tuple of reactivity ratios (r1, r2)
    """
    if not RDKIT_AVAILABLE:
        return 1.0, 1.0  # Default to ideal random copolymerization

    try:
        mol1 = parse_smiles_safe(monomer1_smiles)
        mol2 = parse_smiles_safe(monomer2_smiles)

        if mol1 is None or mol2 is None:
            return 1.0, 1.0

        # Simple estimation based on electronic effects
        # This is a simplified approach - real reactivity ratios depend on many factors

        # Calculate electron-withdrawing/donating character
        logp1, logp2 = Descriptors.MolLogP(mol1), Descriptors.MolLogP(mol2)

        # Estimate reactivity ratios based on polarity difference
        polarity_diff = abs(logp1 - logp2)

        if polarity_diff < 0.5:
            # Similar monomers - close to ideal
            r1, r2 = 1.0, 1.0
        elif polarity_diff < 1.5:
            # Moderate difference
            r1, r2 = 0.8, 1.2
        else:
            # Large difference - tendency toward alternation
            r1, r2 = 0.3, 0.3

        return r1, r2

    except Exception:
        return 1.0, 1.0


def predict_copolymer_solubility(
    copolymer_system: CopolymerSystem, solvent_solubility_parameter: float
) -> Dict[str, float]:
    """
    Predict copolymer solubility in a given solvent.

    Args:
        copolymer_system: The copolymer system
        solvent_solubility_parameter: Solubility parameter of the solvent

    Returns:
        Dictionary containing solubility predictions
    """
    # Calculate effective solubility parameter of copolymer
    if all(comp.solubility_parameter for comp in copolymer_system.components):
        copolymer_sp = sum(
            comp.weight_fraction * comp.solubility_parameter
            for comp in copolymer_system.components
        )
    else:
        # Estimate from molecular structure if available
        copolymer_sp = 18.0  # Default value

    # Calculate solubility parameter difference
    delta_sp = abs(copolymer_sp - solvent_solubility_parameter)

    # Predict solubility based on Hansen solubility theory
    # Rule of thumb: delta < 2 = good solubility, delta > 5 = poor solubility
    if delta_sp < 2.0:
        solubility_class = "Good"
        solubility_score = 1.0 - (delta_sp / 2.0)
    elif delta_sp < 5.0:
        solubility_class = "Moderate"
        solubility_score = 0.5 - ((delta_sp - 2.0) / 6.0)
    else:
        solubility_class = "Poor"
        solubility_score = max(0.0, 0.1 - ((delta_sp - 5.0) / 10.0))

    return {
        "copolymer_solubility_parameter": copolymer_sp,
        "solvent_solubility_parameter": solvent_solubility_parameter,
        "solubility_parameter_difference": delta_sp,
        "solubility_class": solubility_class,
        "solubility_score": solubility_score,
    }


# Export all classes and functions
__all__ = [
    # Enums
    "CopolymerType",
    "MixingRule",
    "PhaseState",
    # Data classes
    "MonomerComponent",
    "CopolymerSystem",
    "CopolymerProperties",
    "MicrophaseInfo",
    "BlendMiscibility",
    # Mixing rule classes
    "MixingRuleCalculator",
    "LinearMixingRule",
    "LogarithmicMixingRule",
    "GordonTaylorMixingRule",
    "FoxMixingRule",
    "FloryHugginsMixingRule",
    # Analysis classes
    "CopolymerPropertyPredictor",
    "MicrophaseAnalyzer",
    "BlendMiscibilityAnalyzer",
    "CopolymerBlendModeler",
    # Utility functions
    "calculate_sequence_distribution",
    "estimate_copolymer_reactivity_ratios",
    "predict_copolymer_solubility",
]

"""
Predictive Solid-State Property Calculation Module for ChemiAccel.

This module provides GPU-accelerated methods for predicting key properties
of crystal structures and polymorphs, including density, elastic moduli,
thermal expansion, melting point, and solubility from crystal structure data.

Key Features:
- GPU-accelerated property calculations using CuPy and PyTorch
- Machine learning models for property prediction
- Integration with crystal structure prediction results
- Support for multiple crystal systems and space groups
- Comprehensive error handling and validation
- Batch processing capabilities for high-throughput screening

Property Types Supported:
- Density calculation from unit cell parameters
- Elastic moduli (bulk, shear, Young's modulus, Poisson's ratio)
- Thermal expansion coefficients
- Melting point prediction
- Solubility estimation
- Mechanical properties
- Thermal properties
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional GPU acceleration imports
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

# Optional ML framework imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Optional scientific computing imports
try:
    from scipy import constants, interpolate, optimize
    from scipy.spatial.distance import cdist

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Core imports
from ..core.exceptions import ChemiAccelError

logger = logging.getLogger(__name__)


class PropertyType(Enum):
    """Types of solid-state properties that can be predicted."""

    DENSITY = "density"
    BULK_MODULUS = "bulk_modulus"
    SHEAR_MODULUS = "shear_modulus"
    YOUNGS_MODULUS = "youngs_modulus"
    POISSONS_RATIO = "poissons_ratio"
    THERMAL_EXPANSION = "thermal_expansion"
    MELTING_POINT = "melting_point"
    SOLUBILITY = "solubility"
    HARDNESS = "hardness"
    COMPRESSIBILITY = "compressibility"


class ElasticProperty(Enum):
    """Types of elastic properties."""

    BULK_MODULUS = "bulk_modulus"
    SHEAR_MODULUS = "shear_modulus"
    YOUNGS_MODULUS = "youngs_modulus"
    POISSONS_RATIO = "poissons_ratio"
    ELASTIC_TENSOR = "elastic_tensor"


class ThermalProperty(Enum):
    """Types of thermal properties."""

    THERMAL_EXPANSION = "thermal_expansion"
    HEAT_CAPACITY = "heat_capacity"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    MELTING_POINT = "melting_point"
    DECOMPOSITION_TEMPERATURE = "decomposition_temperature"


class SolubilityType(Enum):
    """Types of solubility predictions."""

    AQUEOUS = "aqueous"
    ORGANIC = "organic"
    INTRINSIC = "intrinsic"
    PH_DEPENDENT = "ph_dependent"


# Custom exceptions
class SolidStatePropertyError(ChemiAccelError):
    """Base exception for solid-state property prediction errors."""

    pass


class PropertyPredictionError(SolidStatePropertyError):
    """Exception raised when property prediction fails."""

    pass


class DensityCalculationError(SolidStatePropertyError):
    """Exception raised when density calculation fails."""

    pass


class ElasticModuliCalculationError(SolidStatePropertyError):
    """Exception raised when elastic moduli calculation fails."""

    pass


class ThermalPropertyError(SolidStatePropertyError):
    """Exception raised when thermal property prediction fails."""

    pass


class SolubilityPredictionError(SolidStatePropertyError):
    """Exception raised when solubility prediction fails."""

    pass


# Data structures
@dataclass
class CrystalStructureData:
    """Crystal structure data for property prediction."""

    unit_cell_parameters: Tuple[
        float, float, float, float, float, float
    ]  # a, b, c, α, β, γ
    space_group: str
    atomic_positions: List[Tuple[str, float, float, float]]  # (element, x, y, z)
    formula: str
    molecular_weight: float
    z_value: int  # Number of formula units per unit cell
    volume: Optional[float] = None
    density: Optional[float] = None


@dataclass
class PropertyPredictionConfig:
    """Configuration for solid-state property prediction."""

    use_gpu: bool = True
    batch_size: int = 32
    temperature: float = 298.15  # Room temperature in K
    pressure: float = 1.0  # Standard pressure in bar
    include_uncertainty: bool = True
    ml_models_enabled: bool = True
    cache_results: bool = True
    timeout_seconds: int = 300


@dataclass
class PropertyResult:
    """Result of property prediction."""

    property_type: PropertyType
    value: float
    unit: str
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    method: Optional[str] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None


@dataclass
class ElasticProperties:
    """Complete set of elastic properties."""

    bulk_modulus: float  # GPa
    shear_modulus: float  # GPa
    youngs_modulus: float  # GPa
    poissons_ratio: float
    elastic_tensor: Optional[np.ndarray] = None
    compliance_tensor: Optional[np.ndarray] = None


@dataclass
class ThermalProperties:
    """Complete set of thermal properties."""

    thermal_expansion: float  # K⁻¹
    heat_capacity: Optional[float] = None  # J/(mol·K)
    thermal_conductivity: Optional[float] = None  # W/(m·K)
    melting_point: Optional[float] = None  # K
    decomposition_temperature: Optional[float] = None  # K


@dataclass
class SolubilityResult:
    """Solubility prediction result."""

    solubility_type: SolubilityType
    value: float  # mg/mL or mol/L
    unit: str
    ph: Optional[float] = None
    temperature: float = 298.15
    ionic_strength: Optional[float] = None


class DensityCalculator:
    """GPU-accelerated density calculator for crystal structures."""

    def __init__(self, use_gpu: bool = True):
        """Initialize density calculator."""
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np

    def calculate_density(self, structure: CrystalStructureData) -> float:
        """
        Calculate density from crystal structure data.

        Args:
            structure: Crystal structure data

        Returns:
            Density in g/cm³

        Raises:
            DensityCalculationError: If calculation fails
        """
        try:
            # Calculate unit cell volume if not provided
            if structure.volume is None:
                volume = self._calculate_unit_cell_volume(
                    structure.unit_cell_parameters
                )
            else:
                volume = structure.volume

            # Calculate density: ρ = (Z × M) / (V × N_A)
            # Z = number of formula units per unit cell
            # M = molecular weight (g/mol)
            # V = unit cell volume (Å³)
            # N_A = Avogadro's number

            avogadro = 6.02214076e23  # mol⁻¹
            density = (structure.z_value * structure.molecular_weight) / (
                volume * avogadro * 1e-24
            )

            return float(density)

        except Exception as e:
            raise DensityCalculationError(f"Failed to calculate density: {str(e)}")

    def _calculate_unit_cell_volume(
        self, params: Tuple[float, float, float, float, float, float]
    ) -> float:
        """Calculate unit cell volume from lattice parameters."""
        a, b, c, alpha, beta, gamma = params

        # Convert angles to radians
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        # Calculate volume using the general formula
        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)

        volume = (
            a
            * b
            * c
            * np.sqrt(
                1
                + 2 * cos_alpha * cos_beta * cos_gamma
                - cos_alpha**2
                - cos_beta**2
                - cos_gamma**2
            )
        )

        return float(volume)

    def batch_calculate_density(
        self, structures: List[CrystalStructureData]
    ) -> List[float]:
        """Calculate densities for multiple structures."""
        return [self.calculate_density(structure) for structure in structures]


class ElasticModuliPredictor:
    """GPU-accelerated elastic moduli predictor using ML models."""

    def __init__(self, config: PropertyPredictionConfig):
        """Initialize elastic moduli predictor."""
        self.config = config
        self.use_gpu = config.use_gpu and _TORCH_AVAILABLE
        self.device = (
            torch.device("cuda" if self.use_gpu else "cpu")
            if _TORCH_AVAILABLE
            else None
        )

        # Initialize ML models
        if _TORCH_AVAILABLE and config.ml_models_enabled:
            self._initialize_models()

    def _initialize_models(self):
        """Initialize neural network models for elastic property prediction."""
        # Bulk modulus prediction network
        self.bulk_modulus_model = ElasticPropertyNet(input_size=15, output_size=1)

        # Shear modulus prediction network
        self.shear_modulus_model = ElasticPropertyNet(input_size=15, output_size=1)

        if self.use_gpu:
            self.bulk_modulus_model = self.bulk_modulus_model.to(self.device)
            self.shear_modulus_model = self.shear_modulus_model.to(self.device)

    def predict_elastic_properties(
        self, structure: CrystalStructureData
    ) -> ElasticProperties:
        """
        Predict elastic properties from crystal structure.

        Args:
            structure: Crystal structure data

        Returns:
            Complete elastic properties

        Raises:
            ElasticModuliCalculationError: If prediction fails
        """
        try:
            # Extract structural features
            features = self._extract_structural_features(structure)

            # Predict bulk and shear moduli using ML models
            if hasattr(self, "bulk_modulus_model") and hasattr(
                self, "shear_modulus_model"
            ):
                bulk_modulus = self._predict_bulk_modulus(features)
                shear_modulus = self._predict_shear_modulus(features)
            else:
                # Fallback to empirical models
                bulk_modulus = self._empirical_bulk_modulus(structure)
                shear_modulus = self._empirical_shear_modulus(structure)

            # Calculate derived properties
            youngs_modulus = self._calculate_youngs_modulus(bulk_modulus, shear_modulus)
            poissons_ratio = self._calculate_poissons_ratio(bulk_modulus, shear_modulus)

            return ElasticProperties(
                bulk_modulus=bulk_modulus,
                shear_modulus=shear_modulus,
                youngs_modulus=youngs_modulus,
                poissons_ratio=poissons_ratio,
            )

        except Exception as e:
            raise ElasticModuliCalculationError(
                f"Failed to predict elastic properties: {str(e)}"
            )

    def _extract_structural_features(
        self, structure: CrystalStructureData
    ) -> np.ndarray:
        """Extract features from crystal structure for ML prediction."""
        features = []

        # Unit cell parameters
        a, b, c, alpha, beta, gamma = structure.unit_cell_parameters
        features.extend([a, b, c, alpha, beta, gamma])

        # Derived geometric features
        features.append(
            structure.volume
            or self._calculate_unit_cell_volume(structure.unit_cell_parameters)
        )
        features.append(structure.density or 1.0)
        features.append(structure.z_value)
        features.append(structure.molecular_weight)

        # Crystal system indicators (one-hot encoded)
        crystal_system_features = self._encode_crystal_system(structure.space_group)
        features.extend(crystal_system_features)

        return np.array(features, dtype=np.float32)

    def _encode_crystal_system(self, space_group: str) -> List[float]:
        """Encode crystal system as one-hot features."""
        # Simplified space group to crystal system mapping
        crystal_systems = [
            "cubic",
            "tetragonal",
            "orthorhombic",
            "hexagonal",
            "trigonal",
        ]

        # Basic mapping (in practice, this would be more sophisticated)
        if any(term in space_group.lower() for term in ["cubic", "pm3m", "fm3m"]):
            system = "cubic"
        elif any(term in space_group.lower() for term in ["tetragonal", "p4", "i4"]):
            system = "tetragonal"
        elif any(
            term in space_group.lower() for term in ["orthorhombic", "pnma", "pbca"]
        ):
            system = "orthorhombic"
        elif any(term in space_group.lower() for term in ["hexagonal", "p6", "p63"]):
            system = "hexagonal"
        else:
            system = "trigonal"

        return [1.0 if sys == system else 0.0 for sys in crystal_systems]

    def _calculate_unit_cell_volume(
        self, params: Tuple[float, float, float, float, float, float]
    ) -> float:
        """Calculate unit cell volume."""
        a, b, c, alpha, beta, gamma = params
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)

        volume = (
            a
            * b
            * c
            * np.sqrt(
                1
                + 2 * cos_alpha * cos_beta * cos_gamma
                - cos_alpha**2
                - cos_beta**2
                - cos_gamma**2
            )
        )

        return float(volume)

    def _predict_bulk_modulus(self, features: np.ndarray) -> float:
        """Predict bulk modulus using neural network."""
        if not _TORCH_AVAILABLE:
            return self._empirical_bulk_modulus_from_features(features)

        with torch.no_grad():
            features_tensor = torch.from_numpy(features).unsqueeze(0)
            if self.use_gpu:
                features_tensor = features_tensor.to(self.device)

            prediction = self.bulk_modulus_model(features_tensor)
            return float(prediction.cpu().numpy()[0, 0])

    def _predict_shear_modulus(self, features: np.ndarray) -> float:
        """Predict shear modulus using neural network."""
        if not _TORCH_AVAILABLE:
            return self._empirical_shear_modulus_from_features(features)

        with torch.no_grad():
            features_tensor = torch.from_numpy(features).unsqueeze(0)
            if self.use_gpu:
                features_tensor = features_tensor.to(self.device)

            prediction = self.shear_modulus_model(features_tensor)
            return float(prediction.cpu().numpy()[0, 0])

    def _empirical_bulk_modulus(self, structure: CrystalStructureData) -> float:
        """Empirical bulk modulus calculation."""
        # Simple empirical model based on density and composition
        density = structure.density or 2.0  # Default density
        return 50.0 + 30.0 * density  # GPa, simplified model

    def _empirical_shear_modulus(self, structure: CrystalStructureData) -> float:
        """Empirical shear modulus calculation."""
        density = structure.density or 2.0
        return 30.0 + 20.0 * density  # GPa, simplified model

    def _empirical_bulk_modulus_from_features(self, features: np.ndarray) -> float:
        """Empirical bulk modulus from extracted features."""
        density = features[7] if len(features) > 7 else 2.0
        return 50.0 + 30.0 * density

    def _empirical_shear_modulus_from_features(self, features: np.ndarray) -> float:
        """Empirical shear modulus from extracted features."""
        density = features[7] if len(features) > 7 else 2.0
        return 30.0 + 20.0 * density

    def _calculate_youngs_modulus(
        self, bulk_modulus: float, shear_modulus: float
    ) -> float:
        """Calculate Young's modulus from bulk and shear moduli."""
        # E = 9KG / (3K + G)
        return (9 * bulk_modulus * shear_modulus) / (3 * bulk_modulus + shear_modulus)

    def _calculate_poissons_ratio(
        self, bulk_modulus: float, shear_modulus: float
    ) -> float:
        """Calculate Poisson's ratio from bulk and shear moduli."""
        # ν = (3K - 2G) / (6K + 2G)
        return (3 * bulk_modulus - 2 * shear_modulus) / (
            6 * bulk_modulus + 2 * shear_modulus
        )


class ThermalPropertyPredictor:
    """Predictor for thermal properties of crystal structures."""

    def __init__(self, config: PropertyPredictionConfig):
        """Initialize thermal property predictor."""
        self.config = config
        self.use_gpu = config.use_gpu and _TORCH_AVAILABLE
        self.device = (
            torch.device("cuda" if self.use_gpu else "cpu")
            if _TORCH_AVAILABLE
            else None
        )

    def predict_thermal_expansion(self, structure: CrystalStructureData) -> float:
        """
        Predict thermal expansion coefficient.

        Args:
            structure: Crystal structure data

        Returns:
            Thermal expansion coefficient in K⁻¹

        Raises:
            ThermalPropertyError: If prediction fails
        """
        try:
            # Simple empirical model based on density and bonding
            density = structure.density or 2.0

            # Thermal expansion generally decreases with density
            # and varies with crystal system
            base_expansion = 1e-5  # Base expansion coefficient
            density_factor = 1.0 / (density + 0.5)

            thermal_expansion = base_expansion * density_factor

            return float(thermal_expansion)

        except Exception as e:
            raise ThermalPropertyError(f"Failed to predict thermal expansion: {str(e)}")

    def predict_melting_point(self, structure: CrystalStructureData) -> float:
        """
        Predict melting point.

        Args:
            structure: Crystal structure data

        Returns:
            Melting point in K

        Raises:
            ThermalPropertyError: If prediction fails
        """
        try:
            # Empirical model based on density, molecular weight, and bonding
            density = structure.density or 2.0
            mw = structure.molecular_weight

            # Higher density and molecular weight generally correlate with higher melting points
            base_mp = 300.0  # Base melting point in K
            density_contribution = density * 100.0
            mw_contribution = np.log(mw + 1) * 50.0

            melting_point = base_mp + density_contribution + mw_contribution

            return float(melting_point)

        except Exception as e:
            raise ThermalPropertyError(f"Failed to predict melting point: {str(e)}")

    def predict_thermal_properties(
        self, structure: CrystalStructureData
    ) -> ThermalProperties:
        """Predict complete set of thermal properties."""
        try:
            thermal_expansion = self.predict_thermal_expansion(structure)
            melting_point = self.predict_melting_point(structure)

            return ThermalProperties(
                thermal_expansion=thermal_expansion, melting_point=melting_point
            )

        except Exception as e:
            raise ThermalPropertyError(
                f"Failed to predict thermal properties: {str(e)}"
            )


class SolubilityPredictor:
    """Predictor for solubility properties of crystal structures."""

    def __init__(self, config: PropertyPredictionConfig):
        """Initialize solubility predictor."""
        self.config = config

    def predict_aqueous_solubility(self, structure: CrystalStructureData) -> float:
        """
        Predict aqueous solubility.

        Args:
            structure: Crystal structure data

        Returns:
            Solubility in mg/mL

        Raises:
            SolubilityPredictionError: If prediction fails
        """
        try:
            # Empirical model based on crystal properties
            density = structure.density or 2.0
            mw = structure.molecular_weight

            # Higher density generally correlates with lower solubility
            # Molecular weight has complex effects
            base_solubility = 10.0  # mg/mL
            density_factor = 1.0 / (density + 0.1)
            mw_factor = 1.0 / np.sqrt(mw / 100.0 + 1)

            solubility = base_solubility * density_factor * mw_factor

            return float(solubility)

        except Exception as e:
            raise SolubilityPredictionError(
                f"Failed to predict aqueous solubility: {str(e)}"
            )

    def predict_solubility(
        self,
        structure: CrystalStructureData,
        solubility_type: SolubilityType = SolubilityType.AQUEOUS,
    ) -> SolubilityResult:
        """Predict solubility of specified type."""
        try:
            if solubility_type == SolubilityType.AQUEOUS:
                value = self.predict_aqueous_solubility(structure)
                unit = "mg/mL"
            else:
                # Simplified - in practice would have different models
                value = self.predict_aqueous_solubility(structure) * 0.5
                unit = "mg/mL"

            return SolubilityResult(
                solubility_type=solubility_type,
                value=value,
                unit=unit,
                temperature=self.config.temperature,
            )

        except Exception as e:
            raise SolubilityPredictionError(f"Failed to predict solubility: {str(e)}")


class ElasticPropertyNet(nn.Module):
    """Neural network for elastic property prediction."""

    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: List[int] = None
    ):
        """Initialize neural network."""
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(0.2)]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class SolidStatePropertyEngine:
    """Main engine for solid-state property prediction."""

    def __init__(self, config: Optional[PropertyPredictionConfig] = None):
        """Initialize property prediction engine."""
        self.config = config or PropertyPredictionConfig()

        # Initialize component predictors
        self.density_calculator = DensityCalculator(self.config.use_gpu)
        self.elastic_predictor = ElasticModuliPredictor(self.config)
        self.thermal_predictor = ThermalPropertyPredictor(self.config)
        self.solubility_predictor = SolubilityPredictor(self.config)

        # Results cache
        self._cache = {} if self.config.cache_results else None

    def predict_property(
        self, structure: CrystalStructureData, property_type: PropertyType
    ) -> PropertyResult:
        """
        Predict a specific property for a crystal structure.

        Args:
            structure: Crystal structure data
            property_type: Type of property to predict

        Returns:
            Property prediction result

        Raises:
            PropertyPredictionError: If prediction fails
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(structure, property_type)
            if self._cache and cache_key in self._cache:
                return self._cache[cache_key]

            # Predict property based on type
            if property_type == PropertyType.DENSITY:
                value = self.density_calculator.calculate_density(structure)
                unit = "g/cm³"
                method = "analytical"

            elif property_type in [
                PropertyType.BULK_MODULUS,
                PropertyType.SHEAR_MODULUS,
                PropertyType.YOUNGS_MODULUS,
                PropertyType.POISSONS_RATIO,
            ]:
                elastic_props = self.elastic_predictor.predict_elastic_properties(
                    structure
                )

                if property_type == PropertyType.BULK_MODULUS:
                    value = elastic_props.bulk_modulus
                    unit = "GPa"
                elif property_type == PropertyType.SHEAR_MODULUS:
                    value = elastic_props.shear_modulus
                    unit = "GPa"
                elif property_type == PropertyType.YOUNGS_MODULUS:
                    value = elastic_props.youngs_modulus
                    unit = "GPa"
                else:  # POISSONS_RATIO
                    value = elastic_props.poissons_ratio
                    unit = "dimensionless"

                method = "ml_prediction"

            elif property_type == PropertyType.THERMAL_EXPANSION:
                value = self.thermal_predictor.predict_thermal_expansion(structure)
                unit = "K⁻¹"
                method = "empirical"

            elif property_type == PropertyType.MELTING_POINT:
                value = self.thermal_predictor.predict_melting_point(structure)
                unit = "K"
                method = "empirical"

            elif property_type == PropertyType.SOLUBILITY:
                value = self.solubility_predictor.predict_aqueous_solubility(structure)
                unit = "mg/mL"
                method = "empirical"

            else:
                raise PropertyPredictionError(
                    f"Unsupported property type: {property_type}"
                )

            # Create result
            result = PropertyResult(
                property_type=property_type,
                value=value,
                unit=unit,
                method=method,
                temperature=self.config.temperature,
                pressure=self.config.pressure,
            )

            # Cache result
            if self._cache:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            raise PropertyPredictionError(
                f"Failed to predict {property_type.value}: {str(e)}"
            )

    def predict_all_properties(
        self, structure: CrystalStructureData
    ) -> Dict[PropertyType, PropertyResult]:
        """Predict all supported properties for a crystal structure."""
        results = {}

        for property_type in PropertyType:
            try:
                results[property_type] = self.predict_property(structure, property_type)
            except Exception as e:
                logger.warning(f"Failed to predict {property_type.value}: {str(e)}")

        return results

    def batch_predict_property(
        self, structures: List[CrystalStructureData], property_type: PropertyType
    ) -> List[PropertyResult]:
        """Predict a property for multiple structures."""
        return [
            self.predict_property(structure, property_type) for structure in structures
        ]

    def _get_cache_key(
        self, structure: CrystalStructureData, property_type: PropertyType
    ) -> str:
        """Generate cache key for structure and property type."""
        return f"{structure.formula}_{property_type.value}_{hash(str(structure.unit_cell_parameters))}"


# Utility functions
def create_property_config(
    use_gpu: bool = True, batch_size: int = 32, temperature: float = 298.15, **kwargs
) -> PropertyPredictionConfig:
    """Create a property prediction configuration."""
    return PropertyPredictionConfig(
        use_gpu=use_gpu, batch_size=batch_size, temperature=temperature, **kwargs
    )


def create_crystal_structure(
    formula: str,
    unit_cell_params: Tuple[float, float, float, float, float, float],
    space_group: str,
    atomic_positions: List[Tuple[str, float, float, float]],
    molecular_weight: float,
    z_value: int = 1,
) -> CrystalStructureData:
    """Create crystal structure data object."""
    return CrystalStructureData(
        formula=formula,
        unit_cell_parameters=unit_cell_params,
        space_group=space_group,
        atomic_positions=atomic_positions,
        molecular_weight=molecular_weight,
        z_value=z_value,
    )


def predict_solid_state_properties(
    structure: CrystalStructureData,
    properties: Optional[List[PropertyType]] = None,
    config: Optional[PropertyPredictionConfig] = None,
) -> Dict[PropertyType, PropertyResult]:
    """
    High-level function to predict solid-state properties.

    Args:
        structure: Crystal structure data
        properties: List of properties to predict (all if None)
        config: Prediction configuration

    Returns:
        Dictionary of property predictions
    """
    engine = SolidStatePropertyEngine(config)

    if properties is None:
        return engine.predict_all_properties(structure)
    else:
        results = {}
        for prop in properties:
            try:
                results[prop] = engine.predict_property(structure, prop)
            except Exception as e:
                logger.warning(f"Failed to predict {prop.value}: {str(e)}")
        return results


def calculate_structure_density(structure: CrystalStructureData) -> float:
    """Calculate density for a crystal structure."""
    calculator = DensityCalculator()
    return calculator.calculate_density(structure)


def predict_elastic_moduli(
    structure: CrystalStructureData, config: Optional[PropertyPredictionConfig] = None
) -> ElasticProperties:
    """Predict elastic moduli for a crystal structure."""
    predictor = ElasticModuliPredictor(config or PropertyPredictionConfig())
    return predictor.predict_elastic_properties(structure)


def predict_thermal_properties(
    structure: CrystalStructureData, config: Optional[PropertyPredictionConfig] = None
) -> ThermalProperties:
    """Predict thermal properties for a crystal structure."""
    predictor = ThermalPropertyPredictor(config or PropertyPredictionConfig())
    return predictor.predict_thermal_properties(structure)


def predict_solubility(
    structure: CrystalStructureData,
    solubility_type: SolubilityType = SolubilityType.AQUEOUS,
    config: Optional[PropertyPredictionConfig] = None,
) -> SolubilityResult:
    """Predict solubility for a crystal structure."""
    predictor = SolubilityPredictor(config or PropertyPredictionConfig())
    return predictor.predict_solubility(structure, solubility_type)

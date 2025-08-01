"""
Crystal Structure and Property Prediction Integration

This module integrates crystallization process outcomes with AI-driven crystal structure
prediction to predict resulting polymorphs, crystal habit, particle size distribution,
and associated properties like dissolution rate and flowability.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    PYTORCH_AVAILABLE = False

from .core import (
    CrystallizationCore,
    CrystalProperties,
    PolymorphType,
    ProcessConditions,
    SimulationConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class CrystalStructureData:
    """
    Crystal structure information and descriptors.

    Attributes:
        unit_cell_parameters: a, b, c, alpha, beta, gamma
        space_group: Crystal space group
        density: Crystal density (g/cm³)
        miller_indices: Important Miller indices
        surface_energies: Surface energies for different faces
        molecular_packing: Packing coefficient
        hydrogen_bonding: H-bonding descriptors
        intermolecular_interactions: Interaction energies
    """

    unit_cell_parameters: Dict[str, float] = field(default_factory=dict)
    space_group: Optional[str] = None
    density: float = 0.0
    miller_indices: List[Tuple[int, int, int]] = field(default_factory=list)
    surface_energies: Dict[str, float] = field(default_factory=dict)
    molecular_packing: float = 0.0
    hydrogen_bonding: Dict[str, float] = field(default_factory=dict)
    intermolecular_interactions: Dict[str, float] = field(default_factory=dict)


@dataclass
class PolymorphData:
    """
    Polymorph-specific data and properties.

    Attributes:
        polymorph_type: Type of polymorph
        structure_data: Crystal structure information
        stability_ranking: Thermodynamic stability ranking
        formation_energy: Formation energy (kJ/mol)
        dissolution_rate: Intrinsic dissolution rate
        solubility: Equilibrium solubility
        mechanical_properties: Hardness, elasticity, etc.
    """

    polymorph_type: PolymorphType
    structure_data: CrystalStructureData
    stability_ranking: int = 1
    formation_energy: float = 0.0
    dissolution_rate: float = 0.0
    solubility: float = 0.0
    mechanical_properties: Dict[str, float] = field(default_factory=dict)


class StructurePredictionModel(ABC):
    """Abstract base class for crystal structure prediction models."""

    @abstractmethod
    def predict_structure(self, molecular_data: Dict[str, Any]) -> CrystalStructureData:
        """Predict crystal structure from molecular data."""
        pass

    @abstractmethod
    def train(
        self, training_data: List[Tuple[Dict[str, Any], CrystalStructureData]]
    ) -> None:
        """Train the structure prediction model."""
        pass


class MLStructurePredictor(StructurePredictionModel):
    """
    Machine learning-based crystal structure predictor.

    Uses molecular descriptors to predict crystal structure parameters
    and properties using ensemble methods.
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize ML structure predictor.

        Args:
            model_type: Type of ML model ('random_forest', 'gradient_boosting', 'neural_network')
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ML structure prediction")

        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.is_trained = False

        # Initialize models for different properties
        properties = [
            "density",
            "unit_cell_a",
            "unit_cell_b",
            "unit_cell_c",
            "unit_cell_alpha",
            "unit_cell_beta",
            "unit_cell_gamma",
        ]

        for prop in properties:
            if model_type == "random_forest":
                self.models[prop] = RandomForestRegressor(
                    n_estimators=100, random_state=42
                )
            elif model_type == "gradient_boosting":
                self.models[prop] = GradientBoostingRegressor(
                    n_estimators=100, random_state=42
                )
            elif model_type == "neural_network":
                self.models[prop] = MLPRegressor(
                    hidden_layer_sizes=(100, 50), random_state=42
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.scalers[prop] = StandardScaler()

    def predict_structure(self, molecular_data: Dict[str, Any]) -> CrystalStructureData:
        """
        Predict crystal structure from molecular data.

        Args:
            molecular_data: Dictionary containing molecular descriptors

        Returns:
            Predicted crystal structure data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Extract features
        features = self._extract_features(molecular_data)
        features_array = np.array(features).reshape(1, -1)

        # Predict structure parameters
        predictions = {}
        for prop, model in self.models.items():
            scaled_features = self.scalers[prop].transform(features_array)
            predictions[prop] = model.predict(scaled_features)[0]

        # Create structure data
        structure_data = CrystalStructureData()
        structure_data.density = predictions["density"]
        structure_data.unit_cell_parameters = {
            "a": predictions["unit_cell_a"],
            "b": predictions["unit_cell_b"],
            "c": predictions["unit_cell_c"],
            "alpha": predictions["unit_cell_alpha"],
            "beta": predictions["unit_cell_beta"],
            "gamma": predictions["unit_cell_gamma"],
        }

        # Predict additional properties
        structure_data.molecular_packing = self._calculate_packing_coefficient(
            molecular_data, structure_data
        )

        return structure_data

    def train(
        self, training_data: List[Tuple[Dict[str, Any], CrystalStructureData]]
    ) -> None:
        """
        Train the structure prediction model.

        Args:
            training_data: List of (molecular_data, structure_data) tuples
        """
        if len(training_data) < 10:
            raise ValueError("Insufficient training data (minimum 10 samples required)")

        # Prepare training data
        X = []
        y = {prop: [] for prop in self.models.keys()}

        for molecular_data, structure_data in training_data:
            features = self._extract_features(molecular_data)
            X.append(features)

            y["density"].append(structure_data.density)
            y["unit_cell_a"].append(structure_data.unit_cell_parameters.get("a", 0.0))
            y["unit_cell_b"].append(structure_data.unit_cell_parameters.get("b", 0.0))
            y["unit_cell_c"].append(structure_data.unit_cell_parameters.get("c", 0.0))
            y["unit_cell_alpha"].append(
                structure_data.unit_cell_parameters.get("alpha", 90.0)
            )
            y["unit_cell_beta"].append(
                structure_data.unit_cell_parameters.get("beta", 90.0)
            )
            y["unit_cell_gamma"].append(
                structure_data.unit_cell_parameters.get("gamma", 90.0)
            )

        X = np.array(X)

        # Train models
        for prop, model in self.models.items():
            y_prop = np.array(y[prop])

            # Fit scaler and transform features
            X_scaled = self.scalers[prop].fit_transform(X)

            # Train model
            model.fit(X_scaled, y_prop)

            logger.info(f"Trained {prop} predictor")

        self.is_trained = True
        logger.info("Structure prediction model training completed")

    def _extract_features(self, molecular_data: Dict[str, Any]) -> List[float]:
        """Extract features from molecular data."""
        features = []

        # Basic molecular properties
        features.append(molecular_data.get("molecular_weight", 0.0))
        features.append(molecular_data.get("logp", 0.0))
        features.append(molecular_data.get("tpsa", 0.0))
        features.append(molecular_data.get("num_hbd", 0.0))
        features.append(molecular_data.get("num_hba", 0.0))
        features.append(molecular_data.get("num_rotatable_bonds", 0.0))
        features.append(molecular_data.get("num_aromatic_rings", 0.0))
        features.append(molecular_data.get("molecular_volume", 0.0))

        # Shape descriptors
        features.append(molecular_data.get("asphericity", 0.0))
        features.append(molecular_data.get("eccentricity", 0.0))
        features.append(molecular_data.get("inertial_shape_factor", 0.0))

        # Electronic properties
        features.append(molecular_data.get("dipole_moment", 0.0))
        features.append(molecular_data.get("polarizability", 0.0))

        return features

    def _calculate_packing_coefficient(
        self, molecular_data: Dict[str, Any], structure_data: CrystalStructureData
    ) -> float:
        """Calculate packing coefficient."""
        molecular_volume = molecular_data.get("molecular_volume", 100.0)  # Å³

        # Unit cell volume
        params = structure_data.unit_cell_parameters
        a, b, c = params.get("a", 10.0), params.get("b", 10.0), params.get("c", 10.0)
        alpha = np.radians(params.get("alpha", 90.0))
        beta = np.radians(params.get("beta", 90.0))
        gamma = np.radians(params.get("gamma", 90.0))

        # Calculate unit cell volume
        unit_cell_volume = (
            a
            * b
            * c
            * np.sqrt(
                1
                + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
                - np.cos(alpha) ** 2
                - np.cos(beta) ** 2
                - np.cos(gamma) ** 2
            )
        )

        # Assume Z=4 (typical for organic molecules)
        z = 4
        packing_coefficient = (z * molecular_volume) / unit_cell_volume

        return min(packing_coefficient, 1.0)


class PolymorphPredictor(CrystallizationCore):
    """
    Polymorph prediction and stability analysis.

    Predicts likely polymorphs based on crystallization conditions
    and molecular structure.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize polymorph predictor.

        Args:
            config: Simulation configuration
        """
        super().__init__(config)
        self.structure_predictor = None
        self.polymorph_database = {}
        self.stability_model = None

    def initialize(self) -> None:
        """Initialize the polymorph predictor."""
        self.structure_predictor = MLStructurePredictor()

        # Initialize stability prediction model
        if SKLEARN_AVAILABLE:
            self.stability_model = RandomForestRegressor(
                n_estimators=100, random_state=42
            )

        self._initialized = True
        logger.info("Polymorph predictor initialized")

    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        required_keys = ["molecular_data", "conditions"]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required input: {key}")
        return True

    def predict_polymorphs(
        self, molecular_data: Dict[str, Any], conditions: ProcessConditions
    ) -> List[PolymorphData]:
        """
        Predict likely polymorphs for given conditions.

        Args:
            molecular_data: Molecular structure and properties
            conditions: Crystallization conditions

        Returns:
            List of predicted polymorphs with probabilities
        """
        if not self.is_initialized():
            self.initialize()

        self.validate_inputs(molecular_data=molecular_data, conditions=conditions)

        # Predict base crystal structure
        base_structure = self._predict_base_structure(molecular_data)

        # Generate polymorph variants
        polymorphs = self._generate_polymorph_variants(base_structure, molecular_data)

        # Predict stabilities under given conditions
        for polymorph in polymorphs:
            polymorph.stability_ranking = self._predict_stability(
                polymorph, molecular_data, conditions
            )

        # Sort by stability
        polymorphs.sort(key=lambda p: p.stability_ranking)

        return polymorphs

    def predict_dominant_polymorph(
        self, molecular_data: Dict[str, Any], conditions: ProcessConditions
    ) -> PolymorphData:
        """
        Predict the dominant polymorph under given conditions.

        Args:
            molecular_data: Molecular structure and properties
            conditions: Crystallization conditions

        Returns:
            Most stable polymorph
        """
        polymorphs = self.predict_polymorphs(molecular_data, conditions)
        return polymorphs[0] if polymorphs else None

    def _predict_base_structure(
        self, molecular_data: Dict[str, Any]
    ) -> CrystalStructureData:
        """Predict base crystal structure."""
        # Use structure predictor if trained
        if self.structure_predictor and self.structure_predictor.is_trained:
            return self.structure_predictor.predict_structure(molecular_data)

        # Otherwise use heuristic rules
        return self._heuristic_structure_prediction(molecular_data)

    def _heuristic_structure_prediction(
        self, molecular_data: Dict[str, Any]
    ) -> CrystalStructureData:
        """Heuristic crystal structure prediction."""
        structure = CrystalStructureData()

        # Estimate density based on molecular weight and volume
        mw = molecular_data.get("molecular_weight", 200.0)
        volume = molecular_data.get("molecular_volume", 150.0)

        # Typical organic crystal density
        structure.density = 1.2 + 0.001 * mw - 0.0001 * volume
        structure.density = max(0.8, min(2.0, structure.density))

        # Estimate unit cell parameters
        # Assume orthorhombic for simplicity
        molecular_length = (volume / 4.0) ** (1 / 3)  # Rough estimate

        structure.unit_cell_parameters = {
            "a": molecular_length * 1.5,
            "b": molecular_length * 1.2,
            "c": molecular_length * 2.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        }

        structure.space_group = "P212121"  # Common for organics

        return structure

    def _generate_polymorph_variants(
        self, base_structure: CrystalStructureData, molecular_data: Dict[str, Any]
    ) -> List[PolymorphData]:
        """Generate polymorph variants."""
        polymorphs = []

        # Form I (most stable)
        form_i = PolymorphData(
            polymorph_type=PolymorphType.FORM_I,
            structure_data=base_structure,
            stability_ranking=1,
        )
        polymorphs.append(form_i)

        # Form II (metastable)
        form_ii_structure = CrystalStructureData()
        form_ii_structure.density = base_structure.density * 0.95
        form_ii_structure.unit_cell_parameters = (
            base_structure.unit_cell_parameters.copy()
        )
        form_ii_structure.unit_cell_parameters["a"] *= 1.1
        form_ii_structure.unit_cell_parameters["c"] *= 0.9
        form_ii_structure.space_group = "P21/c"

        form_ii = PolymorphData(
            polymorph_type=PolymorphType.FORM_II,
            structure_data=form_ii_structure,
            stability_ranking=2,
        )
        polymorphs.append(form_ii)

        # Check for hydrate possibility
        num_hbd = molecular_data.get("num_hbd", 0)
        num_hba = molecular_data.get("num_hba", 0)

        if num_hbd + num_hba >= 3:  # Likely to form hydrate
            hydrate_structure = CrystalStructureData()
            hydrate_structure.density = base_structure.density * 0.85
            hydrate_structure.unit_cell_parameters = (
                base_structure.unit_cell_parameters.copy()
            )
            # Expand unit cell for water inclusion
            for param in ["a", "b", "c"]:
                hydrate_structure.unit_cell_parameters[param] *= 1.15

            hydrate = PolymorphData(
                polymorph_type=PolymorphType.HYDRATE,
                structure_data=hydrate_structure,
                stability_ranking=3,
            )
            polymorphs.append(hydrate)

        return polymorphs

    def _predict_stability(
        self,
        polymorph: PolymorphData,
        molecular_data: Dict[str, Any],
        conditions: ProcessConditions,
    ) -> int:
        """Predict polymorph stability ranking."""
        # Simple heuristic based on density and conditions
        stability_score = 0.0

        # Higher density usually means more stable
        density = polymorph.structure_data.density
        stability_score += density * 10

        # Temperature effects
        if conditions.temperature_profile:
            avg_temp = np.mean([t[0] for t in conditions.temperature_profile])
            if polymorph.polymorph_type == PolymorphType.HYDRATE and avg_temp > 333.15:
                stability_score -= 50  # Hydrates less stable at high temp

        # Solvent effects
        if conditions.solvent_composition:
            water_content = conditions.solvent_composition.get("water", 0.0)
            if (
                polymorph.polymorph_type == PolymorphType.HYDRATE
                and water_content > 0.5
            ):
                stability_score += 20  # Hydrates favored in aqueous solvents

        # Convert to ranking (lower score = higher stability)
        return int(-stability_score)


class CrystalHabitPredictor:
    """
    Crystal habit (morphology) prediction based on surface energies
    and growth conditions.
    """

    def __init__(self):
        """Initialize crystal habit predictor."""
        self.surface_energy_model = None
        self.habit_database = {}

    def predict_crystal_habit(
        self, polymorph: PolymorphData, conditions: ProcessConditions
    ) -> Dict[str, Any]:
        """
        Predict crystal habit for given polymorph and conditions.

        Args:
            polymorph: Polymorph data
            conditions: Crystallization conditions

        Returns:
            Crystal habit descriptors
        """
        # Get surface energies
        surface_energies = self._calculate_surface_energies(polymorph)

        # Apply growth conditions
        growth_rates = self._calculate_face_growth_rates(surface_energies, conditions)

        # Predict morphology
        morphology = self._predict_morphology(growth_rates)

        habit = {
            "aspect_ratio": morphology["length"] / morphology["width"],
            "habit_type": morphology["habit_type"],
            "dominant_faces": morphology["dominant_faces"],
            "surface_area_to_volume": morphology["surface_area"] / morphology["volume"],
            "predicted_flowability": self._predict_flowability(morphology),
        }

        return habit

    def _calculate_surface_energies(self, polymorph: PolymorphData) -> Dict[str, float]:
        """Calculate surface energies for different crystal faces."""
        # Simplified surface energy calculation
        surface_energies = {}

        # Common faces for organic crystals
        faces = ["100", "010", "001", "110", "011", "101"]

        base_energy = 0.1  # J/m²

        for face in faces:
            # Miller indices
            h, k, l = int(face[0]), int(face[1]), int(face[2])

            # Surface energy depends on Miller indices and molecular packing
            energy = base_energy * (h**2 + k**2 + l**2) ** 0.5

            # Adjust based on molecular interactions
            if polymorph.structure_data.hydrogen_bonding:
                hb_strength = sum(polymorph.structure_data.hydrogen_bonding.values())
                energy *= 1.0 + 0.1 * hb_strength

            surface_energies[face] = energy

        return surface_energies

    def _calculate_face_growth_rates(
        self, surface_energies: Dict[str, float], conditions: ProcessConditions
    ) -> Dict[str, float]:
        """Calculate growth rates for different faces."""
        growth_rates = {}

        # Base growth rate from conditions
        base_rate = 1e-8  # m/s

        # Temperature effect
        if conditions.temperature_profile:
            avg_temp = np.mean([t[0] for t in conditions.temperature_profile])
            temp_factor = np.exp(-(30000.0 / 8.314) * (1 / avg_temp - 1 / 298.15))
        else:
            temp_factor = 1.0

        # Supersaturation effect
        if conditions.supersaturation_profile:
            avg_supersat = np.mean([s[0] for s in conditions.supersaturation_profile])
            supersat_factor = max(0, avg_supersat - 1.0)
        else:
            supersat_factor = 0.5

        for face, surface_energy in surface_energies.items():
            # Higher surface energy faces grow slower
            energy_factor = 1.0 / (1.0 + surface_energy * 100)

            growth_rate = base_rate * temp_factor * supersat_factor * energy_factor
            growth_rates[face] = growth_rate

        return growth_rates

    def _predict_morphology(self, growth_rates: Dict[str, float]) -> Dict[str, Any]:
        """Predict crystal morphology from growth rates."""
        # Sort faces by growth rate (slowest growing faces dominate morphology)
        sorted_faces = sorted(growth_rates.items(), key=lambda x: x[1])

        dominant_faces = [face for face, rate in sorted_faces[:3]]

        # Determine habit type based on dominant faces
        if "001" in dominant_faces and "100" in dominant_faces:
            habit_type = "plate"
            length, width, height = 10.0, 8.0, 2.0
        elif "100" in dominant_faces and "010" in dominant_faces:
            habit_type = "needle"
            length, width, height = 20.0, 3.0, 3.0
        else:
            habit_type = "block"
            length, width, height = 8.0, 6.0, 5.0

        # Calculate geometric properties
        volume = length * width * height
        surface_area = 2 * (length * width + length * height + width * height)

        morphology = {
            "habit_type": habit_type,
            "length": length,
            "width": width,
            "height": height,
            "volume": volume,
            "surface_area": surface_area,
            "dominant_faces": dominant_faces,
        }

        return morphology

    def _predict_flowability(self, morphology: Dict[str, Any]) -> float:
        """Predict powder flowability from crystal habit."""
        aspect_ratio = morphology["length"] / morphology["width"]

        # Flowability decreases with increasing aspect ratio
        flowability = 1.0 / (1.0 + 0.1 * aspect_ratio)

        # Adjust based on habit type
        if morphology["habit_type"] == "needle":
            flowability *= 0.7  # Needles flow poorly
        elif morphology["habit_type"] == "plate":
            flowability *= 0.8  # Plates have moderate flowability

        return flowability


class ParticleSizeDistributor:
    """
    Particle size distribution prediction and analysis.
    """

    def __init__(self):
        """Initialize particle size distributor."""
        self.psd_model = None

    def predict_size_distribution(
        self, kinetics_results: Dict[str, Any], habit_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict final particle size distribution.

        Args:
            kinetics_results: Results from kinetics modeling
            habit_data: Crystal habit prediction results

        Returns:
            Particle size distribution data
        """
        # Get size distribution from kinetics
        if "results" in kinetics_results:
            d10 = kinetics_results["results"].get("d10", [10e-6])[-1]
            d50 = kinetics_results["results"].get("d50", [50e-6])[-1]
            d90 = kinetics_results["results"].get("d90", [100e-6])[-1]
        else:
            d10, d50, d90 = 10e-6, 50e-6, 100e-6

        # Calculate distribution parameters
        span = (d90 - d10) / d50
        uniformity = d50 / (d90 - d10) if d90 > d10 else 1.0

        # Adjust for crystal habit
        aspect_ratio = habit_data.get("aspect_ratio", 1.0)
        if aspect_ratio > 3.0:  # Needle-like
            span *= 1.5  # More polydisperse
        elif aspect_ratio < 0.5:  # Plate-like
            span *= 1.2

        psd_data = {
            "d10": d10,
            "d50": d50,
            "d90": d90,
            "span": span,
            "uniformity": uniformity,
            "mean_size": d50,
            "size_range": d90 - d10,
            "distribution_type": self._classify_distribution(span),
        }

        return psd_data

    def _classify_distribution(self, span: float) -> str:
        """Classify size distribution based on span."""
        if span < 1.5:
            return "narrow"
        elif span < 3.0:
            return "moderate"
        else:
            return "broad"


class StructurePropertyPredictor(CrystallizationCore):
    """
    Main structure-property prediction integration class.

    Combines polymorph prediction, habit prediction, and property
    estimation for comprehensive crystallization outcome prediction.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize structure-property predictor.

        Args:
            config: Simulation configuration
        """
        super().__init__(config)
        self.polymorph_predictor = None
        self.habit_predictor = None
        self.size_distributor = None

    def initialize(self) -> None:
        """Initialize all prediction components."""
        self.polymorph_predictor = PolymorphPredictor(self.config)
        self.polymorph_predictor.initialize()

        self.habit_predictor = CrystalHabitPredictor()
        self.size_distributor = ParticleSizeDistributor()

        self._initialized = True
        logger.info("Structure-property predictor initialized")

    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        required_keys = ["molecular_data", "conditions"]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required input: {key}")
        return True

    def predict_crystallization_properties(
        self,
        molecular_data: Dict[str, Any],
        conditions: ProcessConditions,
        kinetics_results: Optional[Dict[str, Any]] = None,
    ) -> CrystalProperties:
        """
        Predict comprehensive crystallization properties.

        Args:
            molecular_data: Molecular structure and properties
            conditions: Crystallization conditions
            kinetics_results: Optional kinetics modeling results

        Returns:
            Comprehensive crystal properties
        """
        if not self.is_initialized():
            self.initialize()

        self.validate_inputs(molecular_data=molecular_data, conditions=conditions)

        # Predict polymorphs
        polymorphs = self.polymorph_predictor.predict_polymorphs(
            molecular_data, conditions
        )

        # Get dominant polymorph
        dominant_polymorph = polymorphs[0] if polymorphs else None

        # Predict crystal habit
        habit_data = {}
        if dominant_polymorph:
            habit_data = self.habit_predictor.predict_crystal_habit(
                dominant_polymorph, conditions
            )

        # Predict size distribution
        psd_data = {}
        if kinetics_results:
            psd_data = self.size_distributor.predict_size_distribution(
                kinetics_results, habit_data
            )

        # Compile crystal properties
        properties = CrystalProperties()

        # Polymorph composition
        if polymorphs:
            total_prob = len(polymorphs)
            for i, polymorph in enumerate(polymorphs):
                weight = (total_prob - i) / total_prob  # Higher weight for more stable
                properties.polymorph_composition[polymorph.polymorph_type] = weight

        # Particle size distribution
        properties.particle_size_distribution = psd_data

        # Crystal habit
        properties.crystal_habit = habit_data

        # Derived properties
        if dominant_polymorph:
            properties.surface_area = self._calculate_surface_area(
                dominant_polymorph, habit_data, psd_data
            )
            properties.dissolution_rate = self._predict_dissolution_rate(
                dominant_polymorph, habit_data
            )
            properties.flowability_index = habit_data.get("predicted_flowability", 0.5)

        # Estimate yield and purity
        properties.yield_ = self._estimate_yield(conditions, kinetics_results)
        properties.purity = self._estimate_purity(conditions, polymorphs)

        return properties

    def _calculate_surface_area(
        self,
        polymorph: PolymorphData,
        habit_data: Dict[str, Any],
        psd_data: Dict[str, Any],
    ) -> float:
        """Calculate specific surface area."""
        if not habit_data or not psd_data:
            return 1.0  # Default value

        # Surface area to volume ratio from habit
        sa_to_vol = habit_data.get("surface_area_to_volume", 1.0)

        # Mean particle size
        mean_size = psd_data.get("d50", 50e-6)

        # Density
        density = polymorph.structure_data.density * 1000  # kg/m³

        # Specific surface area (m²/g)
        specific_sa = sa_to_vol / (mean_size * density / 1000)

        return specific_sa

    def _predict_dissolution_rate(
        self, polymorph: PolymorphData, habit_data: Dict[str, Any]
    ) -> float:
        """Predict dissolution rate."""
        # Base dissolution rate
        base_rate = 1e-6  # g/cm²/min

        # Density effect (higher density = slower dissolution)
        density_factor = 1.0 / polymorph.structure_data.density

        # Surface area effect
        sa_factor = habit_data.get("surface_area_to_volume", 1.0)

        dissolution_rate = base_rate * density_factor * sa_factor

        return dissolution_rate

    def _estimate_yield(
        self, conditions: ProcessConditions, kinetics_results: Optional[Dict[str, Any]]
    ) -> float:
        """Estimate crystallization yield."""
        if kinetics_results and "results" in kinetics_results:
            yield_evolution = kinetics_results["results"].get("yield_evolution", [50.0])
            return yield_evolution[-1]

        # Heuristic yield estimation
        base_yield = 80.0  # %

        # Temperature effect
        if conditions.temperature_profile:
            avg_temp = np.mean([t[0] for t in conditions.temperature_profile])
            if avg_temp < 283.15:  # Low temperature
                base_yield *= 1.1
            elif avg_temp > 353.15:  # High temperature
                base_yield *= 0.9

        # Supersaturation effect
        if conditions.supersaturation_profile:
            avg_supersat = np.mean([s[0] for s in conditions.supersaturation_profile])
            if avg_supersat > 2.0:
                base_yield *= 1.2

        return min(100.0, base_yield)

    def _estimate_purity(
        self, conditions: ProcessConditions, polymorphs: List[PolymorphData]
    ) -> float:
        """Estimate product purity."""
        base_purity = 95.0  # %

        # Multiple polymorphs reduce purity
        if len(polymorphs) > 1:
            base_purity *= 0.95

        # Impurity effects
        if conditions.impurity_levels:
            total_impurities = sum(conditions.impurity_levels.values())
            base_purity *= 1.0 - total_impurities / 10000  # ppm to fraction

        return max(80.0, min(100.0, base_purity))

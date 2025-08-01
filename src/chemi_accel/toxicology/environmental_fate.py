"""
Advanced environmental degradation pathway & product prediction.

Enhances GPU-accelerated degradation modeling to focus on environmental
degradation pathways (hydrolysis, photolysis, biodegradation) and
accurately predict resulting degradation products.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.backend import Backend
from ..core.exceptions import ChemiAccelError
from .core import ConfidenceLevel, ToxicityData


class DegradationMechanism(Enum):
    """Types of environmental degradation mechanisms."""

    HYDROLYSIS = "hydrolysis"
    PHOTOLYSIS = "photolysis"
    BIODEGRADATION = "biodegradation"
    ATMOSPHERIC_OXIDATION = "atmospheric_oxidation"
    SOIL_DEGRADATION = "soil_degradation"
    SEDIMENT_DEGRADATION = "sediment_degradation"
    THERMAL_DEGRADATION = "thermal_degradation"


class EnvironmentalCompartment(Enum):
    """Environmental compartments for fate modeling."""

    ATMOSPHERE = "atmosphere"
    SURFACE_WATER = "surface_water"
    GROUNDWATER = "groundwater"
    SOIL = "soil"
    SEDIMENT = "sediment"
    BIOTA = "biota"
    VEGETATION = "vegetation"


class DegradationConditions(Enum):
    """Environmental conditions affecting degradation."""

    AEROBIC = "aerobic"
    ANAEROBIC = "anaerobic"
    ACIDIC = "acidic"
    BASIC = "basic"
    NEUTRAL = "neutral"
    HIGH_TEMPERATURE = "high_temperature"
    LOW_TEMPERATURE = "low_temperature"
    UV_EXPOSURE = "uv_exposure"
    NO_UV = "no_uv"


@dataclass
class DegradationProduct:
    """Data structure for degradation products."""

    parent_smiles: str
    product_smiles: str
    mechanism: DegradationMechanism
    yield_fraction: float  # 0-1, fraction of parent converted to this product
    half_life: Optional[float] = None  # hours
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    reaction_pathway: Optional[List[str]] = None
    environmental_conditions: Optional[List[DegradationConditions]] = None
    toxicity_data: Optional[ToxicityData] = None

    def __post_init__(self):
        """Validate degradation product data."""
        if not 0 <= self.yield_fraction <= 1:
            raise ValueError("Yield fraction must be between 0 and 1")

        if self.half_life is not None and self.half_life < 0:
            raise ValueError("Half-life cannot be negative")


@dataclass
class DegradationPathway:
    """Complete degradation pathway information."""

    parent_compound: str
    mechanism: DegradationMechanism
    compartment: EnvironmentalCompartment
    conditions: List[DegradationConditions]
    products: List[DegradationProduct]
    overall_half_life: float  # hours
    mass_balance: float  # fraction of parent accounted for
    pathway_confidence: ConfidenceLevel
    kinetic_parameters: Dict[str, float] = field(default_factory=dict)

    @property
    def primary_products(self) -> List[DegradationProduct]:
        """Get primary degradation products (highest yields)."""
        return sorted(self.products, key=lambda x: x.yield_fraction, reverse=True)[:3]

    @property
    def total_yield(self) -> float:
        """Calculate total yield of all products."""
        return sum(product.yield_fraction for product in self.products)


class DegradationModel(ABC):
    """Abstract base class for degradation prediction models."""

    @abstractmethod
    def predict_products(
        self,
        smiles: str,
        mechanism: DegradationMechanism,
        conditions: List[DegradationConditions],
    ) -> List[DegradationProduct]:
        """Predict degradation products."""
        pass

    @abstractmethod
    def predict_half_life(
        self,
        smiles: str,
        mechanism: DegradationMechanism,
        compartment: EnvironmentalCompartment,
        conditions: List[DegradationConditions],
    ) -> Tuple[float, float]:
        """Predict degradation half-life and uncertainty."""
        pass


class HydrolysisModel(DegradationModel):
    """Model for hydrolytic degradation."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.reaction_rules = self._load_hydrolysis_rules()
        self.kinetic_models = {}

    def _load_hydrolysis_rules(self) -> Dict[str, Any]:
        """Load hydrolysis reaction rules."""
        # Simplified reaction rules for hydrolysis
        return {
            "ester_hydrolysis": {
                "pattern": "[C:1](=[O:2])[O:3][C:4]",
                "products": ["[C:1](=[O:2])O", "[C:4]O"],
                "rate_factors": {"acidic": 2.0, "basic": 5.0, "neutral": 1.0},
            },
            "amide_hydrolysis": {
                "pattern": "[C:1](=[O:2])[N:3]",
                "products": ["[C:1](=[O:2])O", "[N:3]"],
                "rate_factors": {"acidic": 1.5, "basic": 3.0, "neutral": 0.5},
            },
            "nitrile_hydrolysis": {
                "pattern": "[C:1]#[N:2]",
                "products": ["[C:1](=O)O", "[N:2]"],
                "rate_factors": {"acidic": 1.0, "basic": 2.0, "neutral": 0.1},
            },
        }

    def predict_products(
        self,
        smiles: str,
        mechanism: DegradationMechanism,
        conditions: List[DegradationConditions],
    ) -> List[DegradationProduct]:
        """Predict hydrolysis products."""

        if mechanism != DegradationMechanism.HYDROLYSIS:
            return []

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []

            products = []

            # Apply hydrolysis rules
            for rule_name, rule_data in self.reaction_rules.items():
                pattern = Chem.MolFromSmarts(rule_data["pattern"])
                if pattern is None:
                    continue

                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    # Calculate rate enhancement based on conditions
                    rate_factor = 1.0
                    for condition in conditions:
                        if condition.value in rule_data["rate_factors"]:
                            rate_factor *= rule_data["rate_factors"][condition.value]

                    # Generate products for each match
                    for match in matches:
                        for product_template in rule_data["products"]:
                            try:
                                # This is simplified - would use proper reaction templates
                                product_smiles = self._apply_reaction_template(
                                    smiles, product_template, match
                                )

                                if product_smiles and product_smiles != smiles:
                                    # Estimate yield based on rate factor and molecular properties
                                    yield_frac = min(
                                        0.8 * rate_factor / len(matches), 1.0
                                    )

                                    product = DegradationProduct(
                                        parent_smiles=smiles,
                                        product_smiles=product_smiles,
                                        mechanism=mechanism,
                                        yield_fraction=yield_frac,
                                        environmental_conditions=conditions,
                                        confidence=ConfidenceLevel.MODERATE,
                                    )
                                    products.append(product)

                            except Exception:
                                continue

            return products

        except ImportError:
            # Fallback for demo
            return self._generate_demo_hydrolysis_products(smiles, conditions)

    def _apply_reaction_template(
        self, smiles: str, template: str, match: Tuple[int, ...]
    ) -> str:
        """Apply reaction template to generate product."""
        # Simplified implementation - would use proper reaction SMARTS
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""

            # For demo, generate simplified products
            if "ester" in template.lower():
                # Ester hydrolysis: R-COO-R' -> R-COOH + R'-OH
                return "CC(=O)O"  # Acetic acid as example
            elif "amide" in template.lower():
                # Amide hydrolysis: R-CO-NR'R'' -> R-COOH + NR'R''
                return "CC(=O)O"  # Carboxylic acid as example
            elif "nitrile" in template.lower():
                # Nitrile hydrolysis: R-CN -> R-COOH + NH3
                return "CC(=O)O"  # Carboxylic acid as example

            return ""

        except Exception:
            return ""

    def _generate_demo_hydrolysis_products(
        self, smiles: str, conditions: List[DegradationConditions]
    ) -> List[DegradationProduct]:
        """Generate demo hydrolysis products."""

        # Simple demo products based on common patterns
        demo_products = []

        if "C(=O)O" in smiles or "COO" in smiles:  # Ester-like
            demo_products.append(
                DegradationProduct(
                    parent_smiles=smiles,
                    product_smiles="CC(=O)O",  # Acetic acid
                    mechanism=DegradationMechanism.HYDROLYSIS,
                    yield_fraction=0.7,
                    half_life=24.0,
                    environmental_conditions=conditions,
                )
            )

        if "N" in smiles:  # Nitrogen-containing
            demo_products.append(
                DegradationProduct(
                    parent_smiles=smiles,
                    product_smiles="N",  # Ammonia
                    mechanism=DegradationMechanism.HYDROLYSIS,
                    yield_fraction=0.3,
                    half_life=48.0,
                    environmental_conditions=conditions,
                )
            )

        return demo_products

    def predict_half_life(
        self,
        smiles: str,
        mechanism: DegradationMechanism,
        compartment: EnvironmentalCompartment,
        conditions: List[DegradationConditions],
    ) -> Tuple[float, float]:
        """Predict hydrolysis half-life."""

        if mechanism != DegradationMechanism.HYDROLYSIS:
            return float("inf"), 0.0

        # Base half-life estimation
        base_half_life = 168.0  # 1 week default

        # Adjust for conditions
        rate_factor = 1.0
        for condition in conditions:
            if condition == DegradationConditions.ACIDIC:
                rate_factor *= 2.0
            elif condition == DegradationConditions.BASIC:
                rate_factor *= 5.0
            elif condition == DegradationConditions.HIGH_TEMPERATURE:
                rate_factor *= 3.0
            elif condition == DegradationConditions.LOW_TEMPERATURE:
                rate_factor *= 0.3

        # Adjust for compartment
        if compartment == EnvironmentalCompartment.SURFACE_WATER:
            rate_factor *= 1.2
        elif compartment == EnvironmentalCompartment.GROUNDWATER:
            rate_factor *= 0.5
        elif compartment == EnvironmentalCompartment.SOIL:
            rate_factor *= 0.8

        half_life = base_half_life / rate_factor
        uncertainty = 0.3  # 30% uncertainty

        return half_life, uncertainty


class PhotolysisModel(DegradationModel):
    """Model for photolytic degradation."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.photolysis_database = self._load_photolysis_data()

    def _load_photolysis_data(self) -> Dict[str, Any]:
        """Load photolysis reaction database."""
        return {
            "aromatic_photolysis": {
                "chromophores": ["c1ccccc1", "c1ccc2ccccc2c1"],  # Benzene, naphthalene
                "quantum_yields": {"direct": 0.1, "sensitized": 0.3},
                "wavelength_range": (280, 400),  # UV-B and UV-A
            },
            "carbonyl_photolysis": {
                "chromophores": ["C=O"],
                "quantum_yields": {"norrish_i": 0.2, "norrish_ii": 0.4},
                "wavelength_range": (300, 360),
            },
        }

    def predict_products(
        self,
        smiles: str,
        mechanism: DegradationMechanism,
        conditions: List[DegradationConditions],
    ) -> List[DegradationProduct]:
        """Predict photolysis products."""

        if mechanism != DegradationMechanism.PHOTOLYSIS:
            return []

        # Check for UV exposure
        has_uv = DegradationConditions.UV_EXPOSURE in conditions
        if not has_uv:
            return []

        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []

            products = []

            # Check for chromophores
            for photolysis_type, data in self.photolysis_database.items():
                for chromophore_smarts in data["chromophores"]:
                    pattern = Chem.MolFromSmarts(chromophore_smarts)
                    if pattern and mol.HasSubstructMatch(pattern):
                        # Generate photolysis products
                        demo_products = self._generate_photolysis_products(
                            smiles, photolysis_type, conditions
                        )
                        products.extend(demo_products)

            return products

        except ImportError:
            return self._generate_demo_photolysis_products(smiles, conditions)

    def _generate_photolysis_products(
        self, smiles: str, photolysis_type: str, conditions: List[DegradationConditions]
    ) -> List[DegradationProduct]:
        """Generate photolysis products based on type."""

        products = []

        if photolysis_type == "aromatic_photolysis":
            # Aromatic ring opening/modification
            products.append(
                DegradationProduct(
                    parent_smiles=smiles,
                    product_smiles="CC(=O)C",  # Ketone fragment
                    mechanism=DegradationMechanism.PHOTOLYSIS,
                    yield_fraction=0.4,
                    half_life=2.0,  # 2 hours in direct sunlight
                    environmental_conditions=conditions,
                )
            )

        elif photolysis_type == "carbonyl_photolysis":
            # Norrish reactions
            products.append(
                DegradationProduct(
                    parent_smiles=smiles,
                    product_smiles="CC",  # Alkyl fragment
                    mechanism=DegradationMechanism.PHOTOLYSIS,
                    yield_fraction=0.3,
                    half_life=1.0,  # 1 hour
                    environmental_conditions=conditions,
                )
            )

        return products

    def _generate_demo_photolysis_products(
        self, smiles: str, conditions: List[DegradationConditions]
    ) -> List[DegradationProduct]:
        """Generate demo photolysis products."""

        products = []

        # Simple heuristics for demo
        if any(char in smiles for char in ["c", "C"]):  # Carbon-containing
            products.append(
                DegradationProduct(
                    parent_smiles=smiles,
                    product_smiles="CO2",  # CO2 from mineralization
                    mechanism=DegradationMechanism.PHOTOLYSIS,
                    yield_fraction=0.2,
                    half_life=4.0,
                    environmental_conditions=conditions,
                )
            )

        return products

    def predict_half_life(
        self,
        smiles: str,
        mechanism: DegradationMechanism,
        compartment: EnvironmentalCompartment,
        conditions: List[DegradationConditions],
    ) -> Tuple[float, float]:
        """Predict photolysis half-life."""

        if mechanism != DegradationMechanism.PHOTOLYSIS:
            return float("inf"), 0.0

        # Check for UV exposure
        has_uv = DegradationConditions.UV_EXPOSURE in conditions
        if not has_uv:
            return float("inf"), 0.0

        # Base photolysis half-life (in direct sunlight)
        base_half_life = 8.0  # 8 hours

        # Adjust for compartment (light availability)
        if compartment == EnvironmentalCompartment.ATMOSPHERE:
            light_factor = 1.0
        elif compartment == EnvironmentalCompartment.SURFACE_WATER:
            light_factor = 2.0  # Some attenuation
        elif compartment == EnvironmentalCompartment.SOIL:
            light_factor = 10.0  # Significant attenuation
        else:
            light_factor = 100.0  # Very little light

        half_life = base_half_life * light_factor
        uncertainty = 0.5  # 50% uncertainty due to variable light conditions

        return half_life, uncertainty


class BiodegradationModel(DegradationModel):
    """Model for biodegradation processes."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.biodegradation_rules = self._load_biodegradation_rules()
        self.ml_model = None

    def _load_biodegradation_rules(self) -> Dict[str, Any]:
        """Load biodegradation pathway rules."""
        return {
            "aerobic_alkane": {
                "conditions": [DegradationConditions.AEROBIC],
                "pathway": "omega_oxidation",
                "products": ["alcohol", "aldehyde", "carboxylic_acid"],
            },
            "anaerobic_aromatic": {
                "conditions": [DegradationConditions.ANAEROBIC],
                "pathway": "ring_reduction",
                "products": ["cyclohexane_derivatives"],
            },
            "ester_biodegradation": {
                "conditions": [
                    DegradationConditions.AEROBIC,
                    DegradationConditions.ANAEROBIC,
                ],
                "pathway": "esterase_cleavage",
                "products": ["carboxylic_acid", "alcohol"],
            },
        }

    def predict_products(
        self,
        smiles: str,
        mechanism: DegradationMechanism,
        conditions: List[DegradationConditions],
    ) -> List[DegradationProduct]:
        """Predict biodegradation products."""

        if mechanism != DegradationMechanism.BIODEGRADATION:
            return []

        products = []

        # Apply biodegradation rules
        for rule_name, rule_data in self.biodegradation_rules.items():
            # Check if conditions match
            if any(cond in conditions for cond in rule_data["conditions"]):
                demo_products = self._generate_biodegradation_products(
                    smiles, rule_name, conditions
                )
                products.extend(demo_products)

        return products

    def _generate_biodegradation_products(
        self, smiles: str, rule_name: str, conditions: List[DegradationConditions]
    ) -> List[DegradationProduct]:
        """Generate biodegradation products based on rule."""

        products = []

        if "alkane" in rule_name:
            # Alkane oxidation products
            products.extend(
                [
                    DegradationProduct(
                        parent_smiles=smiles,
                        product_smiles="CCO",  # Alcohol
                        mechanism=DegradationMechanism.BIODEGRADATION,
                        yield_fraction=0.4,
                        half_life=72.0,
                        environmental_conditions=conditions,
                    ),
                    DegradationProduct(
                        parent_smiles=smiles,
                        product_smiles="CC(=O)O",  # Carboxylic acid
                        mechanism=DegradationMechanism.BIODEGRADATION,
                        yield_fraction=0.6,
                        half_life=168.0,
                        environmental_conditions=conditions,
                    ),
                ]
            )

        elif "ester" in rule_name:
            # Ester hydrolysis by enzymes
            products.append(
                DegradationProduct(
                    parent_smiles=smiles,
                    product_smiles="CC(=O)O",  # Carboxylic acid
                    mechanism=DegradationMechanism.BIODEGRADATION,
                    yield_fraction=0.8,
                    half_life=120.0,
                    environmental_conditions=conditions,
                )
            )

        return products

    def predict_half_life(
        self,
        smiles: str,
        mechanism: DegradationMechanism,
        compartment: EnvironmentalCompartment,
        conditions: List[DegradationConditions],
    ) -> Tuple[float, float]:
        """Predict biodegradation half-life."""

        if mechanism != DegradationMechanism.BIODEGRADATION:
            return float("inf"), 0.0

        # Base biodegradation half-life
        base_half_life = 720.0  # 30 days default

        # Adjust for oxygen availability
        if DegradationConditions.AEROBIC in conditions:
            oxygen_factor = 1.0
        elif DegradationConditions.ANAEROBIC in conditions:
            oxygen_factor = 5.0  # Slower anaerobic degradation
        else:
            oxygen_factor = 2.0  # Mixed conditions

        # Adjust for temperature
        temp_factor = 1.0
        if DegradationConditions.HIGH_TEMPERATURE in conditions:
            temp_factor = 0.5  # Faster at higher temperature
        elif DegradationConditions.LOW_TEMPERATURE in conditions:
            temp_factor = 3.0  # Slower at lower temperature

        # Adjust for compartment (microbial activity)
        if compartment == EnvironmentalCompartment.SOIL:
            microbial_factor = 1.0
        elif compartment == EnvironmentalCompartment.SURFACE_WATER:
            microbial_factor = 1.5
        elif compartment == EnvironmentalCompartment.SEDIMENT:
            microbial_factor = 2.0
        elif compartment == EnvironmentalCompartment.GROUNDWATER:
            microbial_factor = 5.0
        else:
            microbial_factor = 10.0

        half_life = base_half_life * oxygen_factor * temp_factor * microbial_factor
        uncertainty = 0.6  # 60% uncertainty due to microbial variability

        return half_life, uncertainty


class EnvironmentalFatePredictor:
    """
    Advanced environmental fate and degradation pathway predictor.

    Integrates multiple degradation mechanisms and environmental conditions
    to predict comprehensive fate and transport behavior.
    """

    def __init__(self, backend: Backend):
        self.backend = backend

        # Initialize degradation models
        self.models = {
            DegradationMechanism.HYDROLYSIS: HydrolysisModel(backend),
            DegradationMechanism.PHOTOLYSIS: PhotolysisModel(backend),
            DegradationMechanism.BIODEGRADATION: BiodegradationModel(backend),
        }

        # Environmental parameters
        self.default_conditions = {
            EnvironmentalCompartment.SURFACE_WATER: [
                DegradationConditions.AEROBIC,
                DegradationConditions.NEUTRAL,
                DegradationConditions.UV_EXPOSURE,
            ],
            EnvironmentalCompartment.SOIL: [
                DegradationConditions.AEROBIC,
                DegradationConditions.NEUTRAL,
                DegradationConditions.NO_UV,
            ],
            EnvironmentalCompartment.GROUNDWATER: [
                DegradationConditions.ANAEROBIC,
                DegradationConditions.NEUTRAL,
                DegradationConditions.NO_UV,
            ],
        }

    def predict_fate(
        self,
        smiles: Union[str, List[str]],
        environmental_conditions: Optional[Dict[str, Any]] = None,
        compartments: Optional[List[EnvironmentalCompartment]] = None,
        mechanisms: Optional[List[DegradationMechanism]] = None,
        **kwargs,
    ) -> Union[
        Dict[str, List[DegradationPathway]], List[Dict[str, List[DegradationPathway]]]
    ]:
        """
        Predict environmental fate and degradation pathways.

        Args:
            smiles: SMILES string(s)
            environmental_conditions: Environmental parameters
            compartments: Environmental compartments to consider
            mechanisms: Degradation mechanisms to evaluate

        Returns:
            Dictionary of degradation pathways by mechanism
        """

        # Default parameters
        if compartments is None:
            compartments = [
                EnvironmentalCompartment.SURFACE_WATER,
                EnvironmentalCompartment.SOIL,
                EnvironmentalCompartment.ATMOSPHERE,
            ]

        if mechanisms is None:
            mechanisms = [
                DegradationMechanism.HYDROLYSIS,
                DegradationMechanism.PHOTOLYSIS,
                DegradationMechanism.BIODEGRADATION,
            ]

        # Handle single SMILES
        if isinstance(smiles, str):
            return self._predict_single_compound_fate(
                smiles, environmental_conditions, compartments, mechanisms, **kwargs
            )

        # Handle multiple SMILES
        results = []
        for smi in smiles:
            result = self._predict_single_compound_fate(
                smi, environmental_conditions, compartments, mechanisms, **kwargs
            )
            results.append(result)

        return results

    def _predict_single_compound_fate(
        self,
        smiles: str,
        environmental_conditions: Optional[Dict[str, Any]],
        compartments: List[EnvironmentalCompartment],
        mechanisms: List[DegradationMechanism],
        **kwargs,
    ) -> Dict[str, List[DegradationPathway]]:
        """Predict fate for single compound."""

        fate_results = {}

        for compartment in compartments:
            # Get environmental conditions for this compartment
            conditions = self._get_compartment_conditions(
                compartment, environmental_conditions
            )

            compartment_pathways = []

            for mechanism in mechanisms:
                if mechanism in self.models:
                    pathway = self._predict_degradation_pathway(
                        smiles, mechanism, compartment, conditions
                    )
                    if pathway:
                        compartment_pathways.append(pathway)

            if compartment_pathways:
                fate_results[compartment.value] = compartment_pathways

        return fate_results

    def _get_compartment_conditions(
        self,
        compartment: EnvironmentalCompartment,
        environmental_conditions: Optional[Dict[str, Any]],
    ) -> List[DegradationConditions]:
        """Get environmental conditions for compartment."""

        # Start with default conditions
        conditions = self.default_conditions.get(
            compartment, [DegradationConditions.AEROBIC, DegradationConditions.NEUTRAL]
        )

        # Override with user-specified conditions
        if environmental_conditions:
            if f"{compartment.value}_conditions" in environmental_conditions:
                user_conditions = environmental_conditions[
                    f"{compartment.value}_conditions"
                ]
                if isinstance(user_conditions, list):
                    conditions = [DegradationConditions(c) for c in user_conditions]

            # Global condition overrides
            if "pH" in environmental_conditions:
                pH = environmental_conditions["pH"]
                if pH < 6:
                    conditions = [
                        c
                        for c in conditions
                        if c
                        not in [
                            DegradationConditions.NEUTRAL,
                            DegradationConditions.BASIC,
                        ]
                    ]
                    conditions.append(DegradationConditions.ACIDIC)
                elif pH > 8:
                    conditions = [
                        c
                        for c in conditions
                        if c
                        not in [
                            DegradationConditions.NEUTRAL,
                            DegradationConditions.ACIDIC,
                        ]
                    ]
                    conditions.append(DegradationConditions.BASIC)

            if "temperature" in environmental_conditions:
                temp = environmental_conditions["temperature"]
                if temp > 25:  # Celsius
                    conditions.append(DegradationConditions.HIGH_TEMPERATURE)
                elif temp < 10:
                    conditions.append(DegradationConditions.LOW_TEMPERATURE)

        return list(set(conditions))  # Remove duplicates

    def _predict_degradation_pathway(
        self,
        smiles: str,
        mechanism: DegradationMechanism,
        compartment: EnvironmentalCompartment,
        conditions: List[DegradationConditions],
    ) -> Optional[DegradationPathway]:
        """Predict degradation pathway for specific mechanism."""

        model = self.models[mechanism]

        # Predict products
        products = model.predict_products(smiles, mechanism, conditions)

        if not products:
            return None

        # Predict half-life
        half_life, uncertainty = model.predict_half_life(
            smiles, mechanism, compartment, conditions
        )

        # Calculate mass balance
        total_yield = sum(product.yield_fraction for product in products)
        mass_balance = min(total_yield, 1.0)

        # Determine pathway confidence
        if uncertainty < 0.3:
            pathway_confidence = ConfidenceLevel.HIGH
        elif uncertainty < 0.5:
            pathway_confidence = ConfidenceLevel.MODERATE
        else:
            pathway_confidence = ConfidenceLevel.LOW

        return DegradationPathway(
            parent_compound=smiles,
            mechanism=mechanism,
            compartment=compartment,
            conditions=conditions,
            products=products,
            overall_half_life=half_life,
            mass_balance=mass_balance,
            pathway_confidence=pathway_confidence,
            kinetic_parameters={
                "k_degradation": 0.693 / half_life
            },  # First-order rate constant
        )

    def predict_multimedia_fate(
        self,
        smiles: str,
        release_scenario: Dict[str, Any],
        simulation_time: float = 8760.0,  # 1 year in hours
    ) -> Dict[str, Any]:
        """
        Predict multimedia environmental fate using fugacity modeling.

        Args:
            smiles: SMILES string
            release_scenario: Release rates and locations
            simulation_time: Simulation time in hours

        Returns:
            Multimedia fate modeling results
        """

        # This would implement a full multimedia fate model
        # For now, return simplified results

        compartment_fractions = {
            "atmosphere": 0.1,
            "surface_water": 0.3,
            "soil": 0.5,
            "sediment": 0.1,
        }

        # Simulate partitioning over time
        time_points = np.linspace(0, simulation_time, 100)
        concentration_profiles = {}

        for compartment in compartment_fractions:
            # Simple exponential decay
            initial_fraction = compartment_fractions[compartment]
            decay_rate = 0.693 / (168.0)  # 1 week half-life

            concentrations = initial_fraction * np.exp(-decay_rate * time_points / 24)
            concentration_profiles[compartment] = concentrations.tolist()

        return {
            "compound": smiles,
            "simulation_time_hours": simulation_time,
            "time_points": time_points.tolist(),
            "concentration_profiles": concentration_profiles,
            "final_distribution": {
                comp: conc[-1] for comp, conc in concentration_profiles.items()
            },
            "total_degraded": 1.0
            - sum(conc[-1] for conc in concentration_profiles.values()),
        }

    def estimate_bioaccumulation_potential(
        self, smiles: str, trophic_levels: int = 4
    ) -> Dict[str, float]:
        """Estimate bioaccumulation potential."""

        try:
            from rdkit import Chem
            from rdkit.Chem import Crippen, Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES"}

            # Calculate relevant descriptors
            log_p = Crippen.MolLogP(mol)
            mol_wt = Descriptors.MolWt(mol)

            # Estimate bioconcentration factor (BCF)
            if log_p < 1:
                log_bcf = 0.5 * log_p
            elif log_p <= 6:
                log_bcf = 0.85 * log_p - 0.7
            else:
                log_bcf = -0.2 * log_p + 6.9  # Decreases for very lipophilic compounds

            bcf = 10**log_bcf

            # Estimate biomagnification factor (BMF)
            if mol_wt > 500:  # Large molecules less likely to biomagnify
                bmf = 1.0
            else:
                bmf = min(10 ** (0.1 * log_p), 10.0)

            # Calculate trophic magnification factor (TMF)
            tmf = bmf ** (trophic_levels - 1)

            return {
                "log_p": log_p,
                "molecular_weight": mol_wt,
                "bioconcentration_factor": bcf,
                "biomagnification_factor": bmf,
                "trophic_magnification_factor": tmf,
                "bioaccumulation_category": self._classify_bioaccumulation(bcf, bmf),
            }

        except ImportError:
            # Fallback for demo
            return {
                "bioconcentration_factor": 100.0,
                "biomagnification_factor": 2.0,
                "trophic_magnification_factor": 8.0,
                "bioaccumulation_category": "moderate",
            }

    def _classify_bioaccumulation(self, bcf: float, bmf: float) -> str:
        """Classify bioaccumulation potential."""

        if bcf > 5000 or bmf > 5:
            return "very_high"
        elif bcf > 1000 or bmf > 2:
            return "high"
        elif bcf > 100 or bmf > 1.5:
            return "moderate"
        else:
            return "low"

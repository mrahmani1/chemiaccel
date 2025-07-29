"""
Recyclability and Re-usability Assessment Module.

This module analyzes molecular structure and material composition to predict
the feasibility and efficiency of various recycling processes including
mechanical, chemical, and biological recycling approaches.
"""

import asyncio
import json
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Optional dependencies
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class RecyclingMethod(Enum):
    """Types of recycling methods."""

    MECHANICAL_RECYCLING = "mechanical_recycling"
    CHEMICAL_RECYCLING = "chemical_recycling"
    BIOLOGICAL_RECYCLING = "biological_recycling"
    THERMAL_RECYCLING = "thermal_recycling"
    SOLVENT_BASED_RECYCLING = "solvent_based_recycling"
    DEPOLYMERIZATION = "depolymerization"
    PYROLYSIS = "pyrolysis"
    GASIFICATION = "gasification"
    ENZYMATIC_RECYCLING = "enzymatic_recycling"
    HYDROLYSIS = "hydrolysis"


class MaterialCategory(Enum):
    """Categories of materials for recycling assessment."""

    THERMOPLASTIC = "thermoplastic"
    THERMOSET = "thermoset"
    ELASTOMER = "elastomer"
    FIBER = "fiber"
    COMPOSITE = "composite"
    BIODEGRADABLE = "biodegradable"
    METAL = "metal"
    CERAMIC = "ceramic"
    GLASS = "glass"
    NATURAL_POLYMER = "natural_polymer"


class RecyclabilityMetric(Enum):
    """Metrics for recyclability assessment."""

    PROCESSING_EASE = "processing_ease"
    MATERIAL_RECOVERY_RATE = "material_recovery_rate"
    QUALITY_RETENTION = "quality_retention"
    ECONOMIC_VIABILITY = "economic_viability"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    CONTAMINATION_TOLERANCE = "contamination_tolerance"
    INFRASTRUCTURE_COMPATIBILITY = "infrastructure_compatibility"
    CIRCULARITY_POTENTIAL = "circularity_potential"


@dataclass
class MaterialComposition:
    """Material composition and structure information."""

    polymer_content: Dict[str, float]  # Polymer types and percentages
    additive_content: Dict[str, float]  # Additives and percentages
    filler_content: Dict[str, float]  # Fillers and percentages
    contaminants: Dict[str, float] = field(default_factory=dict)
    molecular_weight_distribution: Optional[Tuple[float, float]] = None  # (Mn, Mw)
    crystallinity: Optional[float] = None
    crosslink_density: Optional[float] = None
    glass_transition_temp: Optional[float] = None
    melting_point: Optional[float] = None
    degradation_temp: Optional[float] = None
    chemical_structure: Optional[str] = None  # SMILES or other representation


@dataclass
class RecyclingConditions:
    """Conditions for recycling process."""

    method: RecyclingMethod
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    time: Optional[float] = None
    solvent: Optional[str] = None
    catalyst: Optional[str] = None
    ph: Optional[float] = None
    atmosphere: Optional[str] = None
    preprocessing_required: bool = False
    energy_requirements: Optional[float] = None  # kWh/kg
    chemical_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class RecyclabilityScore:
    """Recyclability assessment score for a specific method."""

    method: RecyclingMethod
    overall_score: float  # 0.0 to 1.0
    metric_scores: Dict[RecyclabilityMetric, float]
    feasibility: bool
    confidence: float
    processing_efficiency: Optional[float] = None
    material_recovery_rate: Optional[float] = None
    quality_degradation: Optional[float] = None
    economic_score: Optional[float] = None
    environmental_score: Optional[float] = None
    technical_barriers: List[str] = field(default_factory=list)
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)


@dataclass
class CircularityMetrics:
    """Comprehensive circularity assessment metrics."""

    circularity_index: float  # 0.0 to 1.0
    recyclability_score: float  # 0.0 to 1.0
    reusability_score: float  # 0.0 to 1.0
    biodegradability_score: float  # 0.0 to 1.0
    durability_score: float  # 0.0 to 1.0
    design_for_recycling_score: float  # 0.0 to 1.0
    material_compatibility_score: float  # 0.0 to 1.0
    economic_viability_score: float  # 0.0 to 1.0
    infrastructure_readiness_score: float  # 0.0 to 1.0
    lifecycle_impact_score: float  # 0.0 to 1.0 (lower is better)


@dataclass
class RecyclabilityAssessment:
    """Complete recyclability assessment results."""

    material_id: str
    material_composition: MaterialComposition
    material_category: MaterialCategory
    recycling_scores: Dict[RecyclingMethod, RecyclabilityScore]
    recommended_methods: List[RecyclingMethod]
    circularity_metrics: CircularityMetrics
    processing_recommendations: Dict[str, Any]
    contamination_sensitivity: Dict[str, float]
    market_potential: Dict[str, float]
    regulatory_considerations: List[str] = field(default_factory=list)
    assessment_date: datetime = field(default_factory=datetime.now)


class RecyclabilityPredictor(ABC):
    """Abstract base class for recyclability predictors."""

    @abstractmethod
    def assess_recyclability(
        self, composition: MaterialComposition
    ) -> RecyclabilityAssessment:
        """Assess material recyclability."""
        pass

    @abstractmethod
    def predict_method_feasibility(
        self, composition: MaterialComposition, method: RecyclingMethod
    ) -> RecyclabilityScore:
        """Predict feasibility of specific recycling method."""
        pass


class MolecularStructureAnalyzer:
    """Analyze molecular structure for recyclability prediction."""

    def __init__(self):
        self.structure_patterns = {
            "ester_bonds": "C(=O)O",
            "amide_bonds": "C(=O)N",
            "ether_bonds": "COC",
            "aromatic_rings": "c1ccccc1",
            "crosslinks": "C1CC1",  # Simplified
            "degradable_groups": ["C(=O)O", "C(=O)N", "COC"],
        }

    def analyze_structure(self, smiles: str) -> Dict[str, Any]:
        """Analyze molecular structure from SMILES."""

        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, using simplified analysis")
            return self._simplified_structure_analysis(smiles)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._simplified_structure_analysis(smiles)

            analysis = {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Crippen.MolLogP(mol),
                "aromatic_atoms": rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
                "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "hydrogen_bond_donors": rdMolDescriptors.CalcNumHBD(mol),
                "hydrogen_bond_acceptors": rdMolDescriptors.CalcNumHBA(mol),
                "topological_polar_surface_area": rdMolDescriptors.CalcTPSA(mol),
                "ring_count": rdMolDescriptors.CalcNumRings(mol),
                "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
                "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "saturated_rings": rdMolDescriptors.CalcNumSaturatedRings(mol),
            }

            # Check for specific functional groups
            analysis["functional_groups"] = self._identify_functional_groups(mol)

            # Assess degradability based on structure
            analysis["hydrolyzable_bonds"] = self._count_hydrolyzable_bonds(mol)
            analysis["oxidizable_sites"] = self._count_oxidizable_sites(mol)

            return analysis

        except Exception as e:
            logger.error(f"Error in RDKit analysis: {e}")
            return self._simplified_structure_analysis(smiles)

    def _simplified_structure_analysis(self, smiles: str) -> Dict[str, Any]:
        """Simplified structure analysis without RDKit."""

        analysis = {
            "molecular_weight": len(smiles) * 10,  # Rough approximation
            "aromatic_rings": smiles.count("c"),
            "functional_groups": [],
            "hydrolyzable_bonds": 0,
            "oxidizable_sites": 0,
        }

        # Simple pattern matching
        if "C(=O)O" in smiles:
            analysis["functional_groups"].append("ester")
            analysis["hydrolyzable_bonds"] += smiles.count("C(=O)O")

        if "C(=O)N" in smiles:
            analysis["functional_groups"].append("amide")
            analysis["hydrolyzable_bonds"] += smiles.count("C(=O)N")

        if "COC" in smiles:
            analysis["functional_groups"].append("ether")

        return analysis

    def _identify_functional_groups(self, mol) -> List[str]:
        """Identify functional groups using SMARTS patterns."""

        patterns = {
            "ester": "[C](=O)[O]",
            "amide": "[C](=O)[N]",
            "ether": "[C][O][C]",
            "alcohol": "[C][OH]",
            "carboxylic_acid": "[C](=O)[OH]",
            "aromatic": "c1ccccc1",
            "alkene": "[C]=[C]",
            "alkyne": "[C]#[C]",
        }

        functional_groups = []

        for group_name, pattern in patterns.items():
            try:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    functional_groups.append(group_name)
            except:
                continue

        return functional_groups

    def _count_hydrolyzable_bonds(self, mol) -> int:
        """Count hydrolyzable bonds (esters, amides)."""

        hydrolyzable_patterns = [
            "[C](=O)[O]",  # Ester
            "[C](=O)[N]",  # Amide
        ]

        count = 0
        for pattern in hydrolyzable_patterns:
            try:
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
                count += len(matches)
            except:
                continue

        return count

    def _count_oxidizable_sites(self, mol) -> int:
        """Count sites susceptible to oxidation."""

        oxidizable_patterns = [
            "[C][C]",  # Alkyl chains
            "[C]=[C]",  # Alkenes
            "[C][OH]",  # Alcohols
        ]

        count = 0
        for pattern in oxidizable_patterns:
            try:
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
                count += len(matches)
            except:
                continue

        return count


class MechanicalRecyclingPredictor(RecyclabilityPredictor):
    """Predictor for mechanical recycling feasibility."""

    def __init__(self):
        self.thermal_stability_threshold = 200.0  # °C
        self.melt_flow_threshold = 10.0  # g/10min
        self.contamination_tolerance = {
            "paper": 0.05,
            "metal": 0.01,
            "other_polymers": 0.10,
            "adhesives": 0.02,
        }

    def assess_recyclability(
        self, composition: MaterialComposition
    ) -> RecyclabilityAssessment:
        """Assess mechanical recyclability."""

        # Determine material category
        category = self._determine_material_category(composition)

        # Assess mechanical recycling
        mechanical_score = self.predict_method_feasibility(
            composition, RecyclingMethod.MECHANICAL_RECYCLING
        )

        # Generate circularity metrics
        circularity_metrics = self._calculate_circularity_metrics(composition)

        return RecyclabilityAssessment(
            material_id=f"mech_{hash(str(composition))}",
            material_composition=composition,
            material_category=category,
            recycling_scores={RecyclingMethod.MECHANICAL_RECYCLING: mechanical_score},
            recommended_methods=[RecyclingMethod.MECHANICAL_RECYCLING]
            if mechanical_score.feasibility
            else [],
            circularity_metrics=circularity_metrics,
            processing_recommendations=self._generate_processing_recommendations(
                composition
            ),
            contamination_sensitivity=self._assess_contamination_sensitivity(
                composition
            ),
        )

    def predict_method_feasibility(
        self, composition: MaterialComposition, method: RecyclingMethod
    ) -> RecyclabilityScore:
        """Predict mechanical recycling feasibility."""

        if method != RecyclingMethod.MECHANICAL_RECYCLING:
            raise ValueError("This predictor only handles mechanical recycling")

        # Score different aspects
        thermal_score = self._score_thermal_properties(composition)
        purity_score = self._score_material_purity(composition)
        processing_score = self._score_processing_ease(composition)
        contamination_score = self._score_contamination_tolerance(composition)

        # Calculate overall score
        weights = {
            RecyclabilityMetric.PROCESSING_EASE: 0.25,
            RecyclabilityMetric.MATERIAL_RECOVERY_RATE: 0.20,
            RecyclabilityMetric.QUALITY_RETENTION: 0.25,
            RecyclabilityMetric.CONTAMINATION_TOLERANCE: 0.15,
            RecyclabilityMetric.ECONOMIC_VIABILITY: 0.15,
        }

        metric_scores = {
            RecyclabilityMetric.PROCESSING_EASE: processing_score,
            RecyclabilityMetric.MATERIAL_RECOVERY_RATE: purity_score,
            RecyclabilityMetric.QUALITY_RETENTION: thermal_score,
            RecyclabilityMetric.CONTAMINATION_TOLERANCE: contamination_score,
            RecyclabilityMetric.ECONOMIC_VIABILITY: (processing_score + purity_score)
            / 2,
        }

        overall_score = sum(
            weights.get(metric, 0.0) * score for metric, score in metric_scores.items()
        )

        # Determine feasibility
        feasibility = overall_score > 0.6 and thermal_score > 0.5

        # Identify barriers and advantages
        barriers, advantages = self._identify_barriers_and_advantages(
            composition, metric_scores
        )

        return RecyclabilityScore(
            method=method,
            overall_score=overall_score,
            metric_scores=metric_scores,
            feasibility=feasibility,
            confidence=0.8,  # Fixed confidence for mechanical recycling
            processing_efficiency=processing_score,
            material_recovery_rate=purity_score,
            quality_degradation=1.0 - thermal_score,
            technical_barriers=barriers,
            advantages=advantages,
        )

    def _determine_material_category(
        self, composition: MaterialComposition
    ) -> MaterialCategory:
        """Determine material category based on composition."""

        # Check for thermosets
        if composition.crosslink_density and composition.crosslink_density > 0.1:
            return MaterialCategory.THERMOSET

        # Check for thermoplastics
        main_polymers = ["PE", "PP", "PS", "PVC", "PET", "PC", "PA", "PMMA", "ABS"]

        for polymer in main_polymers:
            if any(polymer in key for key in composition.polymer_content.keys()):
                return MaterialCategory.THERMOPLASTIC

        # Check for biodegradable materials
        biodegradable_polymers = ["PLA", "PHA", "starch", "cellulose"]
        for polymer in biodegradable_polymers:
            if any(polymer in key for key in composition.polymer_content.keys()):
                return MaterialCategory.BIODEGRADABLE

        # Default to thermoplastic
        return MaterialCategory.THERMOPLASTIC

    def _score_thermal_properties(self, composition: MaterialComposition) -> float:
        """Score thermal stability for mechanical recycling."""

        if composition.degradation_temp is None:
            # Estimate based on polymer type
            degradation_temp = 250.0  # Default estimate
        else:
            degradation_temp = composition.degradation_temp

        # Score based on thermal stability
        if degradation_temp > 300.0:
            return 1.0
        elif degradation_temp > 250.0:
            return 0.8
        elif degradation_temp > 200.0:
            return 0.6
        elif degradation_temp > 150.0:
            return 0.4
        else:
            return 0.2

    def _score_material_purity(self, composition: MaterialComposition) -> float:
        """Score material purity."""

        # Calculate polymer purity
        total_polymer = sum(composition.polymer_content.values())
        total_additives = sum(composition.additive_content.values())
        total_fillers = sum(composition.filler_content.values())
        total_contaminants = sum(composition.contaminants.values())

        total_content = (
            total_polymer + total_additives + total_fillers + total_contaminants
        )

        if total_content == 0:
            return 0.5

        polymer_fraction = total_polymer / total_content

        # High polymer content is good for mechanical recycling
        if polymer_fraction > 0.9:
            return 1.0
        elif polymer_fraction > 0.8:
            return 0.8
        elif polymer_fraction > 0.7:
            return 0.6
        elif polymer_fraction > 0.6:
            return 0.4
        else:
            return 0.2

    def _score_processing_ease(self, composition: MaterialComposition) -> float:
        """Score ease of processing."""

        score = 0.7  # Base score

        # Adjust for melting point
        if composition.melting_point:
            if 150 <= composition.melting_point <= 250:
                score += 0.2  # Ideal processing range
            elif composition.melting_point > 300:
                score -= 0.3  # Too high
            elif composition.melting_point < 100:
                score -= 0.2  # Too low

        # Adjust for crystallinity
        if composition.crystallinity:
            if 0.2 <= composition.crystallinity <= 0.6:
                score += 0.1  # Good balance
            elif composition.crystallinity > 0.8:
                score -= 0.2  # Very crystalline, harder to process

        return max(0.0, min(1.0, score))

    def _score_contamination_tolerance(self, composition: MaterialComposition) -> float:
        """Score tolerance to contamination."""

        total_contaminants = sum(composition.contaminants.values())

        if total_contaminants == 0:
            return 1.0
        elif total_contaminants < 0.05:
            return 0.8
        elif total_contaminants < 0.10:
            return 0.6
        elif total_contaminants < 0.20:
            return 0.4
        else:
            return 0.2

    def _identify_barriers_and_advantages(
        self,
        composition: MaterialComposition,
        metric_scores: Dict[RecyclabilityMetric, float],
    ) -> Tuple[List[str], List[str]]:
        """Identify technical barriers and advantages."""

        barriers = []
        advantages = []

        # Check thermal properties
        if metric_scores[RecyclabilityMetric.QUALITY_RETENTION] < 0.5:
            barriers.append("Low thermal stability")
        else:
            advantages.append("Good thermal stability")

        # Check purity
        if metric_scores[RecyclabilityMetric.MATERIAL_RECOVERY_RATE] < 0.6:
            barriers.append("High additive/filler content")
        else:
            advantages.append("High polymer purity")

        # Check contamination
        if metric_scores[RecyclabilityMetric.CONTAMINATION_TOLERANCE] < 0.6:
            barriers.append("High contamination sensitivity")

        # Check crosslinking
        if composition.crosslink_density and composition.crosslink_density > 0.1:
            barriers.append("Crosslinked structure")

        return barriers, advantages

    def _calculate_circularity_metrics(
        self, composition: MaterialComposition
    ) -> CircularityMetrics:
        """Calculate comprehensive circularity metrics."""

        # Simple implementation - can be expanded
        recyclability = self._score_material_purity(composition) * 0.8
        reusability = (
            0.7
            if composition.crosslink_density and composition.crosslink_density < 0.05
            else 0.3
        )
        biodegradability = 0.1  # Most synthetic polymers have low biodegradability
        durability = self._score_thermal_properties(composition)

        circularity_index = (
            recyclability + reusability + biodegradability + durability
        ) / 4

        return CircularityMetrics(
            circularity_index=circularity_index,
            recyclability_score=recyclability,
            reusability_score=reusability,
            biodegradability_score=biodegradability,
            durability_score=durability,
            design_for_recycling_score=0.6,
            material_compatibility_score=0.7,
            economic_viability_score=0.6,
            infrastructure_readiness_score=0.8,
            lifecycle_impact_score=0.3,
        )

    def _generate_processing_recommendations(
        self, composition: MaterialComposition
    ) -> Dict[str, Any]:
        """Generate processing recommendations."""

        recommendations = {
            "preprocessing": [],
            "processing_conditions": {},
            "quality_improvements": [],
        }

        # Preprocessing recommendations
        if sum(composition.contaminants.values()) > 0.05:
            recommendations["preprocessing"].append("Contamination removal")

        if sum(composition.additive_content.values()) > 0.3:
            recommendations["preprocessing"].append("Additive extraction")

        # Processing conditions
        if composition.melting_point:
            recommendations["processing_conditions"]["temperature"] = (
                composition.melting_point + 20
            )

        # Quality improvements
        recommendations["quality_improvements"].extend(
            ["Controlled cooling", "Compatibilizer addition", "Antioxidant addition"]
        )

        return recommendations

    def _assess_contamination_sensitivity(
        self, composition: MaterialComposition
    ) -> Dict[str, float]:
        """Assess sensitivity to different types of contamination."""

        return {
            "paper": 0.8,  # Moderate tolerance
            "metal": 0.3,  # Low tolerance
            "other_polymers": 0.6,  # Moderate tolerance
            "adhesives": 0.2,  # Low tolerance
            "colorants": 0.9,  # High tolerance
            "moisture": 0.4,  # Low tolerance
        }


class ChemicalRecyclingPredictor(RecyclabilityPredictor):
    """Predictor for chemical recycling feasibility."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.structure_analyzer = MolecularStructureAnalyzer()

        # Chemical recycling methods and their requirements
        self.method_requirements = {
            RecyclingMethod.DEPOLYMERIZATION: {
                "hydrolyzable_bonds": 2,
                "temperature_range": (150, 300),
                "suitable_polymers": ["PET", "PA", "PC", "PU"],
            },
            RecyclingMethod.PYROLYSIS: {
                "temperature_range": (400, 800),
                "suitable_polymers": ["PE", "PP", "PS", "mixed"],
            },
            RecyclingMethod.SOLVENT_BASED_RECYCLING: {
                "solubility_required": True,
                "suitable_polymers": ["PS", "PC", "PMMA"],
            },
            RecyclingMethod.HYDROLYSIS: {
                "hydrolyzable_bonds": 1,
                "suitable_polymers": ["PET", "PA", "PU"],
            },
        }

    def assess_recyclability(
        self, composition: MaterialComposition
    ) -> RecyclabilityAssessment:
        """Assess chemical recyclability."""

        category = self._determine_material_category(composition)

        # Assess multiple chemical recycling methods
        recycling_scores = {}
        for method in [
            RecyclingMethod.DEPOLYMERIZATION,
            RecyclingMethod.PYROLYSIS,
            RecyclingMethod.SOLVENT_BASED_RECYCLING,
            RecyclingMethod.HYDROLYSIS,
        ]:
            score = self.predict_method_feasibility(composition, method)
            recycling_scores[method] = score

        # Find recommended methods
        recommended_methods = [
            method
            for method, score in recycling_scores.items()
            if score.feasibility and score.overall_score > 0.6
        ]

        # Generate circularity metrics
        circularity_metrics = self._calculate_circularity_metrics(
            composition, recycling_scores
        )

        return RecyclabilityAssessment(
            material_id=f"chem_{hash(str(composition))}",
            material_composition=composition,
            material_category=category,
            recycling_scores=recycling_scores,
            recommended_methods=recommended_methods,
            circularity_metrics=circularity_metrics,
            processing_recommendations=self._generate_processing_recommendations(
                composition
            ),
            contamination_sensitivity=self._assess_contamination_sensitivity(
                composition
            ),
            market_potential=self._assess_market_potential(composition),
        )

    def predict_method_feasibility(
        self, composition: MaterialComposition, method: RecyclingMethod
    ) -> RecyclabilityScore:
        """Predict chemical recycling method feasibility."""

        # Analyze molecular structure if available
        structure_analysis = {}
        if composition.chemical_structure:
            structure_analysis = self.structure_analyzer.analyze_structure(
                composition.chemical_structure
            )

        # Method-specific assessment
        if method == RecyclingMethod.DEPOLYMERIZATION:
            return self._assess_depolymerization(composition, structure_analysis)
        elif method == RecyclingMethod.PYROLYSIS:
            return self._assess_pyrolysis(composition, structure_analysis)
        elif method == RecyclingMethod.SOLVENT_BASED_RECYCLING:
            return self._assess_solvent_recycling(composition, structure_analysis)
        elif method == RecyclingMethod.HYDROLYSIS:
            return self._assess_hydrolysis(composition, structure_analysis)
        else:
            return self._default_assessment(method, composition)

    def _assess_depolymerization(
        self, composition: MaterialComposition, structure_analysis: Dict[str, Any]
    ) -> RecyclabilityScore:
        """Assess depolymerization feasibility."""

        # Check for hydrolyzable bonds
        hydrolyzable_bonds = structure_analysis.get("hydrolyzable_bonds", 0)
        functional_groups = structure_analysis.get("functional_groups", [])

        # Score components
        structural_score = 0.8 if hydrolyzable_bonds > 0 else 0.3
        functional_score = (
            0.9 if "ester" in functional_groups or "amide" in functional_groups else 0.4
        )
        purity_score = self._calculate_purity_score(composition)

        # Check polymer compatibility
        compatible_polymers = ["PET", "PA", "PC", "PU"]
        polymer_compatibility = any(
            any(compat in polymer for compat in compatible_polymers)
            for polymer in composition.polymer_content.keys()
        )
        compatibility_score = 0.9 if polymer_compatibility else 0.2

        metric_scores = {
            RecyclabilityMetric.PROCESSING_EASE: structural_score,
            RecyclabilityMetric.MATERIAL_RECOVERY_RATE: 0.85,
            RecyclabilityMetric.QUALITY_RETENTION: 0.9,
            RecyclabilityMetric.ECONOMIC_VIABILITY: compatibility_score,
            RecyclabilityMetric.ENVIRONMENTAL_IMPACT: 0.7,
        }

        overall_score = np.mean(list(metric_scores.values()))
        feasibility = overall_score > 0.6 and hydrolyzable_bonds > 0

        barriers = []
        advantages = []

        if hydrolyzable_bonds == 0:
            barriers.append("No hydrolyzable bonds")
        else:
            advantages.append("Contains hydrolyzable bonds")

        if not polymer_compatibility:
            barriers.append("Polymer not suitable for depolymerization")
        else:
            advantages.append("Compatible polymer type")

        return RecyclabilityScore(
            method=RecyclingMethod.DEPOLYMERIZATION,
            overall_score=overall_score,
            metric_scores=metric_scores,
            feasibility=feasibility,
            confidence=0.8,
            material_recovery_rate=0.85,
            quality_degradation=0.1,
            technical_barriers=barriers,
            advantages=advantages,
        )

    def _assess_pyrolysis(
        self, composition: MaterialComposition, structure_analysis: Dict[str, Any]
    ) -> RecyclabilityScore:
        """Assess pyrolysis feasibility."""

        # Pyrolysis is generally applicable to most organic polymers
        thermal_score = 0.8  # Most polymers can be pyrolyzed

        # Check for problematic elements
        problematic_elements = ["Cl", "Br", "F", "S"]  # Can form toxic gases
        safety_score = 0.9

        # Simplified check - in real implementation, would analyze composition
        purity_score = self._calculate_purity_score(composition)

        # Energy requirements are high for pyrolysis
        energy_score = 0.6

        metric_scores = {
            RecyclabilityMetric.PROCESSING_EASE: thermal_score,
            RecyclabilityMetric.MATERIAL_RECOVERY_RATE: 0.7,
            RecyclabilityMetric.QUALITY_RETENTION: 0.6,  # Products are different
            RecyclabilityMetric.ECONOMIC_VIABILITY: energy_score,
            RecyclabilityMetric.ENVIRONMENTAL_IMPACT: safety_score,
        }

        overall_score = np.mean(list(metric_scores.values()))
        feasibility = overall_score > 0.5

        advantages = [
            "Applicable to mixed polymer waste",
            "Produces valuable chemicals",
            "Handles contaminated materials",
        ]

        barriers = [
            "High energy requirements",
            "Complex product separation",
            "Potential toxic emissions",
        ]

        return RecyclabilityScore(
            method=RecyclingMethod.PYROLYSIS,
            overall_score=overall_score,
            metric_scores=metric_scores,
            feasibility=feasibility,
            confidence=0.9,
            material_recovery_rate=0.7,
            quality_degradation=0.4,
            technical_barriers=barriers,
            advantages=advantages,
        )

    def _assess_solvent_recycling(
        self, composition: MaterialComposition, structure_analysis: Dict[str, Any]
    ) -> RecyclabilityScore:
        """Assess solvent-based recycling feasibility."""

        # Check solubility indicators
        logp = structure_analysis.get("logp", 0)
        molecular_weight = structure_analysis.get("molecular_weight", 50000)

        # Lower LogP and MW generally indicate better solubility
        solubility_score = max(0.2, 1.0 - abs(logp) / 5.0)
        if molecular_weight > 100000:
            solubility_score *= 0.7

        # Crosslinked materials are not soluble
        if composition.crosslink_density and composition.crosslink_density > 0.05:
            solubility_score = 0.1

        purity_score = self._calculate_purity_score(composition)
        selectivity_score = 0.8  # Solvent methods can be selective

        metric_scores = {
            RecyclabilityMetric.PROCESSING_EASE: solubility_score,
            RecyclabilityMetric.MATERIAL_RECOVERY_RATE: 0.8,
            RecyclabilityMetric.QUALITY_RETENTION: 0.9,
            RecyclabilityMetric.ECONOMIC_VIABILITY: 0.6,  # Solvent costs
            RecyclabilityMetric.ENVIRONMENTAL_IMPACT: 0.5,  # Solvent emissions
        }

        overall_score = np.mean(list(metric_scores.values()))
        feasibility = overall_score > 0.6 and solubility_score > 0.4

        barriers = []
        advantages = []

        if solubility_score < 0.4:
            barriers.append("Poor solubility")
        else:
            advantages.append("Good solubility expected")

        if composition.crosslink_density and composition.crosslink_density > 0.05:
            barriers.append("Crosslinked structure prevents dissolution")

        advantages.extend(
            ["High quality recovered material", "Selective separation possible"]
        )

        barriers.extend(
            ["Solvent recovery required", "Environmental concerns with solvents"]
        )

        return RecyclabilityScore(
            method=RecyclingMethod.SOLVENT_BASED_RECYCLING,
            overall_score=overall_score,
            metric_scores=metric_scores,
            feasibility=feasibility,
            confidence=0.7,
            material_recovery_rate=0.8,
            quality_degradation=0.1,
            technical_barriers=barriers,
            advantages=advantages,
        )

    def _assess_hydrolysis(
        self, composition: MaterialComposition, structure_analysis: Dict[str, Any]
    ) -> RecyclabilityScore:
        """Assess hydrolysis feasibility."""

        # Similar to depolymerization but specifically water-based
        hydrolyzable_bonds = structure_analysis.get("hydrolyzable_bonds", 0)
        functional_groups = structure_analysis.get("functional_groups", [])

        structural_score = 0.9 if hydrolyzable_bonds > 0 else 0.2

        # Hydrolysis works best with ester bonds
        if "ester" in functional_groups:
            bond_score = 0.9
        elif "amide" in functional_groups:
            bond_score = 0.6  # Amides are less reactive
        else:
            bond_score = 0.1

        # Environmental friendliness of water-based process
        environmental_score = 0.9

        metric_scores = {
            RecyclabilityMetric.PROCESSING_EASE: structural_score,
            RecyclabilityMetric.MATERIAL_RECOVERY_RATE: 0.8,
            RecyclabilityMetric.QUALITY_RETENTION: 0.85,
            RecyclabilityMetric.ECONOMIC_VIABILITY: 0.7,
            RecyclabilityMetric.ENVIRONMENTAL_IMPACT: environmental_score,
        }

        overall_score = np.mean(list(metric_scores.values()))
        feasibility = overall_score > 0.6 and hydrolyzable_bonds > 0

        barriers = []
        advantages = []

        if hydrolyzable_bonds == 0:
            barriers.append("No hydrolyzable bonds")
        else:
            advantages.append("Contains hydrolyzable bonds")

        advantages.extend(
            [
                "Environmentally friendly process",
                "Water-based system",
                "Mild conditions possible",
            ]
        )

        if bond_score < 0.5:
            barriers.append("Bonds not easily hydrolyzed")

        return RecyclabilityScore(
            method=RecyclingMethod.HYDROLYSIS,
            overall_score=overall_score,
            metric_scores=metric_scores,
            feasibility=feasibility,
            confidence=0.8,
            material_recovery_rate=0.8,
            quality_degradation=0.15,
            technical_barriers=barriers,
            advantages=advantages,
        )

    def _default_assessment(
        self, method: RecyclingMethod, composition: MaterialComposition
    ) -> RecyclabilityScore:
        """Default assessment for other methods."""

        metric_scores = {
            RecyclabilityMetric.PROCESSING_EASE: 0.5,
            RecyclabilityMetric.MATERIAL_RECOVERY_RATE: 0.5,
            RecyclabilityMetric.QUALITY_RETENTION: 0.5,
            RecyclabilityMetric.ECONOMIC_VIABILITY: 0.4,
            RecyclabilityMetric.ENVIRONMENTAL_IMPACT: 0.5,
        }

        overall_score = np.mean(list(metric_scores.values()))

        return RecyclabilityScore(
            method=method,
            overall_score=overall_score,
            metric_scores=metric_scores,
            feasibility=False,
            confidence=0.3,
            technical_barriers=["Method not fully assessed"],
            advantages=[],
        )

    def _calculate_purity_score(self, composition: MaterialComposition) -> float:
        """Calculate material purity score."""

        total_polymer = sum(composition.polymer_content.values())
        total_other = (
            sum(composition.additive_content.values())
            + sum(composition.filler_content.values())
            + sum(composition.contaminants.values())
        )

        total = total_polymer + total_other
        if total == 0:
            return 0.5

        purity = total_polymer / total
        return min(1.0, purity)

    def _determine_material_category(
        self, composition: MaterialComposition
    ) -> MaterialCategory:
        """Determine material category."""

        # Similar to mechanical recycling predictor
        if composition.crosslink_density and composition.crosslink_density > 0.1:
            return MaterialCategory.THERMOSET

        return MaterialCategory.THERMOPLASTIC

    def _calculate_circularity_metrics(
        self,
        composition: MaterialComposition,
        recycling_scores: Dict[RecyclingMethod, RecyclabilityScore],
    ) -> CircularityMetrics:
        """Calculate circularity metrics for chemical recycling."""

        # Average recyclability across methods
        recyclability = np.mean(
            [score.overall_score for score in recycling_scores.values()]
        )

        # Chemical recycling generally produces lower quality materials for reuse
        reusability = 0.4

        # Biodegradability (low for most synthetic polymers)
        biodegradability = 0.1

        # Durability
        durability = (
            0.8
            if composition.crosslink_density and composition.crosslink_density > 0.05
            else 0.6
        )

        circularity_index = (
            recyclability + reusability + biodegradability + durability
        ) / 4

        return CircularityMetrics(
            circularity_index=circularity_index,
            recyclability_score=recyclability,
            reusability_score=reusability,
            biodegradability_score=biodegradability,
            durability_score=durability,
            design_for_recycling_score=0.5,
            material_compatibility_score=0.6,
            economic_viability_score=0.5,
            infrastructure_readiness_score=0.4,  # Chemical recycling infrastructure is limited
            lifecycle_impact_score=0.4,
        )

    def _generate_processing_recommendations(
        self, composition: MaterialComposition
    ) -> Dict[str, Any]:
        """Generate chemical recycling processing recommendations."""

        recommendations = {
            "preprocessing": ["Sorting", "Cleaning"],
            "optimal_methods": [],
            "process_conditions": {},
            "product_applications": [],
        }

        # Method-specific recommendations would be added here

        return recommendations

    def _assess_contamination_sensitivity(
        self, composition: MaterialComposition
    ) -> Dict[str, float]:
        """Assess contamination sensitivity for chemical recycling."""

        return {
            "organic_contaminants": 0.6,  # Can often be processed
            "inorganic_contaminants": 0.3,  # More problematic
            "other_polymers": 0.7,  # Often acceptable
            "additives": 0.8,  # Usually not problematic
            "moisture": 0.5,  # Moderate sensitivity
        }

    def _assess_market_potential(
        self, composition: MaterialComposition
    ) -> Dict[str, float]:
        """Assess market potential for recycled products."""

        return {
            "monomers": 0.9,  # High value
            "oligomers": 0.7,  # Medium value
            "fuels": 0.5,  # Lower value
            "chemicals": 0.8,  # High value
        }


class BiologicalRecyclingPredictor(RecyclabilityPredictor):
    """Predictor for biological recycling feasibility."""

    def __init__(self):
        self.biodegradation_patterns = {
            "ester": 0.9,  # Highly biodegradable
            "amide": 0.6,  # Moderately biodegradable
            "ether": 0.3,  # Poorly biodegradable
            "aromatic": 0.2,  # Very poorly biodegradable
            "aliphatic": 0.7,  # Well biodegradable
        }

    def assess_recyclability(
        self, composition: MaterialComposition
    ) -> RecyclabilityAssessment:
        """Assess biological recyclability."""

        category = (
            MaterialCategory.BIODEGRADABLE
        )  # Assume biodegradable for biological recycling

        # Assess biological recycling methods
        recycling_scores = {}
        for method in [
            RecyclingMethod.BIOLOGICAL_RECYCLING,
            RecyclingMethod.ENZYMATIC_RECYCLING,
        ]:
            score = self.predict_method_feasibility(composition, method)
            recycling_scores[method] = score

        recommended_methods = [
            method for method, score in recycling_scores.items() if score.feasibility
        ]

        circularity_metrics = self._calculate_circularity_metrics(composition)

        return RecyclabilityAssessment(
            material_id=f"bio_{hash(str(composition))}",
            material_composition=composition,
            material_category=category,
            recycling_scores=recycling_scores,
            recommended_methods=recommended_methods,
            circularity_metrics=circularity_metrics,
            processing_recommendations=self._generate_processing_recommendations(
                composition
            ),
            contamination_sensitivity=self._assess_contamination_sensitivity(
                composition
            ),
        )

    def predict_method_feasibility(
        self, composition: MaterialComposition, method: RecyclingMethod
    ) -> RecyclabilityScore:
        """Predict biological recycling feasibility."""

        if method == RecyclingMethod.BIOLOGICAL_RECYCLING:
            return self._assess_general_biodegradation(composition)
        elif method == RecyclingMethod.ENZYMATIC_RECYCLING:
            return self._assess_enzymatic_recycling(composition)
        else:
            return RecyclabilityScore(
                method=method,
                overall_score=0.0,
                metric_scores={},
                feasibility=False,
                confidence=0.0,
            )

    def _assess_general_biodegradation(
        self, composition: MaterialComposition
    ) -> RecyclabilityScore:
        """Assess general biodegradation potential."""

        # Check for biodegradable polymers
        biodegradable_polymers = ["PLA", "PHA", "PBS", "PBAT", "starch", "cellulose"]

        biodegradability_score = 0.0
        for polymer, fraction in composition.polymer_content.items():
            for bio_polymer in biodegradable_polymers:
                if bio_polymer.lower() in polymer.lower():
                    biodegradability_score += fraction * 0.9
                    break
            else:
                # Not a known biodegradable polymer
                biodegradability_score += fraction * 0.1

        # Adjust for additives and fillers
        total_additives = sum(composition.additive_content.values())
        total_fillers = sum(composition.filler_content.values())

        if total_additives > 0.2:
            biodegradability_score *= 0.7
        if total_fillers > 0.3:
            biodegradability_score *= 0.8

        metric_scores = {
            RecyclabilityMetric.PROCESSING_EASE: 0.9,  # Biological processes are gentle
            RecyclabilityMetric.MATERIAL_RECOVERY_RATE: biodegradability_score,
            RecyclabilityMetric.ENVIRONMENTAL_IMPACT: 0.95,  # Very environmentally friendly
            RecyclabilityMetric.ECONOMIC_VIABILITY: 0.6,  # Moderate economics
        }

        overall_score = np.mean(list(metric_scores.values()))
        feasibility = biodegradability_score > 0.6

        barriers = []
        advantages = ["Environmentally friendly", "Complete mineralization possible"]

        if biodegradability_score < 0.6:
            barriers.append("Low biodegradability")

        if total_additives > 0.2:
            barriers.append("High additive content may inhibit biodegradation")

        return RecyclabilityScore(
            method=RecyclingMethod.BIOLOGICAL_RECYCLING,
            overall_score=overall_score,
            metric_scores=metric_scores,
            feasibility=feasibility,
            confidence=0.7,
            material_recovery_rate=biodegradability_score,
            environmental_score=0.95,
            technical_barriers=barriers,
            advantages=advantages,
        )

    def _assess_enzymatic_recycling(
        self, composition: MaterialComposition
    ) -> RecyclabilityScore:
        """Assess enzymatic recycling potential."""

        # Enzymatic recycling works best with specific polymer types
        enzymatically_recyclable = ["PET", "PLA", "PA", "PU"]

        enzyme_compatibility = 0.0
        for polymer, fraction in composition.polymer_content.items():
            for recyclable in enzymatically_recyclable:
                if recyclable in polymer:
                    enzyme_compatibility += fraction * 0.9
                    break
            else:
                enzyme_compatibility += fraction * 0.2

        selectivity_score = 0.9  # Enzymes are highly selective
        mild_conditions_score = 0.95  # Enzymes work under mild conditions

        metric_scores = {
            RecyclabilityMetric.PROCESSING_EASE: mild_conditions_score,
            RecyclabilityMetric.MATERIAL_RECOVERY_RATE: enzyme_compatibility,
            RecyclabilityMetric.QUALITY_RETENTION: 0.9,  # High quality products
            RecyclabilityMetric.ENVIRONMENTAL_IMPACT: 0.9,
            RecyclabilityMetric.ECONOMIC_VIABILITY: 0.5,  # Enzyme costs are high
        }

        overall_score = np.mean(list(metric_scores.values()))
        feasibility = enzyme_compatibility > 0.6

        advantages = [
            "Highly selective process",
            "Mild reaction conditions",
            "High quality products",
            "Environmentally friendly",
        ]

        barriers = [
            "High enzyme costs",
            "Limited to specific polymers",
            "Requires pure feedstock",
        ]

        if enzyme_compatibility < 0.6:
            barriers.append("Polymer not compatible with available enzymes")

        return RecyclabilityScore(
            method=RecyclingMethod.ENZYMATIC_RECYCLING,
            overall_score=overall_score,
            metric_scores=metric_scores,
            feasibility=feasibility,
            confidence=0.8,
            material_recovery_rate=enzyme_compatibility,
            quality_degradation=0.1,
            technical_barriers=barriers,
            advantages=advantages,
        )

    def _calculate_circularity_metrics(
        self, composition: MaterialComposition
    ) -> CircularityMetrics:
        """Calculate circularity metrics for biological recycling."""

        # Biological recycling provides excellent end-of-life solution
        biodegradability = 0.9

        # But typically doesn't enable direct reuse
        reusability = 0.2
        recyclability = 0.8  # Through biological pathways

        circularity_index = (biodegradability + reusability + recyclability) / 3

        return CircularityMetrics(
            circularity_index=circularity_index,
            recyclability_score=recyclability,
            reusability_score=reusability,
            biodegradability_score=biodegradability,
            durability_score=0.6,  # Biodegradable materials may be less durable
            design_for_recycling_score=0.8,
            material_compatibility_score=0.7,
            economic_viability_score=0.6,
            infrastructure_readiness_score=0.5,
            lifecycle_impact_score=0.1,  # Very low impact
        )

    def _generate_processing_recommendations(
        self, composition: MaterialComposition
    ) -> Dict[str, Any]:
        """Generate biological recycling recommendations."""

        return {
            "pretreatment": ["Remove non-biodegradable components"],
            "optimal_conditions": {
                "temperature": "25-60°C",
                "pH": "6-8",
                "moisture": "50-70%",
            },
            "composting_guidelines": [
                "Ensure adequate aeration",
                "Monitor temperature",
                "Turn regularly",
            ],
        }

    def _assess_contamination_sensitivity(
        self, composition: MaterialComposition
    ) -> Dict[str, float]:
        """Assess contamination sensitivity for biological recycling."""

        return {
            "heavy_metals": 0.1,  # Very sensitive
            "pesticides": 0.2,  # Very sensitive
            "other_polymers": 0.3,  # Sensitive
            "food_waste": 0.9,  # Not sensitive
            "paper": 0.8,  # Not very sensitive
        }


class ComprehensiveRecyclabilityAssessor:
    """Comprehensive recyclability assessment using multiple approaches."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu

        # Initialize different recycling predictors
        self.predictors = {
            "mechanical": MechanicalRecyclingPredictor(),
            "chemical": ChemicalRecyclingPredictor(use_gpu),
            "biological": BiologicalRecyclingPredictor(),
        }

        self.ensemble_weights = {"mechanical": 0.4, "chemical": 0.4, "biological": 0.2}

    async def assess_comprehensive_recyclability(
        self, composition: MaterialComposition
    ) -> RecyclabilityAssessment:
        """Perform comprehensive recyclability assessment."""

        # Run assessments in parallel
        tasks = []
        for name, predictor in self.predictors.items():
            task = asyncio.create_task(
                self._run_assessment(predictor, composition, name)
            )
            tasks.append((name, task))

        # Collect results
        assessments = {}
        for name, task in tasks:
            try:
                assessment = await task
                assessments[name] = assessment
            except Exception as e:
                logger.error(f"Error in {name} assessment: {e}")

        # Combine assessments
        combined_assessment = self._combine_assessments(assessments, composition)

        return combined_assessment

    async def _run_assessment(
        self,
        predictor: RecyclabilityPredictor,
        composition: MaterialComposition,
        name: str,
    ) -> RecyclabilityAssessment:
        """Run assessment for a specific predictor."""

        # Run in executor for CPU-bound operations
        loop = asyncio.get_event_loop()
        assessment = await loop.run_in_executor(
            None, predictor.assess_recyclability, composition
        )

        return assessment

    def _combine_assessments(
        self,
        assessments: Dict[str, RecyclabilityAssessment],
        composition: MaterialComposition,
    ) -> RecyclabilityAssessment:
        """Combine multiple assessments into comprehensive result."""

        if not assessments:
            raise ValueError("No successful assessments to combine")

        # Combine recycling scores
        all_recycling_scores = {}
        for assessment in assessments.values():
            all_recycling_scores.update(assessment.recycling_scores)

        # Find best methods across all approaches
        recommended_methods = []
        for assessment in assessments.values():
            recommended_methods.extend(assessment.recommended_methods)

        # Remove duplicates and sort by score
        unique_methods = list(set(recommended_methods))

        # Calculate weighted circularity metrics
        combined_circularity = self._combine_circularity_metrics(assessments)

        # Combine processing recommendations
        combined_recommendations = self._combine_processing_recommendations(assessments)

        return RecyclabilityAssessment(
            material_id=f"comprehensive_{hash(str(composition))}",
            material_composition=composition,
            material_category=MaterialCategory.COMPOSITE,  # Conservative assumption
            recycling_scores=all_recycling_scores,
            recommended_methods=unique_methods,
            circularity_metrics=combined_circularity,
            processing_recommendations=combined_recommendations,
            contamination_sensitivity=self._combine_contamination_sensitivity(
                assessments
            ),
            market_potential={"comprehensive_assessment": 0.7},
        )

    def _combine_circularity_metrics(
        self, assessments: Dict[str, RecyclabilityAssessment]
    ) -> CircularityMetrics:
        """Combine circularity metrics using weighted average."""

        weighted_metrics = {}
        total_weight = 0

        for name, assessment in assessments.items():
            weight = self.ensemble_weights.get(name, 0.33)
            total_weight += weight

            for attr in [
                "circularity_index",
                "recyclability_score",
                "reusability_score",
                "biodegradability_score",
                "durability_score",
                "design_for_recycling_score",
                "material_compatibility_score",
                "economic_viability_score",
                "infrastructure_readiness_score",
                "lifecycle_impact_score",
            ]:
                value = getattr(assessment.circularity_metrics, attr)
                if attr not in weighted_metrics:
                    weighted_metrics[attr] = 0
                weighted_metrics[attr] += weight * value

        # Normalize by total weight
        if total_weight > 0:
            for attr in weighted_metrics:
                weighted_metrics[attr] /= total_weight

        return CircularityMetrics(**weighted_metrics)

    def _combine_processing_recommendations(
        self, assessments: Dict[str, RecyclabilityAssessment]
    ) -> Dict[str, Any]:
        """Combine processing recommendations."""

        combined = {
            "mechanical_preprocessing": [],
            "chemical_processing": [],
            "biological_processing": [],
            "general_recommendations": [],
        }

        for assessment in assessments.values():
            for key, value in assessment.processing_recommendations.items():
                if key not in combined:
                    combined[key] = []
                if isinstance(value, list):
                    combined[key].extend(value)
                else:
                    combined[key].append(str(value))

        return combined

    def _combine_contamination_sensitivity(
        self, assessments: Dict[str, RecyclabilityAssessment]
    ) -> Dict[str, float]:
        """Combine contamination sensitivity assessments."""

        all_contaminants = set()
        for assessment in assessments.values():
            all_contaminants.update(assessment.contamination_sensitivity.keys())

        combined_sensitivity = {}
        for contaminant in all_contaminants:
            sensitivities = []
            for assessment in assessments.values():
                if contaminant in assessment.contamination_sensitivity:
                    sensitivities.append(
                        assessment.contamination_sensitivity[contaminant]
                    )

            if sensitivities:
                # Use most conservative (lowest) sensitivity
                combined_sensitivity[contaminant] = min(sensitivities)

        return combined_sensitivity


# Utility functions
def create_standard_plastic_composition(
    plastic_type: str = "PET",
) -> MaterialComposition:
    """Create standard plastic composition for testing."""

    compositions = {
        "PET": MaterialComposition(
            polymer_content={"PET": 0.95},
            additive_content={"stabilizer": 0.03, "colorant": 0.01},
            filler_content={"glass_fiber": 0.01},
            molecular_weight_distribution=(25000, 50000),
            crystallinity=0.3,
            glass_transition_temp=343.15,  # 70°C
            melting_point=533.15,  # 260°C
            degradation_temp=573.15,  # 300°C
            chemical_structure="CC(=O)OCC",  # Simplified
        ),
        "PE": MaterialComposition(
            polymer_content={"HDPE": 0.98},
            additive_content={"antioxidant": 0.015, "slip_agent": 0.005},
            filler_content={},
            molecular_weight_distribution=(100000, 200000),
            crystallinity=0.7,
            melting_point=403.15,  # 130°C
            degradation_temp=623.15,  # 350°C
            chemical_structure="CCCCCC",  # Simplified
        ),
        "PP": MaterialComposition(
            polymer_content={"PP": 0.97},
            additive_content={"stabilizer": 0.02, "nucleating_agent": 0.01},
            filler_content={},
            molecular_weight_distribution=(150000, 300000),
            crystallinity=0.6,
            melting_point=433.15,  # 160°C
            degradation_temp=633.15,  # 360°C
            chemical_structure="CC(C)CCC",  # Simplified
        ),
    }

    return compositions.get(plastic_type, compositions["PET"])


def create_biodegradable_composition(polymer_type: str = "PLA") -> MaterialComposition:
    """Create biodegradable polymer composition."""

    if polymer_type == "PLA":
        return MaterialComposition(
            polymer_content={"PLA": 0.98},
            additive_content={"plasticizer": 0.02},
            filler_content={},
            molecular_weight_distribution=(50000, 150000),
            crystallinity=0.2,
            glass_transition_temp=333.15,  # 60°C
            melting_point=448.15,  # 175°C
            degradation_temp=473.15,  # 200°C
            chemical_structure="CC(=O)OC(C)C(=O)O",  # Simplified
        )
    elif polymer_type == "PHA":
        return MaterialComposition(
            polymer_content={"PHA": 0.99},
            additive_content={"processing_aid": 0.01},
            filler_content={},
            molecular_weight_distribution=(200000, 800000),
            crystallinity=0.6,
            melting_point=448.15,  # 175°C
            degradation_temp=523.15,  # 250°C
            chemical_structure="CC(C)C(=O)O",  # Simplified
        )
    else:
        raise ValueError(f"Unknown biodegradable polymer: {polymer_type}")


async def assess_material_recyclability(
    composition: MaterialComposition,
    methods: Optional[List[str]] = None,
    use_gpu: bool = True,
) -> RecyclabilityAssessment:
    """Assess material recyclability using specified methods."""

    assessor = ComprehensiveRecyclabilityAssessor(use_gpu=use_gpu)

    if methods:
        # Filter predictors based on requested methods
        filtered_predictors = {}
        for method in methods:
            if method in assessor.predictors:
                filtered_predictors[method] = assessor.predictors[method]
        assessor.predictors = filtered_predictors

    assessment = await assessor.assess_comprehensive_recyclability(composition)

    return assessment


# Exception classes
class RecyclabilityAssessmentError(Exception):
    """Base exception for recyclability assessment."""

    pass


class InvalidCompositionError(RecyclabilityAssessmentError):
    """Exception raised for invalid material composition."""

    pass


class UnsupportedRecyclingMethodError(RecyclabilityAssessmentError):
    """Exception raised for unsupported recycling method."""

    pass

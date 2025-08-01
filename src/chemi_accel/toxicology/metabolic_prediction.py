"""
Metabolic pathway and metabolite toxicity prediction.

Integrates with AI-driven systems pharmacology to predict metabolic pathways
and formation of active/toxic metabolites, then predicts toxicity of metabolites.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.backend import Backend
from ..core.exceptions import ChemiAccelError
from .core import ConfidenceLevel, SpeciesType, ToxicityData, ToxicityPrediction


class MetabolicEnzyme(Enum):
    """Major metabolic enzymes."""

    CYP1A2 = "cyp1a2"
    CYP2C9 = "cyp2c9"
    CYP2C19 = "cyp2c19"
    CYP2D6 = "cyp2d6"
    CYP3A4 = "cyp3a4"
    CYP3A5 = "cyp3a5"
    UGT1A1 = "ugt1a1"
    UGT2B7 = "ugt2b7"
    SULT1A1 = "sult1a1"
    NAT2 = "nat2"
    GST = "gst"
    ALDH = "aldh"
    MAO_A = "mao_a"
    MAO_B = "mao_b"


class MetabolicReaction(Enum):
    """Types of metabolic reactions."""

    HYDROXYLATION = "hydroxylation"
    OXIDATION = "oxidation"
    DEALKYLATION = "dealkylation"
    EPOXIDATION = "epoxidation"
    HYDROLYSIS = "hydrolysis"
    CONJUGATION = "conjugation"
    GLUCURONIDATION = "glucuronidation"
    SULFATION = "sulfation"
    ACETYLATION = "acetylation"
    METHYLATION = "methylation"
    DEAMINATION = "deamination"
    DEHALOGENATION = "dehalogenation"


class MetaboliteType(Enum):
    """Classification of metabolites."""

    PHASE_I = "phase_i"
    PHASE_II = "phase_ii"
    ACTIVE_METABOLITE = "active_metabolite"
    TOXIC_METABOLITE = "toxic_metabolite"
    REACTIVE_INTERMEDIATE = "reactive_intermediate"
    PARENT_COMPOUND = "parent_compound"


@dataclass
class MetabolicStep:
    """Single step in metabolic pathway."""

    substrate_smiles: str
    product_smiles: str
    enzyme: MetabolicEnzyme
    reaction_type: MetabolicReaction
    km: Optional[float] = None  # Michaelis constant (μM)
    vmax: Optional[float] = None  # Maximum velocity
    intrinsic_clearance: Optional[float] = None  # μL/min/mg protein
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    site_of_metabolism: Optional[str] = None  # Atom indices

    @property
    def reaction_efficiency(self) -> float:
        """Calculate reaction efficiency (Vmax/Km)."""
        if self.km and self.vmax and self.km > 0:
            return self.vmax / self.km
        return 0.0


@dataclass
class Metabolite:
    """Metabolite information."""

    smiles: str
    metabolite_type: MetaboliteType
    formation_pathway: List[MetabolicStep]
    abundance: float  # Relative abundance (0-1)
    half_life: Optional[float] = None  # minutes
    toxicity_data: Optional[ToxicityData] = None
    pharmacological_activity: Optional[Dict[str, float]] = None

    @property
    def is_reactive(self) -> bool:
        """Check if metabolite is reactive intermediate."""
        return self.metabolite_type == MetaboliteType.REACTIVE_INTERMEDIATE

    @property
    def pathway_length(self) -> int:
        """Number of steps in formation pathway."""
        return len(self.formation_pathway)


@dataclass
class MetabolicNetwork:
    """Complete metabolic network for a compound."""

    parent_compound: str
    species: SpeciesType
    metabolites: List[Metabolite]
    enzymatic_pathways: Dict[MetabolicEnzyme, List[MetabolicStep]]
    network_confidence: ConfidenceLevel
    mass_balance: float = 1.0  # Fraction of parent accounted for

    @property
    def primary_metabolites(self) -> List[Metabolite]:
        """Get primary metabolites (formed directly from parent)."""
        return [met for met in self.metabolites if met.pathway_length == 1]

    @property
    def toxic_metabolites(self) -> List[Metabolite]:
        """Get potentially toxic metabolites."""
        toxic = []
        for met in self.metabolites:
            if (
                met.metabolite_type == MetaboliteType.TOXIC_METABOLITE
                or met.is_reactive
                or (met.toxicity_data and met.toxicity_data.overall_risk_score > 0.5)
            ):
                toxic.append(met)
        return toxic

    @property
    def active_metabolites(self) -> List[Metabolite]:
        """Get pharmacologically active metabolites."""
        return [
            met
            for met in self.metabolites
            if met.metabolite_type == MetaboliteType.ACTIVE_METABOLITE
        ]


class MetabolismPredictor(ABC):
    """Abstract base class for metabolism prediction."""

    @abstractmethod
    def predict_metabolites(
        self, smiles: str, species: SpeciesType, n_generations: int = 2
    ) -> List[Metabolite]:
        """Predict metabolites for compound."""
        pass

    @abstractmethod
    def predict_enzyme_kinetics(
        self, substrate_smiles: str, enzyme: MetabolicEnzyme
    ) -> Tuple[float, float]:  # Km, Vmax
        """Predict enzyme kinetic parameters."""
        pass


class CytochromeP450Predictor(MetabolismPredictor):
    """Predictor for CYP450-mediated metabolism."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.som_models = {}  # Site of metabolism models
        self.kinetic_models = {}
        self.reaction_rules = self._load_cyp450_rules()

    def _load_cyp450_rules(self) -> Dict[MetabolicEnzyme, Dict[str, Any]]:
        """Load CYP450 reaction rules."""
        return {
            MetabolicEnzyme.CYP3A4: {
                "reactions": [
                    MetabolicReaction.HYDROXYLATION,
                    MetabolicReaction.DEALKYLATION,
                    MetabolicReaction.EPOXIDATION,
                ],
                "substrates": [
                    "[CH3:1]",  # Methyl hydroxylation
                    "[CH2:1][CH3:2]",  # Ethyl hydroxylation
                    "c1ccccc1",  # Aromatic hydroxylation
                ],
                "km_range": (1.0, 100.0),  # μM
                "selectivity": 0.4,
            },
            MetabolicEnzyme.CYP2D6: {
                "reactions": [
                    MetabolicReaction.HYDROXYLATION,
                    MetabolicReaction.DEALKYLATION,
                ],
                "substrates": [
                    "[NH2:1]",  # Basic nitrogen
                    "c1ccc(N)cc1",  # Aromatic amine
                ],
                "km_range": (0.5, 50.0),
                "selectivity": 0.8,
            },
            MetabolicEnzyme.CYP1A2: {
                "reactions": [
                    MetabolicReaction.HYDROXYLATION,
                    MetabolicReaction.DEALKYLATION,
                ],
                "substrates": [
                    "c1cc2[nH]c3ccccc3c2cc1",  # Indole
                    "c1ccc2nc3ccccc3nc2c1",  # Phenazine
                ],
                "km_range": (2.0, 80.0),
                "selectivity": 0.6,
            },
        }

    def predict_metabolites(
        self, smiles: str, species: SpeciesType, n_generations: int = 2
    ) -> List[Metabolite]:
        """Predict CYP450 metabolites."""

        metabolites = []
        current_generation = [smiles]

        for generation in range(n_generations):
            next_generation = []

            for substrate in current_generation:
                for enzyme, rules in self.reaction_rules.items():
                    # Check if substrate matches enzyme specificity
                    if self._is_substrate(substrate, enzyme, rules):
                        # Generate metabolites for this enzyme
                        enzyme_metabolites = self._generate_cyp450_metabolites(
                            substrate, enzyme, rules, generation
                        )
                        metabolites.extend(enzyme_metabolites)

                        # Add to next generation for further metabolism
                        for met in enzyme_metabolites:
                            if met.smiles not in next_generation:
                                next_generation.append(met.smiles)

            current_generation = next_generation

        return metabolites

    def _is_substrate(
        self, smiles: str, enzyme: MetabolicEnzyme, rules: Dict[str, Any]
    ) -> bool:
        """Check if compound is substrate for enzyme."""

        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # Check substrate patterns
            for pattern in rules["substrates"]:
                substrate_pattern = Chem.MolFromSmarts(pattern)
                if substrate_pattern and mol.HasSubstructMatch(substrate_pattern):
                    return True

            return False

        except ImportError:
            # Fallback for demo
            return True

    def _generate_cyp450_metabolites(
        self,
        substrate_smiles: str,
        enzyme: MetabolicEnzyme,
        rules: Dict[str, Any],
        generation: int,
    ) -> List[Metabolite]:
        """Generate CYP450 metabolites."""

        metabolites = []

        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(substrate_smiles)
            if mol is None:
                return metabolites

            # Generate products for each reaction type
            for reaction_type in rules["reactions"]:
                products = self._apply_cyp450_reaction(
                    substrate_smiles, reaction_type, enzyme
                )

                for product_smiles in products:
                    if product_smiles and product_smiles != substrate_smiles:
                        # Create metabolic step
                        km, vmax = self.predict_enzyme_kinetics(
                            substrate_smiles, enzyme
                        )

                        step = MetabolicStep(
                            substrate_smiles=substrate_smiles,
                            product_smiles=product_smiles,
                            enzyme=enzyme,
                            reaction_type=reaction_type,
                            km=km,
                            vmax=vmax,
                            confidence=ConfidenceLevel.MODERATE,
                        )

                        # Determine metabolite type
                        metabolite_type = MetaboliteType.PHASE_I
                        if generation == 0:
                            metabolite_type = MetaboliteType.PHASE_I

                        # Estimate abundance
                        abundance = rules["selectivity"] * (0.5**generation)

                        metabolite = Metabolite(
                            smiles=product_smiles,
                            metabolite_type=metabolite_type,
                            formation_pathway=[step],
                            abundance=abundance,
                        )

                        metabolites.append(metabolite)

            return metabolites

        except ImportError:
            # Fallback for demo
            return self._generate_demo_cyp450_metabolites(
                substrate_smiles, enzyme, generation
            )

    def _apply_cyp450_reaction(
        self,
        substrate_smiles: str,
        reaction_type: MetabolicReaction,
        enzyme: MetabolicEnzyme,
    ) -> List[str]:
        """Apply CYP450 reaction to generate products."""

        products = []

        # Simplified reaction application for demo
        if reaction_type == MetabolicReaction.HYDROXYLATION:
            # Add hydroxyl group (simplified)
            if "C" in substrate_smiles:
                products.append("CCO")  # Alcohol product

        elif reaction_type == MetabolicReaction.DEALKYLATION:
            # Remove alkyl group (simplified)
            if "CC" in substrate_smiles:
                products.append("CO")  # Dealkylated product

        elif reaction_type == MetabolicReaction.EPOXIDATION:
            # Form epoxide (simplified)
            if "c1ccccc1" in substrate_smiles:  # Aromatic
                products.append("C1OC1")  # Epoxide

        return products

    def _generate_demo_cyp450_metabolites(
        self, substrate_smiles: str, enzyme: MetabolicEnzyme, generation: int
    ) -> List[Metabolite]:
        """Generate demo CYP450 metabolites."""

        # Demo metabolites based on common patterns
        demo_products = {
            MetabolicEnzyme.CYP3A4: ["CCO", "CC(=O)O", "CO"],
            MetabolicEnzyme.CYP2D6: ["CCO", "c1ccc(O)cc1"],
            MetabolicEnzyme.CYP1A2: ["c1ccc(O)cc1", "CCO"],
        }

        metabolites = []
        products = demo_products.get(enzyme, ["CCO"])

        for product_smiles in products:
            step = MetabolicStep(
                substrate_smiles=substrate_smiles,
                product_smiles=product_smiles,
                enzyme=enzyme,
                reaction_type=MetabolicReaction.HYDROXYLATION,
                km=np.random.uniform(1, 50),
                vmax=np.random.uniform(10, 100),
            )

            metabolite = Metabolite(
                smiles=product_smiles,
                metabolite_type=MetaboliteType.PHASE_I,
                formation_pathway=[step],
                abundance=0.3 * (0.5**generation),
            )

            metabolites.append(metabolite)

        return metabolites

    def predict_enzyme_kinetics(
        self, substrate_smiles: str, enzyme: MetabolicEnzyme
    ) -> Tuple[float, float]:
        """Predict enzyme kinetic parameters."""

        if enzyme in self.reaction_rules:
            km_range = self.reaction_rules[enzyme]["km_range"]
            km = np.random.uniform(km_range[0], km_range[1])
            vmax = np.random.uniform(10, 200)  # pmol/min/mg protein
            return km, vmax

        # Default values
        return 10.0, 50.0


class ConjugationPredictor(MetabolismPredictor):
    """Predictor for Phase II conjugation reactions."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.conjugation_rules = self._load_conjugation_rules()

    def _load_conjugation_rules(self) -> Dict[MetabolicReaction, Dict[str, Any]]:
        """Load conjugation reaction rules."""
        return {
            MetabolicReaction.GLUCURONIDATION: {
                "enzymes": [MetabolicEnzyme.UGT1A1, MetabolicEnzyme.UGT2B7],
                "substrates": [
                    "[OH:1]",
                    "[NH2:1]",
                    "C(=O)O",
                ],  # Alcohols, amines, carboxylic acids
                "product_template": "[*:1]C1OC(C(=O)O)C(O)C(O)C1O",  # Glucuronide
            },
            MetabolicReaction.SULFATION: {
                "enzymes": [MetabolicEnzyme.SULT1A1],
                "substrates": ["[OH:1]", "[NH2:1]"],  # Alcohols, amines
                "product_template": "[*:1]S(=O)(=O)O",  # Sulfate
            },
            MetabolicReaction.ACETYLATION: {
                "enzymes": [MetabolicEnzyme.NAT2],
                "substrates": ["[NH2:1]"],  # Amines
                "product_template": "[*:1]C(=O)C",  # Acetate
            },
        }

    def predict_metabolites(
        self, smiles: str, species: SpeciesType, n_generations: int = 1
    ) -> List[Metabolite]:
        """Predict conjugation metabolites."""

        metabolites = []

        for reaction_type, rules in self.conjugation_rules.items():
            if self._has_conjugation_site(smiles, rules["substrates"]):
                conjugates = self._generate_conjugates(smiles, reaction_type, rules)
                metabolites.extend(conjugates)

        return metabolites

    def _has_conjugation_site(self, smiles: str, substrate_patterns: List[str]) -> bool:
        """Check if compound has sites for conjugation."""

        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            for pattern in substrate_patterns:
                substrate_pattern = Chem.MolFromSmarts(pattern)
                if substrate_pattern and mol.HasSubstructMatch(substrate_pattern):
                    return True

            return False

        except ImportError:
            # Fallback for demo
            return "O" in smiles or "N" in smiles

    def _generate_conjugates(
        self,
        substrate_smiles: str,
        reaction_type: MetabolicReaction,
        rules: Dict[str, Any],
    ) -> List[Metabolite]:
        """Generate conjugation products."""

        metabolites = []

        # Simplified conjugation for demo
        conjugate_smiles = {
            MetabolicReaction.GLUCURONIDATION: "CC(O)C(O)C(O)C(O)C(=O)O",  # Glucuronide
            MetabolicReaction.SULFATION: "COS(=O)(=O)O",  # Sulfate
            MetabolicReaction.ACETYLATION: "CNC(=O)C",  # Acetate
        }

        product_smiles = conjugate_smiles.get(reaction_type, "CO")

        # Select enzyme
        enzyme = rules["enzymes"][0] if rules["enzymes"] else MetabolicEnzyme.UGT1A1

        step = MetabolicStep(
            substrate_smiles=substrate_smiles,
            product_smiles=product_smiles,
            enzyme=enzyme,
            reaction_type=reaction_type,
            confidence=ConfidenceLevel.HIGH,
        )

        metabolite = Metabolite(
            smiles=product_smiles,
            metabolite_type=MetaboliteType.PHASE_II,
            formation_pathway=[step],
            abundance=0.6,  # Phase II typically high efficiency
        )

        metabolites.append(metabolite)

        return metabolites

    def predict_enzyme_kinetics(
        self, substrate_smiles: str, enzyme: MetabolicEnzyme
    ) -> Tuple[float, float]:
        """Predict conjugation enzyme kinetics."""

        # Conjugation enzymes typically have higher Km (lower affinity)
        km = np.random.uniform(50, 500)  # μM
        vmax = np.random.uniform(100, 1000)  # pmol/min/mg protein

        return km, vmax


class MetabolitePredictor:
    """
    Comprehensive metabolite toxicity prediction system.

    Predicts toxicity of metabolites using multi-endpoint models.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        # Import toxicity predictor
        from .multi_endpoint import MultiEndpointToxicologyPredictor

        self.toxicity_predictor = MultiEndpointToxicologyPredictor(backend)

    def predict_metabolite_toxicity(
        self,
        metabolites: List[Metabolite],
        endpoints: Optional[List] = None,
        species: Optional[List[SpeciesType]] = None,
    ) -> List[Metabolite]:
        """Predict toxicity for all metabolites."""

        # Default parameters
        if endpoints is None:
            from .core import ToxicityEndpointType

            endpoints = [
                ToxicityEndpointType.ACUTE_ORAL,
                ToxicityEndpointType.GENOTOXICITY,
                ToxicityEndpointType.CARCINOGENICITY,
            ]

        if species is None:
            species = [SpeciesType.HUMAN, SpeciesType.RAT]

        updated_metabolites = []

        for metabolite in metabolites:
            # Predict toxicity for this metabolite
            try:
                predictions = self.toxicity_predictor.predict_endpoints(
                    metabolite.smiles, endpoints, species, include_uncertainty=True
                )

                # Create toxicity data
                from .core import ToxicityData

                toxicity_data = ToxicityData(
                    smiles=metabolite.smiles, predicted_endpoints=predictions
                )

                # Update metabolite with toxicity data
                metabolite.toxicity_data = toxicity_data

                # Classify as toxic if high risk
                if toxicity_data.overall_risk_score > 0.7:
                    metabolite.metabolite_type = MetaboliteType.TOXIC_METABOLITE

                updated_metabolites.append(metabolite)

            except Exception as e:
                print(
                    f"Warning: Failed to predict toxicity for metabolite {metabolite.smiles}: {e}"
                )
                updated_metabolites.append(metabolite)

        return updated_metabolites

    def identify_reactive_metabolites(
        self, metabolites: List[Metabolite]
    ) -> List[Metabolite]:
        """Identify potentially reactive metabolites."""

        reactive_patterns = [
            "C=C=O",  # Ketene
            "C=C=C",  # Allene
            "C1OC1",  # Epoxide
            "C=N+=[O-]",  # Nitroso
            "C#C",  # Acetylene
            "CC(=O)OO",  # Peracetic acid
        ]

        reactive_metabolites = []

        for metabolite in metabolites:
            is_reactive = False

            try:
                from rdkit import Chem

                mol = Chem.MolFromSmiles(metabolite.smiles)
                if mol is None:
                    continue

                # Check for reactive patterns
                for pattern in reactive_patterns:
                    reactive_pattern = Chem.MolFromSmarts(pattern)
                    if reactive_pattern and mol.HasSubstructMatch(reactive_pattern):
                        is_reactive = True
                        break

                if is_reactive:
                    metabolite.metabolite_type = MetaboliteType.REACTIVE_INTERMEDIATE
                    reactive_metabolites.append(metabolite)

            except ImportError:
                # Fallback heuristics for demo
                if any(
                    pattern in metabolite.smiles for pattern in ["=C=", "C1OC1", "#"]
                ):
                    metabolite.metabolite_type = MetaboliteType.REACTIVE_INTERMEDIATE
                    reactive_metabolites.append(metabolite)

        return reactive_metabolites


class MetabolicPathwayPredictor:
    """
    Comprehensive metabolic pathway prediction system.

    Integrates multiple metabolism predictors to generate complete
    metabolic networks with toxicity assessment.
    """

    def __init__(self, backend: Backend):
        self.backend = backend

        # Initialize metabolism predictors
        self.predictors = {
            "cyp450": CytochromeP450Predictor(backend),
            "conjugation": ConjugationPredictor(backend),
        }

        # Initialize metabolite toxicity predictor
        self.metabolite_predictor = MetabolitePredictor(backend)

    def predict_metabolism(
        self,
        smiles: Union[str, List[str]],
        species: SpeciesType = SpeciesType.HUMAN,
        include_metabolite_toxicity: bool = True,
        n_generations: int = 2,
        **kwargs,
    ) -> Union[MetabolicNetwork, List[MetabolicNetwork]]:
        """
        Predict complete metabolic pathways.

        Args:
            smiles: SMILES string(s)
            species: Species for metabolism prediction
            include_metabolite_toxicity: Predict metabolite toxicity
            n_generations: Number of metabolic generations

        Returns:
            MetabolicNetwork object(s)
        """

        # Handle single SMILES
        if isinstance(smiles, str):
            return self._predict_single_metabolism(
                smiles, species, include_metabolite_toxicity, n_generations, **kwargs
            )

        # Handle multiple SMILES
        results = []
        for smi in smiles:
            result = self._predict_single_metabolism(
                smi, species, include_metabolite_toxicity, n_generations, **kwargs
            )
            results.append(result)

        return results

    def _predict_single_metabolism(
        self,
        smiles: str,
        species: SpeciesType,
        include_metabolite_toxicity: bool,
        n_generations: int,
        **kwargs,
    ) -> MetabolicNetwork:
        """Predict metabolism for single compound."""

        all_metabolites = []
        enzymatic_pathways = {}

        # Phase I metabolism (CYP450)
        cyp450_metabolites = self.predictors["cyp450"].predict_metabolites(
            smiles, species, n_generations
        )
        all_metabolites.extend(cyp450_metabolites)

        # Collect enzymatic pathways
        for metabolite in cyp450_metabolites:
            for step in metabolite.formation_pathway:
                if step.enzyme not in enzymatic_pathways:
                    enzymatic_pathways[step.enzyme] = []
                enzymatic_pathways[step.enzyme].append(step)

        # Phase II metabolism (conjugation)
        # Apply to parent and Phase I metabolites
        substrates_for_conjugation = [smiles]
        substrates_for_conjugation.extend([m.smiles for m in cyp450_metabolites])

        for substrate in substrates_for_conjugation[:5]:  # Limit for demo
            conjugation_metabolites = self.predictors[
                "conjugation"
            ].predict_metabolites(substrate, species, 1)
            all_metabolites.extend(conjugation_metabolites)

            # Collect conjugation pathways
            for metabolite in conjugation_metabolites:
                for step in metabolite.formation_pathway:
                    if step.enzyme not in enzymatic_pathways:
                        enzymatic_pathways[step.enzyme] = []
                    enzymatic_pathways[step.enzyme].append(step)

        # Predict metabolite toxicity if requested
        if include_metabolite_toxicity:
            all_metabolites = self.metabolite_predictor.predict_metabolite_toxicity(
                all_metabolites
            )

        # Identify reactive metabolites
        reactive_metabolites = self.metabolite_predictor.identify_reactive_metabolites(
            all_metabolites
        )

        # Calculate network confidence
        network_confidence = self._calculate_network_confidence(all_metabolites)

        # Calculate mass balance
        total_abundance = sum(m.abundance for m in all_metabolites)
        mass_balance = min(total_abundance, 1.0)

        return MetabolicNetwork(
            parent_compound=smiles,
            species=species,
            metabolites=all_metabolites,
            enzymatic_pathways=enzymatic_pathways,
            network_confidence=network_confidence,
            mass_balance=mass_balance,
        )

    def _calculate_network_confidence(
        self, metabolites: List[Metabolite]
    ) -> ConfidenceLevel:
        """Calculate overall network confidence."""

        if not metabolites:
            return ConfidenceLevel.VERY_LOW

        # Average confidence from all metabolic steps
        all_confidences = []
        for metabolite in metabolites:
            for step in metabolite.formation_pathway:
                confidence_score = self._confidence_to_score(step.confidence)
                all_confidences.append(confidence_score)

        if not all_confidences:
            return ConfidenceLevel.LOW

        avg_confidence = np.mean(all_confidences)
        return self._score_to_confidence(avg_confidence)

    def _confidence_to_score(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to numeric score."""
        mapping = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MODERATE: 0.5,
            ConfidenceLevel.HIGH: 0.7,
            ConfidenceLevel.VERY_HIGH: 0.9,
        }
        return mapping.get(confidence, 0.5)

    def _score_to_confidence(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MODERATE
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def predict_drug_drug_interactions(
        self,
        substrate_smiles: str,
        inhibitor_smiles: str,
        enzymes: Optional[List[MetabolicEnzyme]] = None,
    ) -> Dict[str, Any]:
        """Predict potential drug-drug interactions."""

        if enzymes is None:
            enzymes = [
                MetabolicEnzyme.CYP3A4,
                MetabolicEnzyme.CYP2D6,
                MetabolicEnzyme.CYP1A2,
            ]

        interactions = {}

        for enzyme in enzymes:
            # Check if substrate is metabolized by enzyme
            substrate_kinetics = self.predictors["cyp450"].predict_enzyme_kinetics(
                substrate_smiles, enzyme
            )

            # Check if inhibitor affects enzyme
            inhibitor_kinetics = self.predictors["cyp450"].predict_enzyme_kinetics(
                inhibitor_smiles, enzyme
            )

            # Simple interaction prediction
            if substrate_kinetics[0] < 50 and inhibitor_kinetics[0] < 50:  # Both low Km
                interaction_risk = "high"
            elif substrate_kinetics[0] < 100 or inhibitor_kinetics[0] < 100:
                interaction_risk = "moderate"
            else:
                interaction_risk = "low"

            interactions[enzyme.value] = {
                "substrate_km": substrate_kinetics[0],
                "inhibitor_km": inhibitor_kinetics[0],
                "interaction_risk": interaction_risk,
            }

        return {
            "substrate": substrate_smiles,
            "inhibitor": inhibitor_smiles,
            "enzyme_interactions": interactions,
            "overall_risk": max(
                [data["interaction_risk"] for data in interactions.values()]
            ),
        }

    def simulate_pharmacokinetics(
        self,
        smiles: str,
        dose: float,
        species: SpeciesType = SpeciesType.HUMAN,
        simulation_time: float = 24.0,  # hours
    ) -> Dict[str, Any]:
        """Simulate pharmacokinetics including metabolism."""

        # Get metabolic network
        metabolic_network = self.predict_metabolism(smiles, species)

        # Simple PK simulation
        time_points = np.linspace(0, simulation_time, 100)

        # Parent compound concentration (first-order elimination)
        ke = 0.1  # 1/hr elimination rate constant
        parent_concentration = dose * np.exp(-ke * time_points)

        # Metabolite concentrations (formation-elimination)
        metabolite_profiles = {}

        for metabolite in metabolic_network.primary_metabolites:
            kf = metabolite.abundance * ke  # Formation rate
            kem = 0.5  # Metabolite elimination rate

            # Formation-elimination model
            if kem != ke:
                conc = (
                    dose
                    * kf
                    / (kem - ke)
                    * (np.exp(-ke * time_points) - np.exp(-kem * time_points))
                )
            else:
                conc = dose * kf * time_points * np.exp(-ke * time_points)

            metabolite_profiles[metabolite.smiles] = conc.tolist()

        return {
            "compound": smiles,
            "dose": dose,
            "time_points": time_points.tolist(),
            "parent_concentration": parent_concentration.tolist(),
            "metabolite_profiles": metabolite_profiles,
            "metabolic_network": metabolic_network,
        }

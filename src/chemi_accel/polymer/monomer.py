"""
Monomer analysis and processing module.

This module provides functionality for analyzing monomer reactivity,
polymerization potential, stereochemistry, and ring-opening polymerization.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning(
        "RDKit not available. Some monomer analysis features will be limited."
    )

from ..core.exceptions import ChemiAccelError, InvalidMoleculeError
from ..utils.rdkit_integration import parse_smiles_safe


class PolymerizationType(Enum):
    """Types of polymerization mechanisms."""

    ADDITION = "addition"
    CONDENSATION = "condensation"
    RING_OPENING = "ring_opening"
    METATHESIS = "metathesis"
    RADICAL = "radical"
    IONIC = "ionic"
    COORDINATION = "coordination"


class FunctionalGroupType(Enum):
    """Types of functional groups relevant to polymerization."""

    VINYL = "vinyl"
    ACRYLATE = "acrylate"
    METHACRYLATE = "methacrylate"
    STYRENE = "styrene"
    EPOXIDE = "epoxide"
    LACTONE = "lactone"
    LACTAM = "lactam"
    ANHYDRIDE = "anhydride"
    ISOCYANATE = "isocyanate"
    ALCOHOL = "alcohol"
    AMINE = "amine"
    CARBOXYLIC_ACID = "carboxylic_acid"
    ALDEHYDE = "aldehyde"
    KETONE = "ketone"


@dataclass
class FunctionalGroup:
    """Represents a functional group in a monomer."""

    group_type: FunctionalGroupType
    smarts_pattern: str
    reactivity_score: float
    polymerization_types: List[PolymerizationType]
    atom_indices: Optional[List[int]] = None


@dataclass
class ReactivityProfile:
    """Reactivity profile for a monomer."""

    functional_groups: List[FunctionalGroup]
    overall_reactivity: float
    preferred_mechanisms: List[PolymerizationType]
    steric_hindrance: float
    electronic_effects: Dict[str, float]


@dataclass
class PolymerizationPotential:
    """Assessment of polymerization potential."""

    can_polymerize: bool
    polymerization_types: List[PolymerizationType]
    reactivity_score: float
    stability_score: float
    selectivity_factors: Dict[str, float]


@dataclass
class StereochemistryInfo:
    """Stereochemistry information for polymer formation."""

    chiral_centers: List[int]
    tacticity_potential: str  # isotactic, syndiotactic, atactic
    stereoisomers: List[str]
    stereoregularity_score: float


class ReactivityAnalyzer:
    """Analyzes monomer reactivity and functional group identification."""

    def __init__(self):
        """Initialize the reactivity analyzer."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for monomer reactivity analysis")

        # Define functional group patterns and their properties
        self.functional_groups = {
            FunctionalGroupType.VINYL: FunctionalGroup(
                FunctionalGroupType.VINYL,
                "[C]=[C]",
                0.8,
                [PolymerizationType.ADDITION, PolymerizationType.RADICAL],
            ),
            FunctionalGroupType.ACRYLATE: FunctionalGroup(
                FunctionalGroupType.ACRYLATE,
                "[C]=[C]-[C](=O)-[O]",
                0.9,
                [PolymerizationType.ADDITION, PolymerizationType.RADICAL],
            ),
            FunctionalGroupType.METHACRYLATE: FunctionalGroup(
                FunctionalGroupType.METHACRYLATE,
                "[C]=[C]([C])-[C](=O)-[O]",
                0.85,
                [PolymerizationType.ADDITION, PolymerizationType.RADICAL],
            ),
            FunctionalGroupType.STYRENE: FunctionalGroup(
                FunctionalGroupType.STYRENE,
                "[c][C]=[C]",
                0.75,
                [
                    PolymerizationType.ADDITION,
                    PolymerizationType.RADICAL,
                    PolymerizationType.IONIC,
                ],
            ),
            FunctionalGroupType.EPOXIDE: FunctionalGroup(
                FunctionalGroupType.EPOXIDE,
                "[C]1[O][C]1",
                0.95,
                [PolymerizationType.RING_OPENING, PolymerizationType.IONIC],
            ),
            FunctionalGroupType.LACTONE: FunctionalGroup(
                FunctionalGroupType.LACTONE,
                "[C](=O)[O][C]",
                0.7,
                [PolymerizationType.RING_OPENING, PolymerizationType.COORDINATION],
            ),
            FunctionalGroupType.ISOCYANATE: FunctionalGroup(
                FunctionalGroupType.ISOCYANATE,
                "[N]=[C]=[O]",
                0.9,
                [PolymerizationType.CONDENSATION],
            ),
            FunctionalGroupType.ALCOHOL: FunctionalGroup(
                FunctionalGroupType.ALCOHOL,
                "[O][H]",
                0.6,
                [PolymerizationType.CONDENSATION],
            ),
            FunctionalGroupType.AMINE: FunctionalGroup(
                FunctionalGroupType.AMINE,
                "[N]([H])[H]",
                0.65,
                [PolymerizationType.CONDENSATION],
            ),
            FunctionalGroupType.CARBOXYLIC_ACID: FunctionalGroup(
                FunctionalGroupType.CARBOXYLIC_ACID,
                "[C](=O)[O][H]",
                0.7,
                [PolymerizationType.CONDENSATION],
            ),
        }

    def identify_functional_groups(self, smiles: str) -> List[FunctionalGroup]:
        """
        Identify functional groups in a monomer.

        Args:
            smiles: SMILES string of the monomer

        Returns:
            List of identified functional groups
        """
        mol = parse_smiles_safe(smiles)
        if mol is None:
            raise InvalidMoleculeError(
                smiles, "Invalid SMILES for functional group analysis"
            )

        identified_groups = []

        for group_type, group_info in self.functional_groups.items():
            pattern = Chem.MolFromSmarts(group_info.smarts_pattern)
            if pattern is None:
                continue

            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                group = FunctionalGroup(
                    group_type=group_info.group_type,
                    smarts_pattern=group_info.smarts_pattern,
                    reactivity_score=group_info.reactivity_score,
                    polymerization_types=group_info.polymerization_types,
                    atom_indices=list(match),
                )
                identified_groups.append(group)

        return identified_groups

    def calculate_reactivity_profile(self, smiles: str) -> ReactivityProfile:
        """
        Calculate comprehensive reactivity profile for a monomer.

        Args:
            smiles: SMILES string of the monomer

        Returns:
            ReactivityProfile object
        """
        mol = parse_smiles_safe(smiles)
        if mol is None:
            raise InvalidMoleculeError(smiles, "Invalid SMILES for reactivity analysis")

        functional_groups = self.identify_functional_groups(smiles)

        # Calculate overall reactivity
        if functional_groups:
            overall_reactivity = max(
                group.reactivity_score for group in functional_groups
            )
        else:
            overall_reactivity = 0.1  # Low reactivity for no identified groups

        # Determine preferred mechanisms
        all_mechanisms = []
        for group in functional_groups:
            all_mechanisms.extend(group.polymerization_types)
        preferred_mechanisms = list(set(all_mechanisms))

        # Calculate steric hindrance
        steric_hindrance = self._calculate_steric_hindrance(mol)

        # Calculate electronic effects
        electronic_effects = self._calculate_electronic_effects(mol)

        return ReactivityProfile(
            functional_groups=functional_groups,
            overall_reactivity=overall_reactivity,
            preferred_mechanisms=preferred_mechanisms,
            steric_hindrance=steric_hindrance,
            electronic_effects=electronic_effects,
        )

    def _calculate_steric_hindrance(self, mol: Chem.Mol) -> float:
        """Calculate steric hindrance factor."""
        # Simple approximation based on molecular complexity
        num_rotatable_bonds = CalcNumRotatableBonds(mol)
        num_heavy_atoms = mol.GetNumHeavyAtoms()

        # Normalize to 0-1 scale
        steric_factor = min(1.0, (num_rotatable_bonds + num_heavy_atoms * 0.1) / 20.0)
        return steric_factor

    def _calculate_electronic_effects(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate electronic effects relevant to polymerization."""
        effects = {}

        # Calculate basic electronic descriptors
        try:
            effects["logp"] = Descriptors.MolLogP(mol)
            effects["tpsa"] = Descriptors.TPSA(mol)
            effects["num_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
            effects["num_heteroatoms"] = rdMolDescriptors.CalcNumHeteroatoms(mol)
        except Exception as e:
            logging.warning(f"Error calculating electronic effects: {e}")
            effects = {
                "logp": 0.0,
                "tpsa": 0.0,
                "num_aromatic_rings": 0,
                "num_heteroatoms": 0,
            }

        return effects


class PolymerizationPotentialAnalyzer:
    """Analyzes polymerization potential of monomers."""

    def __init__(self):
        """Initialize the polymerization potential analyzer."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError(
                "RDKit is required for polymerization potential analysis"
            )

        self.reactivity_analyzer = ReactivityAnalyzer()

    def assess_polymerization_potential(self, smiles: str) -> PolymerizationPotential:
        """
        Assess the polymerization potential of a monomer.

        Args:
            smiles: SMILES string of the monomer

        Returns:
            PolymerizationPotential assessment
        """
        mol = parse_smiles_safe(smiles)
        if mol is None:
            raise InvalidMoleculeError(
                smiles, "Invalid SMILES for polymerization assessment"
            )

        reactivity_profile = self.reactivity_analyzer.calculate_reactivity_profile(
            smiles
        )

        # Determine if molecule can polymerize
        can_polymerize = len(reactivity_profile.functional_groups) > 0

        # Get polymerization types
        polymerization_types = reactivity_profile.preferred_mechanisms

        # Calculate reactivity score
        reactivity_score = reactivity_profile.overall_reactivity

        # Calculate stability score (inverse of reactivity with adjustments)
        stability_score = self._calculate_stability_score(mol, reactivity_profile)

        # Calculate selectivity factors
        selectivity_factors = self._calculate_selectivity_factors(
            mol, reactivity_profile
        )

        return PolymerizationPotential(
            can_polymerize=can_polymerize,
            polymerization_types=polymerization_types,
            reactivity_score=reactivity_score,
            stability_score=stability_score,
            selectivity_factors=selectivity_factors,
        )

    def _calculate_stability_score(
        self, mol: Chem.Mol, profile: ReactivityProfile
    ) -> float:
        """Calculate molecular stability score."""
        # Base stability inversely related to reactivity
        base_stability = 1.0 - profile.overall_reactivity

        # Adjust for aromatic stabilization
        aromatic_rings = profile.electronic_effects.get("num_aromatic_rings", 0)
        aromatic_bonus = min(0.2, aromatic_rings * 0.1)

        # Adjust for steric protection
        steric_protection = profile.steric_hindrance * 0.1

        stability_score = base_stability + aromatic_bonus + steric_protection
        return min(1.0, max(0.0, stability_score))

    def _calculate_selectivity_factors(
        self, mol: Chem.Mol, profile: ReactivityProfile
    ) -> Dict[str, float]:
        """Calculate selectivity factors for different polymerization conditions."""
        factors = {}

        # Temperature selectivity (higher for more stable monomers)
        factors["temperature_tolerance"] = profile.overall_reactivity * 0.8 + 0.2

        # Solvent selectivity (based on polarity)
        tpsa = profile.electronic_effects.get("tpsa", 0)
        factors["polar_solvent_compatibility"] = min(1.0, tpsa / 100.0)
        factors["nonpolar_solvent_compatibility"] = (
            1.0 - factors["polar_solvent_compatibility"]
        )

        # Catalyst selectivity
        factors["radical_initiation"] = (
            1.0 if PolymerizationType.RADICAL in profile.preferred_mechanisms else 0.3
        )
        factors["ionic_initiation"] = (
            1.0 if PolymerizationType.IONIC in profile.preferred_mechanisms else 0.3
        )

        return factors


class StereochemistryHandler:
    """Handles stereochemistry analysis for polymer formation."""

    def __init__(self):
        """Initialize the stereochemistry handler."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for stereochemistry analysis")

    def analyze_stereochemistry(self, smiles: str) -> StereochemistryInfo:
        """
        Analyze stereochemistry relevant to polymer formation.

        Args:
            smiles: SMILES string of the monomer

        Returns:
            StereochemistryInfo object
        """
        mol = parse_smiles_safe(smiles)
        if mol is None:
            raise InvalidMoleculeError(
                smiles, "Invalid SMILES for stereochemistry analysis"
            )

        # Find chiral centers
        chiral_centers = self._find_chiral_centers(mol)

        # Assess tacticity potential
        tacticity_potential = self._assess_tacticity_potential(mol, chiral_centers)

        # Generate stereoisomers
        stereoisomers = self._generate_stereoisomers(mol)

        # Calculate stereoregularity score
        stereoregularity_score = self._calculate_stereoregularity_score(
            mol, chiral_centers
        )

        return StereochemistryInfo(
            chiral_centers=chiral_centers,
            tacticity_potential=tacticity_potential,
            stereoisomers=stereoisomers,
            stereoregularity_score=stereoregularity_score,
        )

    def _find_chiral_centers(self, mol: Chem.Mol) -> List[int]:
        """Find chiral centers in the molecule."""
        chiral_centers = []
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                chiral_centers.append(atom.GetIdx())
        return chiral_centers

    def _assess_tacticity_potential(
        self, mol: Chem.Mol, chiral_centers: List[int]
    ) -> str:
        """Assess potential for tacticity in polymer formation."""
        if not chiral_centers:
            return "atactic"

        # Simple heuristic based on molecular structure
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        if len(chiral_centers) == 1 and num_heavy_atoms < 20:
            return "isotactic/syndiotactic"
        elif len(chiral_centers) > 1:
            return "complex_tacticity"
        else:
            return "atactic"

    def _generate_stereoisomers(self, mol: Chem.Mol) -> List[str]:
        """Generate possible stereoisomers."""
        try:
            # Use RDKit's stereoisomer enumeration
            stereoisomers = []
            opts = Chem.EnumerateStereoisomers.StereoEnumerationOptions()
            isomers = tuple(
                Chem.EnumerateStereoisomers.EnumerateStereoisomers(mol, options=opts)
            )

            for isomer in isomers[:10]:  # Limit to first 10 isomers
                try:
                    smiles = Chem.MolToSmiles(isomer)
                    stereoisomers.append(smiles)
                except:
                    continue

            return stereoisomers
        except Exception as e:
            logging.warning(f"Error generating stereoisomers: {e}")
            return []

    def _calculate_stereoregularity_score(
        self, mol: Chem.Mol, chiral_centers: List[int]
    ) -> float:
        """Calculate potential for stereoregular polymer formation."""
        if not chiral_centers:
            return 0.0

        # Simple scoring based on molecular complexity and chiral centers
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        complexity_factor = min(1.0, num_heavy_atoms / 30.0)
        chiral_factor = min(1.0, len(chiral_centers) / 3.0)

        return (1.0 - complexity_factor) * chiral_factor


class RingOpeningAnalyzer:
    """Analyzes ring-opening polymerization potential."""

    def __init__(self):
        """Initialize the ring-opening analyzer."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for ring-opening analysis")

        # Define ring patterns for ring-opening polymerization
        self.ring_patterns = {
            "three_membered": [
                "[C]1[O][C]1",
                "[C]1[N][C]1",
                "[C]1[S][C]1",
            ],  # Epoxides, aziridines, thiiranes
            "four_membered": [
                "[C]1[C][C][O]1",
                "[C]1[C][C][N]1",
            ],  # Oxetanes, azetidines
            "five_membered": [
                "[C]1[C][C][C][O]1",
                "[C]1[C][C][C][N]1",
            ],  # THF derivatives
            "six_membered": [
                "[C]1[C][C][C][C][O]1",
                "[C]1[C][C][C][C][N]1",
            ],  # THP derivatives
            "lactones": ["[C](=O)[O][C]", "[C](=O)[O][C][C]"],  # Various lactones
            "lactams": ["[C](=O)[N][C]", "[C](=O)[N][C][C]"],  # Various lactams
        }

    def analyze_ring_opening_potential(
        self, smiles: str
    ) -> Dict[str, Union[bool, float, List[str]]]:
        """
        Analyze ring-opening polymerization potential.

        Args:
            smiles: SMILES string of the monomer

        Returns:
            Dictionary with ring-opening analysis results
        """
        mol = parse_smiles_safe(smiles)
        if mol is None:
            raise InvalidMoleculeError(
                smiles, "Invalid SMILES for ring-opening analysis"
            )

        results = {
            "can_ring_open": False,
            "ring_strain_score": 0.0,
            "ring_types": [],
            "polymerization_likelihood": 0.0,
            "ring_sizes": [],
        }

        # Analyze ring systems
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        results["ring_sizes"] = ring_sizes

        # Check for specific ring patterns
        identified_rings = []
        max_strain_score = 0.0

        for ring_type, patterns in self.ring_patterns.items():
            for pattern in patterns:
                try:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        identified_rings.append(ring_type)
                        # Calculate strain score based on ring size and type
                        strain_score = self._calculate_ring_strain(
                            ring_type, ring_sizes
                        )
                        max_strain_score = max(max_strain_score, strain_score)
                except:
                    continue

        results["ring_types"] = list(set(identified_rings))
        results["ring_strain_score"] = max_strain_score
        results["can_ring_open"] = max_strain_score > 0.3
        results[
            "polymerization_likelihood"
        ] = self._calculate_polymerization_likelihood(
            max_strain_score, identified_rings, ring_sizes
        )

        return results

    def _calculate_ring_strain(self, ring_type: str, ring_sizes: List[int]) -> float:
        """Calculate ring strain score."""
        strain_scores = {
            "three_membered": 1.0,  # High strain
            "four_membered": 0.8,  # High strain
            "five_membered": 0.3,  # Moderate strain
            "six_membered": 0.1,  # Low strain
            "lactones": 0.7,  # Variable strain
            "lactams": 0.6,  # Variable strain
        }

        base_score = strain_scores.get(ring_type, 0.0)

        # Adjust based on actual ring sizes
        if ring_sizes:
            min_ring_size = min(ring_sizes)
            if min_ring_size <= 4:
                base_score *= 1.2  # Increase for small rings
            elif min_ring_size >= 7:
                base_score *= 0.8  # Decrease for large rings

        return min(1.0, base_score)

    def _calculate_polymerization_likelihood(
        self, strain_score: float, ring_types: List[str], ring_sizes: List[int]
    ) -> float:
        """Calculate likelihood of successful ring-opening polymerization."""
        if strain_score < 0.3:
            return 0.1  # Low likelihood

        # Base likelihood from strain
        likelihood = strain_score * 0.8

        # Bonus for known polymerizable ring types
        polymerizable_types = ["three_membered", "lactones", "lactams"]
        if any(rt in polymerizable_types for rt in ring_types):
            likelihood += 0.2

        return min(1.0, likelihood)


class MonomerAnalyzer:
    """Main class for comprehensive monomer analysis."""

    def __init__(self):
        """Initialize the monomer analyzer."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for monomer analysis")

        self.reactivity_analyzer = ReactivityAnalyzer()
        self.polymerization_analyzer = PolymerizationPotentialAnalyzer()
        self.stereochemistry_handler = StereochemistryHandler()
        self.ring_opening_analyzer = RingOpeningAnalyzer()

    def analyze_monomer(
        self, smiles: str
    ) -> Dict[
        str,
        Union[ReactivityProfile, PolymerizationPotential, StereochemistryInfo, Dict],
    ]:
        """
        Perform comprehensive monomer analysis.

        Args:
            smiles: SMILES string of the monomer

        Returns:
            Dictionary containing all analysis results
        """
        try:
            results = {
                "reactivity_profile": self.reactivity_analyzer.calculate_reactivity_profile(
                    smiles
                ),
                "polymerization_potential": self.polymerization_analyzer.assess_polymerization_potential(
                    smiles
                ),
                "stereochemistry_info": self.stereochemistry_handler.analyze_stereochemistry(
                    smiles
                ),
                "ring_opening_analysis": self.ring_opening_analyzer.analyze_ring_opening_potential(
                    smiles
                ),
            }

            return results

        except Exception as e:
            raise ChemiAccelError(f"Error in monomer analysis: {str(e)}")

    def batch_analyze_monomers(self, smiles_list: List[str]) -> List[Dict]:
        """
        Analyze multiple monomers in batch.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of analysis results
        """
        results = []
        for smiles in smiles_list:
            try:
                result = self.analyze_monomer(smiles)
                result["smiles"] = smiles
                result["success"] = True
                results.append(result)
            except Exception as e:
                results.append({"smiles": smiles, "success": False, "error": str(e)})

        return results

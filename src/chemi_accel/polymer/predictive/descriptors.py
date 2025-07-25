"""
Specialized polymer descriptor calculation algorithms.

This module implements polymer-specific descriptors that are particularly
relevant for property prediction, including structural, topological, and
thermodynamic descriptors.
"""

import logging
from typing import Dict, List, Union

import pandas as pd

try:
    import cudf
    import cupy as cp

    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcTPSA

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ...core.calculator import Calculator
from ...utils.validation import validate_smiles_series

logger = logging.getLogger(__name__)


class PolymerDescriptorCalculator:
    """
    Specialized descriptor calculator for polymer systems.

    This class calculates descriptors that are particularly relevant for
    polymer property prediction, including:
    - Monomer reactivity descriptors
    - Chain flexibility indicators
    - Thermal property predictors
    - Mechanical property correlates
    """

    def __init__(self, calculator: Calculator):
        """
        Initialize the polymer descriptor calculator.

        Args:
            calculator: ChemiAccel calculator instance
        """
        self.calculator = calculator

    def calculate_polymer_descriptors(
        self, molecules: Union[pd.Series, List[str]]
    ) -> pd.DataFrame:
        """
        Calculate comprehensive polymer-relevant descriptors.

        Args:
            molecules: SMILES strings of monomers or oligomers

        Returns:
            DataFrame with calculated descriptors
        """
        if isinstance(molecules, list):
            molecules = pd.Series(molecules)

        validate_smiles_series(molecules)

        # Calculate basic molecular descriptors
        basic_descriptors = self._calculate_basic_descriptors(molecules)

        # Calculate polymer-specific descriptors
        polymer_descriptors = self._calculate_polymer_specific_descriptors(molecules)

        # Calculate thermal property predictors
        thermal_descriptors = self._calculate_thermal_descriptors(molecules)

        # Calculate mechanical property predictors
        mechanical_descriptors = self._calculate_mechanical_descriptors(molecules)

        # Combine all descriptors
        all_descriptors = pd.concat(
            [
                basic_descriptors,
                polymer_descriptors,
                thermal_descriptors,
                mechanical_descriptors,
            ],
            axis=1,
        )

        return all_descriptors

    def _calculate_basic_descriptors(self, molecules: pd.Series) -> pd.DataFrame:
        """Calculate basic molecular descriptors relevant to polymers."""
        descriptors = {}

        for idx, smiles in molecules.items():
            if not RDKIT_AVAILABLE:
                # Fallback to simple descriptors
                desc = self._calculate_simple_descriptors(smiles)
            else:
                desc = self._calculate_rdkit_descriptors(smiles)

            descriptors[idx] = desc

        return pd.DataFrame.from_dict(descriptors, orient="index")

    def _calculate_rdkit_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate RDKit-based descriptors."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._get_default_descriptors()

            descriptors = {
                # Basic molecular properties
                "molecular_weight": Descriptors.MolWt(mol),
                "heavy_atom_count": mol.GetNumHeavyAtoms(),
                "atom_count": mol.GetNumAtoms(),
                "bond_count": mol.GetNumBonds(),
                # Lipophilicity and solubility
                "logp": Crippen.MolLogP(mol),
                "tpsa": CalcTPSA(mol),
                # Flexibility and conformational properties
                "rotatable_bonds": CalcNumRotatableBonds(mol),
                "ring_count": rdMolDescriptors.CalcNumRings(mol),
                "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
                # Hydrogen bonding
                "hbd_count": rdMolDescriptors.CalcNumHBD(mol),
                "hba_count": rdMolDescriptors.CalcNumHBA(mol),
                # Complexity measures
                "bertz_ct": rdMolDescriptors.BertzCT(mol),
                "chi0v": rdMolDescriptors.Chi0v(mol),
                "chi1v": rdMolDescriptors.Chi1v(mol),
                "chi2v": rdMolDescriptors.Chi2v(mol),
                # Electrotopological state
                "estate_vsa1": rdMolDescriptors.EState_VSA1(mol),
                "estate_vsa2": rdMolDescriptors.EState_VSA2(mol),
                "estate_vsa3": rdMolDescriptors.EState_VSA3(mol),
            }

            return descriptors

        except Exception as e:
            logger.warning(f"Failed to calculate RDKit descriptors for {smiles}: {e}")
            return self._get_default_descriptors()

    def _calculate_simple_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate simple descriptors without RDKit."""
        # Basic SMILES-based descriptors
        descriptors = {
            "smiles_length": len(smiles),
            "carbon_count": smiles.count("C"),
            "nitrogen_count": smiles.count("N"),
            "oxygen_count": smiles.count("O"),
            "sulfur_count": smiles.count("S"),
            "ring_indicators": smiles.count("1") + smiles.count("2"),
            "branch_indicators": smiles.count("(") + smiles.count(")"),
            "double_bond_count": smiles.count("="),
            "triple_bond_count": smiles.count("#"),
            "aromatic_indicators": smiles.count("c")
            + smiles.count("n")
            + smiles.count("o"),
        }

        # Estimate molecular weight from atom counts
        atomic_weights = {"C": 12.01, "N": 14.01, "O": 16.00, "S": 32.07, "H": 1.008}
        estimated_mw = (
            descriptors["carbon_count"] * atomic_weights["C"]
            + descriptors["nitrogen_count"] * atomic_weights["N"]
            + descriptors["oxygen_count"] * atomic_weights["O"]
            + descriptors["sulfur_count"] * atomic_weights["S"]
        )

        descriptors["estimated_molecular_weight"] = estimated_mw

        return descriptors

    def _get_default_descriptors(self) -> Dict[str, float]:
        """Return default descriptor values for failed calculations."""
        return {
            "molecular_weight": 0.0,
            "heavy_atom_count": 0.0,
            "atom_count": 0.0,
            "bond_count": 0.0,
            "logp": 0.0,
            "tpsa": 0.0,
            "rotatable_bonds": 0.0,
            "ring_count": 0.0,
            "aromatic_rings": 0.0,
            "aliphatic_rings": 0.0,
            "hbd_count": 0.0,
            "hba_count": 0.0,
            "bertz_ct": 0.0,
            "chi0v": 0.0,
            "chi1v": 0.0,
            "chi2v": 0.0,
            "estate_vsa1": 0.0,
            "estate_vsa2": 0.0,
            "estate_vsa3": 0.0,
        }

    def _calculate_polymer_specific_descriptors(
        self, molecules: pd.Series
    ) -> pd.DataFrame:
        """Calculate polymer-specific structural descriptors."""
        descriptors = {}

        for idx, smiles in molecules.items():
            desc = {
                # Reactivity indicators
                "vinyl_groups": self._count_vinyl_groups(smiles),
                "epoxy_groups": self._count_epoxy_groups(smiles),
                "hydroxyl_groups": self._count_hydroxyl_groups(smiles),
                "carboxyl_groups": self._count_carboxyl_groups(smiles),
                "amine_groups": self._count_amine_groups(smiles),
                # Chain flexibility indicators
                "backbone_flexibility": self._estimate_backbone_flexibility(smiles),
                "side_chain_length": self._estimate_side_chain_length(smiles),
                "branching_factor": self._estimate_branching_factor(smiles),
                # Stereochemistry indicators
                "chiral_centers": self._count_chiral_centers(smiles),
                "stereochemistry_complexity": self._estimate_stereochemistry_complexity(
                    smiles
                ),
                # Polymerization potential
                "polymerization_sites": self._count_polymerization_sites(smiles),
                "crosslinking_potential": self._estimate_crosslinking_potential(smiles),
            }

            descriptors[idx] = desc

        return pd.DataFrame.from_dict(descriptors, orient="index")

    def _calculate_thermal_descriptors(self, molecules: pd.Series) -> pd.DataFrame:
        """Calculate descriptors relevant to thermal properties."""
        descriptors = {}

        for idx, smiles in molecules.items():
            desc = {
                # Glass transition predictors
                "tg_predictor_1": self._calculate_tg_predictor_1(smiles),
                "tg_predictor_2": self._calculate_tg_predictor_2(smiles),
                "tg_predictor_3": self._calculate_tg_predictor_3(smiles),
                # Thermal stability indicators
                "aromatic_content": self._calculate_aromatic_content(smiles),
                "thermal_stability_index": self._calculate_thermal_stability_index(
                    smiles
                ),
                "decomposition_predictor": self._calculate_decomposition_predictor(
                    smiles
                ),
                # Crystallinity indicators
                "crystallinity_index": self._calculate_crystallinity_index(smiles),
                "packing_efficiency": self._estimate_packing_efficiency(smiles),
            }

            descriptors[idx] = desc

        return pd.DataFrame.from_dict(descriptors, orient="index")

    def _calculate_mechanical_descriptors(self, molecules: pd.Series) -> pd.DataFrame:
        """Calculate descriptors relevant to mechanical properties."""
        descriptors = {}

        for idx, smiles in molecules.items():
            desc = {
                # Young's modulus predictors
                "modulus_predictor_1": self._calculate_modulus_predictor_1(smiles),
                "modulus_predictor_2": self._calculate_modulus_predictor_2(smiles),
                "modulus_predictor_3": self._calculate_modulus_predictor_3(smiles),
                # Stiffness indicators
                "backbone_stiffness": self._calculate_backbone_stiffness(smiles),
                "crosslink_density_predictor": self._calculate_crosslink_density_predictor(
                    smiles
                ),
                # Toughness indicators
                "toughness_predictor": self._calculate_toughness_predictor(smiles),
                "brittleness_index": self._calculate_brittleness_index(smiles),
                # Density predictors
                "density_predictor": self._calculate_density_predictor(smiles),
                "free_volume_predictor": self._calculate_free_volume_predictor(smiles),
            }

            descriptors[idx] = desc

        return pd.DataFrame.from_dict(descriptors, orient="index")

    # Helper methods for specific descriptor calculations
    def _count_vinyl_groups(self, smiles: str) -> int:
        """Count vinyl groups (C=C) in the molecule."""
        return smiles.count("C=C")

    def _count_epoxy_groups(self, smiles: str) -> int:
        """Count epoxy groups in the molecule."""
        # Simple pattern matching for epoxy rings
        return smiles.count("C1OC1")

    def _count_hydroxyl_groups(self, smiles: str) -> int:
        """Count hydroxyl groups in the molecule."""
        return smiles.count("O")  # Simplified

    def _count_carboxyl_groups(self, smiles: str) -> int:
        """Count carboxyl groups in the molecule."""
        return smiles.count("C(=O)O")

    def _count_amine_groups(self, smiles: str) -> int:
        """Count amine groups in the molecule."""
        return smiles.count("N")  # Simplified

    def _estimate_backbone_flexibility(self, smiles: str) -> float:
        """Estimate backbone flexibility based on rotatable bonds."""
        # Simple heuristic based on single bonds and chain length
        single_bonds = smiles.count("-")
        total_length = len(smiles)
        return single_bonds / max(total_length, 1)

    def _estimate_side_chain_length(self, smiles: str) -> float:
        """Estimate average side chain length."""
        branches = smiles.count("(")
        if branches == 0:
            return 0.0

        # Simple heuristic
        return len(smiles) / (branches * 10)

    def _estimate_branching_factor(self, smiles: str) -> float:
        """Estimate degree of branching."""
        branches = smiles.count("(") + smiles.count(")")
        return branches / max(len(smiles), 1)

    def _count_chiral_centers(self, smiles: str) -> int:
        """Count chiral centers (simplified)."""
        return smiles.count("@")

    def _estimate_stereochemistry_complexity(self, smiles: str) -> float:
        """Estimate stereochemistry complexity."""
        stereo_indicators = smiles.count("@") + smiles.count("/")
        return stereo_indicators / max(len(smiles), 1)

    def _count_polymerization_sites(self, smiles: str) -> int:
        """Count potential polymerization sites."""
        vinyl = self._count_vinyl_groups(smiles)
        epoxy = self._count_epoxy_groups(smiles)
        return vinyl + epoxy

    def _estimate_crosslinking_potential(self, smiles: str) -> float:
        """Estimate crosslinking potential."""
        functional_groups = (
            self._count_vinyl_groups(smiles)
            + self._count_epoxy_groups(smiles)
            + self._count_hydroxyl_groups(smiles)
        )
        return functional_groups / max(len(smiles), 1)

    # Thermal property predictors
    def _calculate_tg_predictor_1(self, smiles: str) -> float:
        """First glass transition temperature predictor."""
        # Based on aromatic content and flexibility
        aromatic = smiles.count("c") + smiles.count("n") + smiles.count("o")
        flexibility = self._estimate_backbone_flexibility(smiles)
        return aromatic * 10 - flexibility * 50

    def _calculate_tg_predictor_2(self, smiles: str) -> float:
        """Second glass transition temperature predictor."""
        # Based on molecular weight and ring content
        mw_estimate = len(smiles) * 5  # Rough estimate
        rings = smiles.count("1") + smiles.count("2")
        return mw_estimate * 0.1 + rings * 20

    def _calculate_tg_predictor_3(self, smiles: str) -> float:
        """Third glass transition temperature predictor."""
        # Based on hydrogen bonding potential
        hb_donors = self._count_hydroxyl_groups(smiles) + self._count_amine_groups(
            smiles
        )
        hb_acceptors = smiles.count("O") + smiles.count("N")
        return (hb_donors + hb_acceptors) * 15

    def _calculate_aromatic_content(self, smiles: str) -> float:
        """Calculate aromatic content fraction."""
        aromatic_chars = smiles.count("c") + smiles.count("n") + smiles.count("o")
        total_chars = len(smiles)
        return aromatic_chars / max(total_chars, 1)

    def _calculate_thermal_stability_index(self, smiles: str) -> float:
        """Calculate thermal stability index."""
        aromatic = self._calculate_aromatic_content(smiles)
        rings = (smiles.count("1") + smiles.count("2")) / max(len(smiles), 1)
        return aromatic * 0.7 + rings * 0.3

    def _calculate_decomposition_predictor(self, smiles: str) -> float:
        """Predict decomposition temperature."""
        stability = self._calculate_thermal_stability_index(smiles)
        return stability * 400 + 200  # Rough estimate in Celsius

    def _calculate_crystallinity_index(self, smiles: str) -> float:
        """Calculate crystallinity index."""
        # Based on regularity and symmetry
        regularity = 1.0 / max(len(set(smiles)), 1)  # Character diversity
        return regularity

    def _estimate_packing_efficiency(self, smiles: str) -> float:
        """Estimate packing efficiency."""
        branching = self._estimate_branching_factor(smiles)
        return 1.0 - branching  # Less branching = better packing

    # Mechanical property predictors
    def _calculate_modulus_predictor_1(self, smiles: str) -> float:
        """First Young's modulus predictor."""
        # Based on aromatic content and crosslinking
        aromatic = self._calculate_aromatic_content(smiles)
        crosslink = self._estimate_crosslinking_potential(smiles)
        return aromatic * 1000 + crosslink * 500

    def _calculate_modulus_predictor_2(self, smiles: str) -> float:
        """Second Young's modulus predictor."""
        # Based on backbone stiffness
        rings = (smiles.count("1") + smiles.count("2")) / max(len(smiles), 1)
        double_bonds = smiles.count("=") / max(len(smiles), 1)
        return (rings + double_bonds) * 800

    def _calculate_modulus_predictor_3(self, smiles: str) -> float:
        """Third Young's modulus predictor."""
        # Based on crystallinity and packing
        crystallinity = self._calculate_crystallinity_index(smiles)
        packing = self._estimate_packing_efficiency(smiles)
        return (crystallinity + packing) * 600

    def _calculate_backbone_stiffness(self, smiles: str) -> float:
        """Calculate backbone stiffness."""
        aromatic = self._calculate_aromatic_content(smiles)
        double_bonds = smiles.count("=") / max(len(smiles), 1)
        return aromatic * 0.8 + double_bonds * 0.2

    def _calculate_crosslink_density_predictor(self, smiles: str) -> float:
        """Predict crosslink density."""
        return self._estimate_crosslinking_potential(smiles) * 100

    def _calculate_toughness_predictor(self, smiles: str) -> float:
        """Predict toughness."""
        flexibility = self._estimate_backbone_flexibility(smiles)
        crosslink = self._estimate_crosslinking_potential(smiles)
        return flexibility * 0.6 + crosslink * 0.4

    def _calculate_brittleness_index(self, smiles: str) -> float:
        """Calculate brittleness index."""
        stiffness = self._calculate_backbone_stiffness(smiles)
        flexibility = self._estimate_backbone_flexibility(smiles)
        return stiffness / max(flexibility, 0.1)

    def _calculate_density_predictor(self, smiles: str) -> float:
        """Predict polymer density."""
        packing = self._estimate_packing_efficiency(smiles)
        aromatic = self._calculate_aromatic_content(smiles)
        return 0.8 + packing * 0.4 + aromatic * 0.3  # g/cm³

    def _calculate_free_volume_predictor(self, smiles: str) -> float:
        """Predict free volume fraction."""
        packing = self._estimate_packing_efficiency(smiles)
        return 1.0 - packing

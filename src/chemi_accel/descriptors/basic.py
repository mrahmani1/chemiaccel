"""
Basic molecular descriptor calculations.

This module provides fundamental molecular property calculations including
molecular weight, atom/bond counts, ring information, and basic 2D descriptors
like TPSA and LogP using RDKit as the reference implementation.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem,
        Crippen,
        Descriptors,
        Lipinski,
        rdchem,
        rdMolDescriptors,
    )

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ..core.exceptions import ChemiAccelError, DescriptorCalculationError
from ..core.molecular import MolecularData
from ..utils.logging import get_logger

logger = get_logger()


class BasicDescriptorCalculator:
    """Calculator for basic molecular descriptors using RDKit."""

    def __init__(self):
        """Initialize basic descriptor calculator."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for basic descriptor calculations")

        # Define descriptor functions with metadata
        self.descriptor_registry = {
            # Fundamental molecular properties
            "molecular_weight": {
                "function": Descriptors.MolWt,
                "description": "Molecular weight in Daltons",
                "category": "constitutional",
            },
            "exact_molecular_weight": {
                "function": Descriptors.ExactMolWt,
                "description": "Exact molecular weight using isotopic masses",
                "category": "constitutional",
            },
            # Atom and bond counts
            "num_atoms": {
                "function": lambda mol: mol.GetNumAtoms(),
                "description": "Total number of atoms",
                "category": "constitutional",
            },
            "num_heavy_atoms": {
                "function": Descriptors.HeavyAtomCount,
                "description": "Number of heavy (non-hydrogen) atoms",
                "category": "constitutional",
            },
            "num_bonds": {
                "function": lambda mol: mol.GetNumBonds(),
                "description": "Total number of bonds",
                "category": "constitutional",
            },
            # Ring information
            "ring_count": {
                "function": Descriptors.RingCount,
                "description": "Total number of rings",
                "category": "topological",
            },
            "num_aromatic_rings": {
                "function": Descriptors.NumAromaticRings,
                "description": "Number of aromatic rings",
                "category": "topological",
            },
            "num_saturated_rings": {
                "function": Descriptors.NumSaturatedRings,
                "description": "Number of saturated rings",
                "category": "topological",
            },
            "num_aliphatic_rings": {
                "function": Descriptors.NumAliphaticRings,
                "description": "Number of aliphatic rings",
                "category": "topological",
            },
            "largest_ring_size": {
                "function": self._calculate_largest_ring_size,
                "description": "Size of the largest ring",
                "category": "topological",
            },
            # Lipophilicity and solubility
            "logp": {
                "function": Crippen.MolLogP,
                "description": "Octanol-water partition coefficient (LogP)",
                "category": "physicochemical",
            },
            "tpsa": {
                "function": Descriptors.TPSA,
                "description": "Topological polar surface area",
                "category": "physicochemical",
            },
            # Structural features
            "num_rotatable_bonds": {
                "function": Descriptors.NumRotatableBonds,
                "description": "Number of rotatable bonds",
                "category": "topological",
            },
            "num_heteroatoms": {
                "function": Descriptors.NumHeteroatoms,
                "description": "Number of heteroatoms",
                "category": "constitutional",
            },
            # Hydrogen bonding
            "num_hbd": {
                "function": Descriptors.NumHDonors,
                "description": "Number of hydrogen bond donors",
                "category": "physicochemical",
            },
            "num_hba": {
                "function": Descriptors.NumHAcceptors,
                "description": "Number of hydrogen bond acceptors",
                "category": "physicochemical",
            },
            # Complexity measures
            "bertz_complexity": {
                "function": Descriptors.BertzCT,
                "description": "Bertz complexity index",
                "category": "complexity",
            },
            "fraction_csp3": {
                "function": Descriptors.FractionCSP3,
                "description": "Fraction of sp3 carbons",
                "category": "constitutional",
            },
            # Additional 2D descriptors
            "molar_refractivity": {
                "function": Crippen.MolMR,
                "description": "Molar refractivity",
                "category": "physicochemical",
            },
            "num_radical_electrons": {
                "function": Descriptors.NumRadicalElectrons,
                "description": "Number of radical electrons",
                "category": "electronic",
            },
            "formal_charge": {
                "function": Chem.rdmolops.GetFormalCharge,
                "description": "Total formal charge",
                "category": "electronic",
            },
        }

        logger.info(
            f"BasicDescriptorCalculator initialized with {len(self.descriptor_registry)} descriptors"
        )

    def _calculate_largest_ring_size(self, mol: Chem.Mol) -> int:
        """Calculate the size of the largest ring in the molecule."""
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        if not atom_rings:
            return 0
        return max(len(ring) for ring in atom_rings)

    def get_available_descriptors(self) -> List[str]:
        """Get list of available descriptor names."""
        return list(self.descriptor_registry.keys())

    def get_descriptor_info(self, descriptor_name: str) -> Dict[str, str]:
        """Get information about a specific descriptor."""
        if descriptor_name not in self.descriptor_registry:
            raise ValueError(f"Unknown descriptor: {descriptor_name}")

        info = self.descriptor_registry[descriptor_name].copy()
        info.pop("function")  # Remove function reference
        return info

    def get_descriptors_by_category(self, category: str) -> List[str]:
        """Get descriptors by category."""
        return [
            name
            for name, info in self.descriptor_registry.items()
            if info["category"] == category
        ]

    def calculate_single_descriptor(
        self, mol_data: MolecularData, descriptor_name: str
    ) -> float:
        """Calculate a single descriptor for a molecule.

        Args:
            mol_data: MolecularData with RDKit mol object
            descriptor_name: Name of descriptor to calculate

        Returns:
            Calculated descriptor value

        Raises:
            DescriptorCalculationError: If calculation fails
        """
        if descriptor_name not in self.descriptor_registry:
            raise DescriptorCalculationError(
                descriptor=descriptor_name,
                molecule_id=mol_data.mol_id,
                reason="Unknown descriptor",
            )

        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise DescriptorCalculationError(
                descriptor=descriptor_name,
                molecule_id=mol_data.mol_id,
                reason="No RDKit mol object found in molecular data",
            )

        try:
            desc_func = self.descriptor_registry[descriptor_name]["function"]
            value = desc_func(rdkit_mol)
            return float(value) if value is not None else 0.0
        except Exception as e:
            raise DescriptorCalculationError(
                descriptor=descriptor_name,
                molecule_id=mol_data.mol_id,
                reason=f"Failed to calculate {descriptor_name}: {e}",
            )

    def calculate_descriptors(
        self, mol_data: MolecularData, descriptor_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate multiple descriptors for a single molecule.

        Args:
            mol_data: MolecularData with RDKit mol object
            descriptor_names: List of descriptor names (None for all)

        Returns:
            Dictionary of descriptor values
        """
        if descriptor_names is None:
            descriptor_names = self.get_available_descriptors()

        descriptors = {}
        errors = []

        for desc_name in descriptor_names:
            try:
                value = self.calculate_single_descriptor(mol_data, desc_name)
                descriptors[desc_name] = value
            except DescriptorCalculationError as e:
                logger.warning(f"Descriptor calculation failed: {e}")
                descriptors[desc_name] = np.nan
                errors.append(str(e))

        # Store any errors in the result for debugging
        if errors:
            descriptors["_calculation_errors"] = errors

        return descriptors

    def calculate_descriptors_batch(
        self,
        molecules: List[MolecularData],
        descriptor_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Calculate descriptors for a batch of molecules.

        Args:
            molecules: List of MolecularData objects
            descriptor_names: List of descriptor names to calculate

        Returns:
            DataFrame with calculated descriptors
        """
        if not molecules:
            return pd.DataFrame()

        if descriptor_names is None:
            descriptor_names = self.get_available_descriptors()

        results = []

        for mol_data in molecules:
            try:
                descriptors = self.calculate_descriptors(mol_data, descriptor_names)

                # Add molecule identifiers
                result_row = {
                    "mol_id": mol_data.mol_id,
                    "smiles": mol_data.smiles,
                    **descriptors,
                }
                results.append(result_row)

            except Exception as e:
                logger.error(
                    f"Failed to calculate descriptors for molecule {mol_data.mol_id}: {e}"
                )
                # Add row with NaN values
                result_row = {
                    "mol_id": mol_data.mol_id,
                    "smiles": mol_data.smiles,
                    **dict.fromkeys(descriptor_names, np.nan),
                }
                results.append(result_row)

        df = pd.DataFrame(results)

        # Reorder columns: identifiers first, then descriptors
        id_cols = ["mol_id", "smiles"]
        desc_cols = [
            col for col in df.columns if col not in id_cols and not col.startswith("_")
        ]
        error_cols = [col for col in df.columns if col.startswith("_")]

        df = df[id_cols + desc_cols + error_cols]

        logger.info(
            f"Calculated {len(descriptor_names)} descriptors for {len(df)} molecules"
        )
        return df

    def get_descriptor_statistics(
        self, df: pd.DataFrame, descriptor_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate statistics for descriptors in a DataFrame.

        Args:
            df: DataFrame with calculated descriptors
            descriptor_names: List of descriptor names to analyze

        Returns:
            DataFrame with descriptor statistics
        """
        if descriptor_names is None:
            # Get numeric columns that are likely descriptors
            descriptor_names = [
                col
                for col in df.columns
                if col not in ["mol_id", "smiles"]
                and df[col].dtype in [np.float64, np.int64]
            ]

        stats = df[descriptor_names].describe()

        # Add additional statistics
        stats.loc["missing"] = df[descriptor_names].isna().sum()
        stats.loc["missing_pct"] = (df[descriptor_names].isna().sum() / len(df)) * 100

        return stats


def calculate_basic_properties(molecules: List[MolecularData]) -> pd.DataFrame:
    """Convenience function to calculate basic molecular properties.

    Args:
        molecules: List of MolecularData objects

    Returns:
        DataFrame with basic molecular properties
    """
    calculator = BasicDescriptorCalculator()
    basic_descriptors = [
        "molecular_weight",
        "num_atoms",
        "num_heavy_atoms",
        "num_bonds",
        "ring_count",
        "num_aromatic_rings",
        "logp",
        "tpsa",
        "num_rotatable_bonds",
        "num_hbd",
        "num_hba",
    ]
    return calculator.calculate_descriptors_batch(molecules, basic_descriptors)


def get_lipinski_descriptors(molecules: List[MolecularData]) -> pd.DataFrame:
    """Calculate Lipinski Rule of Five descriptors.

    Args:
        molecules: List of MolecularData objects

    Returns:
        DataFrame with Lipinski descriptors and violations
    """
    calculator = BasicDescriptorCalculator()
    lipinski_descriptors = ["molecular_weight", "logp", "num_hbd", "num_hba"]

    df = calculator.calculate_descriptors_batch(molecules, lipinski_descriptors)

    # Add Lipinski violations
    df["lipinski_violations"] = (
        (df["molecular_weight"] > 500).astype(int)
        + (df["logp"] > 5).astype(int)
        + (df["num_hbd"] > 5).astype(int)
        + (df["num_hba"] > 10).astype(int)
    )

    df["lipinski_compliant"] = df["lipinski_violations"] == 0

    return df

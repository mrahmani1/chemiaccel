"""
Reference Implementation Wrappers for Benchmarking.

This module provides wrappers for RDKit and Mordred implementations
to enable fair performance comparisons with ChemiAccel.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import numpy as np
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcTPSA

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    Descriptors = None

# Mordred imports
try:
    from mordred import Calculator as MordredCalculator
    from mordred import descriptors as mordred_descriptors

    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False
    MordredCalculator = None

try:
    from ..src.chemi_accel.utils.logging import get_logger
except ImportError:
    # Fallback for standalone execution
    import logging

    def get_logger():
        return logging.getLogger(__name__)


logger = get_logger()


@dataclass
class ReferenceResult:
    """Result from reference implementation."""

    implementation: str
    descriptor: str
    values: List[float]
    execution_time: float
    success_count: int
    error_count: int
    errors: List[str]


class ReferenceImplementation(ABC):
    """Abstract base class for reference implementations."""

    def __init__(self, name: str):
        """Initialize reference implementation.

        Args:
            name: Name of the implementation
        """
        self.name = name
        self.supported_descriptors = set()
        self._setup_descriptors()

    @abstractmethod
    def _setup_descriptors(self) -> None:
        """Setup supported descriptors."""
        pass

    @abstractmethod
    def calculate_descriptor(
        self, molecules: List[str], descriptor: str
    ) -> ReferenceResult:
        """Calculate descriptor for molecules.

        Args:
            molecules: List of SMILES strings
            descriptor: Descriptor name

        Returns:
            Reference result
        """
        pass

    def is_descriptor_supported(self, descriptor: str) -> bool:
        """Check if descriptor is supported.

        Args:
            descriptor: Descriptor name

        Returns:
            True if supported
        """
        return descriptor in self.supported_descriptors

    def get_supported_descriptors(self) -> List[str]:
        """Get list of supported descriptors.

        Returns:
            List of descriptor names
        """
        return list(self.supported_descriptors)


class RDKitReference(ReferenceImplementation):
    """RDKit reference implementation wrapper."""

    def __init__(self):
        """Initialize RDKit reference."""
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit not available")

        super().__init__("rdkit")
        logger.info("Initialized RDKit reference implementation")

    def _setup_descriptors(self) -> None:
        """Setup RDKit descriptors."""
        self.supported_descriptors = {
            "molecular_weight",
            "logp",
            "tpsa",
            "num_rotatable_bonds",
            "num_aromatic_rings",
            "num_heavy_atoms",
            "num_heteroatoms",
            "num_rings",
            "fraction_csp3",
            "num_hbd",  # Hydrogen bond donors
            "num_hba",  # Hydrogen bond acceptors
            "slogp",
            "smr",  # Molar refractivity
        }

        # Descriptor function mapping
        self.descriptor_functions = {
            "molecular_weight": lambda mol: Descriptors.MolWt(mol),
            "logp": lambda mol: Crippen.MolLogP(mol),
            "tpsa": lambda mol: CalcTPSA(mol),
            "num_rotatable_bonds": lambda mol: CalcNumRotatableBonds(mol),
            "num_aromatic_rings": lambda mol: Descriptors.NumAromaticRings(mol),
            "num_heavy_atoms": lambda mol: mol.GetNumHeavyAtoms(),
            "num_heteroatoms": lambda mol: Descriptors.NumHeteroatoms(mol),
            "num_rings": lambda mol: Descriptors.RingCount(mol),
            "fraction_csp3": lambda mol: Descriptors.FractionCsp3(mol),
            "num_hbd": lambda mol: Descriptors.NumHDonors(mol),
            "num_hba": lambda mol: Descriptors.NumHAcceptors(mol),
            "slogp": lambda mol: Descriptors.SlogP_VSA1(mol),
            "smr": lambda mol: Descriptors.SMR_VSA1(mol),
        }

    def _parse_smiles(self, smiles: str) -> Optional[Chem.Mol]:
        """Parse SMILES string to RDKit molecule.

        Args:
            smiles: SMILES string

        Returns:
            RDKit molecule or None if parsing fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Add hydrogens for more accurate calculations
                mol = Chem.AddHs(mol)
            return mol
        except Exception as e:
            logger.debug(f"Failed to parse SMILES '{smiles}': {e}")
            return None

    def calculate_descriptor(
        self, molecules: List[str], descriptor: str
    ) -> ReferenceResult:
        """Calculate descriptor using RDKit.

        Args:
            molecules: List of SMILES strings
            descriptor: Descriptor name

        Returns:
            Reference result
        """
        if descriptor not in self.supported_descriptors:
            raise ValueError(
                f"Descriptor '{descriptor}' not supported by RDKit reference"
            )

        start_time = time.perf_counter()

        values = []
        errors = []
        success_count = 0
        error_count = 0

        descriptor_func = self.descriptor_functions[descriptor]

        for i, smiles in enumerate(molecules):
            try:
                mol = self._parse_smiles(smiles)
                if mol is None:
                    values.append(float("nan"))
                    errors.append(f"Failed to parse SMILES at index {i}: {smiles}")
                    error_count += 1
                    continue

                value = descriptor_func(mol)
                values.append(float(value) if value is not None else float("nan"))
                success_count += 1

            except Exception as e:
                values.append(float("nan"))
                errors.append(f"Error calculating {descriptor} at index {i}: {str(e)}")
                error_count += 1

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        result = ReferenceResult(
            implementation=self.name,
            descriptor=descriptor,
            values=values,
            execution_time=execution_time,
            success_count=success_count,
            error_count=error_count,
            errors=errors,
        )

        logger.info(
            f"RDKit {descriptor}: {success_count}/{len(molecules)} successful, "
            f"{execution_time:.3f}s"
        )

        return result

    def calculate_multiple_descriptors(
        self, molecules: List[str], descriptors: List[str]
    ) -> Dict[str, ReferenceResult]:
        """Calculate multiple descriptors efficiently.

        Args:
            molecules: List of SMILES strings
            descriptors: List of descriptor names

        Returns:
            Dictionary mapping descriptor names to results
        """
        # Pre-parse all molecules once
        parsed_mols = []
        for smiles in molecules:
            mol = self._parse_smiles(smiles)
            parsed_mols.append(mol)

        results = {}

        for descriptor in descriptors:
            if descriptor not in self.supported_descriptors:
                logger.warning(f"Skipping unsupported descriptor: {descriptor}")
                continue

            start_time = time.perf_counter()

            values = []
            errors = []
            success_count = 0
            error_count = 0

            descriptor_func = self.descriptor_functions[descriptor]

            for i, mol in enumerate(parsed_mols):
                try:
                    if mol is None:
                        values.append(float("nan"))
                        error_count += 1
                        continue

                    value = descriptor_func(mol)
                    values.append(float(value) if value is not None else float("nan"))
                    success_count += 1

                except Exception as e:
                    values.append(float("nan"))
                    errors.append(
                        f"Error calculating {descriptor} at index {i}: {str(e)}"
                    )
                    error_count += 1

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            results[descriptor] = ReferenceResult(
                implementation=self.name,
                descriptor=descriptor,
                values=values,
                execution_time=execution_time,
                success_count=success_count,
                error_count=error_count,
                errors=errors,
            )

        return results


class MordredReference(ReferenceImplementation):
    """Mordred reference implementation wrapper."""

    def __init__(self):
        """Initialize Mordred reference."""
        if not MORDRED_AVAILABLE:
            raise ImportError("Mordred not available")

        super().__init__("mordred")

        # Initialize Mordred calculator
        self.calculator = MordredCalculator(mordred_descriptors, ignore_3D=True)

        logger.info("Initialized Mordred reference implementation")

    def _setup_descriptors(self) -> None:
        """Setup Mordred descriptors."""
        # Map common descriptor names to Mordred equivalents
        self.supported_descriptors = {
            "molecular_weight",
            "logp",
            "tpsa",
            "num_rotatable_bonds",
            "num_aromatic_rings",
            "num_heavy_atoms",
            "wiener_index",
            "zagreb_m1",
            "zagreb_m2",
            "randic_index",
            "balaban_j",
            "kappa1",
            "kappa2",
            "kappa3",
        }

        # Mordred descriptor mapping
        self.descriptor_mapping = {
            "molecular_weight": "MW",
            "logp": "SlogP",
            "tpsa": "TPSA",
            "num_rotatable_bonds": "nRotB",
            "num_aromatic_rings": "nAromRing",
            "num_heavy_atoms": "nHeavyAtom",
            "wiener_index": "W",
            "zagreb_m1": "Zagreb1",
            "zagreb_m2": "Zagreb2",
            "randic_index": "Chi0",
            "balaban_j": "BalabanJ",
            "kappa1": "Kappa1",
            "kappa2": "Kappa2",
            "kappa3": "Kappa3",
        }

    def _parse_smiles(self, smiles: str) -> Optional[Chem.Mol]:
        """Parse SMILES string to RDKit molecule for Mordred.

        Args:
            smiles: SMILES string

        Returns:
            RDKit molecule or None if parsing fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
            return mol
        except Exception as e:
            logger.debug(f"Failed to parse SMILES '{smiles}': {e}")
            return None

    def calculate_descriptor(
        self, molecules: List[str], descriptor: str
    ) -> ReferenceResult:
        """Calculate descriptor using Mordred.

        Args:
            molecules: List of SMILES strings
            descriptor: Descriptor name

        Returns:
            Reference result
        """
        if descriptor not in self.supported_descriptors:
            raise ValueError(
                f"Descriptor '{descriptor}' not supported by Mordred reference"
            )

        start_time = time.perf_counter()

        # Parse molecules
        parsed_mols = []
        for smiles in molecules:
            mol = self._parse_smiles(smiles)
            parsed_mols.append(mol)

        # Calculate descriptors
        try:
            # Calculate all descriptors at once (Mordred's efficient approach)
            df = self.calculator.pandas(parsed_mols)

            # Extract specific descriptor
            mordred_name = self.descriptor_mapping.get(descriptor, descriptor)

            if mordred_name in df.columns:
                values = df[mordred_name].tolist()
            else:
                # Try to find similar column name
                matching_cols = [
                    col for col in df.columns if mordred_name.lower() in col.lower()
                ]
                if matching_cols:
                    values = df[matching_cols[0]].tolist()
                else:
                    raise ValueError(
                        f"Descriptor {mordred_name} not found in Mordred results"
                    )

            # Convert to float and handle errors
            processed_values = []
            errors = []
            success_count = 0
            error_count = 0

            for i, value in enumerate(values):
                try:
                    if pd.isna(value) or value is None:
                        processed_values.append(float("nan"))
                        error_count += 1
                    else:
                        processed_values.append(float(value))
                        success_count += 1
                except Exception as e:
                    processed_values.append(float("nan"))
                    errors.append(f"Error processing value at index {i}: {str(e)}")
                    error_count += 1

        except Exception as e:
            # Fallback: return NaN for all molecules
            processed_values = [float("nan")] * len(molecules)
            errors = [f"Mordred calculation failed: {str(e)}"]
            success_count = 0
            error_count = len(molecules)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        result = ReferenceResult(
            implementation=self.name,
            descriptor=descriptor,
            values=processed_values,
            execution_time=execution_time,
            success_count=success_count,
            error_count=error_count,
            errors=errors,
        )

        logger.info(
            f"Mordred {descriptor}: {success_count}/{len(molecules)} successful, "
            f"{execution_time:.3f}s"
        )

        return result

    def calculate_all_descriptors(self, molecules: List[str]) -> pd.DataFrame:
        """Calculate all available Mordred descriptors.

        Args:
            molecules: List of SMILES strings

        Returns:
            DataFrame with all descriptor values
        """
        # Parse molecules
        parsed_mols = []
        for smiles in molecules:
            mol = self._parse_smiles(smiles)
            parsed_mols.append(mol)

        # Calculate all descriptors
        start_time = time.perf_counter()
        df = self.calculator.pandas(parsed_mols)
        end_time = time.perf_counter()

        logger.info(
            f"Calculated {len(df.columns)} Mordred descriptors for "
            f"{len(molecules)} molecules in {end_time - start_time:.3f}s"
        )

        return df


class CPUReference(ReferenceImplementation):
    """CPU-only reference implementation using basic algorithms."""

    def __init__(self):
        """Initialize CPU reference."""
        super().__init__("cpu_reference")
        logger.info("Initialized CPU reference implementation")

    def _setup_descriptors(self) -> None:
        """Setup CPU reference descriptors."""
        self.supported_descriptors = {
            "molecular_weight",
            "atom_count",
            "bond_count",
        }

        # Atomic weights
        self.atomic_weights = {
            "C": 12.01,
            "H": 1.008,
            "O": 15.999,
            "N": 14.007,
            "S": 32.06,
            "P": 30.974,
            "F": 18.998,
            "Cl": 35.45,
            "Br": 79.904,
            "I": 126.90,
        }

    def _estimate_molecular_weight(self, smiles: str) -> float:
        """Estimate molecular weight from SMILES."""
        weight = 0.0

        # Count explicit atoms
        for atom, atomic_weight in self.atomic_weights.items():
            count = smiles.count(atom) + smiles.count(atom.lower())
            weight += count * atomic_weight

        return weight

    def calculate_descriptor(
        self, molecules: List[str], descriptor: str
    ) -> ReferenceResult:
        """Calculate descriptor using CPU reference.

        Args:
            molecules: List of SMILES strings
            descriptor: Descriptor name

        Returns:
            Reference result
        """
        if descriptor not in self.supported_descriptors:
            raise ValueError(
                f"Descriptor '{descriptor}' not supported by CPU reference"
            )

        start_time = time.perf_counter()

        values = []
        errors = []
        success_count = 0
        error_count = 0

        for i, smiles in enumerate(molecules):
            try:
                if descriptor == "molecular_weight":
                    value = self._estimate_molecular_weight(smiles)
                elif descriptor == "atom_count":
                    value = len([c for c in smiles if c.isupper()])
                elif descriptor == "bond_count":
                    value = smiles.count("-") + smiles.count("=") + smiles.count("#")
                else:
                    raise ValueError(f"Unknown descriptor: {descriptor}")

                values.append(float(value))
                success_count += 1

            except Exception as e:
                values.append(float("nan"))
                errors.append(f"Error calculating {descriptor} at index {i}: {str(e)}")
                error_count += 1

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        result = ReferenceResult(
            implementation=self.name,
            descriptor=descriptor,
            values=values,
            execution_time=execution_time,
            success_count=success_count,
            error_count=error_count,
            errors=errors,
        )

        logger.info(
            f"CPU reference {descriptor}: {success_count}/{len(molecules)} successful, "
            f"{execution_time:.3f}s"
        )

        return result


def get_reference_implementations() -> Dict[str, ReferenceImplementation]:
    """Get available reference implementations.

    Returns:
        Dictionary mapping implementation names to instances
    """
    implementations = {}

    # Try to initialize RDKit
    if RDKIT_AVAILABLE:
        try:
            implementations["rdkit"] = RDKitReference()
        except Exception as e:
            logger.warning(f"Failed to initialize RDKit reference: {e}")

    # Try to initialize Mordred
    if MORDRED_AVAILABLE:
        try:
            implementations["mordred"] = MordredReference()
        except Exception as e:
            logger.warning(f"Failed to initialize Mordred reference: {e}")

    # Always include CPU reference
    try:
        implementations["cpu_reference"] = CPUReference()
    except Exception as e:
        logger.warning(f"Failed to initialize CPU reference: {e}")

    logger.info(f"Available reference implementations: {list(implementations.keys())}")

    return implementations

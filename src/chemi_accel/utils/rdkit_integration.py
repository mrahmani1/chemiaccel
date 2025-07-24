"""
RDKit integration utilities for ChemiAccel.

This module provides comprehensive RDKit-based molecular parsing,
validation, and basic property extraction capabilities.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# RDKit imports with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    from rdkit.Chem import AllChem, rdchem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit import RDLogger
    
    # Suppress RDKit warnings by default
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
    
except ImportError as e:
    RDKIT_AVAILABLE = False
    _RDKIT_IMPORT_ERROR = str(e)

from ..utils.logging import get_logger
from ..core.exceptions import InvalidMoleculeError, ChemiAccelError
from ..core.molecular import MolecularData


logger = get_logger()


def check_rdkit_availability() -> None:
    """Check if RDKit is available and raise error if not."""
    if not RDKIT_AVAILABLE:
        raise ChemiAccelError(
            f"RDKit is not available: {_RDKIT_IMPORT_ERROR}. "
            "Please install RDKit: conda install -c conda-forge rdkit"
        )


class RDKitMolecularParser:
    """RDKit-based molecular parser and validator."""
    
    def __init__(self, 
                 sanitize: bool = True,
                 remove_hs: bool = True,
                 standardize: bool = False):
        """Initialize RDKit molecular parser.
        
        Args:
            sanitize: Whether to sanitize molecules during parsing
            remove_hs: Whether to remove explicit hydrogens
            standardize: Whether to standardize molecules (tautomers, etc.)
        """
        check_rdkit_availability()
        
        self.sanitize = sanitize
        self.remove_hs = remove_hs
        self.standardize = standardize
        
        # Initialize standardizer if needed
        if self.standardize:
            try:
                self.standardizer = rdMolStandardize.Standardizer()
                self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
            except AttributeError:
                # Fallback for different RDKit versions
                logger.warning("RDKit standardization not available in this version")
                self.standardize = False
        
        logger.debug(f"RDKit parser initialized: sanitize={sanitize}, "
                    f"remove_hs={remove_hs}, standardize={standardize}")
    
    def parse_smiles(self, smiles: str, mol_id: Optional[Union[str, int]] = None) -> MolecularData:
        """Parse SMILES string into MolecularData.
        
        Args:
            smiles: SMILES string to parse
            mol_id: Optional molecule identifier
            
        Returns:
            MolecularData with RDKit mol object and basic properties
            
        Raises:
            InvalidMoleculeError: If SMILES parsing fails
        """
        if not isinstance(smiles, str) or not smiles.strip():
            raise InvalidMoleculeError(
                molecule_id=mol_id or smiles,
                reason="SMILES string is empty or not a string",
                suggestions=["Provide a valid SMILES string"]
            )
        
        smiles = smiles.strip()
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise InvalidMoleculeError(
                    molecule_id=mol_id or smiles,
                    reason="RDKit failed to parse SMILES",
                    suggestions=[
                        "Check SMILES syntax",
                        "Verify chemical validity",
                        "Try molecular standardization"
                    ]
                )
            
            # Apply processing options
            if self.sanitize:
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    raise InvalidMoleculeError(
                        molecule_id=mol_id or smiles,
                        reason=f"Molecule sanitization failed: {e}",
                        suggestions=["Check for unusual valences or structures"]
                    )
            
            if self.remove_hs:
                mol = Chem.RemoveHs(mol)
            
            if self.standardize and hasattr(self, 'standardizer'):
                try:
                    mol = self.standardizer.standardize(mol)
                    # Get canonical tautomer
                    mol = self.tautomer_enumerator.Canonicalize(mol)
                except Exception as e:
                    logger.warning(f"Standardization failed for {smiles}: {e}")
            
            # Extract basic properties
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            molecular_weight = Descriptors.MolWt(mol)
            
            # Create MolecularData with RDKit mol object
            mol_data = MolecularData(
                smiles=Chem.MolToSmiles(mol),  # Use canonical SMILES
                mol_id=mol_id,
                num_atoms=num_atoms,
                num_bonds=num_bonds,
                molecular_weight=molecular_weight,
                backend_data={"rdkit_mol": mol}
            )
            
            logger.debug(f"Successfully parsed molecule: {mol_id or smiles} "
                        f"({num_atoms} atoms, {num_bonds} bonds)")
            
            return mol_data
            
        except InvalidMoleculeError:
            raise
        except Exception as e:
            raise InvalidMoleculeError(
                molecule_id=mol_id or smiles,
                reason=f"Unexpected error during parsing: {e}",
                suggestions=["Check SMILES format and chemical validity"]
            )
    
    def parse_smiles_batch(self, 
                          smiles_list: List[str], 
                          mol_ids: Optional[List[Union[str, int]]] = None,
                          error_handling: str = "skip") -> List[MolecularData]:
        """Parse batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            mol_ids: Optional list of molecule identifiers
            error_handling: How to handle errors ("skip", "raise", "collect")
            
        Returns:
            List of successfully parsed MolecularData objects
            
        Raises:
            ChemiAccelError: If error_handling="raise" and parsing fails
        """
        if mol_ids is None:
            mol_ids = [None] * len(smiles_list)
        elif len(mol_ids) != len(smiles_list):
            raise ChemiAccelError(
                f"mol_ids length ({len(mol_ids)}) must match "
                f"smiles_list length ({len(smiles_list)})"
            )
        
        molecules = []
        errors = []
        
        for i, (smiles, mol_id) in enumerate(zip(smiles_list, mol_ids)):
            try:
                mol_data = self.parse_smiles(smiles, mol_id or i)
                molecules.append(mol_data)
            except InvalidMoleculeError as e:
                error_info = {
                    "index": i,
                    "smiles": smiles,
                    "mol_id": mol_id,
                    "error": str(e)
                }
                errors.append(error_info)
                
                if error_handling == "raise":
                    raise ChemiAccelError(f"Failed to parse molecule at index {i}: {e}")
                elif error_handling == "skip":
                    logger.warning(f"Skipping invalid molecule at index {i}: {e}")
                    continue
        
        if error_handling == "collect" and errors:
            logger.warning(f"Failed to parse {len(errors)} out of {len(smiles_list)} molecules")
            # Store errors in first molecule's backend_data for reference
            if molecules:
                molecules[0].backend_data["parsing_errors"] = errors
        
        logger.info(f"Successfully parsed {len(molecules)} out of {len(smiles_list)} molecules")
        return molecules
    
    def parse_sdf_file(self, file_path: Union[str, Path]) -> List[MolecularData]:
        """Parse SDF file into list of MolecularData.
        
        Args:
            file_path: Path to SDF file
            
        Returns:
            List of MolecularData objects
            
        Raises:
            ChemiAccelError: If file reading fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ChemiAccelError(f"SDF file not found: {file_path}")
        
        try:
            molecules = []
            supplier = Chem.SDMolSupplier(str(file_path))
            
            for i, mol in enumerate(supplier):
                if mol is None:
                    logger.warning(f"Failed to parse molecule at index {i} in SDF file")
                    continue
                
                # Extract properties
                mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
                smiles = Chem.MolToSmiles(mol)
                
                mol_data = MolecularData(
                    smiles=smiles,
                    mol_id=mol_id,
                    num_atoms=mol.GetNumAtoms(),
                    num_bonds=mol.GetNumBonds(),
                    molecular_weight=Descriptors.MolWt(mol),
                    backend_data={"rdkit_mol": mol}
                )
                molecules.append(mol_data)
            
            logger.info(f"Parsed {len(molecules)} molecules from SDF file: {file_path}")
            return molecules
            
        except Exception as e:
            raise ChemiAccelError(f"Failed to parse SDF file {file_path}: {e}")
    
    def validate_molecule(self, mol_data: MolecularData) -> Tuple[bool, List[str]]:
        """Validate molecular data using RDKit.
        
        Args:
            mol_data: MolecularData to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if RDKit mol object exists
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            issues.append("No RDKit mol object found")
            return False, issues
        
        # Basic structural validation
        try:
            # Check for valid structure
            if rdkit_mol.GetNumAtoms() == 0:
                issues.append("Molecule has no atoms")
            
            # Check for unusual valences
            for atom in rdkit_mol.GetAtoms():
                if atom.GetTotalValence() > 8:  # Reasonable upper limit
                    issues.append(f"Unusual valence for atom {atom.GetIdx()}: {atom.GetTotalValence()}")
            
            # Check for disconnected fragments
            fragments = Chem.GetMolFrags(rdkit_mol)
            if len(fragments) > 1:
                issues.append(f"Molecule has {len(fragments)} disconnected fragments")
            
            # Try to generate 2D coordinates (basic structure check)
            try:
                AllChem.Compute2DCoords(rdkit_mol)
            except Exception as e:
                issues.append(f"Failed to generate 2D coordinates: {e}")
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class RDKitDescriptorCalculator:
    """Calculate basic molecular descriptors using RDKit."""
    
    def __init__(self):
        """Initialize RDKit descriptor calculator."""
        check_rdkit_availability()
        
        # Define available descriptors
        self.descriptor_functions = {
            # Basic molecular properties
            "molecular_weight": Descriptors.MolWt,
            "num_atoms": lambda mol: mol.GetNumAtoms(),
            "num_bonds": lambda mol: mol.GetNumBonds(),
            "num_heavy_atoms": Descriptors.HeavyAtomCount,
            
            # Lipophilicity and solubility
            "logp": Crippen.MolLogP,
            "tpsa": Descriptors.TPSA,
            
            # Structural features
            "num_rotatable_bonds": Descriptors.NumRotatableBonds,
            "num_aromatic_rings": Descriptors.NumAromaticRings,
            "num_saturated_rings": Descriptors.NumSaturatedRings,
            "num_aliphatic_rings": Descriptors.NumAliphaticRings,
            
            # Hydrogen bonding
            "num_hbd": Descriptors.NumHDonors,
            "num_hba": Descriptors.NumHAcceptors,
            
            # Ring information
            "ring_count": Descriptors.RingCount,
            "largest_ring_size": lambda mol: max([len(ring) for ring in mol.GetRingInfo().AtomRings()]) if mol.GetRingInfo().AtomRings() else 0,
            
            # Complexity measures
            "bertz_complexity": Descriptors.BertzCT,
            "fraction_csp3": Descriptors.FractionCSP3,
        }
        
        logger.debug(f"RDKit descriptor calculator initialized with {len(self.descriptor_functions)} descriptors")
    
    def calculate_descriptors(self, 
                            mol_data: MolecularData, 
                            descriptor_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate descriptors for a single molecule.
        
        Args:
            mol_data: MolecularData with RDKit mol object
            descriptor_names: List of descriptor names to calculate (None for all)
            
        Returns:
            Dictionary of descriptor values
            
        Raises:
            ChemiAccelError: If calculation fails
        """
        rdkit_mol = mol_data.backend_data.get("rdkit_mol")
        if rdkit_mol is None:
            raise ChemiAccelError(f"No RDKit mol object found for molecule {mol_data.mol_id}")
        
        if descriptor_names is None:
            descriptor_names = list(self.descriptor_functions.keys())
        
        descriptors = {}
        
        for desc_name in descriptor_names:
            if desc_name not in self.descriptor_functions:
                logger.warning(f"Unknown descriptor: {desc_name}")
                continue
            
            try:
                desc_func = self.descriptor_functions[desc_name]
                value = desc_func(rdkit_mol)
                descriptors[desc_name] = float(value) if value is not None else 0.0
            except Exception as e:
                logger.warning(f"Failed to calculate {desc_name} for molecule {mol_data.mol_id}: {e}")
                descriptors[desc_name] = 0.0
        
        return descriptors
    
    def calculate_descriptors_batch(self, 
                                  molecules: List[MolecularData],
                                  descriptor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate descriptors for batch of molecules.
        
        Args:
            molecules: List of MolecularData objects
            descriptor_names: List of descriptor names to calculate
            
        Returns:
            DataFrame with calculated descriptors
        """
        if not molecules:
            return pd.DataFrame()
        
        results = []
        
        for mol_data in molecules:
            try:
                descriptors = self.calculate_descriptors(mol_data, descriptor_names)
                descriptors["mol_id"] = mol_data.mol_id
                descriptors["smiles"] = mol_data.smiles
                results.append(descriptors)
            except Exception as e:
                logger.warning(f"Failed to calculate descriptors for molecule {mol_data.mol_id}: {e}")
                # Add row with NaN values
                descriptors = {name: np.nan for name in (descriptor_names or self.descriptor_functions.keys())}
                descriptors["mol_id"] = mol_data.mol_id
                descriptors["smiles"] = mol_data.smiles
                results.append(descriptors)
        
        df = pd.DataFrame(results)
        
        # Reorder columns to put identifiers first
        id_cols = ["mol_id", "smiles"]
        desc_cols = [col for col in df.columns if col not in id_cols]
        df = df[id_cols + desc_cols]
        
        logger.info(f"Calculated descriptors for {len(df)} molecules")
        return df
    
    def get_available_descriptors(self) -> List[str]:
        """Get list of available descriptor names.
        
        Returns:
            List of descriptor names
        """
        return list(self.descriptor_functions.keys())


# Convenience functions for easy access
def parse_smiles(smiles: str, mol_id: Optional[Union[str, int]] = None, **kwargs) -> MolecularData:
    """Parse single SMILES string (convenience function).
    
    Args:
        smiles: SMILES string to parse
        mol_id: Optional molecule identifier
        **kwargs: Additional arguments for RDKitMolecularParser
        
    Returns:
        MolecularData object
    """
    parser = RDKitMolecularParser(**kwargs)
    return parser.parse_smiles(smiles, mol_id)


def parse_smiles_batch(smiles_list: List[str], 
                      mol_ids: Optional[List[Union[str, int]]] = None,
                      **kwargs) -> List[MolecularData]:
    """Parse batch of SMILES strings (convenience function).
    
    Args:
        smiles_list: List of SMILES strings
        mol_ids: Optional list of molecule identifiers
        **kwargs: Additional arguments for RDKitMolecularParser
        
    Returns:
        List of MolecularData objects
    """
    parser = RDKitMolecularParser(**kwargs)
    return parser.parse_smiles_batch(smiles_list, mol_ids)


def calculate_basic_descriptors(molecules: List[MolecularData]) -> pd.DataFrame:
    """Calculate basic descriptors for molecules (convenience function).
    
    Args:
        molecules: List of MolecularData objects with RDKit mol objects
        
    Returns:
        DataFrame with calculated descriptors
    """
    calculator = RDKitDescriptorCalculator()
    return calculator.calculate_descriptors_batch(molecules)


def parse_smiles_safe(smiles: str) -> Optional[Chem.Mol]:
    """
    Safely parse SMILES string and return RDKit Mol object.
    
    Args:
        smiles: SMILES string to parse
        
    Returns:
        RDKit Mol object or None if parsing fails
    """
    if not RDKIT_AVAILABLE:
        return None
    
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            # Try to sanitize the molecule
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None
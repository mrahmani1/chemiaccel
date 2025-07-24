"""
Molecular data structures and representations for ChemiAccel.

This module defines the core molecular data structures used across
different backends for consistent molecular representation.
"""

from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
import numpy as np

from ..utils.logging import get_logger
from .exceptions import InvalidMoleculeError, ValidationError


logger = get_logger()


@dataclass
class MolecularData:
    """Core molecular data structure for ChemiAccel.
    
    This class provides a unified interface for molecular data across
    different backends while maintaining backend-specific optimizations.
    """
    
    # Molecular identifiers
    smiles: Optional[str] = None
    mol_id: Optional[Union[str, int]] = None
    
    # Basic molecular properties
    num_atoms: Optional[int] = None
    num_bonds: Optional[int] = None
    molecular_weight: Optional[float] = None
    
    # Structural representations (populated by backends)
    atom_features: Optional[np.ndarray] = None
    bond_features: Optional[np.ndarray] = None
    adjacency_matrix: Optional[np.ndarray] = None
    
    # Backend-specific data
    backend_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate molecular data after initialization."""
        if self.backend_data is None:
            self.backend_data = {}
        
        # Basic validation
        if self.smiles is None and self.atom_features is None:
            raise ValidationError(
                input_name="molecular_data",
                input_value=self,
                expected_type="SMILES string or atom features",
                reason="Either SMILES or structural data must be provided"
            )
    
    @classmethod
    def from_smiles(cls, smiles: str, mol_id: Optional[Union[str, int]] = None) -> 'MolecularData':
        """Create MolecularData from SMILES string.
        
        Args:
            smiles: SMILES representation of the molecule
            mol_id: Optional molecule identifier
            
        Returns:
            MolecularData instance
            
        Raises:
            InvalidMoleculeError: If SMILES is invalid
        """
        if not isinstance(smiles, str) or not smiles.strip():
            raise InvalidMoleculeError(
                molecule_id=mol_id or smiles,
                reason="SMILES string is empty or invalid",
                suggestions=["Provide a valid SMILES string"]
            )
        
        return cls(smiles=smiles.strip(), mol_id=mol_id)
    
    @classmethod
    def from_smiles_list(cls, smiles_list: List[str], mol_ids: Optional[List[Union[str, int]]] = None) -> List['MolecularData']:
        """Create list of MolecularData from SMILES list.
        
        Args:
            smiles_list: List of SMILES strings
            mol_ids: Optional list of molecule identifiers
            
        Returns:
            List of MolecularData instances
        """
        if mol_ids is None:
            mol_ids = [None] * len(smiles_list)
        elif len(mol_ids) != len(smiles_list):
            raise ValidationError(
                input_name="mol_ids",
                input_value=mol_ids,
                expected_type=f"list of length {len(smiles_list)}",
                reason="mol_ids length must match smiles_list length"
            )
        
        molecules = []
        for i, (smiles, mol_id) in enumerate(zip(smiles_list, mol_ids)):
            try:
                molecules.append(cls.from_smiles(smiles, mol_id or i))
            except InvalidMoleculeError as e:
                logger.warning(f"Skipping invalid molecule at index {i}: {e}")
                continue
        
        return molecules
    
    def is_valid(self) -> bool:
        """Check if molecular data is valid.
        
        Returns:
            True if molecular data is valid
        """
        try:
            # Basic structural validation
            if self.num_atoms is not None and self.num_atoms <= 0:
                return False
            
            if self.num_bonds is not None and self.num_bonds < 0:
                return False
            
            if self.molecular_weight is not None and self.molecular_weight <= 0:
                return False
            
            # Array shape validation
            if self.atom_features is not None:
                if self.atom_features.ndim != 2:
                    return False
                if self.num_atoms is not None and self.atom_features.shape[0] != self.num_atoms:
                    return False
            
            if self.bond_features is not None:
                if self.bond_features.ndim != 2:
                    return False
                if self.num_bonds is not None and self.bond_features.shape[0] != self.num_bonds:
                    return False
            
            if self.adjacency_matrix is not None:
                if self.adjacency_matrix.ndim != 2:
                    return False
                if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
                    return False
                if self.num_atoms is not None and self.adjacency_matrix.shape[0] != self.num_atoms:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of molecular data.
        
        Returns:
            Dictionary with molecular data summary
        """
        return {
            "mol_id": self.mol_id,
            "smiles": self.smiles,
            "num_atoms": self.num_atoms,
            "num_bonds": self.num_bonds,
            "molecular_weight": self.molecular_weight,
            "has_atom_features": self.atom_features is not None,
            "has_bond_features": self.bond_features is not None,
            "has_adjacency_matrix": self.adjacency_matrix is not None,
            "is_valid": self.is_valid()
        }
    
    def __repr__(self) -> str:
        """String representation of molecular data."""
        return f"MolecularData(mol_id={self.mol_id}, smiles='{self.smiles}', atoms={self.num_atoms})"


class MolecularBatch:
    """Container for batched molecular data processing.
    
    This class handles batching of molecular data for efficient
    processing across different backends.
    """
    
    def __init__(self, molecules: List[MolecularData], batch_size: Optional[int] = None):
        """Initialize molecular batch.
        
        Args:
            molecules: List of molecular data
            batch_size: Optional batch size for processing
        """
        self.molecules = molecules
        self.batch_size = batch_size or len(molecules)
        self._validate_batch()
    
    def _validate_batch(self) -> None:
        """Validate batch consistency."""
        if not self.molecules:
            raise ValidationError(
                input_name="molecules",
                input_value=self.molecules,
                expected_type="non-empty list",
                reason="Batch cannot be empty"
            )
        
        # Check for consistent data types
        valid_molecules = [mol for mol in self.molecules if mol.is_valid()]
        if len(valid_molecules) != len(self.molecules):
            invalid_count = len(self.molecules) - len(valid_molecules)
            logger.warning(f"Batch contains {invalid_count} invalid molecules")
    
    def get_batches(self) -> List[List[MolecularData]]:
        """Split molecules into batches.
        
        Returns:
            List of molecular data batches
        """
        batches = []
        for i in range(0, len(self.molecules), self.batch_size):
            batch = self.molecules[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def get_smiles_list(self) -> List[str]:
        """Get list of SMILES strings from batch.
        
        Returns:
            List of SMILES strings
        """
        return [mol.smiles for mol in self.molecules if mol.smiles is not None]
    
    def get_valid_molecules(self) -> List[MolecularData]:
        """Get only valid molecules from batch.
        
        Returns:
            List of valid molecular data
        """
        return [mol for mol in self.molecules if mol.is_valid()]
    
    def __len__(self) -> int:
        """Get number of molecules in batch."""
        return len(self.molecules)
    
    def __getitem__(self, index: int) -> MolecularData:
        """Get molecule by index."""
        return self.molecules[index]
    
    def __repr__(self) -> str:
        """String representation of molecular batch."""
        return f"MolecularBatch(size={len(self.molecules)}, batch_size={self.batch_size})"
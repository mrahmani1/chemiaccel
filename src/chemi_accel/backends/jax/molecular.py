"""
JAX molecular processing functions and data structures.

This module implements JAX-based molecular representations and processing
functions using padded arrays and masking for variable-sized molecules.
"""

from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import Array
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create dummy types for type hints when JAX is not available
    Array = Any
    jnp = None

from ...core.molecular import MolecularData, MolecularBatch
from ...core.exceptions import BackendUnavailableError, ValidationError
from ...utils.logging import get_logger

logger = get_logger()


@dataclass
class JAXMolecule:
    """JAX-based molecular representation with padded arrays.
    
    This class represents molecules using JAX arrays with padding and masking
    to handle variable-sized molecules efficiently in batch operations.
    """
    
    # Core molecular arrays (padded to max_atoms/max_bonds)
    atom_features: Array  # (max_atoms, num_atom_features)
    bond_features: Array  # (max_bonds, num_bond_features)
    adjacency_matrix: Array  # (max_atoms, max_atoms)
    
    # Masks for valid atoms/bonds
    atom_mask: Array  # (max_atoms,) boolean mask
    bond_mask: Array  # (max_bonds,) boolean mask
    
    # Molecular metadata
    num_atoms: int
    num_bonds: int
    mol_id: Optional[Union[str, int]] = None
    smiles: Optional[str] = None
    
    def __post_init__(self):
        """Validate JAX molecular data after initialization."""
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")
        
        # Validate array shapes
        max_atoms = self.atom_features.shape[0]
        max_bonds = self.bond_features.shape[0]
        
        if self.adjacency_matrix.shape != (max_atoms, max_atoms):
            raise ValidationError(
                input_name="adjacency_matrix",
                input_value=self.adjacency_matrix.shape,
                expected_type=f"({max_atoms}, {max_atoms})",
                reason="Adjacency matrix shape must match atom features"
            )
        
        if self.atom_mask.shape != (max_atoms,):
            raise ValidationError(
                input_name="atom_mask",
                input_value=self.atom_mask.shape,
                expected_type=f"({max_atoms},)",
                reason="Atom mask shape must match atom features"
            )
        
        if self.bond_mask.shape != (max_bonds,):
            raise ValidationError(
                input_name="bond_mask", 
                input_value=self.bond_mask.shape,
                expected_type=f"({max_bonds},)",
                reason="Bond mask shape must match bond features"
            )
    
    @property
    def actual_num_atoms(self) -> int:
        """Get actual number of atoms (sum of atom mask)."""
        return int(jnp.sum(self.atom_mask))
    
    @property
    def actual_num_bonds(self) -> int:
        """Get actual number of bonds (sum of bond mask)."""
        return int(jnp.sum(self.bond_mask))
    
    def get_valid_atom_features(self) -> Array:
        """Get atom features for valid atoms only."""
        return self.atom_features[self.atom_mask]
    
    def get_valid_bond_features(self) -> Array:
        """Get bond features for valid bonds only."""
        return self.bond_features[self.bond_mask]
    
    def get_valid_adjacency(self) -> Array:
        """Get adjacency matrix for valid atoms only."""
        valid_indices = jnp.where(self.atom_mask)[0]
        return self.adjacency_matrix[jnp.ix_(valid_indices, valid_indices)]


@dataclass
class JAXMolecularBatch:
    """Batched JAX molecular representation for efficient processing.
    
    This class handles batches of molecules with consistent padding
    for efficient vectorized operations.
    """
    
    # Batched molecular arrays (batch_size, max_atoms/bonds, features)
    atom_features: Array  # (batch_size, max_atoms, num_atom_features)
    bond_features: Array  # (batch_size, max_bonds, num_bond_features)
    adjacency_matrices: Array  # (batch_size, max_atoms, max_atoms)
    
    # Batched masks
    atom_masks: Array  # (batch_size, max_atoms)
    bond_masks: Array  # (batch_size, max_bonds)
    
    # Batch metadata
    batch_size: int
    max_atoms: int
    max_bonds: int
    molecule_sizes: Array  # (batch_size,) actual number of atoms per molecule
    
    # Optional metadata
    mol_ids: Optional[List[Union[str, int]]] = None
    smiles_list: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate batched molecular data."""
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")
        
        # Validate batch dimensions
        expected_shapes = {
            "atom_features": (self.batch_size, self.max_atoms, self.atom_features.shape[2]),
            "bond_features": (self.batch_size, self.max_bonds, self.bond_features.shape[2]),
            "adjacency_matrices": (self.batch_size, self.max_atoms, self.max_atoms),
            "atom_masks": (self.batch_size, self.max_atoms),
            "bond_masks": (self.batch_size, self.max_bonds),
            "molecule_sizes": (self.batch_size,)
        }
        
        for name, expected_shape in expected_shapes.items():
            actual_shape = getattr(self, name).shape
            if actual_shape != expected_shape:
                raise ValidationError(
                    input_name=name,
                    input_value=actual_shape,
                    expected_type=str(expected_shape),
                    reason=f"Shape mismatch for {name}"
                )
    
    def get_molecule(self, index: int) -> JAXMolecule:
        """Extract a single molecule from the batch.
        
        Args:
            index: Index of molecule in batch
            
        Returns:
            JAXMolecule instance
        """
        if index >= self.batch_size:
            raise IndexError(f"Index {index} out of range for batch size {self.batch_size}")
        
        return JAXMolecule(
            atom_features=self.atom_features[index],
            bond_features=self.bond_features[index],
            adjacency_matrix=self.adjacency_matrices[index],
            atom_mask=self.atom_masks[index],
            bond_mask=self.bond_masks[index],
            num_atoms=int(self.molecule_sizes[index]),
            num_bonds=int(jnp.sum(self.bond_masks[index])),
            mol_id=self.mol_ids[index] if self.mol_ids else None,
            smiles=self.smiles_list[index] if self.smiles_list else None
        )
    
    def __len__(self) -> int:
        """Get batch size."""
        return self.batch_size
    
    def __getitem__(self, index: int) -> JAXMolecule:
        """Get molecule by index."""
        return self.get_molecule(index)


class JAXMolecularProcessor:
    """JAX-based molecular processing utilities.
    
    This class provides utilities for converting molecular data to JAX
    representations and handling padding/masking operations.
    """
    
    def __init__(self, max_atoms: int = 100, max_bonds: int = 200):
        """Initialize JAX molecular processor.
        
        Args:
            max_atoms: Maximum number of atoms for padding
            max_bonds: Maximum number of bonds for padding
        """
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")
        
        self.max_atoms = max_atoms
        self.max_bonds = max_bonds
        
        # Standard atom features (can be extended)
        self.atom_feature_names = [
            'atomic_number', 'formal_charge', 'num_hydrogens', 
            'hybridization', 'is_aromatic', 'is_in_ring'
        ]
        self.num_atom_features = len(self.atom_feature_names)
        
        # Standard bond features
        self.bond_feature_names = [
            'bond_type', 'is_conjugated', 'is_in_ring', 'is_aromatic'
        ]
        self.num_bond_features = len(self.bond_feature_names)
        
        logger.info(f"Initialized JAX processor with max_atoms={max_atoms}, max_bonds={max_bonds}")
    
    def create_padded_arrays(self, 
                           atom_features: np.ndarray,
                           bond_features: np.ndarray,
                           adjacency_matrix: np.ndarray) -> Tuple[Array, Array, Array, Array, Array]:
        """Create padded JAX arrays from molecular data.
        
        Args:
            atom_features: Atom feature matrix (num_atoms, num_features)
            bond_features: Bond feature matrix (num_bonds, num_features)
            adjacency_matrix: Adjacency matrix (num_atoms, num_atoms)
            
        Returns:
            Tuple of (padded_atoms, padded_bonds, padded_adjacency, atom_mask, bond_mask)
        """
        num_atoms, num_bonds = atom_features.shape[0], bond_features.shape[0]
        
        # Validate input sizes
        if num_atoms > self.max_atoms:
            raise ValidationError(
                input_name="atom_features",
                input_value=num_atoms,
                expected_type=f"<= {self.max_atoms}",
                reason=f"Molecule has {num_atoms} atoms, exceeds max_atoms={self.max_atoms}"
            )
        
        if num_bonds > self.max_bonds:
            raise ValidationError(
                input_name="bond_features",
                input_value=num_bonds,
                expected_type=f"<= {self.max_bonds}",
                reason=f"Molecule has {num_bonds} bonds, exceeds max_bonds={self.max_bonds}"
            )
        
        # Create padded atom features
        padded_atoms = jnp.zeros((self.max_atoms, self.num_atom_features))
        padded_atoms = padded_atoms.at[:num_atoms].set(atom_features)
        
        # Create padded bond features
        padded_bonds = jnp.zeros((self.max_bonds, self.num_bond_features))
        padded_bonds = padded_bonds.at[:num_bonds].set(bond_features)
        
        # Create padded adjacency matrix
        padded_adjacency = jnp.zeros((self.max_atoms, self.max_atoms))
        padded_adjacency = padded_adjacency.at[:num_atoms, :num_atoms].set(adjacency_matrix)
        
        # Create masks
        atom_mask = jnp.zeros(self.max_atoms, dtype=bool)
        atom_mask = atom_mask.at[:num_atoms].set(True)
        
        bond_mask = jnp.zeros(self.max_bonds, dtype=bool)
        bond_mask = bond_mask.at[:num_bonds].set(True)
        
        return padded_atoms, padded_bonds, padded_adjacency, atom_mask, bond_mask
    
    def molecular_data_to_jax(self, mol_data: MolecularData) -> JAXMolecule:
        """Convert MolecularData to JAX representation.
        
        Args:
            mol_data: MolecularData instance
            
        Returns:
            JAXMolecule instance
        """
        if mol_data.atom_features is None:
            raise ValidationError(
                input_name="mol_data.atom_features",
                input_value=None,
                expected_type="numpy.ndarray",
                reason="Atom features must be populated for JAX conversion"
            )
        
        # Create padded arrays
        bond_features = mol_data.bond_features if mol_data.bond_features is not None else np.zeros((0, self.num_bond_features))
        adjacency_matrix = mol_data.adjacency_matrix if mol_data.adjacency_matrix is not None else np.zeros((mol_data.num_atoms or 0, mol_data.num_atoms or 0))
        
        padded_atoms, padded_bonds, padded_adjacency, atom_mask, bond_mask = \
            self.create_padded_arrays(
                mol_data.atom_features,
                bond_features,
                adjacency_matrix
            )
        
        return JAXMolecule(
            atom_features=padded_atoms,
            bond_features=padded_bonds,
            adjacency_matrix=padded_adjacency,
            atom_mask=atom_mask,
            bond_mask=bond_mask,
            num_atoms=mol_data.num_atoms or 0,
            num_bonds=mol_data.num_bonds or 0,
            mol_id=mol_data.mol_id,
            smiles=mol_data.smiles
        )
    
    def create_batch(self, molecules: List[MolecularData]) -> JAXMolecularBatch:
        """Create batched JAX representation from list of molecules.
        
        Args:
            molecules: List of MolecularData instances
            
        Returns:
            JAXMolecularBatch instance
        """
        if not molecules:
            raise ValidationError(
                input_name="molecules",
                input_value=molecules,
                expected_type="non-empty list",
                reason="Cannot create batch from empty molecule list"
            )
        
        batch_size = len(molecules)
        
        # Convert all molecules to JAX format
        jax_molecules = [self.molecular_data_to_jax(mol) for mol in molecules]
        
        # Stack into batch arrays
        atom_features_batch = jnp.stack([mol.atom_features for mol in jax_molecules])
        bond_features_batch = jnp.stack([mol.bond_features for mol in jax_molecules])
        adjacency_batch = jnp.stack([mol.adjacency_matrix for mol in jax_molecules])
        atom_masks_batch = jnp.stack([mol.atom_mask for mol in jax_molecules])
        bond_masks_batch = jnp.stack([mol.bond_mask for mol in jax_molecules])
        
        # Create molecule sizes array
        molecule_sizes = jnp.array([mol.num_atoms for mol in jax_molecules])
        
        # Extract metadata
        mol_ids = [mol.mol_id for mol in jax_molecules]
        smiles_list = [mol.smiles for mol in jax_molecules]
        
        return JAXMolecularBatch(
            atom_features=atom_features_batch,
            bond_features=bond_features_batch,
            adjacency_matrices=adjacency_batch,
            atom_masks=atom_masks_batch,
            bond_masks=bond_masks_batch,
            batch_size=batch_size,
            max_atoms=self.max_atoms,
            max_bonds=self.max_bonds,
            molecule_sizes=molecule_sizes,
            mol_ids=mol_ids,
            smiles_list=smiles_list
        )
    
    def optimize_batch_size(self, molecules: List[MolecularData]) -> int:
        """Determine optimal batch size based on molecule sizes.
        
        Args:
            molecules: List of molecules to analyze
            
        Returns:
            Recommended batch size
        """
        if not molecules:
            return 1
        
        # Analyze molecule sizes
        atom_counts = [mol.num_atoms or 0 for mol in molecules if mol.num_atoms]
        if not atom_counts:
            return len(molecules)
        
        avg_atoms = np.mean(atom_counts)
        max_atoms = max(atom_counts)
        
        # Heuristic for batch size based on memory considerations
        # Smaller molecules can have larger batches
        if avg_atoms < 20:
            recommended_batch = min(64, len(molecules))
        elif avg_atoms < 50:
            recommended_batch = min(32, len(molecules))
        else:
            recommended_batch = min(16, len(molecules))
        
        logger.debug(f"Recommended batch size: {recommended_batch} "
                    f"(avg_atoms={avg_atoms:.1f}, max_atoms={max_atoms})")
        
        return recommended_batch


def create_synthetic_molecule(num_atoms: int = 10, 
                            num_bonds: int = 9,
                            mol_id: Optional[Union[str, int]] = None,
                            smiles: Optional[str] = None) -> MolecularData:
    """Create synthetic molecular data for testing.
    
    Args:
        num_atoms: Number of atoms
        num_bonds: Number of bonds
        mol_id: Optional molecule ID
        smiles: Optional SMILES string
        
    Returns:
        MolecularData with synthetic features
    """
    # Create synthetic atom features (atomic_number, formal_charge, etc.)
    atom_features = np.random.randint(1, 18, size=(num_atoms, 6)).astype(np.float32)
    atom_features[:, 1] = np.random.randint(-2, 3, size=num_atoms)  # formal charge
    atom_features[:, 2] = np.random.randint(0, 4, size=num_atoms)   # num hydrogens
    atom_features[:, 3] = np.random.randint(1, 5, size=num_atoms)   # hybridization
    atom_features[:, 4] = np.random.choice([0, 1], size=num_atoms)  # is_aromatic
    atom_features[:, 5] = np.random.choice([0, 1], size=num_atoms)  # is_in_ring
    
    # Create synthetic bond features
    bond_features = np.random.randint(1, 4, size=(num_bonds, 4)).astype(np.float32)
    bond_features[:, 1] = np.random.choice([0, 1], size=num_bonds)  # is_conjugated
    bond_features[:, 2] = np.random.choice([0, 1], size=num_bonds)  # is_in_ring
    bond_features[:, 3] = np.random.choice([0, 1], size=num_bonds)  # is_aromatic
    
    # Create synthetic adjacency matrix (simple chain for testing)
    adjacency = np.zeros((num_atoms, num_atoms))
    for i in range(min(num_bonds, num_atoms - 1)):
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1
    
    return MolecularData(
        smiles=smiles or f"synthetic_{num_atoms}atoms",
        mol_id=mol_id,
        num_atoms=num_atoms,
        num_bonds=num_bonds,
        molecular_weight=float(num_atoms * 12.0),  # Rough estimate
        atom_features=atom_features,
        bond_features=bond_features,
        adjacency_matrix=adjacency
    )
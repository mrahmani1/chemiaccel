"""
JAX-based molecular descriptor calculations.

This module implements molecular descriptor algorithms using JAX NumPy API
with support for padding, masking, and batch processing.
"""

from typing import Dict, Any, Optional, Callable
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import Array
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    Array = Any
    jnp = None

from .molecular import JAXMolecule, JAXMolecularBatch
from ...core.exceptions import BackendUnavailableError, DescriptorCalculationError
from ...utils.logging import get_logger

logger = get_logger()


class JAXDescriptorCalculator:
    """JAX-based molecular descriptor calculator.
    
    This class implements molecular descriptor calculations using JAX
    with support for JIT compilation and automatic vectorization.
    """
    
    def __init__(self):
        """Initialize JAX descriptor calculator."""
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")
        
        # Compile descriptor functions for better performance
        self._compiled_functions = {}
        self._compile_descriptor_functions()
        
        logger.info("Initialized JAX descriptor calculator")
    
    def _compile_descriptor_functions(self):
        """Pre-compile descriptor functions with JIT."""
        try:
            # Compile single molecule descriptor functions
            self._compiled_functions['molecular_weight'] = jax.jit(self._molecular_weight_single)
            self._compiled_functions['atom_count'] = jax.jit(self._atom_count_single)
            self._compiled_functions['bond_count'] = jax.jit(self._bond_count_single)
            self._compiled_functions['ring_count'] = jax.jit(self._ring_count_single)
            self._compiled_functions['aromatic_atom_count'] = jax.jit(self._aromatic_atom_count_single)
            self._compiled_functions['formal_charge'] = jax.jit(self._formal_charge_single)
            
            # Compile batch descriptor functions
            self._compiled_functions['batch_basic_descriptors'] = jax.jit(self._batch_basic_descriptors)
            
            # Compile fingerprint function
            self._compiled_functions['molecular_fingerprint'] = jax.jit(self._molecular_fingerprint_single)
            
            logger.debug("Successfully compiled JAX descriptor functions")
            
        except Exception as e:
            logger.warning(f"Failed to compile some descriptor functions: {e}")
    
    def _molecular_weight_single(self, molecule: JAXMolecule) -> float:
        """Calculate molecular weight for a single molecule.
        
        Args:
            molecule: JAX molecule representation
            
        Returns:
            Molecular weight
        """
        # Atomic weights (simplified - using atomic numbers as proxy)
        atomic_weights = jnp.array([
            0.0, 1.008, 4.003, 6.94, 9.012, 10.81, 12.01, 14.01, 16.00, 19.00, 20.18,
            22.99, 24.31, 26.98, 28.09, 30.97, 32.07, 35.45, 39.95, 39.10, 40.08
        ])
        
        # Get atomic numbers from atom features (first column)
        atomic_numbers = molecule.atom_features[:, 0].astype(int)
        
        # Clip atomic numbers to valid range
        atomic_numbers = jnp.clip(atomic_numbers, 0, len(atomic_weights) - 1)
        
        # Calculate molecular weight using mask
        weights = atomic_weights[atomic_numbers]
        molecular_weight = jnp.sum(weights * molecule.atom_mask)
        
        return molecular_weight
    
    def _atom_count_single(self, molecule: JAXMolecule) -> int:
        """Count atoms in a single molecule.
        
        Args:
            molecule: JAX molecule representation
            
        Returns:
            Number of atoms
        """
        return jnp.sum(molecule.atom_mask)
    
    def _bond_count_single(self, molecule: JAXMolecule) -> int:
        """Count bonds in a single molecule.
        
        Args:
            molecule: JAX molecule representation
            
        Returns:
            Number of bonds
        """
        return jnp.sum(molecule.bond_mask)
    
    def _ring_count_single(self, molecule: JAXMolecule) -> int:
        """Count atoms in rings for a single molecule.
        
        Args:
            molecule: JAX molecule representation
            
        Returns:
            Number of atoms in rings
        """
        # Assuming ring information is in the 6th column (index 5) of atom features
        is_in_ring = molecule.atom_features[:, 5]
        ring_atoms = jnp.sum(is_in_ring * molecule.atom_mask)
        return ring_atoms
    
    def _aromatic_atom_count_single(self, molecule: JAXMolecule) -> int:
        """Count aromatic atoms in a single molecule.
        
        Args:
            molecule: JAX molecule representation
            
        Returns:
            Number of aromatic atoms
        """
        # Assuming aromatic information is in the 5th column (index 4) of atom features
        is_aromatic = molecule.atom_features[:, 4]
        aromatic_atoms = jnp.sum(is_aromatic * molecule.atom_mask)
        return aromatic_atoms
    
    def _formal_charge_single(self, molecule: JAXMolecule) -> float:
        """Calculate total formal charge for a single molecule.
        
        Args:
            molecule: JAX molecule representation
            
        Returns:
            Total formal charge
        """
        # Formal charge is in the 2nd column (index 1) of atom features
        formal_charges = molecule.atom_features[:, 1]
        total_charge = jnp.sum(formal_charges * molecule.atom_mask)
        return total_charge
    
    def _molecular_fingerprint_single(self, molecule: JAXMolecule, radius: int = 2) -> Array:
        """Calculate molecular fingerprint for a single molecule.
        
        Args:
            molecule: JAX molecule representation
            radius: Fingerprint radius
            
        Returns:
            Molecular fingerprint as JAX array
        """
        # Simplified Morgan-like fingerprint calculation
        atom_features = molecule.atom_features
        adj_matrix = molecule.adjacency_matrix
        atom_mask = molecule.atom_mask
        
        # Initialize atom identifiers from features
        atom_ids = jnp.sum(atom_features * jnp.array([1, 2, 4, 8, 16, 32]), axis=1)
        
        # Iteratively update identifiers based on neighbors
        for _ in range(radius):
            neighbor_sums = jnp.matmul(adj_matrix, atom_ids)
            atom_ids = atom_ids + neighbor_sums
            atom_ids = atom_ids * atom_mask  # Apply mask
        
        # Create fingerprint by hashing atom identifiers
        fingerprint_size = 1024
        fingerprint = jnp.zeros(fingerprint_size)
        
        # Simple hash function for demonstration
        valid_ids = atom_ids * atom_mask
        hash_indices = (valid_ids.astype(int) % fingerprint_size)
        
        # Count occurrences at each hash position
        for i in range(fingerprint_size):
            count = jnp.sum((hash_indices == i) * atom_mask)
            fingerprint = fingerprint.at[i].set(count)
        
        return fingerprint
    
    def _batch_basic_descriptors(self, batch: JAXMolecularBatch) -> Dict[str, Array]:
        """Calculate basic descriptors for a batch of molecules.
        
        Args:
            batch: Batch of JAX molecules
            
        Returns:
            Dictionary of descriptor arrays
        """
        # Vectorized calculations across the batch
        
        # Molecular weights
        atomic_weights = jnp.array([
            0.0, 1.008, 4.003, 6.94, 9.012, 10.81, 12.01, 14.01, 16.00, 19.00, 20.18,
            22.99, 24.31, 26.98, 28.09, 30.97, 32.07, 35.45, 39.95, 39.10, 40.08
        ])
        
        atomic_numbers = batch.atom_features[:, :, 0].astype(int)
        atomic_numbers = jnp.clip(atomic_numbers, 0, len(atomic_weights) - 1)
        weights = atomic_weights[atomic_numbers]
        molecular_weights = jnp.sum(weights * batch.atom_masks, axis=1)
        
        # Atom counts
        atom_counts = jnp.sum(batch.atom_masks, axis=1)
        
        # Bond counts
        bond_counts = jnp.sum(batch.bond_masks, axis=1)
        
        # Ring atom counts
        ring_atoms = jnp.sum(batch.atom_features[:, :, 5] * batch.atom_masks, axis=1)
        
        # Aromatic atom counts
        aromatic_atoms = jnp.sum(batch.atom_features[:, :, 4] * batch.atom_masks, axis=1)
        
        # Total formal charges
        formal_charges = jnp.sum(batch.atom_features[:, :, 1] * batch.atom_masks, axis=1)
        
        return {
            'molecular_weight': molecular_weights,
            'atom_count': atom_counts,
            'bond_count': bond_counts,
            'ring_atom_count': ring_atoms,
            'aromatic_atom_count': aromatic_atoms,
            'formal_charge': formal_charges
        }
    
    def calculate_single_molecule(self, molecule: JAXMolecule) -> Dict[str, Any]:
        """Calculate descriptors for a single molecule.
        
        Args:
            molecule: JAX molecule representation
            
        Returns:
            Dictionary of calculated descriptors
        """
        try:
            results = {}
            
            # Use non-compiled functions for single molecules (compiled functions are for batch processing)
            results['molecular_weight'] = float(self._molecular_weight_single(molecule))
            results['atom_count'] = int(self._atom_count_single(molecule))
            results['bond_count'] = int(self._bond_count_single(molecule))
            results['ring_atom_count'] = int(self._ring_count_single(molecule))
            results['aromatic_atom_count'] = int(self._aromatic_atom_count_single(molecule))
            results['formal_charge'] = float(self._formal_charge_single(molecule))
            
            # Add molecule metadata
            results['mol_id'] = molecule.mol_id
            results['smiles'] = molecule.smiles
            
            return results
            
        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="single_molecule_descriptors",
                molecule_id=molecule.mol_id or "unknown",
                reason=str(e)
            )
    
    def calculate_batch(self, batch: JAXMolecularBatch) -> Dict[str, np.ndarray]:
        """Calculate descriptors for a batch of molecules.
        
        Args:
            batch: Batch of JAX molecules
            
        Returns:
            Dictionary of descriptor arrays
        """
        try:
            # Use compiled batch function if available
            if 'batch_basic_descriptors' in self._compiled_functions:
                jax_results = self._compiled_functions['batch_basic_descriptors'](batch)
            else:
                jax_results = self._batch_basic_descriptors(batch)
            
            # Convert JAX arrays to numpy arrays
            results = {}
            for key, values in jax_results.items():
                results[key] = np.array(values)
            
            # Add metadata
            results['mol_ids'] = batch.mol_ids
            results['smiles_list'] = batch.smiles_list
            results['batch_size'] = batch.batch_size
            
            return results
            
        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="batch_descriptors",
                molecule_id=f"batch_size_{batch.batch_size}",
                reason=str(e)
            )
    
    def get_available_descriptors(self) -> Dict[str, str]:
        """Get list of available descriptors.
        
        Returns:
            Dictionary mapping descriptor names to descriptions
        """
        return {
            'molecular_weight': 'Molecular weight in Daltons',
            'atom_count': 'Total number of atoms',
            'bond_count': 'Total number of bonds',
            'ring_atom_count': 'Number of atoms in rings',
            'aromatic_atom_count': 'Number of aromatic atoms',
            'formal_charge': 'Total formal charge'
        }


# Advanced descriptor functions using JAX

def calculate_topological_indices(molecule: JAXMolecule) -> Dict[str, float]:
    """Calculate topological indices using JAX operations.
    
    Args:
        molecule: JAX molecule representation
        
    Returns:
        Dictionary of topological indices
    """
    if not JAX_AVAILABLE:
        raise BackendUnavailableError("JAX backend requires jax package")
    
    # Get valid adjacency matrix
    adj_matrix = molecule.adjacency_matrix
    atom_mask = molecule.atom_mask
    
    # Calculate degree matrix (number of connections per atom)
    degrees = jnp.sum(adj_matrix * atom_mask[:, None] * atom_mask[None, :], axis=1)
    valid_degrees = degrees * atom_mask
    
    # Wiener index (sum of shortest path distances)
    # Simplified calculation using powers of adjacency matrix
    wiener_index = 0.0
    current_paths = adj_matrix
    
    for path_length in range(1, int(jnp.sum(atom_mask))):
        # Count paths of current length
        path_counts = jnp.sum(current_paths * atom_mask[:, None] * atom_mask[None, :])
        wiener_index += path_length * path_counts
        
        # Update for next path length
        current_paths = jnp.matmul(current_paths, adj_matrix)
        
        # Stop if no more paths
        if jnp.sum(current_paths) == 0:
            break
    
    # Zagreb indices
    zagreb_1 = jnp.sum(valid_degrees ** 2)
    zagreb_2 = jnp.sum(jnp.sum(adj_matrix * degrees[:, None] * degrees[None, :], axis=1) * atom_mask)
    
    # Randic connectivity index
    degree_products = jnp.sqrt(degrees[:, None] * degrees[None, :])
    randic_index = jnp.sum(adj_matrix / jnp.where(degree_products > 0, degree_products, 1.0))
    
    return {
        'wiener_index': float(wiener_index),
        'zagreb_1': float(zagreb_1),
        'zagreb_2': float(zagreb_2),
        'randic_index': float(randic_index)
    }


def calculate_molecular_fingerprint(molecule: JAXMolecule, radius: int = 2) -> Array:
    """Calculate molecular fingerprint using JAX operations.
    
    Args:
        molecule: JAX molecule representation
        radius: Fingerprint radius
        
    Returns:
        Molecular fingerprint as JAX array
    """
    if not JAX_AVAILABLE:
        raise BackendUnavailableError("JAX backend requires jax package")
    
    # Simplified Morgan-like fingerprint calculation
    atom_features = molecule.atom_features
    adj_matrix = molecule.adjacency_matrix
    atom_mask = molecule.atom_mask
    
    # Initialize atom identifiers from features
    atom_ids = jnp.sum(atom_features * jnp.array([1, 2, 4, 8, 16, 32]), axis=1)
    
    # Iteratively update identifiers based on neighbors
    for _ in range(radius):
        neighbor_sums = jnp.matmul(adj_matrix, atom_ids)
        atom_ids = atom_ids + neighbor_sums
        atom_ids = atom_ids * atom_mask  # Apply mask
    
    # Create fingerprint by hashing atom identifiers
    fingerprint_size = 1024
    fingerprint = jnp.zeros(fingerprint_size)
    
    # Simple hash function for demonstration
    for i in range(len(atom_ids)):
        if atom_mask[i]:
            hash_val = int(atom_ids[i]) % fingerprint_size
            fingerprint = fingerprint.at[hash_val].add(1)
    
    return fingerprint


def batch_fingerprints(batch: JAXMolecularBatch, radius: int = 2) -> Array:
    """Calculate fingerprints for a batch of molecules.
    
    Args:
        batch: Batch of JAX molecules
        radius: Fingerprint radius
        
    Returns:
        Batch of molecular fingerprints
    """
    if not JAX_AVAILABLE:
        raise BackendUnavailableError("JAX backend requires jax package")
    
    # Vectorized fingerprint calculation
    def single_fingerprint(atom_features, adj_matrix, atom_mask):
        # Initialize atom identifiers
        atom_ids = jnp.sum(atom_features * jnp.array([1, 2, 4, 8, 16, 32]), axis=1)
        
        # Iteratively update
        for _ in range(radius):
            neighbor_sums = jnp.matmul(adj_matrix, atom_ids)
            atom_ids = atom_ids + neighbor_sums
            atom_ids = atom_ids * atom_mask
        
        # Create fingerprint
        fingerprint_size = 1024
        fingerprint = jnp.zeros(fingerprint_size)
        
        # Vectorized hashing (simplified)
        valid_ids = atom_ids * atom_mask
        hash_indices = (valid_ids.astype(int) % fingerprint_size)
        
        # Count occurrences at each hash position
        for i in range(fingerprint_size):
            count = jnp.sum((hash_indices == i) * atom_mask)
            fingerprint = fingerprint.at[i].set(count)
        
        return fingerprint
    
    # Apply to batch using vmap
    vectorized_fingerprint = jax.vmap(single_fingerprint)
    
    return vectorized_fingerprint(
        batch.atom_features,
        batch.adjacency_matrices,
        batch.atom_masks
    )
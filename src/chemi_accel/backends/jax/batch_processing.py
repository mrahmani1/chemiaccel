"""
JAX batch processing utilities for molecular data.

This module provides utilities for efficient batch processing of molecular
data with automatic shape handling and memory management.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

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

from ...core.exceptions import BackendUnavailableError, ValidationError
from ...core.molecular import MolecularData
from ...utils.logging import get_logger
from .molecular import JAXMolecularBatch, JAXMolecularProcessor

logger = get_logger()


class JAXBatchProcessor:
    """JAX batch processor for efficient molecular data processing.

    This class handles automatic batching, shape optimization, and
    memory management for JAX-based molecular calculations.
    """

    def __init__(
        self,
        max_atoms: int = 100,
        max_bonds: int = 200,
        default_batch_size: int = 32,
        memory_limit_mb: Optional[int] = None,
    ):
        """Initialize JAX batch processor.

        Args:
            max_atoms: Maximum number of atoms for padding
            max_bonds: Maximum number of bonds for padding
            default_batch_size: Default batch size for processing
            memory_limit_mb: Memory limit in MB (None for auto-detection)
        """
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")

        self.max_atoms = max_atoms
        self.max_bonds = max_bonds
        self.default_batch_size = default_batch_size
        self.memory_limit_mb = memory_limit_mb

        # Initialize molecular processor
        self.molecular_processor = JAXMolecularProcessor(max_atoms, max_bonds)

        # Batch statistics
        self.processed_batches = 0
        self.total_molecules = 0

        logger.info(
            f"Initialized JAX batch processor with batch_size={default_batch_size}"
        )

    def estimate_memory_usage(
        self, batch_size: int, num_atom_features: int = 6, num_bond_features: int = 4
    ) -> float:
        """Estimate memory usage for a batch in MB.

        Args:
            batch_size: Number of molecules in batch
            num_atom_features: Number of atom features
            num_bond_features: Number of bond features

        Returns:
            Estimated memory usage in MB
        """
        # Calculate array sizes (assuming float32)
        bytes_per_float = 4

        # Atom features: (batch_size, max_atoms, num_atom_features)
        atom_features_size = (
            batch_size * self.max_atoms * num_atom_features * bytes_per_float
        )

        # Bond features: (batch_size, max_bonds, num_bond_features)
        bond_features_size = (
            batch_size * self.max_bonds * num_bond_features * bytes_per_float
        )

        # Adjacency matrices: (batch_size, max_atoms, max_atoms)
        adjacency_size = batch_size * self.max_atoms * self.max_atoms * bytes_per_float

        # Masks: (batch_size, max_atoms) + (batch_size, max_bonds)
        mask_size = batch_size * (self.max_atoms + self.max_bonds) * bytes_per_float

        total_bytes = (
            atom_features_size + bond_features_size + adjacency_size + mask_size
        )
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def optimize_batch_size(self, molecules: List[MolecularData]) -> int:
        """Optimize batch size based on molecule characteristics and memory constraints.

        Args:
            molecules: List of molecules to analyze

        Returns:
            Optimized batch size
        """
        if not molecules:
            return 1

        # Analyze molecule sizes
        atom_counts = [mol.num_atoms or 0 for mol in molecules if mol.num_atoms]
        if not atom_counts:
            return min(self.default_batch_size, len(molecules))

        avg_atoms = np.mean(atom_counts)
        max_atoms = max(atom_counts)

        # Start with default batch size
        optimal_batch_size = self.default_batch_size

        # Adjust based on molecule size
        if avg_atoms > 50:
            optimal_batch_size = max(8, optimal_batch_size // 2)
        elif avg_atoms > 100:
            optimal_batch_size = max(4, optimal_batch_size // 4)

        # Check memory constraints if specified
        if self.memory_limit_mb:
            while optimal_batch_size > 1:
                estimated_memory = self.estimate_memory_usage(optimal_batch_size)
                if estimated_memory <= self.memory_limit_mb:
                    break
                optimal_batch_size = max(1, optimal_batch_size // 2)

        # Don't exceed number of molecules
        optimal_batch_size = min(optimal_batch_size, len(molecules))

        logger.debug(
            f"Optimized batch size: {optimal_batch_size} "
            f"(avg_atoms={avg_atoms:.1f}, max_atoms={max_atoms}, "
            f"estimated_memory={self.estimate_memory_usage(optimal_batch_size):.1f}MB)"
        )

        return optimal_batch_size

    def create_batches(
        self, molecules: List[MolecularData], batch_size: Optional[int] = None
    ) -> List[JAXMolecularBatch]:
        """Create batches from list of molecules.

        Args:
            molecules: List of molecular data
            batch_size: Optional batch size (auto-optimized if None)

        Returns:
            List of JAX molecular batches
        """
        if not molecules:
            return []

        # Optimize batch size if not provided
        if batch_size is None:
            batch_size = self.optimize_batch_size(molecules)

        batches = []
        for i in range(0, len(molecules), batch_size):
            batch_molecules = molecules[i : i + batch_size]
            try:
                batch = self.molecular_processor.create_batch(batch_molecules)
                batches.append(batch)
            except Exception as e:
                logger.warning(f"Failed to create batch {i//batch_size}: {e}")
                continue

        logger.info(f"Created {len(batches)} batches from {len(molecules)} molecules")
        return batches

    def process_batches_iterator(
        self, molecules: List[MolecularData], batch_size: Optional[int] = None
    ) -> Iterator[JAXMolecularBatch]:
        """Create iterator for processing batches one at a time.

        Args:
            molecules: List of molecular data
            batch_size: Optional batch size

        Yields:
            JAX molecular batches
        """
        if batch_size is None:
            batch_size = self.optimize_batch_size(molecules)

        for i in range(0, len(molecules), batch_size):
            batch_molecules = molecules[i : i + batch_size]
            try:
                batch = self.molecular_processor.create_batch(batch_molecules)
                yield batch
                self.processed_batches += 1
                self.total_molecules += len(batch_molecules)
            except Exception as e:
                logger.warning(f"Failed to process batch starting at index {i}: {e}")
                continue

    def adaptive_batch_processing(
        self,
        molecules: List[MolecularData],
        process_function: callable,
        initial_batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Adaptive batch processing with automatic batch size adjustment.

        Args:
            molecules: List of molecular data
            process_function: Function to process each batch
            initial_batch_size: Initial batch size (auto-optimized if None)

        Returns:
            List of processing results
        """
        if not molecules:
            return []

        current_batch_size = initial_batch_size or self.optimize_batch_size(molecules)
        results = []
        failed_batches = 0

        i = 0
        while i < len(molecules):
            batch_molecules = molecules[i : i + current_batch_size]

            try:
                # Create batch
                batch = self.molecular_processor.create_batch(batch_molecules)

                # Process batch
                batch_results = process_function(batch)
                results.append(batch_results)

                # Move to next batch
                i += current_batch_size
                failed_batches = 0  # Reset failure counter

                logger.debug(
                    f"Successfully processed batch {len(results)} "
                    f"(size={len(batch_molecules)}, total_processed={i})"
                )

            except Exception as e:
                failed_batches += 1
                logger.warning(
                    f"Batch processing failed (attempt {failed_batches}): {e}"
                )

                # Adaptive batch size reduction
                if failed_batches >= 2 and current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.info(f"Reduced batch size to {current_batch_size}")
                    failed_batches = 0
                elif failed_batches >= 3:
                    # Skip problematic molecules
                    logger.error(
                        f"Skipping batch starting at index {i} after multiple failures"
                    )
                    i += current_batch_size
                    failed_batches = 0

        logger.info(
            f"Completed adaptive batch processing: {len(results)} successful batches"
        )
        return results

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "processed_batches": self.processed_batches,
            "total_molecules": self.total_molecules,
            "avg_molecules_per_batch": self.total_molecules
            / max(1, self.processed_batches),
            "max_atoms": self.max_atoms,
            "max_bonds": self.max_bonds,
            "default_batch_size": self.default_batch_size,
            "memory_limit_mb": self.memory_limit_mb,
        }

    def reset_stats(self):
        """Reset processing statistics."""
        self.processed_batches = 0
        self.total_molecules = 0


class JAXShapeHandler:
    """Utilities for handling variable molecular shapes in JAX.

    This class provides utilities for padding, masking, and shape
    transformations needed for efficient JAX processing.
    """

    @staticmethod
    def pad_to_shape(
        array: np.ndarray, target_shape: Tuple[int, ...], pad_value: float = 0.0
    ) -> Array:
        """Pad array to target shape.

        Args:
            array: Input array to pad
            target_shape: Target shape for padding
            pad_value: Value to use for padding

        Returns:
            Padded JAX array
        """
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")

        current_shape = array.shape

        if len(current_shape) != len(target_shape):
            raise ValidationError(
                input_name="array",
                input_value=current_shape,
                expected_type=f"shape with {len(target_shape)} dimensions",
                reason=f"Shape dimension mismatch: {len(current_shape)} vs {len(target_shape)}",
            )

        # Calculate padding for each dimension
        pad_widths = []
        for current_dim, target_dim in zip(current_shape, target_shape):
            if current_dim > target_dim:
                raise ValidationError(
                    input_name="array",
                    input_value=current_dim,
                    expected_type=f"<= {target_dim}",
                    reason=f"Array dimension {current_dim} exceeds target {target_dim}",
                )
            pad_widths.append((0, target_dim - current_dim))

        # Convert to JAX array and pad
        jax_array = jnp.array(array)
        padded_array = jnp.pad(jax_array, pad_widths, constant_values=pad_value)

        return padded_array

    @staticmethod
    def create_mask(actual_size: int, max_size: int) -> Array:
        """Create boolean mask for valid elements.

        Args:
            actual_size: Actual number of valid elements
            max_size: Maximum size (padded size)

        Returns:
            Boolean mask array
        """
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")

        if actual_size > max_size:
            raise ValidationError(
                input_name="actual_size",
                input_value=actual_size,
                expected_type=f"<= {max_size}",
                reason="Actual size cannot exceed maximum size",
            )

        mask = jnp.zeros(max_size, dtype=bool)
        mask = mask.at[:actual_size].set(True)
        return mask

    @staticmethod
    def apply_mask(array: Array, mask: Array, axis: int = 0) -> Array:
        """Apply mask to array along specified axis.

        Args:
            array: Input array
            mask: Boolean mask
            axis: Axis to apply mask along

        Returns:
            Masked array
        """
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")

        # Expand mask dimensions to match array
        expanded_mask = mask
        for _ in range(array.ndim - mask.ndim):
            expanded_mask = jnp.expand_dims(expanded_mask, axis=-1)

        # Move mask axis to match target axis
        if axis != 0:
            expanded_mask = jnp.moveaxis(expanded_mask, 0, axis)

        return array * expanded_mask

    @staticmethod
    def unpad_array(padded_array: Array, actual_size: int, axis: int = 0) -> Array:
        """Remove padding from array.

        Args:
            padded_array: Padded array
            actual_size: Actual size (without padding)
            axis: Axis to unpad along

        Returns:
            Unpadded array
        """
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")

        # Create slice for unpadding
        slices = [slice(None)] * padded_array.ndim
        slices[axis] = slice(0, actual_size)

        return padded_array[tuple(slices)]


def create_test_batch(batch_size: int = 4, max_atoms: int = 20) -> JAXMolecularBatch:
    """Create a test batch for development and testing.

    Args:
        batch_size: Number of molecules in batch
        max_atoms: Maximum number of atoms

    Returns:
        Test JAX molecular batch
    """
    from .molecular import create_synthetic_molecule

    # Create synthetic molecules with varying sizes
    molecules = []
    for i in range(batch_size):
        num_atoms = np.random.randint(5, max_atoms)
        num_bonds = max(0, num_atoms - 1)
        mol = create_synthetic_molecule(
            num_atoms=num_atoms,
            num_bonds=num_bonds,
            mol_id=f"test_mol_{i}",
            smiles=f"test_smiles_{i}",
        )
        molecules.append(mol)

    # Create batch processor and generate batch
    processor = JAXBatchProcessor(max_atoms=max_atoms)
    batch = processor.molecular_processor.create_batch(molecules)

    return batch

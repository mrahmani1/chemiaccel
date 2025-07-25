"""
JAX vectorization and JIT compilation utilities.

This module implements jax.vmap vectorization for batch descriptor calculations,
JIT-compiled descriptor calculation graphs, and automatic differentiation support.
"""

from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import Array, grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    Array = Any
    jnp = None
    vmap = None
    jit = None
    grad = None

from ...core.exceptions import BackendUnavailableError, DescriptorCalculationError
from ...utils.logging import get_logger
from .molecular import JAXMolecularBatch, JAXMolecule

logger = get_logger()


class JAXVectorizedCalculator:
    """JAX vectorized molecular descriptor calculator.

    This class implements vectorized descriptor calculations using jax.vmap
    and JIT compilation for optimal performance on TPUs and GPUs.
    """

    def __init__(self, compile_on_init: bool = True):
        """Initialize vectorized calculator.

        Args:
            compile_on_init: Whether to compile functions during initialization
        """
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")

        self.compile_on_init = compile_on_init
        self._compiled_functions = {}
        self._vectorized_functions = {}

        if compile_on_init:
            self._compile_all_functions()

        logger.info("Initialized JAX vectorized calculator")

    def _compile_all_functions(self):
        """Compile all descriptor functions with JIT and vectorization."""
        try:
            # Single molecule descriptor functions
            self._compiled_functions["molecular_weight"] = jit(
                self._molecular_weight_single
            )
            self._compiled_functions["atom_count"] = jit(self._atom_count_single)
            self._compiled_functions["bond_count"] = jit(self._bond_count_single)
            self._compiled_functions["formal_charge"] = jit(self._formal_charge_single)
            self._compiled_functions["aromatic_atoms"] = jit(
                self._aromatic_atoms_single
            )

            # Vectorized versions using vmap
            self._vectorized_functions["molecular_weight"] = jit(
                vmap(self._molecular_weight_single)
            )
            self._vectorized_functions["atom_count"] = jit(
                vmap(self._atom_count_single)
            )
            self._vectorized_functions["bond_count"] = jit(
                vmap(self._bond_count_single)
            )
            self._vectorized_functions["formal_charge"] = jit(
                vmap(self._formal_charge_single)
            )
            self._vectorized_functions["aromatic_atoms"] = jit(
                vmap(self._aromatic_atoms_single)
            )

            # Complex descriptor functions
            self._compiled_functions["topological_descriptors"] = jit(
                self._topological_descriptors_single
            )
            self._vectorized_functions["topological_descriptors"] = jit(
                vmap(self._topological_descriptors_single)
            )

            # Fingerprint functions
            self._compiled_functions["morgan_fingerprint"] = jit(
                partial(self._morgan_fingerprint_single, radius=2)
            )
            self._vectorized_functions["morgan_fingerprint"] = jit(
                vmap(partial(self._morgan_fingerprint_single, radius=2))
            )

            # Note: batch_all_descriptors cannot be JIT compiled directly due to complex input structure
            # We'll use the individual vectorized functions instead

            logger.debug("Successfully compiled all JAX vectorized functions")

        except Exception as e:
            logger.warning(f"Failed to compile some vectorized functions: {e}")

    def _molecular_weight_single(self, atom_features: Array, atom_mask: Array) -> float:
        """Calculate molecular weight for single molecule.

        Args:
            atom_features: Atom feature matrix (max_atoms, num_features)
            atom_mask: Boolean mask for valid atoms (max_atoms,)

        Returns:
            Molecular weight
        """
        # Atomic weights lookup table
        atomic_weights = jnp.array(
            [
                0.0,
                1.008,
                4.003,
                6.94,
                9.012,
                10.81,
                12.01,
                14.01,
                16.00,
                19.00,
                20.18,
                22.99,
                24.31,
                26.98,
                28.09,
                30.97,
                32.07,
                35.45,
                39.95,
                39.10,
                40.08,
                44.96,
                47.87,
                50.94,
                51.996,
                54.94,
                55.85,
                58.93,
                58.69,
                63.55,
                65.38,
            ]
        )

        # Get atomic numbers and clip to valid range
        atomic_numbers = atom_features[:, 0].astype(int)
        atomic_numbers = jnp.clip(atomic_numbers, 0, len(atomic_weights) - 1)

        # Calculate molecular weight
        weights = atomic_weights[atomic_numbers]
        return jnp.sum(weights * atom_mask)

    def _atom_count_single(self, atom_features: Array, atom_mask: Array) -> int:
        """Count atoms in single molecule."""
        return jnp.sum(atom_mask)

    def _bond_count_single(self, bond_features: Array, bond_mask: Array) -> int:
        """Count bonds in single molecule."""
        return jnp.sum(bond_mask)

    def _formal_charge_single(self, atom_features: Array, atom_mask: Array) -> float:
        """Calculate total formal charge for single molecule."""
        formal_charges = atom_features[:, 1]  # Formal charge is in column 1
        return jnp.sum(formal_charges * atom_mask)

    def _aromatic_atoms_single(self, atom_features: Array, atom_mask: Array) -> int:
        """Count aromatic atoms in single molecule."""
        is_aromatic = atom_features[:, 4]  # Aromatic flag is in column 4
        return jnp.sum(is_aromatic * atom_mask)

    def _topological_descriptors_single(
        self, atom_features: Array, adjacency_matrix: Array, atom_mask: Array
    ) -> Dict[str, float]:
        """Calculate topological descriptors for single molecule.

        Args:
            atom_features: Atom feature matrix
            adjacency_matrix: Adjacency matrix
            atom_mask: Boolean mask for valid atoms

        Returns:
            Dictionary of topological descriptors
        """
        # Calculate degree matrix
        degrees = jnp.sum(
            adjacency_matrix * atom_mask[:, None] * atom_mask[None, :], axis=1
        )
        valid_degrees = degrees * atom_mask

        # Zagreb indices
        zagreb_1 = jnp.sum(valid_degrees**2)

        # Zagreb 2 (sum of products of degrees of adjacent atoms)
        degree_products = degrees[:, None] * degrees[None, :]
        zagreb_2 = (
            jnp.sum(
                adjacency_matrix
                * degree_products
                * atom_mask[:, None]
                * atom_mask[None, :]
            )
            / 2
        )

        # Randic connectivity index
        sqrt_degree_products = jnp.sqrt(degree_products)
        # Avoid division by zero
        safe_sqrt_products = jnp.where(
            sqrt_degree_products > 0, sqrt_degree_products, 1.0
        )
        randic_index = (
            jnp.sum(
                adjacency_matrix
                / safe_sqrt_products
                * atom_mask[:, None]
                * atom_mask[None, :]
            )
            / 2
        )

        # Wiener index (simplified - sum of shortest paths)
        # For efficiency, we approximate using powers of adjacency matrix
        num_atoms = jnp.sum(atom_mask)
        wiener_approx = 0.0

        # Calculate path matrix up to diameter
        path_matrix = adjacency_matrix
        for path_length in range(1, int(num_atoms)):
            path_counts = jnp.sum(path_matrix * atom_mask[:, None] * atom_mask[None, :])
            wiener_approx += path_length * path_counts
            path_matrix = jnp.matmul(path_matrix, adjacency_matrix)

            # Early termination if no more paths
            if jnp.sum(path_matrix) == 0:
                break

        return {
            "zagreb_1": zagreb_1,
            "zagreb_2": zagreb_2,
            "randic_index": randic_index,
            "wiener_index": wiener_approx,
        }

    def _morgan_fingerprint_single(
        self,
        atom_features: Array,
        adjacency_matrix: Array,
        atom_mask: Array,
        radius: int = 2,
    ) -> Array:
        """Calculate Morgan fingerprint for single molecule.

        Args:
            atom_features: Atom feature matrix
            adjacency_matrix: Adjacency matrix
            atom_mask: Boolean mask for valid atoms
            radius: Fingerprint radius

        Returns:
            Morgan fingerprint as bit vector
        """
        # Initialize atom identifiers from features
        atom_ids = jnp.sum(atom_features * jnp.array([1, 2, 4, 8, 16, 32]), axis=1)

        # Iteratively update identifiers based on neighbors
        for _ in range(radius):
            neighbor_sums = jnp.matmul(adjacency_matrix, atom_ids)
            atom_ids = atom_ids + neighbor_sums
            atom_ids = atom_ids * atom_mask  # Apply mask

        # Create fingerprint by hashing atom identifiers
        fingerprint_size = 1024
        fingerprint = jnp.zeros(fingerprint_size)

        # Hash valid atom identifiers to fingerprint positions
        valid_ids = atom_ids * atom_mask
        hash_indices = valid_ids.astype(int) % fingerprint_size

        # Count occurrences at each hash position
        for i in range(fingerprint_size):
            count = jnp.sum((hash_indices == i) * atom_mask)
            fingerprint = fingerprint.at[i].add(count)

        return fingerprint

    def _batch_all_descriptors(self, batch: JAXMolecularBatch) -> Dict[str, Array]:
        """Calculate all descriptors for a batch using vectorized operations.

        Args:
            batch: Batch of JAX molecules

        Returns:
            Dictionary of descriptor arrays
        """
        # Extract batch arrays
        atom_features = batch.atom_features  # (batch_size, max_atoms, num_features)
        bond_features = batch.bond_features  # (batch_size, max_bonds, num_features)
        adjacency_matrices = (
            batch.adjacency_matrices
        )  # (batch_size, max_atoms, max_atoms)
        atom_masks = batch.atom_masks  # (batch_size, max_atoms)
        bond_masks = batch.bond_masks  # (batch_size, max_bonds)

        results = {}

        # Basic descriptors using vectorized functions
        if "molecular_weight" in self._vectorized_functions:
            results["molecular_weight"] = self._vectorized_functions[
                "molecular_weight"
            ](atom_features, atom_masks)

        if "atom_count" in self._vectorized_functions:
            results["atom_count"] = self._vectorized_functions["atom_count"](
                atom_features, atom_masks
            )

        if "bond_count" in self._vectorized_functions:
            results["bond_count"] = self._vectorized_functions["bond_count"](
                bond_features, bond_masks
            )

        if "formal_charge" in self._vectorized_functions:
            results["formal_charge"] = self._vectorized_functions["formal_charge"](
                atom_features, atom_masks
            )

        if "aromatic_atoms" in self._vectorized_functions:
            results["aromatic_atoms"] = self._vectorized_functions["aromatic_atoms"](
                atom_features, atom_masks
            )

        # Note: Topological descriptors are skipped in batch mode due to complexity
        # They would need special handling for dictionary returns in vmap

        # Morgan fingerprints
        if "morgan_fingerprint" in self._vectorized_functions:
            results["morgan_fingerprint"] = self._vectorized_functions[
                "morgan_fingerprint"
            ](atom_features, adjacency_matrices, atom_masks)

        return results

    def calculate_batch_vectorized(
        self, batch: JAXMolecularBatch
    ) -> Dict[str, np.ndarray]:
        """Calculate descriptors for batch using vectorized operations.

        Args:
            batch: Batch of JAX molecules

        Returns:
            Dictionary of descriptor arrays
        """
        try:
            # Use the non-compiled version for now due to complex input structure
            jax_results = self._batch_all_descriptors(batch)

            # Convert JAX arrays to numpy
            results = {}
            for key, values in jax_results.items():
                if isinstance(values, jnp.ndarray):
                    results[key] = np.array(values)
                else:
                    results[key] = values

            return results

        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="batch_descriptors",
                molecule_id=f"batch_size_{batch.batch_size}",
                reason=str(e),
            )

    def calculate_single_vectorized(self, molecule: JAXMolecule) -> Dict[str, Any]:
        """Calculate descriptors for single molecule using compiled functions.

        Args:
            molecule: JAX molecule representation

        Returns:
            Dictionary of calculated descriptors
        """
        try:
            results = {}

            # Use compiled single molecule functions
            if "molecular_weight" in self._compiled_functions:
                results["molecular_weight"] = float(
                    self._compiled_functions["molecular_weight"](
                        molecule.atom_features, molecule.atom_mask
                    )
                )

            if "atom_count" in self._compiled_functions:
                results["atom_count"] = int(
                    self._compiled_functions["atom_count"](
                        molecule.atom_features, molecule.atom_mask
                    )
                )

            if "bond_count" in self._compiled_functions:
                results["bond_count"] = int(
                    self._compiled_functions["bond_count"](
                        molecule.bond_features, molecule.bond_mask
                    )
                )

            if "formal_charge" in self._compiled_functions:
                results["formal_charge"] = float(
                    self._compiled_functions["formal_charge"](
                        molecule.atom_features, molecule.atom_mask
                    )
                )

            if "aromatic_atoms" in self._compiled_functions:
                results["aromatic_atoms"] = int(
                    self._compiled_functions["aromatic_atoms"](
                        molecule.atom_features, molecule.atom_mask
                    )
                )

            # Morgan fingerprint
            if "morgan_fingerprint" in self._compiled_functions:
                fp = self._compiled_functions["morgan_fingerprint"](
                    molecule.atom_features,
                    molecule.adjacency_matrix,
                    molecule.atom_mask,
                )
                results["morgan_fingerprint"] = np.array(fp)

            return results

        except Exception as e:
            raise DescriptorCalculationError(
                descriptor="vectorized_single",
                molecule_id=molecule.mol_id or "unknown",
                reason=str(e),
            )

    def get_compiled_functions(self) -> List[str]:
        """Get list of compiled function names.

        Returns:
            List of compiled function names
        """
        return list(self._compiled_functions.keys())

    def get_vectorized_functions(self) -> List[str]:
        """Get list of vectorized function names.

        Returns:
            List of vectorized function names
        """
        return list(self._vectorized_functions.keys())


class JAXAutoDiffCalculator:
    """JAX automatic differentiation calculator for ML integration.

    This class provides automatic differentiation support for molecular
    descriptors, enabling gradient-based optimization and ML workflows.
    """

    def __init__(self):
        """Initialize automatic differentiation calculator."""
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")

        self._grad_functions = {}
        self._compile_grad_functions()

        logger.info("Initialized JAX automatic differentiation calculator")

    def _compile_grad_functions(self):
        """Compile gradient functions for key descriptors."""
        try:
            # Gradient of molecular weight with respect to atom features
            self._grad_functions["molecular_weight_grad"] = jit(
                grad(self._molecular_weight_differentiable)
            )

            # Gradient of formal charge
            self._grad_functions["formal_charge_grad"] = jit(
                grad(self._formal_charge_differentiable)
            )

            logger.debug("Successfully compiled gradient functions")

        except Exception as e:
            logger.warning(f"Failed to compile gradient functions: {e}")

    def _molecular_weight_differentiable(self, atom_features: Array) -> float:
        """Differentiable molecular weight calculation.

        Args:
            atom_features: Atom feature matrix (differentiable)

        Returns:
            Molecular weight
        """
        # Simplified differentiable version
        # Use continuous approximation of atomic weights
        atomic_numbers = atom_features[:, 0]  # Keep as float for differentiability

        # Smooth approximation of atomic weight lookup
        # Using polynomial approximation for common elements
        weights = 12.0 + atomic_numbers * 0.8  # Rough approximation

        return jnp.sum(weights)

    def _formal_charge_differentiable(self, atom_features: Array) -> float:
        """Differentiable formal charge calculation.

        Args:
            atom_features: Atom feature matrix

        Returns:
            Total formal charge
        """
        formal_charges = atom_features[:, 1]
        return jnp.sum(formal_charges)

    def calculate_gradients(self, molecule: JAXMolecule, descriptor: str) -> Array:
        """Calculate gradients of descriptor with respect to atom features.

        Args:
            molecule: JAX molecule representation
            descriptor: Name of descriptor to differentiate

        Returns:
            Gradient array
        """
        if descriptor not in self._grad_functions:
            raise ValueError(
                f"Gradient function not available for descriptor: {descriptor}"
            )

        try:
            grad_func = self._grad_functions[f"{descriptor}_grad"]
            gradients = grad_func(molecule.atom_features)
            return gradients

        except Exception as e:
            raise DescriptorCalculationError(
                descriptor=f"{descriptor}_gradient",
                molecule_id=molecule.mol_id or "unknown",
                reason=str(e),
            )

    def get_available_gradients(self) -> List[str]:
        """Get list of descriptors with gradient support.

        Returns:
            List of descriptor names
        """
        return [name.replace("_grad", "") for name in self._grad_functions.keys()]


class JAXMemoryEfficientBatcher:
    """Memory-efficient batching strategies for JAX calculations.

    This class implements efficient batching strategies for memory-constrained
    environments, including gradient accumulation and chunked processing.
    """

    def __init__(self, memory_limit_mb: Optional[int] = None):
        """Initialize memory-efficient batcher.

        Args:
            memory_limit_mb: Memory limit in MB (None for auto-detection)
        """
        if not JAX_AVAILABLE:
            raise BackendUnavailableError("JAX backend requires jax package")

        self.memory_limit_mb = memory_limit_mb
        self.vectorized_calc = JAXVectorizedCalculator(compile_on_init=True)

        logger.info(
            f"Initialized memory-efficient batcher with limit: {memory_limit_mb}MB"
        )

    def chunked_batch_processing(
        self, batch: JAXMolecularBatch, chunk_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Process batch in chunks to manage memory usage.

        Args:
            batch: Large batch to process
            chunk_size: Size of chunks (auto-determined if None)

        Returns:
            Combined results from all chunks
        """
        if chunk_size is None:
            chunk_size = self._determine_optimal_chunk_size(batch)

        all_results = {}
        batch_size = batch.batch_size

        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)

            # Create chunk
            chunk = self._extract_chunk(batch, start_idx, end_idx)

            # Process chunk
            chunk_results = self.vectorized_calc.calculate_batch_vectorized(chunk)

            # Accumulate results
            for key, values in chunk_results.items():
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append(values)

        # Concatenate all results
        final_results = {}
        for key, value_list in all_results.items():
            if isinstance(value_list[0], np.ndarray):
                final_results[key] = np.concatenate(value_list, axis=0)
            else:
                final_results[key] = value_list

        return final_results

    def _determine_optimal_chunk_size(self, batch: JAXMolecularBatch) -> int:
        """Determine optimal chunk size based on memory constraints.

        Args:
            batch: Batch to analyze

        Returns:
            Optimal chunk size
        """
        if self.memory_limit_mb is None:
            return min(16, batch.batch_size)  # Default conservative chunk size

        # Estimate memory per molecule
        memory_per_mol = self._estimate_memory_per_molecule(batch)

        # Calculate maximum molecules that fit in memory limit
        max_molecules = int(self.memory_limit_mb / memory_per_mol)

        # Use conservative estimate (80% of limit)
        chunk_size = max(1, int(max_molecules * 0.8))

        return min(chunk_size, batch.batch_size)

    def _estimate_memory_per_molecule(self, batch: JAXMolecularBatch) -> float:
        """Estimate memory usage per molecule in MB.

        Args:
            batch: Batch to analyze

        Returns:
            Memory per molecule in MB
        """
        # Rough estimate based on array sizes
        bytes_per_float = 4

        atom_features_size = batch.max_atoms * 6 * bytes_per_float
        bond_features_size = batch.max_bonds * 4 * bytes_per_float
        adjacency_size = batch.max_atoms * batch.max_atoms * bytes_per_float

        total_bytes = atom_features_size + bond_features_size + adjacency_size
        return total_bytes / (1024 * 1024)  # Convert to MB

    def _extract_chunk(
        self, batch: JAXMolecularBatch, start_idx: int, end_idx: int
    ) -> JAXMolecularBatch:
        """Extract a chunk from the batch.

        Args:
            batch: Original batch
            start_idx: Start index for chunk
            end_idx: End index for chunk

        Returns:
            Chunk as new JAXMolecularBatch
        """
        chunk_size = end_idx - start_idx

        # Extract chunk arrays
        chunk_atom_features = batch.atom_features[start_idx:end_idx]
        chunk_bond_features = batch.bond_features[start_idx:end_idx]
        chunk_adjacency = batch.adjacency_matrices[start_idx:end_idx]
        chunk_atom_masks = batch.atom_masks[start_idx:end_idx]
        chunk_bond_masks = batch.bond_masks[start_idx:end_idx]
        chunk_molecule_sizes = batch.molecule_sizes[start_idx:end_idx]

        # Extract metadata
        chunk_mol_ids = batch.mol_ids[start_idx:end_idx] if batch.mol_ids else None
        chunk_smiles = (
            batch.smiles_list[start_idx:end_idx] if batch.smiles_list else None
        )

        return JAXMolecularBatch(
            atom_features=chunk_atom_features,
            bond_features=chunk_bond_features,
            adjacency_matrices=chunk_adjacency,
            atom_masks=chunk_atom_masks,
            bond_masks=chunk_bond_masks,
            batch_size=chunk_size,
            max_atoms=batch.max_atoms,
            max_bonds=batch.max_bonds,
            molecule_sizes=chunk_molecule_sizes,
            mol_ids=chunk_mol_ids,
            smiles_list=chunk_smiles,
        )

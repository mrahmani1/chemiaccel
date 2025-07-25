"""
Graph-Based CUDA Kernels for Molecular Descriptor Calculations.

This module provides CUDA kernel implementations for graph-based molecular
descriptors including topological indices, substructure matching, and
molecular similarity calculations using CSR graph representations.
"""

from typing import Any, Dict, Optional, Tuple

try:
    import cupy as cp
    from cupy import RawKernel

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    RawKernel = None

# Type definitions for when CUDA is not available
if not CUDA_AVAILABLE:
    from typing import Any

    class MockArray:
        pass

    class MockRawKernel:
        pass

    # Create mock modules for type annotations
    class MockCp:
        ndarray = MockArray

    if cp is None:
        cp = MockCp()
    if RawKernel is None:
        RawKernel = MockRawKernel

from ...utils.logging import get_logger
from .memory import get_memory_manager

logger = get_logger()


# CUDA kernel source code for graph-based molecular descriptors
WIENER_INDEX_KERNEL = r"""
extern "C" __global__
void calculate_wiener_index(
    const int* csr_indices,
    const int* csr_indptr,
    const int* molecule_boundaries,
    const int* molecule_csr_boundaries,
    float* results,
    int num_molecules,
    int max_atoms
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int num_atoms = end_atom - start_atom;

    int start_csr = (mol_idx == 0) ? 0 : molecule_csr_boundaries[mol_idx - 1];

    // Shared memory for distance matrix (limited by max_atoms)
    extern __shared__ int dist[];

    if (num_atoms > max_atoms) {
        results[mol_idx] = -1.0f; // Error indicator
        return;
    }

    // Initialize distance matrix with large values
    for (int i = 0; i < num_atoms * num_atoms; i++) {
        dist[i] = 999999;
    }

    // Set diagonal to 0
    for (int i = 0; i < num_atoms; i++) {
        dist[i * num_atoms + i] = 0;
    }

    // Set direct connections to 1
    for (int i = 0; i < num_atoms; i++) {
        int atom_csr_start = csr_indptr[start_atom + i] - start_csr;
        int atom_csr_end = csr_indptr[start_atom + i + 1] - start_csr;

        for (int j = atom_csr_start; j < atom_csr_end; j++) {
            int neighbor = csr_indices[start_csr + j] - start_atom;
            if (neighbor >= 0 && neighbor < num_atoms) {
                dist[i * num_atoms + neighbor] = 1;
            }
        }
    }

    // Floyd-Warshall algorithm for shortest paths
    for (int k = 0; k < num_atoms; k++) {
        for (int i = 0; i < num_atoms; i++) {
            for (int j = 0; j < num_atoms; j++) {
                int ik_dist = dist[i * num_atoms + k];
                int kj_dist = dist[k * num_atoms + j];
                int ij_dist = dist[i * num_atoms + j];

                if (ik_dist != 999999 && kj_dist != 999999) {
                    int new_dist = ik_dist + kj_dist;
                    if (new_dist < ij_dist) {
                        dist[i * num_atoms + j] = new_dist;
                    }
                }
            }
        }
    }

    // Calculate Wiener index (sum of all shortest path distances)
    float wiener_sum = 0.0f;
    for (int i = 0; i < num_atoms; i++) {
        for (int j = i + 1; j < num_atoms; j++) {
            int distance = dist[i * num_atoms + j];
            if (distance != 999999) {
                wiener_sum += distance;
            }
        }
    }

    results[mol_idx] = wiener_sum;
}
"""

ZAGREB_INDEX_KERNEL = r"""
extern "C" __global__
void calculate_zagreb_indices(
    const int* csr_indices,
    const int* csr_indptr,
    const int* molecule_boundaries,
    const int* molecule_csr_boundaries,
    float* zagreb1_results,
    float* zagreb2_results,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int start_csr = (mol_idx == 0) ? 0 : molecule_csr_boundaries[mol_idx - 1];

    float zagreb1 = 0.0f;
    float zagreb2 = 0.0f;

    // Calculate degree for each atom and Zagreb indices
    for (int i = start_atom; i < end_atom; i++) {
        int atom_csr_start = csr_indptr[i] - start_csr;
        int atom_csr_end = csr_indptr[i + 1] - start_csr;
        int degree = atom_csr_end - atom_csr_start;

        // Zagreb M1 index: sum of squared degrees
        zagreb1 += degree * degree;

        // Zagreb M2 index: sum of products of degrees of adjacent atoms
        for (int j = atom_csr_start; j < atom_csr_end; j++) {
            int neighbor = csr_indices[start_csr + j];
            if (neighbor > i) { // Avoid double counting
                int neighbor_csr_start = csr_indptr[neighbor] - start_csr;
                int neighbor_csr_end = csr_indptr[neighbor + 1] - start_csr;
                int neighbor_degree = neighbor_csr_end - neighbor_csr_start;

                zagreb2 += degree * neighbor_degree;
            }
        }
    }

    zagreb1_results[mol_idx] = zagreb1;
    zagreb2_results[mol_idx] = zagreb2;
}
"""

SUBSTRUCTURE_MATCH_KERNEL = r"""
extern "C" __global__
void substructure_matching(
    const int* mol_csr_indices,
    const int* mol_csr_indptr,
    const int* mol_atom_types,
    const int* molecule_boundaries,
    const int* molecule_csr_boundaries,
    const int* pattern_csr_indices,
    const int* pattern_csr_indptr,
    const int* pattern_atom_types,
    int pattern_size,
    int* match_counts,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int num_atoms = end_atom - start_atom;
    int start_csr = (mol_idx == 0) ? 0 : molecule_csr_boundaries[mol_idx - 1];

    int match_count = 0;

    // Simple substructure matching - try each atom as potential match start
    for (int start_idx = 0; start_idx < num_atoms; start_idx++) {
        bool is_match = true;

        // Check if this starting position could match the pattern
        for (int p_idx = 0; p_idx < pattern_size && is_match; p_idx++) {
            int mol_atom_idx = start_atom + start_idx + p_idx;

            if (mol_atom_idx >= end_atom) {
                is_match = false;
                break;
            }

            // Check atom type match
            if (mol_atom_types[mol_atom_idx] != pattern_atom_types[p_idx]) {
                is_match = false;
                break;
            }

            // Check connectivity (simplified - just degree comparison)
            int mol_degree = csr_indptr[mol_atom_idx + 1] - csr_indptr[mol_atom_idx];
            int pattern_degree = pattern_csr_indptr[p_idx + 1] - pattern_csr_indptr[p_idx];

            if (mol_degree < pattern_degree) {
                is_match = false;
                break;
            }
        }

        if (is_match) {
            match_count++;
        }
    }

    match_counts[mol_idx] = match_count;
}
"""

TANIMOTO_SIMILARITY_KERNEL = r"""
extern "C" __global__
void calculate_tanimoto_similarity(
    const int* fingerprint1,
    const int* fingerprint2,
    const int* molecule_boundaries,
    float* similarities,
    int num_molecules,
    int fingerprint_size
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_fp = mol_idx * fingerprint_size;

    int intersection = 0;
    int union_count = 0;

    // Calculate intersection and union of fingerprints
    for (int i = 0; i < fingerprint_size; i++) {
        int bit1 = fingerprint1[start_fp + i];
        int bit2 = fingerprint2[start_fp + i];

        intersection += (bit1 & bit2);
        union_count += (bit1 | bit2);
    }

    // Tanimoto coefficient = intersection / union
    float similarity = 0.0f;
    if (union_count > 0) {
        similarity = (float)intersection / (float)union_count;
    }

    similarities[mol_idx] = similarity;
}
"""

RANDIC_INDEX_KERNEL = r"""
extern "C" __global__
void calculate_randic_index(
    const int* csr_indices,
    const int* csr_indptr,
    const int* molecule_boundaries,
    const int* molecule_csr_boundaries,
    float* results,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int start_csr = (mol_idx == 0) ? 0 : molecule_csr_boundaries[mol_idx - 1];

    float randic_sum = 0.0f;

    // Calculate Randic index: sum of 1/sqrt(degree_i * degree_j) for all edges
    for (int i = start_atom; i < end_atom; i++) {
        int atom_csr_start = csr_indptr[i] - start_csr;
        int atom_csr_end = csr_indptr[i + 1] - start_csr;
        int degree_i = atom_csr_end - atom_csr_start;

        for (int j = atom_csr_start; j < atom_csr_end; j++) {
            int neighbor = csr_indices[start_csr + j];

            if (neighbor > i) { // Avoid double counting edges
                int neighbor_csr_start = csr_indptr[neighbor] - start_csr;
                int neighbor_csr_end = csr_indptr[neighbor + 1] - start_csr;
                int degree_j = neighbor_csr_end - neighbor_csr_start;

                if (degree_i > 0 && degree_j > 0) {
                    randic_sum += 1.0f / sqrtf((float)(degree_i * degree_j));
                }
            }
        }
    }

    results[mol_idx] = randic_sum;
}
"""


class GraphKernelManager:
    """Manages graph-based CUDA kernels for molecular descriptor calculations."""

    def __init__(self, memory_manager=None):
        """Initialize graph kernel manager.

        Args:
            memory_manager: CUDA memory manager instance
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self.memory_manager = memory_manager or get_memory_manager()
        self._kernels = {}
        self._compile_kernels()

    def _compile_kernels(self):
        """Compile all graph-based CUDA kernels."""
        kernel_sources = {
            "wiener_index": WIENER_INDEX_KERNEL,
            "zagreb_indices": ZAGREB_INDEX_KERNEL,
            "substructure_matching": SUBSTRUCTURE_MATCH_KERNEL,
            "tanimoto_similarity": TANIMOTO_SIMILARITY_KERNEL,
            "randic_index": RANDIC_INDEX_KERNEL,
        }

        for name, source in kernel_sources.items():
            try:
                kernel = RawKernel(source, f"calculate_{name}")
                self._kernels[name] = kernel
                logger.debug(f"Compiled graph CUDA kernel: {name}")
            except Exception as e:
                logger.error(f"Failed to compile graph kernel {name}: {e}")
                raise

        logger.info(f"Compiled {len(self._kernels)} graph-based CUDA kernels")

    def calculate_wiener_index(
        self,
        csr_indices: cp.ndarray,
        csr_indptr: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        molecule_csr_boundaries: cp.ndarray,
        max_atoms: int = 50,
    ) -> cp.ndarray:
        """Calculate Wiener index using CUDA kernel.

        Args:
            csr_indices: CSR format indices array
            csr_indptr: CSR format indptr array
            molecule_boundaries: Array indicating end of each molecule
            molecule_csr_boundaries: Array indicating CSR boundaries for each molecule
            max_atoms: Maximum atoms per molecule for shared memory allocation

        Returns:
            Array of Wiener indices for each molecule
        """
        num_molecules = len(molecule_boundaries)
        results = cp.zeros(num_molecules, dtype=cp.float32)

        # Launch kernel with shared memory for distance matrix
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size
        shared_mem_size = max_atoms * max_atoms * 4  # 4 bytes per int

        kernel = self._kernels["wiener_index"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                csr_indices,
                csr_indptr,
                molecule_boundaries,
                molecule_csr_boundaries,
                results,
                num_molecules,
                max_atoms,
            ),
            shared_mem=shared_mem_size,
        )

        return results

    def calculate_zagreb_indices(
        self,
        csr_indices: cp.ndarray,
        csr_indptr: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        molecule_csr_boundaries: cp.ndarray,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Calculate Zagreb M1 and M2 indices.

        Args:
            csr_indices: CSR format indices array
            csr_indptr: CSR format indptr array
            molecule_boundaries: Array indicating end of each molecule
            molecule_csr_boundaries: Array indicating CSR boundaries for each molecule

        Returns:
            Tuple of (Zagreb M1 indices, Zagreb M2 indices)
        """
        num_molecules = len(molecule_boundaries)
        zagreb1_results = cp.zeros(num_molecules, dtype=cp.float32)
        zagreb2_results = cp.zeros(num_molecules, dtype=cp.float32)

        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["zagreb_indices"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                csr_indices,
                csr_indptr,
                molecule_boundaries,
                molecule_csr_boundaries,
                zagreb1_results,
                zagreb2_results,
                num_molecules,
            ),
        )

        return zagreb1_results, zagreb2_results

    def calculate_randic_index(
        self,
        csr_indices: cp.ndarray,
        csr_indptr: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        molecule_csr_boundaries: cp.ndarray,
    ) -> cp.ndarray:
        """Calculate Randic connectivity index.

        Args:
            csr_indices: CSR format indices array
            csr_indptr: CSR format indptr array
            molecule_boundaries: Array indicating end of each molecule
            molecule_csr_boundaries: Array indicating CSR boundaries for each molecule

        Returns:
            Array of Randic indices for each molecule
        """
        num_molecules = len(molecule_boundaries)
        results = cp.zeros(num_molecules, dtype=cp.float32)

        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["randic_index"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                csr_indices,
                csr_indptr,
                molecule_boundaries,
                molecule_csr_boundaries,
                results,
                num_molecules,
            ),
        )

        return results

    def substructure_matching(
        self,
        mol_csr_indices: cp.ndarray,
        mol_csr_indptr: cp.ndarray,
        mol_atom_types: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        molecule_csr_boundaries: cp.ndarray,
        pattern_csr_indices: cp.ndarray,
        pattern_csr_indptr: cp.ndarray,
        pattern_atom_types: cp.ndarray,
    ) -> cp.ndarray:
        """Perform substructure matching using CUDA kernel.

        Args:
            mol_csr_indices: Molecule CSR indices
            mol_csr_indptr: Molecule CSR indptr
            mol_atom_types: Molecule atom types
            molecule_boundaries: Molecule boundaries
            molecule_csr_boundaries: Molecule CSR boundaries
            pattern_csr_indices: Pattern CSR indices
            pattern_csr_indptr: Pattern CSR indptr
            pattern_atom_types: Pattern atom types

        Returns:
            Array of match counts for each molecule
        """
        num_molecules = len(molecule_boundaries)
        pattern_size = len(pattern_atom_types)
        match_counts = cp.zeros(num_molecules, dtype=cp.int32)

        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["substructure_matching"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                mol_csr_indices,
                mol_csr_indptr,
                mol_atom_types,
                molecule_boundaries,
                molecule_csr_boundaries,
                pattern_csr_indices,
                pattern_csr_indptr,
                pattern_atom_types,
                pattern_size,
                match_counts,
                num_molecules,
            ),
        )

        return match_counts

    def calculate_tanimoto_similarity(
        self,
        fingerprint1: cp.ndarray,
        fingerprint2: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        fingerprint_size: int,
    ) -> cp.ndarray:
        """Calculate Tanimoto similarity between fingerprints.

        Args:
            fingerprint1: First set of fingerprints
            fingerprint2: Second set of fingerprints
            molecule_boundaries: Molecule boundaries
            fingerprint_size: Size of each fingerprint

        Returns:
            Array of Tanimoto similarities
        """
        num_molecules = len(molecule_boundaries)
        similarities = cp.zeros(num_molecules, dtype=cp.float32)

        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["tanimoto_similarity"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                fingerprint1,
                fingerprint2,
                molecule_boundaries,
                similarities,
                num_molecules,
                fingerprint_size,
            ),
        )

        return similarities


class GraphDescriptorCalculator:
    """High-level interface for graph-based molecular descriptor calculations."""

    def __init__(self, memory_manager=None):
        """Initialize graph descriptor calculator.

        Args:
            memory_manager: CUDA memory manager instance
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self.memory_manager = memory_manager or get_memory_manager()
        self.kernel_manager = GraphKernelManager(memory_manager)

    def calculate_topological_indices(
        self, graph_data: Dict[str, Any]
    ) -> Dict[str, cp.ndarray]:
        """Calculate topological indices for molecular graphs.

        Args:
            graph_data: Dictionary containing graph data arrays

        Returns:
            Dictionary containing calculated topological indices
        """
        results = {}

        required_keys = [
            "csr_indices",
            "csr_indptr",
            "molecule_boundaries",
            "molecule_csr_boundaries",
        ]
        if not all(key in graph_data for key in required_keys):
            logger.warning("Missing required keys for topological indices calculation")
            return results

        # Wiener index
        try:
            results["wiener_index"] = self.kernel_manager.calculate_wiener_index(
                graph_data["csr_indices"],
                graph_data["csr_indptr"],
                graph_data["molecule_boundaries"],
                graph_data["molecule_csr_boundaries"],
            )
        except Exception as e:
            logger.warning(f"Failed to calculate Wiener index: {e}")

        # Zagreb indices
        try:
            zagreb1, zagreb2 = self.kernel_manager.calculate_zagreb_indices(
                graph_data["csr_indices"],
                graph_data["csr_indptr"],
                graph_data["molecule_boundaries"],
                graph_data["molecule_csr_boundaries"],
            )
            results["zagreb_m1"] = zagreb1
            results["zagreb_m2"] = zagreb2
        except Exception as e:
            logger.warning(f"Failed to calculate Zagreb indices: {e}")

        # Randic index
        try:
            results["randic_index"] = self.kernel_manager.calculate_randic_index(
                graph_data["csr_indices"],
                graph_data["csr_indptr"],
                graph_data["molecule_boundaries"],
                graph_data["molecule_csr_boundaries"],
            )
        except Exception as e:
            logger.warning(f"Failed to calculate Randic index: {e}")

        logger.info(f"Calculated {len(results)} topological indices")
        return results

    def calculate_molecular_similarity(
        self,
        fingerprints1: cp.ndarray,
        fingerprints2: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        fingerprint_size: int,
        similarity_type: str = "tanimoto",
    ) -> cp.ndarray:
        """Calculate molecular similarity using fingerprints.

        Args:
            fingerprints1: First set of molecular fingerprints
            fingerprints2: Second set of molecular fingerprints
            molecule_boundaries: Molecule boundaries
            fingerprint_size: Size of each fingerprint
            similarity_type: Type of similarity ('tanimoto', 'dice')

        Returns:
            Array of similarity values
        """
        if similarity_type.lower() == "tanimoto":
            return self.kernel_manager.calculate_tanimoto_similarity(
                fingerprints1, fingerprints2, molecule_boundaries, fingerprint_size
            )
        else:
            raise ValueError(f"Unsupported similarity type: {similarity_type}")

    def perform_substructure_search(
        self, molecular_data: Dict[str, Any], pattern_data: Dict[str, Any]
    ) -> cp.ndarray:
        """Perform substructure search across molecules.

        Args:
            molecular_data: Dictionary containing molecular graph data
            pattern_data: Dictionary containing pattern graph data

        Returns:
            Array of match counts for each molecule
        """
        required_mol_keys = [
            "csr_indices",
            "csr_indptr",
            "atom_types",
            "molecule_boundaries",
            "molecule_csr_boundaries",
        ]
        required_pattern_keys = ["csr_indices", "csr_indptr", "atom_types"]

        if not all(key in molecular_data for key in required_mol_keys):
            raise ValueError("Missing required molecular data keys")

        if not all(key in pattern_data for key in required_pattern_keys):
            raise ValueError("Missing required pattern data keys")

        return self.kernel_manager.substructure_matching(
            molecular_data["csr_indices"],
            molecular_data["csr_indptr"],
            molecular_data["atom_types"],
            molecular_data["molecule_boundaries"],
            molecular_data["molecule_csr_boundaries"],
            pattern_data["csr_indices"],
            pattern_data["csr_indptr"],
            pattern_data["atom_types"],
        )


# Global graph kernel manager instance
_graph_kernel_manager: Optional[GraphKernelManager] = None


def get_graph_kernel_manager() -> GraphKernelManager:
    """Get global graph CUDA kernel manager instance."""
    global _graph_kernel_manager
    if _graph_kernel_manager is None:
        _graph_kernel_manager = GraphKernelManager()
    return _graph_kernel_manager

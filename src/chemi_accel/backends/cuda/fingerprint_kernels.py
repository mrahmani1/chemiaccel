"""
CUDA Kernels for Molecular Fingerprint Calculations.

This module provides GPU-accelerated implementations of molecular fingerprint
generation, focusing on Morgan (circular) fingerprints with parallel processing
of circular substructure enumeration and bit vector generation.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

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


# CUDA kernel for Morgan fingerprint generation
MORGAN_FINGERPRINT_KERNEL = r"""
extern "C" __global__
void generate_morgan_fingerprint(
    const int* atomic_numbers,
    const int* adjacency_indices,
    const int* adjacency_indptr,
    const int* molecule_boundaries,
    const int* molecule_atom_offsets,
    int radius,
    int n_bits,
    unsigned int* fingerprints,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    // Get molecule boundaries
    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int num_atoms = end_atom - start_atom;

    // Initialize fingerprint for this molecule
    unsigned int* mol_fp = fingerprints + mol_idx * n_bits;
    for (int i = 0; i < n_bits; i++) {
        mol_fp[i] = 0;
    }

    // Temporary arrays for atom identifiers at each radius
    extern __shared__ unsigned int shared_mem[];
    unsigned int* current_ids = shared_mem + threadIdx.x * num_atoms * 2;
    unsigned int* next_ids = current_ids + num_atoms;

    // Initialize atom identifiers (radius 0)
    for (int atom = 0; atom < num_atoms; atom++) {
        current_ids[atom] = atomic_numbers[start_atom + atom];
    }

    // Generate identifiers for each radius
    for (int r = 0; r <= radius; r++) {
        // Hash current identifiers into fingerprint
        for (int atom = 0; atom < num_atoms; atom++) {
            unsigned int hash_val = current_ids[atom];

            // Simple hash function (can be improved)
            hash_val = hash_val * 2654435761U;  // Knuth's multiplicative hash
            int bit_idx = hash_val % n_bits;

            // Set bit in fingerprint (atomic operation for thread safety)
            atomicOr(&mol_fp[bit_idx / 32], 1U << (bit_idx % 32));
        }

        // If not at maximum radius, compute next radius identifiers
        if (r < radius) {
            for (int atom = 0; atom < num_atoms; atom++) {
                unsigned int new_id = current_ids[atom];

                // Get neighbors from adjacency list
                int atom_global_idx = start_atom + atom;
                int adj_start = adjacency_indptr[atom_global_idx];
                int adj_end = adjacency_indptr[atom_global_idx + 1];

                // Collect neighbor identifiers
                for (int j = adj_start; j < adj_end; j++) {
                    int neighbor_global = adjacency_indices[j];
                    int neighbor_local = neighbor_global - start_atom;

                    if (neighbor_local >= 0 && neighbor_local < num_atoms) {
                        // Combine with neighbor identifier
                        new_id = new_id * 37 + current_ids[neighbor_local];
                    }
                }

                next_ids[atom] = new_id;
            }

            // Swap arrays
            unsigned int* temp = current_ids;
            current_ids = next_ids;
            next_ids = temp;
        }
    }
}
"""

# CUDA kernel for Morgan fingerprint with counts (not binary)
MORGAN_COUNTS_KERNEL = r"""
extern "C" __global__
void generate_morgan_counts(
    const int* atomic_numbers,
    const int* adjacency_indices,
    const int* adjacency_indptr,
    const int* molecule_boundaries,
    const int* molecule_atom_offsets,
    int radius,
    unsigned int* hash_values,
    unsigned int* hash_counts,
    int* result_sizes,
    int max_unique_hashes,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    // Get molecule boundaries
    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int num_atoms = end_atom - start_atom;

    // Initialize result arrays for this molecule
    unsigned int* mol_hashes = hash_values + mol_idx * max_unique_hashes;
    unsigned int* mol_counts = hash_counts + mol_idx * max_unique_hashes;
    int unique_count = 0;

    // Temporary arrays for atom identifiers
    extern __shared__ unsigned int shared_mem[];
    unsigned int* current_ids = shared_mem + threadIdx.x * num_atoms * 2;
    unsigned int* next_ids = current_ids + num_atoms;

    // Initialize atom identifiers (radius 0)
    for (int atom = 0; atom < num_atoms; atom++) {
        current_ids[atom] = atomic_numbers[start_atom + atom];
    }

    // Generate identifiers for each radius
    for (int r = 0; r <= radius; r++) {
        // Process current identifiers
        for (int atom = 0; atom < num_atoms; atom++) {
            unsigned int hash_val = current_ids[atom];

            // Find or add hash value
            bool found = false;
            for (int i = 0; i < unique_count; i++) {
                if (mol_hashes[i] == hash_val) {
                    mol_counts[i]++;
                    found = true;
                    break;
                }
            }

            if (!found && unique_count < max_unique_hashes) {
                mol_hashes[unique_count] = hash_val;
                mol_counts[unique_count] = 1;
                unique_count++;
            }
        }

        // Compute next radius identifiers
        if (r < radius) {
            for (int atom = 0; atom < num_atoms; atom++) {
                unsigned int new_id = current_ids[atom];

                // Get neighbors
                int atom_global_idx = start_atom + atom;
                int adj_start = adjacency_indptr[atom_global_idx];
                int adj_end = adjacency_indptr[atom_global_idx + 1];

                for (int j = adj_start; j < adj_end; j++) {
                    int neighbor_global = adjacency_indices[j];
                    int neighbor_local = neighbor_global - start_atom;

                    if (neighbor_local >= 0 && neighbor_local < num_atoms) {
                        new_id = new_id * 37 + current_ids[neighbor_local];
                    }
                }

                next_ids[atom] = new_id;
            }

            // Swap arrays
            unsigned int* temp = current_ids;
            current_ids = next_ids;
            next_ids = temp;
        }
    }

    result_sizes[mol_idx] = unique_count;
}
"""

# CUDA kernel for circular substructure enumeration
CIRCULAR_SUBSTRUCTURE_KERNEL = r"""
extern "C" __global__
void enumerate_circular_substructures(
    const int* atomic_numbers,
    const int* bond_types,
    const int* adjacency_indices,
    const int* adjacency_indptr,
    const int* molecule_boundaries,
    int radius,
    unsigned int* substructure_hashes,
    int* substructure_counts,
    int max_substructures,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int num_atoms = end_atom - start_atom;

    unsigned int* mol_hashes = substructure_hashes + mol_idx * max_substructures;
    int* mol_counts = substructure_counts + mol_idx * max_substructures;
    int unique_count = 0;

    // For each atom as center
    for (int center = 0; center < num_atoms; center++) {
        int center_global = start_atom + center;

        // BFS to enumerate substructures up to radius
        extern __shared__ int shared_queue[];
        int* queue = shared_queue + threadIdx.x * num_atoms * 2;
        int* distances = queue + num_atoms;

        // Initialize BFS
        for (int i = 0; i < num_atoms; i++) {
            distances[i] = -1;
        }

        queue[0] = center;
        distances[center] = 0;
        int queue_front = 0;
        int queue_back = 1;

        // BFS traversal
        while (queue_front < queue_back) {
            int current_local = queue[queue_front++];
            int current_global = start_atom + current_local;
            int current_dist = distances[current_local];

            if (current_dist < radius) {
                // Add neighbors
                int adj_start = adjacency_indptr[current_global];
                int adj_end = adjacency_indptr[current_global + 1];

                for (int j = adj_start; j < adj_end; j++) {
                    int neighbor_global = adjacency_indices[j];
                    int neighbor_local = neighbor_global - start_atom;

                    if (neighbor_local >= 0 && neighbor_local < num_atoms &&
                        distances[neighbor_local] == -1) {
                        distances[neighbor_local] = current_dist + 1;
                        queue[queue_back++] = neighbor_local;
                    }
                }
            }
        }

        // Generate hash for this substructure
        unsigned int substructure_hash = 0;
        for (int i = 0; i < num_atoms; i++) {
            if (distances[i] >= 0 && distances[i] <= radius) {
                // Include atom in hash based on distance and atomic number
                unsigned int atom_contribution =
                    atomic_numbers[start_atom + i] * (distances[i] + 1);
                substructure_hash = substructure_hash * 31 + atom_contribution;
            }
        }

        // Add to unique substructures
        bool found = false;
        for (int i = 0; i < unique_count; i++) {
            if (mol_hashes[i] == substructure_hash) {
                mol_counts[i]++;
                found = true;
                break;
            }
        }

        if (!found && unique_count < max_substructures) {
            mol_hashes[unique_count] = substructure_hash;
            mol_counts[unique_count] = 1;
            unique_count++;
        }
    }
}
"""

# CUDA kernel for MACCS keys generation
MACCS_KEYS_KERNEL = r"""
extern "C" __global__
void generate_maccs_keys(
    const int* atomic_numbers,
    const int* bond_types,
    const int* adjacency_indices,
    const int* adjacency_indptr,
    const int* molecule_boundaries,
    unsigned int* maccs_fingerprints,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int num_atoms = end_atom - start_atom;

    // Initialize MACCS keys (167 bits, stored as 6 uint32s)
    unsigned int* mol_maccs = maccs_fingerprints + mol_idx * 6;
    for (int i = 0; i < 6; i++) {
        mol_maccs[i] = 0;
    }

    // Simple MACCS-like patterns (simplified version)
    // In practice, this would implement the full 166 MACCS key patterns

    // Key 1: Any atom
    if (num_atoms > 0) {
        atomicOr(&mol_maccs[0], 1U << 0);
    }

    // Key 2: More than 1 atom
    if (num_atoms > 1) {
        atomicOr(&mol_maccs[0], 1U << 1);
    }

    // Key 3: More than 2 atoms
    if (num_atoms > 2) {
        atomicOr(&mol_maccs[0], 1U << 2);
    }

    // Check for specific atomic numbers and patterns
    for (int atom = 0; atom < num_atoms; atom++) {
        int atom_num = atomic_numbers[start_atom + atom];

        // Key 4: Carbon
        if (atom_num == 6) {
            atomicOr(&mol_maccs[0], 1U << 3);
        }

        // Key 5: Nitrogen
        if (atom_num == 7) {
            atomicOr(&mol_maccs[0], 1U << 4);
        }

        // Key 6: Oxygen
        if (atom_num == 8) {
            atomicOr(&mol_maccs[0], 1U << 5);
        }

        // Key 7: Sulfur
        if (atom_num == 16) {
            atomicOr(&mol_maccs[0], 1U << 6);
        }

        // Key 8: Phosphorus
        if (atom_num == 15) {
            atomicOr(&mol_maccs[0], 1U << 7);
        }

        // Key 9: Halogen (F, Cl, Br, I)
        if (atom_num == 9 || atom_num == 17 || atom_num == 35 || atom_num == 53) {
            atomicOr(&mol_maccs[0], 1U << 8);
        }

        // Check connectivity patterns
        int atom_global = start_atom + atom;
        int adj_start = adjacency_indptr[atom_global];
        int adj_end = adjacency_indptr[atom_global + 1];
        int degree = adj_end - adj_start;

        // Key 10: Atom with degree > 3
        if (degree > 3) {
            atomicOr(&mol_maccs[0], 1U << 9);
        }

        // Key 11: Atom with degree > 4
        if (degree > 4) {
            atomicOr(&mol_maccs[0], 1U << 10);
        }
    }

    // Additional pattern matching would go here...
    // This is a simplified implementation for demonstration
}
"""

# CUDA kernel for topological fingerprint (RDK-like)
TOPOLOGICAL_FINGERPRINT_KERNEL = r"""
extern "C" __global__
void generate_topological_fingerprint(
    const int* atomic_numbers,
    const int* bond_types,
    const int* adjacency_indices,
    const int* adjacency_indptr,
    const int* molecule_boundaries,
    int n_bits,
    unsigned int* fingerprints,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int num_atoms = end_atom - start_atom;

    // Initialize fingerprint
    unsigned int* mol_fp = fingerprints + mol_idx * (n_bits / 32);
    for (int i = 0; i < n_bits / 32; i++) {
        mol_fp[i] = 0;
    }

    // Generate simple path-based hashes (simplified topological approach)
    // For each pair of atoms, create a hash based on their atomic numbers and distance
    for (int i = 0; i < num_atoms; i++) {
        for (int j = i + 1; j < num_atoms; j++) {
            // Simple path hash based on atomic numbers
            int atom1_num = atomic_numbers[start_atom + i];
            int atom2_num = atomic_numbers[start_atom + j];

            // Ensure consistent ordering
            if (atom1_num > atom2_num) {
                int temp = atom1_num;
                atom1_num = atom2_num;
                atom2_num = temp;
            }

            // Generate hash for this atom pair
            unsigned int pair_hash = atom1_num * 100 + atom2_num;
            pair_hash = pair_hash * 2654435761U;  // Knuth's hash

            int bit_idx = pair_hash % n_bits;
            atomicOr(&mol_fp[bit_idx / 32], 1U << (bit_idx % 32));
        }
    }

    // Add individual atom contributions
    for (int i = 0; i < num_atoms; i++) {
        unsigned int atom_hash = atomic_numbers[start_atom + i] * 2654435761U;
        int bit_idx = atom_hash % n_bits;
        atomicOr(&mol_fp[bit_idx / 32], 1U << (bit_idx % 32));
    }
}
"""

# CUDA kernel for atom pair fingerprint
ATOM_PAIR_FINGERPRINT_KERNEL = r"""
extern "C" __global__
void generate_atom_pair_fingerprint(
    const int* atomic_numbers,
    const int* adjacency_indices,
    const int* adjacency_indptr,
    const int* molecule_boundaries,
    int n_bits,
    unsigned int* fingerprints,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int num_atoms = end_atom - start_atom;

    // Initialize fingerprint
    unsigned int* mol_fp = fingerprints + mol_idx * (n_bits / 32);
    for (int i = 0; i < n_bits / 32; i++) {
        mol_fp[i] = 0;
    }

    // Calculate shortest path distances using Floyd-Warshall (simplified)
    extern __shared__ int distances[];
    int* dist_matrix = distances + threadIdx.x * num_atoms * num_atoms;

    // Initialize distance matrix
    for (int i = 0; i < num_atoms; i++) {
        for (int j = 0; j < num_atoms; j++) {
            if (i == j) {
                dist_matrix[i * num_atoms + j] = 0;
            } else {
                dist_matrix[i * num_atoms + j] = 999;  // Large number for infinity
            }
        }
    }

    // Set direct connections
    for (int i = 0; i < num_atoms; i++) {
        int atom_global = start_atom + i;
        int adj_start = adjacency_indptr[atom_global];
        int adj_end = adjacency_indptr[atom_global + 1];

        for (int j = adj_start; j < adj_end; j++) {
            int neighbor_global = adjacency_indices[j];
            int neighbor_local = neighbor_global - start_atom;

            if (neighbor_local >= 0 && neighbor_local < num_atoms) {
                dist_matrix[i * num_atoms + neighbor_local] = 1;
            }
        }
    }

    // Floyd-Warshall algorithm (simplified for small molecules)
    for (int k = 0; k < num_atoms; k++) {
        for (int i = 0; i < num_atoms; i++) {
            for (int j = 0; j < num_atoms; j++) {
                int dist_ik = dist_matrix[i * num_atoms + k];
                int dist_kj = dist_matrix[k * num_atoms + j];
                int dist_ij = dist_matrix[i * num_atoms + j];

                if (dist_ik != 999 && dist_kj != 999 && dist_ik + dist_kj < dist_ij) {
                    dist_matrix[i * num_atoms + j] = dist_ik + dist_kj;
                }
            }
        }
    }

    // Generate atom pair descriptors
    for (int i = 0; i < num_atoms; i++) {
        for (int j = i + 1; j < num_atoms; j++) {
            int distance = dist_matrix[i * num_atoms + j];

            if (distance < 999 && distance <= 15) {  // Max distance cutoff
                // Create atom pair descriptor
                int atom1_num = atomic_numbers[start_atom + i];
                int atom2_num = atomic_numbers[start_atom + j];

                // Ensure consistent ordering
                if (atom1_num > atom2_num) {
                    int temp = atom1_num;
                    atom1_num = atom2_num;
                    atom2_num = temp;
                }

                // Generate hash for atom pair
                unsigned int pair_hash = atom1_num * 1000 + atom2_num * 100 + distance;
                pair_hash = pair_hash * 2654435761U;  // Knuth's hash

                int bit_idx = pair_hash % n_bits;
                atomicOr(&mol_fp[bit_idx / 32], 1U << (bit_idx % 32));
            }
        }
    }
}
"""

# CUDA kernel for topological torsion fingerprint
TOPOLOGICAL_TORSION_FINGERPRINT_KERNEL = r"""
extern "C" __global__
void generate_topological_torsion_fingerprint(
    const int* atomic_numbers,
    const int* adjacency_indices,
    const int* adjacency_indptr,
    const int* molecule_boundaries,
    int n_bits,
    unsigned int* fingerprints,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int num_atoms = end_atom - start_atom;

    // Initialize fingerprint
    unsigned int* mol_fp = fingerprints + mol_idx * (n_bits / 32);
    for (int i = 0; i < n_bits / 32; i++) {
        mol_fp[i] = 0;
    }

    // Find all 4-atom paths (torsions)
    for (int a1 = 0; a1 < num_atoms; a1++) {
        int a1_global = start_atom + a1;
        int adj1_start = adjacency_indptr[a1_global];
        int adj1_end = adjacency_indptr[a1_global + 1];

        // For each neighbor of a1
        for (int i = adj1_start; i < adj1_end; i++) {
            int a2_global = adjacency_indices[i];
            int a2 = a2_global - start_atom;

            if (a2 < 0 || a2 >= num_atoms || a2 == a1) continue;

            int adj2_start = adjacency_indptr[a2_global];
            int adj2_end = adjacency_indptr[a2_global + 1];

            // For each neighbor of a2 (except a1)
            for (int j = adj2_start; j < adj2_end; j++) {
                int a3_global = adjacency_indices[j];
                int a3 = a3_global - start_atom;

                if (a3 < 0 || a3 >= num_atoms || a3 == a1 || a3 == a2) continue;

                int adj3_start = adjacency_indptr[a3_global];
                int adj3_end = adjacency_indptr[a3_global + 1];

                // For each neighbor of a3 (except a2)
                for (int k = adj3_start; k < adj3_end; k++) {
                    int a4_global = adjacency_indices[k];
                    int a4 = a4_global - start_atom;

                    if (a4 < 0 || a4 >= num_atoms || a4 == a1 || a4 == a2 || a4 == a3) continue;

                    // We have a 4-atom torsion: a1-a2-a3-a4
                    // Generate torsion descriptor
                    int atom1_num = atomic_numbers[start_atom + a1];
                    int atom2_num = atomic_numbers[start_atom + a2];
                    int atom3_num = atomic_numbers[start_atom + a3];
                    int atom4_num = atomic_numbers[start_atom + a4];

                    // Create canonical torsion representation
                    unsigned int torsion_hash;
                    if (atom1_num < atom4_num || (atom1_num == atom4_num && atom2_num <= atom3_num)) {
                        torsion_hash = atom1_num * 1000000 + atom2_num * 10000 + atom3_num * 100 + atom4_num;
                    } else {
                        torsion_hash = atom4_num * 1000000 + atom3_num * 10000 + atom2_num * 100 + atom1_num;
                    }

                    torsion_hash = torsion_hash * 2654435761U;  // Knuth's hash

                    int bit_idx = torsion_hash % n_bits;
                    atomicOr(&mol_fp[bit_idx / 32], 1U << (bit_idx % 32));
                }
            }
        }
    }
}
"""


class CUDAFingerprintKernelManager:
    """Manages CUDA kernels for molecular fingerprint calculations."""

    def __init__(self, memory_manager=None):
        """Initialize CUDA fingerprint kernel manager.

        Args:
            memory_manager: CUDA memory manager instance
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self.memory_manager = memory_manager or get_memory_manager()
        self._kernels = {}
        self._compile_kernels()

    def _compile_kernels(self):
        """Compile all CUDA fingerprint kernels."""
        kernel_sources = {
            "morgan_fingerprint": MORGAN_FINGERPRINT_KERNEL,
            "morgan_counts": MORGAN_COUNTS_KERNEL,
            "circular_substructures": CIRCULAR_SUBSTRUCTURE_KERNEL,
            "maccs_keys": MACCS_KEYS_KERNEL,
            "topological_fingerprint": TOPOLOGICAL_FINGERPRINT_KERNEL,
            "atom_pair_fingerprint": ATOM_PAIR_FINGERPRINT_KERNEL,
            "topological_torsion_fingerprint": TOPOLOGICAL_TORSION_FINGERPRINT_KERNEL,
        }

        kernel_functions = {
            "morgan_fingerprint": "generate_morgan_fingerprint",
            "morgan_counts": "generate_morgan_counts",
            "circular_substructures": "enumerate_circular_substructures",
            "maccs_keys": "generate_maccs_keys",
            "topological_fingerprint": "generate_topological_fingerprint",
            "atom_pair_fingerprint": "generate_atom_pair_fingerprint",
            "topological_torsion_fingerprint": "generate_topological_torsion_fingerprint",
        }

        for name, source in kernel_sources.items():
            try:
                kernel = RawKernel(source, kernel_functions[name])
                self._kernels[name] = kernel
                logger.debug(f"Compiled CUDA fingerprint kernel: {name}")
            except Exception as e:
                logger.error(f"Failed to compile fingerprint kernel {name}: {e}")
                raise

        logger.info(f"Compiled {len(self._kernels)} CUDA fingerprint kernels")

    def generate_morgan_fingerprints(
        self,
        atomic_numbers: cp.ndarray,
        adjacency_indices: cp.ndarray,
        adjacency_indptr: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        molecule_atom_offsets: cp.ndarray,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> cp.ndarray:
        """Generate Morgan fingerprints using CUDA kernel.

        Args:
            atomic_numbers: Array of atomic numbers
            adjacency_indices: CSR adjacency matrix indices
            adjacency_indptr: CSR adjacency matrix indptr
            molecule_boundaries: Array indicating end of each molecule
            molecule_atom_offsets: Array of atom offsets for each molecule
            radius: Morgan fingerprint radius
            n_bits: Number of bits in fingerprint

        Returns:
            Array of Morgan fingerprints (num_molecules, n_bits//32) as uint32
        """
        num_molecules = len(molecule_boundaries)
        max_atoms = int(
            cp.max(cp.diff(cp.concatenate([cp.array([0]), molecule_boundaries])))
        )

        # Allocate output array (packed bits as uint32)
        fingerprints = cp.zeros((num_molecules, n_bits // 32), dtype=cp.uint32)

        # Calculate shared memory size needed
        shared_mem_size = max_atoms * 2 * 4  # 2 arrays of max_atoms uint32s

        # Launch kernel
        block_size = min(256, num_molecules)
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["morgan_fingerprint"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                atomic_numbers,
                adjacency_indices,
                adjacency_indptr,
                molecule_boundaries,
                molecule_atom_offsets,
                radius,
                n_bits,
                fingerprints,
                num_molecules,
            ),
            shared_mem=shared_mem_size,
        )

        return fingerprints

    def generate_morgan_counts(
        self,
        atomic_numbers: cp.ndarray,
        adjacency_indices: cp.ndarray,
        adjacency_indptr: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        molecule_atom_offsets: cp.ndarray,
        radius: int = 2,
        max_unique_hashes: int = 1000,
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Generate Morgan fingerprint counts using CUDA kernel.

        Args:
            atomic_numbers: Array of atomic numbers
            adjacency_indices: CSR adjacency matrix indices
            adjacency_indptr: CSR adjacency matrix indptr
            molecule_boundaries: Array indicating end of each molecule
            molecule_atom_offsets: Array of atom offsets for each molecule
            radius: Morgan fingerprint radius
            max_unique_hashes: Maximum number of unique hash values per molecule

        Returns:
            Tuple of (hash_values, hash_counts, result_sizes)
        """
        num_molecules = len(molecule_boundaries)
        max_atoms = int(
            cp.max(cp.diff(cp.concatenate([cp.array([0]), molecule_boundaries])))
        )

        # Allocate output arrays
        hash_values = cp.zeros((num_molecules, max_unique_hashes), dtype=cp.uint32)
        hash_counts = cp.zeros((num_molecules, max_unique_hashes), dtype=cp.uint32)
        result_sizes = cp.zeros(num_molecules, dtype=cp.int32)

        # Calculate shared memory size
        shared_mem_size = max_atoms * 2 * 4  # 2 arrays of max_atoms uint32s

        # Launch kernel
        block_size = min(256, num_molecules)
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["morgan_counts"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                atomic_numbers,
                adjacency_indices,
                adjacency_indptr,
                molecule_boundaries,
                molecule_atom_offsets,
                radius,
                hash_values,
                hash_counts,
                result_sizes,
                max_unique_hashes,
                num_molecules,
            ),
            shared_mem=shared_mem_size,
        )

        return hash_values, hash_counts, result_sizes

    def enumerate_circular_substructures(
        self,
        atomic_numbers: cp.ndarray,
        bond_types: cp.ndarray,
        adjacency_indices: cp.ndarray,
        adjacency_indptr: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        radius: int = 2,
        max_substructures: int = 1000,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Enumerate circular substructures using CUDA kernel.

        Args:
            atomic_numbers: Array of atomic numbers
            bond_types: Array of bond types
            adjacency_indices: CSR adjacency matrix indices
            adjacency_indptr: CSR adjacency matrix indptr
            molecule_boundaries: Array indicating end of each molecule
            radius: Substructure radius
            max_substructures: Maximum substructures per molecule

        Returns:
            Tuple of (substructure_hashes, substructure_counts)
        """
        num_molecules = len(molecule_boundaries)
        max_atoms = int(
            cp.max(cp.diff(cp.concatenate([cp.array([0]), molecule_boundaries])))
        )

        # Allocate output arrays
        substructure_hashes = cp.zeros(
            (num_molecules, max_substructures), dtype=cp.uint32
        )
        substructure_counts = cp.zeros(
            (num_molecules, max_substructures), dtype=cp.int32
        )

        # Calculate shared memory size (queue + distances)
        shared_mem_size = max_atoms * 2 * 4  # 2 arrays of max_atoms int32s

        # Launch kernel
        block_size = min(256, num_molecules)
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["circular_substructures"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                atomic_numbers,
                bond_types,
                adjacency_indices,
                adjacency_indptr,
                molecule_boundaries,
                radius,
                substructure_hashes,
                substructure_counts,
                max_substructures,
                num_molecules,
            ),
            shared_mem=shared_mem_size,
        )

        return substructure_hashes, substructure_counts

    def generate_maccs_keys(
        self,
        atomic_numbers: cp.ndarray,
        bond_types: cp.ndarray,
        adjacency_indices: cp.ndarray,
        adjacency_indptr: cp.ndarray,
        molecule_boundaries: cp.ndarray,
    ) -> cp.ndarray:
        """Generate MACCS keys using CUDA kernel.

        Args:
            atomic_numbers: Array of atomic numbers
            bond_types: Array of bond types
            adjacency_indices: CSR adjacency matrix indices
            adjacency_indptr: CSR adjacency matrix indptr
            molecule_boundaries: Array indicating end of each molecule

        Returns:
            Array of MACCS keys (num_molecules, 6) as uint32 (167 bits packed)
        """
        num_molecules = len(molecule_boundaries)

        # Allocate output array (167 bits packed into 6 uint32s)
        maccs_fingerprints = cp.zeros((num_molecules, 6), dtype=cp.uint32)

        # Launch kernel
        block_size = min(256, num_molecules)
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["maccs_keys"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                atomic_numbers,
                bond_types,
                adjacency_indices,
                adjacency_indptr,
                molecule_boundaries,
                maccs_fingerprints,
                num_molecules,
            ),
        )

        return maccs_fingerprints

    def generate_topological_fingerprints(
        self,
        atomic_numbers: cp.ndarray,
        bond_types: cp.ndarray,
        adjacency_indices: cp.ndarray,
        adjacency_indptr: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        n_bits: int = 2048,
    ) -> cp.ndarray:
        """Generate topological fingerprints using CUDA kernel.

        Args:
            atomic_numbers: Array of atomic numbers
            bond_types: Array of bond types
            adjacency_indices: CSR adjacency matrix indices
            adjacency_indptr: CSR adjacency matrix indptr
            molecule_boundaries: Array indicating end of each molecule
            n_bits: Number of bits in fingerprint

        Returns:
            Array of topological fingerprints (num_molecules, n_bits//32) as uint32
        """
        num_molecules = len(molecule_boundaries)
        max_atoms = int(
            cp.max(cp.diff(cp.concatenate([cp.array([0]), molecule_boundaries])))
        )

        # Allocate output array (packed bits as uint32)
        fingerprints = cp.zeros((num_molecules, n_bits // 32), dtype=cp.uint32)

        # Calculate shared memory size (path + visited arrays)
        shared_mem_size = (8 + max_atoms) * 4  # path array + visited array

        # Launch kernel
        block_size = min(256, num_molecules)
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["topological_fingerprint"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                atomic_numbers,
                bond_types,
                adjacency_indices,
                adjacency_indptr,
                molecule_boundaries,
                n_bits,
                fingerprints,
                num_molecules,
            ),
            shared_mem=shared_mem_size,
        )

        return fingerprints

    def generate_atom_pair_fingerprints(
        self,
        atomic_numbers: cp.ndarray,
        adjacency_indices: cp.ndarray,
        adjacency_indptr: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        n_bits: int = 2048,
    ) -> cp.ndarray:
        """Generate atom pair fingerprints using CUDA kernel.

        Args:
            atomic_numbers: Array of atomic numbers
            adjacency_indices: CSR adjacency matrix indices
            adjacency_indptr: CSR adjacency matrix indptr
            molecule_boundaries: Array indicating end of each molecule
            n_bits: Number of bits in fingerprint

        Returns:
            Array of atom pair fingerprints (num_molecules, n_bits//32) as uint32
        """
        num_molecules = len(molecule_boundaries)
        max_atoms = int(
            cp.max(cp.diff(cp.concatenate([cp.array([0]), molecule_boundaries])))
        )

        # Allocate output array (packed bits as uint32)
        fingerprints = cp.zeros((num_molecules, n_bits // 32), dtype=cp.uint32)

        # Calculate shared memory size (distance matrix)
        shared_mem_size = max_atoms * max_atoms * 4  # int32 distance matrix

        # Launch kernel
        block_size = min(
            64, num_molecules
        )  # Smaller block size due to shared memory usage
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["atom_pair_fingerprint"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                atomic_numbers,
                adjacency_indices,
                adjacency_indptr,
                molecule_boundaries,
                n_bits,
                fingerprints,
                num_molecules,
            ),
            shared_mem=shared_mem_size,
        )

        return fingerprints

    def generate_topological_torsion_fingerprints(
        self,
        atomic_numbers: cp.ndarray,
        adjacency_indices: cp.ndarray,
        adjacency_indptr: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        n_bits: int = 2048,
    ) -> cp.ndarray:
        """Generate topological torsion fingerprints using CUDA kernel.

        Args:
            atomic_numbers: Array of atomic numbers
            adjacency_indices: CSR adjacency matrix indices
            adjacency_indptr: CSR adjacency matrix indptr
            molecule_boundaries: Array indicating end of each molecule
            n_bits: Number of bits in fingerprint

        Returns:
            Array of topological torsion fingerprints (num_molecules, n_bits//32) as uint32
        """
        num_molecules = len(molecule_boundaries)

        # Allocate output array (packed bits as uint32)
        fingerprints = cp.zeros((num_molecules, n_bits // 32), dtype=cp.uint32)

        # Launch kernel
        block_size = min(256, num_molecules)
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["topological_torsion_fingerprint"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                atomic_numbers,
                adjacency_indices,
                adjacency_indptr,
                molecule_boundaries,
                n_bits,
                fingerprints,
                num_molecules,
            ),
        )

        return fingerprints


class GPUFingerprintGenerator:
    """GPU-accelerated fingerprint generator for all fingerprint types."""

    def __init__(self, memory_manager=None):
        """Initialize GPU fingerprint generator.

        Args:
            memory_manager: CUDA memory manager instance
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self.memory_manager = memory_manager or get_memory_manager()
        self.kernel_manager = CUDAFingerprintKernelManager(memory_manager)

    def generate_batch_fingerprints(
        self,
        molecular_data: Dict[str, Any],
        radius: int = 2,
        n_bits: int = 2048,
        return_counts: bool = False,
    ) -> Dict[str, cp.ndarray]:
        """Generate Morgan fingerprints for a batch of molecules.

        Args:
            molecular_data: Dictionary containing molecular graph data
            radius: Morgan fingerprint radius
            n_bits: Number of bits in fingerprint (for binary version)
            return_counts: Whether to return count-based fingerprints

        Returns:
            Dictionary containing fingerprint results
        """
        required_keys = [
            "atomic_numbers",
            "adjacency_indices",
            "adjacency_indptr",
            "molecule_boundaries",
            "molecule_atom_offsets",
        ]

        if not all(key in molecular_data for key in required_keys):
            missing = [key for key in required_keys if key not in molecular_data]
            raise ValueError(f"Missing required molecular data keys: {missing}")

        results = {}

        if return_counts:
            # Generate count-based fingerprints
            (
                hash_values,
                hash_counts,
                result_sizes,
            ) = self.kernel_manager.generate_morgan_counts(
                molecular_data["atomic_numbers"],
                molecular_data["adjacency_indices"],
                molecular_data["adjacency_indptr"],
                molecular_data["molecule_boundaries"],
                molecular_data["molecule_atom_offsets"],
                radius=radius,
            )

            results["morgan_hash_values"] = hash_values
            results["morgan_hash_counts"] = hash_counts
            results["morgan_result_sizes"] = result_sizes
        else:
            # Generate binary fingerprints
            fingerprints = self.kernel_manager.generate_morgan_fingerprints(
                molecular_data["atomic_numbers"],
                molecular_data["adjacency_indices"],
                molecular_data["adjacency_indptr"],
                molecular_data["molecule_boundaries"],
                molecular_data["molecule_atom_offsets"],
                radius=radius,
                n_bits=n_bits,
            )

            results["morgan_fingerprints"] = fingerprints

        # Also enumerate circular substructures if bond data available
        if "bond_types" in molecular_data:
            (
                substructure_hashes,
                substructure_counts,
            ) = self.kernel_manager.enumerate_circular_substructures(
                molecular_data["atomic_numbers"],
                molecular_data["bond_types"],
                molecular_data["adjacency_indices"],
                molecular_data["adjacency_indptr"],
                molecular_data["molecule_boundaries"],
                radius=radius,
            )

            results["substructure_hashes"] = substructure_hashes
            results["substructure_counts"] = substructure_counts

        logger.info(
            f"Generated Morgan fingerprints for {len(molecular_data['molecule_boundaries'])} molecules"
        )
        return results

    def convert_to_numpy_bits(
        self, packed_fingerprints: cp.ndarray, n_bits: int
    ) -> np.ndarray:
        """Convert packed uint32 fingerprints to numpy bit arrays.

        Args:
            packed_fingerprints: Packed fingerprints as uint32 arrays
            n_bits: Number of bits per fingerprint

        Returns:
            Numpy array of shape (num_molecules, n_bits) with binary values
        """
        num_molecules = packed_fingerprints.shape[0]

        # Convert to CPU
        packed_cpu = cp.asnumpy(packed_fingerprints)

        # Unpack bits
        bit_fingerprints = np.zeros((num_molecules, n_bits), dtype=np.int8)

        for mol_idx in range(num_molecules):
            for word_idx in range(n_bits // 32):
                word = packed_cpu[mol_idx, word_idx]
                for bit_idx in range(32):
                    if word & (1 << bit_idx):
                        bit_fingerprints[mol_idx, word_idx * 32 + bit_idx] = 1

        return bit_fingerprints

    def generate_batch_maccs_keys(
        self, molecular_data: Dict[str, Any]
    ) -> Dict[str, cp.ndarray]:
        """Generate MACCS keys for a batch of molecules.

        Args:
            molecular_data: Dictionary containing molecular graph data

        Returns:
            Dictionary containing MACCS keys results
        """
        required_keys = [
            "atomic_numbers",
            "bond_types",
            "adjacency_indices",
            "adjacency_indptr",
            "molecule_boundaries",
        ]

        if not all(key in molecular_data for key in required_keys):
            missing = [key for key in required_keys if key not in molecular_data]
            raise ValueError(f"Missing required molecular data keys: {missing}")

        # Generate MACCS keys using GPU
        maccs_keys = self.kernel_manager.generate_maccs_keys(
            molecular_data["atomic_numbers"],
            molecular_data["bond_types"],
            molecular_data["adjacency_indices"],
            molecular_data["adjacency_indptr"],
            molecular_data["molecule_boundaries"],
        )

        results = {"maccs_keys": maccs_keys}

        logger.info(
            f"Generated MACCS keys for {len(molecular_data['molecule_boundaries'])} molecules"
        )
        return results

    def generate_batch_topological_fingerprints(
        self, molecular_data: Dict[str, Any], n_bits: int = 2048
    ) -> Dict[str, cp.ndarray]:
        """Generate topological fingerprints for a batch of molecules.

        Args:
            molecular_data: Dictionary containing molecular graph data
            n_bits: Number of bits in fingerprint

        Returns:
            Dictionary containing topological fingerprint results
        """
        required_keys = [
            "atomic_numbers",
            "bond_types",
            "adjacency_indices",
            "adjacency_indptr",
            "molecule_boundaries",
        ]

        if not all(key in molecular_data for key in required_keys):
            missing = [key for key in required_keys if key not in molecular_data]
            raise ValueError(f"Missing required molecular data keys: {missing}")

        # Generate topological fingerprints using GPU
        fingerprints = self.kernel_manager.generate_topological_fingerprints(
            molecular_data["atomic_numbers"],
            molecular_data["bond_types"],
            molecular_data["adjacency_indices"],
            molecular_data["adjacency_indptr"],
            molecular_data["molecule_boundaries"],
            n_bits=n_bits,
        )

        results = {"topological_fingerprints": fingerprints}

        logger.info(
            f"Generated topological fingerprints for {len(molecular_data['molecule_boundaries'])} molecules"
        )
        return results

    def generate_batch_atom_pair_fingerprints(
        self, molecular_data: Dict[str, Any], n_bits: int = 2048
    ) -> Dict[str, cp.ndarray]:
        """Generate atom pair fingerprints for a batch of molecules.

        Args:
            molecular_data: Dictionary containing molecular graph data
            n_bits: Number of bits in fingerprint

        Returns:
            Dictionary containing atom pair fingerprint results
        """
        required_keys = [
            "atomic_numbers",
            "adjacency_indices",
            "adjacency_indptr",
            "molecule_boundaries",
        ]

        if not all(key in molecular_data for key in required_keys):
            missing = [key for key in required_keys if key not in molecular_data]
            raise ValueError(f"Missing required molecular data keys: {missing}")

        # Generate atom pair fingerprints using GPU
        fingerprints = self.kernel_manager.generate_atom_pair_fingerprints(
            molecular_data["atomic_numbers"],
            molecular_data["adjacency_indices"],
            molecular_data["adjacency_indptr"],
            molecular_data["molecule_boundaries"],
            n_bits=n_bits,
        )

        results = {"atom_pair_fingerprints": fingerprints}

        logger.info(
            f"Generated atom pair fingerprints for {len(molecular_data['molecule_boundaries'])} molecules"
        )
        return results

    def generate_batch_topological_torsion_fingerprints(
        self, molecular_data: Dict[str, Any], n_bits: int = 2048
    ) -> Dict[str, cp.ndarray]:
        """Generate topological torsion fingerprints for a batch of molecules.

        Args:
            molecular_data: Dictionary containing molecular graph data
            n_bits: Number of bits in fingerprint

        Returns:
            Dictionary containing topological torsion fingerprint results
        """
        required_keys = [
            "atomic_numbers",
            "adjacency_indices",
            "adjacency_indptr",
            "molecule_boundaries",
        ]

        if not all(key in molecular_data for key in required_keys):
            missing = [key for key in required_keys if key not in molecular_data]
            raise ValueError(f"Missing required molecular data keys: {missing}")

        # Generate topological torsion fingerprints using GPU
        fingerprints = self.kernel_manager.generate_topological_torsion_fingerprints(
            molecular_data["atomic_numbers"],
            molecular_data["adjacency_indices"],
            molecular_data["adjacency_indptr"],
            molecular_data["molecule_boundaries"],
            n_bits=n_bits,
        )

        results = {"topological_torsion_fingerprints": fingerprints}

        logger.info(
            f"Generated topological torsion fingerprints for {len(molecular_data['molecule_boundaries'])} molecules"
        )
        return results

    def convert_maccs_to_numpy_bits(self, packed_maccs: cp.ndarray) -> np.ndarray:
        """Convert packed uint32 MACCS keys to numpy bit arrays.

        Args:
            packed_maccs: Packed MACCS keys as uint32 arrays (num_molecules, 6)

        Returns:
            Numpy array of shape (num_molecules, 167) with binary values
        """
        num_molecules = packed_maccs.shape[0]

        # Convert to CPU
        packed_cpu = cp.asnumpy(packed_maccs)

        # Unpack bits (167 bits total)
        bit_maccs = np.zeros((num_molecules, 167), dtype=np.int8)

        for mol_idx in range(num_molecules):
            for word_idx in range(6):  # 6 uint32 words for 167 bits
                word = packed_cpu[mol_idx, word_idx]
                bits_in_word = min(32, 167 - word_idx * 32)  # Handle last word

                for bit_idx in range(bits_in_word):
                    if word & (1 << bit_idx):
                        bit_maccs[mol_idx, word_idx * 32 + bit_idx] = 1

        return bit_maccs


# Maintain backward compatibility
GPUMorganFingerprintGenerator = GPUFingerprintGenerator

# Global fingerprint kernel manager instance
_fingerprint_kernel_manager: Optional[CUDAFingerprintKernelManager] = None


def get_fingerprint_kernel_manager() -> CUDAFingerprintKernelManager:
    """Get global CUDA fingerprint kernel manager instance."""
    global _fingerprint_kernel_manager
    if _fingerprint_kernel_manager is None:
        _fingerprint_kernel_manager = CUDAFingerprintKernelManager()
    return _fingerprint_kernel_manager

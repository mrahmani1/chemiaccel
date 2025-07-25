"""
CUDA Kernels for Molecular Descriptor Calculations.

This module provides CUDA kernel implementations for embarrassingly parallel
molecular descriptor calculations including molecular weight, atom counts,
bond counts, TPSA, rotatable bonds, and aromatic ring detection.
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


# CUDA kernel source code for basic molecular descriptors
MOLECULAR_WEIGHT_KERNEL = r"""
extern "C" __global__
void calculate_molecular_weight(
    const float* atom_weights,
    const int* molecule_boundaries,
    float* results,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];

    float total_weight = 0.0f;
    for (int i = start_atom; i < end_atom; i++) {
        total_weight += atom_weights[i];
    }

    results[mol_idx] = total_weight;
}
"""

ATOM_COUNT_KERNEL = r"""
extern "C" __global__
void calculate_atom_counts(
    const int* atomic_numbers,
    const int* molecule_boundaries,
    int* heavy_atom_counts,
    int* total_atom_counts,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];

    int heavy_count = 0;
    int total_count = end_atom - start_atom;

    for (int i = start_atom; i < end_atom; i++) {
        if (atomic_numbers[i] != 1) {  // Not hydrogen
            heavy_count++;
        }
    }

    heavy_atom_counts[mol_idx] = heavy_count;
    total_atom_counts[mol_idx] = total_count;
}
"""

BOND_COUNT_KERNEL = r"""
extern "C" __global__
void calculate_bond_counts(
    const int* bond_types,
    const int* molecule_bond_boundaries,
    int* single_bonds,
    int* double_bonds,
    int* triple_bonds,
    int* total_bonds,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_bond = (mol_idx == 0) ? 0 : molecule_bond_boundaries[mol_idx - 1];
    int end_bond = molecule_bond_boundaries[mol_idx];

    int single_count = 0;
    int double_count = 0;
    int triple_count = 0;
    int total_count = end_bond - start_bond;

    for (int i = start_bond; i < end_bond; i++) {
        switch (bond_types[i]) {
            case 1: single_count++; break;
            case 2: double_count++; break;
            case 3: triple_count++; break;
        }
    }

    single_bonds[mol_idx] = single_count;
    double_bonds[mol_idx] = double_count;
    triple_bonds[mol_idx] = triple_count;
    total_bonds[mol_idx] = total_count;
}
"""

TPSA_KERNEL = r"""
extern "C" __global__
void calculate_tpsa(
    const int* atomic_numbers,
    const int* hybridizations,
    const int* hydrogen_counts,
    const bool* is_aromatic,
    const int* molecule_boundaries,
    float* results,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];

    float tpsa = 0.0f;

    for (int i = start_atom; i < end_atom; i++) {
        int atomic_num = atomic_numbers[i];
        int hyb = hybridizations[i];
        int h_count = hydrogen_counts[i];
        bool aromatic = is_aromatic[i];

        // TPSA contributions based on atom type and environment
        if (atomic_num == 7) {  // Nitrogen
            if (aromatic) {
                tpsa += 12.89f;  // Aromatic nitrogen
            } else if (hyb == 3) {  // sp3
                tpsa += 3.24f;
            } else if (hyb == 2) {  // sp2
                if (h_count == 0) {
                    tpsa += 12.36f;  // Tertiary amine
                } else {
                    tpsa += 26.02f;  // Secondary/primary amine
                }
            }
        } else if (atomic_num == 8) {  // Oxygen
            if (hyb == 3) {  // sp3 (alcohol, ether)
                if (h_count > 0) {
                    tpsa += 20.23f;  // Alcohol
                } else {
                    tpsa += 9.23f;   // Ether
                }
            } else if (hyb == 2) {  // sp2 (carbonyl)
                tpsa += 17.07f;
            }
        } else if (atomic_num == 16) {  // Sulfur
            if (hyb == 3) {
                tpsa += 25.30f;  // Sulfide
            } else if (hyb == 2) {
                tpsa += 32.09f;  // Sulfoxide/sulfone
            }
        }
    }

    results[mol_idx] = tpsa;
}
"""

ROTATABLE_BONDS_KERNEL = r"""
extern "C" __global__
void calculate_rotatable_bonds(
    const int* bond_atom1,
    const int* bond_atom2,
    const int* bond_types,
    const bool* bond_in_ring,
    const int* atomic_numbers,
    const int* molecule_bond_boundaries,
    int* results,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_bond = (mol_idx == 0) ? 0 : molecule_bond_boundaries[mol_idx - 1];
    int end_bond = molecule_bond_boundaries[mol_idx];

    int rotatable_count = 0;

    for (int i = start_bond; i < end_bond; i++) {
        // Check if bond is rotatable:
        // 1. Single bond (type 1)
        // 2. Not in ring
        // 3. Not terminal (both atoms have degree > 1)
        // 4. Not involving hydrogen

        if (bond_types[i] == 1 && !bond_in_ring[i]) {
            int atom1 = bond_atom1[i];
            int atom2 = bond_atom2[i];

            // Skip if either atom is hydrogen
            if (atomic_numbers[atom1] == 1 || atomic_numbers[atom2] == 1) {
                continue;
            }

            // This is a simplified check - in practice would need to count
            // actual degrees of atoms, but for this kernel we assume
            // non-hydrogen atoms in the bond list have degree > 1
            rotatable_count++;
        }
    }

    results[mol_idx] = rotatable_count;
}
"""

AROMATIC_RINGS_KERNEL = r"""
extern "C" __global__
void calculate_aromatic_rings(
    const bool* atom_is_aromatic,
    const int* bond_atom1,
    const int* bond_atom2,
    const bool* bond_is_aromatic,
    const int* molecule_boundaries,
    const int* molecule_bond_boundaries,
    int* results,
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mol_idx >= num_molecules) return;

    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int start_bond = (mol_idx == 0) ? 0 : molecule_bond_boundaries[mol_idx - 1];
    int end_bond = molecule_bond_boundaries[mol_idx];

    // Count aromatic atoms
    int aromatic_atom_count = 0;
    for (int i = start_atom; i < end_atom; i++) {
        if (atom_is_aromatic[i]) {
            aromatic_atom_count++;
        }
    }

    // Estimate aromatic rings based on aromatic atoms
    // This is a simplified heuristic: aromatic_atoms / 6 for benzene-like rings
    int estimated_rings = (aromatic_atom_count + 5) / 6;  // Round up division

    results[mol_idx] = estimated_rings;
}
"""


class CUDAKernelManager:
    """Manages CUDA kernels for molecular descriptor calculations."""

    def __init__(self, memory_manager=None):
        """Initialize CUDA kernel manager.

        Args:
            memory_manager: CUDA memory manager instance
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self.memory_manager = memory_manager or get_memory_manager()
        self._kernels = {}
        self._compile_kernels()

    def _compile_kernels(self):
        """Compile all CUDA kernels."""
        kernel_sources = {
            "molecular_weight": MOLECULAR_WEIGHT_KERNEL,
            "atom_counts": ATOM_COUNT_KERNEL,
            "bond_counts": BOND_COUNT_KERNEL,
            "tpsa": TPSA_KERNEL,
            "rotatable_bonds": ROTATABLE_BONDS_KERNEL,
            "aromatic_rings": AROMATIC_RINGS_KERNEL,
        }

        for name, source in kernel_sources.items():
            try:
                kernel = RawKernel(source, f"calculate_{name}")
                self._kernels[name] = kernel
                logger.debug(f"Compiled CUDA kernel: {name}")
            except Exception as e:
                logger.error(f"Failed to compile kernel {name}: {e}")
                raise

        logger.info(f"Compiled {len(self._kernels)} CUDA kernels")

    def calculate_molecular_weights(
        self, atom_weights: cp.ndarray, molecule_boundaries: cp.ndarray
    ) -> cp.ndarray:
        """Calculate molecular weights using CUDA kernel.

        Args:
            atom_weights: Array of atomic weights
            molecule_boundaries: Array indicating end of each molecule

        Returns:
            Array of molecular weights for each molecule
        """
        num_molecules = len(molecule_boundaries)
        results = cp.zeros(num_molecules, dtype=cp.float32)

        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["molecular_weight"]
        kernel(
            (grid_size,),
            (block_size,),
            (atom_weights, molecule_boundaries, results, num_molecules),
        )

        return results

    def calculate_atom_counts(
        self, atomic_numbers: cp.ndarray, molecule_boundaries: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Calculate heavy atom and total atom counts.

        Args:
            atomic_numbers: Array of atomic numbers
            molecule_boundaries: Array indicating end of each molecule

        Returns:
            Tuple of (heavy_atom_counts, total_atom_counts)
        """
        num_molecules = len(molecule_boundaries)
        heavy_counts = cp.zeros(num_molecules, dtype=cp.int32)
        total_counts = cp.zeros(num_molecules, dtype=cp.int32)

        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["atom_counts"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                atomic_numbers,
                molecule_boundaries,
                heavy_counts,
                total_counts,
                num_molecules,
            ),
        )

        return heavy_counts, total_counts

    def calculate_bond_counts(
        self, bond_types: cp.ndarray, molecule_bond_boundaries: cp.ndarray
    ) -> Dict[str, cp.ndarray]:
        """Calculate bond counts by type.

        Args:
            bond_types: Array of bond types (1=single, 2=double, 3=triple)
            molecule_bond_boundaries: Array indicating end of bonds for each molecule

        Returns:
            Dictionary with bond count arrays
        """
        num_molecules = len(molecule_bond_boundaries)
        single_bonds = cp.zeros(num_molecules, dtype=cp.int32)
        double_bonds = cp.zeros(num_molecules, dtype=cp.int32)
        triple_bonds = cp.zeros(num_molecules, dtype=cp.int32)
        total_bonds = cp.zeros(num_molecules, dtype=cp.int32)

        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["bond_counts"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                bond_types,
                molecule_bond_boundaries,
                single_bonds,
                double_bonds,
                triple_bonds,
                total_bonds,
                num_molecules,
            ),
        )

        return {
            "single_bonds": single_bonds,
            "double_bonds": double_bonds,
            "triple_bonds": triple_bonds,
            "total_bonds": total_bonds,
        }

    def calculate_tpsa(
        self,
        atomic_numbers: cp.ndarray,
        hybridizations: cp.ndarray,
        hydrogen_counts: cp.ndarray,
        is_aromatic: cp.ndarray,
        molecule_boundaries: cp.ndarray,
    ) -> cp.ndarray:
        """Calculate Topological Polar Surface Area (TPSA).

        Args:
            atomic_numbers: Array of atomic numbers
            hybridizations: Array of hybridization states
            hydrogen_counts: Array of hydrogen counts per atom
            is_aromatic: Array of aromatic flags per atom
            molecule_boundaries: Array indicating end of each molecule

        Returns:
            Array of TPSA values for each molecule
        """
        num_molecules = len(molecule_boundaries)
        results = cp.zeros(num_molecules, dtype=cp.float32)

        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["tpsa"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                atomic_numbers,
                hybridizations,
                hydrogen_counts,
                is_aromatic,
                molecule_boundaries,
                results,
                num_molecules,
            ),
        )

        return results

    def calculate_rotatable_bonds(
        self,
        bond_atom1: cp.ndarray,
        bond_atom2: cp.ndarray,
        bond_types: cp.ndarray,
        bond_in_ring: cp.ndarray,
        atomic_numbers: cp.ndarray,
        molecule_bond_boundaries: cp.ndarray,
    ) -> cp.ndarray:
        """Calculate number of rotatable bonds.

        Args:
            bond_atom1: Array of first atom indices for each bond
            bond_atom2: Array of second atom indices for each bond
            bond_types: Array of bond types
            bond_in_ring: Array of ring membership flags for bonds
            atomic_numbers: Array of atomic numbers
            molecule_bond_boundaries: Array indicating end of bonds for each molecule

        Returns:
            Array of rotatable bond counts for each molecule
        """
        num_molecules = len(molecule_bond_boundaries)
        results = cp.zeros(num_molecules, dtype=cp.int32)

        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["rotatable_bonds"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                bond_atom1,
                bond_atom2,
                bond_types,
                bond_in_ring,
                atomic_numbers,
                molecule_bond_boundaries,
                results,
                num_molecules,
            ),
        )

        return results

    def calculate_aromatic_rings(
        self,
        atom_is_aromatic: cp.ndarray,
        bond_atom1: cp.ndarray,
        bond_atom2: cp.ndarray,
        bond_is_aromatic: cp.ndarray,
        molecule_boundaries: cp.ndarray,
        molecule_bond_boundaries: cp.ndarray,
    ) -> cp.ndarray:
        """Calculate number of aromatic rings.

        Args:
            atom_is_aromatic: Array of aromatic flags per atom
            bond_atom1: Array of first atom indices for each bond
            bond_atom2: Array of second atom indices for each bond
            bond_is_aromatic: Array of aromatic flags per bond
            molecule_boundaries: Array indicating end of each molecule
            molecule_bond_boundaries: Array indicating end of bonds for each molecule

        Returns:
            Array of aromatic ring counts for each molecule
        """
        num_molecules = len(molecule_boundaries)
        results = cp.zeros(num_molecules, dtype=cp.int32)

        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size

        kernel = self._kernels["aromatic_rings"]
        kernel(
            (grid_size,),
            (block_size,),
            (
                atom_is_aromatic,
                bond_atom1,
                bond_atom2,
                bond_is_aromatic,
                molecule_boundaries,
                molecule_bond_boundaries,
                results,
                num_molecules,
            ),
        )

        return results


class BasicDescriptorCalculator:
    """High-level interface for basic molecular descriptor calculations."""

    def __init__(self, memory_manager=None):
        """Initialize descriptor calculator.

        Args:
            memory_manager: CUDA memory manager instance
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self.memory_manager = memory_manager or get_memory_manager()
        self.kernel_manager = CUDAKernelManager(memory_manager)

    def calculate_basic_descriptors(
        self, molecular_data: Dict[str, Any]
    ) -> Dict[str, cp.ndarray]:
        """Calculate all basic descriptors for a batch of molecules.

        Args:
            molecular_data: Dictionary containing molecular data arrays

        Returns:
            Dictionary containing calculated descriptors
        """
        results = {}

        # Molecular weights
        if "atom_weights" in molecular_data and "molecule_boundaries" in molecular_data:
            results[
                "molecular_weight"
            ] = self.kernel_manager.calculate_molecular_weights(
                molecular_data["atom_weights"], molecular_data["molecule_boundaries"]
            )

        # Atom counts
        if (
            "atomic_numbers" in molecular_data
            and "molecule_boundaries" in molecular_data
        ):
            heavy_counts, total_counts = self.kernel_manager.calculate_atom_counts(
                molecular_data["atomic_numbers"], molecular_data["molecule_boundaries"]
            )
            results["heavy_atom_count"] = heavy_counts
            results["total_atom_count"] = total_counts

        # Bond counts
        if (
            "bond_types" in molecular_data
            and "molecule_bond_boundaries" in molecular_data
        ):
            bond_counts = self.kernel_manager.calculate_bond_counts(
                molecular_data["bond_types"], molecular_data["molecule_bond_boundaries"]
            )
            results.update(bond_counts)

        # TPSA
        tpsa_required = [
            "atomic_numbers",
            "hybridizations",
            "hydrogen_counts",
            "is_aromatic",
            "molecule_boundaries",
        ]
        if all(key in molecular_data for key in tpsa_required):
            results["tpsa"] = self.kernel_manager.calculate_tpsa(
                molecular_data["atomic_numbers"],
                molecular_data["hybridizations"],
                molecular_data["hydrogen_counts"],
                molecular_data["is_aromatic"],
                molecular_data["molecule_boundaries"],
            )

        # Rotatable bonds
        rot_bonds_required = [
            "bond_atom1",
            "bond_atom2",
            "bond_types",
            "bond_in_ring",
            "atomic_numbers",
            "molecule_bond_boundaries",
        ]
        if all(key in molecular_data for key in rot_bonds_required):
            results["rotatable_bonds"] = self.kernel_manager.calculate_rotatable_bonds(
                molecular_data["bond_atom1"],
                molecular_data["bond_atom2"],
                molecular_data["bond_types"],
                molecular_data["bond_in_ring"],
                molecular_data["atomic_numbers"],
                molecular_data["molecule_bond_boundaries"],
            )

        # Aromatic rings
        arom_rings_required = [
            "atom_is_aromatic",
            "bond_atom1",
            "bond_atom2",
            "bond_is_aromatic",
            "molecule_boundaries",
            "molecule_bond_boundaries",
        ]
        if all(key in molecular_data for key in arom_rings_required):
            results["aromatic_rings"] = self.kernel_manager.calculate_aromatic_rings(
                molecular_data["atom_is_aromatic"],
                molecular_data["bond_atom1"],
                molecular_data["bond_atom2"],
                molecular_data["bond_is_aromatic"],
                molecular_data["molecule_boundaries"],
                molecular_data["molecule_bond_boundaries"],
            )

        logger.info(f"Calculated {len(results)} basic descriptors")
        return results


# Global kernel manager instance
_kernel_manager: Optional[CUDAKernelManager] = None


def get_kernel_manager() -> CUDAKernelManager:
    """Get global CUDA kernel manager instance."""
    global _kernel_manager
    if _kernel_manager is None:
        _kernel_manager = CUDAKernelManager()
    return _kernel_manager

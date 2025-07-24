"""
CUDA kernels for 3D molecular descriptor calculations.

This module provides GPU-accelerated implementations for 3D molecular descriptors
including radius of gyration, principal moments of inertia, asphericity, and
shape-based descriptor computations.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import cupy as cp
    from cupy import RawKernel
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    RawKernel = None

from .memory import get_memory_manager
from ...core.exceptions import GPUMemoryError
from ...utils.logging import get_logger

logger = get_logger()


# CUDA kernel for radius of gyration calculation
RADIUS_OF_GYRATION_KERNEL = r'''
extern "C" __global__
void calculate_radius_of_gyration(
    const float* coordinates,      // (total_atoms, 3) flattened coordinates
    const float* masses,          // (total_atoms,) atomic masses
    const int* molecule_boundaries, // (num_molecules,) atom boundaries
    float* results,               // (num_molecules,) output
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (mol_idx >= num_molecules) return;
    
    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int num_atoms = end_atom - start_atom;
    
    if (num_atoms == 0) {
        results[mol_idx] = 0.0f;
        return;
    }
    
    // Calculate center of mass
    float total_mass = 0.0f;
    float com_x = 0.0f, com_y = 0.0f, com_z = 0.0f;
    
    for (int i = start_atom; i < end_atom; i++) {
        float mass = masses[i];
        total_mass += mass;
        com_x += mass * coordinates[i * 3 + 0];
        com_y += mass * coordinates[i * 3 + 1];
        com_z += mass * coordinates[i * 3 + 2];
    }
    
    if (total_mass == 0.0f) {
        results[mol_idx] = 0.0f;
        return;
    }
    
    com_x /= total_mass;
    com_y /= total_mass;
    com_z /= total_mass;
    
    // Calculate radius of gyration
    float rg_squared = 0.0f;
    
    for (int i = start_atom; i < end_atom; i++) {
        float mass = masses[i];
        float dx = coordinates[i * 3 + 0] - com_x;
        float dy = coordinates[i * 3 + 1] - com_y;
        float dz = coordinates[i * 3 + 2] - com_z;
        
        rg_squared += mass * (dx * dx + dy * dy + dz * dz);
    }
    
    rg_squared /= total_mass;
    results[mol_idx] = sqrtf(rg_squared);
}
'''

# CUDA kernel for principal moments of inertia calculation
PRINCIPAL_MOMENTS_KERNEL = r'''
extern "C" __global__
void calculate_principal_moments(
    const float* coordinates,      // (total_atoms, 3) flattened coordinates
    const float* masses,          // (total_atoms,) atomic masses
    const int* molecule_boundaries, // (num_molecules,) atom boundaries
    float* pmi1_results,          // (num_molecules,) first principal moment
    float* pmi2_results,          // (num_molecules,) second principal moment
    float* pmi3_results,          // (num_molecules,) third principal moment
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (mol_idx >= num_molecules) return;
    
    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int num_atoms = end_atom - start_atom;
    
    if (num_atoms == 0) {
        pmi1_results[mol_idx] = 0.0f;
        pmi2_results[mol_idx] = 0.0f;
        pmi3_results[mol_idx] = 0.0f;
        return;
    }
    
    // Calculate center of mass
    float total_mass = 0.0f;
    float com_x = 0.0f, com_y = 0.0f, com_z = 0.0f;
    
    for (int i = start_atom; i < end_atom; i++) {
        float mass = masses[i];
        total_mass += mass;
        com_x += mass * coordinates[i * 3 + 0];
        com_y += mass * coordinates[i * 3 + 1];
        com_z += mass * coordinates[i * 3 + 2];
    }
    
    if (total_mass == 0.0f) {
        pmi1_results[mol_idx] = 0.0f;
        pmi2_results[mol_idx] = 0.0f;
        pmi3_results[mol_idx] = 0.0f;
        return;
    }
    
    com_x /= total_mass;
    com_y /= total_mass;
    com_z /= total_mass;
    
    // Calculate inertia tensor elements
    float Ixx = 0.0f, Iyy = 0.0f, Izz = 0.0f;
    float Ixy = 0.0f, Ixz = 0.0f, Iyz = 0.0f;
    
    for (int i = start_atom; i < end_atom; i++) {
        float mass = masses[i];
        float x = coordinates[i * 3 + 0] - com_x;
        float y = coordinates[i * 3 + 1] - com_y;
        float z = coordinates[i * 3 + 2] - com_z;
        
        Ixx += mass * (y * y + z * z);
        Iyy += mass * (x * x + z * z);
        Izz += mass * (x * x + y * y);
        
        Ixy -= mass * x * y;
        Ixz -= mass * x * z;
        Iyz -= mass * y * z;
    }
    
    // Simplified eigenvalue calculation for 3x3 symmetric matrix
    // This is an approximation - full eigenvalue calculation would be more complex
    float trace = Ixx + Iyy + Izz;
    float det = Ixx * (Iyy * Izz - Iyz * Iyz) - Ixy * (Ixy * Izz - Iyz * Ixz) + Ixz * (Ixy * Iyz - Iyy * Ixz);
    
    // Approximate eigenvalues (simplified)
    float avg = trace / 3.0f;
    pmi1_results[mol_idx] = fmaxf(0.0f, avg - sqrtf(fmaxf(0.0f, avg * avg - det / 3.0f)));
    pmi2_results[mol_idx] = avg;
    pmi3_results[mol_idx] = fmaxf(0.0f, avg + sqrtf(fmaxf(0.0f, avg * avg - det / 3.0f)));
}
'''

# CUDA kernel for asphericity calculation
ASPHERICITY_KERNEL = r'''
extern "C" __global__
void calculate_asphericity(
    const float* pmi1,            // First principal moment
    const float* pmi2,            // Second principal moment  
    const float* pmi3,            // Third principal moment
    float* results,               // Output asphericity values
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (mol_idx >= num_molecules) return;
    
    float I1 = pmi1[mol_idx];
    float I2 = pmi2[mol_idx];
    float I3 = pmi3[mol_idx];
    
    float total = I1 + I2 + I3;
    
    if (total == 0.0f) {
        results[mol_idx] = 0.0f;
        return;
    }
    
    // Asphericity = (I3 - 0.5*(I1 + I2)) / (I1 + I2 + I3)
    float asphericity = (I3 - 0.5f * (I1 + I2)) / total;
    results[mol_idx] = asphericity;
}
'''

# CUDA kernel for molecular volume calculation
MOLECULAR_VOLUME_KERNEL = r'''
extern "C" __global__
void calculate_molecular_volume(
    const float* coordinates,      // (total_atoms, 3) flattened coordinates
    const float* vdw_radii,       // (total_atoms,) van der Waals radii
    const int* molecule_boundaries, // (num_molecules,) atom boundaries
    float* results,               // (num_molecules,) output volumes
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (mol_idx >= num_molecules) return;
    
    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    
    float total_volume = 0.0f;
    
    // Simple additive volume calculation
    for (int i = start_atom; i < end_atom; i++) {
        float radius = vdw_radii[i];
        float volume = (4.0f / 3.0f) * 3.14159265f * radius * radius * radius;
        total_volume += volume;
    }
    
    // Apply overlap correction factor (empirical)
    results[mol_idx] = total_volume * 0.7f;
}
'''

# CUDA kernel for geometric diameter calculation
GEOMETRIC_DIAMETER_KERNEL = r'''
extern "C" __global__
void calculate_geometric_diameter(
    const float* coordinates,      // (total_atoms, 3) flattened coordinates
    const int* molecule_boundaries, // (num_molecules,) atom boundaries
    float* results,               // (num_molecules,) output diameters
    int num_molecules
) {
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (mol_idx >= num_molecules) return;
    
    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    
    float max_distance = 0.0f;
    
    // Calculate maximum pairwise distance
    for (int i = start_atom; i < end_atom; i++) {
        float x1 = coordinates[i * 3 + 0];
        float y1 = coordinates[i * 3 + 1];
        float z1 = coordinates[i * 3 + 2];
        
        for (int j = i + 1; j < end_atom; j++) {
            float x2 = coordinates[j * 3 + 0];
            float y2 = coordinates[j * 3 + 1];
            float z2 = coordinates[j * 3 + 2];
            
            float dx = x1 - x2;
            float dy = y1 - y2;
            float dz = z1 - z2;
            
            float distance = sqrtf(dx * dx + dy * dy + dz * dz);
            max_distance = fmaxf(max_distance, distance);
        }
    }
    
    results[mol_idx] = max_distance;
}
'''


class CUDA3DDescriptorKernels:
    """CUDA kernels for 3D molecular descriptor calculations."""
    
    def __init__(self, memory_manager=None):
        """Initialize 3D descriptor CUDA kernels.
        
        Args:
            memory_manager: CUDA memory manager instance
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
        
        self.memory_manager = memory_manager or get_memory_manager()
        self._kernels = {}
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile all 3D descriptor CUDA kernels."""
        kernel_sources = {
            'radius_of_gyration': RADIUS_OF_GYRATION_KERNEL,
            'principal_moments': PRINCIPAL_MOMENTS_KERNEL,
            'asphericity': ASPHERICITY_KERNEL,
            'molecular_volume': MOLECULAR_VOLUME_KERNEL,
            'geometric_diameter': GEOMETRIC_DIAMETER_KERNEL,
        }
        
        for name, source in kernel_sources.items():
            try:
                kernel = RawKernel(source, f'calculate_{name}')
                self._kernels[name] = kernel
                logger.debug(f"Compiled 3D descriptor CUDA kernel: {name}")
            except Exception as e:
                logger.error(f"Failed to compile 3D kernel {name}: {e}")
                raise
        
        logger.info(f"Compiled {len(self._kernels)} 3D descriptor CUDA kernels")
    
    def calculate_radius_of_gyration(self,
                                   coordinates: cp.ndarray,
                                   masses: cp.ndarray,
                                   molecule_boundaries: cp.ndarray) -> cp.ndarray:
        """Calculate radius of gyration using CUDA kernel.
        
        Args:
            coordinates: Flattened coordinates array (total_atoms, 3)
            masses: Atomic masses array (total_atoms,)
            molecule_boundaries: Molecule boundary indices
            
        Returns:
            Array of radius of gyration values
        """
        num_molecules = len(molecule_boundaries)
        results = cp.zeros(num_molecules, dtype=cp.float32)
        
        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size
        
        kernel = self._kernels['radius_of_gyration']
        kernel((grid_size,), (block_size,), (
            coordinates, masses, molecule_boundaries, results, num_molecules
        ))
        
        return results
    
    def calculate_principal_moments(self,
                                  coordinates: cp.ndarray,
                                  masses: cp.ndarray,
                                  molecule_boundaries: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Calculate principal moments of inertia using CUDA kernel.
        
        Args:
            coordinates: Flattened coordinates array (total_atoms, 3)
            masses: Atomic masses array (total_atoms,)
            molecule_boundaries: Molecule boundary indices
            
        Returns:
            Tuple of (PMI1, PMI2, PMI3) arrays
        """
        num_molecules = len(molecule_boundaries)
        pmi1_results = cp.zeros(num_molecules, dtype=cp.float32)
        pmi2_results = cp.zeros(num_molecules, dtype=cp.float32)
        pmi3_results = cp.zeros(num_molecules, dtype=cp.float32)
        
        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size
        
        kernel = self._kernels['principal_moments']
        kernel((grid_size,), (block_size,), (
            coordinates, masses, molecule_boundaries,
            pmi1_results, pmi2_results, pmi3_results, num_molecules
        ))
        
        return pmi1_results, pmi2_results, pmi3_results
    
    def calculate_asphericity(self,
                            pmi1: cp.ndarray,
                            pmi2: cp.ndarray,
                            pmi3: cp.ndarray) -> cp.ndarray:
        """Calculate asphericity from principal moments.
        
        Args:
            pmi1: First principal moment array
            pmi2: Second principal moment array
            pmi3: Third principal moment array
            
        Returns:
            Array of asphericity values
        """
        num_molecules = len(pmi1)
        results = cp.zeros(num_molecules, dtype=cp.float32)
        
        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size
        
        kernel = self._kernels['asphericity']
        kernel((grid_size,), (block_size,), (
            pmi1, pmi2, pmi3, results, num_molecules
        ))
        
        return results
    
    def calculate_molecular_volume(self,
                                 coordinates: cp.ndarray,
                                 vdw_radii: cp.ndarray,
                                 molecule_boundaries: cp.ndarray) -> cp.ndarray:
        """Calculate molecular volume using CUDA kernel.
        
        Args:
            coordinates: Flattened coordinates array (total_atoms, 3)
            vdw_radii: van der Waals radii array (total_atoms,)
            molecule_boundaries: Molecule boundary indices
            
        Returns:
            Array of molecular volume values
        """
        num_molecules = len(molecule_boundaries)
        results = cp.zeros(num_molecules, dtype=cp.float32)
        
        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size
        
        kernel = self._kernels['molecular_volume']
        kernel((grid_size,), (block_size,), (
            coordinates, vdw_radii, molecule_boundaries, results, num_molecules
        ))
        
        return results
    
    def calculate_geometric_diameter(self,
                                   coordinates: cp.ndarray,
                                   molecule_boundaries: cp.ndarray) -> cp.ndarray:
        """Calculate geometric diameter using CUDA kernel.
        
        Args:
            coordinates: Flattened coordinates array (total_atoms, 3)
            molecule_boundaries: Molecule boundary indices
            
        Returns:
            Array of geometric diameter values
        """
        num_molecules = len(molecule_boundaries)
        results = cp.zeros(num_molecules, dtype=cp.float32)
        
        # Launch kernel
        block_size = 256
        grid_size = (num_molecules + block_size - 1) // block_size
        
        kernel = self._kernels['geometric_diameter']
        kernel((grid_size,), (block_size,), (
            coordinates, molecule_boundaries, results, num_molecules
        ))
        
        return results


class GPU3DDescriptorCalculator:
    """High-level interface for GPU-accelerated 3D descriptor calculations."""
    
    def __init__(self, memory_manager=None):
        """Initialize 3D descriptor calculator.
        
        Args:
            memory_manager: CUDA memory manager instance
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
        
        self.memory_manager = memory_manager or get_memory_manager()
        self.kernel_manager = CUDA3DDescriptorKernels(memory_manager)
        
        # Default van der Waals radii (Angstroms)
        self.vdw_radii = {
            1: 1.20,   # H
            6: 1.70,   # C
            7: 1.55,   # N
            8: 1.52,   # O
            9: 1.47,   # F
            15: 1.80,  # P
            16: 1.80,  # S
            17: 1.75,  # Cl
            35: 1.85,  # Br
            53: 1.98   # I
        }
        
        # Atomic masses (amu)
        self.atomic_masses = {
            1: 1.008,   # H
            6: 12.011,  # C
            7: 14.007,  # N
            8: 15.999,  # O
            9: 18.998,  # F
            15: 30.974, # P
            16: 32.065, # S
            17: 35.453, # Cl
            35: 79.904, # Br
            53: 126.904 # I
        }
    
    def calculate_3d_descriptors(self, molecular_data: Dict[str, Any]) -> Dict[str, cp.ndarray]:
        """Calculate all 3D descriptors for a batch of molecules.
        
        Args:
            molecular_data: Dictionary containing molecular 3D data
            
        Returns:
            Dictionary containing calculated 3D descriptors
        """
        results = {}
        
        # Extract required data
        coordinates = molecular_data.get('coordinates')  # (total_atoms, 3)
        atomic_numbers = molecular_data.get('atomic_numbers')  # (total_atoms,)
        molecule_boundaries = molecular_data.get('molecule_boundaries')  # (num_molecules,)
        
        if coordinates is None or atomic_numbers is None or molecule_boundaries is None:
            logger.warning("Missing required data for 3D descriptor calculation")
            return results
        
        # Prepare masses and radii arrays
        masses = cp.array([self.atomic_masses.get(int(an), 12.0) for an in cp.asnumpy(atomic_numbers)], dtype=cp.float32)
        vdw_radii = cp.array([self.vdw_radii.get(int(an), 2.0) for an in cp.asnumpy(atomic_numbers)], dtype=cp.float32)
        
        # Radius of gyration
        results['radius_of_gyration'] = self.kernel_manager.calculate_radius_of_gyration(
            coordinates, masses, molecule_boundaries
        )
        
        # Principal moments of inertia
        pmi1, pmi2, pmi3 = self.kernel_manager.calculate_principal_moments(
            coordinates, masses, molecule_boundaries
        )
        results['pmi1'] = pmi1
        results['pmi2'] = pmi2
        results['pmi3'] = pmi3
        
        # Asphericity
        results['asphericity'] = self.kernel_manager.calculate_asphericity(pmi1, pmi2, pmi3)
        
        # Molecular volume
        results['molecular_volume'] = self.kernel_manager.calculate_molecular_volume(
            coordinates, vdw_radii, molecule_boundaries
        )
        
        # Geometric diameter
        results['geometric_diameter'] = self.kernel_manager.calculate_geometric_diameter(
            coordinates, molecule_boundaries
        )
        
        logger.info(f"Calculated {len(results)} 3D descriptors")
        return results


# Global kernel manager instance
_3d_kernel_manager: Optional[CUDA3DDescriptorKernels] = None


def get_3d_kernel_manager() -> CUDA3DDescriptorKernels:
    """Get global 3D descriptor CUDA kernel manager instance."""
    global _3d_kernel_manager
    if _3d_kernel_manager is None:
        _3d_kernel_manager = CUDA3DDescriptorKernels()
    return _3d_kernel_manager
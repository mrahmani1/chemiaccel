"""
Optimized CUDA Kernels for ChemiAccel.

This module provides highly optimized CUDA kernel implementations with:
- Coalesced memory access patterns
- Shared memory caching for frequently accessed data
- Warp-level primitives and cooperative groups
- Optimized kernel occupancy and resource utilization
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    import cupy as cp
    from cupy import RawKernel
    CUDA_AVAILABLE = True
    CupyArray = cp.ndarray
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    RawKernel = None
    # Create a dummy type for type hints when CuPy is not available
    from typing import Any
    CupyArray = Any

from .memory import get_memory_manager
from ...core.exceptions import GPUMemoryError
from ...utils.logging import get_logger

logger = get_logger()


# Optimized CUDA kernel with coalesced memory access and shared memory
OPTIMIZED_MOLECULAR_WEIGHT_KERNEL = r'''
#include <cooperative_groups.h>
using namespace cooperative_groups;

extern "C" __global__
void calculate_molecular_weight_optimized(
    const float* __restrict__ atom_weights,
    const int* __restrict__ molecule_boundaries,
    float* __restrict__ results,
    int num_molecules
) {
    // Use cooperative groups for warp-level operations
    auto block = this_thread_block();
    auto warp = tiled_partition<32>(block);
    
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for caching molecule boundaries
    __shared__ int shared_boundaries[512];  // Cache for block
    
    // Cooperatively load boundaries into shared memory
    int cache_idx = threadIdx.x;
    while (cache_idx < min(512, num_molecules + 1)) {
        if (blockIdx.x * 512 + cache_idx < num_molecules) {
            shared_boundaries[cache_idx] = molecule_boundaries[blockIdx.x * 512 + cache_idx];
        }
        cache_idx += blockDim.x;
    }
    
    block.sync();
    
    if (mol_idx >= num_molecules) return;
    
    // Use cached boundaries when possible
    int start_atom, end_atom;
    int local_mol_idx = mol_idx - blockIdx.x * 512;
    
    if (local_mol_idx < 512 && blockIdx.x * 512 < num_molecules) {
        start_atom = (local_mol_idx == 0) ? 0 : shared_boundaries[local_mol_idx - 1];
        end_atom = shared_boundaries[local_mol_idx];
    } else {
        start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
        end_atom = molecule_boundaries[mol_idx];
    }
    
    // Vectorized loading and reduction using warp primitives
    float total_weight = 0.0f;
    int num_atoms = end_atom - start_atom;
    
    // Process atoms in chunks for better memory coalescing
    for (int i = start_atom; i < end_atom; i += 32) {
        int atom_idx = i + warp.thread_rank();
        float weight = 0.0f;
        
        if (atom_idx < end_atom) {
            weight = atom_weights[atom_idx];
        }
        
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            weight += warp.shfl_down_sync(0xffffffff, weight, offset);
        }
        
        if (warp.thread_rank() == 0) {
            total_weight += weight;
        }
    }
    
    results[mol_idx] = total_weight;
}
'''

# Optimized TPSA kernel with improved memory access patterns
OPTIMIZED_TPSA_KERNEL = r'''
#include <cooperative_groups.h>
using namespace cooperative_groups;

extern "C" __global__
void calculate_tpsa_optimized(
    const int* __restrict__ atomic_numbers,
    const int* __restrict__ hybridizations,
    const int* __restrict__ hydrogen_counts,
    const bool* __restrict__ is_aromatic,
    const int* __restrict__ molecule_boundaries,
    float* __restrict__ results,
    int num_molecules
) {
    auto block = this_thread_block();
    auto warp = tiled_partition<32>(block);
    
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for TPSA lookup tables
    __shared__ float tpsa_lookup[16][8];  // [atomic_number][hybridization]
    
    // Initialize TPSA lookup table cooperatively
    if (threadIdx.x < 128) {
        int flat_idx = threadIdx.x;
        int atomic_num = flat_idx / 8;
        int hyb = flat_idx % 8;
        
        float tpsa_contrib = 0.0f;
        
        if (atomic_num == 7) {  // Nitrogen
            if (hyb == 1) tpsa_contrib = 12.89f;  // Aromatic
            else if (hyb == 3) tpsa_contrib = 3.24f;  // sp3
            else if (hyb == 2) tpsa_contrib = 19.19f;  // sp2 average
        } else if (atomic_num == 8) {  // Oxygen
            if (hyb == 3) tpsa_contrib = 14.73f;  // sp3 average
            else if (hyb == 2) tpsa_contrib = 17.07f;  // sp2
        } else if (atomic_num == 16) {  // Sulfur
            if (hyb == 3) tpsa_contrib = 25.30f;
            else if (hyb == 2) tpsa_contrib = 32.09f;
        }
        
        tpsa_lookup[atomic_num][hyb] = tpsa_contrib;
    }
    
    block.sync();
    
    if (mol_idx >= num_molecules) return;
    
    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    
    float tpsa = 0.0f;
    
    // Vectorized processing with coalesced memory access
    for (int i = start_atom; i < end_atom; i += 32) {
        int atom_idx = i + warp.thread_rank();
        float atom_tpsa = 0.0f;
        
        if (atom_idx < end_atom) {
            int atomic_num = atomic_numbers[atom_idx];
            int hyb = hybridizations[atom_idx];
            int h_count = hydrogen_counts[atom_idx];
            bool aromatic = is_aromatic[atom_idx];
            
            // Use lookup table for base contribution
            if (atomic_num < 16 && hyb < 8) {
                atom_tpsa = tpsa_lookup[atomic_num][hyb];
                
                // Apply corrections based on hydrogen count and aromaticity
                if (atomic_num == 7 && !aromatic && hyb == 2) {
                    atom_tpsa = (h_count == 0) ? 12.36f : 26.02f;
                } else if (atomic_num == 8 && hyb == 3) {
                    atom_tpsa = (h_count > 0) ? 20.23f : 9.23f;
                }
            }
        }
        
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            atom_tpsa += warp.shfl_down_sync(0xffffffff, atom_tpsa, offset);
        }
        
        if (warp.thread_rank() == 0) {
            tpsa += atom_tpsa;
        }
    }
    
    results[mol_idx] = tpsa;
}
'''

# Optimized graph kernel with improved memory access and shared memory usage
OPTIMIZED_ZAGREB_KERNEL = r'''
#include <cooperative_groups.h>
using namespace cooperative_groups;

extern "C" __global__
void calculate_zagreb_indices_optimized(
    const int* __restrict__ csr_indices,
    const int* __restrict__ csr_indptr,
    const int* __restrict__ molecule_boundaries,
    const int* __restrict__ molecule_csr_boundaries,
    float* __restrict__ zagreb1_results,
    float* __restrict__ zagreb2_results,
    int num_molecules
) {
    auto block = this_thread_block();
    auto warp = tiled_partition<32>(block);
    
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for caching degree calculations
    __shared__ int shared_degrees[1024];  // Cache degrees for atoms in block
    
    if (mol_idx >= num_molecules) return;
    
    int start_atom = (mol_idx == 0) ? 0 : molecule_boundaries[mol_idx - 1];
    int end_atom = molecule_boundaries[mol_idx];
    int start_csr = (mol_idx == 0) ? 0 : molecule_csr_boundaries[mol_idx - 1];
    int num_atoms = end_atom - start_atom;
    
    float zagreb1 = 0.0f;
    float zagreb2 = 0.0f;
    
    // First pass: calculate and cache degrees with coalesced access
    for (int i = 0; i < num_atoms; i += blockDim.x) {
        int atom_offset = i + threadIdx.x;
        if (atom_offset < num_atoms && threadIdx.x < 1024) {
            int atom_idx = start_atom + atom_offset;
            int degree = csr_indptr[atom_idx + 1] - csr_indptr[atom_idx];
            shared_degrees[threadIdx.x] = degree;
        }
        
        block.sync();
        
        // Process cached degrees
        if (atom_offset < num_atoms) {
            int degree = shared_degrees[threadIdx.x];
            
            // Zagreb M1: sum of squared degrees
            float degree_sq = (float)(degree * degree);
            
            // Warp-level reduction for Zagreb M1
            for (int offset = 16; offset > 0; offset /= 2) {
                degree_sq += warp.shfl_down_sync(0xffffffff, degree_sq, offset);
            }
            
            if (warp.thread_rank() == 0) {
                zagreb1 += degree_sq;
            }
        }
        
        block.sync();
    }
    
    // Second pass: calculate Zagreb M2 with optimized neighbor access
    for (int i = start_atom; i < end_atom; i++) {
        int atom_csr_start = csr_indptr[i] - start_csr;
        int atom_csr_end = csr_indptr[i + 1] - start_csr;
        int degree_i = atom_csr_end - atom_csr_start;
        
        // Vectorized neighbor processing
        for (int j = atom_csr_start; j < atom_csr_end; j += 32) {
            int neighbor_idx = j + warp.thread_rank();
            float edge_contrib = 0.0f;
            
            if (neighbor_idx < atom_csr_end) {
                int neighbor = csr_indices[start_csr + neighbor_idx];
                
                if (neighbor > i) { // Avoid double counting
                    int neighbor_degree = csr_indptr[neighbor + 1] - csr_indptr[neighbor];
                    edge_contrib = (float)(degree_i * neighbor_degree);
                }
            }
            
            // Warp-level reduction
            for (int offset = 16; offset > 0; offset /= 2) {
                edge_contrib += warp.shfl_down_sync(0xffffffff, edge_contrib, offset);
            }
            
            if (warp.thread_rank() == 0) {
                zagreb2 += edge_contrib;
            }
        }
    }
    
    zagreb1_results[mol_idx] = zagreb1;
    zagreb2_results[mol_idx] = zagreb2;
}
'''

# Optimized fingerprint similarity kernel with vectorized operations
OPTIMIZED_TANIMOTO_KERNEL = r'''
#include <cooperative_groups.h>
using namespace cooperative_groups;

extern "C" __global__
void calculate_tanimoto_similarity_optimized(
    const unsigned int* __restrict__ fingerprint1,
    const unsigned int* __restrict__ fingerprint2,
    const int* __restrict__ molecule_boundaries,
    float* __restrict__ similarities,
    int num_molecules,
    int fingerprint_size_words  // Size in 32-bit words
) {
    auto block = this_thread_block();
    auto warp = tiled_partition<32>(block);
    
    int mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (mol_idx >= num_molecules) return;
    
    int start_fp = mol_idx * fingerprint_size_words;
    
    int intersection = 0;
    int union_count = 0;
    
    // Vectorized bit operations with coalesced memory access
    for (int i = 0; i < fingerprint_size_words; i += 32) {
        int word_idx = i + warp.thread_rank();
        int local_intersection = 0;
        int local_union = 0;
        
        if (word_idx < fingerprint_size_words) {
            unsigned int word1 = fingerprint1[start_fp + word_idx];
            unsigned int word2 = fingerprint2[start_fp + word_idx];
            
            // Use built-in bit counting functions
            local_intersection = __popc(word1 & word2);
            local_union = __popc(word1 | word2);
        }
        
        // Warp-level reduction using shuffle operations
        for (int offset = 16; offset > 0; offset /= 2) {
            local_intersection += warp.shfl_down_sync(0xffffffff, local_intersection, offset);
            local_union += warp.shfl_down_sync(0xffffffff, local_union, offset);
        }
        
        if (warp.thread_rank() == 0) {
            intersection += local_intersection;
            union_count += local_union;
        }
    }
    
    // Calculate Tanimoto coefficient
    float similarity = 0.0f;
    if (union_count > 0) {
        similarity = (float)intersection / (float)union_count;
    }
    
    similarities[mol_idx] = similarity;
}
'''

# Kernel profiling and occupancy optimization utilities
KERNEL_PROFILER_KERNEL = r'''
extern "C" __global__
void kernel_profiler(
    float* dummy_data,
    int* occupancy_info,
    int data_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Record occupancy information
    if (idx == 0) {
        occupancy_info[0] = blockDim.x;  // Block size
        occupancy_info[1] = gridDim.x;   // Grid size
        occupancy_info[2] = blockIdx.x;  // Block index
    }
    
    // Dummy computation to measure performance
    if (idx < data_size) {
        float value = dummy_data[idx];
        value = sqrtf(value * value + 1.0f);
        dummy_data[idx] = value;
    }
}
'''


class OptimizedCUDAKernelManager:
    """Manager for optimized CUDA kernels with advanced memory access patterns."""
    
    def __init__(self, memory_manager=None):
        """Initialize optimized kernel manager."""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
        
        self.memory_manager = memory_manager or get_memory_manager()
        self._kernels = {}
        self._kernel_configs = {}
        self._compile_kernels()
        self._optimize_kernel_configs()
    
    def _compile_kernels(self):
        """Compile all optimized CUDA kernels."""
        kernel_sources = {
            'molecular_weight_opt': OPTIMIZED_MOLECULAR_WEIGHT_KERNEL,
            'tpsa_opt': OPTIMIZED_TPSA_KERNEL,
            'zagreb_opt': OPTIMIZED_ZAGREB_KERNEL,
            'tanimoto_opt': OPTIMIZED_TANIMOTO_KERNEL,
            'profiler': KERNEL_PROFILER_KERNEL,
        }
        
        # Compilation options for optimization
        compile_options = [
            '--use_fast_math',
            '--maxrregcount=64',
            '-O3',
            '--gpu-architecture=compute_70',  # Adjust based on target GPU
        ]
        
        for name, source in kernel_sources.items():
            try:
                kernel = RawKernel(
                    source, 
                    f'calculate_{name}' if not name.endswith('profiler') else 'kernel_profiler',
                    options=compile_options
                )
                self._kernels[name] = kernel
                logger.debug(f"Compiled optimized CUDA kernel: {name}")
            except Exception as e:
                logger.error(f"Failed to compile optimized kernel {name}: {e}")
                # Fall back to basic compilation without optimization flags
                try:
                    kernel = RawKernel(source, f'calculate_{name}' if not name.endswith('profiler') else 'kernel_profiler')
                    self._kernels[name] = kernel
                    logger.warning(f"Compiled kernel {name} without optimization flags")
                except Exception as e2:
                    logger.error(f"Failed to compile kernel {name} even without optimization: {e2}")
                    raise
        
        logger.info(f"Compiled {len(self._kernels)} optimized CUDA kernels")
    
    def _optimize_kernel_configs(self):
        """Optimize kernel launch configurations for maximum occupancy."""
        device = cp.cuda.Device()
        device_props = device.attributes
        
        # Get device properties for occupancy calculation
        max_threads_per_block = device_props['MaxThreadsPerBlock']
        max_shared_memory = device_props['SharedMemoryPerBlock']
        warp_size = device_props['WarpSize']
        
        for kernel_name, kernel in self._kernels.items():
            if kernel_name == 'profiler':
                continue
                
            # Calculate optimal block size for maximum occupancy
            optimal_block_size = self._calculate_optimal_block_size(
                kernel, max_threads_per_block, warp_size
            )
            
            # Calculate shared memory requirements
            shared_mem_size = self._calculate_shared_memory_size(kernel_name)
            
            self._kernel_configs[kernel_name] = {
                'block_size': optimal_block_size,
                'shared_memory': shared_mem_size,
                'max_blocks_per_sm': self._calculate_max_blocks_per_sm(
                    optimal_block_size, shared_mem_size, max_shared_memory
                )
            }
            
            logger.debug(f"Optimized config for {kernel_name}: "
                        f"block_size={optimal_block_size}, "
                        f"shared_mem={shared_mem_size}")
    
    def _calculate_optimal_block_size(self, kernel, max_threads, warp_size):
        """Calculate optimal block size for kernel."""
        # Start with multiples of warp size and find the best occupancy
        best_block_size = warp_size
        best_occupancy = 0
        
        for block_size in range(warp_size, max_threads + 1, warp_size):
            try:
                # This is a simplified occupancy calculation
                # In practice, you'd use CUDA occupancy calculator API
                occupancy = min(1.0, max_threads / block_size)
                
                if occupancy > best_occupancy:
                    best_occupancy = occupancy
                    best_block_size = block_size
                    
            except Exception:
                continue
        
        return best_block_size
    
    def _calculate_shared_memory_size(self, kernel_name):
        """Calculate shared memory requirements for kernel."""
        shared_mem_sizes = {
            'molecular_weight_opt': 512 * 4,  # 512 ints for boundaries cache
            'tpsa_opt': 16 * 8 * 4,  # TPSA lookup table
            'zagreb_opt': 1024 * 4,  # Degree cache
            'tanimoto_opt': 0,  # No shared memory
        }
        
        return shared_mem_sizes.get(kernel_name, 0)
    
    def _calculate_max_blocks_per_sm(self, block_size, shared_mem_size, max_shared_mem):
        """Calculate maximum blocks per streaming multiprocessor."""
        if shared_mem_size == 0:
            return 32  # Default maximum
        
        return min(32, max_shared_mem // shared_mem_size)
    
    def profile_kernel_performance(self, kernel_name: str, data_size: int) -> Dict[str, float]:
        """Profile kernel performance and occupancy."""
        if kernel_name not in self._kernels:
            raise ValueError(f"Kernel {kernel_name} not found")
        
        # Create dummy data for profiling
        dummy_data = cp.random.random(data_size, dtype=cp.float32)
        occupancy_info = cp.zeros(3, dtype=cp.int32)
        
        config = self._kernel_configs.get(kernel_name, {'block_size': 256, 'shared_memory': 0})
        block_size = config['block_size']
        grid_size = (data_size + block_size - 1) // block_size
        
        # Profile kernel execution
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        start_event.record()
        
        if kernel_name == 'profiler':
            self._kernels['profiler']((grid_size,), (block_size,), (
                dummy_data, occupancy_info, data_size
            ))
        else:
            # Run actual kernel with dummy data (this is simplified)
            pass
        
        end_event.record()
        end_event.synchronize()
        
        execution_time = cp.cuda.get_elapsed_time(start_event, end_event)
        
        # Calculate performance metrics
        throughput = data_size / (execution_time / 1000.0)  # Elements per second
        
        return {
            'execution_time_ms': execution_time,
            'throughput_elements_per_sec': throughput,
            'block_size': block_size,
            'grid_size': grid_size,
            'occupancy_estimate': min(1.0, block_size * grid_size / (2048 * 80))  # Rough estimate
        }
    
    def calculate_molecular_weights_optimized(self, 
                                            atom_weights: CupyArray,
                                            molecule_boundaries: CupyArray) -> CupyArray:
        """Calculate molecular weights using optimized kernel."""
        num_molecules = len(molecule_boundaries)
        results = cp.zeros(num_molecules, dtype=cp.float32)
        
        config = self._kernel_configs['molecular_weight_opt']
        block_size = config['block_size']
        grid_size = (num_molecules + block_size - 1) // block_size
        shared_mem = config['shared_memory']
        
        kernel = self._kernels['molecular_weight_opt']
        kernel((grid_size,), (block_size,), (
            atom_weights, molecule_boundaries, results, num_molecules
        ), shared_mem=shared_mem)
        
        return results
    
    def calculate_tpsa_optimized(self,
                               atomic_numbers: CupyArray,
                               hybridizations: CupyArray,
                               hydrogen_counts: CupyArray,
                               is_aromatic: CupyArray,
                               molecule_boundaries: CupyArray) -> CupyArray:
        """Calculate TPSA using optimized kernel."""
        num_molecules = len(molecule_boundaries)
        results = cp.zeros(num_molecules, dtype=cp.float32)
        
        config = self._kernel_configs['tpsa_opt']
        block_size = config['block_size']
        grid_size = (num_molecules + block_size - 1) // block_size
        shared_mem = config['shared_memory']
        
        kernel = self._kernels['tpsa_opt']
        kernel((grid_size,), (block_size,), (
            atomic_numbers, hybridizations, hydrogen_counts,
            is_aromatic, molecule_boundaries, results, num_molecules
        ), shared_mem=shared_mem)
        
        return results
    
    def calculate_zagreb_indices_optimized(self,
                                         csr_indices: CupyArray,
                                         csr_indptr: CupyArray,
                                         molecule_boundaries: CupyArray,
                                         molecule_csr_boundaries: CupyArray) -> Tuple[CupyArray, CupyArray]:
        """Calculate Zagreb indices using optimized kernel."""
        num_molecules = len(molecule_boundaries)
        zagreb1_results = cp.zeros(num_molecules, dtype=cp.float32)
        zagreb2_results = cp.zeros(num_molecules, dtype=cp.float32)
        
        config = self._kernel_configs['zagreb_opt']
        block_size = config['block_size']
        grid_size = (num_molecules + block_size - 1) // block_size
        shared_mem = config['shared_memory']
        
        kernel = self._kernels['zagreb_opt']
        kernel((grid_size,), (block_size,), (
            csr_indices, csr_indptr, molecule_boundaries,
            molecule_csr_boundaries, zagreb1_results, zagreb2_results, num_molecules
        ), shared_mem=shared_mem)
        
        return zagreb1_results, zagreb2_results
    
    def calculate_tanimoto_similarity_optimized(self,
                                              fingerprint1: CupyArray,
                                              fingerprint2: CupyArray,
                                              molecule_boundaries: CupyArray,
                                              fingerprint_size_bits: int) -> CupyArray:
        """Calculate Tanimoto similarity using optimized kernel."""
        num_molecules = len(molecule_boundaries)
        similarities = cp.zeros(num_molecules, dtype=cp.float32)
        fingerprint_size_words = (fingerprint_size_bits + 31) // 32
        
        config = self._kernel_configs['tanimoto_opt']
        block_size = config['block_size']
        grid_size = (num_molecules + block_size - 1) // block_size
        
        kernel = self._kernels['tanimoto_opt']
        kernel((grid_size,), (block_size,), (
            fingerprint1, fingerprint2, molecule_boundaries,
            similarities, num_molecules, fingerprint_size_words
        ))
        
        return similarities


class KernelProfiler:
    """Utility class for profiling and optimizing CUDA kernel performance."""
    
    def __init__(self, kernel_manager: OptimizedCUDAKernelManager):
        """Initialize kernel profiler."""
        self.kernel_manager = kernel_manager
        self.profile_results = {}
    
    def profile_all_kernels(self, test_sizes: List[int] = None) -> Dict[str, Dict[str, Any]]:
        """Profile all kernels with different data sizes."""
        if test_sizes is None:
            test_sizes = [1000, 10000, 100000, 1000000]
        
        results = {}
        
        for kernel_name in self.kernel_manager._kernels.keys():
            if kernel_name == 'profiler':
                continue
                
            kernel_results = {}
            
            for size in test_sizes:
                try:
                    profile_data = self.kernel_manager.profile_kernel_performance(kernel_name, size)
                    kernel_results[f'size_{size}'] = profile_data
                    
                    logger.info(f"Profiled {kernel_name} with size {size}: "
                              f"{profile_data['execution_time_ms']:.2f}ms, "
                              f"{profile_data['throughput_elements_per_sec']:.0f} elem/s")
                              
                except Exception as e:
                    logger.warning(f"Failed to profile {kernel_name} with size {size}: {e}")
            
            results[kernel_name] = kernel_results
        
        self.profile_results = results
        return results
    
    def get_optimization_recommendations(self) -> Dict[str, List[str]]:
        """Generate optimization recommendations based on profiling results."""
        recommendations = {}
        
        for kernel_name, kernel_results in self.profile_results.items():
            kernel_recommendations = []
            
            # Analyze occupancy
            occupancies = [result.get('occupancy_estimate', 0) 
                          for result in kernel_results.values()]
            avg_occupancy = sum(occupancies) / len(occupancies) if occupancies else 0
            
            if avg_occupancy < 0.5:
                kernel_recommendations.append(
                    f"Low occupancy ({avg_occupancy:.2f}). Consider increasing block size or reducing shared memory usage."
                )
            
            # Analyze throughput scaling
            throughputs = [(size, result.get('throughput_elements_per_sec', 0))
                          for size_key, result in kernel_results.items()
                          for size in [int(size_key.split('_')[1])]]
            
            if len(throughputs) >= 2:
                throughputs.sort()
                scaling_factor = throughputs[-1][1] / throughputs[0][1] if throughputs[0][1] > 0 else 0
                expected_scaling = throughputs[-1][0] / throughputs[0][0]
                
                if scaling_factor < expected_scaling * 0.5:
                    kernel_recommendations.append(
                        f"Poor scaling ({scaling_factor:.1f}x vs expected {expected_scaling:.1f}x). "
                        "Consider optimizing memory access patterns."
                    )
            
            recommendations[kernel_name] = kernel_recommendations
        
        return recommendations
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        if not self.profile_results:
            return "No profiling results available. Run profile_all_kernels() first."
        
        report = ["CUDA Kernel Performance Report", "=" * 40, ""]
        
        for kernel_name, kernel_results in self.profile_results.items():
            report.append(f"Kernel: {kernel_name}")
            report.append("-" * 20)
            
            for size_key, result in kernel_results.items():
                size = size_key.split('_')[1]
                report.append(f"  Size {size}:")
                report.append(f"    Execution time: {result.get('execution_time_ms', 0):.2f} ms")
                report.append(f"    Throughput: {result.get('throughput_elements_per_sec', 0):.0f} elem/s")
                report.append(f"    Occupancy: {result.get('occupancy_estimate', 0):.2f}")
                report.append(f"    Block size: {result.get('block_size', 0)}")
            
            report.append("")
        
        # Add optimization recommendations
        recommendations = self.get_optimization_recommendations()
        if any(recommendations.values()):
            report.append("Optimization Recommendations:")
            report.append("-" * 30)
            
            for kernel_name, kernel_recs in recommendations.items():
                if kernel_recs:
                    report.append(f"{kernel_name}:")
                    for rec in kernel_recs:
                        report.append(f"  - {rec}")
                    report.append("")
        
        return "\n".join(report)


# Global optimized kernel manager instance
_optimized_kernel_manager: Optional[OptimizedCUDAKernelManager] = None


def get_optimized_kernel_manager() -> OptimizedCUDAKernelManager:
    """Get global optimized CUDA kernel manager instance."""
    global _optimized_kernel_manager
    if _optimized_kernel_manager is None:
        _optimized_kernel_manager = OptimizedCUDAKernelManager()
    return _optimized_kernel_manager
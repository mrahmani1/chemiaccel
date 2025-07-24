"""
CuPy Array Operations for ChemiAccel.

This module provides direct cupy array operations for numerical computations,
optimized memory layouts, and interoperability utilities for existing cupy workflows.
"""

from typing import List, Optional, Dict, Any, Union, Tuple, TYPE_CHECKING
import logging

try:
    import cupy as cp
    import cupyx
    from cupyx.scipy import sparse as cp_sparse
    CUPY_AVAILABLE = True
except ImportError as e:
    cp = None
    cupyx = None
    cp_sparse = None
    CUPY_AVAILABLE = False
    _import_error = e

# Type checking imports
if TYPE_CHECKING:
    import cupy as cp

import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse

from ...core.exceptions import ChemiAccelError, BackendUnavailableError
from ...utils.logging import get_logger

logger = get_logger()


class CupyOperationsError(ChemiAccelError):
    """Raised when cupy operations fail."""
    pass


class CupyArrayProcessor:
    """Handles cupy array operations for molecular descriptor calculations."""
    
    def __init__(self, device_id: Optional[int] = None):
        """Initialize cupy array processor.
        
        Args:
            device_id: CUDA device ID to use (None for default)
            
        Raises:
            BackendUnavailableError: If cupy is not available
        """
        if not CUPY_AVAILABLE:
            raise BackendUnavailableError(
                "cupy",
                f"CuPy not available: {_import_error}. "
                "Install with: pip install cupy-cuda11x"
            )
        
        self.device_id = device_id
        self._initialize_device()
        
        logger.info(f"CupyArrayProcessor initialized on device {self.device_id}")
    
    def _initialize_device(self) -> None:
        """Initialize CUDA device for cupy operations."""
        try:
            if self.device_id is not None:
                cp.cuda.Device(self.device_id).use()
            
            # Get current device info
            device = cp.cuda.Device()
            self.device_id = device.id
            
            logger.debug(f"Using CUDA device {self.device_id}: {device.attributes}")
            
        except Exception as e:
            raise CupyOperationsError(f"Failed to initialize CUDA device: {e}")
    
    def numpy_to_cupy(self, array: np.ndarray, copy: bool = False) -> "cp.ndarray":
        """Convert NumPy array to CuPy array with optimized memory layout.
        
        Args:
            array: NumPy array to convert
            copy: Whether to force a copy (default: False for zero-copy when possible)
            
        Returns:
            CuPy array with optimized layout
            
        Raises:
            CupyOperationsError: If conversion fails
        """
        try:
            # Convert to cupy array
            if copy:
                cupy_array = cp.array(array, copy=True)
            else:
                cupy_array = cp.asarray(array)
            
            # Optimize memory layout for GPU operations
            if not cupy_array.flags.c_contiguous:
                cupy_array = cp.ascontiguousarray(cupy_array)
            
            logger.debug(f"Converted array shape {array.shape} to CuPy")
            return cupy_array
            
        except Exception as e:
            raise CupyOperationsError(f"Failed to convert NumPy to CuPy: {e}")
    
    def cupy_to_numpy(self, array: "cp.ndarray") -> np.ndarray:
        """Convert CuPy array back to NumPy array.
        
        Args:
            array: CuPy array to convert
            
        Returns:
            NumPy array
            
        Raises:
            CupyOperationsError: If conversion fails
        """
        try:
            numpy_array = cp.asnumpy(array)
            logger.debug(f"Converted CuPy array shape {array.shape} to NumPy")
            return numpy_array
            
        except Exception as e:
            raise CupyOperationsError(f"Failed to convert CuPy to NumPy: {e}")
    
    def optimize_array_layout(self, array: "cp.ndarray") -> "cp.ndarray":
        """Optimize array memory layout for maximum GPU throughput.
        
        Args:
            array: CuPy array to optimize
            
        Returns:
            Optimized CuPy array
        """
        try:
            # Ensure C-contiguous layout for optimal memory access
            if not array.flags.c_contiguous:
                array = cp.ascontiguousarray(array)
            
            # For 2D arrays, consider transpose if it improves memory access patterns
            if array.ndim == 2 and array.shape[0] > array.shape[1]:
                # Wide matrices might benefit from transpose for certain operations
                logger.debug(f"Array shape {array.shape} - considering layout optimization")
            
            return array
            
        except Exception as e:
            logger.warning(f"Array layout optimization failed: {e}")
            return array
    
    def batch_molecular_descriptors(
        self, 
        molecular_matrices: List[np.ndarray],
        descriptor_func: callable,
        batch_size: Optional[int] = None
    ) -> "cp.ndarray":
        """Calculate molecular descriptors in batches using CuPy operations.
        
        Args:
            molecular_matrices: List of molecular feature matrices
            descriptor_func: Function to calculate descriptors (should work with CuPy)
            batch_size: Batch size for processing (None for auto)
            
        Returns:
            CuPy array with calculated descriptors
            
        Raises:
            CupyOperationsError: If batch processing fails
        """
        try:
            if not molecular_matrices:
                return cp.array([])
            
            # Determine optimal batch size
            if batch_size is None:
                batch_size = self._determine_optimal_batch_size(molecular_matrices)
            
            results = []
            
            for i in range(0, len(molecular_matrices), batch_size):
                batch = molecular_matrices[i:i + batch_size]
                
                # Convert batch to CuPy arrays
                cupy_batch = [self.numpy_to_cupy(mat) for mat in batch]
                
                # Process batch
                batch_results = []
                for cupy_mat in cupy_batch:
                    result = descriptor_func(cupy_mat)
                    batch_results.append(result)
                
                # Combine batch results
                if batch_results:
                    batch_array = cp.stack(batch_results)
                    results.append(batch_array)
            
            # Combine all results
            if results:
                final_result = cp.concatenate(results, axis=0)
            else:
                final_result = cp.array([])
            
            logger.debug(f"Processed {len(molecular_matrices)} molecules in batches")
            return final_result
            
        except Exception as e:
            raise CupyOperationsError(f"Batch processing failed: {e}")
    
    def _determine_optimal_batch_size(self, matrices: List[np.ndarray]) -> int:
        """Determine optimal batch size based on available GPU memory.
        
        Args:
            matrices: List of matrices to process
            
        Returns:
            Optimal batch size
        """
        try:
            # Get available GPU memory
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            
            # Estimate memory per matrix
            if matrices:
                sample_matrix = matrices[0]
                bytes_per_matrix = sample_matrix.nbytes * 4  # Factor for intermediate results
                
                # Use 70% of available memory
                usable_memory = int(free_bytes * 0.7)
                batch_size = max(1, usable_memory // bytes_per_matrix)
                
                # Cap at reasonable maximum
                batch_size = min(batch_size, 1000)
            else:
                batch_size = 100
            
            logger.debug(f"Determined optimal batch size: {batch_size}")
            return batch_size
            
        except Exception as e:
            logger.warning(f"Failed to determine optimal batch size: {e}")
            return 100  # Default fallback
    
    def sparse_matrix_operations(
        self, 
        adjacency_matrices: List[np.ndarray]
    ) -> "cp.ndarray":
        """Perform efficient sparse matrix operations for molecular graphs.
        
        Args:
            adjacency_matrices: List of molecular adjacency matrices
            
        Returns:
            CuPy array with processed results
            
        Raises:
            CupyOperationsError: If sparse operations fail
        """
        try:
            results = []
            
            for adj_matrix in adjacency_matrices:
                # Convert to sparse format for efficiency
                if not sp_sparse.issparse(adj_matrix):
                    sparse_adj = sp_sparse.csr_matrix(adj_matrix)
                else:
                    sparse_adj = adj_matrix
                
                # Convert to CuPy sparse matrix
                cupy_sparse = cp_sparse.csr_matrix(sparse_adj)
                
                # Example operations on sparse molecular graphs
                # Calculate graph properties efficiently
                
                # Degree centrality (sum of each row)
                degrees = cp.array(cupy_sparse.sum(axis=1)).flatten()
                
                # Simple graph metrics
                num_edges = cupy_sparse.nnz // 2  # Undirected graph
                num_nodes = cupy_sparse.shape[0]
                
                # Combine results
                graph_features = cp.array([
                    num_nodes,
                    num_edges,
                    cp.mean(degrees),
                    cp.std(degrees),
                    cp.max(degrees)
                ])
                
                results.append(graph_features)
            
            if results:
                return cp.stack(results)
            else:
                return cp.array([])
            
        except Exception as e:
            raise CupyOperationsError(f"Sparse matrix operations failed: {e}")
    
    def vectorized_molecular_properties(
        self, 
        atom_features: "cp.ndarray",
        bond_features: Optional["cp.ndarray"] = None
    ) -> "cp.ndarray":
        """Calculate molecular properties using vectorized CuPy operations.
        
        Args:
            atom_features: CuPy array of atomic features (n_molecules, max_atoms, n_features)
            bond_features: Optional CuPy array of bond features
            
        Returns:
            CuPy array with calculated molecular properties
            
        Raises:
            CupyOperationsError: If calculation fails
        """
        try:
            if atom_features.ndim != 3:
                raise CupyOperationsError(
                    f"Expected 3D atom_features array, got shape {atom_features.shape}"
                )
            
            n_molecules, max_atoms, n_features = atom_features.shape
            
            # Calculate basic molecular properties using vectorized operations
            properties = []
            
            # Molecular weight (assuming first feature is atomic weight)
            if n_features > 0:
                molecular_weights = cp.sum(atom_features[:, :, 0], axis=1)
                properties.append(molecular_weights)
            
            # Number of atoms (non-zero atomic numbers, assuming second feature)
            if n_features > 1:
                atom_counts = cp.sum(atom_features[:, :, 1] > 0, axis=1)
                properties.append(atom_counts)
            
            # Average atomic properties
            if n_features > 2:
                # Mask for valid atoms
                valid_atoms = atom_features[:, :, 1] > 0  # Non-zero atomic numbers
                
                for feat_idx in range(2, min(n_features, 6)):  # Process first few features
                    # Calculate mean for each molecule, ignoring invalid atoms
                    masked_features = cp.where(
                        valid_atoms, 
                        atom_features[:, :, feat_idx], 
                        0
                    )
                    feature_sums = cp.sum(masked_features, axis=1)
                    atom_counts_for_avg = cp.sum(valid_atoms, axis=1)
                    
                    # Avoid division by zero
                    avg_features = cp.where(
                        atom_counts_for_avg > 0,
                        feature_sums / atom_counts_for_avg,
                        0
                    )
                    properties.append(avg_features)
            
            # Combine all properties
            if properties:
                result = cp.stack(properties, axis=1)
            else:
                result = cp.zeros((n_molecules, 1))
            
            logger.debug(f"Calculated {result.shape[1]} properties for {n_molecules} molecules")
            return result
            
        except Exception as e:
            raise CupyOperationsError(f"Vectorized property calculation failed: {e}")
    
    def benchmark_vs_numpy(
        self, 
        operation_func: callable,
        test_data: np.ndarray,
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """Benchmark CuPy operations against NumPy equivalents.
        
        Args:
            operation_func: Function to benchmark (should work with both NumPy and CuPy)
            test_data: Test data for benchmarking
            num_iterations: Number of iterations for timing
            
        Returns:
            Dictionary with timing results and speedup
        """
        try:
            import time
            
            # NumPy benchmark
            numpy_times = []
            for _ in range(num_iterations):
                start_time = time.time()
                numpy_result = operation_func(test_data)
                numpy_times.append(time.time() - start_time)
            
            # CuPy benchmark
            cupy_data = self.numpy_to_cupy(test_data)
            cp.cuda.Stream.null.synchronize()  # Ensure data transfer is complete
            
            cupy_times = []
            for _ in range(num_iterations):
                start_time = time.time()
                cupy_result = operation_func(cupy_data)
                cp.cuda.Stream.null.synchronize()  # Wait for GPU computation
                cupy_times.append(time.time() - start_time)
            
            # Calculate statistics
            numpy_mean = np.mean(numpy_times)
            cupy_mean = np.mean(cupy_times)
            speedup = numpy_mean / cupy_mean if cupy_mean > 0 else 0
            
            results = {
                'numpy_mean_time': numpy_mean,
                'cupy_mean_time': cupy_mean,
                'speedup': speedup,
                'numpy_std': np.std(numpy_times),
                'cupy_std': np.std(cupy_times)
            }
            
            logger.info(f"Benchmark results: {speedup:.2f}x speedup with CuPy")
            return results
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {}
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current CUDA device.
        
        Returns:
            Dictionary with device information
        """
        try:
            device = cp.cuda.Device(self.device_id)
            
            info = {
                'device_id': self.device_id,
                'device_name': device.attributes['Name'],
                'compute_capability': device.compute_capability,
                'total_memory': device.mem_info[1],
                'free_memory': device.mem_info[0],
                'multiprocessor_count': device.attributes['MultiProcessorCount'],
                'max_threads_per_block': device.attributes['MaxThreadsPerBlock'],
                'max_block_dim': [
                    device.attributes['MaxBlockDimX'],
                    device.attributes['MaxBlockDimY'], 
                    device.attributes['MaxBlockDimZ']
                ],
                'max_grid_dim': [
                    device.attributes['MaxGridDimX'],
                    device.attributes['MaxGridDimY'],
                    device.attributes['MaxGridDimZ']
                ]
            }
            
            return info
            
        except Exception as e:
            logger.warning(f"Failed to get device info: {e}")
            return {'device_id': self.device_id, 'error': str(e)}


def check_cupy_availability() -> bool:
    """Check if CuPy is available.
    
    Returns:
        True if CuPy is available, False otherwise
    """
    return CUPY_AVAILABLE


def get_cupy_version_info() -> Dict[str, str]:
    """Get CuPy version information.
    
    Returns:
        Dictionary with version information
    """
    if not CUPY_AVAILABLE:
        return {"error": "CuPy not available"}
    
    try:
        info = {
            'cupy': cp.__version__,
            'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
            'cuda_driver': cp.cuda.runtime.driverGetVersion()
        }
        return info
    except Exception as e:
        return {"error": str(e)}


# Convenience functions for common operations
def molecular_weight_cupy(atom_weights: "cp.ndarray") -> "cp.ndarray":
    """Calculate molecular weights using CuPy.
    
    Args:
        atom_weights: CuPy array of atomic weights
        
    Returns:
        CuPy array of molecular weights
    """
    return cp.sum(atom_weights, axis=-1)


def tanimoto_similarity_cupy(
    fingerprints1: "cp.ndarray", 
    fingerprints2: "cp.ndarray"
) -> "cp.ndarray":
    """Calculate Tanimoto similarity using CuPy operations.
    
    Args:
        fingerprints1: First set of binary fingerprints
        fingerprints2: Second set of binary fingerprints
        
    Returns:
        CuPy array of Tanimoto similarities
    """
    intersection = cp.sum(fingerprints1 & fingerprints2, axis=-1)
    union = cp.sum(fingerprints1 | fingerprints2, axis=-1)
    
    # Avoid division by zero
    similarity = cp.where(union > 0, intersection / union, 0.0)
    return similarity
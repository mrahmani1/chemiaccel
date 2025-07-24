"""
cudf Integration for ChemiAccel.

This module provides seamless integration with cudf for GPU-native DataFrame operations,
enabling zero-copy data transfers and efficient memory management through RAPIDS.
"""

from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
import logging
import warnings

try:
    import cudf
    import cupy as cp
    from rmm import DeviceBuffer
    import rmm
    CUDF_AVAILABLE = True
except ImportError as e:
    cudf = None
    cp = None
    DeviceBuffer = None
    rmm = None
    CUDF_AVAILABLE = False
    _import_error = e

# Type checking imports
if TYPE_CHECKING:
    import cudf

import pandas as pd
import numpy as np

from ...core.exceptions import ChemiAccelError, BackendUnavailableError
from ...utils.logging import get_logger
from ...utils.rdkit_integration import RDKitMolecularParser, RDKitDescriptorCalculator

logger = get_logger()


class CudfIntegrationError(ChemiAccelError):
    """Raised when cudf integration operations fail."""
    pass


class CudfProcessor:
    """Handles cudf Series input processing and DataFrame output generation."""
    
    def __init__(self, memory_pool_size: Optional[int] = None):
        """Initialize cudf processor with RAPIDS Memory Manager.
        
        Args:
            memory_pool_size: Size of RMM memory pool in bytes (None for auto)
            
        Raises:
            BackendUnavailableError: If cudf/RAPIDS is not available
        """
        if not CUDF_AVAILABLE:
            raise BackendUnavailableError(
                "cudf",
                f"cudf/RAPIDS not available: {_import_error}. "
                "Install with: pip install cudf-cu11 cupy-cuda11x"
            )
        
        self.memory_pool_size = memory_pool_size
        self._initialize_memory_manager()
        
        # Initialize RDKit components for molecular processing
        self.molecular_parser = RDKitMolecularParser(
            sanitize=True,
            remove_hs=True,
            standardize=False
        )
        self.descriptor_calculator = RDKitDescriptorCalculator()
        
        logger.info("CudfProcessor initialized with RAPIDS Memory Manager")
    
    def _initialize_memory_manager(self) -> None:
        """Initialize RAPIDS Memory Manager for efficient GPU memory handling."""
        try:
            # Initialize RMM memory pool if not already done
            if not rmm.is_initialized():
                if self.memory_pool_size:
                    rmm.reinitialize(
                        pool_allocator=True,
                        initial_pool_size=self.memory_pool_size
                    )
                else:
                    # Auto-determine pool size based on available GPU memory
                    import cupy
                    mempool = cupy.get_default_memory_pool()
                    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
                    # Use 80% of available memory for pool
                    pool_size = int(free_bytes * 0.8)
                    rmm.reinitialize(
                        pool_allocator=True,
                        initial_pool_size=pool_size
                    )
                
                logger.info(f"Initialized RMM memory pool with size: {rmm.get_info()}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize RMM memory pool: {e}")
            # Continue without memory pool - cudf will use default allocation
    
    def process_cudf_series(self, molecules: "cudf.Series") -> List[str]:
        """Process cudf.Series input for SMILES strings.
        
        Args:
            molecules: cudf Series containing SMILES strings
            
        Returns:
            List of SMILES strings for processing
            
        Raises:
            CudfIntegrationError: If processing fails
        """
        try:
            if not isinstance(molecules, cudf.Series):
                raise CudfIntegrationError(
                    f"Expected cudf.Series, got {type(molecules)}"
                )
            
            # Convert cudf Series to list efficiently
            # Use zero-copy transfer to host memory when possible
            if molecules.dtype == 'object' or molecules.dtype.kind in ['U', 'S']:
                # String data - convert to pandas then to list
                pandas_series = molecules.to_pandas()
                smiles_list = pandas_series.tolist()
            else:
                raise CudfIntegrationError(
                    f"Expected string data in cudf.Series, got dtype: {molecules.dtype}"
                )
            
            logger.debug(f"Processed cudf.Series with {len(smiles_list)} SMILES strings")
            return smiles_list
            
        except Exception as e:
            raise CudfIntegrationError(f"Failed to process cudf.Series: {e}")
    
    def create_cudf_dataframe(
        self, 
        results_df: pd.DataFrame, 
        original_index: Optional["cudf.Index"] = None
    ) -> "cudf.DataFrame":
        """Create cudf.DataFrame from calculation results with zero-copy when possible.
        
        Args:
            results_df: Pandas DataFrame with calculation results
            original_index: Original cudf Index to preserve
            
        Returns:
            cudf DataFrame with results
            
        Raises:
            CudfIntegrationError: If DataFrame creation fails
        """
        try:
            # Convert pandas DataFrame to cudf DataFrame
            cudf_df = cudf.from_pandas(results_df)
            
            # Preserve original index if provided
            if original_index is not None:
                # Ensure index lengths match
                if len(original_index) == len(cudf_df):
                    cudf_df.index = original_index
                else:
                    logger.warning(
                        f"Index length mismatch: original={len(original_index)}, "
                        f"results={len(cudf_df)}. Using default index."
                    )
            
            logger.debug(f"Created cudf.DataFrame with shape {cudf_df.shape}")
            return cudf_df
            
        except Exception as e:
            raise CudfIntegrationError(f"Failed to create cudf.DataFrame: {e}")
    
    def calculate_descriptors_cudf(
        self, 
        molecules: "cudf.Series", 
        descriptor_sets: List[str]
    ) -> "cudf.DataFrame":
        """Calculate molecular descriptors with cudf input/output.
        
        Args:
            molecules: cudf Series of SMILES strings
            descriptor_sets: List of descriptor names to calculate
            
        Returns:
            cudf DataFrame with calculated descriptors
            
        Raises:
            CudfIntegrationError: If calculation fails
        """
        try:
            logger.info(f"Starting cudf descriptor calculation for {len(molecules)} molecules")
            
            # Store original index for preservation
            original_index = molecules.index
            
            # Process cudf Series to get SMILES list
            smiles_list = self.process_cudf_series(molecules)
            
            if not smiles_list:
                # Return empty cudf DataFrame
                return cudf.DataFrame()
            
            # Parse molecules using RDKit
            molecular_data = self.molecular_parser.parse_smiles_batch(
                smiles_list,
                mol_ids=original_index.to_pandas().tolist(),
                error_handling="skip"
            )
            
            if not molecular_data:
                raise CudfIntegrationError("No valid molecules found after parsing")
            
            # Calculate descriptors using RDKit
            results_df = self.descriptor_calculator.calculate_descriptors_batch(
                molecular_data,
                descriptor_sets
            )
            
            # Convert results to cudf DataFrame
            cudf_results = self.create_cudf_dataframe(results_df, original_index)
            
            logger.info(
                f"Completed cudf descriptor calculation: {len(cudf_results)} molecules, "
                f"{len(cudf_results.columns)} descriptors"
            )
            
            return cudf_results
            
        except Exception as e:
            logger.error(f"cudf descriptor calculation failed: {e}")
            raise CudfIntegrationError(f"Failed to calculate descriptors: {e}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory usage information.
        
        Returns:
            Dictionary with memory usage statistics
        """
        try:
            info = {}
            
            # RMM memory info
            if rmm.is_initialized():
                rmm_info = rmm.get_info()
                info['rmm'] = {
                    'total_bytes': rmm_info.total,
                    'free_bytes': rmm_info.free,
                    'used_bytes': rmm_info.total - rmm_info.free
                }
            
            # CuPy memory info
            if cp is not None:
                mempool = cp.get_default_memory_pool()
                info['cupy'] = {
                    'used_bytes': mempool.used_bytes(),
                    'total_bytes': mempool.total_bytes()
                }
                
                # GPU device memory info
                free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
                info['gpu_device'] = {
                    'free_bytes': free_bytes,
                    'total_bytes': total_bytes,
                    'used_bytes': total_bytes - free_bytes
                }
            
            return info
            
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {}
    
    def optimize_memory_usage(self) -> None:
        """Optimize GPU memory usage by cleaning up unused allocations."""
        try:
            # Clear CuPy memory pool
            if cp is not None:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                logger.debug("Cleared CuPy memory pool")
            
            # Trigger RMM cleanup if available
            if rmm.is_initialized():
                # RMM doesn't have explicit cleanup, but we can get current stats
                info = rmm.get_info()
                logger.debug(f"RMM memory after cleanup: {info}")
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")


def check_cudf_availability() -> bool:
    """Check if cudf and RAPIDS components are available.
    
    Returns:
        True if cudf is available, False otherwise
    """
    return CUDF_AVAILABLE


def get_cudf_version_info() -> Dict[str, str]:
    """Get version information for RAPIDS components.
    
    Returns:
        Dictionary with version information
    """
    if not CUDF_AVAILABLE:
        return {"error": "cudf not available"}
    
    try:
        info = {}
        if cudf is not None:
            info['cudf'] = cudf.__version__
        if cp is not None:
            info['cupy'] = cp.__version__
        if rmm is not None:
            info['rmm'] = rmm.__version__
        return info
    except Exception as e:
        return {"error": str(e)}


# Convenience function for direct usage
def process_molecules_cudf(
    molecules: "cudf.Series", 
    descriptor_sets: List[str],
    memory_pool_size: Optional[int] = None
) -> "cudf.DataFrame":
    """Convenience function for processing molecules with cudf.
    
    Args:
        molecules: cudf Series of SMILES strings
        descriptor_sets: List of descriptor names to calculate
        memory_pool_size: RMM memory pool size in bytes
        
    Returns:
        cudf DataFrame with calculated descriptors
    """
    processor = CudfProcessor(memory_pool_size=memory_pool_size)
    return processor.calculate_descriptors_cudf(molecules, descriptor_sets)
"""
Dask integration for ChemiAccel.

This module provides distributed computing capabilities for large-scale
molecular descriptor calculations using Dask for out-of-core processing
and distributed computation across multiple workers.
"""

from typing import List, Optional, Union, Dict, Any, Callable, TYPE_CHECKING
import warnings
from pathlib import Path

# Dask imports with error handling
try:
    import dask
    import dask.dataframe as dd
    import dask.bag as db
    from dask.distributed import Client, as_completed, get_client
    from dask.delayed import delayed
    DASK_AVAILABLE = True
except ImportError:
    dask = dd = db = Client = as_completed = get_client = delayed = None
    DASK_AVAILABLE = False

# Type checking imports
if TYPE_CHECKING:
    import dask.dataframe as dd

import pandas as pd
import numpy as np

from ..core.calculator import Calculator
from ..core.exceptions import ChemiAccelError, ValidationError
from ..utils.logging import get_logger

logger = get_logger()


class DaskChemiAccelCalculator:
    """Distributed ChemiAccel calculator using Dask.
    
    This class provides distributed molecular descriptor calculation
    capabilities using Dask for processing large datasets that don't
    fit in memory or require distributed computation.
    
    Parameters:
        backend (str): ChemiAccel backend to use on workers
        descriptor_set: Descriptors to calculate
        chunk_size (int): Size of chunks for processing
        client: Optional Dask client instance
        
    Example:
        >>> import dask
        >>> from dask.distributed import Client
        >>> import chemi_accel as ca
        >>> 
        >>> # Start Dask client
        >>> client = Client('scheduler-address:8786')
        >>> 
        >>> # Create distributed calculator
        >>> calc = ca.DaskChemiAccelCalculator(
        >>>     backend='cuda',
        >>>     descriptor_set='basic',
        >>>     chunk_size=10000
        >>> )
        >>> 
        >>> # Process large dataset
        >>> large_smiles_list = [...]  # millions of SMILES
        >>> results = calc.calculate_distributed(large_smiles_list)
        >>> results.to_parquet('molecular_descriptors.parquet')
    """
    
    def __init__(self,
                 backend: str = 'auto',
                 descriptor_set: Union[str, List[str]] = 'basic',
                 chunk_size: int = 10000,
                 client: Optional[Client] = None):
        """Initialize distributed ChemiAccel calculator.
        
        Args:
            backend: Backend to use on Dask workers
            descriptor_set: Descriptors to calculate
            chunk_size: Number of molecules per chunk
            client: Optional Dask client instance
        """
        if not DASK_AVAILABLE:
            raise ChemiAccelError(
                "Dask is required for distributed processing. "
                "Install with: pip install dask[distributed]"
            )
        
        self.backend = backend
        self.descriptor_set = descriptor_set
        self.chunk_size = chunk_size
        self.client = client or self._get_or_create_client()
        
        # Validate client
        if self.client is None:
            raise ChemiAccelError("Could not create or connect to Dask client")
        
        logger.info(f"DaskChemiAccelCalculator initialized with {len(self.client.scheduler_info()['workers'])} workers")
    
    def _get_or_create_client(self) -> Optional[Client]:
        """Get existing Dask client or create a new one."""
        try:
            # Try to get existing client
            return get_client()
        except ValueError:
            # No existing client, create a local one
            try:
                return Client(processes=True, silence_logs=False)
            except Exception as e:
                logger.warning(f"Could not create Dask client: {e}")
                return None
    
    def calculate_distributed(self, 
                            molecules: Union[List[str], pd.Series, str, Path],
                            output_format: str = 'dataframe') -> Union["dd.DataFrame", pd.DataFrame]:
        """Calculate descriptors using distributed processing.
        
        Args:
            molecules: Input molecules in various formats:
                - List[str]: List of SMILES strings
                - pd.Series: Pandas Series of SMILES
                - str/Path: Path to file containing SMILES (CSV, parquet, etc.)
            output_format: Output format ('dataframe', 'dask_dataframe', 'parquet')
            
        Returns:
            Calculated descriptors as DataFrame or Dask DataFrame
            
        Raises:
            ChemiAccelError: If distributed calculation fails
        """
        logger.info("Starting distributed descriptor calculation")
        
        # Convert input to Dask bag
        if isinstance(molecules, (str, Path)):
            # Load from file
            molecules_bag = self._load_molecules_from_file(molecules)
        elif isinstance(molecules, (list, pd.Series)):
            # Convert to Dask bag
            if isinstance(molecules, pd.Series):
                molecules = molecules.tolist()
            molecules_bag = db.from_sequence(molecules, partition_size=self.chunk_size)
        else:
            raise ValidationError(
                input_name="molecules",
                input_value=type(molecules).__name__,
                expected_type="list, pandas Series, or file path",
                reason="Unsupported input type for distributed processing"
            )
        
        # Create delayed calculation function
        calc_func = self._create_calculation_function()
        
        # Map calculation across partitions
        logger.info(f"Processing {molecules_bag.npartitions} partitions with chunk size {self.chunk_size}")
        
        results_bag = molecules_bag.map_partitions(calc_func)
        
        # Convert to DataFrame
        if output_format == 'dask_dataframe':
            # Convert bag to Dask DataFrame
            results_df = results_bag.to_dataframe()
            logger.info("Returning Dask DataFrame")
            return results_df
        elif output_format == 'dataframe':
            # Compute and return pandas DataFrame
            results_list = results_bag.compute()
            results_df = pd.concat([df for df in results_list if not df.empty], ignore_index=True)
            logger.info(f"Computed results: {len(results_df)} molecules")
            return results_df
        elif output_format == 'parquet':
            # Save directly to parquet and return path
            output_path = 'distributed_descriptors.parquet'
            results_bag.to_dataframe().to_parquet(output_path)
            logger.info(f"Results saved to {output_path}")
            return output_path
        else:
            raise ValidationError(
                input_name="output_format",
                input_value=output_format,
                expected_type="'dataframe', 'dask_dataframe', or 'parquet'",
                reason="Invalid output format"
            )
    
    def _load_molecules_from_file(self, file_path: Union[str, Path]) -> "db.Bag":
        """Load molecules from file into Dask bag.
        
        Args:
            file_path: Path to file containing SMILES
            
        Returns:
            Dask bag with SMILES strings
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ChemiAccelError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            # Load CSV file
            df = dd.read_csv(file_path)
            # Assume first column contains SMILES
            smiles_column = df.columns[0]
            return df[smiles_column].to_bag()
        elif file_path.suffix.lower() == '.parquet':
            # Load parquet file
            df = dd.read_parquet(file_path)
            smiles_column = df.columns[0]
            return df[smiles_column].to_bag()
        else:
            # Try to read as text file with one SMILES per line
            with open(file_path, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
            return db.from_sequence(smiles_list, partition_size=self.chunk_size)
    
    def _create_calculation_function(self) -> Callable:
        """Create function for distributed calculation."""
        
        @delayed
        def calculate_chunk(smiles_chunk):
            """Calculate descriptors for a chunk of SMILES."""
            try:
                # Initialize calculator on worker
                calc = Calculator(backend=self.backend)
                calc.configure_descriptors(self.descriptor_set)
                
                # Convert chunk to list if needed
                if hasattr(smiles_chunk, 'tolist'):
                    smiles_list = smiles_chunk.tolist()
                else:
                    smiles_list = list(smiles_chunk)
                
                if not smiles_list:
                    return pd.DataFrame()
                
                # Calculate descriptors
                results = calc.pandas(smiles_list)
                
                logger.debug(f"Processed chunk: {len(smiles_list)} molecules -> {len(results)} results")
                return results
                
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                # Return empty DataFrame on error
                return pd.DataFrame()
        
        return calculate_chunk
    
    def calculate_from_csv(self, 
                          csv_path: Union[str, Path],
                          smiles_column: str = 'smiles',
                          output_path: Optional[Union[str, Path]] = None) -> "dd.DataFrame":
        """Calculate descriptors from CSV file with SMILES.
        
        Args:
            csv_path: Path to CSV file
            smiles_column: Name of column containing SMILES
            output_path: Optional path to save results
            
        Returns:
            Dask DataFrame with results
        """
        # Read CSV as Dask DataFrame
        df = dd.read_csv(csv_path)
        
        if smiles_column not in df.columns:
            raise ValidationError(
                input_name="smiles_column",
                input_value=smiles_column,
                expected_type=f"one of {list(df.columns)}",
                reason="SMILES column not found in CSV"
            )
        
        # Extract SMILES
        smiles_series = df[smiles_column]
        
        # Calculate descriptors
        results = self.calculate_distributed(smiles_series.compute(), output_format='dask_dataframe')
        
        # Save if output path specified
        if output_path:
            results.to_parquet(output_path)
            logger.info(f"Results saved to {output_path}")
        
        return results
    
    def benchmark_distributed(self, 
                            num_molecules: int = 100000,
                            num_workers: Optional[int] = None) -> Dict[str, Any]:
        """Benchmark distributed processing performance.
        
        Args:
            num_molecules: Number of test molecules
            num_workers: Number of workers to use (None for all available)
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        # Generate test molecules
        test_molecules = [
            'CCO', 'c1ccccc1', 'CC(=O)O', 'CCN(CC)CC', 'c1ccc2c(c1)ccccc2'
        ] * (num_molecules // 5 + 1)
        test_molecules = test_molecules[:num_molecules]
        
        logger.info(f"Starting distributed benchmark: {num_molecules} molecules")
        
        start_time = time.time()
        
        # Calculate descriptors
        results = self.calculate_distributed(test_molecules, output_format='dataframe')
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        molecules_per_second = len(results) / total_time
        num_descriptors = len([col for col in results.columns if col not in ['mol_id', 'smiles']])
        descriptors_per_second = molecules_per_second * num_descriptors
        
        # Get worker info
        worker_info = self.client.scheduler_info()['workers']
        num_workers_used = len(worker_info)
        
        benchmark_results = {
            'backend': self.backend,
            'num_molecules': len(results),
            'num_descriptors': num_descriptors,
            'num_workers': num_workers_used,
            'chunk_size': self.chunk_size,
            'total_time_seconds': total_time,
            'molecules_per_second': molecules_per_second,
            'descriptors_per_second': descriptors_per_second,
            'speedup_estimate': molecules_per_second / 100  # Rough estimate vs single core
        }
        
        logger.info(f"Distributed benchmark complete: {molecules_per_second:.1f} molecules/sec")
        return benchmark_results
    
    def close(self):
        """Close the Dask client."""
        if self.client:
            self.client.close()
            logger.info("Dask client closed")


# Convenience functions
def distributed_calculate_descriptors(molecules: Union[List[str], str, Path],
                                    backend: str = 'auto',
                                    descriptor_set: Union[str, List[str]] = 'basic',
                                    chunk_size: int = 10000,
                                    client: Optional[Client] = None) -> pd.DataFrame:
    """Calculate descriptors using distributed processing (convenience function).
    
    Args:
        molecules: Input molecules (list, file path, etc.)
        backend: ChemiAccel backend to use
        descriptor_set: Descriptors to calculate
        chunk_size: Chunk size for processing
        client: Optional Dask client
        
    Returns:
        DataFrame with calculated descriptors
        
    Example:
        >>> import chemi_accel as ca
        >>> 
        >>> # Process large SMILES list
        >>> smiles = [...]  # large list
        >>> results = ca.distributed_calculate_descriptors(
        >>>     smiles, 
        >>>     backend='cuda',
        >>>     chunk_size=5000
        >>> )
    """
    calc = DaskChemiAccelCalculator(
        backend=backend,
        descriptor_set=descriptor_set,
        chunk_size=chunk_size,
        client=client
    )
    
    try:
        return calc.calculate_distributed(molecules)
    finally:
        if client is None:  # Only close if we created the client
            calc.close()


def setup_dask_cluster(n_workers: int = 4, 
                      threads_per_worker: int = 2,
                      memory_limit: str = '4GB') -> Client:
    """Set up a local Dask cluster for ChemiAccel processing.
    
    Args:
        n_workers: Number of worker processes
        threads_per_worker: Number of threads per worker
        memory_limit: Memory limit per worker
        
    Returns:
        Dask Client instance
        
    Example:
        >>> import chemi_accel as ca
        >>> 
        >>> # Set up local cluster
        >>> client = ca.setup_dask_cluster(n_workers=8, memory_limit='8GB')
        >>> 
        >>> # Use with distributed calculator
        >>> calc = ca.DaskChemiAccelCalculator(client=client)
    """
    if not DASK_AVAILABLE:
        raise ChemiAccelError("Dask not available for cluster setup")
    
    from dask.distributed import LocalCluster
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        silence_logs=False
    )
    
    client = Client(cluster)
    
    logger.info(f"Dask cluster started: {n_workers} workers, {client.scheduler_info()['address']}")
    return client


def process_large_dataset(input_path: Union[str, Path],
                         output_path: Union[str, Path],
                         smiles_column: str = 'smiles',
                         backend: str = 'auto',
                         descriptor_set: Union[str, List[str]] = 'basic',
                         chunk_size: int = 10000) -> None:
    """Process large molecular dataset from file to file.
    
    This function provides a high-level interface for processing large
    molecular datasets that don't fit in memory, reading from input files
    and writing results to output files.
    
    Args:
        input_path: Path to input file (CSV, parquet, etc.)
        output_path: Path to output file (parquet recommended)
        smiles_column: Name of SMILES column in input
        backend: ChemiAccel backend to use
        descriptor_set: Descriptors to calculate
        chunk_size: Processing chunk size
        
    Example:
        >>> import chemi_accel as ca
        >>> 
        >>> # Process large CSV file
        >>> ca.process_large_dataset(
        >>>     'large_molecules.csv',
        >>>     'molecular_descriptors.parquet',
        >>>     smiles_column='canonical_smiles',
        >>>     backend='cuda',
        >>>     chunk_size=20000
        >>> )
    """
    logger.info(f"Processing large dataset: {input_path} -> {output_path}")
    
    calc = DaskChemiAccelCalculator(
        backend=backend,
        descriptor_set=descriptor_set,
        chunk_size=chunk_size
    )
    
    try:
        # Process file
        results = calc.calculate_from_csv(
            input_path,
            smiles_column=smiles_column,
            output_path=output_path
        )
        
        logger.info(f"Dataset processing complete: {output_path}")
        
    finally:
        calc.close()
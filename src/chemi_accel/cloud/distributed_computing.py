"""
Distributed computing integration for ChemiAccel.

Provides Dask and Ray cluster integration for distributed molecular processing
with automatic scaling and fault tolerance.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Iterator, Callable
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for distributed computing clusters."""
    cluster_type: str  # 'dask', 'ray'
    scheduler_address: str = None
    n_workers: int = 4
    threads_per_worker: int = 1
    memory_per_worker: str = "8GB"
    resources_per_worker: Dict[str, int] = None
    auto_scaling: bool = True
    min_workers: int = 1
    max_workers: int = 10


class DaskIntegration:
    """Dask cluster integration for distributed ChemiAccel processing."""
    
    def __init__(self, cluster_config: ClusterConfig = None):
        """
        Initialize Dask integration.
        
        Args:
            cluster_config: Cluster configuration
        """
        self.config = cluster_config or ClusterConfig(cluster_type='dask')
        self.client = None
        self.cluster = None
        
        try:
            import dask
            import dask.distributed
            import dask_cuda
            self.dask_available = True
        except ImportError:
            logger.warning("Dask not available. Install with: pip install dask[distributed] dask-cuda")
            self.dask_available = False
    
    def setup_cluster(self, cluster_type: str = 'local') -> bool:
        """
        Setup Dask cluster.
        
        Args:
            cluster_type: Type of cluster ('local', 'slurm', 'kubernetes', 'existing')
            
        Returns:
            Success status
        """
        if not self.dask_available:
            return False
        
        try:
            import dask.distributed as dd
            import dask_cuda
            
            if cluster_type == 'local':
                # Local GPU cluster
                self.cluster = dask_cuda.LocalCUDACluster(
                    n_workers=self.config.n_workers,
                    threads_per_worker=self.config.threads_per_worker,
                    memory_limit=self.config.memory_per_worker
                )
                
            elif cluster_type == 'slurm':
                # SLURM cluster
                from dask_jobqueue import SLURMCluster
                
                self.cluster = SLURMCluster(
                    cores=self.config.threads_per_worker,
                    memory=self.config.memory_per_worker,
                    processes=1,
                    queue='gpu',
                    walltime='24:00:00',
                    job_extra=['--gres=gpu:1'],
                    env_extra=['export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID']
                )
                
                if self.config.auto_scaling:
                    self.cluster.adapt(
                        minimum=self.config.min_workers,
                        maximum=self.config.max_workers
                    )
                else:
                    self.cluster.scale(self.config.n_workers)
            
            elif cluster_type == 'kubernetes':
                # Kubernetes cluster
                from dask_kubernetes import KubeCluster
                
                self.cluster = KubeCluster.from_yaml('dask-worker-gpu.yaml')
                
                if self.config.auto_scaling:
                    self.cluster.adapt(
                        minimum=self.config.min_workers,
                        maximum=self.config.max_workers
                    )
            
            elif cluster_type == 'existing':
                # Connect to existing cluster
                if not self.config.scheduler_address:
                    raise ValueError("scheduler_address required for existing cluster")
                
                self.client = dd.Client(self.config.scheduler_address)
                return True
            
            else:
                raise ValueError(f"Unsupported cluster type: {cluster_type}")
            
            # Create client
            self.client = dd.Client(self.cluster)
            
            logger.info(f"Dask cluster setup complete: {self.client}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Dask cluster: {e}")
            return False
    
    def process_molecules_distributed(self, molecules: pd.Series,
                                    descriptor_config: Dict[str, Any] = None,
                                    chunk_size: int = 1000) -> pd.DataFrame:
        """
        Process molecules using distributed Dask computation.
        
        Args:
            molecules: Series of SMILES strings
            descriptor_config: Descriptor calculation configuration
            chunk_size: Chunk size for processing
            
        Returns:
            DataFrame with calculated descriptors
        """
        if not self.client:
            raise RuntimeError("Dask cluster not initialized")
        
        try:
            import dask.dataframe as dd
            from dask import delayed
            
            # Create Dask DataFrame
            df = pd.DataFrame({'smiles': molecules})
            ddf = dd.from_pandas(df, npartitions=max(1, len(df) // chunk_size))
            
            # Define processing function
            @delayed
            def process_chunk(chunk_df):
                from chemi_accel import Calculator
                
                calc = Calculator(backend='cuda', batch_size=min(len(chunk_df), 500))
                descriptors = calc.pandas(chunk_df['smiles'])
                
                return pd.concat([chunk_df, descriptors], axis=1)
            
            # Process chunks in parallel
            futures = []
            for partition in ddf.to_delayed():
                future = process_chunk(partition)
                futures.append(future)
            
            # Compute results
            results = dd.from_delayed(futures)
            final_result = results.compute()
            
            logger.info(f"Processed {len(final_result)} molecules using Dask")
            return final_result
            
        except Exception as e:
            logger.error(f"Dask processing failed: {e}")
            raise
    
    def process_cudf_distributed(self, cudf_series,
                                descriptor_config: Dict[str, Any] = None) -> 'cudf.DataFrame':
        """
        Process cuDF Series using distributed computation.
        
        Args:
            cudf_series: cuDF Series of SMILES strings
            descriptor_config: Descriptor configuration
            
        Returns:
            cuDF DataFrame with descriptors
        """
        if not self.client:
            raise RuntimeError("Dask cluster not initialized")
        
        try:
            import dask_cudf
            from dask import delayed
            
            # Create Dask-cuDF DataFrame
            df = cudf_series.to_frame('smiles')
            ddf = dask_cudf.from_cudf(df, npartitions=4)
            
            # Define processing function
            def process_cudf_chunk(chunk_df):
                from chemi_accel import Calculator
                
                calc = Calculator(backend='cuda')
                descriptors = calc.cudf(chunk_df['smiles'])
                
                return descriptors
            
            # Apply processing to each partition
            result_ddf = ddf.map_partitions(process_cudf_chunk, meta=None)
            
            # Compute final result
            final_result = result_ddf.compute()
            
            logger.info(f"Processed {len(final_result)} molecules using Dask-cuDF")
            return final_result
            
        except Exception as e:
            logger.error(f"Dask-cuDF processing failed: {e}")
            raise
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status and metrics."""
        if not self.client:
            return {}
        
        try:
            status = {
                'scheduler_info': self.client.scheduler_info(),
                'worker_count': len(self.client.scheduler_info()['workers']),
                'total_cores': sum(w['nthreads'] for w in self.client.scheduler_info()['workers'].values()),
                'total_memory': sum(w['memory_limit'] for w in self.client.scheduler_info()['workers'].values()),
                'tasks': self.client.scheduler_info()['tasks']
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {}
    
    def shutdown_cluster(self):
        """Shutdown Dask cluster."""
        try:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()
            
            logger.info("Dask cluster shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down Dask cluster: {e}")


class RayIntegration:
    """Ray cluster integration for distributed ChemiAccel processing."""
    
    def __init__(self, cluster_config: ClusterConfig = None):
        """
        Initialize Ray integration.
        
        Args:
            cluster_config: Cluster configuration
        """
        self.config = cluster_config or ClusterConfig(cluster_type='ray')
        self.ray_initialized = False
        
        try:
            import ray
            self.ray_available = True
        except ImportError:
            logger.warning("Ray not available. Install with: pip install ray[default]")
            self.ray_available = False
    
    def setup_cluster(self, cluster_type: str = 'local') -> bool:
        """
        Setup Ray cluster.
        
        Args:
            cluster_type: Type of cluster ('local', 'existing', 'kubernetes')
            
        Returns:
            Success status
        """
        if not self.ray_available:
            return False
        
        try:
            import ray
            
            if cluster_type == 'local':
                # Local Ray cluster
                ray.init(
                    num_cpus=self.config.n_workers * self.config.threads_per_worker,
                    num_gpus=self.config.n_workers,
                    object_store_memory=2000000000  # 2GB
                )
                
            elif cluster_type == 'existing':
                # Connect to existing Ray cluster
                if not self.config.scheduler_address:
                    raise ValueError("scheduler_address required for existing cluster")
                
                ray.init(address=self.config.scheduler_address)
            
            elif cluster_type == 'kubernetes':
                # Ray on Kubernetes
                from ray.util.kubernetes import KubernetesJobSubmissionClient
                
                # This would require additional setup
                ray.init(address='ray://ray-head:10001')
            
            else:
                raise ValueError(f"Unsupported cluster type: {cluster_type}")
            
            self.ray_initialized = True
            logger.info(f"Ray cluster initialized: {ray.cluster_resources()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Ray cluster: {e}")
            return False
    
    def process_molecules_distributed(self, molecules: pd.Series,
                                    descriptor_config: Dict[str, Any] = None,
                                    chunk_size: int = 1000) -> pd.DataFrame:
        """
        Process molecules using distributed Ray computation.
        
        Args:
            molecules: Series of SMILES strings
            descriptor_config: Descriptor configuration
            chunk_size: Chunk size for processing
            
        Returns:
            DataFrame with calculated descriptors
        """
        if not self.ray_initialized:
            raise RuntimeError("Ray cluster not initialized")
        
        try:
            import ray
            
            # Define remote processing function
            @ray.remote(num_gpus=1)
            class ChemiAccelWorker:
                def __init__(self):
                    from chemi_accel import Calculator
                    self.calc = Calculator(backend='cuda', batch_size=500)
                
                def process_chunk(self, smiles_list):
                    import pandas as pd
                    
                    chunk_df = pd.DataFrame({'smiles': smiles_list})
                    descriptors = self.calc.pandas(chunk_df['smiles'])
                    
                    return pd.concat([chunk_df, descriptors], axis=1)
            
            # Create workers
            num_workers = min(self.config.n_workers, len(molecules) // chunk_size + 1)
            workers = [ChemiAccelWorker.remote() for _ in range(num_workers)]
            
            # Split molecules into chunks
            chunks = []
            for i in range(0, len(molecules), chunk_size):
                chunk = molecules.iloc[i:i + chunk_size].tolist()
                chunks.append(chunk)
            
            # Process chunks in parallel
            futures = []
            for i, chunk in enumerate(chunks):
                worker = workers[i % num_workers]
                future = worker.process_chunk.remote(chunk)
                futures.append(future)
            
            # Get results
            results = ray.get(futures)
            
            # Combine results
            final_result = pd.concat(results, ignore_index=True)
            
            logger.info(f"Processed {len(final_result)} molecules using Ray")
            return final_result
            
        except Exception as e:
            logger.error(f"Ray processing failed: {e}")
            raise
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current Ray cluster status."""
        if not self.ray_initialized:
            return {}
        
        try:
            import ray
            
            status = {
                'cluster_resources': ray.cluster_resources(),
                'available_resources': ray.available_resources(),
                'nodes': ray.nodes(),
                'tasks': ray.timeline()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get Ray cluster status: {e}")
            return {}
    
    def shutdown_cluster(self):
        """Shutdown Ray cluster."""
        try:
            import ray
            ray.shutdown()
            self.ray_initialized = False
            
            logger.info("Ray cluster shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down Ray cluster: {e}")


class DistributedChemiAccelCalculator:
    """Unified distributed calculator for ChemiAccel."""
    
    def __init__(self, framework: str = 'dask', 
                 cluster_config: ClusterConfig = None):
        """
        Initialize distributed calculator.
        
        Args:
            framework: Distributed framework ('dask', 'ray')
            cluster_config: Cluster configuration
        """
        self.framework = framework
        self.config = cluster_config or ClusterConfig(cluster_type=framework)
        
        if framework == 'dask':
            self.backend = DaskIntegration(cluster_config)
        elif framework == 'ray':
            self.backend = RayIntegration(cluster_config)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def setup_cluster(self, cluster_type: str = 'local') -> bool:
        """Setup distributed cluster."""
        return self.backend.setup_cluster(cluster_type)
    
    def pandas(self, molecules: pd.Series, 
               descriptor_config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Calculate descriptors using distributed processing.
        
        Args:
            molecules: Series of SMILES strings
            descriptor_config: Descriptor configuration
            
        Returns:
            DataFrame with calculated descriptors
        """
        return self.backend.process_molecules_distributed(
            molecules, descriptor_config
        )
    
    def cudf(self, molecules, descriptor_config: Dict[str, Any] = None):
        """
        Calculate descriptors using distributed cuDF processing.
        
        Args:
            molecules: cuDF Series of SMILES strings
            descriptor_config: Descriptor configuration
            
        Returns:
            cuDF DataFrame with descriptors
        """
        if self.framework != 'dask':
            raise NotImplementedError("cuDF processing only supported with Dask")
        
        return self.backend.process_cudf_distributed(molecules, descriptor_config)
    
    def process_large_dataset(self, input_path: str, output_path: str,
                             descriptor_config: Dict[str, Any] = None,
                             chunk_size: int = 10000) -> bool:
        """
        Process large dataset from file.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            descriptor_config: Descriptor configuration
            chunk_size: Processing chunk size
            
        Returns:
            Success status
        """
        try:
            # Read data in chunks
            all_results = []
            
            if input_path.endswith('.parquet'):
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(input_path)
                
                for batch in parquet_file.iter_batches(batch_size=chunk_size):
                    chunk_df = batch.to_pandas()
                    if 'smiles' in chunk_df.columns:
                        result = self.pandas(chunk_df['smiles'], descriptor_config)
                        all_results.append(result)
            
            elif input_path.endswith('.csv'):
                for chunk_df in pd.read_csv(input_path, chunksize=chunk_size):
                    if 'smiles' in chunk_df.columns:
                        result = self.pandas(chunk_df['smiles'], descriptor_config)
                        all_results.append(result)
            
            else:
                raise ValueError(f"Unsupported file format: {input_path}")
            
            # Combine and save results
            if all_results:
                final_result = pd.concat(all_results, ignore_index=True)
                
                if output_path.endswith('.parquet'):
                    final_result.to_parquet(output_path, index=False)
                elif output_path.endswith('.csv'):
                    final_result.to_csv(output_path, index=False)
                else:
                    raise ValueError(f"Unsupported output format: {output_path}")
                
                logger.info(f"Processed {len(final_result)} molecules, saved to {output_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to process large dataset: {e}")
            return False
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status."""
        return self.backend.get_cluster_status()
    
    def shutdown(self):
        """Shutdown distributed cluster."""
        self.backend.shutdown_cluster()


# Utility functions
def setup_dask_cluster(scheduler_address: str = None,
                      n_workers: int = 4) -> DaskIntegration:
    """
    Convenience function to setup Dask cluster.
    
    Args:
        scheduler_address: Existing scheduler address
        n_workers: Number of workers for local cluster
        
    Returns:
        DaskIntegration instance
    """
    config = ClusterConfig(
        cluster_type='dask',
        scheduler_address=scheduler_address,
        n_workers=n_workers
    )
    
    dask_integration = DaskIntegration(config)
    
    if scheduler_address:
        dask_integration.setup_cluster('existing')
    else:
        dask_integration.setup_cluster('local')
    
    return dask_integration


def setup_ray_cluster(cluster_address: str = None,
                     n_workers: int = 4) -> RayIntegration:
    """
    Convenience function to setup Ray cluster.
    
    Args:
        cluster_address: Existing cluster address
        n_workers: Number of workers for local cluster
        
    Returns:
        RayIntegration instance
    """
    config = ClusterConfig(
        cluster_type='ray',
        scheduler_address=cluster_address,
        n_workers=n_workers
    )
    
    ray_integration = RayIntegration(config)
    
    if cluster_address:
        ray_integration.setup_cluster('existing')
    else:
        ray_integration.setup_cluster('local')
    
    return ray_integration


def distributed_calculate_descriptors(molecules: pd.Series,
                                     framework: str = 'dask',
                                     cluster_config: ClusterConfig = None) -> pd.DataFrame:
    """
    Convenience function for distributed descriptor calculation.
    
    Args:
        molecules: Series of SMILES strings
        framework: Distributed framework ('dask', 'ray')
        cluster_config: Cluster configuration
        
    Returns:
        DataFrame with calculated descriptors
    """
    calc = DistributedChemiAccelCalculator(framework, cluster_config)
    calc.setup_cluster('local')
    
    try:
        result = calc.pandas(molecules)
        return result
    finally:
        calc.shutdown()


def process_large_dataset(input_path: str, output_path: str,
                         framework: str = 'dask',
                         chunk_size: int = 10000) -> bool:
    """
    Convenience function to process large datasets.
    
    Args:
        input_path: Input file path
        output_path: Output file path
        framework: Distributed framework
        chunk_size: Processing chunk size
        
    Returns:
        Success status
    """
    calc = DistributedChemiAccelCalculator(framework)
    calc.setup_cluster('local')
    
    try:
        return calc.process_large_dataset(input_path, output_path, chunk_size=chunk_size)
    finally:
        calc.shutdown()
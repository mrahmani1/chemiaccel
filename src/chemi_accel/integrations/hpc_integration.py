"""
HPC and Distributed Computing Integration for ChemiAccel.

This module provides integration with high-performance computing environments,
including MPI, SLURM, and distributed computing frameworks.
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except (ImportError, RuntimeError):
    _MPI_AVAILABLE = False
    MPI = None

try:
    import dask
    import dask.distributed
    from dask.distributed import Client, as_completed
    import dask.dataframe as dd
    _DASK_AVAILABLE = True
except ImportError:
    _DASK_AVAILABLE = False
    dask = None

try:
    import ray
    _RAY_AVAILABLE = True
except (ImportError, RuntimeError):
    _RAY_AVAILABLE = False
    ray = None

from ..core.calculator import Calculator
from ..core.exceptions import ChemiAccelError, DistributedComputingError
from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class HPCJobConfig:
    """Configuration for HPC cluster jobs."""
    scheduler: str  # 'slurm', 'pbs', 'sge'
    nodes: int
    gpus_per_node: int
    walltime: str
    partition: str
    modules_to_load: List[str]
    mpi_enabled: bool
    job_name: str = "chemiaccel_job"
    memory_per_node: str = "32GB"
    output_file: str = "chemiaccel_%j.out"
    error_file: str = "chemiaccel_%j.err"
    email: Optional[str] = None
    account: Optional[str] = None


@dataclass
class MPIConfig:
    """MPI configuration and environment information."""
    world_size: int
    rank: int
    processor_name: str
    is_root: bool
    communicator: Any = None


class MPIIntegration:
    """MPI integration with automatic world size detection."""
    
    def __init__(self):
        """Initialize MPI integration."""
        if not _MPI_AVAILABLE:
            raise DistributedComputingError("MPI not available. Install mpi4py.")
        
        self.comm = MPI.COMM_WORLD
        self.config = MPIConfig(
            world_size=self.comm.Get_size(),
            rank=self.comm.Get_rank(),
            processor_name=MPI.Get_processor_name(),
            is_root=self.comm.Get_rank() == 0,
            communicator=self.comm
        )
        
        logger.info(f"MPI initialized: rank {self.config.rank}/{self.config.world_size} "
                   f"on {self.config.processor_name}")
    
    def distribute_workload(self, molecules: pd.Series) -> pd.Series:
        """Distribute molecular dataset across MPI ranks."""
        if not _MPI_AVAILABLE:
            return molecules
        
        total_molecules = len(molecules)
        molecules_per_rank = total_molecules // self.config.world_size
        remainder = total_molecules % self.config.world_size
        
        # Calculate start and end indices for this rank
        start_idx = self.config.rank * molecules_per_rank
        if self.config.rank < remainder:
            start_idx += self.config.rank
            end_idx = start_idx + molecules_per_rank + 1
        else:
            start_idx += remainder
            end_idx = start_idx + molecules_per_rank
        
        local_molecules = molecules.iloc[start_idx:end_idx]
        
        logger.info(f"Rank {self.config.rank}: processing {len(local_molecules)} molecules "
                   f"(indices {start_idx}:{end_idx})")
        
        return local_molecules
    
    def gather_results(self, local_results: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Gather results from all MPI ranks."""
        if not _MPI_AVAILABLE:
            return local_results
        
        # Gather all results to root process
        all_results = self.comm.gather(local_results, root=0)
        
        if self.config.is_root:
            # Concatenate all results
            combined_results = pd.concat(all_results, ignore_index=True)
            logger.info(f"Gathered results from {self.config.world_size} ranks: "
                       f"{len(combined_results)} total molecules processed")
            return combined_results
        
        return None
    
    def broadcast_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast configuration from root to all ranks."""
        if not _MPI_AVAILABLE:
            return config
        
        return self.comm.bcast(config, root=0)
    
    def barrier(self):
        """Synchronize all MPI processes."""
        if _MPI_AVAILABLE:
            self.comm.Barrier()


class SLURMJobManager:
    """SLURM job script generation and submission utilities, including Google Cloud Run."""
    
    def __init__(self):
        """Initialize SLURM job manager."""
        self.slurm_available = self._check_slurm_availability()
        self.gcloud_available = self._check_gcloud_availability()
    
    def _check_slurm_availability(self) -> bool:
        """Check if SLURM is available on the system."""
        try:
            subprocess.run(['sinfo', '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_gcloud_availability(self) -> bool:
        """Check if Google Cloud SDK is available."""
        try:
            subprocess.run(['gcloud', '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def generate_job_script(self, config: HPCJobConfig, 
                          python_script: str,
                          script_args: List[str] = None) -> str:
        """Generate SLURM job script."""
        if script_args is None:
            script_args = []
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name={config.job_name}
#SBATCH --nodes={config.nodes}
#SBATCH --ntasks-per-node={config.gpus_per_node}
#SBATCH --gres=gpu:{config.gpus_per_node}
#SBATCH --time={config.walltime}
#SBATCH --partition={config.partition}
#SBATCH --mem={config.memory_per_node}
#SBATCH --output={config.output_file}
#SBATCH --error={config.error_file}
"""
        
        if config.account:
            script_content += f"#SBATCH --account={config.account}\n"
        
        if config.email:
            script_content += f"#SBATCH --mail-user={config.email}\n"
            script_content += "#SBATCH --mail-type=BEGIN,END,FAIL\n"
        
        script_content += "\n# Load required modules\n"
        for module in config.modules_to_load:
            script_content += f"module load {module}\n"
        
        script_content += "\n# Set environment variables\n"
        script_content += "export OMP_NUM_THREADS=1\n"
        script_content += "export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID\n"
        
        if config.mpi_enabled:
            script_content += f"\n# Run with MPI\n"
            script_content += f"mpirun -np {config.nodes * config.gpus_per_node} "
            script_content += f"python {python_script} {' '.join(script_args)}\n"
        else:
            script_content += f"\n# Run without MPI\n"
            script_content += f"python {python_script} {' '.join(script_args)}\n"
        
        return script_content
    
    def submit_job(self, job_script_path: str) -> Optional[str]:
        """Submit job to SLURM scheduler."""
        if not self.slurm_available:
            raise DistributedComputingError("SLURM not available on this system")
        
        try:
            result = subprocess.run(['sbatch', job_script_path], 
                                  capture_output=True, text=True, check=True)
            
            # Extract job ID from output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if 'Submitted batch job' in line:
                    job_id = line.split()[-1]
                    logger.info(f"Submitted SLURM job {job_id}")
                    return job_id
            
            return None
            
        except subprocess.CalledProcessError as e:
            raise DistributedComputingError(f"Failed to submit SLURM job: {e.stderr}")
    
    def check_job_status(self, job_id: str) -> Dict[str, str]:
        """Check status of SLURM job."""
        if not self.slurm_available:
            return {"status": "unknown", "reason": "SLURM not available"}
        
        try:
            result = subprocess.run(['squeue', '-j', job_id, '--format=%T,%R'], 
                                  capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                status_line = lines[1]
                parts = status_line.split(',')
                return {
                    "status": parts[0] if parts else "unknown",
                    "reason": parts[1] if len(parts) > 1 else ""
                }
            else:
                return {"status": "completed", "reason": "Job not in queue"}
                
        except subprocess.CalledProcessError:
            return {"status": "unknown", "reason": "Failed to query job status"}
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel SLURM job."""
        if not self.slurm_available:
            return False
        
        try:
            subprocess.run(['scancel', job_id], check=True)
            logger.info(f"Cancelled SLURM job {job_id}")
            return True
        except subprocess.CalledProcessError:
            logger.error(f"Failed to cancel SLURM job {job_id}")
            return False
    
    def submit_google_cloud_run_job(self, config: HPCJobConfig, 
                                   container_image: str,
                                   input_data_uri: str,
                                   output_data_uri: str,
                                   project_id: str,
                                   region: str = 'us-central1') -> Optional[str]:
        """Submit job to Google Cloud Run."""
        if not self.gcloud_available:
            raise DistributedComputingError("Google Cloud SDK not available")
        
        try:
            # Create Cloud Run job configuration
            job_name = f"{config.job_name}-{int(time.time())}"
            
            # Build gcloud command
            cmd = [
                'gcloud', 'run', 'jobs', 'create', job_name,
                '--image', container_image,
                '--region', region,
                '--project', project_id,
                '--task-timeout', config.walltime,
                '--memory', config.memory_per_node,
                '--cpu', str(config.gpus_per_node),  # Use as CPU count for Cloud Run
                '--parallelism', str(config.nodes),
                '--set-env-vars', f'INPUT_DATA_URI={input_data_uri}',
                '--set-env-vars', f'OUTPUT_DATA_URI={output_data_uri}',
                '--format', 'value(metadata.name)'
            ]
            
            # Add GPU support if available (currently limited in Cloud Run)
            if config.gpus_per_node > 0:
                logger.warning("GPU support in Cloud Run is limited. Using CPU configuration.")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            job_id = result.stdout.strip()
            logger.info(f"Created Google Cloud Run job {job_id}")
            
            # Execute the job
            execute_cmd = [
                'gcloud', 'run', 'jobs', 'execute', job_id,
                '--region', region,
                '--project', project_id,
                '--format', 'value(metadata.name)'
            ]
            
            execute_result = subprocess.run(execute_cmd, capture_output=True, text=True, check=True)
            execution_id = execute_result.stdout.strip()
            
            logger.info(f"Executed Google Cloud Run job {execution_id}")
            return execution_id
            
        except subprocess.CalledProcessError as e:
            raise DistributedComputingError(f"Failed to submit Google Cloud Run job: {e.stderr}")
    
    def check_google_cloud_run_status(self, execution_id: str, 
                                     project_id: str,
                                     region: str = 'us-central1') -> Dict[str, str]:
        """Check status of Google Cloud Run job execution."""
        if not self.gcloud_available:
            return {"status": "unknown", "reason": "Google Cloud SDK not available"}
        
        try:
            cmd = [
                'gcloud', 'run', 'jobs', 'executions', 'describe', execution_id,
                '--region', region,
                '--project', project_id,
                '--format', 'value(status.conditions[0].type,status.conditions[0].status)'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.stdout.strip():
                parts = result.stdout.strip().split('\t')
                condition_type = parts[0] if parts else "Unknown"
                condition_status = parts[1] if len(parts) > 1 else "Unknown"
                
                # Map Cloud Run status to standard status
                if condition_type == "Completed" and condition_status == "True":
                    status = "completed"
                elif condition_type == "Running" and condition_status == "True":
                    status = "running"
                elif condition_type == "Failed" and condition_status == "True":
                    status = "failed"
                else:
                    status = "pending"
                
                return {"status": status, "reason": f"{condition_type}:{condition_status}"}
            else:
                return {"status": "unknown", "reason": "No status information"}
                
        except subprocess.CalledProcessError as e:
            return {"status": "error", "reason": f"Failed to query status: {e.stderr}"}
    
    def cancel_google_cloud_run_job(self, execution_id: str,
                                   project_id: str,
                                   region: str = 'us-central1') -> bool:
        """Cancel Google Cloud Run job execution."""
        if not self.gcloud_available:
            return False
        
        try:
            cmd = [
                'gcloud', 'run', 'jobs', 'executions', 'cancel', execution_id,
                '--region', region,
                '--project', project_id
            ]
            
            subprocess.run(cmd, check=True)
            logger.info(f"Cancelled Google Cloud Run execution {execution_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel Google Cloud Run execution {execution_id}: {e}")
            return False
    
    def generate_cloud_run_dockerfile(self, base_image: str = 'nvidia/cuda:11.8-runtime-ubuntu20.04') -> str:
        """Generate Dockerfile for Google Cloud Run deployment."""
        dockerfile_content = f"""FROM {base_image}

# Install Python and dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install ChemiAccel and dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt

# Copy application code
COPY . /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Create entrypoint script
RUN echo '#!/bin/bash\\n\\
python3 -c "\\
import os\\n\\
from chemi_accel import Calculator\\n\\
from chemi_accel.integrations.hpc_integration import ParallelIOManager\\n\\
import pandas as pd\\n\\
\\n\\
# Get input and output URIs from environment\\n\\
input_uri = os.environ.get(\\"INPUT_DATA_URI\\")\\n\\
output_uri = os.environ.get(\\"OUTPUT_DATA_URI\\")\\n\\
\\n\\
# Process molecules\\n\\
if input_uri and output_uri:\\n\\
    # Read data from cloud storage\\n\\
    if input_uri.startswith(\\"gs://\\"):\\n\\
        import subprocess\\n\\
        subprocess.run([\\"gsutil\\", \\"cp\\", input_uri, \\"/tmp/input.csv\\"])\\n\\
        molecules = pd.read_csv(\\"/tmp/input.csv\\")[\\"smiles\\"]\\n\\
    else:\\n\\
        molecules = pd.read_csv(input_uri)[\\"smiles\\"]\\n\\
    \\n\\
    # Calculate descriptors\\n\\
    calc = Calculator(backend=\\"cuda\\")\\n\\
    results = calc.pandas(molecules)\\n\\
    \\n\\
    # Write results to cloud storage\\n\\
    if output_uri.startswith(\\"gs://\\"):\\n\\
        results.to_csv(\\"/tmp/output.csv\\", index=False)\\n\\
        subprocess.run([\\"gsutil\\", \\"cp\\", \\"/tmp/output.csv\\", output_uri])\\n\\
    else:\\n\\
        results.to_csv(output_uri, index=False)\\n\\
    \\n\\
    print(f\\"Processed {{len(results)}} molecules\\")\\n\\
else:\\n\\
    print(\\"INPUT_DATA_URI and OUTPUT_DATA_URI environment variables required\\")\\n\\
"' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
"""
        return dockerfile_content


class DaskClusterIntegration:
    """Enhanced Dask cluster integration for distributed processing."""
    
    def __init__(self, scheduler_address: Optional[str] = None):
        """Initialize Dask cluster integration."""
        if not _DASK_AVAILABLE:
            raise DistributedComputingError("Dask not available. Install dask[distributed].")
        
        self.client = None
        self.scheduler_address = scheduler_address
        
        if scheduler_address:
            self.connect_to_cluster(scheduler_address)
    
    def connect_to_cluster(self, scheduler_address: str):
        """Connect to existing Dask cluster."""
        try:
            self.client = Client(scheduler_address)
            logger.info(f"Connected to Dask cluster at {scheduler_address}")
            logger.info(f"Cluster info: {self.client}")
        except Exception as e:
            raise DistributedComputingError(f"Failed to connect to Dask cluster: {e}")
    
    def create_local_cluster(self, n_workers: int = None, 
                           threads_per_worker: int = 2,
                           memory_limit: str = '4GB'):
        """Create local Dask cluster."""
        try:
            from dask.distributed import LocalCluster
            
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit
            )
            self.client = Client(cluster)
            logger.info(f"Created local Dask cluster with {n_workers} workers")
            
        except Exception as e:
            raise DistributedComputingError(f"Failed to create local Dask cluster: {e}")
    
    def distribute_calculation(self, molecules: pd.Series, 
                             calculator_config: Dict[str, Any],
                             chunk_size: int = 1000) -> pd.DataFrame:
        """Distribute ChemiAccel calculation across Dask cluster."""
        if not self.client:
            raise DistributedComputingError("No Dask client available")
        
        # Split molecules into chunks
        chunks = [molecules.iloc[i:i+chunk_size] 
                 for i in range(0, len(molecules), chunk_size)]
        
        # Submit tasks to cluster
        futures = []
        for chunk in chunks:
            future = self.client.submit(self._calculate_chunk, 
                                      chunk, calculator_config)
            futures.append(future)
        
        # Gather results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Chunk calculation failed: {e}")
        
        # Combine results
        if results:
            combined_results = pd.concat(results, ignore_index=True)
            logger.info(f"Distributed calculation completed: {len(combined_results)} molecules")
            return combined_results
        else:
            raise DistributedComputingError("All chunk calculations failed")
    
    @staticmethod
    def _calculate_chunk(molecules: pd.Series, 
                        calculator_config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate descriptors for a chunk of molecules."""
        calculator = Calculator(**calculator_config)
        return calculator.pandas(molecules)
    
    def close(self):
        """Close Dask client connection."""
        if self.client:
            self.client.close()
            logger.info("Closed Dask client connection")


class RayClusterIntegration:
    """Ray cluster integration for distributed processing."""
    
    def __init__(self, ray_address: Optional[str] = None):
        """Initialize Ray cluster integration."""
        if not _RAY_AVAILABLE:
            raise DistributedComputingError("Ray not available. Install ray.")
        
        self.ray_address = ray_address
        self.initialized = False
        
        if ray_address:
            self.connect_to_cluster(ray_address)
        else:
            self.init_local_ray()
    
    def connect_to_cluster(self, ray_address: str):
        """Connect to existing Ray cluster."""
        try:
            ray.init(address=ray_address)
            self.initialized = True
            logger.info(f"Connected to Ray cluster at {ray_address}")
            logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
        except Exception as e:
            raise DistributedComputingError(f"Failed to connect to Ray cluster: {e}")
    
    def init_local_ray(self, num_cpus: Optional[int] = None,
                      num_gpus: Optional[int] = None):
        """Initialize local Ray instance."""
        try:
            ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
            self.initialized = True
            logger.info("Initialized local Ray instance")
            logger.info(f"Ray resources: {ray.cluster_resources()}")
        except Exception as e:
            raise DistributedComputingError(f"Failed to initialize Ray: {e}")
    
    def distribute_calculation(self, molecules: pd.Series,
                             calculator_config: Dict[str, Any],
                             chunk_size: int = 1000) -> pd.DataFrame:
        """Distribute ChemiAccel calculation across Ray cluster."""
        if not self.initialized:
            raise DistributedComputingError("Ray not initialized")
        
        # Split molecules into chunks
        chunks = [molecules.iloc[i:i+chunk_size] 
                 for i in range(0, len(molecules), chunk_size)]
        
        # Submit tasks to Ray cluster
        futures = []
        for chunk in chunks:
            future = self._calculate_chunk_remote.remote(chunk, calculator_config)
            futures.append(future)
        
        # Gather results
        try:
            results = ray.get(futures)
            combined_results = pd.concat(results, ignore_index=True)
            logger.info(f"Ray distributed calculation completed: {len(combined_results)} molecules")
            return combined_results
        except Exception as e:
            raise DistributedComputingError(f"Ray calculation failed: {e}")
    
    def _calculate_chunk_remote(molecules: pd.Series, 
                              calculator_config: Dict[str, Any]) -> pd.DataFrame:
        """Ray remote function for calculating descriptors."""
        calculator = Calculator(**calculator_config)
        return calculator.pandas(molecules)
    
    # Apply Ray decorator only if Ray is available
    if _RAY_AVAILABLE:
        _calculate_chunk_remote = ray.remote(_calculate_chunk_remote)
    
    def shutdown(self):
        """Shutdown Ray."""
        if self.initialized:
            ray.shutdown()
            self.initialized = False
            logger.info("Ray shutdown completed")


class ParallelIOManager:
    """Parallel I/O for high-performance file systems."""
    
    def __init__(self, file_system: str = 'lustre'):
        """Initialize parallel I/O manager."""
        self.file_system = file_system
        self.mpi_integration = None
        
        if _MPI_AVAILABLE:
            try:
                self.mpi_integration = MPIIntegration()
            except:
                pass
    
    def parallel_read_csv(self, file_path: str, 
                         chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Read CSV file in parallel across MPI ranks."""
        if not self.mpi_integration:
            return pd.read_csv(file_path)
        
        if self.mpi_integration.config.is_root:
            # Root process determines file size and chunks
            file_size = os.path.getsize(file_path)
            
            # Read header to determine structure
            header_df = pd.read_csv(file_path, nrows=1)
            
            # Estimate rows per chunk
            if chunk_size is None:
                total_lines = sum(1 for _ in open(file_path)) - 1  # Subtract header
                chunk_size = total_lines // self.mpi_integration.config.world_size
            
            chunk_info = {
                'chunk_size': chunk_size,
                'columns': header_df.columns.tolist(),
                'total_lines': total_lines
            }
        else:
            chunk_info = None
        
        # Broadcast chunk information
        chunk_info = self.mpi_integration.broadcast_config(chunk_info)
        
        # Each rank reads its portion
        skip_rows = self.mpi_integration.config.rank * chunk_info['chunk_size'] + 1
        nrows = chunk_info['chunk_size']
        
        try:
            local_df = pd.read_csv(file_path, skiprows=skip_rows, nrows=nrows,
                                 names=chunk_info['columns'])
            logger.info(f"Rank {self.mpi_integration.config.rank}: "
                       f"read {len(local_df)} rows")
            return local_df
        except Exception as e:
            logger.error(f"Rank {self.mpi_integration.config.rank}: "
                        f"failed to read file chunk: {e}")
            return pd.DataFrame()
    
    def parallel_write_csv(self, df: pd.DataFrame, 
                          output_path: str,
                          gather_results: bool = True) -> bool:
        """Write DataFrame to CSV in parallel."""
        if not self.mpi_integration:
            df.to_csv(output_path, index=False)
            return True
        
        if gather_results:
            # Gather all results to root and write
            combined_df = self.mpi_integration.gather_results(df)
            if self.mpi_integration.config.is_root and combined_df is not None:
                combined_df.to_csv(output_path, index=False)
                logger.info(f"Wrote {len(combined_df)} rows to {output_path}")
                return True
        else:
            # Each rank writes its own file
            rank_output = f"{output_path}.rank_{self.mpi_integration.config.rank}"
            df.to_csv(rank_output, index=False)
            logger.info(f"Rank {self.mpi_integration.config.rank}: "
                       f"wrote {len(df)} rows to {rank_output}")
            return True
        
        return False
    
    def optimize_for_filesystem(self, file_path: str) -> Dict[str, Any]:
        """Optimize I/O parameters for specific file system."""
        optimizations = {
            'lustre': {
                'stripe_count': 4,
                'stripe_size': '1M',
                'buffer_size': 1024*1024,
                'direct_io': True,
                'prefetch_threads': 8
            },
            'gpfs': {
                'block_size': '1M',
                'buffer_size': 2*1024*1024,
                'prefetch_threads': 4,
                'write_behind': True
            },
            'nfs': {
                'buffer_size': 64*1024,
                'cache_size': '256M'
            },
            'beegfs': {
                'stripe_count': 8,
                'chunk_size': '512K',
                'buffer_size': 2*1024*1024
            },
            'ceph': {
                'object_size': '4M',
                'stripe_unit': '4M',
                'buffer_size': 4*1024*1024
            }
        }
        
        return optimizations.get(self.file_system, {'buffer_size': 64*1024})
    
    def setup_lustre_striping(self, directory_path: str, 
                             stripe_count: int = 4,
                             stripe_size: str = '1M') -> bool:
        """Setup Lustre file system striping for optimal performance."""
        if self.file_system != 'lustre':
            return False
        
        try:
            # Check if lfs command is available
            subprocess.run(['lfs', '--version'], capture_output=True, check=True)
            
            # Set stripe parameters for directory
            cmd = ['lfs', 'setstripe', '-c', str(stripe_count), '-S', stripe_size, directory_path]
            subprocess.run(cmd, check=True)
            
            logger.info(f"Set Lustre striping: count={stripe_count}, size={stripe_size} for {directory_path}")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to set Lustre striping: {e}")
            return False
    
    def setup_gpfs_prefetch(self, file_path: str, prefetch_threads: int = 4) -> bool:
        """Setup GPFS prefetch optimization."""
        if self.file_system != 'gpfs':
            return False
        
        try:
            # Set GPFS prefetch parameters
            cmd = ['mmchattr', '-P', str(prefetch_threads), file_path]
            subprocess.run(cmd, check=True)
            
            logger.info(f"Set GPFS prefetch threads: {prefetch_threads} for {file_path}")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to set GPFS prefetch: {e}")
            return False
    
    def parallel_read_hdf5(self, file_path: str, dataset_name: str = 'molecules') -> pd.DataFrame:
        """Read HDF5 file in parallel using MPI-IO."""
        try:
            import h5py
            
            if self.mpi_integration and self.mpi_integration.config.world_size > 1:
                # Use parallel HDF5 with MPI
                with h5py.File(file_path, 'r', driver='mpio', 
                              comm=self.mpi_integration.comm) as f:
                    dataset = f[dataset_name]
                    
                    # Calculate chunk for this rank
                    total_size = dataset.shape[0]
                    chunk_size = total_size // self.mpi_integration.config.world_size
                    start_idx = self.mpi_integration.config.rank * chunk_size
                    
                    if self.mpi_integration.config.rank == self.mpi_integration.config.world_size - 1:
                        end_idx = total_size  # Last rank gets remainder
                    else:
                        end_idx = start_idx + chunk_size
                    
                    # Read chunk
                    chunk_data = dataset[start_idx:end_idx]
                    
                    # Convert to DataFrame
                    if len(chunk_data.shape) == 1:
                        df = pd.DataFrame({'smiles': chunk_data})
                    else:
                        df = pd.DataFrame(chunk_data)
                    
                    logger.info(f"Rank {self.mpi_integration.config.rank}: "
                               f"read {len(df)} rows from HDF5")
                    return df
            else:
                # Regular HDF5 read
                with h5py.File(file_path, 'r') as f:
                    data = f[dataset_name][:]
                    if len(data.shape) == 1:
                        return pd.DataFrame({'smiles': data})
                    else:
                        return pd.DataFrame(data)
                        
        except ImportError:
            logger.error("h5py not available for HDF5 support")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to read HDF5 file: {e}")
            return pd.DataFrame()
    
    def parallel_write_hdf5(self, df: pd.DataFrame, file_path: str, 
                           dataset_name: str = 'molecules',
                           compression: str = 'gzip') -> bool:
        """Write DataFrame to HDF5 file in parallel."""
        try:
            import h5py
            
            if self.mpi_integration and self.mpi_integration.config.world_size > 1:
                # Gather data to root for writing
                all_data = self.mpi_integration.gather_results(df)
                
                if self.mpi_integration.config.is_root and all_data is not None:
                    with h5py.File(file_path, 'w') as f:
                        # Convert DataFrame to appropriate format
                        if len(all_data.columns) == 1 and 'smiles' in all_data.columns:
                            data = all_data['smiles'].values.astype('S')
                        else:
                            data = all_data.values
                        
                        f.create_dataset(dataset_name, data=data, compression=compression)
                    
                    logger.info(f"Wrote {len(all_data)} rows to HDF5 file {file_path}")
                    return True
            else:
                # Regular HDF5 write
                with h5py.File(file_path, 'w') as f:
                    if len(df.columns) == 1 and 'smiles' in df.columns:
                        data = df['smiles'].values.astype('S')
                    else:
                        data = df.values
                    
                    f.create_dataset(dataset_name, data=data, compression=compression)
                
                logger.info(f"Wrote {len(df)} rows to HDF5 file {file_path}")
                return True
                
        except ImportError:
            logger.error("h5py not available for HDF5 support")
            return False
        except Exception as e:
            logger.error(f"Failed to write HDF5 file: {e}")
            return False
        
        return False


class HPCCalculator(Calculator):
    """HPC-aware Calculator with distributed computing capabilities."""
    
    def __init__(self, backend='auto', device=None, batch_size='auto',
                 use_mpi: bool = False,
                 dask_scheduler: Optional[str] = None,
                 ray_address: Optional[str] = None,
                 parallel_io_filesystem: str = 'lustre'):
        """
        Initialize HPC Calculator.
        
        Args:
            backend: Backend type ('cuda', 'jax', 'cpu')
            device: Device specification
            batch_size: Batch size for processing
            use_mpi: Enable MPI distribution
            dask_scheduler: Dask scheduler address
            ray_address: Ray cluster address
            parallel_io_filesystem: Filesystem type for parallel I/O
        """
        super().__init__(backend=backend, device=device, batch_size=batch_size)
        
        self.use_mpi = use_mpi
        self.dask_scheduler = dask_scheduler
        self.ray_address = ray_address
        
        # Initialize distributed computing integrations
        self.mpi_integration = None
        self.dask_integration = None
        self.ray_integration = None
        self.parallel_io = ParallelIOManager(file_system=parallel_io_filesystem)
        
        if self.use_mpi and _MPI_AVAILABLE:
            try:
                self.mpi_integration = MPIIntegration()
                logger.info("MPI integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MPI: {e}")
        
        if self.dask_scheduler and _DASK_AVAILABLE:
            try:
                self.dask_integration = DaskClusterIntegration(self.dask_scheduler)
                logger.info("Dask integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Dask: {e}")
        
        if self.ray_address and _RAY_AVAILABLE:
            try:
                self.ray_integration = RayClusterIntegration(self.ray_address)
                logger.info("Ray integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Ray: {e}")
    
    def pandas(self, molecules: pd.Series) -> pd.DataFrame:
        """Calculate descriptors with HPC distribution."""
        # Use MPI distribution if available
        if self.mpi_integration:
            local_molecules = self.mpi_integration.distribute_workload(molecules)
            local_results = super().pandas(local_molecules)
            combined_results = self.mpi_integration.gather_results(local_results)
            
            if combined_results is not None:
                return combined_results
            else:
                # Non-root processes return empty DataFrame
                return pd.DataFrame()
        
        # Use Dask distribution if available
        elif self.dask_integration:
            calculator_config = {
                'backend': self.backend_type.value,
                'device': self.device,
                'batch_size': self.batch_size
            }
            return self.dask_integration.distribute_calculation(
                molecules, calculator_config)
        
        # Use Ray distribution if available
        elif self.ray_integration:
            calculator_config = {
                'backend': self.backend_type.value,
                'device': self.device,
                'batch_size': self.batch_size
            }
            return self.ray_integration.distribute_calculation(
                molecules, calculator_config)
        
        # Fall back to standard calculation
        else:
            return super().pandas(molecules)
    
    def __del__(self):
        """Cleanup distributed computing resources."""
        if self.dask_integration:
            self.dask_integration.close()
        if self.ray_integration:
            self.ray_integration.shutdown()


# Utility functions for HPC integration
def detect_hpc_environment() -> Dict[str, Any]:
    """Detect HPC environment and available resources."""
    environment = {
        'scheduler': None,
        'mpi_available': _MPI_AVAILABLE,
        'dask_available': _DASK_AVAILABLE,
        'ray_available': _RAY_AVAILABLE,
        'slurm_available': False,
        'pbs_available': False,
        'node_count': 1,
        'gpu_count': 0
    }
    
    # Check for SLURM
    if 'SLURM_JOB_ID' in os.environ:
        environment['scheduler'] = 'slurm'
        environment['slurm_available'] = True
        environment['node_count'] = int(os.environ.get('SLURM_NNODES', 1))
        environment['job_id'] = os.environ.get('SLURM_JOB_ID')
    
    # Check for PBS
    elif 'PBS_JOBID' in os.environ:
        environment['scheduler'] = 'pbs'
        environment['pbs_available'] = True
        environment['job_id'] = os.environ.get('PBS_JOBID')
    
    # Detect GPUs
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
        if cuda_devices:
            environment['gpu_count'] = len(cuda_devices.split(','))
    
    # MPI world size
    if _MPI_AVAILABLE:
        try:
            comm = MPI.COMM_WORLD
            environment['mpi_world_size'] = comm.Get_size()
            environment['mpi_rank'] = comm.Get_rank()
        except:
            pass
    
    return environment


# Utility functions for HPC integration
def detect_hpc_environment() -> Dict[str, Any]:
    """Detect HPC environment and available resources."""
    environment = {
        'scheduler': None,
        'mpi_available': _MPI_AVAILABLE,
        'dask_available': _DASK_AVAILABLE,
        'ray_available': _RAY_AVAILABLE,
        'slurm_available': False,
        'pbs_available': False,
        'node_count': 1,
        'gpu_count': 0
    }
    
    # Check for SLURM
    if 'SLURM_JOB_ID' in os.environ:
        environment['scheduler'] = 'slurm'
        environment['slurm_available'] = True
        environment['node_count'] = int(os.environ.get('SLURM_NNODES', 1))
        environment['job_id'] = os.environ.get('SLURM_JOB_ID')
        environment['partition'] = os.environ.get('SLURM_JOB_PARTITION')
        environment['cpus_per_node'] = int(os.environ.get('SLURM_CPUS_ON_NODE', 1))
    
    # Check for PBS
    elif 'PBS_JOBID' in os.environ:
        environment['scheduler'] = 'pbs'
        environment['pbs_available'] = True
        environment['job_id'] = os.environ.get('PBS_JOBID')
        environment['queue'] = os.environ.get('PBS_QUEUE')
    
    # Check for SGE
    elif 'JOB_ID' in os.environ and 'SGE_ROOT' in os.environ:
        environment['scheduler'] = 'sge'
        environment['job_id'] = os.environ.get('JOB_ID')
    
    # Detect GPUs
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
        if cuda_devices and cuda_devices != '':
            environment['gpu_count'] = len(cuda_devices.split(','))
    
    # MPI world size
    if _MPI_AVAILABLE:
        try:
            comm = MPI.COMM_WORLD
            environment['mpi_world_size'] = comm.Get_size()
            environment['mpi_rank'] = comm.Get_rank()
        except:
            pass
    
    # Check for common HPC file systems
    environment['lustre_available'] = os.path.exists('/usr/bin/lfs')
    environment['gpfs_available'] = os.path.exists('/usr/lpp/mmfs/bin/mmlsfs')
    environment['beegfs_available'] = os.path.exists('/opt/beegfs/sbin/beegfs-ctl')
    
    return environment


def create_hpc_calculator(**kwargs) -> HPCCalculator:
    """Create HPC-optimized Calculator based on environment."""
    env = detect_hpc_environment()
    
    # Auto-configure based on environment
    if env['mpi_available'] and env.get('mpi_world_size', 1) > 1:
        kwargs.setdefault('use_mpi', True)
    
    # Set filesystem based on availability
    if env.get('lustre_available'):
        kwargs.setdefault('parallel_io_filesystem', 'lustre')
    elif env.get('gpfs_available'):
        kwargs.setdefault('parallel_io_filesystem', 'gpfs')
    elif env.get('beegfs_available'):
        kwargs.setdefault('parallel_io_filesystem', 'beegfs')
    
    return HPCCalculator(**kwargs)


def submit_hpc_job(scheduler: str,
                   job_config: HPCJobConfig,
                   input_path: str,
                   output_path: str,
                   descriptor_config: Dict[str, Any] = None) -> Optional[str]:
    """
    Utility function to submit HPC job for ChemiAccel processing.
    
    Args:
        scheduler: Scheduler type ('slurm', 'pbs', 'sge')
        job_config: Job configuration
        input_path: Input data file path
        output_path: Output results file path
        descriptor_config: Descriptor calculation configuration
        
    Returns:
        Job ID if successful, None otherwise
    """
    try:
        if scheduler.lower() == 'slurm':
            slurm_manager = SLURMJobManager()
            
            # Generate Python script for processing
            python_script = f"""
import sys
import pandas as pd
from chemi_accel.integrations.hpc_integration import create_hpc_calculator

def main():
    # Create HPC calculator
    calc = create_hpc_calculator(
        backend='{descriptor_config.get("backend", "cuda") if descriptor_config else "cuda"}',
        batch_size={descriptor_config.get("batch_size", 1000) if descriptor_config else 1000}
    )
    
    # Process file
    success = calc.process_file_parallel(
        input_path='{input_path}',
        output_path='{output_path}'
    )
    
    if success:
        print(f"Successfully processed molecules: {input_path} -> {output_path}")
        sys.exit(0)
    else:
        print("Processing failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
"""
            
            # Write script to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(python_script)
                script_path = f.name
            
            # Generate and submit job
            job_script = slurm_manager.generate_job_script(
                job_config, script_path, []
            )
            
            # Write job script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(job_script)
                job_script_path = f.name
            
            # Submit job
            job_id = slurm_manager.submit_job(job_script_path)
            
            logger.info(f"Submitted HPC job {job_id} for ChemiAccel processing")
            return job_id
            
        else:
            raise ValueError(f"Scheduler {scheduler} not yet supported")
            
    except Exception as e:
        logger.error(f"Failed to submit HPC job: {e}")
        return None
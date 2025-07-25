"""
HPC and distributed computing integration for ChemiAccel.

Provides MPI integration, SLURM job management, and distributed processing
capabilities for high-performance computing environments.
"""

import logging
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HPCJobConfig:
    """Configuration for HPC cluster jobs."""

    job_name: str
    nodes: int
    gpus_per_node: int = 1
    cpus_per_node: int = 8
    memory_per_node: str = "32GB"
    walltime: str = "24:00:00"
    partition: str = "gpu"
    account: str = None
    modules_to_load: List[str] = None
    environment_variables: Dict[str, str] = None
    working_directory: str = None
    output_file: str = None
    error_file: str = None


@dataclass
class MPIConfig:
    """Configuration for MPI execution."""

    world_size: int
    rank: int
    master_addr: str = "localhost"
    master_port: str = "29500"
    backend: str = "nccl"  # or "gloo"


class HPCScheduler(ABC):
    """Abstract base class for HPC schedulers."""

    @abstractmethod
    def submit_job(self, job_config: HPCJobConfig, script_content: str) -> str:
        """Submit job to scheduler and return job ID."""
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> str:
        """Get job status."""
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel job."""
        pass

    @abstractmethod
    def get_job_info(self, job_id: str) -> Dict[str, Any]:
        """Get detailed job information."""
        pass


class SLURMScheduler(HPCScheduler):
    """SLURM workload manager integration."""

    def __init__(self):
        """Initialize SLURM scheduler."""
        # Check if SLURM commands are available
        if not self._check_slurm_available():
            raise RuntimeError("SLURM commands not available on this system")

    def _check_slurm_available(self) -> bool:
        """Check if SLURM commands are available."""
        try:
            subprocess.run(["sinfo", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def submit_job(self, job_config: HPCJobConfig, script_content: str) -> str:
        """Submit job to SLURM."""
        # Create job script
        job_script = self._create_slurm_script(job_config, script_content)

        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(job_script)
            script_path = f.name

        try:
            # Submit job
            result = subprocess.run(
                ["sbatch", script_path], capture_output=True, text=True, check=True
            )

            # Extract job ID from output
            output_lines = result.stdout.strip().split("\n")
            job_id = output_lines[-1].split()[-1]

            logger.info(f"Submitted SLURM job: {job_id}")
            return job_id

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit SLURM job: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise
        finally:
            # Clean up temporary script
            os.unlink(script_path)

    def get_job_status(self, job_id: str) -> str:
        """Get SLURM job status."""
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "UNKNOWN"

    def cancel_job(self, job_id: str) -> bool:
        """Cancel SLURM job."""
        try:
            subprocess.run(["scancel", job_id], check=True)
            logger.info(f"Cancelled SLURM job: {job_id}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel SLURM job {job_id}: {e}")
            return False

    def get_job_info(self, job_id: str) -> Dict[str, Any]:
        """Get detailed SLURM job information."""
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse job info
            job_info = {}
            for line in result.stdout.split("\n"):
                if "=" in line:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        job_info[key] = value

            return job_info

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get SLURM job info for {job_id}: {e}")
            return {}

    def _create_slurm_script(
        self, job_config: HPCJobConfig, script_content: str
    ) -> str:
        """Create SLURM job script."""
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_config.job_name}",
            f"#SBATCH --nodes={job_config.nodes}",
            f"#SBATCH --ntasks-per-node={job_config.cpus_per_node}",
            f"#SBATCH --gres=gpu:{job_config.gpus_per_node}",
            f"#SBATCH --mem={job_config.memory_per_node}",
            f"#SBATCH --time={job_config.walltime}",
            f"#SBATCH --partition={job_config.partition}",
        ]

        if job_config.account:
            script_lines.append(f"#SBATCH --account={job_config.account}")

        if job_config.output_file:
            script_lines.append(f"#SBATCH --output={job_config.output_file}")

        if job_config.error_file:
            script_lines.append(f"#SBATCH --error={job_config.error_file}")

        script_lines.extend(
            [
                "",
                "# Load modules",
            ]
        )

        if job_config.modules_to_load:
            for module in job_config.modules_to_load:
                script_lines.append(f"module load {module}")

        script_lines.extend(
            [
                "",
                "# Set environment variables",
            ]
        )

        if job_config.environment_variables:
            for key, value in job_config.environment_variables.items():
                script_lines.append(f"export {key}={value}")

        if job_config.working_directory:
            script_lines.extend(
                [
                    "",
                    f"cd {job_config.working_directory}",
                ]
            )

        script_lines.extend(["", "# Job execution", script_content])

        return "\n".join(script_lines)


class PBSScheduler(HPCScheduler):
    """PBS/Torque scheduler integration."""

    def __init__(self):
        """Initialize PBS scheduler."""
        if not self._check_pbs_available():
            raise RuntimeError("PBS commands not available on this system")

    def _check_pbs_available(self) -> bool:
        """Check if PBS commands are available."""
        try:
            subprocess.run(["qstat", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def submit_job(self, job_config: HPCJobConfig, script_content: str) -> str:
        """Submit job to PBS."""
        job_script = self._create_pbs_script(job_config, script_content)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pbs", delete=False) as f:
            f.write(job_script)
            script_path = f.name

        try:
            result = subprocess.run(
                ["qsub", script_path], capture_output=True, text=True, check=True
            )

            job_id = result.stdout.strip()
            logger.info(f"Submitted PBS job: {job_id}")
            return job_id

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit PBS job: {e}")
            raise
        finally:
            os.unlink(script_path)

    def get_job_status(self, job_id: str) -> str:
        """Get PBS job status."""
        try:
            result = subprocess.run(
                ["qstat", "-f", job_id], capture_output=True, text=True, check=True
            )

            # Parse status from output
            for line in result.stdout.split("\n"):
                if "job_state" in line:
                    return line.split("=")[-1].strip()

            return "UNKNOWN"
        except subprocess.CalledProcessError:
            return "UNKNOWN"

    def cancel_job(self, job_id: str) -> bool:
        """Cancel PBS job."""
        try:
            subprocess.run(["qdel", job_id], check=True)
            logger.info(f"Cancelled PBS job: {job_id}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel PBS job {job_id}: {e}")
            return False

    def get_job_info(self, job_id: str) -> Dict[str, Any]:
        """Get detailed PBS job information."""
        try:
            result = subprocess.run(
                ["qstat", "-f", job_id], capture_output=True, text=True, check=True
            )

            job_info = {}
            for line in result.stdout.split("\n"):
                if "=" in line and not line.startswith("\t"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        job_info[key] = value

            return job_info

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get PBS job info for {job_id}: {e}")
            return {}

    def _create_pbs_script(self, job_config: HPCJobConfig, script_content: str) -> str:
        """Create PBS job script."""
        script_lines = [
            "#!/bin/bash",
            f"#PBS -N {job_config.job_name}",
            f"#PBS -l nodes={job_config.nodes}:ppn={job_config.cpus_per_node}:gpus={job_config.gpus_per_node}",
            f"#PBS -l mem={job_config.memory_per_node}",
            f"#PBS -l walltime={job_config.walltime}",
            f"#PBS -q {job_config.partition}",
        ]

        if job_config.account:
            script_lines.append(f"#PBS -A {job_config.account}")

        if job_config.output_file:
            script_lines.append(f"#PBS -o {job_config.output_file}")

        if job_config.error_file:
            script_lines.append(f"#PBS -e {job_config.error_file}")

        script_lines.extend(
            [
                "",
                "# Change to working directory",
                "cd $PBS_O_WORKDIR",
                "",
                "# Load modules",
            ]
        )

        if job_config.modules_to_load:
            for module in job_config.modules_to_load:
                script_lines.append(f"module load {module}")

        script_lines.extend(
            [
                "",
                "# Set environment variables",
            ]
        )

        if job_config.environment_variables:
            for key, value in job_config.environment_variables.items():
                script_lines.append(f"export {key}={value}")

        script_lines.extend(["", "# Job execution", script_content])

        return "\n".join(script_lines)


class MPIIntegration:
    """MPI integration for distributed ChemiAccel processing."""

    def __init__(self):
        """Initialize MPI integration."""
        self.mpi_config = self._detect_mpi_environment()
        self.is_mpi_available = self._check_mpi_available()

    def _check_mpi_available(self) -> bool:
        """Check if MPI is available."""
        try:
            subprocess.run(["mpirun", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _detect_mpi_environment(self) -> MPIConfig:
        """Detect MPI environment configuration."""
        world_size = int(
            os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1"))
        )
        rank = int(
            os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0"))
        )

        return MPIConfig(
            world_size=world_size,
            rank=rank,
            master_addr=os.environ.get("MASTER_ADDR", "localhost"),
            master_port=os.environ.get("MASTER_PORT", "29500"),
        )

    def distribute_workload(self, molecules: pd.Series) -> pd.Series:
        """
        Distribute molecular dataset across MPI ranks.

        Args:
            molecules: Series of SMILES strings

        Returns:
            Subset of molecules for current rank
        """
        if not self.is_mpi_available or self.mpi_config.world_size == 1:
            return molecules

        # Calculate chunk size for current rank
        total_molecules = len(molecules)
        chunk_size = total_molecules // self.mpi_config.world_size
        remainder = total_molecules % self.mpi_config.world_size

        # Calculate start and end indices for current rank
        start_idx = self.mpi_config.rank * chunk_size
        if self.mpi_config.rank < remainder:
            start_idx += self.mpi_config.rank
            chunk_size += 1
        else:
            start_idx += remainder

        end_idx = start_idx + chunk_size

        logger.info(
            f"MPI rank {self.mpi_config.rank}/{self.mpi_config.world_size}: "
            f"processing molecules {start_idx}-{end_idx}"
        )

        return molecules.iloc[start_idx:end_idx]

    def gather_results(self, local_results: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Gather results from all MPI ranks.

        Args:
            local_results: Results from current rank

        Returns:
            Combined results (only on rank 0)
        """
        if not self.is_mpi_available:
            return local_results

        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD

            # Gather all results to rank 0
            all_results = comm.gather(local_results, root=0)

            if self.mpi_config.rank == 0:
                # Combine results
                combined_results = pd.concat(all_results, ignore_index=True)
                logger.info(
                    f"Combined results from {self.mpi_config.world_size} ranks: "
                    f"{len(combined_results)} total molecules"
                )
                return combined_results
            else:
                return None

        except ImportError:
            logger.warning("mpi4py not available, returning local results")
            return local_results

    def barrier(self):
        """Synchronize all MPI ranks."""
        if self.is_mpi_available:
            try:
                from mpi4py import MPI

                MPI.COMM_WORLD.barrier()
            except ImportError:
                pass


class ParallelIOManager:
    """Parallel I/O manager for high-performance file systems."""

    def __init__(self, file_system: str = "lustre"):
        """
        Initialize parallel I/O manager.

        Args:
            file_system: File system type ('lustre', 'gpfs', 'nfs')
        """
        self.file_system = file_system
        self.stripe_config = self._get_stripe_config()

    def _get_stripe_config(self) -> Dict[str, Any]:
        """Get optimal striping configuration for file system."""
        if self.file_system == "lustre":
            return {"stripe_count": 4, "stripe_size": "1M"}
        elif self.file_system == "gpfs":
            return {"block_size": "1M", "prefetch_threads": 4}
        else:
            return {}

    def setup_output_directory(self, output_dir: str) -> bool:
        """
        Setup output directory with optimal striping.

        Args:
            output_dir: Output directory path

        Returns:
            Success status
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            if self.file_system == "lustre":
                # Set Lustre striping
                cmd = [
                    "lfs",
                    "setstripe",
                    "-c",
                    str(self.stripe_config["stripe_count"]),
                    "-S",
                    self.stripe_config["stripe_size"],
                    output_dir,
                ]
                subprocess.run(cmd, check=True)
                logger.info(f"Set Lustre striping for {output_dir}")

            return True

        except Exception as e:
            logger.error(f"Failed to setup output directory {output_dir}: {e}")
            return False

    def parallel_write(
        self, data: pd.DataFrame, output_path: str, format: str = "parquet"
    ) -> bool:
        """
        Write data using parallel I/O strategies.

        Args:
            data: DataFrame to write
            output_path: Output file path
            format: Output format ('parquet', 'csv', 'hdf5')

        Returns:
            Success status
        """
        try:
            if format == "parquet":
                # Use pyarrow for parallel writing
                data.to_parquet(output_path, index=False, engine="pyarrow")
            elif format == "csv":
                data.to_csv(output_path, index=False)
            elif format == "hdf5":
                data.to_hdf(output_path, key="data", mode="w", format="table")
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Wrote {len(data)} rows to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write data to {output_path}: {e}")
            return False

    def parallel_read(
        self, input_path: str, chunk_size: int = 10000
    ) -> Iterator[pd.DataFrame]:
        """
        Read data using parallel I/O strategies.

        Args:
            input_path: Input file path
            chunk_size: Chunk size for reading

        Yields:
            DataFrame chunks
        """
        try:
            file_ext = Path(input_path).suffix.lower()

            if file_ext == ".parquet":
                # Read parquet in chunks
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(input_path)

                for batch in parquet_file.iter_batches(batch_size=chunk_size):
                    yield batch.to_pandas()

            elif file_ext == ".csv":
                # Read CSV in chunks
                for chunk in pd.read_csv(input_path, chunksize=chunk_size):
                    yield chunk

            elif file_ext == ".hdf5":
                # Read HDF5 in chunks
                with pd.HDFStore(input_path, "r") as store:
                    total_rows = store.get_storer("data").nrows

                    for start in range(0, total_rows, chunk_size):
                        end = min(start + chunk_size, total_rows)
                        chunk = store.select("data", start=start, stop=end)
                        yield chunk

            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

        except Exception as e:
            logger.error(f"Failed to read data from {input_path}: {e}")
            raise


class HPCIntegration:
    """Main HPC integration class for ChemiAccel."""

    def __init__(self, scheduler: str = "auto"):
        """
        Initialize HPC integration.

        Args:
            scheduler: HPC scheduler ('slurm', 'pbs', 'auto')
        """
        self.scheduler = self._create_scheduler(scheduler)
        self.mpi = MPIIntegration()
        self.io_manager = ParallelIOManager()

    def _create_scheduler(self, scheduler: str) -> HPCScheduler:
        """Create appropriate scheduler instance."""
        if scheduler == "auto":
            # Auto-detect scheduler
            if SLURMScheduler()._check_slurm_available():
                return SLURMScheduler()
            elif PBSScheduler()._check_pbs_available():
                return PBSScheduler()
            else:
                raise RuntimeError("No supported HPC scheduler found")
        elif scheduler == "slurm":
            return SLURMScheduler()
        elif scheduler == "pbs":
            return PBSScheduler()
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

    def submit_chemi_accel_job(
        self,
        job_config: HPCJobConfig,
        input_data_path: str,
        output_data_path: str,
        descriptor_config: Dict[str, Any] = None,
    ) -> str:
        """
        Submit ChemiAccel processing job to HPC cluster.

        Args:
            job_config: HPC job configuration
            input_data_path: Path to input molecular data
            output_data_path: Path for output results
            descriptor_config: ChemiAccel descriptor configuration

        Returns:
            Job ID
        """
        # Create processing script
        script_content = self._create_hpc_processing_script(
            input_data_path, output_data_path, descriptor_config
        )

        # Submit job
        return self.scheduler.submit_job(job_config, script_content)

    def _create_hpc_processing_script(
        self, input_path: str, output_path: str, descriptor_config: Dict[str, Any]
    ) -> str:
        """Create HPC processing script."""
        script_template = f"""
# MPI execution
mpirun -np $SLURM_NTASKS python3 << 'EOF'
import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Import ChemiAccel
        from chemi_accel import Calculator
        from chemi_accel.cloud.hpc_integration import MPIIntegration, ParallelIOManager

        # Initialize components
        calc = Calculator(backend='cuda', batch_size=1000)
        mpi = MPIIntegration()
        io_manager = ParallelIOManager()

        logger.info(f"MPI rank {{mpi.mpi_config.rank}}/{{mpi.mpi_config.world_size}}")

        # Setup output directory (rank 0 only)
        if mpi.mpi_config.rank == 0:
            io_manager.setup_output_directory(os.path.dirname("{output_path}"))

        mpi.barrier()

        # Load and distribute data
        if mpi.mpi_config.rank == 0:
            logger.info("Loading molecular data...")

            # Read data in chunks to handle large files
            all_molecules = []
            for chunk in io_manager.parallel_read("{input_path}"):
                if 'smiles' in chunk.columns:
                    all_molecules.extend(chunk['smiles'].tolist())

            molecules_series = pd.Series(all_molecules)
            logger.info(f"Loaded {{len(molecules_series)}} molecules")
        else:
            molecules_series = pd.Series([])

        # Distribute workload across MPI ranks
        local_molecules = mpi.distribute_workload(molecules_series)

        if len(local_molecules) > 0:
            logger.info(f"Rank {{mpi.mpi_config.rank}}: Processing {{len(local_molecules)}} molecules")

            # Calculate descriptors
            descriptors = calc.pandas(local_molecules)

            # Create results DataFrame
            local_results = pd.DataFrame({{'smiles': local_molecules}})
            local_results = pd.concat([local_results, descriptors], axis=1)
        else:
            local_results = pd.DataFrame()

        # Gather results
        combined_results = mpi.gather_results(local_results)

        # Save results (rank 0 only)
        if mpi.mpi_config.rank == 0 and combined_results is not None:
            logger.info(f"Saving {{len(combined_results)}} results to {output_path}")
            io_manager.parallel_write(combined_results, "{output_path}", format='parquet')
            logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF
"""
        return script_template

    def monitor_job(self, job_id: str) -> str:
        """Monitor HPC job status."""
        return self.scheduler.get_job_status(job_id)

    def wait_for_completion(self, job_id: str, poll_interval: int = 60) -> str:
        """
        Wait for job completion.

        Args:
            job_id: Job ID to monitor
            poll_interval: Polling interval in seconds

        Returns:
            Final job status
        """
        while True:
            status = self.monitor_job(job_id)

            if status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                logger.info(f"Job {job_id} finished with status: {status}")
                return status

            logger.info(f"Job {job_id} status: {status}")
            time.sleep(poll_interval)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel HPC job."""
        return self.scheduler.cancel_job(job_id)

    def get_job_info(self, job_id: str) -> Dict[str, Any]:
        """Get detailed job information."""
        return self.scheduler.get_job_info(job_id)

    def generate_slurm_script(
        self,
        job_config: HPCJobConfig,
        input_path: str,
        output_path: str,
        descriptor_config: Dict[str, Any] = None,
    ) -> str:
        """
        Generate SLURM script for ChemiAccel processing.

        Args:
            job_config: HPC job configuration
            input_path: Input data path
            output_path: Output data path
            descriptor_config: Descriptor configuration

        Returns:
            SLURM script content
        """
        script_content = self._create_hpc_processing_script(
            input_path, output_path, descriptor_config
        )

        if isinstance(self.scheduler, SLURMScheduler):
            return self.scheduler._create_slurm_script(job_config, script_content)
        else:
            raise ValueError(
                "SLURM script generation only supported for SLURM scheduler"
            )


# Utility functions
def submit_hpc_job(
    scheduler: str,
    job_config: HPCJobConfig,
    input_path: str,
    output_path: str,
    descriptor_config: Dict[str, Any] = None,
) -> str:
    """
    Convenience function to submit ChemiAccel job to HPC cluster.

    Args:
        scheduler: HPC scheduler ('slurm', 'pbs', 'auto')
        job_config: Job configuration
        input_path: Input data path
        output_path: Output data path
        descriptor_config: ChemiAccel descriptor configuration

    Returns:
        Job ID
    """
    hpc = HPCIntegration(scheduler)
    return hpc.submit_chemi_accel_job(
        job_config, input_path, output_path, descriptor_config
    )


def create_dask_cluster_config(scheduler_address: str = None) -> Dict[str, Any]:
    """
    Create Dask cluster configuration for distributed processing.

    Args:
        scheduler_address: Dask scheduler address

    Returns:
        Dask cluster configuration
    """
    return {
        "scheduler_address": scheduler_address or "tcp://localhost:8786",
        "worker_options": {
            "nthreads": 1,
            "memory_limit": "8GB",
            "resources": {"GPU": 1},
        },
        "cluster_options": {"n_workers": 4, "threads_per_worker": 1},
    }


def create_ray_cluster_config(head_node_address: str = None) -> Dict[str, Any]:
    """
    Create Ray cluster configuration for distributed processing.

    Args:
        head_node_address: Ray head node address

    Returns:
        Ray cluster configuration
    """
    return {
        "address": head_node_address or "ray://localhost:10001",
        "runtime_env": {
            "pip": ["chemi-accel"],
            "env_vars": {"CUDA_VISIBLE_DEVICES": "0"},
        },
        "resources": {"num_cpus": 8, "num_gpus": 1},
    }

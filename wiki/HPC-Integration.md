# HPC Integration

ChemiAccel provides seamless integration with High-Performance Computing (HPC) clusters, enabling massive-scale molecular descriptor calculations across thousands of nodes. This guide covers SLURM, PBS/Torque integration, and optimization strategies for HPC environments.

## Overview

HPC integration features include:
- **Job Scheduler Support**: Native SLURM, PBS/Torque, and LSF integration
- **Multi-Node Scaling**: Distributed processing across hundreds of compute nodes
- **GPU Cluster Support**: Efficient utilization of GPU-accelerated HPC systems
- **Resource Management**: Intelligent job partitioning and resource allocation
- **Fault Tolerance**: Automatic job recovery and checkpoint/restart capabilities
- **Performance Optimization**: NUMA awareness, interconnect optimization

## Quick Start

### Basic HPC Job Submission

```python
from chemi_accel.hpc import SLURMManager, HPCJobConfig
from chemi_accel import Calculator

# Initialize HPC manager
hpc = SLURMManager(
    cluster_name="my-hpc-cluster",
    partition="gpu",
    account="research-group"
)

# Configure HPC job
job_config = HPCJobConfig(
    job_name="chemiaccel-large-scale",
    nodes=10,
    tasks_per_node=4,
    cpus_per_task=8,
    gpus_per_node=2,
    time_limit="24:00:00",
    memory="64GB"
)

# Submit molecular descriptor calculation job
molecules = load_large_molecule_dataset()  # 1M+ molecules
job_id = hpc.submit_descriptor_job(
    molecules=molecules,
    descriptors="comprehensive",
    job_config=job_config,
    output_path="/scratch/user/results/"
)

print(f"Job submitted with ID: {job_id}")
```

## SLURM Integration

### Advanced SLURM Job Management

```python
from chemi_accel.hpc.slurm import SLURMJobManager, SLURMConfig
import subprocess
import time

class ChemiAccelSLURMIntegration:
    """Advanced SLURM integration for ChemiAccel."""
    
    def __init__(self, cluster_config):
        self.cluster_config = cluster_config
        self.job_manager = SLURMJobManager(cluster_config)
    
    def create_job_script(self, job_config, molecules_file, output_dir):
        """Create SLURM job script for ChemiAccel."""
        
        script_content = f'''#!/bin/bash
#SBATCH --job-name={job_config.job_name}
#SBATCH --partition={job_config.partition}
#SBATCH --account={job_config.account}
#SBATCH --nodes={job_config.nodes}
#SBATCH --ntasks-per-node={job_config.tasks_per_node}
#SBATCH --cpus-per-task={job_config.cpus_per_task}
#SBATCH --gres=gpu:{job_config.gpus_per_node}
#SBATCH --time={job_config.time_limit}
#SBATCH --mem={job_config.memory}
#SBATCH --output={output_dir}/slurm-%j.out
#SBATCH --error={output_dir}/slurm-%j.err

# Load required modules
module load cuda/11.4
module load python/3.9
module load openmpi/4.1.0

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1

# Activate virtual environment
source /home/user/chemiaccel-env/bin/activate

# Run ChemiAccel distributed calculation
srun python -m chemi_accel.hpc.distributed_runner \\
    --molecules-file {molecules_file} \\
    --descriptors comprehensive \\
    --output-dir {output_dir} \\
    --backend cuda \\
    --nodes $SLURM_NNODES \\
    --tasks-per-node $SLURM_NTASKS_PER_NODE \\
    --batch-size 10000 \\
    --checkpoint-interval 3600

# Post-processing
python -m chemi_accel.hpc.postprocess \\
    --input-dir {output_dir} \\
    --output-file {output_dir}/final_results.csv \\
    --validate-results
'''
        
        return script_content
    
    def submit_array_job(self, molecules_batches, job_config):
        """Submit SLURM array job for multiple molecule batches."""
        
        array_size = len(molecules_batches)
        
        array_script = f'''#!/bin/bash
#SBATCH --job-name={job_config.job_name}-array
#SBATCH --partition={job_config.partition}
#SBATCH --account={job_config.account}
#SBATCH --array=0-{array_size-1}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={job_config.tasks_per_node}
#SBATCH --cpus-per-task={job_config.cpus_per_task}
#SBATCH --gres=gpu:{job_config.gpus_per_node}
#SBATCH --time={job_config.time_limit}
#SBATCH --mem={job_config.memory}
#SBATCH --output=array-%A_%a.out
#SBATCH --error=array-%A_%a.err

# Load modules and environment
module load cuda/11.4 python/3.9
source /home/user/chemiaccel-env/bin/activate

# Get array task ID
TASK_ID=$SLURM_ARRAY_TASK_ID

# Process specific batch
python -m chemi_accel.hpc.array_processor \\
    --batch-id $TASK_ID \\
    --molecules-dir /scratch/molecules/batches/ \\
    --output-dir /scratch/results/batch_$TASK_ID/ \\
    --descriptors comprehensive \\
    --backend cuda
'''
        
        # Write script to file
        script_file = f"/tmp/chemiaccel_array_{int(time.time())}.sh"
        with open(script_file, 'w') as f:
            f.write(array_script)
        
        # Submit array job
        cmd = ["sbatch", script_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            return job_id
        else:
            raise RuntimeError(f"Job submission failed: {result.stderr}")
    
    def submit_dependent_job_chain(self, job_configs):
        """Submit chain of dependent jobs."""
        
        job_ids = []
        dependency_string = ""
        
        for i, config in enumerate(job_configs):
            script_content = self.create_job_script(
                config, 
                config.molecules_file,
                config.output_dir
            )
            
            script_file = f"/tmp/job_{i}_{int(time.time())}.sh"
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # Prepare submission command
            cmd = ["sbatch"]
            
            # Add dependency if not first job
            if job_ids:
                dependency_string = f"afterok:{':'.join(job_ids)}"
                cmd.extend(["--dependency", dependency_string])
            
            cmd.append(script_file)
            
            # Submit job
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
                print(f"Job {i+1} submitted with ID: {job_id}")
            else:
                raise RuntimeError(f"Job {i+1} submission failed: {result.stderr}")
        
        return job_ids
    
    def monitor_job_progress(self, job_id, check_interval=60):
        """Monitor job progress and performance metrics."""
        
        monitoring_data = []
        
        while True:
            # Check job status
            status_cmd = ["squeue", "-j", str(job_id), "--format=%T,%r,%M,%l", "--noheader"]
            status_result = subprocess.run(status_cmd, capture_output=True, text=True)
            
            if status_result.returncode != 0 or not status_result.stdout.strip():
                print(f"Job {job_id} completed or not found")
                break
            
            status_info = status_result.stdout.strip().split(',')
            job_state = status_info[0]
            
            if job_state in ['COMPLETED', 'FAILED', 'CANCELLED']:
                print(f"Job {job_id} finished with state: {job_state}")
                break
            
            # Get detailed job information
            sstat_cmd = ["sstat", "-j", str(job_id), "--format=JobID,MaxRSS,AveCPU,AveVMSize", "--parsable2"]
            sstat_result = subprocess.run(sstat_cmd, capture_output=True, text=True)
            
            if sstat_result.returncode == 0:
                # Parse performance metrics
                lines = sstat_result.stdout.strip().split('\n')
                if len(lines) > 1:
                    metrics = lines[1].split('|')
                    monitoring_data.append({
                        'timestamp': time.time(),
                        'job_id': job_id,
                        'max_rss': metrics[1],
                        'avg_cpu': metrics[2],
                        'avg_vmsize': metrics[3]
                    })
            
            print(f"Job {job_id} status: {job_state}")
            time.sleep(check_interval)
        
        return monitoring_data

# Example: Multi-stage HPC workflow
slurm_integration = ChemiAccelSLURMIntegration({
    "cluster_name": "supercomputer",
    "default_partition": "gpu",
    "default_account": "chemistry-dept"
})

# Stage 1: Data preprocessing
prep_config = HPCJobConfig(
    job_name="chemiaccel-prep",
    partition="cpu",
    nodes=1,
    tasks_per_node=8,
    cpus_per_task=4,
    time_limit="02:00:00",
    memory="32GB",
    molecules_file="/data/raw_molecules.smi",
    output_dir="/scratch/preprocessed/"
)

# Stage 2: Descriptor calculation
calc_config = HPCJobConfig(
    job_name="chemiaccel-calc",
    partition="gpu",
    nodes=20,
    tasks_per_node=2,
    cpus_per_task=16,
    gpus_per_node=4,
    time_limit="12:00:00",
    memory="128GB",
    molecules_file="/scratch/preprocessed/molecules.csv",
    output_dir="/scratch/descriptors/"
)

# Stage 3: Results aggregation
agg_config = HPCJobConfig(
    job_name="chemiaccel-aggregate",
    partition="cpu",
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=32,
    time_limit="01:00:00",
    memory="256GB",
    molecules_file="",  # Not needed for aggregation
    output_dir="/scratch/final_results/"
)

# Submit dependent job chain
job_configs = [prep_config, calc_config, agg_config]
job_chain_ids = slurm_integration.submit_dependent_job_chain(job_configs)

print(f"Job chain submitted: {job_chain_ids}")

# Monitor main calculation job
monitoring_data = slurm_integration.monitor_job_progress(job_chain_ids[1])
```

### GPU Cluster Optimization

```python
from chemi_accel.hpc import GPUClusterOptimizer

class HPCGPUOptimization:
    """GPU optimization for HPC clusters."""
    
    def __init__(self, cluster_info):
        self.cluster_info = cluster_info
        self.gpu_optimizer = GPUClusterOptimizer()
    
    def optimize_gpu_allocation(self, total_molecules, available_nodes):
        """Optimize GPU allocation across cluster nodes."""
        
        # Get cluster GPU topology
        gpu_topology = self.get_cluster_gpu_info()
        
        optimization_config = {
            'total_molecules': total_molecules,
            'available_nodes': available_nodes,
            'gpu_topology': gpu_topology,
            'interconnect_bandwidth': self.cluster_info.get('interconnect_bw', '100Gb/s'),
            'memory_per_gpu': self.cluster_info.get('gpu_memory', '40GB'),
            'target_efficiency': 0.85
        }
        
        optimal_allocation = self.gpu_optimizer.optimize_allocation(optimization_config)
        
        return optimal_allocation
    
    def create_multi_gpu_job_script(self, allocation_config):
        """Create optimized multi-GPU job script."""
        
        script = f'''#!/bin/bash
#SBATCH --job-name=chemiaccel-multi-gpu
#SBATCH --nodes={allocation_config['nodes']}
#SBATCH --ntasks-per-node={allocation_config['tasks_per_node']}
#SBATCH --gres=gpu:{allocation_config['gpus_per_node']}
#SBATCH --cpus-per-task={allocation_config['cpus_per_task']}

# GPU topology optimization
export CUDA_VISIBLE_DEVICES={allocation_config['gpu_mapping']}
export NCCL_TOPOLOGY_FILE=/etc/nccl.conf
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# Memory optimization
export CUDA_MEMORY_POOL_ENABLED=1
export CUDA_MEMORY_POOL_SIZE={allocation_config['memory_pool_size']}

# Launch distributed ChemiAccel
mpirun -np {allocation_config['total_processes']} \\
    --bind-to socket \\
    --map-by ppr:{allocation_config['processes_per_socket']}:socket \\
    python -m chemi_accel.hpc.multi_gpu_runner \\
    --molecules-file $1 \\
    --output-dir $2 \\
    --backend cuda \\
    --distributed-backend nccl \\
    --batch-size {allocation_config['optimal_batch_size']}
'''
        
        return script
    
    def get_cluster_gpu_info(self):
        """Query cluster GPU topology and capabilities."""
        
        # Query SLURM for GPU information
        cmd = ["sinfo", "-o", "%N %G %f", "--noheader"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        gpu_info = {}
        for line in result.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                nodes = parts[0]
                gpu_config = parts[1]
                features = parts[2] if len(parts) > 2 else ""
                
                gpu_info[nodes] = {
                    'gpu_config': gpu_config,
                    'features': features
                }
        
        return gpu_info

# Example: Optimize for large-scale calculation
gpu_optimizer = HPCGPUOptimization({
    'cluster_name': 'summit',
    'interconnect_bw': '200Gb/s',
    'gpu_memory': '32GB',
    'cpu_cores_per_node': 176,
    'gpus_per_node': 6
})

# Optimize allocation for 10M molecules
allocation = gpu_optimizer.optimize_gpu_allocation(
    total_molecules=10000000,
    available_nodes=100
)

print(f"Optimal allocation: {allocation}")

# Generate optimized job script
job_script = gpu_optimizer.create_multi_gpu_job_script(allocation)
```

## PBS/Torque Integration

### PBS Job Management

```python
from chemi_accel.hpc.pbs import PBSJobManager, PBSConfig

class ChemiAccelPBSIntegration:
    """PBS/Torque integration for ChemiAccel."""
    
    def __init__(self, pbs_config):
        self.pbs_config = pbs_config
        self.job_manager = PBSJobManager(pbs_config)
    
    def create_pbs_script(self, job_config, calculation_params):
        """Create PBS job script for ChemiAccel calculation."""
        
        script = f'''#!/bin/bash
#PBS -N {job_config.job_name}
#PBS -q {job_config.queue}
#PBS -l nodes={job_config.nodes}:ppn={job_config.ppn}:gpus={job_config.gpus_per_node}
#PBS -l walltime={job_config.walltime}
#PBS -l mem={job_config.memory}
#PBS -o {job_config.output_dir}/job.out
#PBS -e {job_config.output_dir}/job.err
#PBS -V

# Change to working directory
cd $PBS_O_WORKDIR

# Load modules
module load cuda/11.4
module load python3/3.9
module load openmpi/4.0.5

# Set environment variables
export OMP_NUM_THREADS={job_config.ppn}
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Create node list for MPI
cat $PBS_NODEFILE | uniq > nodelist.txt
NNODES=$(cat nodelist.txt | wc -l)

# Activate ChemiAccel environment
source /home/user/chemiaccel/bin/activate

# Run distributed calculation
mpiexec -np {job_config.nodes * job_config.tasks_per_node} \\
    -machinefile nodelist.txt \\
    python -m chemi_accel.hpc.pbs_runner \\
    --molecules-file {calculation_params['molecules_file']} \\
    --descriptors {calculation_params['descriptors']} \\
    --output-dir {job_config.output_dir} \\
    --backend cuda \\
    --batch-size {calculation_params['batch_size']} \\
    --checkpoint-dir {job_config.output_dir}/checkpoints

# Cleanup
rm -f nodelist.txt

echo "Job completed at $(date)"
'''
        
        return script
    
    def submit_parametric_job(self, base_config, parameter_sweep):
        """Submit parametric study with different descriptor sets."""
        
        job_ids = []
        
        for i, params in enumerate(parameter_sweep):
            # Modify job config for this parameter set
            job_config = base_config.copy()
            job_config.job_name = f"{base_config.job_name}_{i}"
            job_config.output_dir = f"{base_config.output_dir}/param_{i}"
            
            # Create job script
            script_content = self.create_pbs_script(job_config, params)
            
            # Write script file
            script_file = f"/tmp/pbs_job_{i}_{int(time.time())}.pbs"
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # Submit job
            cmd = ["qsub", script_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                job_id = result.stdout.strip()
                job_ids.append(job_id)
                print(f"Parameter set {i} submitted with job ID: {job_id}")
            else:
                print(f"Failed to submit parameter set {i}: {result.stderr}")
        
        return job_ids
    
    def monitor_queue_status(self, user_jobs_only=True):
        """Monitor PBS queue status and job progress."""
        
        cmd = ["qstat", "-f"]
        if user_jobs_only:
            import getpass
            username = getpass.getuser()
            cmd.extend(["-u", username])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return self.parse_qstat_output(result.stdout)
        else:
            raise RuntimeError(f"Failed to query queue status: {result.stderr}")
    
    def parse_qstat_output(self, qstat_output):
        """Parse qstat output to extract job information."""
        
        jobs = []
        current_job = {}
        
        for line in qstat_output.split('\n'):
            line = line.strip()
            
            if line.startswith('Job Id:'):
                if current_job:
                    jobs.append(current_job)
                current_job = {'job_id': line.split(':')[1].strip()}
            
            elif '=' in line and current_job:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                current_job[key] = value
        
        if current_job:
            jobs.append(current_job)
        
        return jobs

# Example: PBS parametric study
pbs_integration = ChemiAccelPBSIntegration({
    'server': 'torque.hpc.university.edu',
    'default_queue': 'gpu'
})

# Base job configuration
base_job_config = PBSConfig(
    job_name="chemiaccel_study",
    queue="gpu",
    nodes=4,
    ppn=16,
    gpus_per_node=2,
    walltime="24:00:00",
    memory="128GB",
    output_dir="/scratch/user/chemiaccel_results"
)

# Parameter sweep: different descriptor combinations
parameter_sweep = [
    {
        'molecules_file': '/data/molecules.smi',
        'descriptors': 'constitutional',
        'batch_size': 5000
    },
    {
        'molecules_file': '/data/molecules.smi',
        'descriptors': 'physicochemical',
        'batch_size': 5000
    },
    {
        'molecules_file': '/data/molecules.smi',
        'descriptors': 'fingerprints',
        'batch_size': 2000
    },
    {
        'molecules_file': '/data/molecules.smi',
        'descriptors': 'comprehensive',
        'batch_size': 1000
    }
]

# Submit parametric jobs
parametric_job_ids = pbs_integration.submit_parametric_job(
    base_job_config, 
    parameter_sweep
)

print(f"Submitted {len(parametric_job_ids)} parametric jobs")

# Monitor progress
while True:
    job_status = pbs_integration.monitor_queue_status()
    
    running_jobs = [j for j in job_status if j.get('job_state') == 'R']
    queued_jobs = [j for j in job_status if j.get('job_state') == 'Q']
    
    print(f"Running: {len(running_jobs)}, Queued: {len(queued_jobs)}")
    
    if not running_jobs and not queued_jobs:
        print("All jobs completed")
        break
    
    time.sleep(300)  # Check every 5 minutes
```

## Fault Tolerance and Checkpointing

### Checkpoint/Restart System

```python
from chemi_accel.hpc import CheckpointManager, JobRecovery

class HPCFaultTolerance:
    """Fault tolerance system for HPC calculations."""
    
    def __init__(self, checkpoint_dir="/scratch/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.recovery_manager = JobRecovery()
    
    def create_checkpoint(self, job_state, molecules_processed, results):
        """Create checkpoint of current job state."""
        
        checkpoint_data = {
            'timestamp': time.time(),
            'job_id': job_state.get('job_id'),
            'molecules_processed': molecules_processed,
            'total_molecules': job_state.get('total_molecules'),
            'current_batch': job_state.get('current_batch'),
            'descriptors_config': job_state.get('descriptors_config'),
            'backend_config': job_state.get('backend_config'),
            'partial_results': results
        }
        
        checkpoint_file = self.checkpoint_manager.save_checkpoint(
            job_state['job_id'],
            checkpoint_data
        )
        
        return checkpoint_file
    
    def restore_from_checkpoint(self, job_id):
        """Restore job state from checkpoint."""
        
        checkpoint_data = self.checkpoint_manager.load_checkpoint(job_id)
        
        if checkpoint_data:
            print(f"Restoring job {job_id} from checkpoint")
            print(f"Processed {checkpoint_data['molecules_processed']} molecules")
            print(f"Resuming from batch {checkpoint_data['current_batch']}")
            
            return checkpoint_data
        else:
            return None
    
    def setup_periodic_checkpointing(self, job_config, interval_seconds=3600):
        """Set up periodic checkpointing during job execution."""
        
        checkpoint_script = f'''
import signal
import time
import pickle
from chemi_accel.hpc import CheckpointManager

class PeriodicCheckpointer:
    def __init__(self, interval={interval_seconds}):
        self.interval = interval
        self.last_checkpoint = time.time()
        self.checkpoint_manager = CheckpointManager("{self.checkpoint_dir}")
        
        # Set up signal handler for checkpointing
        signal.signal(signal.SIGUSR1, self.checkpoint_signal_handler)
        signal.signal(signal.SIGTERM, self.termination_handler)
    
    def checkpoint_signal_handler(self, signum, frame):
        self.create_checkpoint_if_needed(force=True)
    
    def termination_handler(self, signum, frame):
        print("Received termination signal, creating final checkpoint...")
        self.create_checkpoint_if_needed(force=True)
        exit(0)
    
    def create_checkpoint_if_needed(self, force=False):
        current_time = time.time()
        if force or (current_time - self.last_checkpoint) >= self.interval:
            # Create checkpoint with current state
            self.checkpoint_manager.save_checkpoint(
                "{job_config.job_id}",
                self.get_current_state()
            )
            self.last_checkpoint = current_time
            print(f"Checkpoint created at {{current_time}}")
    
    def get_current_state(self):
        # Get current job state from global variables
        return {{
            'molecules_processed': globals().get('molecules_processed', 0),
            'current_results': globals().get('current_results', []),
            'batch_index': globals().get('batch_index', 0)
        }}

# Initialize checkpointer
checkpointer = PeriodicCheckpointer()
'''
        
        return checkpoint_script
    
    def create_recovery_job(self, failed_job_id, original_config):
        """Create recovery job for failed calculation."""
        
        checkpoint_data = self.restore_from_checkpoint(failed_job_id)
        
        if not checkpoint_data:
            print(f"No checkpoint found for job {failed_job_id}")
            return None
        
        # Modify job config for recovery
        recovery_config = original_config.copy()
        recovery_config.job_name = f"recovery_{failed_job_id}"
        recovery_config.restart_from_checkpoint = True
        recovery_config.checkpoint_file = self.checkpoint_manager.get_checkpoint_path(failed_job_id)
        
        # Calculate remaining work
        molecules_remaining = (checkpoint_data['total_molecules'] - 
                             checkpoint_data['molecules_processed'])
        
        # Adjust resources based on remaining work
        if molecules_remaining < original_config.total_molecules * 0.5:
            recovery_config.nodes = max(1, original_config.nodes // 2)
            recovery_config.time_limit = original_config.time_limit // 2
        
        recovery_script = self.create_recovery_script(recovery_config, checkpoint_data)
        
        return recovery_script, recovery_config
    
    def create_recovery_script(self, recovery_config, checkpoint_data):
        """Create job script for recovery calculation."""
        
        script = f'''#!/bin/bash
#SBATCH --job-name={recovery_config.job_name}
#SBATCH --nodes={recovery_config.nodes}
#SBATCH --time={recovery_config.time_limit}

echo "Starting recovery job for {checkpoint_data['job_id']}"
echo "Resuming from {checkpoint_data['molecules_processed']} processed molecules"

# Load checkpoint and resume calculation
python -m chemi_accel.hpc.recovery_runner \\
    --checkpoint-file {recovery_config.checkpoint_file} \\
    --output-dir {recovery_config.output_dir} \\
    --backend {checkpoint_data['backend_config']['backend']} \\
    --validate-results

echo "Recovery job completed"
'''
        
        return script

# Example: Set up fault-tolerant HPC calculation
fault_tolerance = HPCFaultTolerance("/scratch/checkpoints")

# Configure job with checkpointing
job_config = {
    'job_id': 'large_calculation_001',
    'total_molecules': 5000000,
    'descriptors_config': {'descriptors': 'comprehensive'},
    'backend_config': {'backend': 'cuda', 'batch_size': 10000}
}

# Set up periodic checkpointing (every hour)
checkpoint_script = fault_tolerance.setup_periodic_checkpointing(
    job_config, 
    interval_seconds=3600
)

# Monitor for failed jobs and create recovery jobs
def monitor_and_recover():
    while True:
        # Check for failed jobs
        failed_jobs = get_failed_job_list()  # Your implementation
        
        for failed_job in failed_jobs:
            print(f"Detected failed job: {failed_job['job_id']}")
            
            # Create recovery job
            recovery_script, recovery_config = fault_tolerance.create_recovery_job(
                failed_job['job_id'],
                failed_job['original_config']
            )
            
            if recovery_script:
                # Submit recovery job
                recovery_job_id = submit_recovery_job(recovery_script)
                print(f"Recovery job submitted: {recovery_job_id}")
        
        time.sleep(600)  # Check every 10 minutes

# Start monitoring (run in background)
monitor_and_recover()
```

## Performance Optimization

### NUMA and Memory Optimization

```python
from chemi_accel.hpc import NUMAOptimizer, MemoryManager

class HPCPerformanceOptimizer:
    """Performance optimization for HPC environments."""
    
    def __init__(self):
        self.numa_optimizer = NUMAOptimizer()
        self.memory_manager = MemoryManager()
    
    def optimize_numa_placement(self, node_config):
        """Optimize NUMA placement for ChemiAccel processes."""
        
        numa_topology = self.numa_optimizer.get_numa_topology()
        
        # Determine optimal process placement
        placement_strategy = self.numa_optimizer.calculate_optimal_placement(
            num_processes=node_config['processes_per_node'],
            num_gpus=node_config['gpus_per_node'],
            numa_topology=numa_topology
        )
        
        numa_script = f'''
# NUMA optimization for ChemiAccel
export OMP_PROC_BIND=true
export OMP_PLACES={placement_strategy['omp_places']}

# Set memory policy
numactl --membind={placement_strategy['memory_nodes']} \\
        --cpunodebind={placement_strategy['cpu_nodes']} \\
        --preferred={placement_strategy['preferred_node']} \\
        python -m chemi_accel.main $@
'''
        
        return numa_script
    
    def optimize_memory_usage(self, total_molecules, available_memory):
        """Optimize memory usage for large-scale calculations."""
        
        memory_config = self.memory_manager.calculate_optimal_memory_config(
            total_molecules=total_molecules,
            available_memory_gb=available_memory,
            descriptor_complexity='comprehensive'
        )
        
        optimization_settings = {
            'batch_size': memory_config['optimal_batch_size'],
            'memory_pool_size': memory_config['memory_pool_size'],
            'buffer_size': memory_config['io_buffer_size'],
            'streaming_enabled': memory_config['enable_streaming'],
            'compression_enabled': memory_config['enable_compression']
        }
        
        return optimization_settings
    
    def create_optimized_launcher(self, job_config, optimization_settings):
        """Create optimized job launcher script."""
        
        launcher_script = f'''#!/bin/bash

# Performance optimizations
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS={job_config['cpus_per_task']}
export OMP_STACKSIZE=64M

# Memory optimizations
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000"
export MEMKIND_HBW_NODES=1

# GPU optimizations
export CUDA_MEMORY_POOL_ENABLED=1
export CUDA_MEMORY_POOL_SIZE={optimization_settings['memory_pool_size']}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# I/O optimizations
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/dev/shm

# Launch optimized ChemiAccel
python -c "
import os
from chemi_accel import Calculator
from chemi_accel.hpc import OptimizedRunner

# Configure optimized calculator
calc_config = {{
    'backend': 'cuda',
    'batch_size': {optimization_settings['batch_size']},
    'memory_pool_size': '{optimization_settings['memory_pool_size']}',
    'enable_streaming': {optimization_settings['streaming_enabled']},
    'enable_compression': {optimization_settings['compression_enabled']}
}}

runner = OptimizedRunner(calc_config)
runner.run_calculation(
    molecules_file=os.environ['MOLECULES_FILE'],
    output_dir=os.environ['OUTPUT_DIR'],
    descriptors=os.environ.get('DESCRIPTORS', 'comprehensive')
)
"
'''
        
        return launcher_script

# Example: Create optimized HPC job
optimizer = HPCPerformanceOptimizer()

# Node configuration
node_config = {
    'processes_per_node': 4,
    'gpus_per_node': 4,
    'cpus_per_task': 16,
    'memory_per_node': '256GB'
}

# Optimize NUMA placement
numa_script = optimizer.optimize_numa_placement(node_config)

# Optimize memory usage for 10M molecules with 256GB RAM
memory_settings = optimizer.optimize_memory_usage(
    total_molecules=10000000,
    available_memory=256
)

print(f"Optimal batch size: {memory_settings['batch_size']}")
print(f"Memory pool size: {memory_settings['memory_pool_size']}")

# Create optimized launcher
job_config = {
    'cpus_per_task': 16,
    'memory_per_task': '64GB'
}

launcher = optimizer.create_optimized_launcher(job_config, memory_settings)
```

## Best Practices

### 1. Resource Planning
```python
# Calculate resource requirements before job submission
def calculate_resource_requirements(molecules, descriptors, target_time):
    calc = Calculator()
    
    # Benchmark with small sample
    sample_molecules = molecules[:100]
    start_time = time.time()
    calc.pandas(sample_molecules, descriptors=descriptors)
    sample_time = time.time() - start_time
    
    # Extrapolate resource needs
    total_time_estimate = (len(molecules) / 100) * sample_time
    required_nodes = max(1, int(total_time_estimate / target_time))
    
    return {
        'estimated_time': total_time_estimate,
        'recommended_nodes': required_nodes,
        'memory_per_node': estimate_memory_needs(molecules, descriptors)
    }
```

### 2. Job Monitoring and Alerts
```python
# Set up automated monitoring and alerts
def setup_job_monitoring(job_id, alert_config):
    monitor_script = f'''
#!/bin/bash
while true; do
    status=$(squeue -j {job_id} -h -o %T)
    if [[ "$status" == "FAILED" ]]; then
        echo "Job {job_id} failed!" | mail -s "HPC Job Alert" {alert_config['email']}
        break
    fi
    sleep 300
done
'''
    return monitor_script
```

### 3. Data Management
```python
# Efficient data staging and cleanup
def create_data_management_script(input_files, output_dir, cleanup=True):
    script = f'''
# Stage input data to local storage
rsync -av {input_files} $TMPDIR/
export MOLECULES_FILE=$TMPDIR/molecules.smi

# Run calculation with local data
python -m chemi_accel.main --input $MOLECULES_FILE --output {output_dir}

# Stage results back and cleanup
rsync -av $TMPDIR/results/ {output_dir}/
{'rm -rf $TMPDIR/*' if cleanup else '# Cleanup disabled'}
'''
    return script
```

---

*For cloud-based HPC and distributed computing, see the [Cloud Deployment](Cloud-Deployment) and [Distributed Computing](Distributed-Computing) guides.*

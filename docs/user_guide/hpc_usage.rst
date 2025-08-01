HPC Usage Guide
===============

This guide covers deploying ChemiAccel on High Performance Computing (HPC) clusters using SLURM, PBS, and other job schedulers.

Overview
--------

ChemiAccel provides comprehensive HPC integration for:

* **Multi-node distributed processing** across HPC clusters
* **GPU acceleration** on HPC systems with NVIDIA hardware
* **Job scheduling** integration with SLURM, PBS, and other schedulers
* **MPI parallelization** for large-scale molecular processing
* **Container deployment** using Singularity and Docker
* **Resource optimization** for cost-effective HPC usage

HPC Architecture Patterns
--------------------------

Single-Node Multi-GPU
~~~~~~~~~~~~~~~~~~~~~~

Optimal for moderate datasets with GPU acceleration:

.. code-block:: text

    [Login Node] → [Compute Node: 4x GPU] → [ChemiAccel CUDA Backend]

Multi-Node CPU Cluster
~~~~~~~~~~~~~~~~~~~~~~~

For CPU-intensive workloads across many nodes:

.. code-block:: text

    [Head Node] → [Node 1: 48 cores] → [ChemiAccel CPU Backend]
                → [Node 2: 48 cores] → [ChemiAccel CPU Backend]
                → [Node N: 48 cores] → [ChemiAccel CPU Backend]

Hybrid CPU-GPU Cluster
~~~~~~~~~~~~~~~~~~~~~~~

Mixed workloads with optimal resource utilization:

.. code-block:: text

    [Scheduler] → [GPU Nodes: 2x V100] → [Descriptor Calculation]
              → [CPU Nodes: 64 cores] → [Post-processing]
              → [Storage Nodes] → [Results Aggregation]

SLURM Integration
-----------------

Basic SLURM Job Submission
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple single-node GPU job:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=chemi-accel-gpu
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=8
    #SBATCH --gres=gpu:2
    #SBATCH --time=02:00:00
    #SBATCH --mem=32GB
    #SBATCH --partition=gpu

    # Load modules
    module load cuda/11.8
    module load python/3.10
    module load singularity

    # Activate environment
    source /path/to/chemi-accel-env/bin/activate

    # Run ChemiAccel
    python -c "
    from chemi_accel import Calculator
    import pandas as pd

    # Load data
    df = pd.read_csv('compounds.csv')
    smiles = df['smiles'].tolist()

    # Process with GPU acceleration
    calc = Calculator(backend='cuda')
    results = calc.pandas(smiles)
    results.to_csv('results.csv', index=False)
    "

Multi-Node Distributed Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scale across multiple nodes:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=chemi-accel-distributed
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=2
    #SBATCH --cpus-per-task=12
    #SBATCH --gres=gpu:2
    #SBATCH --time=04:00:00
    #SBATCH --mem=64GB
    #SBATCH --partition=gpu

    # Load modules
    module load cuda/11.8
    module load openmpi/4.1.1
    module load python/3.10

    # Set up MPI environment
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=29500
    export WORLD_SIZE=$SLURM_NTASKS
    export RANK=$SLURM_PROCID

    # Run distributed ChemiAccel
    srun python -m chemi_accel.distributed.launcher \
        --backend cuda \
        --input-file large_dataset.csv \
        --output-file distributed_results.csv \
        --batch-size 10000 \
        --descriptors molecular_weight,logp,tpsa,morgan_fingerprint

Array Jobs for Parameter Sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process multiple datasets or parameter combinations:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=chemi-accel-array
    #SBATCH --array=1-10
    #SBATCH --nodes=1
    #SBATCH --gres=gpu:1
    #SBATCH --time=01:00:00
    #SBATCH --mem=16GB

    # Define input files
    INPUT_FILES=(
        "dataset_1.csv"
        "dataset_2.csv"
        "dataset_3.csv"
        "dataset_4.csv"
        "dataset_5.csv"
        "dataset_6.csv"
        "dataset_7.csv"
        "dataset_8.csv"
        "dataset_9.csv"
        "dataset_10.csv"
    )

    # Get current input file
    INPUT_FILE=${INPUT_FILES[$SLURM_ARRAY_TASK_ID-1]}
    OUTPUT_FILE="results_${SLURM_ARRAY_TASK_ID}.csv"

    # Process dataset
    python process_dataset.py --input $INPUT_FILE --output $OUTPUT_FILE

PBS Integration
---------------

Basic PBS Job Script
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    #!/bin/bash
    #PBS -N chemi-accel-gpu
    #PBS -l select=1:ncpus=8:ngpus=2:mem=32GB
    #PBS -l walltime=02:00:00
    #PBS -q gpu

    # Change to working directory
    cd $PBS_O_WORKDIR

    # Load modules
    module load cuda/11.8
    module load python/3.10

    # Run ChemiAccel
    python molecular_analysis.py

Multi-Node PBS Job
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    #!/bin/bash
    #PBS -N chemi-accel-multinode
    #PBS -l select=4:ncpus=24:ngpus=1:mem=64GB
    #PBS -l walltime=04:00:00
    #PBS -q gpu

    # Get node list
    NODES=$(cat $PBS_NODEFILE | sort | uniq)
    
    # Set up MPI
    module load openmpi/4.1.1
    
    # Run distributed processing
    mpirun -np 4 -hostfile $PBS_NODEFILE python distributed_processing.py

Container Integration
---------------------

Singularity Containers
~~~~~~~~~~~~~~~~~~~~~~

Build and run ChemiAccel in Singularity:

.. code-block:: bash

    # Build container (on local machine or build node)
    singularity build chemi-accel.sif docker://chemi-accel/chemi-accel:latest

    # SLURM job with Singularity
    #!/bin/bash
    #SBATCH --job-name=chemi-accel-singularity
    #SBATCH --gres=gpu:2
    #SBATCH --time=02:00:00

    module load singularity

    # Run container with GPU support
    singularity exec --nv chemi-accel.sif python -c "
    from chemi_accel import Calculator
    calc = Calculator(backend='cuda')
    print('ChemiAccel container ready')
    "

Docker with Podman
~~~~~~~~~~~~~~~~~~

For systems supporting rootless containers:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=chemi-accel-podman
    #SBATCH --gres=gpu:1

    module load podman

    # Run with GPU access
    podman run --gpus all \
        -v $(pwd):/workspace \
        -w /workspace \
        chemi-accel/chemi-accel:latest \
        python analysis.py

Python Integration Examples
---------------------------

Distributed Computing Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # distributed_chemi_accel.py
    import os
    import pandas as pd
    from mpi4py import MPI
    from chemi_accel import Calculator
    from chemi_accel.distributed import DistributedCalculator

    def main():
        # Initialize MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Load data on master node
        if rank == 0:
            df = pd.read_csv('large_dataset.csv')
            smiles_list = df['smiles'].tolist()
            
            # Split data among processes
            chunk_size = len(smiles_list) // size
            chunks = [smiles_list[i*chunk_size:(i+1)*chunk_size] 
                     for i in range(size)]
        else:
            chunks = None

        # Scatter data to all processes
        local_smiles = comm.scatter(chunks, root=0)

        # Process local chunk
        calc = Calculator(backend='cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')
        local_results = calc.pandas(local_smiles)

        # Gather results
        all_results = comm.gather(local_results, root=0)

        # Save results on master node
        if rank == 0:
            final_results = pd.concat(all_results, ignore_index=True)
            final_results.to_csv('distributed_results.csv', index=False)
            print(f"Processed {len(final_results)} molecules across {size} processes")

    if __name__ == "__main__":
        main()

GPU Resource Management
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # gpu_resource_manager.py
    import os
    import GPUtil
    from chemi_accel import Calculator

    def select_optimal_gpu():
        """Select GPU with most available memory"""
        gpus = GPUtil.getGPUs()
        if not gpus:
            return None
        
        best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
        return best_gpu.id

    def main():
        # Select optimal GPU
        gpu_id = select_optimal_gpu()
        
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            calc = Calculator(backend='cuda')
            print(f"Using GPU {gpu_id}")
        else:
            calc = Calculator(backend='cpu')
            print("Using CPU backend")

        # Process data
        # ... your processing code here ...

    if __name__ == "__main__":
        main()

Advanced HPC Patterns
----------------------

Dynamic Resource Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adjust resources based on workload:

.. code-block:: python

    # adaptive_hpc_processing.py
    import math
    from chemi_accel import Calculator
    from chemi_accel.utils.distributed import estimate_processing_time

    def calculate_optimal_resources(dataset_size, time_limit_hours):
        """Calculate optimal number of nodes and GPUs needed"""
        
        # Estimate processing time per molecule
        time_per_molecule = 0.001  # seconds (adjust based on descriptors)
        total_time_needed = dataset_size * time_per_molecule
        
        # Calculate required parallel processes
        max_time_seconds = time_limit_hours * 3600
        min_processes = math.ceil(total_time_needed / max_time_seconds)
        
        # Account for GPU memory limitations
        molecules_per_gpu = 50000  # adjust based on GPU memory
        min_gpus = math.ceil(dataset_size / molecules_per_gpu)
        
        return max(min_processes, min_gpus)

    def submit_adaptive_job(dataset_path, time_limit=2):
        """Submit job with optimal resource allocation"""
        
        # Analyze dataset
        import pandas as pd
        df = pd.read_csv(dataset_path)
        dataset_size = len(df)
        
        # Calculate resources
        required_gpus = calculate_optimal_resources(dataset_size, time_limit)
        
        # Generate SLURM job script
        job_script = f"""#!/bin/bash
    #SBATCH --job-name=adaptive-chemi-accel
    #SBATCH --gres=gpu:{min(required_gpus, 8)}
    #SBATCH --time={time_limit:02d}:00:00
    #SBATCH --mem={required_gpus * 16}GB

    python distributed_processing.py --dataset {dataset_path}
    """
        
        # Submit job
        with open('adaptive_job.sh', 'w') as f:
            f.write(job_script)
        
        os.system('sbatch adaptive_job.sh')

Fault Tolerance
~~~~~~~~~~~~~~~

Handle node failures gracefully:

.. code-block:: python

    # fault_tolerant_processing.py
    import pickle
    import time
    from pathlib import Path
    from chemi_accel import Calculator

    class FaultTolerantProcessor:
        def __init__(self, checkpoint_dir='checkpoints'):
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(exist_ok=True)
            
        def save_checkpoint(self, processed_data, remaining_data, checkpoint_id):
            """Save processing checkpoint"""
            checkpoint = {
                'processed': processed_data,
                'remaining': remaining_data,
                'timestamp': time.time()
            }
            
            checkpoint_file = self.checkpoint_dir / f'checkpoint_{checkpoint_id}.pkl'
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
                
        def load_checkpoint(self, checkpoint_id):
            """Load processing checkpoint"""
            checkpoint_file = self.checkpoint_dir / f'checkpoint_{checkpoint_id}.pkl'
            if checkpoint_file.exists():
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
            return None
            
        def process_with_checkpoints(self, smiles_list, batch_size=1000):
            """Process data with periodic checkpointing"""
            calc = Calculator(backend='cuda')
            processed_results = []
            
            # Try to resume from checkpoint
            checkpoint = self.load_checkpoint('main')
            if checkpoint:
                processed_results = checkpoint['processed']
                remaining_smiles = checkpoint['remaining']
                print(f"Resumed from checkpoint: {len(processed_results)} already processed")
            else:
                remaining_smiles = smiles_list
                
            # Process in batches with checkpointing
            while remaining_smiles:
                batch = remaining_smiles[:batch_size]
                remaining_smiles = remaining_smiles[batch_size:]
                
                try:
                    batch_results = calc.pandas(batch)
                    processed_results.append(batch_results)
                    
                    # Save checkpoint every 10 batches
                    if len(processed_results) % 10 == 0:
                        self.save_checkpoint(processed_results, remaining_smiles, 'main')
                        print(f"Checkpoint saved: {len(processed_results)} batches completed")
                        
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Save checkpoint and exit
                    self.save_checkpoint(processed_results, remaining_smiles, 'error')
                    raise
                    
            return processed_results

Performance Optimization
-------------------------

Memory Management
~~~~~~~~~~~~~~~~~

Optimize memory usage for large datasets:

.. code-block:: python

    # memory_optimized_processing.py
    import gc
    import psutil
    from chemi_accel import Calculator

    def get_available_memory():
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / (1024**3)

    def calculate_optimal_batch_size(backend='cuda'):
        """Calculate optimal batch size based on available memory"""
        if backend == 'cuda':
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            available_gb = mem_info.free / (1024**3)
            # Estimate 1MB per molecule for GPU processing
            return int(available_gb * 1000)
        else:
            available_gb = get_available_memory()
            # Estimate 100KB per molecule for CPU processing
            return int(available_gb * 10000)

    def memory_efficient_processing(smiles_list):
        """Process large datasets with memory management"""
        calc = Calculator(backend='cuda')
        batch_size = calculate_optimal_batch_size('cuda')
        
        print(f"Processing {len(smiles_list)} molecules in batches of {batch_size}")
        
        results = []
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            
            # Process batch
            batch_results = calc.pandas(batch)
            results.append(batch_results)
            
            # Force garbage collection
            gc.collect()
            
            # Monitor memory usage
            if i % (batch_size * 10) == 0:
                memory_usage = psutil.virtual_memory().percent
                print(f"Processed {i} molecules, Memory usage: {memory_usage:.1f}%")
        
        # Combine results efficiently
        import pandas as pd
        final_results = pd.concat(results, ignore_index=True)
        
        return final_results

I/O Optimization
~~~~~~~~~~~~~~~~

Optimize file I/O for HPC storage systems:

.. code-block:: python

    # io_optimized_processing.py
    import pandas as pd
    from pathlib import Path
    from chemi_accel import Calculator

    def parallel_file_reading(file_paths, n_processes=4):
        """Read multiple files in parallel"""
        from concurrent.futures import ProcessPoolExecutor
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            dataframes = list(executor.map(pd.read_csv, file_paths))
            
        return pd.concat(dataframes, ignore_index=True)

    def streaming_processing(input_file, output_file, chunk_size=10000):
        """Process large files in streaming fashion"""
        calc = Calculator(backend='cuda')
        
        # Process file in chunks
        chunk_results = []
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            smiles = chunk['smiles'].tolist()
            results = calc.pandas(smiles)
            chunk_results.append(results)
            
            # Write intermediate results
            if len(chunk_results) >= 10:  # Write every 10 chunks
                combined = pd.concat(chunk_results, ignore_index=True)
                
                # Append to output file
                mode = 'a' if Path(output_file).exists() else 'w'
                header = mode == 'w'
                combined.to_csv(output_file, mode=mode, header=header, index=False)
                
                chunk_results = []  # Clear memory

        # Write final chunks
        if chunk_results:
            combined = pd.concat(chunk_results, ignore_index=True)
            mode = 'a' if Path(output_file).exists() else 'w'
            header = mode == 'w'
            combined.to_csv(output_file, mode=mode, header=header, index=False)

Best Practices
--------------

Resource Planning
~~~~~~~~~~~~~~~~~

1. **Profile your workload** before large-scale deployment
2. **Use appropriate queue/partition** for your job requirements
3. **Monitor resource utilization** during job execution
4. **Plan for data transfer times** to/from shared storage

Job Submission Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use job arrays** for embarrassingly parallel workloads
2. **Implement checkpointing** for long-running jobs
3. **Set appropriate time limits** to avoid unnecessary queuing
4. **Use dependency chains** for multi-stage workflows

Debugging on HPC
~~~~~~~~~~~~~~~~~

1. **Test with small datasets** before scaling up
2. **Use interactive sessions** for debugging
3. **Check module compatibility** before job submission
4. **Monitor job logs** in real-time when possible

Common HPC Issues
-----------------

Module Loading Problems
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Check available modules
    module avail cuda
    module avail python

    # Load in correct order
    module load cuda/11.8
    module load python/3.10
    module load singularity

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Reduce batch size
    calc = Calculator(backend='cuda', config={'batch_size': 1000})

    # Monitor memory usage
    import psutil
    print(f"Memory usage: {psutil.virtual_memory().percent:.1f}%")

Job Timeout Issues
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Implement checkpointing in job script
    #!/bin/bash
    #SBATCH --time=01:59:00  # Leave 1 minute buffer

    # Save checkpoint before timeout
    python process_with_checkpoints.py

File System Limits
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Check quotas
    quota -u $USER

    # Use scratch storage for temporary files
    export TMPDIR=/scratch/$USER
    mkdir -p $TMPDIR

For additional HPC-specific support, consult your cluster documentation or contact your HPC support team.
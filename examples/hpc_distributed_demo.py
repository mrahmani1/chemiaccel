#!/usr/bin/env python3
"""
HPC and Distributed Computing Demo for ChemiAccel.

This script demonstrates the HPC and distributed computing capabilities
of ChemiAccel, including MPI, SLURM, Dask, and Ray integration.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemi_accel.integrations.hpc_integration import (
    DaskClusterIntegration,
    HPCJobConfig,
    MPIIntegration,
    ParallelIOManager,
    RayClusterIntegration,
    SLURMJobManager,
    create_hpc_calculator,
    detect_hpc_environment,
)
from chemi_accel.utils.distributed import (
    generate_test_molecules,
    validate_hpc_environment,
)


def demo_environment_detection():
    """Demonstrate HPC environment detection."""
    print("=== HPC Environment Detection ===")

    env = detect_hpc_environment()

    print(f"Scheduler: {env.get('scheduler', 'None detected')}")
    print(f"MPI Available: {env['mpi_available']}")
    print(f"Dask Available: {env['dask_available']}")
    print(f"Ray Available: {env['ray_available']}")
    print(f"Node Count: {env['node_count']}")
    print(f"GPU Count: {env['gpu_count']}")

    if env.get("job_id"):
        print(f"Job ID: {env['job_id']}")
    if env.get("partition"):
        print(f"Partition: {env['partition']}")

    # Validate environment
    validation = validate_hpc_environment()
    print("\nEnvironment Validation:")
    print(f"Lustre Available: {validation['lustre_available']}")
    print(f"GPFS Available: {validation['gpfs_available']}")
    print(f"InfiniBand Available: {validation['infiniband_available']}")

    if validation["recommendations"]:
        print("\nRecommendations:")
        for rec in validation["recommendations"]:
            print(f"  - {rec}")

    print()


def demo_mpi_integration(molecules: pd.Series):
    """Demonstrate MPI integration."""
    print("=== MPI Integration Demo ===")

    try:
        mpi_integration = MPIIntegration()

        print(f"MPI World Size: {mpi_integration.config.world_size}")
        print(f"MPI Rank: {mpi_integration.config.rank}")
        print(f"Processor Name: {mpi_integration.config.processor_name}")
        print(f"Is Root: {mpi_integration.config.is_root}")

        # Distribute workload
        local_molecules = mpi_integration.distribute_workload(molecules)
        print(
            f"Local molecules for rank {mpi_integration.config.rank}: {len(local_molecules)}"
        )

        # Simulate processing
        local_results = pd.DataFrame(
            {
                "smiles": local_molecules,
                "rank": mpi_integration.config.rank,
                "molecular_weight": np.random.rand(len(local_molecules)) * 500,
            }
        )

        # Gather results
        combined_results = mpi_integration.gather_results(local_results)

        if mpi_integration.config.is_root and combined_results is not None:
            print(f"Combined results: {len(combined_results)} molecules")
            print(f"Ranks represented: {sorted(combined_results['rank'].unique())}")

    except Exception as e:
        print(f"MPI integration failed: {e}")

    print()


def demo_slurm_job_generation():
    """Demonstrate SLURM job script generation."""
    print("=== SLURM Job Generation Demo ===")

    job_config = HPCJobConfig(
        scheduler="slurm",
        nodes=4,
        gpus_per_node=2,
        walltime="02:00:00",
        partition="gpu",
        modules_to_load=["cuda/11.8", "python/3.9", "openmpi/4.1"],
        mpi_enabled=True,
        job_name="chemiaccel_demo",
        memory_per_node="64GB",
        account="research_project",
    )

    slurm_manager = SLURMJobManager()

    # Generate job script
    script = slurm_manager.generate_job_script(
        job_config,
        "chemiaccel_processing.py",
        ["--input", "/data/molecules.csv", "--output", "/data/results.parquet"],
    )

    print("Generated SLURM job script:")
    print("-" * 50)
    print(script)
    print("-" * 50)

    # Check SLURM availability
    if slurm_manager.slurm_available:
        print("SLURM is available on this system")
    else:
        print("SLURM is not available on this system")

    # Check Google Cloud Run availability
    if slurm_manager.gcloud_available:
        print("Google Cloud SDK is available")
    else:
        print("Google Cloud SDK is not available")

    print()


def demo_dask_integration(molecules: pd.Series):
    """Demonstrate Dask cluster integration."""
    print("=== Dask Integration Demo ===")

    try:
        # Create local Dask cluster
        dask_integration = DaskClusterIntegration()
        dask_integration.create_local_cluster(n_workers=2, threads_per_worker=1)

        print("Dask cluster created with 2 workers")

        # Get cluster status
        if dask_integration.client:
            scheduler_info = dask_integration.client.scheduler_info()
            print(f"Workers: {len(scheduler_info['workers'])}")
            print(
                f"Total cores: {sum(w['nthreads'] for w in scheduler_info['workers'].values())}"
            )

        # Simulate distributed calculation
        calculator_config = {"backend": "cpu", "batch_size": 50}

        print("Running distributed calculation...")
        start_time = time.time()

        # Mock the calculation for demo
        def mock_calculate_chunk(molecules, config):
            time.sleep(0.1)  # Simulate processing time
            return pd.DataFrame(
                {
                    "smiles": molecules,
                    "molecular_weight": np.random.rand(len(molecules)) * 500,
                }
            )

        # Replace the actual method for demo
        original_method = DaskClusterIntegration._calculate_chunk
        DaskClusterIntegration._calculate_chunk = staticmethod(mock_calculate_chunk)

        try:
            results = dask_integration.distribute_calculation(
                molecules, calculator_config, chunk_size=50
            )

            total_time = time.time() - start_time
            print(f"Processed {len(results)} molecules in {total_time:.2f}s")
            print(f"Throughput: {len(results)/total_time:.1f} molecules/second")

        finally:
            # Restore original method
            DaskClusterIntegration._calculate_chunk = original_method

        dask_integration.close()

    except Exception as e:
        print(f"Dask integration failed: {e}")

    print()


def demo_ray_integration(molecules: pd.Series):
    """Demonstrate Ray cluster integration."""
    print("=== Ray Integration Demo ===")

    try:
        # Initialize Ray
        ray_integration = RayClusterIntegration()
        ray_integration.init_local_ray(num_cpus=2, num_gpus=0)

        print("Ray cluster initialized")

        # Get cluster resources
        try:
            import ray

            resources = ray.cluster_resources()
            print(f"Cluster resources: {resources}")
        except:
            pass

        # Simulate distributed calculation
        calculator_config = {"backend": "cpu", "batch_size": 50}

        print("Running Ray distributed calculation...")
        start_time = time.time()

        # For demo, we'll simulate the calculation
        print(f"Processing {len(molecules)} molecules with Ray...")
        time.sleep(0.5)  # Simulate processing

        # Create mock results
        results = pd.DataFrame(
            {
                "smiles": molecules,
                "molecular_weight": np.random.rand(len(molecules)) * 500,
            }
        )

        total_time = time.time() - start_time
        print(f"Processed {len(results)} molecules in {total_time:.2f}s")
        print(f"Throughput: {len(results)/total_time:.1f} molecules/second")

        ray_integration.shutdown()

    except Exception as e:
        print(f"Ray integration failed: {e}")

    print()


def demo_parallel_io():
    """Demonstrate parallel I/O capabilities."""
    print("=== Parallel I/O Demo ===")

    # Test different file systems
    filesystems = ["lustre", "gpfs", "beegfs", "nfs"]

    for fs in filesystems:
        print(f"\n{fs.upper()} Configuration:")
        io_manager = ParallelIOManager(file_system=fs)
        optimizations = io_manager.optimize_for_filesystem("/fake/path")

        for key, value in optimizations.items():
            print(f"  {key}: {value}")

    # Demonstrate I/O benchmarking
    print("\nI/O Performance Benchmarking:")

    # Create test data
    test_data = pd.DataFrame(
        {
            "smiles": generate_test_molecules(1000),
            "molecular_weight": np.random.rand(1000) * 500,
            "logp": np.random.rand(1000) * 5 - 2,
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        io_manager = ParallelIOManager(file_system="lustre")

        # Benchmark I/O performance
        benchmark_results = io_manager.benchmark_io_performance(test_data, temp_dir)

        print("Format Performance (seconds):")
        for format_name in ["csv", "parquet", "hdf5"]:
            write_key = f"{format_name}_write"
            read_key = f"{format_name}_read"
            throughput_key = f"{format_name}_throughput"

            if write_key in benchmark_results:
                print(f"  {format_name.upper()}:")
                print(f"    Write: {benchmark_results[write_key]:.3f}s")
                print(f"    Read:  {benchmark_results[read_key]:.3f}s")
                print(f"    Throughput: {benchmark_results[throughput_key]:.1f} mol/s")

    print()


def demo_hpc_calculator_scaling(molecules: pd.Series):
    """Demonstrate HPC calculator scaling."""
    print("=== HPC Calculator Scaling Demo ===")

    # Create HPC calculator
    calculator = create_hpc_calculator(backend="cpu", batch_size=50)

    print("Calculator Configuration:")
    info = calculator.get_cluster_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Benchmark scaling performance
    print("\nScaling Performance Benchmark:")
    scaling_results = calculator.benchmark_scaling(molecules, node_counts=[1, 2, 4])

    print("Node Count | Time (s) | Efficiency | Speedup | Throughput (mol/s)")
    print("-" * 65)

    for node_count, result in scaling_results.items():
        if "error" not in result:
            print(
                f"{node_count:9d} | {result['total_time']:8.2f} | "
                f"{result['efficiency']:10.2f} | {result['speedup']:7.2f} | "
                f"{result['throughput']:14.1f}"
            )
        else:
            print(f"{node_count:9d} | ERROR: {result['error']}")

    print()


def demo_file_processing():
    """Demonstrate parallel file processing."""
    print("=== Parallel File Processing Demo ===")

    # Create test data file
    test_molecules = generate_test_molecules(500)

    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "molecules.csv"
        output_file = Path(temp_dir) / "results.parquet"

        # Write test data
        test_df = pd.DataFrame({"smiles": test_molecules})
        test_df.to_csv(input_file, index=False)

        print(f"Created test file with {len(test_molecules)} molecules")

        # Create HPC calculator
        calculator = create_hpc_calculator(
            backend="cpu", batch_size=100, parallel_io_filesystem="lustre"
        )

        # Process file
        print("Processing file with parallel I/O...")
        start_time = time.time()

        success = calculator.process_file_parallel(
            str(input_file),
            str(output_file),
            input_format="csv",
            output_format="parquet",
        )

        total_time = time.time() - start_time

        if success:
            # Read and verify results
            results = pd.read_parquet(output_file)
            print(
                f"Successfully processed {len(results)} molecules in {total_time:.2f}s"
            )
            print(f"Throughput: {len(results)/total_time:.1f} molecules/second")
            print(f"Output columns: {list(results.columns)}")
        else:
            print("File processing failed")

    print()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="ChemiAccel HPC and Distributed Computing Demo"
    )
    parser.add_argument(
        "--molecules",
        type=int,
        default=200,
        help="Number of test molecules to generate",
    )
    parser.add_argument(
        "--demo",
        choices=[
            "all",
            "environment",
            "mpi",
            "slurm",
            "dask",
            "ray",
            "io",
            "scaling",
            "file",
        ],
        default="all",
        help="Which demo to run",
    )

    args = parser.parse_args()

    print("ChemiAccel HPC and Distributed Computing Demo")
    print("=" * 50)

    # Generate test molecules
    molecules = generate_test_molecules(args.molecules, complexity="mixed")
    print(f"Generated {len(molecules)} test molecules")
    print()

    # Run selected demos
    if args.demo in ["all", "environment"]:
        demo_environment_detection()

    if args.demo in ["all", "mpi"]:
        demo_mpi_integration(molecules)

    if args.demo in ["all", "slurm"]:
        demo_slurm_job_generation()

    if args.demo in ["all", "dask"]:
        demo_dask_integration(molecules)

    if args.demo in ["all", "ray"]:
        demo_ray_integration(molecules)

    if args.demo in ["all", "io"]:
        demo_parallel_io()

    if args.demo in ["all", "scaling"]:
        demo_hpc_calculator_scaling(molecules)

    if args.demo in ["all", "file"]:
        demo_file_processing()

    print("Demo completed!")


if __name__ == "__main__":
    main()

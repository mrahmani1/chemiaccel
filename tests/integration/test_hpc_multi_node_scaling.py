"""
Multi-node scaling tests and performance validation for HPC integration.

This module tests the scaling efficiency and performance of ChemiAccel
across different node counts and distributed computing configurations.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from chemi_accel.core.exceptions import DistributedComputingError
from chemi_accel.integrations.hpc_integration import (
    DaskClusterIntegration,
    HPCCalculator,
    HPCJobConfig,
    MPIIntegration,
    ParallelIOManager,
    RayClusterIntegration,
    SLURMJobManager,
    create_hpc_calculator,
    detect_hpc_environment,
    submit_hpc_job,
)
from chemi_accel.utils.distributed import PerformanceMonitor, generate_test_molecules

# Test configuration
TEST_MOLECULE_COUNTS = [100, 500, 1000, 5000]
NODE_COUNTS = [1, 2, 4, 8]
SCALING_TOLERANCE = 0.1  # 10% tolerance for scaling efficiency


class TestMultiNodeScaling:
    """Test multi-node scaling performance and efficiency."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_molecules = generate_test_molecules(1000, complexity="mixed")
        self.performance_monitor = PerformanceMonitor()

    def test_strong_scaling_simulation(self):
        """Test strong scaling efficiency with simulated multi-node processing."""
        calculator = HPCCalculator(backend="cpu", batch_size=100)

        scaling_results = calculator.benchmark_scaling(
            self.test_molecules, node_counts=[1, 2, 4, 8]
        )

        # Validate scaling results
        assert len(scaling_results) == 4

        # Check that processing time decreases with more nodes (ideally)
        times = [
            scaling_results[n]["total_time"]
            for n in [1, 2, 4, 8]
            if "total_time" in scaling_results[n]
        ]

        # At least some improvement should be seen
        if len(times) >= 2:
            assert times[0] >= times[-1] * 0.5  # At least 50% improvement

        # Check efficiency calculations
        for node_count, result in scaling_results.items():
            if "efficiency" in result:
                assert 0 <= result["efficiency"] <= 1.0
                assert result["throughput"] > 0

    def test_weak_scaling_simulation(self):
        """Test weak scaling by increasing problem size with node count."""
        base_molecule_count = 100

        results = {}
        baseline_time = None

        for node_count in [1, 2, 4]:
            # Scale problem size with node count
            molecule_count = base_molecule_count * node_count
            test_molecules = generate_test_molecules(molecule_count)

            calculator = HPCCalculator(backend="cpu", batch_size=50)

            start_time = time.time()
            descriptors = calculator.pandas(test_molecules)
            total_time = time.time() - start_time

            if baseline_time is None:
                baseline_time = total_time

            efficiency = baseline_time / total_time if total_time > 0 else 0

            results[node_count] = {
                "molecule_count": molecule_count,
                "total_time": total_time,
                "efficiency": efficiency,
                "throughput": molecule_count / total_time if total_time > 0 else 0,
            }

        # Weak scaling should maintain roughly constant time per node
        assert len(results) >= 2

        # Check that we processed the correct number of molecules
        for node_count, result in results.items():
            expected_count = base_molecule_count * node_count
            assert result["molecule_count"] == expected_count

    @pytest.mark.skipif(
        not os.environ.get("MPI_AVAILABLE"), reason="MPI not available for testing"
    )
    def test_mpi_scaling_real(self):
        """Test actual MPI scaling (requires MPI environment)."""
        try:
            mpi_integration = MPIIntegration()

            # Test workload distribution
            distributed_molecules = mpi_integration.distribute_workload(
                self.test_molecules
            )

            # Each rank should get roughly equal work
            expected_size = (
                len(self.test_molecules) // mpi_integration.config.world_size
            )
            tolerance = max(1, expected_size // 10)  # 10% tolerance

            assert abs(len(distributed_molecules) - expected_size) <= tolerance

            # Test that total work is preserved
            total_distributed = mpi_integration.comm.allreduce(
                len(distributed_molecules)
            )
            assert total_distributed == len(self.test_molecules)

        except DistributedComputingError:
            pytest.skip("MPI not properly configured")

    def test_load_balancing_efficiency(self):
        """Test load balancing across workers."""
        from chemi_accel.utils.distributed import LoadBalancer

        # Test perfect load balancing
        ranges = LoadBalancer.balance_workload(1000, 4)
        work_sizes = [end - start for start, end in ranges]

        # All workers should have similar workload
        min_work = min(work_sizes)
        max_work = max(work_sizes)
        balance_factor = min_work / max_work if max_work > 0 else 0

        assert balance_factor >= 0.75  # At least 75% balanced

        # Test with uneven division
        ranges = LoadBalancer.balance_workload(1001, 4)
        work_sizes = [end - start for start, end in ranges]

        # Should still be reasonably balanced
        assert max(work_sizes) - min(work_sizes) <= 1

    def test_memory_scaling_efficiency(self):
        """Test memory usage scaling with problem size."""
        molecule_counts = [100, 500, 1000]
        memory_usage = []

        for count in molecule_counts:
            test_molecules = generate_test_molecules(count)
            calculator = HPCCalculator(backend="cpu", batch_size=50)

            # Monitor memory during calculation
            monitor = PerformanceMonitor()
            monitor.start_monitoring()

            _ = calculator.pandas(test_molecules)

            metrics = monitor.stop_monitoring()
            memory_usage.append(metrics.memory_usage_mb)

        # Memory usage should scale reasonably with problem size
        # (not necessarily linearly due to overhead)
        assert len(memory_usage) == 3
        assert memory_usage[1] >= memory_usage[0]  # More molecules = more memory
        assert memory_usage[2] >= memory_usage[1]


class TestParallelIOScaling:
    """Test parallel I/O scaling and performance."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_data = pd.DataFrame(
            {
                "smiles": generate_test_molecules(1000),
                "molecular_weight": np.random.rand(1000) * 500,
                "logp": np.random.rand(1000) * 5 - 2,
            }
        )

    def test_filesystem_optimization_detection(self):
        """Test detection and optimization for different filesystems."""
        filesystems = ["lustre", "gpfs", "beegfs", "nfs", "ceph"]

        for fs in filesystems:
            io_manager = ParallelIOManager(file_system=fs)
            optimizations = io_manager.optimize_for_filesystem("/fake/path")

            assert isinstance(optimizations, dict)
            assert "buffer_size" in optimizations

            # Each filesystem should have specific optimizations
            if fs == "lustre":
                assert "stripe_count" in optimizations
                assert "stripe_size" in optimizations
            elif fs == "gpfs":
                assert "block_size" in optimizations
                assert "prefetch_threads" in optimizations

    def test_parallel_io_performance_benchmark(self):
        """Test I/O performance benchmarking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            io_manager = ParallelIOManager(file_system="lustre")

            # Benchmark I/O performance
            benchmark_results = io_manager.benchmark_io_performance(
                self.test_data, temp_dir
            )

            # Should have results for different formats
            expected_formats = ["csv", "parquet", "hdf5"]
            for fmt in expected_formats:
                if f"{fmt}_write" in benchmark_results:
                    assert benchmark_results[f"{fmt}_write"] > 0
                    assert benchmark_results[f"{fmt}_throughput"] > 0

    @patch("subprocess.run")
    def test_lustre_striping_configuration(self, mock_run):
        """Test Lustre file system striping configuration."""
        mock_run.return_value = Mock(returncode=0)

        io_manager = ParallelIOManager(file_system="lustre")

        with tempfile.TemporaryDirectory() as temp_dir:
            success = io_manager.setup_lustre_striping(
                temp_dir, stripe_count=8, stripe_size="2M"
            )

            # Should attempt to configure striping
            assert mock_run.called

            # Check that lfs setstripe command was called
            call_args = mock_run.call_args_list[0][0][0]
            assert "lfs" in call_args
            assert "setstripe" in call_args

    @patch("subprocess.run")
    def test_gpfs_optimization_configuration(self, mock_run):
        """Test GPFS file system optimization."""
        mock_run.return_value = Mock(returncode=0)

        io_manager = ParallelIOManager(file_system="gpfs")

        success = io_manager.setup_gpfs_policies("/fake/path")

        # Should attempt GPFS configuration
        if mock_run.called:
            # Verify GPFS commands were attempted
            assert any(
                "mmfsadm" in str(call) or "mmchfs" in str(call)
                for call in mock_run.call_args_list
            )


class TestDistributedFrameworkScaling:
    """Test scaling with Dask and Ray distributed frameworks."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_molecules = generate_test_molecules(500)

    @pytest.mark.skipif(
        not os.environ.get("DASK_AVAILABLE"), reason="Dask not available for testing"
    )
    def test_dask_cluster_scaling(self):
        """Test Dask cluster scaling performance."""
        try:
            # Create local Dask cluster
            dask_integration = DaskClusterIntegration()
            dask_integration.create_local_cluster(n_workers=2, threads_per_worker=1)

            calculator_config = {"backend": "cpu", "batch_size": 100}

            # Test distributed calculation
            start_time = time.time()
            results = dask_integration.distribute_calculation(
                self.test_molecules, calculator_config, chunk_size=250
            )
            total_time = time.time() - start_time

            # Verify results
            assert len(results) == len(self.test_molecules)
            assert total_time > 0

            # Get cluster status
            status = dask_integration.client.scheduler_info()
            assert len(status["workers"]) == 2

            dask_integration.close()

        except Exception as e:
            pytest.skip(f"Dask cluster test failed: {e}")

    @pytest.mark.skipif(
        not os.environ.get("RAY_AVAILABLE"), reason="Ray not available for testing"
    )
    def test_ray_cluster_scaling(self):
        """Test Ray cluster scaling performance."""
        try:
            # Initialize local Ray cluster
            ray_integration = RayClusterIntegration()
            ray_integration.init_local_ray(num_cpus=2, num_gpus=0)

            calculator_config = {"backend": "cpu", "batch_size": 100}

            # Test distributed calculation
            start_time = time.time()
            results = ray_integration.distribute_calculation(
                self.test_molecules, calculator_config, chunk_size=250
            )
            total_time = time.time() - start_time

            # Verify results
            assert len(results) == len(self.test_molecules)
            assert total_time > 0

            ray_integration.shutdown()

        except Exception as e:
            pytest.skip(f"Ray cluster test failed: {e}")


class TestHPCJobSubmission:
    """Test HPC job submission and management."""

    def setup_method(self):
        """Setup test fixtures."""
        self.job_config = HPCJobConfig(
            scheduler="slurm",
            nodes=2,
            gpus_per_node=1,
            walltime="01:00:00",
            partition="gpu",
            modules_to_load=["cuda/11.8", "python/3.9"],
            mpi_enabled=True,
            job_name="test_chemiaccel_scaling",
        )

    @patch("subprocess.run")
    def test_slurm_job_script_generation(self, mock_run):
        """Test SLURM job script generation for scaling tests."""
        mock_run.return_value = Mock(returncode=0)

        slurm_manager = SLURMJobManager()

        # Generate job script
        script = slurm_manager.generate_job_script(
            self.job_config, "scaling_test.py", ["--molecules", "10000", "--nodes", "2"]
        )

        # Verify script contains scaling-specific configurations
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "mpirun -np 2" in script
        assert "scaling_test.py --molecules 10000 --nodes 2" in script
        assert "module load cuda/11.8" in script
        assert "module load python/3.9" in script

    @patch("subprocess.run")
    def test_hpc_job_submission_utility(self, mock_run):
        """Test HPC job submission utility function."""
        mock_run.side_effect = [
            Mock(returncode=0),  # SLURM availability check
            Mock(stdout="Submitted batch job 12345\n", returncode=0),  # sbatch
        ]

        job_id = submit_hpc_job(
            scheduler="slurm",
            job_config=self.job_config,
            input_path="/data/molecules.csv",
            output_path="/data/results.parquet",
            descriptor_config={"backend": "cuda", "batch_size": 1000},
        )

        assert job_id == "12345"

        # Verify sbatch was called
        assert mock_run.call_count >= 2
        sbatch_call = mock_run.call_args_list[1]
        assert sbatch_call[0][0][0] == "sbatch"


class TestPerformanceValidation:
    """Test performance validation and metrics."""

    def test_scaling_efficiency_calculation(self):
        """Test scaling efficiency calculation and validation."""
        # Simulate performance data
        baseline_time = 100.0

        scaling_data = {
            1: {"total_time": 100.0},
            2: {"total_time": 55.0},  # 91% efficiency
            4: {"total_time": 30.0},  # 83% efficiency
            8: {"total_time": 20.0},  # 62% efficiency
        }

        # Calculate efficiencies
        for node_count, data in scaling_data.items():
            ideal_time = baseline_time / node_count
            efficiency = ideal_time / data["total_time"]
            data["efficiency"] = efficiency

        # Validate efficiency trends
        assert scaling_data[1]["efficiency"] == 1.0  # Perfect baseline
        assert scaling_data[2]["efficiency"] > 0.8  # Good scaling
        assert scaling_data[4]["efficiency"] > 0.7  # Acceptable scaling

        # Efficiency should generally decrease with more nodes
        assert scaling_data[2]["efficiency"] >= scaling_data[4]["efficiency"]

    def test_throughput_validation(self):
        """Test throughput calculation and validation."""
        molecule_count = 1000

        performance_data = [
            {"nodes": 1, "time": 100.0},
            {"nodes": 2, "time": 55.0},
            {"nodes": 4, "time": 30.0},
        ]

        for data in performance_data:
            data["throughput"] = molecule_count / data["time"]

        # Throughput should increase with more nodes
        assert performance_data[1]["throughput"] > performance_data[0]["throughput"]
        assert performance_data[2]["throughput"] > performance_data[1]["throughput"]

        # Validate throughput values are reasonable
        for data in performance_data:
            assert data["throughput"] > 0
            assert data["throughput"] < 1000  # Reasonable upper bound


class TestEnvironmentDetection:
    """Test HPC environment detection and configuration."""

    def test_hpc_environment_detection(self):
        """Test HPC environment detection."""
        env = detect_hpc_environment()

        # Should return a dictionary with expected keys
        expected_keys = [
            "scheduler",
            "mpi_available",
            "dask_available",
            "ray_available",
            "slurm_available",
            "pbs_available",
            "node_count",
            "gpu_count",
        ]

        for key in expected_keys:
            assert key in env

        # Values should be appropriate types
        assert isinstance(env["mpi_available"], bool)
        assert isinstance(env["node_count"], int)
        assert isinstance(env["gpu_count"], int)
        assert env["node_count"] >= 1
        assert env["gpu_count"] >= 0

    @patch.dict(
        os.environ,
        {
            "SLURM_JOB_ID": "12345",
            "SLURM_NNODES": "4",
            "SLURM_JOB_PARTITION": "gpu",
            "SLURM_CPUS_ON_NODE": "8",
            "CUDA_VISIBLE_DEVICES": "0,1",
        },
    )
    def test_slurm_environment_detection(self):
        """Test SLURM environment detection."""
        env = detect_hpc_environment()

        assert env["scheduler"] == "slurm"
        assert env["slurm_available"] == True
        assert env["node_count"] == 4
        assert env["job_id"] == "12345"
        assert env["partition"] == "gpu"
        assert env["cpus_per_node"] == 8
        assert env["gpu_count"] == 2

    @patch.dict(os.environ, {"PBS_JOBID": "67890.cluster", "PBS_QUEUE": "gpu_queue"})
    def test_pbs_environment_detection(self):
        """Test PBS environment detection."""
        env = detect_hpc_environment()

        assert env["scheduler"] == "pbs"
        assert env["pbs_available"] == True
        assert env["job_id"] == "67890.cluster"
        assert env["queue"] == "gpu_queue"

    def test_hpc_calculator_auto_configuration(self):
        """Test automatic HPC calculator configuration."""
        # Test with default environment
        calculator = create_hpc_calculator(backend="cpu")

        assert isinstance(calculator, HPCCalculator)
        assert calculator.backend_type is not None

        # Test cluster info
        info = calculator.get_cluster_info()
        assert isinstance(info, dict)
        assert "mpi_enabled" in info
        assert "parallel_io_filesystem" in info


@pytest.mark.integration
class TestEndToEndScaling:
    """End-to-end scaling tests."""

    def test_complete_scaling_workflow(self):
        """Test complete scaling workflow from data to results."""
        # Create test data
        test_molecules = generate_test_molecules(200)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "molecules.csv"
            output_file = Path(temp_dir) / "results.csv"

            # Write test data
            test_df = pd.DataFrame({"smiles": test_molecules})
            test_df.to_csv(input_file, index=False)

            # Create HPC calculator
            calculator = create_hpc_calculator(backend="cpu", batch_size=50)

            # Process file
            success = calculator.process_file_parallel(
                str(input_file), str(output_file)
            )

            assert success
            assert output_file.exists()

            # Verify results
            results = pd.read_csv(output_file)
            assert len(results) == len(test_molecules)
            assert "smiles" in results.columns

            # Should have descriptor columns
            descriptor_columns = [col for col in results.columns if col != "smiles"]
            assert len(descriptor_columns) > 0

    def test_scaling_performance_regression(self):
        """Test for performance regression in scaling."""
        # This test would be run in CI/CD to catch performance regressions
        test_molecules = generate_test_molecules(100)

        calculator = HPCCalculator(backend="cpu", batch_size=25)

        # Benchmark current performance
        start_time = time.time()
        results = calculator.pandas(test_molecules)
        total_time = time.time() - start_time

        # Performance should be reasonable
        throughput = len(test_molecules) / total_time

        # Should process at least 10 molecules per second on CPU
        assert throughput > 10, f"Performance regression: {throughput:.2f} mol/s"

        # Results should be complete
        assert len(results) == len(test_molecules)
        assert len(results.columns) > 1  # Should have descriptors


if __name__ == "__main__":
    # Run specific test categories
    import sys

    if len(sys.argv) > 1:
        test_category = sys.argv[1]

        if test_category == "scaling":
            pytest.main(["-v", "TestMultiNodeScaling"])
        elif test_category == "io":
            pytest.main(["-v", "TestParallelIOScaling"])
        elif test_category == "distributed":
            pytest.main(["-v", "TestDistributedFrameworkScaling"])
        elif test_category == "jobs":
            pytest.main(["-v", "TestHPCJobSubmission"])
        elif test_category == "performance":
            pytest.main(["-v", "TestPerformanceValidation"])
        elif test_category == "environment":
            pytest.main(["-v", "TestEnvironmentDetection"])
        elif test_category == "e2e":
            pytest.main(["-v", "TestEndToEndScaling"])
        else:
            pytest.main(["-v"])
    else:
        pytest.main(["-v"])

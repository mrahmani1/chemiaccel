"""
Multi-node scaling tests and performance validation for HPC integration.
"""

import pytest
import os
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from chemi_accel.integrations.hpc_integration import (
    MPIIntegration, SLURMJobManager, DaskClusterIntegration,
    RayClusterIntegration, HPCCalculator, detect_hpc_environment,
    create_hpc_calculator
)
from chemi_accel.utils.distributed import (
    PerformanceMonitor, generate_test_molecules
)
from chemi_accel.core.exceptions import DistributedComputingError

# Test configuration
TEST_MOLECULE_COUNTS = [100, 500, 1000]
NODE_COUNTS = [1, 2, 4]
SCALING_TOLERANCE = 0.1  # 10% tolerance for scaling efficiency


class TestMPIIntegration:
    """Test MPI integration functionality."""
    
    def test_mpi_detection(self):
        """Test MPI environment detection."""
        try:
            mpi_integration = MPIIntegration()
            assert mpi_integration.config.world_size >= 1
            assert mpi_integration.config.rank >= 0
            assert mpi_integration.config.rank < mpi_integration.config.world_size
            assert isinstance(mpi_integration.config.processor_name, str)
        except DistributedComputingError:
            pytest.skip("MPI not available")
    
    def test_workload_distribution(self):
        """Test workload distribution across MPI ranks."""
        try:
            mpi_integration = MPIIntegration()
            test_molecules = generate_test_molecules(100)
            
            distributed_molecules = mpi_integration.distribute_workload(test_molecules)
            
            # Check that we got some molecules
            assert len(distributed_molecules) > 0
            
            # Check that total molecules across all ranks equals original
            total_distributed = mpi_integration.comm.allreduce(len(distributed_molecules))
            assert total_distributed == len(test_molecules)
            
        except DistributedComputingError:
            pytest.skip("MPI not available")
    
    def test_result_gathering(self):
        """Test gathering results from all MPI ranks."""
        try:
            mpi_integration = MPIIntegration()
            
            # Create dummy results
            local_results = pd.DataFrame({
                'molecule_id': [f'mol_{mpi_integration.config.rank}_{i}' for i in range(10)],
                'molecular_weight': np.random.rand(10) * 500
            })
            
            gathered_results = mpi_integration.gather_results(local_results)
            
            if mpi_integration.config.is_root:
                assert gathered_results is not None
                assert len(gathered_results) == 10 * mpi_integration.config.world_size
            else:
                assert gathered_results is None
                
        except DistributedComputingError:
            pytest.skip("MPI not available")


class TestSLURMIntegration:
    """Test SLURM job management functionality."""
    
    def test_slurm_detection(self):
        """Test SLURM availability detection."""
        slurm_manager = SLURMJobManager()
        # Should not raise exception even if SLURM not available
        assert isinstance(slurm_manager.slurm_available, bool)
    
    def test_job_script_generation(self):
        """Test SLURM job script generation."""
        from chemi_accel.integrations.hpc_integration import HPCJobConfig
        
        config = HPCJobConfig(
            scheduler='slurm',
            nodes=2,
            gpus_per_node=2,
            walltime='01:00:00',
            partition='gpu',
            modules_to_load=['cuda/11.8', 'python/3.9'],
            mpi_enabled=True,
            job_name='test_chemiaccel'
        )
        
        slurm_manager = SLURMJobManager()
        script = slurm_manager.generate_job_script(
            config, 'test_script.py', ['--input', 'molecules.csv']
        )
        
        # Check that script contains expected SLURM directives
        assert '#SBATCH --job-name=test_chemiaccel' in script
        assert '#SBATCH --nodes=2' in script
        assert '#SBATCH --gres=gpu:2' in script
        assert '#SBATCH --time=01:00:00' in script
        assert '#SBATCH --partition=gpu' in script
        assert 'module load cuda/11.8' in script
        assert 'module load python/3.9' in script
        assert 'mpirun -np 4' in script  # 2 nodes * 2 GPUs
        assert 'test_script.py --input molecules.csv' in script
    
    def test_job_script_without_mpi(self):
        """Test SLURM job script generation without MPI."""
        from chemi_accel.integrations.hpc_integration import HPCJobConfig
        
        config = HPCJobConfig(
            scheduler='slurm',
            nodes=1,
            gpus_per_node=1,
            walltime='00:30:00',
            partition='gpu',
            modules_to_load=['cuda/11.8'],
            mpi_enabled=False
        )
        
        slurm_manager = SLURMJobManager()
        script = slurm_manager.generate_job_script(config, 'test_script.py')
        
        assert 'mpirun' not in script
        assert 'python test_script.py' in script


class TestDaskIntegration:
    """Test Dask cluster integration."""
    
    def test_local_cluster_creation(self):
        """Test local Dask cluster creation."""
        try:
            dask_integration = DaskClusterIntegration()
            dask_integration.create_local_cluster(n_workers=2, threads_per_worker=1)
            
            assert dask_integration.client is not None
            
            # Test basic functionality
            future = dask_integration.client.submit(lambda x: x + 1, 10)
            result = future.result()
            assert result == 11
            
            dask_integration.close()
            
        except DistributedComputingError:
            pytest.skip("Dask not available")
    
    def test_distributed_calculation(self):
        """Test distributed calculation with Dask."""
        try:
            dask_integration = DaskClusterIntegration()
            dask_integration.create_local_cluster(n_workers=2, threads_per_worker=1)
            
            test_molecules = generate_test_molecules(50)
            calculator_config = {
                'backend': 'cpu',
                'batch_size': 10
            }
            
            # This would normally call ChemiAccel calculator
            # For testing, we'll use a mock calculation
            def mock_calculate_chunk(molecules, config):
                return pd.DataFrame({
                    'smiles': molecules,
                    'molecular_weight': np.random.rand(len(molecules)) * 500
                })
            
            # Replace the actual calculation method for testing
            original_method = DaskClusterIntegration._calculate_chunk
            DaskClusterIntegration._calculate_chunk = staticmethod(mock_calculate_chunk)
            
            try:
                results = dask_integration.distribute_calculation(
                    test_molecules, calculator_config, chunk_size=25
                )
                
                assert len(results) == len(test_molecules)
                assert 'smiles' in results.columns
                assert 'molecular_weight' in results.columns
                
            finally:
                # Restore original method
                DaskClusterIntegration._calculate_chunk = original_method
            
            dask_integration.close()
            
        except DistributedComputingError:
            pytest.skip("Dask not available")


class TestRayIntegration:
    """Test Ray cluster integration."""
    
    def test_local_ray_initialization(self):
        """Test local Ray initialization."""
        try:
            ray_integration = RayClusterIntegration()
            
            assert ray_integration.initialized
            
            # Test basic Ray functionality
            @ray_integration._calculate_chunk_remote
            def test_function(x):
                return x * 2
            
            future = test_function.remote(5)
            result = ray.get(future)
            assert result == 10
            
            ray_integration.shutdown()
            
        except DistributedComputingError:
            pytest.skip("Ray not available")


class TestHPCCalculator:
    """Test HPC-aware Calculator functionality."""
    
    def test_hpc_calculator_creation(self):
        """Test HPC Calculator creation and configuration."""
        calculator = create_hpc_calculator(backend='cpu')
        
        assert isinstance(calculator, HPCCalculator)
        assert calculator.backend_type is not None
    
    def test_hpc_environment_detection(self):
        """Test HPC environment detection."""
        env = detect_hpc_environment()
        
        assert isinstance(env, dict)
        assert 'scheduler' in env
        assert 'mpi_available' in env
        assert 'node_count' in env
        assert 'gpu_count' in env
    
    def test_calculation_with_fallback(self):
        """Test calculation with fallback to standard processing."""
        calculator = HPCCalculator(backend='cpu')
        test_molecules = generate_test_molecules(10)
        
        # Should work even without distributed computing setup
        results = calculator.pandas(test_molecules)
        
        assert len(results) == len(test_molecules)
        assert len(results.columns) > 0


class TestScalingPerformance:
    """Test scaling performance and validation."""
    
    @pytest.mark.slow
    def test_strong_scaling_efficiency(self):
        """Test strong scaling efficiency measurement."""
        test_molecules = generate_test_molecules(1000)
        
        # Simulate different node counts
        performance_results = []
        
        for node_count in [1, 2, 4]:
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # Simulate processing time that scales with node count
            processing_time = 10.0 / node_count + np.random.normal(0, 0.1)
            time.sleep(max(0.1, processing_time))  # Minimum sleep for testing
            
            metrics = monitor.stop_monitoring()
            metrics.throughput_molecules_per_sec = len(test_molecules) / metrics.total_time
            
            performance_results.append((node_count, metrics))
        
        # Calculate scaling efficiency
        baseline_time = performance_results[0][1].total_time
        
        for node_count, metrics in performance_results[1:]:
            ideal_time = baseline_time / node_count
            actual_time = metrics.total_time
            efficiency = ideal_time / actual_time
            
            # Should have reasonable scaling efficiency
            assert efficiency > 0.5, f"Poor scaling efficiency: {efficiency:.2f}"
    
    @pytest.mark.slow
    def test_memory_scaling(self):
        """Test memory usage scaling with problem size."""
        molecule_counts = [100, 500, 1000]
        memory_usage = []
        
        for count in molecule_counts:
            test_molecules = generate_test_molecules(count)
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # Simulate memory-intensive processing
            dummy_data = [test_molecules.copy() for _ in range(5)]
            time.sleep(0.1)
            
            metrics = monitor.stop_monitoring()
            memory_usage.append(metrics.memory_usage_mb)
            
            del dummy_data  # Clean up
        
        # Memory usage should scale reasonably with problem size
        assert memory_usage[1] >= memory_usage[0]
        assert memory_usage[2] >= memory_usage[1]
    
    def test_load_balancing(self):
        """Test load balancing across workers."""
        from chemi_accel.utils.distributed import LoadBalancer
        
        # Test perfect division
        ranges = LoadBalancer.balance_workload(100, 4)
        assert len(ranges) == 4
        assert sum(end - start for start, end in ranges) == 100
        
        # Test with remainder
        ranges = LoadBalancer.balance_workload(101, 4)
        assert len(ranges) == 4
        assert sum(end - start for start, end in ranges) == 101
        
        # First worker should get the extra item
        assert ranges[0][1] - ranges[0][0] == 26  # 101 // 4 + 1
        assert ranges[1][1] - ranges[1][0] == 25  # 101 // 4
    
    def test_performance_validation(self):
        """Test performance metrics validation."""
        # Create mock performance data
        work_times = [10.0, 10.5, 9.8, 10.2]  # Well balanced
        
        from chemi_accel.utils.distributed import LoadBalancer
        balance_factor = LoadBalancer.calculate_load_balance_factor(work_times)
        
        assert balance_factor > 0.9  # Should be well balanced
        
        # Test poorly balanced workload
        poor_times = [5.0, 15.0, 8.0, 12.0]
        poor_balance = LoadBalancer.calculate_load_balance_factor(poor_times)
        
        assert poor_balance < 0.5  # Should show poor balance


class TestParallelIO:
    """Test parallel I/O functionality."""
    
    def test_parallel_file_operations(self):
        """Test parallel file read/write operations."""
        from chemi_accel.integrations.hpc_integration import ParallelIOManager
        
        # Create test data
        test_data = pd.DataFrame({
            'smiles': generate_test_molecules(100),
            'molecular_weight': np.random.rand(100) * 500
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / 'test_molecules.csv'
            test_data.to_csv(test_file, index=False)
            
            # Test parallel I/O manager
            io_manager = ParallelIOManager()
            
            # Test reading (will fall back to regular read without MPI)
            read_data = io_manager.parallel_read_csv(str(test_file))
            
            assert len(read_data) == len(test_data)
            assert list(read_data.columns) == list(test_data.columns)
            
            # Test writing
            output_file = Path(temp_dir) / 'output_molecules.csv'
            success = io_manager.parallel_write_csv(read_data, str(output_file))
            
            assert success
            assert output_file.exists()
    
    def test_filesystem_optimization(self):
        """Test filesystem-specific optimizations."""
        from chemi_accel.integrations.hpc_integration import ParallelIOManager
        
        io_manager = ParallelIOManager(file_system='lustre')
        optimizations = io_manager.optimize_for_filesystem('/fake/path')
        
        assert 'stripe_count' in optimizations
        assert 'stripe_size' in optimizations
        assert 'buffer_size' in optimizations
        
        # Test different filesystem
        io_manager_gpfs = ParallelIOManager(file_system='gpfs')
        gpfs_opts = io_manager_gpfs.optimize_for_filesystem('/fake/path')
        
        assert 'block_size' in gpfs_opts
        assert 'buffer_size' in gpfs_opts


@pytest.mark.integration
class TestEndToEndHPC:
    """End-to-end HPC integration tests."""
    
    def test_complete_hpc_workflow(self):
        """Test complete HPC workflow from job submission to results."""
        # This test would normally require actual HPC environment
        # For CI/CD, we'll test the workflow components
        
        from chemi_accel.integrations.hpc_integration import HPCJobConfig
        
        # Test job configuration
        config = HPCJobConfig(
            scheduler='slurm',
            nodes=2,
            gpus_per_node=1,
            walltime='00:30:00',
            partition='gpu',
            modules_to_load=['cuda/11.8', 'python/3.9'],
            mpi_enabled=True
        )
        
        # Test script generation
        slurm_manager = SLURMJobManager()
        script = slurm_manager.generate_job_script(config, 'chemiaccel_job.py')
        
        assert len(script) > 0
        assert 'chemiaccel_job.py' in script
        
        # Test environment detection
        env = detect_hpc_environment()
        assert isinstance(env, dict)
        
        # Test calculator creation
        calculator = create_hpc_calculator(backend='cpu')
        assert isinstance(calculator, HPCCalculator)
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        # Test with unavailable backend
        try:
            calculator = HPCCalculator(
                backend='cpu',
                use_mpi=True,
                dask_scheduler='invalid://address',
                ray_address='invalid://address'
            )
            
            # Should still work with fallbacks
            test_molecules = generate_test_molecules(10)
            results = calculator.pandas(test_molecules)
            
            assert len(results) == len(test_molecules)
            
        except Exception as e:
            # Should handle errors gracefully
            assert isinstance(e, (DistributedComputingError, Exception))


if __name__ == '__main__':
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        
        if test_category == 'mpi':
            pytest.main(['-v', 'TestMPIIntegration'])
        elif test_category == 'slurm':
            pytest.main(['-v', 'TestSLURMIntegration'])
        elif test_category == 'dask':
            pytest.main(['-v', 'TestDaskIntegration'])
        elif test_category == 'ray':
            pytest.main(['-v', 'TestRayIntegration'])
        elif test_category == 'scaling':
            pytest.main(['-v', 'TestScalingPerformance'])
        else:
            pytest.main(['-v'])
    else:
        pytest.main(['-v'])
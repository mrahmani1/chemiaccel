"""
Integration tests for HPC and distributed computing integration.

Tests SLURM, PBS, MPI, and distributed processing capabilities.
"""

import os
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import subprocess

from chemi_accel.cloud.hpc_integration import (
    HPCIntegration,
    MPIIntegration,
    ParallelIOManager,
    HPCJobConfig,
    SLURMScheduler,
    PBSScheduler,
    submit_hpc_job
)


class TestSLURMScheduler:
    """Test SLURM scheduler integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.job_config = HPCJobConfig(
            job_name='test-chemi-accel',
            nodes=2,
            gpus_per_node=2,
            cpus_per_node=8,
            memory_per_node='32GB',
            walltime='24:00:00',
            partition='gpu',
            account='test-account',
            modules_to_load=['cuda/11.8', 'python/3.9'],
            environment_variables={
                'CUDA_VISIBLE_DEVICES': '0,1',
                'OMP_NUM_THREADS': '8'
            }
        )
    
    @patch('subprocess.run')
    def test_slurm_availability_check(self, mock_run):
        """Test SLURM availability detection."""
        # Mock successful sinfo command
        mock_run.return_value = Mock(returncode=0)
        
        scheduler = SLURMScheduler()
        assert scheduler._check_slurm_available()
        
        mock_run.assert_called_with(
            ['sinfo', '--version'],
            capture_output=True,
            check=True
        )
    
    @patch('subprocess.run')
    def test_slurm_job_submission(self, mock_run):
        """Test SLURM job submission."""
        # Mock successful sbatch command
        mock_run.side_effect = [
            Mock(returncode=0),  # sinfo check
            Mock(stdout='Submitted batch job 12345\n', returncode=0)  # sbatch
        ]
        
        scheduler = SLURMScheduler()
        
        script_content = "echo 'Hello from SLURM'"
        job_id = scheduler.submit_job(self.job_config, script_content)
        
        assert job_id == '12345'
        
        # Verify sbatch was called
        sbatch_call = mock_run.call_args_list[1]
        assert sbatch_call[0][0][0] == 'sbatch'
    
    @patch('subprocess.run')
    def test_slurm_job_status(self, mock_run):
        """Test SLURM job status checking."""
        mock_run.side_effect = [
            Mock(returncode=0),  # sinfo check
            Mock(stdout='RUNNING\n', returncode=0)  # squeue
        ]
        
        scheduler = SLURMScheduler()
        status = scheduler.get_job_status('12345')
        
        assert status == 'RUNNING'
        
        # Verify squeue was called
        squeue_call = mock_run.call_args_list[1]
        assert squeue_call[0][0] == ['squeue', '-j', '12345', '-h', '-o', '%T']
    
    @patch('subprocess.run')
    def test_slurm_job_cancellation(self, mock_run):
        """Test SLURM job cancellation."""
        mock_run.side_effect = [
            Mock(returncode=0),  # sinfo check
            Mock(returncode=0)   # scancel
        ]
        
        scheduler = SLURMScheduler()
        success = scheduler.cancel_job('12345')
        
        assert success
        
        # Verify scancel was called
        scancel_call = mock_run.call_args_list[1]
        assert scancel_call[0][0] == ['scancel', '12345']
    
    @patch('subprocess.run')
    def test_slurm_script_generation(self, mock_run):
        """Test SLURM script generation."""
        mock_run.return_value = Mock(returncode=0)  # sinfo check
        
        scheduler = SLURMScheduler()
        script_content = "python process_molecules.py"
        
        slurm_script = scheduler._create_slurm_script(self.job_config, script_content)
        
        # Verify script contains expected SLURM directives
        assert '#SBATCH --job-name=test-chemi-accel' in slurm_script
        assert '#SBATCH --nodes=2' in slurm_script
        assert '#SBATCH --gres=gpu:2' in slurm_script
        assert '#SBATCH --mem=32GB' in slurm_script
        assert '#SBATCH --time=24:00:00' in slurm_script
        assert '#SBATCH --partition=gpu' in slurm_script
        assert '#SBATCH --account=test-account' in slurm_script
        
        # Verify modules and environment variables
        assert 'module load cuda/11.8' in slurm_script
        assert 'module load python/3.9' in slurm_script
        assert 'export CUDA_VISIBLE_DEVICES=0,1' in slurm_script
        assert 'export OMP_NUM_THREADS=8' in slurm_script
        
        # Verify job execution
        assert 'python process_molecules.py' in slurm_script


class TestPBSScheduler:
    """Test PBS scheduler integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.job_config = HPCJobConfig(
            job_name='test-pbs-job',
            nodes=1,
            gpus_per_node=1,
            cpus_per_node=4,
            memory_per_node='16GB',
            walltime='12:00:00',
            partition='gpu_queue'
        )
    
    @patch('subprocess.run')
    def test_pbs_availability_check(self, mock_run):
        """Test PBS availability detection."""
        mock_run.return_value = Mock(returncode=0)
        
        scheduler = PBSScheduler()
        assert scheduler._check_pbs_available()
    
    @patch('subprocess.run')
    def test_pbs_job_submission(self, mock_run):
        """Test PBS job submission."""
        mock_run.side_effect = [
            Mock(returncode=0),  # qstat check
            Mock(stdout='123.cluster\n', returncode=0)  # qsub
        ]
        
        scheduler = PBSScheduler()
        
        script_content = "echo 'Hello from PBS'"
        job_id = scheduler.submit_job(self.job_config, script_content)
        
        assert job_id == '123.cluster'
    
    @patch('subprocess.run')
    def test_pbs_script_generation(self, mock_run):
        """Test PBS script generation."""
        mock_run.return_value = Mock(returncode=0)  # qstat check
        
        scheduler = PBSScheduler()
        script_content = "python process_molecules.py"
        
        pbs_script = scheduler._create_pbs_script(self.job_config, script_content)
        
        # Verify script contains expected PBS directives
        assert '#PBS -N test-pbs-job' in pbs_script
        assert '#PBS -l nodes=1:ppn=4:gpus=1' in pbs_script
        assert '#PBS -l mem=16GB' in pbs_script
        assert '#PBS -l walltime=12:00:00' in pbs_script
        assert '#PBS -q gpu_queue' in pbs_script
        
        # Verify job execution
        assert 'python process_molecules.py' in pbs_script


class TestMPIIntegration:
    """Test MPI integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_molecules = pd.Series([
            'CCO',
            'CC(=O)O', 
            'c1ccccc1',
            'CCN(CC)CC',
            'CC(C)O',
            'CCCCCCCC',
            'CC(C)(C)O',
            'CCO[CH2]'
        ])
    
    @patch.dict(os.environ, {
        'OMPI_COMM_WORLD_SIZE': '4',
        'OMPI_COMM_WORLD_RANK': '1'
    })
    def test_mpi_environment_detection(self):
        """Test MPI environment detection."""
        mpi = MPIIntegration()
        
        assert mpi.mpi_config.world_size == 4
        assert mpi.mpi_config.rank == 1
        assert mpi.mpi_config.master_addr == 'localhost'
        assert mpi.mpi_config.master_port == '29500'
    
    @patch.dict(os.environ, {
        'OMPI_COMM_WORLD_SIZE': '4',
        'OMPI_COMM_WORLD_RANK': '0'
    })
    def test_workload_distribution(self):
        """Test workload distribution across MPI ranks."""
        mpi = MPIIntegration()
        
        # Test rank 0 (should get first 2 molecules)
        distributed_molecules = mpi.distribute_workload(self.test_molecules)
        
        assert len(distributed_molecules) == 2
        assert distributed_molecules.iloc[0] == 'CCO'
        assert distributed_molecules.iloc[1] == 'CC(=O)O'
    
    @patch.dict(os.environ, {
        'OMPI_COMM_WORLD_SIZE': '4',
        'OMPI_COMM_WORLD_RANK': '2'
    })
    def test_workload_distribution_rank_2(self):
        """Test workload distribution for rank 2."""
        mpi = MPIIntegration()
        
        # Test rank 2 (should get molecules 4-5)
        distributed_molecules = mpi.distribute_workload(self.test_molecules)
        
        assert len(distributed_molecules) == 2
        assert distributed_molecules.iloc[0] == 'CCN(CC)CC'
        assert distributed_molecules.iloc[1] == 'CC(C)O'
    
    @patch('subprocess.run')
    def test_mpi_availability_check(self, mock_run):
        """Test MPI availability detection."""
        mock_run.return_value = Mock(returncode=0)
        
        mpi = MPIIntegration()
        assert mpi._check_mpi_available()
        
        mock_run.assert_called_with(
            ['mpirun', '--version'],
            capture_output=True,
            check=True
        )
    
    @patch('mpi4py.MPI.COMM_WORLD')
    def test_results_gathering(self, mock_comm):
        """Test MPI results gathering."""
        # Mock MPI communicator
        mock_comm.gather.return_value = [
            pd.DataFrame({'smiles': ['CCO'], 'mw': [46.07]}),
            pd.DataFrame({'smiles': ['CC(=O)O'], 'mw': [60.05]}),
            pd.DataFrame({'smiles': ['c1ccccc1'], 'mw': [78.11]}),
            pd.DataFrame({'smiles': ['CCN(CC)CC'], 'mw': [101.19]})
        ]
        
        with patch.dict(os.environ, {'OMPI_COMM_WORLD_RANK': '0'}):
            mpi = MPIIntegration()
            
            local_results = pd.DataFrame({'smiles': ['CCO'], 'mw': [46.07]})
            combined_results = mpi.gather_results(local_results)
            
            assert combined_results is not None
            assert len(combined_results) == 4
            assert 'smiles' in combined_results.columns
            assert 'mw' in combined_results.columns


class TestParallelIOManager:
    """Test parallel I/O manager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_data = pd.DataFrame({
            'smiles': ['CCO', 'CC(=O)O', 'c1ccccc1'],
            'mw': [46.07, 60.05, 78.11],
            'logp': [-0.31, -0.17, 2.13]
        })
    
    def test_lustre_stripe_configuration(self):
        """Test Lustre file system stripe configuration."""
        io_manager = ParallelIOManager(file_system='lustre')
        
        assert io_manager.file_system == 'lustre'
        assert io_manager.stripe_config['stripe_count'] == 4
        assert io_manager.stripe_config['stripe_size'] == '1M'
    
    def test_gpfs_configuration(self):
        """Test GPFS file system configuration."""
        io_manager = ParallelIOManager(file_system='gpfs')
        
        assert io_manager.file_system == 'gpfs'
        assert io_manager.stripe_config['block_size'] == '1M'
        assert io_manager.stripe_config['prefetch_threads'] == 4
    
    @patch('subprocess.run')
    def test_lustre_directory_setup(self, mock_run):
        """Test Lustre directory setup with striping."""
        mock_run.return_value = Mock(returncode=0)
        
        io_manager = ParallelIOManager(file_system='lustre')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            success = io_manager.setup_output_directory(temp_dir)
            
            assert success
            
            # Verify lfs setstripe was called
            lfs_call = mock_run.call_args_list[0]
            assert lfs_call[0][0][0] == 'lfs'
            assert lfs_call[0][0][1] == 'setstripe'
            assert '-c' in lfs_call[0][0]
            assert '-S' in lfs_call[0][0]
    
    def test_parallel_write_parquet(self):
        """Test parallel writing in Parquet format."""
        io_manager = ParallelIOManager()
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            success = io_manager.parallel_write(
                self.test_data, 
                temp_file.name, 
                format='parquet'
            )
            
            assert success
            
            # Verify file was written correctly
            read_data = pd.read_parquet(temp_file.name)
            assert len(read_data) == len(self.test_data)
            assert list(read_data.columns) == list(self.test_data.columns)
            
            # Clean up
            os.unlink(temp_file.name)
    
    def test_parallel_read_csv(self):
        """Test parallel reading of CSV files."""
        io_manager = ParallelIOManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            self.test_data.to_csv(temp_file.name, index=False)
            
            # Read in chunks
            chunks = list(io_manager.parallel_read(temp_file.name, chunk_size=2))
            
            assert len(chunks) == 2  # 3 rows with chunk_size=2
            assert len(chunks[0]) == 2
            assert len(chunks[1]) == 1
            
            # Clean up
            os.unlink(temp_file.name)


class TestHPCIntegration:
    """Test main HPC integration class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.job_config = HPCJobConfig(
            job_name='test-hpc-chemi-accel',
            nodes=2,
            gpus_per_node=1,
            walltime='12:00:00',
            partition='gpu'
        )
    
    @patch('subprocess.run')
    def test_scheduler_auto_detection(self, mock_run):
        """Test automatic scheduler detection."""
        # Mock SLURM available
        mock_run.return_value = Mock(returncode=0)
        
        hpc = HPCIntegration(scheduler='auto')
        
        assert isinstance(hpc.scheduler, SLURMScheduler)
    
    @patch('subprocess.run')
    def test_hpc_job_submission(self, mock_run):
        """Test HPC job submission."""
        mock_run.side_effect = [
            Mock(returncode=0),  # sinfo check
            Mock(stdout='Submitted batch job 12345\n', returncode=0)  # sbatch
        ]
        
        hpc = HPCIntegration(scheduler='slurm')
        
        job_id = hpc.submit_chemi_accel_job(
            job_config=self.job_config,
            input_data_path='/data/molecules.csv',
            output_data_path='/data/results.parquet',
            descriptor_config={'backend': 'cuda', 'batch_size': 1000}
        )
        
        assert job_id == '12345'
    
    @patch('subprocess.run')
    def test_slurm_script_generation(self, mock_run):
        """Test SLURM script generation for ChemiAccel."""
        mock_run.return_value = Mock(returncode=0)  # sinfo check
        
        hpc = HPCIntegration(scheduler='slurm')
        
        script = hpc.generate_slurm_script(
            job_config=self.job_config,
            input_path='/data/molecules.csv',
            output_path='/data/results.parquet',
            descriptor_config={'backend': 'cuda'}
        )
        
        # Verify script contains expected elements
        assert '#SBATCH --job-name=test-hpc-chemi-accel' in script
        assert '#SBATCH --nodes=2' in script
        assert 'mpirun -np $SLURM_NTASKS python3' in script
        assert 'from chemi_accel import Calculator' in script
        assert 'from chemi_accel.cloud.hpc_integration import MPIIntegration' in script
    
    @patch('subprocess.run')
    def test_job_monitoring(self, mock_run):
        """Test job monitoring."""
        mock_run.side_effect = [
            Mock(returncode=0),  # sinfo check
            Mock(stdout='RUNNING\n', returncode=0)  # squeue
        ]
        
        hpc = HPCIntegration(scheduler='slurm')
        status = hpc.monitor_job('12345')
        
        assert status == 'RUNNING'
    
    @patch('subprocess.run')
    def test_job_completion_waiting(self, mock_run):
        """Test waiting for job completion."""
        mock_run.side_effect = [
            Mock(returncode=0),  # sinfo check
            Mock(stdout='RUNNING\n', returncode=0),  # squeue 1
            Mock(stdout='RUNNING\n', returncode=0),  # squeue 2
            Mock(stdout='COMPLETED\n', returncode=0)  # squeue 3
        ]
        
        hpc = HPCIntegration(scheduler='slurm')
        
        # Mock time.sleep to speed up test
        with patch('time.sleep'):
            final_status = hpc.wait_for_completion('12345', poll_interval=1)
        
        assert final_status == 'COMPLETED'


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('subprocess.run')
    def test_submit_hpc_job_utility(self, mock_run):
        """Test submit_hpc_job utility function."""
        mock_run.side_effect = [
            Mock(returncode=0),  # sinfo check
            Mock(stdout='Submitted batch job 12345\n', returncode=0)  # sbatch
        ]
        
        job_config = HPCJobConfig(
            job_name='test-utility-job',
            nodes=1,
            gpus_per_node=1,
            walltime='1:00:00',
            partition='gpu'
        )
        
        job_id = submit_hpc_job(
            scheduler='slurm',
            job_config=job_config,
            input_path='/data/input.csv',
            output_path='/data/output.parquet',
            descriptor_config={'backend': 'cuda'}
        )
        
        assert job_id == '12345'


@pytest.mark.integration
class TestRealHPCIntegration:
    """Integration tests with real HPC systems (requires HPC environment)."""
    
    @pytest.mark.skipif(
        not os.environ.get('SLURM_JOB_ID'),
        reason="Not running in SLURM environment"
    )
    def test_real_slurm_integration(self):
        """Test with real SLURM system (requires SLURM environment)."""
        # This would test actual SLURM integration
        # Only runs when in a SLURM environment
        pass
    
    @pytest.mark.skipif(
        not os.environ.get('PBS_JOBID'),
        reason="Not running in PBS environment"
    )
    def test_real_pbs_integration(self):
        """Test with real PBS system (requires PBS environment)."""
        # This would test actual PBS integration
        # Only runs when in a PBS environment
        pass


if __name__ == '__main__':
    pytest.main([__file__])
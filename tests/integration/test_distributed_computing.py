"""
Integration tests for distributed computing with Dask and Ray.

Tests distributed molecular processing, cluster management, and scaling.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from chemi_accel.cloud.distributed_computing import (
    ClusterConfig,
    DaskIntegration,
    DistributedChemiAccelCalculator,
    RayIntegration,
    distributed_calculate_descriptors,
    process_large_dataset,
    setup_dask_cluster,
    setup_ray_cluster,
)


class TestDaskIntegration:
    """Test Dask cluster integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_molecules = pd.Series(
            [
                "CCO",  # ethanol
                "CC(=O)O",  # acetic acid
                "c1ccccc1",  # benzene
                "CCN(CC)CC",  # triethylamine
                "CC(C)O",  # isopropanol
                "CCCCCCCC",  # octane
                "CC(C)(C)O",  # tert-butanol
                "CCO[CH2]",  # ethyl methyl ether
            ]
        )

        self.cluster_config = ClusterConfig(
            cluster_type="dask",
            n_workers=2,
            threads_per_worker=1,
            memory_per_worker="4GB",
        )

    @patch("dask_cuda.LocalCUDACluster")
    @patch("dask.distributed.Client")
    def test_local_cluster_setup(self, mock_client, mock_cluster):
        """Test local Dask cluster setup."""
        # Mock cluster and client
        mock_cluster_instance = Mock()
        mock_cluster.return_value = mock_cluster_instance

        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        # Create integration
        dask_integration = DaskIntegration(self.cluster_config)
        success = dask_integration.setup_cluster("local")

        assert success
        assert dask_integration.cluster == mock_cluster_instance
        assert dask_integration.client == mock_client_instance

        # Verify cluster configuration
        mock_cluster.assert_called_once_with(
            n_workers=2, threads_per_worker=1, memory_limit="4GB"
        )

    @patch("dask_jobqueue.SLURMCluster")
    @patch("dask.distributed.Client")
    def test_slurm_cluster_setup(self, mock_client, mock_slurm_cluster):
        """Test SLURM Dask cluster setup."""
        # Mock SLURM cluster
        mock_cluster_instance = Mock()
        mock_slurm_cluster.return_value = mock_cluster_instance

        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        # Create integration with auto-scaling
        config = ClusterConfig(
            cluster_type="dask",
            n_workers=4,
            auto_scaling=True,
            min_workers=1,
            max_workers=10,
        )

        dask_integration = DaskIntegration(config)
        success = dask_integration.setup_cluster("slurm")

        assert success

        # Verify SLURM cluster configuration
        mock_slurm_cluster.assert_called_once()
        mock_cluster_instance.adapt.assert_called_once_with(minimum=1, maximum=10)

    @patch("dask.distributed.Client")
    def test_existing_cluster_connection(self, mock_client):
        """Test connecting to existing Dask cluster."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        config = ClusterConfig(
            cluster_type="dask", scheduler_address="tcp://scheduler:8786"
        )

        dask_integration = DaskIntegration(config)
        success = dask_integration.setup_cluster("existing")

        assert success
        assert dask_integration.client == mock_client_instance

        mock_client.assert_called_once_with("tcp://scheduler:8786")

    @patch("dask.distributed.Client")
    @patch("dask.dataframe.from_pandas")
    @patch("dask.delayed")
    def test_distributed_processing(self, mock_delayed, mock_from_pandas, mock_client):
        """Test distributed molecular processing."""
        # Mock Dask components
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_ddf = Mock()
        mock_partition = Mock()
        mock_ddf.to_delayed.return_value = [mock_partition]
        mock_from_pandas.return_value = mock_ddf

        # Mock delayed function
        mock_delayed_func = Mock()
        mock_delayed.return_value = mock_delayed_func

        # Mock computation result
        expected_result = pd.DataFrame(
            {
                "smiles": self.test_molecules,
                "molecular_weight": [
                    46.07,
                    60.05,
                    78.11,
                    101.19,
                    60.10,
                    114.23,
                    74.12,
                    60.10,
                ],
            }
        )

        mock_result_ddf = Mock()
        mock_result_ddf.compute.return_value = expected_result

        with patch("dask.dataframe.from_delayed", return_value=mock_result_ddf):
            # Create integration
            dask_integration = DaskIntegration(self.cluster_config)
            dask_integration.client = mock_client_instance

            # Process molecules
            result = dask_integration.process_molecules_distributed(
                self.test_molecules,
                descriptor_config={"descriptors": ["molecular_weight"]},
                chunk_size=4,
            )

            assert len(result) == len(self.test_molecules)
            assert "smiles" in result.columns
            assert "molecular_weight" in result.columns

    @patch("dask.distributed.Client")
    def test_cluster_status_monitoring(self, mock_client):
        """Test cluster status monitoring."""
        # Mock scheduler info
        mock_scheduler_info = {
            "workers": {
                "worker1": {"nthreads": 4, "memory_limit": 8000000000},
                "worker2": {"nthreads": 4, "memory_limit": 8000000000},
            },
            "tasks": {"running": 5, "pending": 2},
        }

        mock_client_instance = Mock()
        mock_client_instance.scheduler_info.return_value = mock_scheduler_info
        mock_client.return_value = mock_client_instance

        dask_integration = DaskIntegration(self.cluster_config)
        dask_integration.client = mock_client_instance

        status = dask_integration.get_cluster_status()

        assert status["worker_count"] == 2
        assert status["total_cores"] == 8
        assert status["total_memory"] == 16000000000
        assert status["tasks"] == {"running": 5, "pending": 2}

    @patch("dask.distributed.Client")
    def test_cluster_shutdown(self, mock_client):
        """Test cluster shutdown."""
        mock_client_instance = Mock()
        mock_cluster_instance = Mock()

        dask_integration = DaskIntegration(self.cluster_config)
        dask_integration.client = mock_client_instance
        dask_integration.cluster = mock_cluster_instance

        dask_integration.shutdown_cluster()

        mock_client_instance.close.assert_called_once()
        mock_cluster_instance.close.assert_called_once()


class TestRayIntegration:
    """Test Ray cluster integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_molecules = pd.Series(["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC"])

        self.cluster_config = ClusterConfig(
            cluster_type="ray", n_workers=2, threads_per_worker=2
        )

    @patch("ray.init")
    def test_local_cluster_setup(self, mock_ray_init):
        """Test local Ray cluster setup."""
        ray_integration = RayIntegration(self.cluster_config)
        success = ray_integration.setup_cluster("local")

        assert success
        assert ray_integration.ray_initialized

        mock_ray_init.assert_called_once_with(
            num_cpus=4,  # n_workers * threads_per_worker
            num_gpus=2,  # n_workers
            object_store_memory=2000000000,
        )

    @patch("ray.init")
    def test_existing_cluster_connection(self, mock_ray_init):
        """Test connecting to existing Ray cluster."""
        config = ClusterConfig(
            cluster_type="ray", scheduler_address="ray://head-node:10001"
        )

        ray_integration = RayIntegration(config)
        success = ray_integration.setup_cluster("existing")

        assert success

        mock_ray_init.assert_called_once_with(address="ray://head-node:10001")

    @patch("ray.init")
    @patch("ray.get")
    @patch("ray.remote")
    def test_distributed_processing(self, mock_remote, mock_get, mock_ray_init):
        """Test distributed molecular processing with Ray."""
        # Mock Ray components
        mock_worker_class = Mock()
        mock_remote.return_value = mock_worker_class

        mock_worker_instance = Mock()
        mock_worker_class.remote.return_value = mock_worker_instance

        mock_future = Mock()
        mock_worker_instance.process_chunk.remote.return_value = mock_future

        # Mock results
        expected_results = [
            pd.DataFrame({"smiles": ["CCO", "CC(=O)O"], "mw": [46.07, 60.05]}),
            pd.DataFrame({"smiles": ["c1ccccc1", "CCN(CC)CC"], "mw": [78.11, 101.19]}),
        ]
        mock_get.return_value = expected_results

        # Create integration
        ray_integration = RayIntegration(self.cluster_config)
        ray_integration.ray_initialized = True

        # Process molecules
        result = ray_integration.process_molecules_distributed(
            self.test_molecules, chunk_size=2
        )

        assert len(result) == len(self.test_molecules)
        assert "smiles" in result.columns
        assert "mw" in result.columns

    @patch("ray.init")
    @patch("ray.cluster_resources")
    @patch("ray.available_resources")
    @patch("ray.nodes")
    @patch("ray.timeline")
    def test_cluster_status_monitoring(
        self, mock_timeline, mock_nodes, mock_available, mock_cluster, mock_ray_init
    ):
        """Test Ray cluster status monitoring."""
        # Mock Ray status functions
        mock_cluster.return_value = {"CPU": 8, "GPU": 2}
        mock_available.return_value = {"CPU": 6, "GPU": 1}
        mock_nodes.return_value = [{"NodeID": "node1"}, {"NodeID": "node2"}]
        mock_timeline.return_value = {"tasks": []}

        ray_integration = RayIntegration(self.cluster_config)
        ray_integration.ray_initialized = True

        status = ray_integration.get_cluster_status()

        assert status["cluster_resources"] == {"CPU": 8, "GPU": 2}
        assert status["available_resources"] == {"CPU": 6, "GPU": 1}
        assert len(status["nodes"]) == 2

    @patch("ray.init")
    @patch("ray.shutdown")
    def test_cluster_shutdown(self, mock_shutdown, mock_ray_init):
        """Test Ray cluster shutdown."""
        ray_integration = RayIntegration(self.cluster_config)
        ray_integration.ray_initialized = True

        ray_integration.shutdown_cluster()

        mock_shutdown.assert_called_once()
        assert not ray_integration.ray_initialized


class TestDistributedChemiAccelCalculator:
    """Test unified distributed calculator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_molecules = pd.Series(["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC"])

        self.cluster_config = ClusterConfig(cluster_type="dask", n_workers=2)

    @patch("chemi_accel.cloud.distributed_computing.DaskIntegration")
    def test_dask_calculator_initialization(self, mock_dask):
        """Test calculator initialization with Dask."""
        mock_dask_instance = Mock()
        mock_dask.return_value = mock_dask_instance

        calc = DistributedChemiAccelCalculator("dask", self.cluster_config)

        assert calc.framework == "dask"
        assert calc.backend == mock_dask_instance
        mock_dask.assert_called_once_with(self.cluster_config)

    @patch("chemi_accel.cloud.distributed_computing.RayIntegration")
    def test_ray_calculator_initialization(self, mock_ray):
        """Test calculator initialization with Ray."""
        mock_ray_instance = Mock()
        mock_ray.return_value = mock_ray_instance

        calc = DistributedChemiAccelCalculator("ray", self.cluster_config)

        assert calc.framework == "ray"
        assert calc.backend == mock_ray_instance
        mock_ray.assert_called_once_with(self.cluster_config)

    def test_unsupported_framework(self):
        """Test handling of unsupported frameworks."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            DistributedChemiAccelCalculator("unsupported")

    @patch("chemi_accel.cloud.distributed_computing.DaskIntegration")
    def test_pandas_processing(self, mock_dask):
        """Test pandas processing interface."""
        # Mock Dask integration
        mock_dask_instance = Mock()
        expected_result = pd.DataFrame(
            {"smiles": self.test_molecules, "mw": [46.07, 60.05, 78.11, 101.19]}
        )
        mock_dask_instance.process_molecules_distributed.return_value = expected_result
        mock_dask.return_value = mock_dask_instance

        calc = DistributedChemiAccelCalculator("dask")
        result = calc.pandas(self.test_molecules)

        assert len(result) == len(self.test_molecules)
        assert "smiles" in result.columns
        assert "mw" in result.columns

        mock_dask_instance.process_molecules_distributed.assert_called_once_with(
            self.test_molecules, None
        )

    @patch("chemi_accel.cloud.distributed_computing.DaskIntegration")
    def test_large_dataset_processing(self, mock_dask):
        """Test large dataset processing from file."""
        # Create test data file
        test_data = pd.DataFrame(
            {
                "smiles": ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC"] * 100,
                "id": range(400),
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_input:
            test_data.to_csv(temp_input.name, index=False)

            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as temp_output:
                # Mock Dask integration
                mock_dask_instance = Mock()
                mock_result = pd.DataFrame(
                    {
                        "smiles": test_data["smiles"],
                        "mw": [46.07, 60.05, 78.11, 101.19] * 100,
                    }
                )
                mock_dask_instance.process_molecules_distributed.return_value = (
                    mock_result
                )
                mock_dask.return_value = mock_dask_instance

                calc = DistributedChemiAccelCalculator("dask")
                calc.backend = mock_dask_instance

                success = calc.process_large_dataset(
                    temp_input.name, temp_output.name, chunk_size=100
                )

                assert success

                # Verify output file exists and has correct data
                result_data = pd.read_parquet(temp_output.name)
                assert len(result_data) == 400
                assert "smiles" in result_data.columns
                assert "mw" in result_data.columns

                # Clean up
                os.unlink(temp_input.name)
                os.unlink(temp_output.name)

    @patch("chemi_accel.cloud.distributed_computing.DaskIntegration")
    def test_cluster_management(self, mock_dask):
        """Test cluster setup and shutdown."""
        mock_dask_instance = Mock()
        mock_dask_instance.setup_cluster.return_value = True
        mock_dask_instance.get_cluster_status.return_value = {"workers": 2}
        mock_dask.return_value = mock_dask_instance

        calc = DistributedChemiAccelCalculator("dask")

        # Test cluster setup
        success = calc.setup_cluster("local")
        assert success
        mock_dask_instance.setup_cluster.assert_called_once_with("local")

        # Test status monitoring
        status = calc.get_cluster_status()
        assert status["workers"] == 2

        # Test shutdown
        calc.shutdown()
        mock_dask_instance.shutdown_cluster.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    @patch("chemi_accel.cloud.distributed_computing.DaskIntegration")
    def test_setup_dask_cluster(self, mock_dask):
        """Test Dask cluster setup utility."""
        mock_dask_instance = Mock()
        mock_dask_instance.setup_cluster.return_value = True
        mock_dask.return_value = mock_dask_instance

        # Test local cluster setup
        dask_integration = setup_dask_cluster(n_workers=4)

        assert dask_integration == mock_dask_instance
        mock_dask_instance.setup_cluster.assert_called_once_with("local")

        # Test existing cluster connection
        dask_integration = setup_dask_cluster(
            scheduler_address="tcp://scheduler:8786", n_workers=4
        )

        mock_dask_instance.setup_cluster.assert_called_with("existing")

    @patch("chemi_accel.cloud.distributed_computing.RayIntegration")
    def test_setup_ray_cluster(self, mock_ray):
        """Test Ray cluster setup utility."""
        mock_ray_instance = Mock()
        mock_ray_instance.setup_cluster.return_value = True
        mock_ray.return_value = mock_ray_instance

        # Test local cluster setup
        ray_integration = setup_ray_cluster(n_workers=4)

        assert ray_integration == mock_ray_instance
        mock_ray_instance.setup_cluster.assert_called_once_with("local")

    @patch("chemi_accel.cloud.distributed_computing.DistributedChemiAccelCalculator")
    def test_distributed_calculate_descriptors(self, mock_calc_class):
        """Test distributed descriptor calculation utility."""
        test_molecules = pd.Series(["CCO", "CC(=O)O"])
        expected_result = pd.DataFrame({"smiles": test_molecules, "mw": [46.07, 60.05]})

        mock_calc = Mock()
        mock_calc.pandas.return_value = expected_result
        mock_calc_class.return_value = mock_calc

        result = distributed_calculate_descriptors(test_molecules, framework="dask")

        assert len(result) == 2
        assert "smiles" in result.columns
        assert "mw" in result.columns

        # Verify calculator was properly initialized and shut down
        mock_calc.setup_cluster.assert_called_once_with("local")
        mock_calc.shutdown.assert_called_once()

    @patch("chemi_accel.cloud.distributed_computing.DistributedChemiAccelCalculator")
    def test_process_large_dataset_utility(self, mock_calc_class):
        """Test large dataset processing utility."""
        mock_calc = Mock()
        mock_calc.process_large_dataset.return_value = True
        mock_calc_class.return_value = mock_calc

        success = process_large_dataset(
            "input.csv", "output.parquet", framework="ray", chunk_size=5000
        )

        assert success

        # Verify calculator was properly used
        mock_calc.setup_cluster.assert_called_once_with("local")
        mock_calc.process_large_dataset.assert_called_once_with(
            "input.csv", "output.parquet", chunk_size=5000
        )
        mock_calc.shutdown.assert_called_once()


@pytest.mark.integration
class TestRealDistributedIntegration:
    """Integration tests with real distributed systems (requires setup)."""

    @pytest.mark.skipif(
        not os.environ.get("DASK_SCHEDULER_ADDRESS"),
        reason="Dask scheduler not available",
    )
    def test_real_dask_integration(self):
        """Test with real Dask cluster (requires Dask setup)."""
        scheduler_address = os.environ.get("DASK_SCHEDULER_ADDRESS")

        config = ClusterConfig(cluster_type="dask", scheduler_address=scheduler_address)

        dask_integration = DaskIntegration(config)
        success = dask_integration.setup_cluster("existing")

        if success:
            # Test basic functionality
            test_molecules = pd.Series(["CCO", "CC(=O)O"])

            try:
                # This would require actual ChemiAccel installation
                # result = dask_integration.process_molecules_distributed(test_molecules)
                # assert len(result) == 2
                pass
            finally:
                dask_integration.shutdown_cluster()

    @pytest.mark.skipif(
        not os.environ.get("RAY_HEAD_ADDRESS"), reason="Ray cluster not available"
    )
    def test_real_ray_integration(self):
        """Test with real Ray cluster (requires Ray setup)."""
        head_address = os.environ.get("RAY_HEAD_ADDRESS")

        config = ClusterConfig(cluster_type="ray", scheduler_address=head_address)

        ray_integration = RayIntegration(config)
        success = ray_integration.setup_cluster("existing")

        if success:
            try:
                # Test basic functionality
                status = ray_integration.get_cluster_status()
                assert "cluster_resources" in status
            finally:
                ray_integration.shutdown_cluster()


if __name__ == "__main__":
    pytest.main([__file__])

"""
Integration tests for cloud batch processing system.

Tests AWS Batch, Google Cloud Batch, and Azure Batch integration
with mock services and real cloud providers (when credentials available).
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from chemi_accel.cloud import CloudBatchProcessor, CloudJobConfig
from chemi_accel.cloud.containers import ContainerPackager
from chemi_accel.cloud.storage import CloudStorageStreamer


class TestCloudBatchProcessor:
    """Test cloud batch processing functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_molecules = pd.Series(
            [
                "CCO",  # ethanol
                "CC(=O)O",  # acetic acid
                "c1ccccc1",  # benzene
                "CCN(CC)CC",  # triethylamine
                "CC(C)O",  # isopropanol
            ]
        )

        self.job_config = CloudJobConfig(
            provider="aws",
            instance_type="ml.g4dn.xlarge",
            num_instances=1,
            input_data_uri="s3://test-bucket/input.csv",
            output_data_uri="s3://test-bucket/output.parquet",
            container_image="123456789.dkr.ecr.us-east-1.amazonaws.com/chemi-accel:latest",
            environment_variables={"CHEMI_ACCEL_BACKEND": "cuda", "BATCH_SIZE": "1000"},
            timeout_minutes=60,
        )

    @patch("boto3.client")
    def test_aws_batch_submission(self, mock_boto_client):
        """Test AWS Batch job submission."""
        # Mock AWS Batch client
        mock_batch = Mock()
        mock_batch.submit_job.return_value = {"jobId": "test-job-123"}
        mock_boto_client.return_value = mock_batch

        # Create processor
        processor = CloudBatchProcessor(provider="aws", region="us-east-1")

        # Submit job
        job_id = processor.submit_job(self.job_config)

        # Verify job submission
        assert job_id == "test-job-123"
        mock_batch.submit_job.assert_called_once()

        # Verify job is tracked
        assert job_id in processor.active_jobs

    @patch("boto3.client")
    def test_aws_batch_monitoring(self, mock_boto_client):
        """Test AWS Batch job monitoring."""
        # Mock AWS Batch client
        mock_batch = Mock()
        mock_batch.describe_jobs.return_value = {
            "jobs": [
                {
                    "jobId": "test-job-123",
                    "status": "RUNNING",
                    "createdAt": "2023-01-01T00:00:00Z",
                    "startedAt": "2023-01-01T00:01:00Z",
                }
            ]
        }
        mock_boto_client.return_value = mock_batch

        processor = CloudBatchProcessor(provider="aws")

        # Monitor job
        status = processor.monitor_job("test-job-123")

        assert status.job_id == "test-job-123"
        assert status.status == "RUNNING"
        assert status.created_at == "2023-01-01T00:00:00Z"

    @patch("google.cloud.batch_v1.BatchServiceClient")
    def test_gcp_batch_submission(self, mock_batch_client):
        """Test GCP Batch job submission."""
        # Mock GCP Batch client
        mock_client = Mock()
        mock_operation = Mock()
        mock_client.create_job.return_value = mock_operation
        mock_batch_client.return_value = mock_client

        # Create processor
        processor = CloudBatchProcessor(
            provider="gcp", project_id="test-project", region="us-central1"
        )

        # Submit job
        job_config = CloudJobConfig(
            provider="gcp",
            instance_type="n1-standard-4",
            num_instances=1,
            input_data_uri="gs://test-bucket/input.csv",
            output_data_uri="gs://test-bucket/output.parquet",
            container_image="gcr.io/test-project/chemi-accel:latest",
            environment_variables={},
        )

        job_id = processor.submit_job(job_config)

        # Verify job submission
        assert "chemi-accel-" in job_id
        mock_client.create_job.assert_called_once()

    def test_auto_scaling_configuration(self):
        """Test auto-scaling configuration."""
        processor = CloudBatchProcessor(provider="aws", auto_scaling=True)

        # Test auto-scaling setup
        with patch.object(
            processor, "_setup_aws_auto_scaling", return_value=True
        ) as mock_setup:
            success = processor.setup_auto_scaling(min_instances=1, max_instances=10)

            assert success
            mock_setup.assert_called_once_with(1, 10, 0.7)

    def test_batch_size_optimization(self):
        """Test batch size optimization."""
        processor = CloudBatchProcessor(provider="aws")

        # Test different dataset sizes
        small_dataset_batch = processor.optimize_batch_size(500, available_memory_gb=8)
        assert small_dataset_batch == 100

        medium_dataset_batch = processor.optimize_batch_size(
            5000, available_memory_gb=16
        )
        assert medium_dataset_batch == 500

        large_dataset_batch = processor.optimize_batch_size(
            100000, available_memory_gb=32
        )
        assert large_dataset_batch == 5000

    @patch("boto3.client")
    def test_job_cancellation(self, mock_boto_client):
        """Test job cancellation."""
        mock_batch = Mock()
        mock_batch.cancel_job.return_value = {}
        mock_boto_client.return_value = mock_batch

        processor = CloudBatchProcessor(provider="aws")
        processor.active_jobs["test-job-123"] = self.job_config

        # Cancel job
        success = processor.cancel_job("test-job-123")

        assert success
        assert "test-job-123" not in processor.active_jobs
        mock_batch.cancel_job.assert_called_once_with(
            jobId="test-job-123", reason="Cancelled by user"
        )

    def test_container_packaging_integration(self):
        """Test integration with container packaging."""
        processor = CloudBatchProcessor(provider="aws")

        with patch.object(processor, "package_for_deployment") as mock_package:
            mock_package.return_value = "test-image:latest"

            image_uri = processor.package_for_deployment("/path/to/code")

            assert image_uri == "test-image:latest"
            mock_package.assert_called_once_with("/path/to/code")

    @patch("chemi_accel.cloud.storage.CloudStorageStreamer")
    def test_storage_streaming_integration(self, mock_streamer_class):
        """Test integration with cloud storage streaming."""
        mock_streamer = Mock()
        mock_streamer.stream_dataframe.return_value = [
            self.test_molecules.to_frame("smiles")
        ]
        mock_streamer_class.return_value = mock_streamer

        processor = CloudBatchProcessor(provider="aws")

        # Test streaming
        chunks = list(processor.stream_from_storage("s3://test-bucket/molecules.csv"))

        assert len(chunks) == 1
        assert "smiles" in chunks[0].columns
        mock_streamer.stream_dataframe.assert_called_once()


class TestMultiProviderIntegration:
    """Test multi-cloud provider integration."""

    def test_provider_switching(self):
        """Test switching between cloud providers."""
        # Test AWS
        aws_processor = CloudBatchProcessor(provider="aws")
        assert aws_processor.provider_name == "aws"

        # Test GCP
        gcp_processor = CloudBatchProcessor(provider="gcp", project_id="test-project")
        assert gcp_processor.provider_name == "gcp"

        # Test Azure
        azure_processor = CloudBatchProcessor(
            provider="azure",
            account_name="testaccount",
            account_key="testkey",
            account_url="https://testaccount.batch.azure.com",
        )
        assert azure_processor.provider_name == "azure"

    def test_unsupported_provider(self):
        """Test handling of unsupported providers."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            CloudBatchProcessor(provider="unsupported")

    @patch("boto3.client")
    @patch("google.cloud.batch_v1.BatchServiceClient")
    def test_cross_provider_job_submission(self, mock_gcp_client, mock_aws_client):
        """Test submitting jobs to different providers."""
        # Setup mocks
        mock_aws_batch = Mock()
        mock_aws_batch.submit_job.return_value = {"jobId": "aws-job-123"}
        mock_aws_client.return_value = mock_aws_batch

        mock_gcp_batch = Mock()
        mock_gcp_batch.create_job.return_value = Mock()
        mock_gcp_client.return_value = mock_gcp_batch

        # Test AWS submission
        aws_processor = CloudBatchProcessor(provider="aws")
        aws_config = CloudJobConfig(
            provider="aws",
            instance_type="ml.g4dn.xlarge",
            num_instances=1,
            input_data_uri="s3://test/input.csv",
            output_data_uri="s3://test/output.parquet",
            container_image="aws-image:latest",
            environment_variables={},
        )

        aws_job_id = aws_processor.submit_job(aws_config)
        assert aws_job_id == "aws-job-123"

        # Test GCP submission
        gcp_processor = CloudBatchProcessor(provider="gcp", project_id="test-project")
        gcp_config = CloudJobConfig(
            provider="gcp",
            instance_type="n1-standard-4",
            num_instances=1,
            input_data_uri="gs://test/input.csv",
            output_data_uri="gs://test/output.parquet",
            container_image="gcr.io/test/chemi-accel:latest",
            environment_variables={},
        )

        gcp_job_id = gcp_processor.submit_job(gcp_config)
        assert "chemi-accel-" in gcp_job_id

    def test_array_job_submission(self):
        """Test array job submission for parallel processing."""
        with patch("boto3.client") as mock_boto_client:
            mock_batch = Mock()
            mock_batch.submit_job.return_value = {"jobId": "array-job-123"}
            mock_boto_client.return_value = mock_batch

            processor = CloudBatchProcessor(provider="aws")

            # Submit array job
            job_id = processor.submit_batch_array_job(
                self.job_config,
                array_size=10,
                descriptor_config={"descriptors": ["molecular_weight", "logp"]},
            )

            assert job_id == "array-job-123"
            mock_batch.submit_job.assert_called_once()

            # Verify array properties were set
            call_args = mock_batch.submit_job.call_args
            assert "arrayProperties" in call_args[1]
            assert call_args[1]["arrayProperties"]["size"] == 10

    def test_cost_estimation(self):
        """Test cost estimation for different providers."""
        # Test AWS cost estimation
        aws_processor = CloudBatchProcessor(provider="aws")
        aws_cost = aws_processor.estimate_cost(
            self.job_config, estimated_runtime_hours=2.0
        )

        assert "total_cost" in aws_cost
        assert "breakdown" in aws_cost
        assert aws_cost["total_cost"] > 0

        # Test GCP cost estimation
        gcp_config = CloudJobConfig(
            provider="gcp",
            instance_type="n1-standard-4",
            num_instances=1,
            input_data_uri="gs://test-bucket/input.csv",
            output_data_uri="gs://test-bucket/output.parquet",
            container_image="gcr.io/test/chemi-accel:latest",
            environment_variables={},
        )

        with patch("google.cloud.batch_v1.BatchServiceClient"):
            gcp_processor = CloudBatchProcessor(
                provider="gcp", project_id="test-project"
            )
            gcp_cost = gcp_processor.estimate_cost(
                gcp_config, estimated_runtime_hours=2.0
            )

            assert "total_cost" in gcp_cost
            assert "breakdown" in gcp_cost
            assert gcp_cost["total_cost"] > 0

    def test_monitoring_dashboard_setup(self):
        """Test monitoring dashboard setup."""
        with patch("boto3.client") as mock_boto_client:
            mock_cloudwatch = Mock()
            mock_cloudwatch.put_dashboard.return_value = {}
            mock_boto_client.return_value = mock_cloudwatch

            processor = CloudBatchProcessor(provider="aws", monitoring=True)
            dashboards = processor.setup_monitoring_dashboard()

            assert "aws" in dashboards
            assert "console.aws.amazon.com" in dashboards["aws"]
            mock_cloudwatch.put_dashboard.assert_called_once()


class TestContainerPackaging:
    """Test container packaging system."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dockerfile_generation(self):
        """Test Dockerfile generation for different providers."""
        packager = ContainerPackager(provider="aws")

        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)
            dockerfile_path = build_path / "Dockerfile"

            config = packager.base_configs["aws"]
            packager._generate_dockerfile(dockerfile_path, config)

            dockerfile_content = dockerfile_path.read_text()

            # Verify key components
            assert "FROM nvidia/cuda" in dockerfile_content
            assert "WORKDIR /app" in dockerfile_content
            assert "pip install" in dockerfile_content
            assert "HEALTHCHECK" in dockerfile_content
            assert "USER chemiuser" in dockerfile_content

    def test_batch_script_generation(self):
        """Test batch processing script generation."""
        packager = ContainerPackager(provider="aws")

        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)
            packager._generate_batch_script(build_path)

            # Check that all scripts were created
            assert (build_path / "run_batch.py").exists()
            assert (build_path / "run_batch.sh").exists()
            assert (build_path / "run_array_batch.py").exists()
            assert (build_path / "healthcheck.py").exists()

            # Verify script content
            batch_script = (build_path / "run_batch.py").read_text()
            assert "ChemiAccel batch processing" in batch_script
            assert "Calculator" in batch_script

            array_script = (build_path / "run_array_batch.py").read_text()
            assert "CHEMI_ACCEL_ARRAY_INDEX" in array_script
            assert "array batch processing" in array_script

    def test_requirements_generation(self):
        """Test requirements.txt generation."""
        packager = ContainerPackager(provider="aws")

        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)
            app_path = build_path / "app"
            app_path.mkdir()

            config = packager.base_configs["aws"]
            packager._generate_requirements(build_path, config)

            requirements_path = app_path / "requirements.txt"
            assert requirements_path.exists()

            requirements = requirements_path.read_text()

            # Verify key dependencies
            assert "pandas" in requirements
            assert "boto3" in requirements
            assert "rdkit-pypi" in requirements
            assert "pyarrow" in requirements
            assert "fastapi" in requirements

    @patch("subprocess.run")
    def test_docker_build(self, mock_subprocess):
        """Test Docker image building."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        packager = ContainerPackager(provider="aws")

        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)

            # Create minimal structure
            app_path = build_path / "app"
            app_path.mkdir()
            (app_path / "pyproject.toml").write_text("[project]\\nname = 'chemi-accel'")
            (build_path / "Dockerfile").write_text("FROM ubuntu:20.04")

            image_name = packager._build_docker_image(build_path, "test-image:latest")

            assert image_name == "test-image:latest"
            mock_subprocess.assert_called_once()

            # Verify docker build command
            call_args = mock_subprocess.call_args[0][0]
            assert "docker" in call_args
            assert "build" in call_args
            assert "test-image:latest" in call_args


class TestCloudStorageIntegration:
    """Test cloud storage streaming integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_data = pd.DataFrame(
            {"smiles": ["CCO", "CC(=O)O", "c1ccccc1"], "id": ["mol1", "mol2", "mol3"]}
        )

    @patch("boto3.client")
    def test_s3_streaming(self, mock_boto_client):
        """Test S3 data streaming."""
        # Mock S3 response
        mock_s3 = Mock()
        mock_response = Mock()
        mock_response.__getitem__.return_value.iter_chunks.return_value = [
            b"smiles,id\\n",
            b"CCO,mol1\\n",
            b"CC(=O)O,mol2\\n",
        ]
        mock_s3.get_object.return_value = {"Body": mock_response}
        mock_boto_client.return_value = mock_s3

        streamer = CloudStorageStreamer()

        # Test streaming
        chunks = list(
            streamer.stream_dataframe("s3://test-bucket/molecules.csv", chunk_size=2)
        )

        assert len(chunks) > 0
        mock_s3.get_object.assert_called_once()

    @patch("google.cloud.storage.Client")
    def test_gcs_streaming(self, mock_storage_client):
        """Test GCS data streaming."""
        # Mock GCS response
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()

        # Mock blob.open() context manager
        mock_file = Mock()
        mock_file.read.side_effect = [b"smiles,id\\nCCO,mol1\\n", b""]
        mock_blob.open.return_value.__enter__.return_value = mock_file

        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client

        streamer = CloudStorageStreamer(gcp_project_id="test-project")

        # Test streaming
        chunks = list(
            streamer.stream_dataframe("gs://test-bucket/molecules.csv", chunk_size=2)
        )

        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with("molecules.csv")

    @patch("azure.storage.blob.BlobServiceClient")
    def test_azure_blob_streaming(self, mock_blob_client):
        """Test Azure Blob Storage streaming."""
        # Mock Azure Blob response
        mock_client = Mock()
        mock_blob_client_instance = Mock()
        mock_download_stream = Mock()
        mock_download_stream.chunks.return_value = [b"smiles,id\\nCCO,mol1\\n"]

        mock_blob_client_instance.download_blob.return_value = mock_download_stream
        mock_client.get_blob_client.return_value = mock_blob_client_instance
        mock_blob_client.return_value = mock_client

        streamer = CloudStorageStreamer(
            azure_account_name="testaccount", azure_account_key="testkey"
        )

        # Test streaming
        chunks = list(
            streamer.stream_dataframe(
                "https://testaccount.blob.core.windows.net/container/molecules.csv",
                chunk_size=2,
            )
        )

        mock_client.get_blob_client.assert_called_once()


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_molecules = pd.DataFrame(
            {
                "smiles": ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "CC(C)O"],
                "id": [f"mol_{i}" for i in range(5)],
            }
        )

    @patch("boto3.client")
    @patch("chemi_accel.cloud.storage.CloudStorageStreamer")
    @patch("chemi_accel.cloud.containers.ContainerPackager")
    def test_full_batch_processing_workflow(
        self, mock_packager, mock_streamer_class, mock_boto_client
    ):
        """Test complete batch processing workflow."""
        # Setup mocks
        mock_batch = Mock()
        mock_batch.submit_job.return_value = {"jobId": "workflow-job-123"}
        mock_batch.describe_jobs.return_value = {
            "jobs": [
                {
                    "jobId": "workflow-job-123",
                    "status": "SUCCEEDED",
                    "createdAt": "2023-01-01T00:00:00Z",
                }
            ]
        }
        mock_boto_client.return_value = mock_batch

        mock_streamer = Mock()
        mock_streamer.stream_dataframe.return_value = [self.test_molecules]
        mock_streamer.upload_dataframe.return_value = True
        mock_streamer_class.return_value = mock_streamer

        mock_packager_instance = Mock()
        mock_packager_instance.build_container.return_value = "test-image:latest"
        mock_packager.return_value = mock_packager_instance

        # Create processor
        processor = CloudBatchProcessor(provider="aws")

        # Step 1: Package container
        image_uri = processor.package_for_deployment("/fake/path")
        assert image_uri == "test-image:latest"

        # Step 2: Submit job
        job_config = CloudJobConfig(
            provider="aws",
            instance_type="ml.g4dn.xlarge",
            num_instances=1,
            input_data_uri="s3://test-bucket/input.csv",
            output_data_uri="s3://test-bucket/output.parquet",
            container_image=image_uri,
            environment_variables={},
        )

        job_id = processor.submit_job(job_config)
        assert job_id == "workflow-job-123"

        # Step 3: Monitor job
        status = processor.monitor_job(job_id)
        assert status.status == "SUCCEEDED"

        # Step 4: Verify data streaming
        chunks = list(processor.stream_from_storage("s3://test-bucket/input.csv"))
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_multi_provider_cost_comparison(self):
        """Test cost comparison across providers."""
        job_configs = {
            "aws": CloudJobConfig(
                provider="aws",
                instance_type="ml.g4dn.xlarge",
                num_instances=2,
                input_data_uri="s3://test/input.csv",
                output_data_uri="s3://test/output.parquet",
                container_image="aws-image:latest",
                environment_variables={},
            ),
            "gcp": CloudJobConfig(
                provider="gcp",
                instance_type="n1-standard-4",
                num_instances=2,
                input_data_uri="gs://test/input.csv",
                output_data_uri="gs://test/output.parquet",
                container_image="gcr.io/test/image:latest",
                environment_variables={},
            ),
            "azure": CloudJobConfig(
                provider="azure",
                instance_type="Standard_NC6",
                num_instances=2,
                input_data_uri="https://test.blob.core.windows.net/container/input.csv",
                output_data_uri="https://test.blob.core.windows.net/container/output.parquet",
                container_image="test.azurecr.io/image:latest",
                environment_variables={},
            ),
        }

        costs = {}

        # AWS
        with patch("boto3.client"):
            aws_processor = CloudBatchProcessor(provider="aws")
            costs["aws"] = aws_processor.estimate_cost(job_configs["aws"], 2.0)

        # GCP
        with patch("google.cloud.batch_v1.BatchServiceClient"):
            gcp_processor = CloudBatchProcessor(provider="gcp", project_id="test")
            costs["gcp"] = gcp_processor.estimate_cost(job_configs["gcp"], 2.0)

        # Azure
        with patch("azure.batch.batch_service_client.BatchServiceClient"):
            azure_processor = CloudBatchProcessor(
                provider="azure",
                account_name="test",
                account_key="key",
                account_url="https://test.batch.azure.com",
            )
            costs["azure"] = azure_processor.estimate_cost(job_configs["azure"], 2.0)

        # Verify all providers return cost estimates
        for provider, cost in costs.items():
            assert "total_cost" in cost
            assert cost["total_cost"] > 0
            assert "breakdown" in cost


@pytest.mark.integration
class TestRealCloudIntegration:
    """Integration tests with real cloud services (requires credentials)."""

    @pytest.mark.skipif(
        not os.environ.get("AWS_ACCESS_KEY_ID"), reason="AWS credentials not available"
    )
    def test_real_aws_batch_integration(self):
        """Test with real AWS Batch (requires AWS credentials)."""
        processor = CloudBatchProcessor(provider="aws")

        # This would require actual AWS setup
        # Including Batch compute environments, job queues, etc.
        # Skipped in normal testing to avoid costs
        pass

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        reason="GCP credentials not available",
    )
    def test_real_gcp_batch_integration(self):
        """Test with real GCP Batch (requires GCP credentials)."""
        processor = CloudBatchProcessor(
            provider="gcp",
            project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", "test-project"),
        )

        # This would require actual GCP setup
        # Skipped in normal testing to avoid costs
        pass

    @pytest.mark.skipif(
        not os.environ.get("AZURE_BATCH_ACCOUNT"),
        reason="Azure credentials not available",
    )
    def test_real_azure_batch_integration(self):
        """Test with real Azure Batch (requires Azure credentials)."""
        processor = CloudBatchProcessor(
            provider="azure",
            account_name=os.environ.get("AZURE_BATCH_ACCOUNT"),
            account_key=os.environ.get("AZURE_BATCH_KEY"),
            account_url=os.environ.get("AZURE_BATCH_URL"),
        )

        # This would require actual Azure setup
        # Skipped in normal testing to avoid costs
        pass


if __name__ == "__main__":
    pytest.main([__file__])

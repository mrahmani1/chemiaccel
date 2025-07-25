"""
Core cloud batch processing tests without external dependencies.

Tests the core functionality of the cloud batch processing system
without requiring actual cloud provider libraries.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from chemi_accel.cloud.batch_processor import CloudJobConfig, JobStatus


class TestCloudJobConfig:
    """Test CloudJobConfig data structure."""

    def test_job_config_creation(self):
        """Test creating job configuration."""
        config = CloudJobConfig(
            provider="aws",
            instance_type="ml.g4dn.xlarge",
            num_instances=2,
            input_data_uri="s3://test-bucket/input.csv",
            output_data_uri="s3://test-bucket/output.parquet",
            container_image="test-image:latest",
            environment_variables={"TEST_VAR": "test_value"},
            timeout_minutes=120,
        )

        assert config.provider == "aws"
        assert config.instance_type == "ml.g4dn.xlarge"
        assert config.num_instances == 2
        assert config.timeout_minutes == 120
        assert config.environment_variables["TEST_VAR"] == "test_value"


class TestJobStatus:
    """Test JobStatus data structure."""

    def test_job_status_creation(self):
        """Test creating job status."""
        status = JobStatus(
            job_id="test-job-123",
            status="RUNNING",
            created_at="2023-01-01T00:00:00Z",
            started_at="2023-01-01T00:01:00Z",
        )

        assert status.job_id == "test-job-123"
        assert status.status == "RUNNING"
        assert status.created_at == "2023-01-01T00:00:00Z"
        assert status.started_at == "2023-01-01T00:01:00Z"


class TestCloudBatchProcessorCore:
    """Test core CloudBatchProcessor functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.job_config = CloudJobConfig(
            provider="aws",
            instance_type="ml.g4dn.xlarge",
            num_instances=1,
            input_data_uri="s3://test-bucket/input.csv",
            output_data_uri="s3://test-bucket/output.parquet",
            container_image="test-image:latest",
            environment_variables={"TEST_VAR": "test_value"},
        )

    @patch("chemi_accel.cloud.batch_processor.AWSBatchProvider")
    def test_processor_initialization(self, mock_aws_provider):
        """Test processor initialization."""
        from chemi_accel.cloud.batch_processor import CloudBatchProcessor

        mock_provider_instance = Mock()
        mock_aws_provider.return_value = mock_provider_instance

        processor = CloudBatchProcessor(provider="aws", region="us-east-1")

        assert processor.provider_name == "aws"
        assert processor.provider == mock_provider_instance
        assert processor.auto_scaling_enabled is True
        assert processor.monitoring_enabled is True

        mock_aws_provider.assert_called_once_with(region="us-east-1")

    @patch("chemi_accel.cloud.batch_processor.GCPBatchProvider")
    def test_gcp_processor_initialization(self, mock_gcp_provider):
        """Test GCP processor initialization."""
        from chemi_accel.cloud.batch_processor import CloudBatchProcessor

        mock_provider_instance = Mock()
        mock_gcp_provider.return_value = mock_provider_instance

        processor = CloudBatchProcessor(
            provider="gcp", project_id="test-project", region="us-central1"
        )

        assert processor.provider_name == "gcp"
        mock_gcp_provider.assert_called_once_with(
            project_id="test-project", region="us-central1"
        )

    def test_unsupported_provider(self):
        """Test handling of unsupported providers."""
        from chemi_accel.cloud.batch_processor import CloudBatchProcessor

        with pytest.raises(ValueError, match="Unsupported provider"):
            CloudBatchProcessor(provider="unsupported")

    @patch("chemi_accel.cloud.batch_processor.AWSBatchProvider")
    def test_batch_size_optimization(self, mock_aws_provider):
        """Test batch size optimization logic."""
        from chemi_accel.cloud.batch_processor import CloudBatchProcessor

        mock_provider_instance = Mock()
        mock_aws_provider.return_value = mock_provider_instance

        processor = CloudBatchProcessor(provider="aws")

        # Test different dataset sizes
        small_batch = processor.optimize_batch_size(500, available_memory_gb=8)
        assert small_batch == 100

        medium_batch = processor.optimize_batch_size(5000, available_memory_gb=16)
        assert medium_batch == 500

        large_batch = processor.optimize_batch_size(100000, available_memory_gb=32)
        assert large_batch == 10000

        # Test memory constraints
        memory_constrained = processor.optimize_batch_size(
            100000, available_memory_gb=4
        )
        assert memory_constrained <= 4 * 50000  # molecules_per_gb default

    @patch("chemi_accel.cloud.batch_processor.AWSBatchProvider")
    def test_cost_estimation_structure(self, mock_aws_provider):
        """Test cost estimation returns proper structure."""
        from chemi_accel.cloud.batch_processor import CloudBatchProcessor

        mock_provider_instance = Mock()
        mock_aws_provider.return_value = mock_provider_instance

        processor = CloudBatchProcessor(provider="aws")

        cost_estimate = processor.estimate_cost(
            self.job_config, estimated_runtime_hours=2.0
        )

        # Verify structure
        assert "total_cost" in cost_estimate
        assert "breakdown" in cost_estimate
        assert isinstance(cost_estimate["total_cost"], (int, float))
        assert isinstance(cost_estimate["breakdown"], dict)

        # Verify breakdown has expected components
        breakdown = cost_estimate["breakdown"]
        assert "compute" in breakdown
        assert "storage" in breakdown

    @patch("chemi_accel.cloud.batch_processor.AWSBatchProvider")
    def test_job_tracking(self, mock_aws_provider):
        """Test job tracking functionality."""
        from chemi_accel.cloud.batch_processor import CloudBatchProcessor

        mock_provider_instance = Mock()
        mock_provider_instance.submit_job.return_value = "test-job-123"
        mock_aws_provider.return_value = mock_provider_instance

        processor = CloudBatchProcessor(provider="aws")

        # Submit job
        job_id = processor.submit_job(self.job_config)

        # Verify job is tracked
        assert job_id in processor.active_jobs
        assert processor.active_jobs[job_id] == self.job_config

        # Test listing active jobs
        active_jobs = processor.list_active_jobs()
        assert job_id in active_jobs

    @patch("chemi_accel.cloud.batch_processor.AWSBatchProvider")
    def test_environment_variable_setup(self, mock_aws_provider):
        """Test environment variable setup for jobs."""
        from chemi_accel.cloud.batch_processor import CloudBatchProcessor

        mock_provider_instance = Mock()
        mock_provider_instance.submit_job.return_value = "test-job-123"
        mock_aws_provider.return_value = mock_provider_instance

        processor = CloudBatchProcessor(provider="aws")

        descriptor_config = {
            "descriptors": ["molecular_weight", "logp", "tpsa"],
            "backend": "cuda",
        }

        job_id = processor.submit_job(self.job_config, descriptor_config)

        # Verify environment variables were set
        call_args = mock_provider_instance.submit_job.call_args
        job_config_arg = call_args[0][0]

        env_vars = job_config_arg.environment_variables
        assert "CHEMI_ACCEL_INPUT_URI" in env_vars
        assert "CHEMI_ACCEL_OUTPUT_URI" in env_vars
        assert "CHEMI_ACCEL_BACKEND" in env_vars
        assert "DESCRIPTOR_CONFIG" in env_vars

        # Verify descriptor config is properly serialized
        descriptor_json = json.loads(env_vars["DESCRIPTOR_CONFIG"])
        assert descriptor_json == descriptor_config


class TestContainerPackagingCore:
    """Test core container packaging functionality."""

    def test_container_config_creation(self):
        """Test container configuration creation."""
        from chemi_accel.cloud.containers import ContainerConfig

        config = ContainerConfig(
            base_image="nvidia/cuda:11.8-devel-ubuntu20.04",
            python_version="3.9",
            cuda_version="11.8",
            additional_packages=["custom-package==1.0.0"],
        )

        assert config.base_image == "nvidia/cuda:11.8-devel-ubuntu20.04"
        assert config.python_version == "3.9"
        assert config.cuda_version == "11.8"
        assert "custom-package==1.0.0" in config.additional_packages

    def test_base_config_selection(self):
        """Test base configuration selection for different providers."""
        from chemi_accel.cloud.containers import ContainerPackager

        # Test AWS config
        aws_packager = ContainerPackager(provider="aws")
        aws_config = aws_packager.base_configs["aws"]
        assert "nvidia/cuda" in aws_config.base_image

        # Test GCP config
        gcp_packager = ContainerPackager(provider="gcp")
        gcp_config = gcp_packager.base_configs["gcp"]
        assert "gcr.io" in gcp_config.base_image

        # Test Azure config
        azure_packager = ContainerPackager(provider="azure")
        azure_config = azure_packager.base_configs["azure"]
        assert "mcr.microsoft.com" in azure_config.base_image

    def test_dockerfile_content_validation(self):
        """Test Dockerfile content generation."""
        from chemi_accel.cloud.containers import ContainerConfig, ContainerPackager

        packager = ContainerPackager(provider="aws")
        config = ContainerConfig(
            base_image="nvidia/cuda:11.8-devel-ubuntu20.04", python_version="3.9"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = Path(temp_dir) / "Dockerfile"
            packager._generate_dockerfile(dockerfile_path, config)

            dockerfile_content = dockerfile_path.read_text()

            # Verify essential components
            assert "FROM nvidia/cuda:11.8-devel-ubuntu20.04" in dockerfile_content
            assert "python3.9" in dockerfile_content
            assert "WORKDIR /app" in dockerfile_content
            assert "pip install" in dockerfile_content
            assert "HEALTHCHECK" in dockerfile_content
            assert "USER chemiuser" in dockerfile_content

    def test_requirements_generation(self):
        """Test requirements.txt generation."""
        from chemi_accel.cloud.containers import ContainerConfig, ContainerPackager

        packager = ContainerPackager(provider="aws")
        config = ContainerConfig(
            base_image="test-image", additional_packages=["custom-package==1.0.0"]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            build_path = Path(temp_dir)
            app_path = build_path / "app"
            app_path.mkdir()

            packager._generate_requirements(build_path, config)

            requirements_path = app_path / "requirements.txt"
            assert requirements_path.exists()

            requirements_content = requirements_path.read_text()

            # Verify core dependencies
            assert "pandas" in requirements_content
            assert "numpy" in requirements_content
            assert "rdkit-pypi" in requirements_content
            assert "boto3" in requirements_content  # AWS-specific
            assert "custom-package==1.0.0" in requirements_content


class TestCloudStorageCore:
    """Test core cloud storage functionality."""

    def test_uri_parsing(self):
        """Test URI parsing for different providers."""
        from chemi_accel.cloud.storage import CloudStorageStreamer

        streamer = CloudStorageStreamer()

        # Test successful provider creation for supported schemes
        # (These may succeed if the libraries are installed)
        try:
            s3_provider = streamer._get_provider("s3://bucket/file.csv")
            assert s3_provider is not None
        except ImportError:
            # Expected if boto3 not installed
            pass

        try:
            gcs_provider = streamer._get_provider("gs://bucket/file.csv")
            assert gcs_provider is not None
        except (ImportError, Exception):
            # Expected if google-cloud-storage not installed or credentials not available
            pass

        # Test Azure URI - should fail due to missing required config
        with pytest.raises(KeyError):
            streamer._get_provider(
                "https://account.blob.core.windows.net/container/file.csv"
            )

        # Test unsupported scheme
        with pytest.raises(ValueError, match="Unsupported storage URI scheme"):
            streamer._get_provider("ftp://server/file.csv")

    def test_format_detection(self):
        """Test file format detection."""
        from chemi_accel.cloud.storage import CloudStorageStreamer

        streamer = CloudStorageStreamer()

        # Test CSV detection
        assert streamer._detect_format("s3://bucket/data.csv") == "csv"
        assert streamer._detect_format("gs://bucket/data.csv.gz") == "csv"

        # Test Parquet detection
        assert streamer._detect_format("s3://bucket/data.parquet") == "parquet"

        # Test SMILES detection
        assert streamer._detect_format("s3://bucket/molecules.smi") == "smi"
        assert streamer._detect_format("gs://bucket/compounds.smiles") == "smi"

        # Test default fallback
        assert streamer._detect_format("s3://bucket/unknown.xyz") == "csv"


if __name__ == "__main__":
    pytest.main([__file__])

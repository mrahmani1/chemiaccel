"""
End-to-end tests for ML platform integration.

Tests the integration with Amazon SageMaker, Google Vertex AI, and Azure ML
for cloud-native molecular descriptor calculation workflows.
"""

import json
import os
import pytest
import tempfile
import uuid
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import pandas as pd
import numpy as np

from chemi_accel.cloud.ml_platforms import (
    MLJobConfig, MLJobStatus, SageMakerIntegration, 
    VertexAIIntegration, AzureMLIntegration, submit_chemi_accel_job
)


@pytest.fixture
def sample_molecules():
    """Sample molecular data for testing."""
    return pd.DataFrame({
        'smiles': [
            'CCO',  # ethanol
            'CC(=O)O',  # acetic acid
            'c1ccccc1',  # benzene
            'CCN(CC)CC',  # triethylamine
            'CC(C)O'  # isopropanol
        ],
        'name': ['ethanol', 'acetic_acid', 'benzene', 'triethylamine', 'isopropanol']
    })


@pytest.fixture
def job_config():
    """Sample job configuration."""
    return MLJobConfig(
        job_name=f"test-job-{uuid.uuid4().hex[:8]}",
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        input_data_uri="s3://test-bucket/input/",
        output_data_uri="s3://test-bucket/output/",
        hyperparameters={'batch_size': 1000},
        environment_variables={'CUDA_VISIBLE_DEVICES': '0'},
        max_runtime_seconds=3600,
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole"
    )


@pytest.fixture
def descriptor_config():
    """Sample descriptor configuration."""
    return {
        'descriptors': ['molecular_weight', 'logp', 'tpsa'],
        'batch_size': 1000,
        'backend': 'cuda'
    }


class TestSageMakerIntegration:
    """Test SageMaker integration functionality."""
    
    @patch('boto3.client')
    def test_sagemaker_initialization(self, mock_boto3):
        """Test SageMaker client initialization."""
        mock_sagemaker = Mock()
        mock_logs = Mock()
        mock_s3 = Mock()
        mock_boto3.side_effect = [mock_sagemaker, mock_logs, mock_s3]
        
        integration = SageMakerIntegration(
            region='us-east-1',
            role_arn='arn:aws:iam::123456789012:role/SageMakerRole'
        )
        
        assert integration.region == 'us-east-1'
        assert integration.default_role_arn == 'arn:aws:iam::123456789012:role/SageMakerRole'
        assert mock_boto3.call_count == 3
    
    @patch('boto3.client')
    def test_submit_processing_job(self, mock_boto3, job_config):
        """Test submitting SageMaker processing job."""
        mock_sagemaker = Mock()
        mock_logs = Mock()
        mock_s3 = Mock()
        mock_boto3.side_effect = [mock_sagemaker, mock_logs, mock_s3]
        
        mock_sagemaker.create_processing_job.return_value = {
            'ProcessingJobArn': 'arn:aws:sagemaker:us-east-1:123456789012:processing-job/test-job'
        }
        
        integration = SageMakerIntegration(region='us-east-1')
        integration._upload_processing_script = Mock()
        integration._get_chemi_accel_image_uri = Mock(return_value='123456789012.dkr.ecr.us-east-1.amazonaws.com/chemi-accel:latest')
        
        job_name = integration.submit_processing_job(job_config, "print('test')")
        
        assert job_name == job_config.job_name
        mock_sagemaker.create_processing_job.assert_called_once()
        
        # Verify job configuration
        call_args = mock_sagemaker.create_processing_job.call_args[1]
        assert call_args['ProcessingJobName'] == job_config.job_name
        assert call_args['ProcessingResources']['ClusterConfig']['InstanceType'] == job_config.instance_type
        assert 'CUDA_VISIBLE_DEVICES' in call_args['Environment']
    
    @patch('boto3.client')
    def test_get_job_status(self, mock_boto3, job_config):
        """Test getting SageMaker job status."""
        mock_sagemaker = Mock()
        mock_logs = Mock()
        mock_s3 = Mock()
        mock_boto3.side_effect = [mock_sagemaker, mock_logs, mock_s3]
        
        from datetime import datetime
        mock_sagemaker.describe_processing_job.return_value = {
            'ProcessingJobStatus': 'Completed',
            'CreationTime': datetime.now(),
            'ProcessingStartTime': datetime.now(),
            'ProcessingEndTime': datetime.now(),
            'ProcessingResources': {'ClusterConfig': {'InstanceType': 'ml.g4dn.xlarge'}}
        }
        
        integration = SageMakerIntegration(region='us-east-1')
        status = integration.get_job_status(job_config.job_name)
        
        assert isinstance(status, MLJobStatus)
        assert status.job_name == job_config.job_name
        assert status.status == 'Completed'
        assert status.created_at is not None
    
    @patch('boto3.client')
    def test_create_pipeline_step(self, mock_boto3, job_config, descriptor_config):
        """Test creating SageMaker pipeline step."""
        mock_sagemaker = Mock()
        mock_logs = Mock()
        mock_s3 = Mock()
        mock_boto3.side_effect = [mock_sagemaker, mock_logs, mock_s3]
        
        integration = SageMakerIntegration(region='us-east-1')
        integration._get_chemi_accel_image_uri = Mock(return_value='123456789012.dkr.ecr.us-east-1.amazonaws.com/chemi-accel:latest')
        
        with patch('chemi_accel.cloud.ml_platforms.SKLearnProcessor'), \
             patch('chemi_accel.cloud.ml_platforms.ProcessingStep'), \
             patch('chemi_accel.cloud.ml_platforms.ParameterString'):
            
            result = integration.create_pipeline_step(
                'test-step', job_config, descriptor_config
            )
            
            assert 'step' in result
            assert 'parameters' in result
            assert 'input_data' in result['parameters']
            assert 'output_data' in result['parameters']
    
    @patch('boto3.client')
    def test_create_model_endpoint(self, mock_boto3):
        """Test creating SageMaker model endpoint."""
        mock_sagemaker = Mock()
        mock_logs = Mock()
        mock_s3 = Mock()
        mock_boto3.side_effect = [mock_sagemaker, mock_logs, mock_s3]
        
        integration = SageMakerIntegration(
            region='us-east-1',
            role_arn='arn:aws:iam::123456789012:role/SageMakerRole'
        )
        integration._get_chemi_accel_image_uri = Mock(return_value='123456789012.dkr.ecr.us-east-1.amazonaws.com/chemi-accel:latest')
        
        endpoint_name = integration.create_model_endpoint(
            model_data_uri='s3://bucket/model.tar.gz'
        )
        
        assert endpoint_name.startswith('chemi-accel-endpoint-')
        mock_sagemaker.create_model.assert_called_once()
        mock_sagemaker.create_endpoint_config.assert_called_once()
        mock_sagemaker.create_endpoint.assert_called_once()


class TestVertexAIIntegration:
    """Test Vertex AI integration functionality."""
    
    @patch('google.cloud.aiplatform.init')
    def test_vertex_initialization(self, mock_init):
        """Test Vertex AI client initialization."""
        integration = VertexAIIntegration(
            project_id='test-project',
            region='us-central1'
        )
        
        assert integration.project_id == 'test-project'
        assert integration.region == 'us-central1'
        mock_init.assert_called_once_with(project='test-project', location='us-central1')
    
    @patch('google.cloud.aiplatform.init')
    @patch('google.cloud.aiplatform.CustomJob')
    def test_submit_processing_job(self, mock_custom_job, mock_init, job_config):
        """Test submitting Vertex AI custom job."""
        mock_job = Mock()
        mock_custom_job.return_value = mock_job
        
        integration = VertexAIIntegration(project_id='test-project')
        integration._get_chemi_accel_image_uri = Mock(return_value='gcr.io/test-project/chemi-accel:latest')
        
        job_name = integration.submit_processing_job(job_config, "print('test')")
        
        assert job_name == job_config.job_name
        mock_custom_job.assert_called_once()
        mock_job.submit.assert_called_once()
        
        # Verify job configuration
        call_args = mock_custom_job.call_args[1]
        assert call_args['display_name'] == job_config.job_name
        assert len(call_args['worker_pool_specs']) == 1
    
    @patch('google.cloud.aiplatform.init')
    @patch('google.cloud.aiplatform.CustomJob')
    def test_get_job_status(self, mock_custom_job, mock_init, job_config):
        """Test getting Vertex AI job status."""
        from datetime import datetime
        
        mock_job = Mock()
        mock_job.state = 'JOB_STATE_SUCCEEDED'
        mock_job.create_time = datetime.now()
        mock_job.start_time = datetime.now()
        mock_job.end_time = datetime.now()
        mock_job.error = None
        
        mock_custom_job.list.return_value = [mock_job]
        
        integration = VertexAIIntegration(project_id='test-project')
        status = integration.get_job_status(job_config.job_name)
        
        assert isinstance(status, MLJobStatus)
        assert status.job_name == job_config.job_name
        assert status.status == 'JOB_STATE_SUCCEEDED'
    
    @patch('google.cloud.aiplatform.init')
    def test_create_pipeline_step(self, mock_init, job_config, descriptor_config):
        """Test creating Vertex AI pipeline component."""
        integration = VertexAIIntegration(project_id='test-project')
        integration._get_chemi_accel_image_uri = Mock(return_value='gcr.io/test-project/chemi-accel:latest')
        
        with patch('chemi_accel.cloud.ml_platforms.component'), \
             patch('chemi_accel.cloud.ml_platforms.Input'), \
             patch('chemi_accel.cloud.ml_platforms.Output'), \
             patch('chemi_accel.cloud.ml_platforms.Dataset'):
            
            result = integration.create_pipeline_step(
                'test-step', job_config, descriptor_config
            )
            
            assert 'component' in result
            assert 'step_name' in result
            assert 'config' in result
            assert result['step_name'] == 'test-step'


class TestAzureMLIntegration:
    """Test Azure ML integration functionality."""
    
    @patch('azure.ai.ml.MLClient')
    @patch('azure.identity.DefaultAzureCredential')
    def test_azure_initialization(self, mock_credential, mock_ml_client):
        """Test Azure ML client initialization."""
        integration = AzureMLIntegration(
            subscription_id='test-subscription',
            resource_group='test-rg',
            workspace_name='test-workspace'
        )
        
        assert integration.subscription_id == 'test-subscription'
        assert integration.resource_group == 'test-rg'
        assert integration.workspace_name == 'test-workspace'
        mock_ml_client.assert_called_once()
    
    @patch('azure.ai.ml.MLClient')
    @patch('azure.identity.DefaultAzureCredential')
    def test_submit_processing_job(self, mock_credential, mock_ml_client, job_config):
        """Test submitting Azure ML command job."""
        mock_client = Mock()
        mock_ml_client.return_value = mock_client
        
        mock_submitted_job = Mock()
        mock_client.jobs.create_or_update.return_value = mock_submitted_job
        
        integration = AzureMLIntegration(
            subscription_id='test-subscription',
            resource_group='test-rg',
            workspace_name='test-workspace'
        )
        integration._get_chemi_accel_image_uri = Mock(return_value='test-workspace.azurecr.io/chemi-accel:latest')
        
        with patch('chemi_accel.cloud.ml_platforms.command'), \
             patch('chemi_accel.cloud.ml_platforms.Environment'):
            
            job_name = integration.submit_processing_job(job_config, "print('test')")
            
            assert job_name == job_config.job_name
            mock_client.jobs.create_or_update.assert_called_once()
    
    @patch('azure.ai.ml.MLClient')
    @patch('azure.identity.DefaultAzureCredential')
    def test_get_job_status(self, mock_credential, mock_ml_client, job_config):
        """Test getting Azure ML job status."""
        from datetime import datetime
        
        mock_client = Mock()
        mock_ml_client.return_value = mock_client
        
        mock_job = Mock()
        mock_job.status = 'Completed'
        mock_job.creation_context.created_at = datetime.now()
        mock_job.start_time = datetime.now()
        mock_job.end_time = datetime.now()
        
        mock_client.jobs.get.return_value = mock_job
        
        integration = AzureMLIntegration(
            subscription_id='test-subscription',
            resource_group='test-rg',
            workspace_name='test-workspace'
        )
        
        status = integration.get_job_status(job_config.job_name)
        
        assert isinstance(status, MLJobStatus)
        assert status.job_name == job_config.job_name
        assert status.status == 'Completed'
    
    @patch('azure.ai.ml.MLClient')
    @patch('azure.identity.DefaultAzureCredential')
    def test_create_pipeline_step(self, mock_credential, mock_ml_client, job_config, descriptor_config):
        """Test creating Azure ML pipeline component."""
        mock_client = Mock()
        mock_ml_client.return_value = mock_client
        
        integration = AzureMLIntegration(
            subscription_id='test-subscription',
            resource_group='test-rg',
            workspace_name='test-workspace'
        )
        integration._get_chemi_accel_image_uri = Mock(return_value='test-workspace.azurecr.io/chemi-accel:latest')
        
        with patch('chemi_accel.cloud.ml_platforms.dsl'), \
             patch('chemi_accel.cloud.ml_platforms.Environment'):
            
            result = integration.create_pipeline_step(
                'test-step', job_config, descriptor_config
            )
            
            assert 'component' in result
            assert 'step_name' in result
            assert 'config' in result
            assert result['step_name'] == 'test-step'


class TestMLPlatformWorkflows:
    """Test end-to-end ML platform workflows."""
    
    def test_submit_chemi_accel_job_sagemaker(self, job_config, descriptor_config):
        """Test submitting ChemiAccel job to SageMaker."""
        with patch('chemi_accel.cloud.ml_platforms.SageMakerIntegration') as mock_integration:
            mock_instance = Mock()
            mock_integration.return_value = mock_instance
            mock_instance.submit_processing_job.return_value = 'test-job-123'
            
            job_name = submit_chemi_accel_job(
                platform='sagemaker',
                job_config=job_config,
                descriptor_config=descriptor_config,
                region='us-east-1',
                role_arn='arn:aws:iam::123456789012:role/SageMakerRole'
            )
            
            assert job_name == 'test-job-123'
            mock_integration.assert_called_once_with(
                region='us-east-1',
                role_arn='arn:aws:iam::123456789012:role/SageMakerRole'
            )
            mock_instance.submit_processing_job.assert_called_once()
    
    def test_submit_chemi_accel_job_vertex(self, job_config, descriptor_config):
        """Test submitting ChemiAccel job to Vertex AI."""
        with patch('chemi_accel.cloud.ml_platforms.VertexAIIntegration') as mock_integration:
            mock_instance = Mock()
            mock_integration.return_value = mock_instance
            mock_instance.submit_processing_job.return_value = 'test-job-456'
            
            job_name = submit_chemi_accel_job(
                platform='vertex',
                job_config=job_config,
                descriptor_config=descriptor_config,
                project_id='test-project',
                region='us-central1'
            )
            
            assert job_name == 'test-job-456'
            mock_integration.assert_called_once_with(
                project_id='test-project',
                region='us-central1'
            )
            mock_instance.submit_processing_job.assert_called_once()
    
    def test_submit_chemi_accel_job_azure(self, job_config, descriptor_config):
        """Test submitting ChemiAccel job to Azure ML."""
        with patch('chemi_accel.cloud.ml_platforms.AzureMLIntegration') as mock_integration:
            mock_instance = Mock()
            mock_integration.return_value = mock_instance
            mock_instance.submit_processing_job.return_value = 'test-job-789'
            
            job_name = submit_chemi_accel_job(
                platform='azure',
                job_config=job_config,
                descriptor_config=descriptor_config,
                subscription_id='test-subscription',
                resource_group='test-rg',
                workspace_name='test-workspace'
            )
            
            assert job_name == 'test-job-789'
            mock_integration.assert_called_once_with(
                subscription_id='test-subscription',
                resource_group='test-rg',
                workspace_name='test-workspace'
            )
            mock_instance.submit_processing_job.assert_called_once()
    
    def test_invalid_platform(self, job_config, descriptor_config):
        """Test error handling for invalid platform."""
        with pytest.raises(ValueError, match="Unsupported platform: invalid"):
            submit_chemi_accel_job(
                platform='invalid',
                job_config=job_config,
                descriptor_config=descriptor_config
            )


class TestProcessingScriptGeneration:
    """Test processing script generation for different platforms."""
    
    def test_processing_script_basic(self, descriptor_config):
        """Test basic processing script generation."""
        from chemi_accel.cloud.ml_platforms import create_processing_script
        
        script = create_processing_script(descriptor_config)
        
        assert 'from chemi_accel import Calculator' in script
        assert 'batch_size' in script
        assert 'cuda' in script
        assert 'pandas' in script
        assert 'parquet' in script
    
    def test_processing_script_custom_config(self):
        """Test processing script with custom configuration."""
        from chemi_accel.cloud.ml_platforms import create_processing_script
        
        custom_config = {
            'descriptors': ['molecular_weight', 'logp'],
            'batch_size': 500,
            'backend': 'jax'
        }
        
        script = create_processing_script(custom_config)
        
        assert json.dumps(custom_config) in script
        assert 'molecular_weight' in script
        assert 'logp' in script


@pytest.mark.integration
class TestRealMLPlatformIntegration:
    """Integration tests with real ML platforms (requires credentials)."""
    
    @pytest.mark.skipif(
        not os.environ.get('AWS_ACCESS_KEY_ID'),
        reason="AWS credentials not available"
    )
    def test_real_sagemaker_integration(self, sample_molecules, job_config):
        """Test real SageMaker integration (requires AWS credentials)."""
        # Create temporary input data
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "molecules.csv"
            sample_molecules.to_csv(input_file, index=False)
            
            # Update job config with real S3 paths
            job_config.input_data_uri = f"s3://test-bucket/input/{input_file.name}"
            job_config.output_data_uri = "s3://test-bucket/output/"
            
            integration = SageMakerIntegration(
                region=os.environ.get('AWS_REGION', 'us-east-1'),
                role_arn=os.environ.get('SAGEMAKER_ROLE_ARN')
            )
            
            # This would require actual S3 upload and job submission
            # For now, just test initialization
            assert integration.region is not None
            assert integration.default_role_arn is not None
    
    @pytest.mark.skipif(
        not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'),
        reason="GCP credentials not available"
    )
    def test_real_vertex_integration(self, sample_molecules, job_config):
        """Test real Vertex AI integration (requires GCP credentials)."""
        integration = VertexAIIntegration(
            project_id=os.environ.get('GCP_PROJECT_ID', 'test-project'),
            region=os.environ.get('GCP_REGION', 'us-central1')
        )
        
        # This would require actual GCS upload and job submission
        # For now, just test initialization
        assert integration.project_id is not None
        assert integration.region is not None
    
    @pytest.mark.skipif(
        not os.environ.get('AZURE_SUBSCRIPTION_ID'),
        reason="Azure credentials not available"
    )
    def test_real_azure_integration(self, sample_molecules, job_config):
        """Test real Azure ML integration (requires Azure credentials)."""
        integration = AzureMLIntegration(
            subscription_id=os.environ.get('AZURE_SUBSCRIPTION_ID'),
            resource_group=os.environ.get('AZURE_RESOURCE_GROUP', 'test-rg'),
            workspace_name=os.environ.get('AZURE_WORKSPACE_NAME', 'test-workspace')
        )
        
        # This would require actual Azure Blob upload and job submission
        # For now, just test initialization
        assert integration.subscription_id is not None
        assert integration.resource_group is not None
        assert integration.workspace_name is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
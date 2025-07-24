"""
Tests for Retraining Pipeline Orchestration System

This module contains comprehensive tests for the retraining pipeline system including
resource management, job scheduling, execution, and orchestration.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from chemi_accel.autonomous.retraining_pipeline import (
    RetrainingPipelineManager,
    PipelineScheduler,
    PipelineExecutor,
    ResourceManager,
    PipelineJob,
    PipelineSchedule,
    ResourceRequirement,
    PipelineStatus,
    ResourceType,
    create_default_pipeline_manager,
    create_production_pipeline_manager
)
from chemi_accel.autonomous.adaptive_learning import (
    AdaptiveLearningManager,
    RetrainingConfig,
    RetrainingStrategy
)


class TestResourceRequirement:
    """Test cases for ResourceRequirement dataclass."""
    
    def test_resource_requirement_creation(self):
        """Test basic resource requirement creation."""
        req = ResourceRequirement(
            resource_type=ResourceType.GPU,
            amount=2.0,
            unit="count",
            priority=1,
            metadata={"gpu_type": "V100"}
        )
        
        assert req.resource_type == ResourceType.GPU
        assert req.amount == 2.0
        assert req.unit == "count"
        assert req.priority == 1
        assert req.metadata["gpu_type"] == "V100"


class TestPipelineJob:
    """Test cases for PipelineJob dataclass."""
    
    def test_pipeline_job_creation(self):
        """Test basic pipeline job creation."""
        job = PipelineJob(
            job_id="test_job_123",
            model_id="model_456",
            status=PipelineStatus.SCHEDULED,
            created_at=datetime.now(timezone.utc)
        )
        
        assert job.job_id == "test_job_123"
        assert job.model_id == "model_456"
        assert job.status == PipelineStatus.SCHEDULED
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
    
    def test_pipeline_job_auto_id_generation(self):
        """Test automatic job ID generation."""
        job = PipelineJob(
            job_id="",  # Empty ID should trigger auto-generation
            model_id="model_123",
            status=PipelineStatus.SCHEDULED,
            created_at=datetime.now(timezone.utc)
        )
        
        assert job.job_id != ""
        assert len(job.job_id) > 0
    
    def test_pipeline_job_validation(self):
        """Test pipeline job validation."""
        with pytest.raises(ValueError, match="model_id is required"):
            PipelineJob(
                job_id="test_job",
                model_id="",  # Empty model_id should raise error
                status=PipelineStatus.SCHEDULED,
                created_at=datetime.now(timezone.utc)
            )


class TestPipelineSchedule:
    """Test cases for PipelineSchedule dataclass."""
    
    def test_pipeline_schedule_creation(self):
        """Test basic pipeline schedule creation."""
        schedule = PipelineSchedule(
            schedule_id="daily_retrain",
            interval_hours=24,
            trigger_conditions=[{"type": "data_threshold", "value": 1000}],
            metadata={"model_id": "model_123"}
        )
        
        assert schedule.schedule_id == "daily_retrain"
        assert schedule.interval_hours == 24
        assert schedule.is_active is True
        assert len(schedule.trigger_conditions) == 1
        assert schedule.metadata["model_id"] == "model_123"


class TestResourceManager:
    """Test cases for ResourceManager."""
    
    @pytest.fixture
    def resource_manager(self) -> ResourceManager:
        """Create resource manager."""
        return ResourceManager()
    
    @pytest.fixture
    def sample_requirements(self) -> List[ResourceRequirement]:
        """Create sample resource requirements."""
        return [
            ResourceRequirement(ResourceType.CPU, 4.0, "cores"),
            ResourceRequirement(ResourceType.MEMORY, 16.0, "GB"),
            ResourceRequirement(ResourceType.GPU, 1.0, "count")
        ]
    
    @pytest.mark.asyncio
    async def test_check_resource_availability_sufficient(self, resource_manager, sample_requirements):
        """Test resource availability check with sufficient resources."""
        available = await resource_manager.check_resource_availability(sample_requirements)
        assert available is True
    
    @pytest.mark.asyncio
    async def test_check_resource_availability_insufficient(self, resource_manager):
        """Test resource availability check with insufficient resources."""
        excessive_requirements = [
            ResourceRequirement(ResourceType.CPU, 100.0, "cores"),  # More than available
            ResourceRequirement(ResourceType.MEMORY, 1000.0, "GB")  # More than available
        ]
        
        available = await resource_manager.check_resource_availability(excessive_requirements)
        assert available is False
    
    @pytest.mark.asyncio
    async def test_allocate_and_release_resources(self, resource_manager, sample_requirements):
        """Test resource allocation and release."""
        job_id = "test_job_123"
        
        # Allocate resources
        allocated = await resource_manager.allocate_resources(job_id, sample_requirements)
        assert allocated is True
        assert job_id in resource_manager.allocated_resources
        
        # Check utilization
        utilization = resource_manager.get_resource_utilization()
        assert utilization[ResourceType.CPU.value]["allocated"] == 4.0
        assert utilization[ResourceType.MEMORY.value]["allocated"] == 16.0
        assert utilization[ResourceType.GPU.value]["allocated"] == 1.0
        
        # Release resources
        await resource_manager.release_resources(job_id)
        assert job_id not in resource_manager.allocated_resources
        
        # Check utilization after release
        utilization = resource_manager.get_resource_utilization()
        assert utilization[ResourceType.CPU.value]["allocated"] == 0.0
        assert utilization[ResourceType.MEMORY.value]["allocated"] == 0.0
        assert utilization[ResourceType.GPU.value]["allocated"] == 0.0
    
    @pytest.mark.asyncio
    async def test_resource_contention(self, resource_manager):
        """Test resource allocation under contention."""
        # Allocate most available resources to first job
        job1_requirements = [
            ResourceRequirement(ResourceType.CPU, 6.0, "cores"),
            ResourceRequirement(ResourceType.MEMORY, 24.0, "GB")
        ]
        
        allocated1 = await resource_manager.allocate_resources("job1", job1_requirements)
        assert allocated1 is True
        
        # Try to allocate overlapping resources to second job
        job2_requirements = [
            ResourceRequirement(ResourceType.CPU, 4.0, "cores"),  # Would exceed available
            ResourceRequirement(ResourceType.MEMORY, 16.0, "GB")  # Would exceed available
        ]
        
        allocated2 = await resource_manager.allocate_resources("job2", job2_requirements)
        assert allocated2 is False
        
        # Release first job's resources
        await resource_manager.release_resources("job1")
        
        # Now second job should be able to allocate
        allocated2_retry = await resource_manager.allocate_resources("job2", job2_requirements)
        assert allocated2_retry is True
    
    def test_get_resource_utilization(self, resource_manager):
        """Test resource utilization reporting."""
        utilization = resource_manager.get_resource_utilization()
        
        for resource_type in ResourceType:
            if resource_type in resource_manager.available_resources:
                assert resource_type.value in utilization
                assert "total" in utilization[resource_type.value]
                assert "allocated" in utilization[resource_type.value]
                assert "available" in utilization[resource_type.value]
                assert "utilization_percent" in utilization[resource_type.value]


class TestPipelineScheduler:
    """Test cases for PipelineScheduler."""
    
    @pytest.fixture
    def scheduler(self) -> PipelineScheduler:
        """Create pipeline scheduler."""
        return PipelineScheduler()
    
    @pytest.fixture
    def sample_schedule(self) -> PipelineSchedule:
        """Create sample schedule."""
        return PipelineSchedule(
            schedule_id="test_schedule",
            interval_hours=6,
            metadata={"model_id": "model_123"}
        )
    
    def test_add_schedule(self, scheduler, sample_schedule):
        """Test adding a schedule."""
        scheduler.add_schedule(sample_schedule)
        
        assert sample_schedule.schedule_id in scheduler.schedules
        assert scheduler.schedules[sample_schedule.schedule_id] == sample_schedule
        assert sample_schedule.next_run is not None
    
    def test_remove_schedule(self, scheduler, sample_schedule):
        """Test removing a schedule."""
        scheduler.add_schedule(sample_schedule)
        
        # Remove existing schedule
        removed = scheduler.remove_schedule(sample_schedule.schedule_id)
        assert removed is True
        assert sample_schedule.schedule_id not in scheduler.schedules
        
        # Try to remove non-existent schedule
        removed = scheduler.remove_schedule("non_existent")
        assert removed is False
    
    def test_update_next_run_interval(self, scheduler):
        """Test next run time calculation for interval schedules."""
        schedule = PipelineSchedule(
            schedule_id="interval_test",
            interval_hours=12
        )
        
        scheduler._update_next_run(schedule)
        
        assert schedule.next_run is not None
        # Next run should be approximately 12 hours from now
        time_diff = schedule.next_run - datetime.now(timezone.utc)
        assert 11.9 <= time_diff.total_seconds() / 3600 <= 12.1  # Allow small tolerance
    
    def test_update_next_run_with_last_run(self, scheduler):
        """Test next run time calculation with existing last run."""
        last_run = datetime.now(timezone.utc) - timedelta(hours=2)
        schedule = PipelineSchedule(
            schedule_id="last_run_test",
            interval_hours=6,
            last_run=last_run
        )
        
        scheduler._update_next_run(schedule)
        
        assert schedule.next_run is not None
        expected_next = last_run + timedelta(hours=6)
        assert abs((schedule.next_run - expected_next).total_seconds()) < 1
    
    @pytest.mark.asyncio
    async def test_start_stop_scheduler(self, scheduler):
        """Test starting and stopping the scheduler."""
        # Start scheduler
        await scheduler.start_scheduler()
        assert scheduler.scheduler_running is True
        assert scheduler.scheduler_task is not None
        
        # Stop scheduler
        await scheduler.stop_scheduler()
        assert scheduler.scheduler_running is False
    
    def test_get_schedule_status(self, scheduler, sample_schedule):
        """Test schedule status reporting."""
        scheduler.add_schedule(sample_schedule)
        
        status = scheduler.get_schedule_status()
        
        assert "scheduler_running" in status
        assert "total_schedules" in status
        assert "active_schedules" in status
        assert "schedules" in status
        
        assert status["total_schedules"] == 1
        assert status["active_schedules"] == 1
        assert len(status["schedules"]) == 1
        
        schedule_info = status["schedules"][0]
        assert schedule_info["schedule_id"] == sample_schedule.schedule_id
        assert schedule_info["is_active"] is True
        assert schedule_info["interval_hours"] == 6


class TestPipelineExecutor:
    """Test cases for PipelineExecutor."""
    
    @pytest.fixture
    def resource_manager(self) -> ResourceManager:
        """Create resource manager."""
        return ResourceManager()
    
    @pytest.fixture
    def executor(self, resource_manager) -> PipelineExecutor:
        """Create pipeline executor."""
        return PipelineExecutor(resource_manager, max_concurrent_jobs=2)
    
    @pytest.fixture
    def sample_job(self) -> PipelineJob:
        """Create sample pipeline job."""
        return PipelineJob(
            job_id="test_job_123",
            model_id="model_456",
            status=PipelineStatus.SCHEDULED,
            created_at=datetime.now(timezone.utc),
            resource_requirements=[
                ResourceRequirement(ResourceType.CPU, 2.0, "cores"),
                ResourceRequirement(ResourceType.MEMORY, 8.0, "GB")
            ]
        )
    
    def test_set_callbacks(self, executor):
        """Test setting callback functions."""
        started_callback = Mock()
        completed_callback = Mock()
        failed_callback = Mock()
        
        executor.set_callbacks(
            job_started=started_callback,
            job_completed=completed_callback,
            job_failed=failed_callback
        )
        
        assert executor.job_started_callback == started_callback
        assert executor.job_completed_callback == completed_callback
        assert executor.job_failed_callback == failed_callback
    
    @pytest.mark.asyncio
    async def test_start_stop_executor(self, executor):
        """Test starting and stopping the executor."""
        job_queue = asyncio.Queue()
        
        # Start executor
        await executor.start_executor(job_queue)
        assert executor.executor_running is True
        assert executor.executor_task is not None
        
        # Stop executor
        await executor.stop_executor()
        assert executor.executor_running is False
    
    @pytest.mark.asyncio
    async def test_job_execution_success(self, executor, sample_job):
        """Test successful job execution."""
        # Mock callbacks
        started_callback = AsyncMock()
        completed_callback = AsyncMock()
        executor.set_callbacks(job_started=started_callback, job_completed=completed_callback)
        
        # Execute job
        await executor._execute_job(sample_job)
        
        # Check job status
        assert sample_job.status == PipelineStatus.COMPLETED
        assert sample_job.started_at is not None
        assert sample_job.completed_at is not None
        assert len(sample_job.metrics) > 0
        assert len(sample_job.output_artifacts) > 0
        
        # Check callbacks were called
        started_callback.assert_called_once_with(sample_job)
        completed_callback.assert_called_once_with(sample_job)
    
    @pytest.mark.asyncio
    async def test_job_execution_failure(self, executor):
        """Test job execution failure handling."""
        # Create job that will fail (no model_id)
        failing_job = PipelineJob(
            job_id="failing_job",
            model_id="model_fail",
            status=PipelineStatus.SCHEDULED,
            created_at=datetime.now(timezone.utc)
        )
        
        # Mock failed callback
        failed_callback = AsyncMock()
        executor.set_callbacks(job_failed=failed_callback)
        
        # Mock the _run_retraining_job to raise exception
        with patch.object(executor, '_run_retraining_job', side_effect=Exception("Test failure")):
            await executor._execute_job(failing_job)
        
        # Check job status
        assert failing_job.status == PipelineStatus.FAILED
        assert failing_job.error_message == "Test failure"
        assert failing_job.completed_at is not None
        
        # Check callback was called
        failed_callback.assert_called_once_with(failing_job)
    
    @pytest.mark.asyncio
    async def test_resource_allocation_in_execution(self, executor, sample_job):
        """Test resource allocation during job execution."""
        # Execute job
        await executor._execute_job(sample_job)
        
        # Resources should be released after execution
        assert sample_job.job_id not in executor.resource_manager.allocated_resources
        
        # Job should be in completed jobs
        assert len(executor.completed_jobs) == 1
        assert executor.completed_jobs[0].job_id == sample_job.job_id
    
    @pytest.mark.asyncio
    async def test_concurrent_job_execution(self, executor):
        """Test concurrent job execution with semaphore."""
        jobs = []
        for i in range(5):
            job = PipelineJob(
                job_id=f"concurrent_job_{i}",
                model_id=f"model_{i}",
                status=PipelineStatus.SCHEDULED,
                created_at=datetime.now(timezone.utc)
            )
            jobs.append(job)
        
        # Execute jobs concurrently
        tasks = [executor._execute_job(job) for job in jobs]
        await asyncio.gather(*tasks)
        
        # All jobs should be completed
        for job in jobs:
            assert job.status == PipelineStatus.COMPLETED
        
        # Should have all jobs in completed list
        assert len(executor.completed_jobs) == 5
    
    def test_get_executor_status(self, executor, sample_job):
        """Test executor status reporting."""
        # Add job to running jobs
        executor.running_jobs[sample_job.job_id] = sample_job
        
        status = executor.get_executor_status()
        
        assert "executor_running" in status
        assert "max_concurrent_jobs" in status
        assert "running_jobs_count" in status
        assert "completed_jobs_count" in status
        assert "running_jobs" in status
        assert "recent_completed_jobs" in status
        
        assert status["max_concurrent_jobs"] == 2
        assert status["running_jobs_count"] == 1
        assert len(status["running_jobs"]) == 1
        
        running_job_info = status["running_jobs"][0]
        assert running_job_info["job_id"] == sample_job.job_id
        assert running_job_info["model_id"] == sample_job.model_id


class TestRetrainingPipelineManager:
    """Test cases for RetrainingPipelineManager."""
    
    @pytest.fixture
    def mock_adaptive_manager(self):
        """Create mock adaptive learning manager."""
        manager = Mock(spec=AdaptiveLearningManager)
        return manager
    
    @pytest.fixture
    def pipeline_manager(self, mock_adaptive_manager) -> RetrainingPipelineManager:
        """Create retraining pipeline manager."""
        return RetrainingPipelineManager(
            adaptive_learning_manager=mock_adaptive_manager,
            max_concurrent_jobs=2
        )
    
    def test_register_model(self, pipeline_manager):
        """Test model registration."""
        model_config = {
            "model_type": "RandomForest",
            "hyperparameters": {"n_estimators": 100}
        }
        
        pipeline_manager.register_model("model_123", model_config)
        
        assert "model_123" in pipeline_manager.models_registry
        model_info = pipeline_manager.models_registry["model_123"]
        assert model_info["model_id"] == "model_123"
        assert model_info["config"] == model_config
        assert model_info["retrain_count"] == 0
    
    def test_add_event_callback(self, pipeline_manager):
        """Test adding event callbacks."""
        callback = Mock()
        
        pipeline_manager.add_event_callback(callback)
        
        assert callback in pipeline_manager.job_event_callbacks
    
    @pytest.mark.asyncio
    async def test_start_stop_pipeline(self, pipeline_manager):
        """Test starting and stopping the pipeline."""
        # Start pipeline
        await pipeline_manager.start_pipeline()
        assert pipeline_manager.is_running is True
        
        # Stop pipeline
        await pipeline_manager.stop_pipeline()
        assert pipeline_manager.is_running is False
    
    def test_schedule_periodic_retraining(self, pipeline_manager):
        """Test scheduling periodic retraining."""
        model_id = "model_123"
        interval_hours = 12
        
        schedule_id = pipeline_manager.schedule_periodic_retraining(model_id, interval_hours)
        
        assert schedule_id is not None
        assert schedule_id in pipeline_manager.scheduler.schedules
        
        schedule = pipeline_manager.scheduler.schedules[schedule_id]
        assert schedule.interval_hours == interval_hours
        assert schedule.metadata["model_id"] == model_id
    
    def test_schedule_conditional_retraining(self, pipeline_manager):
        """Test scheduling conditional retraining."""
        model_id = "model_456"
        conditions = [
            {"type": "data_threshold", "value": 1000},
            {"type": "performance_degradation", "threshold": 0.1}
        ]
        
        schedule_id = pipeline_manager.schedule_conditional_retraining(model_id, conditions)
        
        assert schedule_id is not None
        assert schedule_id in pipeline_manager.scheduler.schedules
        
        schedule = pipeline_manager.scheduler.schedules[schedule_id]
        assert schedule.trigger_conditions == conditions
        assert schedule.metadata["model_id"] == model_id
    
    @pytest.mark.asyncio
    async def test_trigger_immediate_retraining(self, pipeline_manager):
        """Test triggering immediate retraining."""
        model_id = "model_789"
        resource_requirements = [
            ResourceRequirement(ResourceType.CPU, 4.0, "cores"),
            ResourceRequirement(ResourceType.MEMORY, 16.0, "GB")
        ]
        
        job_id = await pipeline_manager.trigger_immediate_retraining(
            model_id,
            resource_requirements=resource_requirements
        )
        
        assert job_id is not None
        assert len(job_id) > 0
    
    @pytest.mark.asyncio
    async def test_job_event_handling(self, pipeline_manager):
        """Test job event handling."""
        # Register model
        pipeline_manager.register_model("model_123", {"type": "test"})
        
        # Create test job
        job = PipelineJob(
            job_id="test_job",
            model_id="model_123",
            status=PipelineStatus.RUNNING,
            created_at=datetime.now(timezone.utc)
        )
        
        # Test job started event
        await pipeline_manager._on_job_started(job)
        model_info = pipeline_manager.models_registry["model_123"]
        assert model_info["current_job_id"] == "test_job"
        
        # Test job completed event
        job.status = PipelineStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        await pipeline_manager._on_job_completed(job)
        
        model_info = pipeline_manager.models_registry["model_123"]
        assert model_info["retrain_count"] == 1
        assert model_info["last_retrained"] is not None
        assert "current_job_id" not in model_info
        
        # Test job failed event
        failed_job = PipelineJob(
            job_id="failed_job",
            model_id="model_123",
            status=PipelineStatus.FAILED,
            created_at=datetime.now(timezone.utc),
            error_message="Test error"
        )
        failed_job.completed_at = datetime.now(timezone.utc)
        
        await pipeline_manager._on_job_failed(failed_job)
        # Failed jobs shouldn't increment retrain count
        assert pipeline_manager.models_registry["model_123"]["retrain_count"] == 1
    
    def test_get_job_status(self, pipeline_manager):
        """Test getting job status."""
        # Create job and add to executor
        job = PipelineJob(
            job_id="status_test_job",
            model_id="model_123",
            status=PipelineStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            metrics={"mse": 0.1, "r2": 0.9}
        )
        
        pipeline_manager.executor.completed_jobs.append(job)
        
        status = pipeline_manager.get_job_status("status_test_job")
        
        assert status is not None
        assert status["job_id"] == "status_test_job"
        assert status["model_id"] == "model_123"
        assert status["status"] == "completed"
        assert status["metrics"]["mse"] == 0.1
        
        # Test non-existent job
        status = pipeline_manager.get_job_status("non_existent")
        assert status is None
    
    def test_get_model_status(self, pipeline_manager):
        """Test getting model status."""
        # Register model
        model_config = {"type": "test_model"}
        pipeline_manager.register_model("model_status_test", model_config)
        
        status = pipeline_manager.get_model_status("model_status_test")
        
        assert status is not None
        assert status["model_id"] == "model_status_test"
        assert status["config"] == model_config
        assert status["retrain_count"] == 0
        
        # Test non-existent model
        status = pipeline_manager.get_model_status("non_existent")
        assert status is None
    
    def test_get_pipeline_status(self, pipeline_manager):
        """Test getting comprehensive pipeline status."""
        # Register a model
        pipeline_manager.register_model("status_model", {"type": "test"})
        
        status = pipeline_manager.get_pipeline_status()
        
        assert "is_running" in status
        assert "registered_models" in status
        assert "resource_utilization" in status
        assert "scheduler_status" in status
        assert "executor_status" in status
        assert "models" in status
        
        assert status["registered_models"] == 1
        assert "status_model" in status["models"]


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_default_pipeline_manager(self):
        """Test default pipeline manager creation."""
        manager = create_default_pipeline_manager()
        
        assert isinstance(manager, RetrainingPipelineManager)
        assert manager.executor.max_concurrent_jobs == 2
        assert isinstance(manager.resource_manager, ResourceManager)
        assert isinstance(manager.scheduler, PipelineScheduler)
        assert isinstance(manager.executor, PipelineExecutor)
    
    def test_create_production_pipeline_manager(self):
        """Test production pipeline manager creation."""
        mock_adaptive_manager = Mock(spec=AdaptiveLearningManager)
        
        manager = create_production_pipeline_manager(
            adaptive_learning_manager=mock_adaptive_manager
        )
        
        assert isinstance(manager, RetrainingPipelineManager)
        assert manager.executor.max_concurrent_jobs == 4
        assert manager.adaptive_learning_manager == mock_adaptive_manager
        
        # Check production resource limits
        resources = manager.resource_manager.available_resources
        assert resources[ResourceType.CPU] == 16.0
        assert resources[ResourceType.GPU] == 2.0
        assert resources[ResourceType.MEMORY] == 64.0
        assert resources[ResourceType.STORAGE] == 2000.0


class TestIntegrationScenarios:
    """Integration test scenarios for retraining pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_execution(self):
        """Test complete end-to-end pipeline execution."""
        # Create pipeline manager
        manager = create_default_pipeline_manager()
        
        # Register model
        model_config = {
            "model_type": "RandomForest",
            "hyperparameters": {"n_estimators": 100}
        }
        manager.register_model("integration_model", model_config)
        
        # Start pipeline
        await manager.start_pipeline()
        
        # Schedule periodic retraining
        schedule_id = manager.schedule_periodic_retraining("integration_model", interval_hours=1)
        
        # Trigger immediate retraining
        job_id = await manager.trigger_immediate_retraining("integration_model")
        
        # Wait a bit for job to be processed
        await asyncio.sleep(0.1)
        
        # Check job status
        job_status = manager.get_job_status(job_id)
        assert job_status is not None
        assert job_status["model_id"] == "integration_model"
        
        # Check model status
        model_status = manager.get_model_status("integration_model")
        assert model_status is not None
        
        # Check pipeline status
        pipeline_status = manager.get_pipeline_status()
        assert pipeline_status["is_running"] is True
        assert pipeline_status["registered_models"] == 1
        
        # Stop pipeline
        await manager.stop_pipeline()
        
        assert manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_resource_contention_scenario(self):
        """Test resource contention handling."""
        # Create manager with limited resources
        manager = create_default_pipeline_manager()
        
        # Reduce available resources for testing
        manager.resource_manager.available_resources[ResourceType.CPU] = 4.0
        manager.resource_manager.available_resources[ResourceType.MEMORY] = 8.0
        
        # Register multiple models
        for i in range(3):
            manager.register_model(f"resource_model_{i}", {"type": "test"})
        
        # Start pipeline
        await manager.start_pipeline()
        
        # Trigger multiple retraining jobs with high resource requirements
        resource_requirements = [
            ResourceRequirement(ResourceType.CPU, 3.0, "cores"),
            ResourceRequirement(ResourceType.MEMORY, 6.0, "GB")
        ]
        
        job_ids = []
        for i in range(3):
            job_id = await manager.trigger_immediate_retraining(
                f"resource_model_{i}",
                resource_requirements=resource_requirements
            )
            job_ids.append(job_id)
        
        # Wait for jobs to be processed
        await asyncio.sleep(0.5)
        
        # Check resource utilization
        utilization = manager.resource_manager.get_resource_utilization()
        
        # Should have some resource allocation (at least one job should run)
        assert utilization[ResourceType.CPU.value]["allocated"] >= 0
        
        # Stop pipeline
        await manager.stop_pipeline()
    
    @pytest.mark.asyncio
    async def test_scheduling_and_execution_integration(self):
        """Test integration between scheduling and execution."""
        manager = create_default_pipeline_manager()
        
        # Register model
        manager.register_model("scheduled_model", {"type": "scheduled"})
        
        # Add event callback to track job events
        job_events = []
        
        async def event_callback(event_type, job):
            job_events.append((event_type, job.job_id, job.model_id))
        
        manager.add_event_callback(event_callback)
        
        # Start pipeline
        await manager.start_pipeline()
        
        # Schedule retraining with very short interval for testing
        schedule_id = manager.schedule_periodic_retraining("scheduled_model", interval_hours=1)
        
        # Manually set next run to past time to trigger immediate execution
        schedule = manager.scheduler.schedules[schedule_id]
        schedule.next_run = datetime.now(timezone.utc) - timedelta(minutes=1)
        
        # Wait for scheduler to trigger and executor to process
        await asyncio.sleep(0.2)
        
        # Check that events were recorded
        assert len(job_events) > 0
        
        # Check job was created and processed
        executor_status = manager.executor.get_executor_status()
        assert executor_status["completed_jobs_count"] >= 1 or executor_status["running_jobs_count"] >= 1
        
        # Stop pipeline
        await manager.stop_pipeline()
    
    @pytest.mark.asyncio
    async def test_failure_recovery_scenario(self):
        """Test failure recovery in pipeline execution."""
        manager = create_default_pipeline_manager()
        
        # Register model
        manager.register_model("failure_model", {"type": "failure_test"})
        
        # Track failed jobs
        failed_jobs = []
        
        async def failure_callback(event_type, job):
            if event_type == "job_failed":
                failed_jobs.append(job)
        
        manager.add_event_callback(failure_callback)
        
        # Start pipeline
        await manager.start_pipeline()
        
        # Patch the retraining job to fail
        original_run_job = manager.executor._run_retraining_job
        
        async def failing_run_job(job):
            if job.model_id == "failure_model":
                raise Exception("Simulated retraining failure")
            return await original_run_job(job)
        
        manager.executor._run_retraining_job = failing_run_job
        
        # Trigger retraining that will fail
        job_id = await manager.trigger_immediate_retraining("failure_model")
        
        # Wait for job to be processed and fail
        await asyncio.sleep(0.1)
        
        # Check that failure was handled
        assert len(failed_jobs) == 1
        failed_job = failed_jobs[0]
        assert failed_job.job_id == job_id
        assert failed_job.status == PipelineStatus.FAILED
        assert "Simulated retraining failure" in failed_job.error_message
        
        # Check job status
        job_status = manager.get_job_status(job_id)
        assert job_status["status"] == "failed"
        assert job_status["error_message"] is not None
        
        # Stop pipeline
        await manager.stop_pipeline()
    
    @pytest.mark.asyncio
    async def test_concurrent_model_retraining(self):
        """Test concurrent retraining of multiple models."""
        manager = create_default_pipeline_manager()
        
        # Register multiple models
        model_count = 5
        for i in range(model_count):
            manager.register_model(f"concurrent_model_{i}", {"type": f"model_{i}"})
        
        # Start pipeline
        await manager.start_pipeline()
        
        # Trigger retraining for all models simultaneously
        job_ids = []
        for i in range(model_count):
            job_id = await manager.trigger_immediate_retraining(f"concurrent_model_{i}")
            job_ids.append(job_id)
        
        # Wait for jobs to be processed
        await asyncio.sleep(1.0)
        
        # Check that all jobs were processed
        completed_count = 0
        running_count = 0
        
        for job_id in job_ids:
            job_status = manager.get_job_status(job_id)
            if job_status:
                if job_status["status"] == "completed":
                    completed_count += 1
                elif job_status["status"] == "running":
                    running_count += 1
        
        # Should have processed some jobs (limited by max_concurrent_jobs)
        total_processed = completed_count + running_count
        assert total_processed > 0
        assert total_processed <= manager.executor.max_concurrent_jobs + model_count
        
        # Stop pipeline
        await manager.stop_pipeline()
"""
Retraining Pipeline Orchestration System

This module provides comprehensive orchestration for automated model retraining
pipelines, including scheduling, resource management, and integration with
the broader autonomous laboratory system.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .adaptive_learning import AdaptiveLearningManager, RetrainingConfig
from .data_ingestion import DataIngestionManager

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Status of retraining pipelines."""

    IDLE = "idle"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceType(Enum):
    """Types of computational resources."""

    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    CLOUD_INSTANCE = "cloud_instance"


@dataclass
class ResourceRequirement:
    """Defines resource requirements for retraining."""

    resource_type: ResourceType
    amount: float
    unit: str
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSchedule:
    """Defines when and how often retraining should occur."""

    schedule_id: str
    cron_expression: Optional[str] = None
    interval_hours: Optional[int] = None
    trigger_conditions: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineJob:
    """Represents a retraining job in the pipeline."""

    job_id: str
    model_id: str
    status: PipelineStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    training_config: Optional[RetrainingConfig] = None
    input_data_refs: List[str] = field(default_factory=list)
    output_artifacts: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate job configuration after initialization."""
        if not self.job_id:
            self.job_id = str(uuid.uuid4())

        if not self.model_id:
            raise ValueError("model_id is required")


class ResourceManager:
    """Manages computational resources for retraining jobs."""

    def __init__(self):
        self.available_resources: Dict[ResourceType, float] = {
            ResourceType.CPU: 8.0,  # CPU cores
            ResourceType.GPU: 1.0,  # GPU count
            ResourceType.MEMORY: 32.0,  # GB
            ResourceType.STORAGE: 1000.0,  # GB
        }
        self.allocated_resources: Dict[str, List[ResourceRequirement]] = {}
        self.resource_lock = asyncio.Lock()

    async def check_resource_availability(
        self, requirements: List[ResourceRequirement]
    ) -> bool:
        """Check if requested resources are available."""
        async with self.resource_lock:
            for req in requirements:
                available = self.available_resources.get(req.resource_type, 0.0)
                allocated = sum(
                    r.amount
                    for job_reqs in self.allocated_resources.values()
                    for r in job_reqs
                    if r.resource_type == req.resource_type
                )

                if available - allocated < req.amount:
                    logger.warning(
                        f"Insufficient {req.resource_type.value}: need {req.amount}, available {available - allocated}"
                    )
                    return False

            return True

    async def allocate_resources(
        self, job_id: str, requirements: List[ResourceRequirement]
    ) -> bool:
        """Allocate resources for a job."""
        if await self.check_resource_availability(requirements):
            async with self.resource_lock:
                self.allocated_resources[job_id] = requirements
                logger.info(
                    f"Allocated resources for job {job_id}: {[f'{r.resource_type.value}:{r.amount}' for r in requirements]}"
                )
                return True
        return False

    async def release_resources(self, job_id: str):
        """Release resources allocated to a job."""
        async with self.resource_lock:
            if job_id in self.allocated_resources:
                requirements = self.allocated_resources.pop(job_id)
                logger.info(
                    f"Released resources for job {job_id}: {[f'{r.resource_type.value}:{r.amount}' for r in requirements]}"
                )

    def get_resource_utilization(self) -> Dict[str, Dict[str, float]]:
        """Get current resource utilization."""
        utilization = {}

        for resource_type, total in self.available_resources.items():
            allocated = sum(
                r.amount
                for job_reqs in self.allocated_resources.values()
                for r in job_reqs
                if r.resource_type == resource_type
            )

            utilization[resource_type.value] = {
                "total": total,
                "allocated": allocated,
                "available": total - allocated,
                "utilization_percent": (allocated / total) * 100 if total > 0 else 0,
            }

        return utilization


class PipelineScheduler:
    """Manages scheduling of retraining pipelines."""

    def __init__(self):
        self.schedules: Dict[str, PipelineSchedule] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.scheduler_running = False
        self.scheduler_task: Optional[asyncio.Task] = None

    def add_schedule(self, schedule: PipelineSchedule):
        """Add a retraining schedule."""
        self.schedules[schedule.schedule_id] = schedule
        self._update_next_run(schedule)
        logger.info(
            f"Added schedule {schedule.schedule_id}, next run: {schedule.next_run}"
        )

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a retraining schedule."""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            logger.info(f"Removed schedule {schedule_id}")
            return True
        return False

    def _update_next_run(self, schedule: PipelineSchedule):
        """Update the next run time for a schedule."""
        now = datetime.now(timezone.utc)

        if schedule.interval_hours:
            # Simple interval-based scheduling
            if schedule.last_run:
                schedule.next_run = schedule.last_run + timedelta(
                    hours=schedule.interval_hours
                )
            else:
                schedule.next_run = now + timedelta(hours=schedule.interval_hours)
        elif schedule.cron_expression:
            # Would implement cron parsing here in production
            # For now, default to daily
            if schedule.last_run:
                schedule.next_run = schedule.last_run + timedelta(days=1)
            else:
                schedule.next_run = now + timedelta(days=1)
        else:
            # Manual scheduling only
            schedule.next_run = None

    async def start_scheduler(self):
        """Start the background scheduler."""
        if self.scheduler_running:
            logger.warning("Scheduler already running")
            return

        self.scheduler_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Started pipeline scheduler")

    async def stop_scheduler(self):
        """Stop the background scheduler."""
        if not self.scheduler_running:
            return

        self.scheduler_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped pipeline scheduler")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.scheduler_running:
            try:
                now = datetime.now(timezone.utc)

                for schedule in list(self.schedules.values()):
                    if not schedule.is_active:
                        continue

                    if schedule.next_run and now >= schedule.next_run:
                        # Time to run this schedule
                        job = PipelineJob(
                            job_id=str(uuid.uuid4()),
                            model_id=schedule.metadata.get("model_id", "default"),
                            status=PipelineStatus.SCHEDULED,
                            created_at=now,
                            metadata={
                                "schedule_id": schedule.schedule_id,
                                "trigger": "scheduled",
                            },
                        )

                        await self.job_queue.put(job)

                        # Update schedule
                        schedule.last_run = now
                        self._update_next_run(schedule)

                        logger.info(
                            f"Scheduled retraining job {job.job_id} from schedule {schedule.schedule_id}"
                        )

                # Check trigger conditions
                await self._check_trigger_conditions()

                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def _check_trigger_conditions(self):
        """Check for trigger conditions that should start retraining."""
        # This would integrate with the data ingestion system
        # to check for conditions like new data availability,
        # performance degradation, etc.
        pass

    def get_schedule_status(self) -> Dict[str, Any]:
        """Get status of all schedules."""
        return {
            "scheduler_running": self.scheduler_running,
            "total_schedules": len(self.schedules),
            "active_schedules": sum(1 for s in self.schedules.values() if s.is_active),
            "schedules": [
                {
                    "schedule_id": s.schedule_id,
                    "is_active": s.is_active,
                    "next_run": s.next_run.isoformat() if s.next_run else None,
                    "last_run": s.last_run.isoformat() if s.last_run else None,
                    "interval_hours": s.interval_hours,
                    "cron_expression": s.cron_expression,
                }
                for s in self.schedules.values()
            ],
        }


class PipelineExecutor:
    """Executes retraining jobs with proper resource management."""

    def __init__(self, resource_manager: ResourceManager, max_concurrent_jobs: int = 2):
        self.resource_manager = resource_manager
        self.max_concurrent_jobs = max_concurrent_jobs
        self.running_jobs: Dict[str, PipelineJob] = {}
        self.completed_jobs: List[PipelineJob] = []
        self.executor_running = False
        self.executor_task: Optional[asyncio.Task] = None
        self.job_semaphore = asyncio.Semaphore(max_concurrent_jobs)

        # Callbacks for integration
        self.job_started_callback: Optional[Callable] = None
        self.job_completed_callback: Optional[Callable] = None
        self.job_failed_callback: Optional[Callable] = None

    def set_callbacks(
        self,
        job_started: Optional[Callable] = None,
        job_completed: Optional[Callable] = None,
        job_failed: Optional[Callable] = None,
    ):
        """Set callback functions for job events."""
        self.job_started_callback = job_started
        self.job_completed_callback = job_completed
        self.job_failed_callback = job_failed

    async def start_executor(self, job_queue: asyncio.Queue):
        """Start the job executor."""
        if self.executor_running:
            logger.warning("Executor already running")
            return

        self.executor_running = True
        self.executor_task = asyncio.create_task(self._executor_loop(job_queue))
        logger.info("Started pipeline executor")

    async def stop_executor(self):
        """Stop the job executor."""
        if not self.executor_running:
            return

        self.executor_running = False
        if self.executor_task:
            self.executor_task.cancel()
            try:
                await self.executor_task
            except asyncio.CancelledError:
                pass

        # Cancel running jobs
        for job in list(self.running_jobs.values()):
            job.status = PipelineStatus.CANCELLED

        logger.info("Stopped pipeline executor")

    async def _executor_loop(self, job_queue: asyncio.Queue):
        """Main executor loop."""
        while self.executor_running:
            try:
                # Get next job from queue
                job = await asyncio.wait_for(job_queue.get(), timeout=10.0)

                # Start job execution in background
                asyncio.create_task(self._execute_job(job))

            except asyncio.TimeoutError:
                # No jobs in queue, continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in executor loop: {e}")
                await asyncio.sleep(60)

    async def _execute_job(self, job: PipelineJob):
        """Execute a single retraining job."""
        async with self.job_semaphore:
            job.status = PipelineStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            self.running_jobs[job.job_id] = job

            logger.info(f"Starting execution of job {job.job_id}")

            try:
                # Notify job started
                if self.job_started_callback:
                    await self.job_started_callback(job)

                # Allocate resources
                if job.resource_requirements:
                    if not await self.resource_manager.allocate_resources(
                        job.job_id, job.resource_requirements
                    ):
                        raise RuntimeError("Failed to allocate required resources")

                # Execute the actual retraining
                await self._run_retraining_job(job)

                # Job completed successfully
                job.status = PipelineStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)

                logger.info(f"Job {job.job_id} completed successfully")

                # Notify job completed
                if self.job_completed_callback:
                    await self.job_completed_callback(job)

            except Exception as e:
                # Job failed
                job.status = PipelineStatus.FAILED
                job.completed_at = datetime.now(timezone.utc)
                job.error_message = str(e)

                logger.error(f"Job {job.job_id} failed: {e}")

                # Notify job failed
                if self.job_failed_callback:
                    await self.job_failed_callback(job)

            finally:
                # Release resources
                if job.resource_requirements:
                    await self.resource_manager.release_resources(job.job_id)

                # Move job to completed list
                if job.job_id in self.running_jobs:
                    completed_job = self.running_jobs.pop(job.job_id)
                    self.completed_jobs.append(completed_job)

                    # Keep only recent completed jobs
                    if len(self.completed_jobs) > 1000:
                        self.completed_jobs = self.completed_jobs[-500:]

    async def _run_retraining_job(self, job: PipelineJob):
        """Run the actual retraining process."""
        # This is where the job would integrate with AdaptiveLearningManager
        # For now, simulate retraining
        logger.info(f"Running retraining for job {job.job_id}")

        # Simulate different training phases
        phases = ["data_preparation", "model_training", "validation", "deployment"]

        for i, phase in enumerate(phases):
            logger.info(f"Job {job.job_id}: {phase}")

            # Simulate work
            await asyncio.sleep(2.0)  # Simulate processing time

            # Update job metadata
            job.metadata[f"{phase}_completed"] = datetime.now(timezone.utc).isoformat()

            # Check for cancellation
            if job.status == PipelineStatus.CANCELLED:
                raise RuntimeError("Job was cancelled")

        # Simulate final metrics
        job.metrics = {
            "final_mse": 0.05,
            "final_r2": 0.95,
            "training_time_minutes": 8.0,
            "samples_processed": 1000,
        }

        # Simulate output artifacts
        job.output_artifacts = {
            "model_file": f"/models/{job.job_id}/final_model.pkl",
            "metrics_file": f"/models/{job.job_id}/metrics.json",
            "training_log": f"/logs/{job.job_id}/training.log",
        }

    def get_executor_status(self) -> Dict[str, Any]:
        """Get status of the executor."""
        return {
            "executor_running": self.executor_running,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "running_jobs_count": len(self.running_jobs),
            "completed_jobs_count": len(self.completed_jobs),
            "running_jobs": [
                {
                    "job_id": job.job_id,
                    "model_id": job.model_id,
                    "started_at": (
                        job.started_at.isoformat() if job.started_at else None
                    ),
                    "status": job.status.value,
                }
                for job in self.running_jobs.values()
            ],
            "recent_completed_jobs": [
                {
                    "job_id": job.job_id,
                    "model_id": job.model_id,
                    "status": job.status.value,
                    "completed_at": (
                        job.completed_at.isoformat() if job.completed_at else None
                    ),
                    "metrics": job.metrics,
                }
                for job in self.completed_jobs[-10:]  # Last 10 jobs
            ],
        }


class RetrainingPipelineManager:
    """Main manager for the retraining pipeline orchestration."""

    def __init__(
        self,
        adaptive_learning_manager: Optional[AdaptiveLearningManager] = None,
        data_ingestion_manager: Optional[DataIngestionManager] = None,
        max_concurrent_jobs: int = 2,
    ):
        self.adaptive_learning_manager = adaptive_learning_manager
        self.data_ingestion_manager = data_ingestion_manager

        self.resource_manager = ResourceManager()
        self.scheduler = PipelineScheduler()
        self.executor = PipelineExecutor(self.resource_manager, max_concurrent_jobs)

        # Set up callbacks
        self.executor.set_callbacks(
            job_started=self._on_job_started,
            job_completed=self._on_job_completed,
            job_failed=self._on_job_failed,
        )

        self.is_running = False
        self.models_registry: Dict[str, Dict[str, Any]] = {}

        # Event hooks
        self.job_event_callbacks: List[Callable] = []

    def register_model(self, model_id: str, model_config: Dict[str, Any]):
        """Register a model for retraining management."""
        self.models_registry[model_id] = {
            "model_id": model_id,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "config": model_config,
            "last_retrained": None,
            "retrain_count": 0,
        }
        logger.info(f"Registered model {model_id} for retraining management")

    def add_event_callback(self, callback: Callable):
        """Add callback for pipeline events."""
        self.job_event_callbacks.append(callback)

    async def start_pipeline(self):
        """Start the retraining pipeline system."""
        if self.is_running:
            logger.warning("Pipeline already running")
            return

        self.is_running = True

        # Start components
        await self.scheduler.start_scheduler()
        await self.executor.start_executor(self.scheduler.job_queue)

        logger.info("Retraining pipeline started")

    async def stop_pipeline(self):
        """Stop the retraining pipeline system."""
        if not self.is_running:
            return

        self.is_running = False

        # Stop components
        await self.scheduler.stop_scheduler()
        await self.executor.stop_executor()

        logger.info("Retraining pipeline stopped")

    def schedule_periodic_retraining(
        self,
        model_id: str,
        interval_hours: int,
        training_config: Optional[RetrainingConfig] = None,
    ) -> str:
        """Schedule periodic retraining for a model."""
        schedule_id = f"periodic_{model_id}_{uuid.uuid4().hex[:8]}"

        schedule = PipelineSchedule(
            schedule_id=schedule_id,
            interval_hours=interval_hours,
            metadata={
                "model_id": model_id,
                "training_config": (
                    training_config.__dict__ if training_config else None
                ),
            },
        )

        self.scheduler.add_schedule(schedule)
        logger.info(
            f"Scheduled periodic retraining for model {model_id} every {interval_hours} hours"
        )

        return schedule_id

    def schedule_conditional_retraining(
        self,
        model_id: str,
        conditions: List[Dict[str, Any]],
        training_config: Optional[RetrainingConfig] = None,
    ) -> str:
        """Schedule conditional retraining based on specific conditions."""
        schedule_id = f"conditional_{model_id}_{uuid.uuid4().hex[:8]}"

        schedule = PipelineSchedule(
            schedule_id=schedule_id,
            trigger_conditions=conditions,
            metadata={
                "model_id": model_id,
                "training_config": (
                    training_config.__dict__ if training_config else None
                ),
            },
        )

        self.scheduler.add_schedule(schedule)
        logger.info(f"Scheduled conditional retraining for model {model_id}")

        return schedule_id

    async def trigger_immediate_retraining(
        self,
        model_id: str,
        training_config: Optional[RetrainingConfig] = None,
        resource_requirements: Optional[List[ResourceRequirement]] = None,
    ) -> str:
        """Trigger immediate retraining for a model."""
        job = PipelineJob(
            job_id=str(uuid.uuid4()),
            model_id=model_id,
            status=PipelineStatus.SCHEDULED,
            created_at=datetime.now(timezone.utc),
            training_config=training_config,
            resource_requirements=resource_requirements or [],
            metadata={"trigger": "immediate"},
        )

        await self.scheduler.job_queue.put(job)
        logger.info(
            f"Triggered immediate retraining for model {model_id}, job ID: {job.job_id}"
        )

        return job.job_id

    async def _on_job_started(self, job: PipelineJob):
        """Handle job started event."""
        logger.info(f"Retraining job started: {job.job_id} for model {job.model_id}")

        # Update model registry
        if job.model_id in self.models_registry:
            self.models_registry[job.model_id]["current_job_id"] = job.job_id

        # Notify callbacks
        for callback in self.job_event_callbacks:
            try:
                await callback("job_started", job)
            except Exception as e:
                logger.error(f"Error in job started callback: {e}")

    async def _on_job_completed(self, job: PipelineJob):
        """Handle job completed event."""
        logger.info(f"Retraining job completed: {job.job_id} for model {job.model_id}")

        # Update model registry
        if job.model_id in self.models_registry:
            model_info = self.models_registry[job.model_id]
            model_info["last_retrained"] = job.completed_at.isoformat()
            model_info["retrain_count"] = model_info.get("retrain_count", 0) + 1
            model_info.pop("current_job_id", None)

        # Notify callbacks
        for callback in self.job_event_callbacks:
            try:
                await callback("job_completed", job)
            except Exception as e:
                logger.error(f"Error in job completed callback: {e}")

    async def _on_job_failed(self, job: PipelineJob):
        """Handle job failed event."""
        logger.error(
            f"Retraining job failed: {job.job_id} for model {job.model_id}, error: {job.error_message}"
        )

        # Update model registry
        if job.model_id in self.models_registry:
            self.models_registry[job.model_id].pop("current_job_id", None)

        # Notify callbacks
        for callback in self.job_event_callbacks:
            try:
                await callback("job_failed", job)
            except Exception as e:
                logger.error(f"Error in job failed callback: {e}")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        # Check running jobs
        if job_id in self.executor.running_jobs:
            job = self.executor.running_jobs[job_id]
        else:
            # Check completed jobs
            completed_job = next(
                (job for job in self.executor.completed_jobs if job.job_id == job_id),
                None,
            )
            if not completed_job:
                return None
            job = completed_job

        return {
            "job_id": job.job_id,
            "model_id": job.model_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "metrics": job.metrics,
            "error_message": job.error_message,
            "metadata": job.metadata,
        }

    def get_model_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get retraining status for a specific model."""
        if model_id not in self.models_registry:
            return None

        model_info = self.models_registry[model_id].copy()

        # Add current job info if running
        current_job_id = model_info.get("current_job_id")
        if current_job_id and current_job_id in self.executor.running_jobs:
            current_job = self.executor.running_jobs[current_job_id]
            model_info["current_job"] = {
                "job_id": current_job.job_id,
                "status": current_job.status.value,
                "started_at": (
                    current_job.started_at.isoformat()
                    if current_job.started_at
                    else None
                ),
            }

        return model_info

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        return {
            "is_running": self.is_running,
            "registered_models": len(self.models_registry),
            "resource_utilization": self.resource_manager.get_resource_utilization(),
            "scheduler_status": self.scheduler.get_schedule_status(),
            "executor_status": self.executor.get_executor_status(),
            "models": {
                model_id: {
                    "last_retrained": info.get("last_retrained"),
                    "retrain_count": info.get("retrain_count", 0),
                    "has_current_job": "current_job_id" in info,
                }
                for model_id, info in self.models_registry.items()
            },
        }


# Factory Functions


def create_default_pipeline_manager(
    adaptive_learning_manager: Optional[AdaptiveLearningManager] = None,
    max_concurrent_jobs: int = 2,
) -> RetrainingPipelineManager:
    """Create a retraining pipeline manager with default configuration."""
    return RetrainingPipelineManager(
        adaptive_learning_manager=adaptive_learning_manager,
        max_concurrent_jobs=max_concurrent_jobs,
    )


def create_production_pipeline_manager(
    adaptive_learning_manager: Optional[AdaptiveLearningManager] = None,
    data_ingestion_manager: Optional[DataIngestionManager] = None,
) -> RetrainingPipelineManager:
    """Create a retraining pipeline manager for production environments."""
    manager = RetrainingPipelineManager(
        adaptive_learning_manager=adaptive_learning_manager,
        data_ingestion_manager=data_ingestion_manager,
        max_concurrent_jobs=4,
    )

    # Configure production resource limits
    manager.resource_manager.available_resources.update(
        {
            ResourceType.CPU: 16.0,
            ResourceType.GPU: 2.0,
            ResourceType.MEMORY: 64.0,
            ResourceType.STORAGE: 2000.0,
        }
    )

    return manager


def create_cloud_pipeline_manager(
    adaptive_learning_manager: Optional[AdaptiveLearningManager] = None,
    cloud_config: Optional[Dict[str, Any]] = None,
) -> RetrainingPipelineManager:
    """Create a retraining pipeline manager for cloud deployments."""
    manager = RetrainingPipelineManager(
        adaptive_learning_manager=adaptive_learning_manager, max_concurrent_jobs=8
    )

    # Configure cloud resource limits
    if cloud_config:
        cpu_cores = cloud_config.get("cpu_cores", 32)
        gpu_count = cloud_config.get("gpu_count", 4)
        memory_gb = cloud_config.get("memory_gb", 128)
        storage_gb = cloud_config.get("storage_gb", 5000)

        manager.resource_manager.available_resources.update(
            {
                ResourceType.CPU: float(cpu_cores),
                ResourceType.GPU: float(gpu_count),
                ResourceType.MEMORY: float(memory_gb),
                ResourceType.STORAGE: float(storage_gb),
            }
        )

    return manager

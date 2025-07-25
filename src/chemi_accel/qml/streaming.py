"""
Streaming prediction system for continuous QML processing.

This module provides streaming interfaces for continuous, real-time QML
property prediction with efficient buffer management and flow control.
"""

import asyncio
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from .base import BaseQMLModel, QMLModelError, QMLPrediction, QuantumProperty
from .memory_manager import MemoryAwareProcessor, MemoryStrategy
from .on_the_fly import (
    BatchPredictionRequest,
    BatchPredictionResult,
    GeometryInput,
    OnTheFlyPredictor,
)


class StreamingMode(Enum):
    """Streaming processing modes."""

    PUSH = "push"  # Push-based streaming
    PULL = "pull"  # Pull-based streaming
    HYBRID = "hybrid"  # Hybrid push-pull


class BufferStrategy(Enum):
    """Buffer management strategies."""

    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based
    ADAPTIVE = "adaptive"  # Adaptive buffer sizing
    SMART = "smart"  # Smart buffering with backpressure


class FlowControl(Enum):
    """Flow control mechanisms."""

    NONE = "none"  # No flow control
    BACKPRESSURE = "backpressure"  # Backpressure-based
    RATE_LIMIT = "rate_limit"  # Rate limiting
    ADAPTIVE_RATE = "adaptive_rate"  # Adaptive rate limiting


@dataclass
class StreamingConfig:
    """Configuration for streaming predictions."""

    buffer_size: int = 64
    max_buffer_size: int = 1024
    timeout: float = 30.0
    batch_timeout: float = 1.0
    max_workers: int = 4
    memory_strategy: MemoryStrategy = MemoryStrategy.BALANCED
    buffer_strategy: BufferStrategy = BufferStrategy.ADAPTIVE
    flow_control: FlowControl = FlowControl.BACKPRESSURE
    enable_async: bool = True
    max_concurrent_batches: int = 3
    prediction_cache_size: int = 1000

    # Performance settings
    target_throughput: float = 100.0  # predictions/second
    max_latency: float = 5.0  # seconds
    min_batch_efficiency: float = 0.8  # minimum batch utilization

    # Quality settings
    enable_prediction_validation: bool = True
    max_retry_attempts: int = 3
    error_tolerance: float = 0.1  # fraction of failed predictions allowed


@dataclass
class StreamingMetrics:
    """Metrics for streaming prediction performance."""

    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    total_batches: int = 0
    average_batch_size: float = 0.0
    throughput: float = 0.0  # predictions/second
    average_latency: float = 0.0  # seconds
    buffer_utilization: float = 0.0
    memory_usage: float = 0.0  # GB
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    uptime: float = 0.0  # seconds

    def update(self, batch_result: BatchPredictionResult):
        """Update metrics with batch result."""
        self.total_batches += 1
        batch_predictions = len(batch_result.predictions)
        batch_successful = batch_predictions - len(batch_result.errors)

        self.total_predictions += batch_predictions
        self.successful_predictions += batch_successful
        self.failed_predictions += len(batch_result.errors)

        # Update averages
        self.average_batch_size = self.total_predictions / self.total_batches
        self.error_rate = self.failed_predictions / max(self.total_predictions, 1)

        if batch_result.execution_time > 0:
            batch_throughput = batch_predictions / batch_result.execution_time
            # Exponential moving average
            self.throughput = 0.9 * self.throughput + 0.1 * batch_throughput


@dataclass
class StreamingRequest:
    """Request for streaming prediction."""

    request_id: str
    molecule: Union[str, Chem.Mol, GeometryInput]
    properties: List[QuantumProperty]
    priority: int = 1
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())


@dataclass
class StreamingResponse:
    """Response from streaming prediction."""

    request_id: str
    prediction: Optional[QMLPrediction]
    error: Optional[str] = None
    processing_time: float = 0.0
    queue_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def total_time(self) -> float:
        return self.processing_time + self.queue_time

    @property
    def success(self) -> bool:
        return self.prediction is not None and self.error is None


class AdaptiveBuffer:
    """Adaptive buffer for streaming predictions with smart sizing."""

    def __init__(
        self,
        initial_size: int = 32,
        max_size: int = 1024,
        strategy: BufferStrategy = BufferStrategy.ADAPTIVE,
    ):
        """
        Initialize adaptive buffer.

        Args:
            initial_size: Initial buffer size
            max_size: Maximum buffer size
            strategy: Buffer management strategy
        """
        self.initial_size = initial_size
        self.max_size = max_size
        self.strategy = strategy
        self.current_size = initial_size

        self._buffer = queue.Queue(maxsize=max_size)
        self._priority_queue = queue.PriorityQueue(maxsize=max_size)
        self._stats = {
            "buffer_hits": 0,
            "buffer_misses": 0,
            "average_wait_time": 0.0,
            "size_adjustments": 0,
        }

        self._lock = threading.RLock()
        self._size_history = []
        self._performance_history = []

    def put(self, item: StreamingRequest, timeout: Optional[float] = None) -> bool:
        """Add item to buffer."""
        start_time = time.time()

        try:
            if self.strategy == BufferStrategy.PRIORITY:
                # Use priority queue
                priority_item = (-item.priority, item.timestamp, item)
                self._priority_queue.put(priority_item, timeout=timeout)
            else:
                # Use regular queue
                self._buffer.put(item, timeout=timeout)

            self._stats["buffer_hits"] += 1
            return True

        except queue.Full:
            self._stats["buffer_misses"] += 1

            # Try adaptive resizing
            if self.strategy == BufferStrategy.ADAPTIVE:
                if self._try_resize_buffer():
                    return self.put(item, timeout)

            return False

        finally:
            wait_time = time.time() - start_time
            self._update_wait_time(wait_time)

    def get(self, timeout: Optional[float] = None) -> Optional[StreamingRequest]:
        """Get item from buffer."""
        try:
            if self.strategy == BufferStrategy.PRIORITY:
                _, _, item = self._priority_queue.get(timeout=timeout)
                self._priority_queue.task_done()
                return item
            else:
                item = self._buffer.get(timeout=timeout)
                self._buffer.task_done()
                return item

        except queue.Empty:
            return None

    def get_batch(
        self, batch_size: int, timeout: float = 1.0
    ) -> List[StreamingRequest]:
        """Get batch of items from buffer."""
        batch = []
        start_time = time.time()

        # Get first item (blocking)
        first_item = self.get(timeout=timeout)
        if first_item is None:
            return batch

        batch.append(first_item)

        # Get remaining items (non-blocking or short timeout)
        remaining_timeout = max(0, timeout - (time.time() - start_time))
        per_item_timeout = remaining_timeout / max(1, batch_size - 1)

        for _ in range(batch_size - 1):
            if time.time() - start_time >= timeout:
                break

            item = self.get(timeout=per_item_timeout)
            if item is None:
                break

            batch.append(item)

        return batch

    def size(self) -> int:
        """Get current buffer size."""
        if self.strategy == BufferStrategy.PRIORITY:
            return self._priority_queue.qsize()
        else:
            return self._buffer.qsize()

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size() >= self.current_size

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.size() == 0

    def _try_resize_buffer(self) -> bool:
        """Try to resize buffer adaptively."""
        with self._lock:
            # Check if we can increase size
            if self.current_size < self.max_size:
                new_size = min(self.current_size * 2, self.max_size)

                # Create new buffer with larger size
                old_items = []

                # Drain old buffer
                while not self._buffer.empty():
                    try:
                        item = self._buffer.get_nowait()
                        old_items.append(item)
                    except queue.Empty:
                        break

                # Create new buffer
                self._buffer = queue.Queue(maxsize=new_size)
                self.current_size = new_size

                # Put items back
                for item in old_items:
                    try:
                        self._buffer.put_nowait(item)
                    except queue.Full:
                        break

                self._stats["size_adjustments"] += 1
                self._size_history.append((time.time(), new_size))

                logging.info(f"Buffer resized to {new_size}")
                return True

        return False

    def _update_wait_time(self, wait_time: float):
        """Update average wait time statistics."""
        current_avg = self._stats["average_wait_time"]
        total_requests = self._stats["buffer_hits"] + self._stats["buffer_misses"]

        if total_requests > 0:
            self._stats["average_wait_time"] = (
                current_avg * (total_requests - 1) + wait_time
            ) / total_requests

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "current_size": self.current_size,
            "max_size": self.max_size,
            "current_fill": self.size(),
            "utilization": self.size() / self.current_size,
            "strategy": self.strategy.value,
            "stats": self._stats.copy(),
            "size_history": self._size_history[-10:],  # Last 10 size changes
        }


class StreamingPredictor:
    """
    Main streaming predictor for continuous QML property prediction.

    Provides efficient streaming interfaces with buffering, flow control,
    and adaptive performance optimization.
    """

    def __init__(self, model: BaseQMLModel, config: StreamingConfig = None):
        """
        Initialize streaming predictor.

        Args:
            model: QML model for predictions
            config: Streaming configuration
        """
        self.model = model
        self.config = config or StreamingConfig()

        # Initialize components
        self.predictor = OnTheFlyPredictor(
            model=model,
            max_batch_size=self.config.buffer_size,
            cache_size=self.config.prediction_cache_size,
        )

        self.memory_processor = MemoryAwareProcessor(
            strategy=self.config.memory_strategy
        )

        self.buffer = AdaptiveBuffer(
            initial_size=self.config.buffer_size,
            max_size=self.config.max_buffer_size,
            strategy=self.config.buffer_strategy,
        )

        # State management
        self._running = False
        self._worker_threads = []
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._response_queue = queue.Queue()
        self._metrics = StreamingMetrics()
        self._start_time = None

        # Flow control
        self._flow_controller = self._create_flow_controller()

        # Async support
        if self.config.enable_async:
            self._async_loop = None
            self._async_tasks = set()

    def start(self):
        """Start streaming prediction service."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()

        # Start worker threads
        for i in range(self.config.max_workers):
            worker = threading.Thread(
                target=self._worker_loop, name=f"StreamingWorker-{i}", daemon=True
            )
            worker.start()
            self._worker_threads.append(worker)

        # Start metrics updater
        metrics_thread = threading.Thread(
            target=self._metrics_loop, name="StreamingMetrics", daemon=True
        )
        metrics_thread.start()
        self._worker_threads.append(metrics_thread)

        logging.info("Streaming predictor started")

    def stop(self):
        """Stop streaming prediction service."""
        if not self._running:
            return

        self._running = False

        # Wait for workers to finish
        for worker in self._worker_threads:
            worker.join(timeout=5.0)

        # Clean up async tasks
        if self.config.enable_async and self._async_loop:
            for task in self._async_tasks:
                task.cancel()

        # Shutdown executor
        self._executor.shutdown(wait=True)

        logging.info("Streaming predictor stopped")

    def predict(
        self,
        molecule: Union[str, Chem.Mol, GeometryInput],
        properties: List[QuantumProperty],
        priority: int = 1,
        timeout: Optional[float] = None,
    ) -> StreamingResponse:
        """
        Submit prediction request (synchronous).

        Args:
            molecule: Input molecule
            properties: Properties to predict
            priority: Request priority (higher = more important)
            timeout: Request timeout

        Returns:
            StreamingResponse with prediction result
        """
        request = StreamingRequest(
            request_id=str(uuid.uuid4()),
            molecule=molecule,
            properties=properties,
            priority=priority,
            timeout=timeout or self.config.timeout,
        )

        # Submit to buffer
        queue_start = time.time()
        if not self.buffer.put(request, timeout=timeout):
            return StreamingResponse(
                request_id=request.request_id,
                error="Buffer full - request rejected",
                queue_time=time.time() - queue_start,
            )

        # Wait for response
        response_timeout = timeout or self.config.timeout
        try:
            while True:
                try:
                    response = self._response_queue.get(timeout=0.1)
                    self._response_queue.task_done()

                    if response.request_id == request.request_id:
                        response.queue_time = (
                            time.time() - queue_start - response.processing_time
                        )
                        return response
                    else:
                        # Put back response for different request
                        self._response_queue.put(response)

                except queue.Empty:
                    if time.time() - queue_start > response_timeout:
                        return StreamingResponse(
                            request_id=request.request_id,
                            error="Request timeout",
                            queue_time=time.time() - queue_start,
                        )
                    continue

        except KeyboardInterrupt:
            return StreamingResponse(
                request_id=request.request_id,
                error="Request interrupted",
                queue_time=time.time() - queue_start,
            )

    def predict_stream(
        self, requests: Iterator[StreamingRequest]
    ) -> Iterator[StreamingResponse]:
        """
        Process streaming requests (generator interface).

        Args:
            requests: Iterator of streaming requests

        Yields:
            StreamingResponse objects
        """
        pending_requests = {}

        for request in requests:
            # Submit request
            if self.buffer.put(request, timeout=self.config.batch_timeout):
                pending_requests[request.request_id] = request
            else:
                yield StreamingResponse(
                    request_id=request.request_id,
                    error="Buffer full - request rejected",
                )

            # Check for completed responses
            while not self._response_queue.empty():
                try:
                    response = self._response_queue.get_nowait()
                    self._response_queue.task_done()

                    if response.request_id in pending_requests:
                        del pending_requests[response.request_id]
                        yield response

                except queue.Empty:
                    break

        # Wait for remaining responses
        while pending_requests:
            try:
                response = self._response_queue.get(timeout=self.config.timeout)
                self._response_queue.task_done()

                if response.request_id in pending_requests:
                    del pending_requests[response.request_id]
                    yield response

            except queue.Empty:
                # Timeout - create error responses for remaining requests
                for request_id in pending_requests.keys():
                    yield StreamingResponse(
                        request_id=request_id, error="Processing timeout"
                    )
                break

    async def predict_async(
        self,
        molecule: Union[str, Chem.Mol, GeometryInput],
        properties: List[QuantumProperty],
        priority: int = 1,
        timeout: Optional[float] = None,
    ) -> StreamingResponse:
        """
        Submit prediction request (asynchronous).

        Args:
            molecule: Input molecule
            properties: Properties to predict
            priority: Request priority
            timeout: Request timeout

        Returns:
            StreamingResponse with prediction result
        """
        if not self.config.enable_async:
            raise QMLModelError("Async not enabled in configuration")

        loop = asyncio.get_event_loop()

        # Run synchronous predict in thread pool
        response = await loop.run_in_executor(
            self._executor, self.predict, molecule, properties, priority, timeout
        )

        return response

    async def predict_stream_async(
        self, requests: AsyncIterator[StreamingRequest]
    ) -> AsyncIterator[StreamingResponse]:
        """
        Process streaming requests (async generator interface).

        Args:
            requests: Async iterator of streaming requests

        Yields:
            StreamingResponse objects
        """
        pending_responses = asyncio.Queue()

        async def response_collector():
            """Collect responses in background."""
            while self._running:
                try:
                    if not self._response_queue.empty():
                        response = self._response_queue.get_nowait()
                        await pending_responses.put(response)
                    else:
                        await asyncio.sleep(0.01)
                except Exception as e:
                    logging.error(f"Response collector error: {e}")

        # Start response collector
        collector_task = asyncio.create_task(response_collector())
        self._async_tasks.add(collector_task)

        try:
            async for request in requests:
                # Submit request
                loop = asyncio.get_event_loop()
                submitted = await loop.run_in_executor(
                    None, self.buffer.put, request, self.config.batch_timeout
                )

                if not submitted:
                    yield StreamingResponse(
                        request_id=request.request_id,
                        error="Buffer full - request rejected",
                    )

                # Yield any available responses
                try:
                    while True:
                        response = await asyncio.wait_for(
                            pending_responses.get(), timeout=0.01
                        )
                        yield response
                except asyncio.TimeoutError:
                    continue

        finally:
            collector_task.cancel()
            self._async_tasks.discard(collector_task)

    def _worker_loop(self):
        """Main worker loop for processing batches."""
        while self._running:
            try:
                # Get batch from buffer
                batch_requests = self.buffer.get_batch(
                    batch_size=self.config.buffer_size,
                    timeout=self.config.batch_timeout,
                )

                if not batch_requests:
                    continue

                # Apply flow control
                if not self._flow_controller.allow_batch(len(batch_requests)):
                    # Put requests back and wait
                    for request in batch_requests:
                        self.buffer.put(request, timeout=0.1)
                    time.sleep(0.1)
                    continue

                # Process batch
                self._process_batch(batch_requests)

            except Exception as e:
                logging.error(f"Worker loop error: {e}")
                time.sleep(0.1)

    def _process_batch(self, requests: List[StreamingRequest]):
        """Process a batch of requests."""
        start_time = time.time()

        try:
            # Create batch prediction request
            molecules = [req.molecule for req in requests]
            all_properties = set()
            for req in requests:
                all_properties.update(req.properties)

            batch_request = BatchPredictionRequest(
                molecules=molecules,
                properties=list(all_properties),
                batch_size=len(requests),
                use_cache=True,
            )

            # Execute batch prediction
            with self.memory_processor.memory_managed_processing():
                batch_result = self.predictor.predict_batch(batch_request)

            # Create individual responses
            for i, request in enumerate(requests):
                try:
                    if i < len(batch_result.predictions):
                        prediction = batch_result.predictions[i]

                        # Filter properties to match request
                        filtered_props = {
                            prop: value
                            for prop, value in prediction.properties.items()
                            if prop in request.properties
                        }

                        filtered_prediction = QMLPrediction(
                            properties=filtered_props,
                            confidence=prediction.confidence,
                            uncertainty=prediction.uncertainty,
                            model_name=prediction.model_name,
                            metadata=prediction.metadata,
                        )

                        response = StreamingResponse(
                            request_id=request.request_id,
                            prediction=filtered_prediction,
                            processing_time=batch_result.execution_time / len(requests),
                        )
                    else:
                        response = StreamingResponse(
                            request_id=request.request_id,
                            error="Batch processing failed",
                        )

                    self._response_queue.put(response, timeout=1.0)

                except Exception as e:
                    error_response = StreamingResponse(
                        request_id=request.request_id,
                        error=f"Response processing error: {e}",
                    )
                    self._response_queue.put(error_response, timeout=1.0)

            # Update metrics
            self._metrics.update(batch_result)

        except Exception as e:
            logging.error(f"Batch processing error: {e}")

            # Create error responses for all requests
            for request in requests:
                error_response = StreamingResponse(
                    request_id=request.request_id, error=f"Batch processing failed: {e}"
                )
                try:
                    self._response_queue.put(error_response, timeout=0.1)
                except queue.Full:
                    pass

    def _create_flow_controller(self) -> "FlowController":
        """Create flow controller based on configuration."""
        if self.config.flow_control == FlowControl.NONE:
            return NoFlowControl()
        elif self.config.flow_control == FlowControl.BACKPRESSURE:
            return BackpressureController(self.config)
        elif self.config.flow_control == FlowControl.RATE_LIMIT:
            return RateLimitController(self.config)
        elif self.config.flow_control == FlowControl.ADAPTIVE_RATE:
            return AdaptiveRateController(self.config)
        else:
            return NoFlowControl()

    def _metrics_loop(self):
        """Update metrics periodically."""
        while self._running:
            try:
                # Update uptime
                if self._start_time:
                    self._metrics.uptime = time.time() - self._start_time

                # Update buffer utilization
                buffer_stats = self.buffer.get_statistics()
                self._metrics.buffer_utilization = buffer_stats["utilization"]

                # Update memory usage
                memory_stats = self.memory_processor.get_memory_statistics()
                if "current_profile" in memory_stats:
                    self._metrics.memory_usage = memory_stats[
                        "current_profile"
                    ].process_memory

                time.sleep(1.0)  # Update every second

            except Exception as e:
                logging.error(f"Metrics update error: {e}")
                time.sleep(1.0)

    def get_metrics(self) -> StreamingMetrics:
        """Get current streaming metrics."""
        return self._metrics

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information."""
        return {
            "running": self._running,
            "metrics": self._metrics,
            "buffer_stats": self.buffer.get_statistics(),
            "memory_stats": self.memory_processor.get_memory_statistics(),
            "config": self.config,
            "worker_threads": len([t for t in self._worker_threads if t.is_alive()]),
            "response_queue_size": self._response_queue.qsize(),
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Flow control implementations
class FlowController:
    """Base class for flow control mechanisms."""

    def allow_batch(self, batch_size: int) -> bool:
        """Check if batch should be allowed to proceed."""
        return True


class NoFlowControl(FlowController):
    """No flow control - always allow."""

    pass


class BackpressureController(FlowController):
    """Backpressure-based flow control."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self._last_check = time.time()
        self._overload_count = 0

    def allow_batch(self, batch_size: int) -> bool:
        """Allow batch based on system load."""
        # Simple backpressure: check memory usage
        current_time = time.time()

        if current_time - self._last_check > 1.0:  # Check every second
            # This would integrate with memory monitor
            # For now, simple implementation
            self._last_check = current_time

            # Allow if not overloaded
            return self._overload_count < 5

        return True


class RateLimitController(FlowController):
    """Rate limiting flow control."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self._last_batch_time = 0
        self._min_interval = 1.0 / config.target_throughput  # seconds per prediction

    def allow_batch(self, batch_size: int) -> bool:
        """Allow batch based on rate limit."""
        current_time = time.time()
        required_interval = self._min_interval * batch_size

        if current_time - self._last_batch_time >= required_interval:
            self._last_batch_time = current_time
            return True

        return False


class AdaptiveRateController(FlowController):
    """Adaptive rate limiting based on performance."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self._performance_history = []
        self._current_rate = config.target_throughput

    def allow_batch(self, batch_size: int) -> bool:
        """Allow batch based on adaptive rate."""
        # This would adapt rate based on recent performance
        # Simplified implementation
        return True  # Always allow for now

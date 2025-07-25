"""
Advanced memory management for large-scale QML batch processing.

This module provides sophisticated memory management capabilities including
GPU memory optimization, smart batching, and memory-aware processing strategies.
"""

import gc
import logging
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

try:
    import numpy as np
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .on_the_fly import GeometryInput


class MemoryStrategy(Enum):
    """Memory management strategies."""

    CONSERVATIVE = "conservative"  # Minimal memory usage
    BALANCED = "balanced"  # Balance speed and memory
    AGGRESSIVE = "aggressive"  # Maximum speed, higher memory usage
    ADAPTIVE = "adaptive"  # Adapt based on available memory


class BatchingStrategy(Enum):
    """Batching strategies for memory optimization."""

    FIXED = "fixed"  # Fixed batch size
    DYNAMIC = "dynamic"  # Dynamic based on memory
    GRADIENT = "gradient"  # Gradient-based adjustment
    MOLECULAR_SIZE = "molecular_size"  # Based on molecular complexity


@dataclass
class MemoryProfile:
    """Memory usage profile."""

    system_total: float  # Total system memory (GB)
    system_available: float  # Available system memory (GB)
    gpu_total: float  # Total GPU memory (GB)
    gpu_available: float  # Available GPU memory (GB)
    process_memory: float  # Current process memory (GB)
    peak_memory: float  # Peak memory usage (GB)
    timestamp: float = field(default_factory=time.time)


@dataclass
class BatchMetrics:
    """Metrics for batch processing."""

    batch_size: int
    memory_usage: float  # GB
    processing_time: float  # seconds
    throughput: float  # molecules/second
    molecules_per_gb: float
    success_rate: float
    avg_molecular_size: float


class MemoryMonitor:
    """Monitor system and GPU memory usage."""

    def __init__(
        self,
        monitoring_interval: float = 1.0,
        warning_threshold: float = 0.9,
        critical_threshold: float = 0.95,
    ):
        """
        Initialize memory monitor.

        Args:
            monitoring_interval: Monitoring interval in seconds
            warning_threshold: Warning threshold (fraction of total memory)
            critical_threshold: Critical threshold (fraction of total memory)
        """
        self.monitoring_interval = monitoring_interval
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self._monitoring = False
        self._monitor_thread = None
        self._memory_history = []
        self._callbacks = {}
        self._lock = threading.Lock()

    def start_monitoring(self):
        """Start memory monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def get_current_profile(self) -> MemoryProfile:
        """Get current memory profile."""
        # System memory
        system_mem = psutil.virtual_memory()
        system_total = system_mem.total / (1024**3)  # GB
        system_available = system_mem.available / (1024**3)

        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024**3)

        # GPU memory
        gpu_total = 0.0
        gpu_available = 0.0

        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_total = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                gpu_available = gpu_total - torch.cuda.memory_allocated(0) / (1024**3)
            except Exception:
                pass

        # Peak memory tracking
        peak_memory = max(
            self._memory_history[-100:] if self._memory_history else [process_memory],
            key=lambda x: x if isinstance(x, (int, float)) else x.process_memory,
            default=process_memory,
        )

        if isinstance(peak_memory, MemoryProfile):
            peak_memory = peak_memory.process_memory

        return MemoryProfile(
            system_total=system_total,
            system_available=system_available,
            gpu_total=gpu_total,
            gpu_available=gpu_available,
            process_memory=process_memory,
            peak_memory=peak_memory,
        )

    def register_callback(self, event: str, callback: Callable):
        """Register callback for memory events."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                profile = self.get_current_profile()

                with self._lock:
                    self._memory_history.append(profile)
                    # Keep only recent history
                    if len(self._memory_history) > 1000:
                        self._memory_history = self._memory_history[-500:]

                # Check thresholds
                system_usage = 1 - (profile.system_available / profile.system_total)
                gpu_usage = (
                    1 - (profile.gpu_available / profile.gpu_total)
                    if profile.gpu_total > 0
                    else 0
                )

                if (
                    system_usage > self.critical_threshold
                    or gpu_usage > self.critical_threshold
                ):
                    self._trigger_callbacks("critical", profile)
                elif (
                    system_usage > self.warning_threshold
                    or gpu_usage > self.warning_threshold
                ):
                    self._trigger_callbacks("warning", profile)

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logging.warning(f"Memory monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def _trigger_callbacks(self, event: str, profile: MemoryProfile):
        """Trigger registered callbacks."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(profile)
                except Exception as e:
                    logging.error(f"Memory callback error: {e}")


class SmartBatcher:
    """Intelligent batching system with memory awareness."""

    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 512,
        strategy: BatchingStrategy = BatchingStrategy.DYNAMIC,
        memory_fraction: float = 0.7,
    ):
        """
        Initialize smart batcher.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            strategy: Batching strategy
            memory_fraction: Fraction of available memory to use
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.strategy = strategy
        self.memory_fraction = memory_fraction

        self.monitor = MemoryMonitor()
        self._batch_history = []
        self._performance_metrics = {}

        # Strategy-specific parameters
        self._gradient_momentum = 0.9
        self._gradient_lr = 0.1
        self._size_complexity_map = {}

    def calculate_optimal_batch_size(
        self, molecules: List[Any], memory_profile: Optional[MemoryProfile] = None
    ) -> int:
        """Calculate optimal batch size for given molecules."""
        if memory_profile is None:
            memory_profile = self.monitor.get_current_profile()

        if self.strategy == BatchingStrategy.FIXED:
            return min(self.current_batch_size, len(molecules))

        elif self.strategy == BatchingStrategy.DYNAMIC:
            return self._dynamic_batch_size(molecules, memory_profile)

        elif self.strategy == BatchingStrategy.GRADIENT:
            return self._gradient_batch_size(molecules, memory_profile)

        elif self.strategy == BatchingStrategy.MOLECULAR_SIZE:
            return self._molecular_size_batch_size(molecules, memory_profile)

        else:
            return self.current_batch_size

    def _dynamic_batch_size(
        self, molecules: List[Any], memory_profile: MemoryProfile
    ) -> int:
        """Calculate batch size based on available memory."""
        # Estimate memory per molecule
        avg_molecule_size = self._estimate_molecule_complexity(molecules[:10])  # Sample

        # Available memory for processing
        if memory_profile.gpu_available > 1.0:  # Use GPU if available
            available_memory = memory_profile.gpu_available * self.memory_fraction
        else:
            available_memory = memory_profile.system_available * self.memory_fraction

        # Estimate molecules per GB (based on complexity)
        molecules_per_gb = max(10, 1000 / avg_molecule_size)  # Rough estimate

        optimal_batch = int(available_memory * molecules_per_gb)

        return max(
            self.min_batch_size, min(optimal_batch, self.max_batch_size, len(molecules))
        )

    def _gradient_batch_size(
        self, molecules: List[Any], memory_profile: MemoryProfile
    ) -> int:
        """Gradient-based batch size adjustment."""
        if not self._batch_history:
            return self._dynamic_batch_size(molecules, memory_profile)

        # Analyze recent performance
        recent_metrics = self._batch_history[-10:]  # Last 10 batches

        if len(recent_metrics) < 3:
            return self.current_batch_size

        # Calculate performance gradient
        throughputs = [m.throughput for m in recent_metrics]
        batch_sizes = [m.batch_size for m in recent_metrics]

        # Simple gradient estimation
        if len(throughputs) >= 2:
            throughput_grad = throughputs[-1] - throughputs[-2]
            batch_grad = batch_sizes[-1] - batch_sizes[-2]

            if batch_grad != 0:
                perf_gradient = throughput_grad / batch_grad

                # Adjust batch size based on gradient
                if perf_gradient > 0:  # Performance improving
                    adjustment = int(self.current_batch_size * self._gradient_lr)
                else:  # Performance declining
                    adjustment = int(-self.current_batch_size * self._gradient_lr)

                new_batch_size = self.current_batch_size + adjustment

                return max(
                    self.min_batch_size,
                    min(new_batch_size, self.max_batch_size, len(molecules)),
                )

        return self.current_batch_size

    def _molecular_size_batch_size(
        self, molecules: List[Any], memory_profile: MemoryProfile
    ) -> int:
        """Batch size based on molecular complexity."""
        complexities = [
            self._estimate_molecule_complexity([mol]) for mol in molecules[:20]
        ]
        avg_complexity = np.mean(complexities) if complexities else 50

        # Adjust batch size inversely to molecular complexity
        base_batch = self._dynamic_batch_size(molecules, memory_profile)
        complexity_factor = 50 / max(avg_complexity, 10)  # Normalize to complexity=50

        adjusted_batch = int(base_batch * complexity_factor)

        return max(
            self.min_batch_size,
            min(adjusted_batch, self.max_batch_size, len(molecules)),
        )

    def _estimate_molecule_complexity(self, molecules: List[Any]) -> float:
        """Estimate average molecular complexity."""
        if not molecules:
            return 50.0  # Default complexity

        total_complexity = 0
        valid_molecules = 0

        for mol in molecules[:10]:  # Sample first 10
            try:
                if isinstance(mol, GeometryInput):
                    complexity = len(mol.atomic_numbers)
                elif isinstance(mol, str):
                    # SMILES complexity estimation
                    complexity = len(mol) * 2  # Rough estimate
                elif hasattr(mol, "GetNumAtoms"):
                    # RDKit molecule
                    complexity = mol.GetNumAtoms()
                else:
                    complexity = 50  # Default

                total_complexity += complexity
                valid_molecules += 1

            except Exception:
                continue

        return total_complexity / max(valid_molecules, 1)

    def update_performance(self, metrics: BatchMetrics):
        """Update performance metrics for batch size optimization."""
        self._batch_history.append(metrics)

        # Keep recent history
        if len(self._batch_history) > 100:
            self._batch_history = self._batch_history[-50:]

        # Update current batch size based on strategy
        if self.strategy in [BatchingStrategy.GRADIENT, BatchingStrategy.DYNAMIC]:
            # Adjust for next batch
            if metrics.success_rate < 0.9:  # Many failures
                self.current_batch_size = max(
                    self.min_batch_size, int(self.current_batch_size * 0.8)
                )
            elif metrics.throughput > self._get_average_throughput() * 1.1:
                # Good performance, try increasing
                self.current_batch_size = min(
                    self.max_batch_size, int(self.current_batch_size * 1.1)
                )

    def _get_average_throughput(self) -> float:
        """Get average throughput from recent batches."""
        if not self._batch_history:
            return 1.0

        recent = self._batch_history[-10:]
        return np.mean([m.throughput for m in recent])


class MemoryAwareProcessor:
    """Memory-aware processor for QML batch operations."""

    def __init__(
        self,
        strategy: MemoryStrategy = MemoryStrategy.BALANCED,
        cache_size_gb: float = 1.0,
        gc_threshold: float = 0.8,
        emergency_threshold: float = 0.95,
    ):
        """
        Initialize memory-aware processor.

        Args:
            strategy: Memory management strategy
            cache_size_gb: Maximum cache size in GB
            gc_threshold: Garbage collection threshold
            emergency_threshold: Emergency cleanup threshold
        """
        self.strategy = strategy
        self.cache_size_gb = cache_size_gb
        self.gc_threshold = gc_threshold
        self.emergency_threshold = emergency_threshold

        self.monitor = MemoryMonitor()
        self.batcher = SmartBatcher()

        # Memory management state
        self._memory_cache = {}
        self._cache_size_bytes = 0
        self._last_gc_time = time.time()
        self._emergency_mode = False

        # Register emergency callbacks
        self.monitor.register_callback("critical", self._emergency_cleanup)
        self.monitor.register_callback("warning", self._proactive_cleanup)

        # Start monitoring
        self.monitor.start_monitoring()

    @contextmanager
    def memory_managed_processing(self, estimated_memory_gb: float = 0.0):
        """Context manager for memory-managed processing."""
        profile_start = self.monitor.get_current_profile()

        try:
            # Pre-processing memory management
            self._prepare_memory_for_processing(estimated_memory_gb)

            yield self

        finally:
            # Post-processing cleanup
            self._cleanup_after_processing()

            profile_end = self.monitor.get_current_profile()

            # Log memory usage
            memory_delta = profile_end.process_memory - profile_start.process_memory
            if memory_delta > 0.1:  # > 100MB
                logging.info(f"Memory delta: {memory_delta:.2f} GB")

    def process_with_memory_awareness(
        self, processing_func: Callable, data: List[Any], **kwargs
    ) -> Any:
        """Process data with full memory awareness."""
        profile = self.monitor.get_current_profile()

        # Determine processing approach based on memory strategy
        if self.strategy == MemoryStrategy.CONSERVATIVE:
            return self._conservative_processing(processing_func, data, **kwargs)
        elif self.strategy == MemoryStrategy.BALANCED:
            return self._balanced_processing(processing_func, data, **kwargs)
        elif self.strategy == MemoryStrategy.AGGRESSIVE:
            return self._aggressive_processing(processing_func, data, **kwargs)
        elif self.strategy == MemoryStrategy.ADAPTIVE:
            return self._adaptive_processing(processing_func, data, profile, **kwargs)
        else:
            return processing_func(data, **kwargs)

    def _conservative_processing(
        self, processing_func: Callable, data: List[Any], **kwargs
    ):
        """Conservative memory processing - minimal memory usage."""
        results = []

        # Process one by one to minimize memory
        for item in data:
            with self.memory_managed_processing(0.1):
                result = processing_func([item], **kwargs)
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)

                # Aggressive cleanup after each item
                self._force_garbage_collection()

        return results

    def _balanced_processing(
        self, processing_func: Callable, data: List[Any], **kwargs
    ):
        """Balanced processing - moderate memory usage."""
        results = []
        batch_size = self.batcher.calculate_optimal_batch_size(data)

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with self.memory_managed_processing(0.5):
                batch_results = processing_func(batch, **kwargs)
                if isinstance(batch_results, list):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)

                # Regular cleanup
                if i % (batch_size * 5) == 0:  # Every 5 batches
                    self._proactive_cleanup()

        return results

    def _aggressive_processing(
        self, processing_func: Callable, data: List[Any], **kwargs
    ):
        """Aggressive processing - maximum speed, higher memory usage."""
        # Use larger batches for speed
        profile = self.monitor.get_current_profile()
        max_batch = min(len(data), self.batcher.max_batch_size)

        # Adjust based on available memory
        if profile.system_available < 2.0:  # Less than 2GB available
            max_batch = max_batch // 2

        results = []

        for i in range(0, len(data), max_batch):
            batch = data[i : i + max_batch]

            with self.memory_managed_processing(1.0):
                batch_results = processing_func(batch, **kwargs)
                if isinstance(batch_results, list):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)

        return results

    def _adaptive_processing(
        self,
        processing_func: Callable,
        data: List[Any],
        profile: MemoryProfile,
        **kwargs,
    ):
        """Adaptive processing based on current memory conditions."""
        # Decide strategy based on memory availability
        system_usage = 1 - (profile.system_available / profile.system_total)

        if system_usage > 0.8:  # High memory usage
            return self._conservative_processing(processing_func, data, **kwargs)
        elif system_usage > 0.6:  # Moderate memory usage
            return self._balanced_processing(processing_func, data, **kwargs)
        else:  # Low memory usage
            return self._aggressive_processing(processing_func, data, **kwargs)

    def _prepare_memory_for_processing(self, estimated_memory_gb: float):
        """Prepare memory for processing."""
        profile = self.monitor.get_current_profile()

        # Check if we have enough memory
        if profile.system_available < estimated_memory_gb * 1.2:  # 20% buffer
            self._proactive_cleanup()

            # Check again
            profile = self.monitor.get_current_profile()
            if profile.system_available < estimated_memory_gb:
                warnings.warn("Insufficient memory for optimal processing")

    def _cleanup_after_processing(self):
        """Cleanup after processing."""
        if time.time() - self._last_gc_time > 30:  # 30 seconds
            self._force_garbage_collection()

    def _proactive_cleanup(self, profile: Optional[MemoryProfile] = None):
        """Proactive memory cleanup."""
        # Clear caches if they're too large
        if self._cache_size_bytes > self.cache_size_gb * 1024**3:
            self._clear_memory_cache()

        # Force garbage collection
        if time.time() - self._last_gc_time > 10:  # 10 seconds
            self._force_garbage_collection()

        # GPU cleanup
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _emergency_cleanup(self, profile: MemoryProfile):
        """Emergency memory cleanup."""
        self._emergency_mode = True

        logging.warning(
            f"Emergency memory cleanup triggered. Available: {profile.system_available:.2f} GB"
        )

        # Aggressive cleanup
        self._clear_memory_cache()
        self._force_garbage_collection()

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Force GPU memory cleanup
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        # Reduce batch sizes
        self.batcher.current_batch_size = max(
            self.batcher.min_batch_size, self.batcher.current_batch_size // 2
        )

        self._emergency_mode = False

    def _force_garbage_collection(self):
        """Force garbage collection."""
        gc.collect()
        self._last_gc_time = time.time()

    def _clear_memory_cache(self):
        """Clear memory caches."""
        self._memory_cache.clear()
        self._cache_size_bytes = 0

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        profile = self.monitor.get_current_profile()

        return {
            "current_profile": profile,
            "strategy": self.strategy.value,
            "cache_size_gb": self._cache_size_bytes / (1024**3),
            "emergency_mode": self._emergency_mode,
            "batch_size": self.batcher.current_batch_size,
            "gc_frequency": 1
            / max(1, time.time() - self._last_gc_time)
            * 60,  # per minute
        }

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "monitor"):
            self.monitor.stop_monitoring()

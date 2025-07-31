"""
CUDA Kernel Optimization Utilities.

This module provides utilities for optimizing CUDA kernel performance including:
- Memory access pattern analysis
- Occupancy optimization
- Resource utilization profiling
- Performance regression testing
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import cupy as cp
    import numpy as np

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    np = None

from ...utils.logging import get_logger
from .memory import get_memory_manager

logger = get_logger()


@dataclass
class KernelPerformanceMetrics:
    """Performance metrics for a CUDA kernel."""

    kernel_name: str
    execution_time_ms: float
    throughput_elements_per_sec: float
    memory_bandwidth_gb_per_sec: float
    occupancy_percentage: float
    block_size: int
    grid_size: int
    shared_memory_bytes: int
    register_count: int
    memory_transactions: int
    cache_hit_rate: float


@dataclass
class OptimizationResult:
    """Result of kernel optimization."""

    original_metrics: KernelPerformanceMetrics
    optimized_metrics: KernelPerformanceMetrics
    speedup_factor: float
    optimization_techniques: List[str]
    recommendations: List[str]


class MemoryAccessAnalyzer:
    """Analyzes memory access patterns for CUDA kernels."""

    def __init__(self):
        """Initialize memory access analyzer."""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

    def analyze_coalescing(
        self, access_pattern: cp.ndarray, warp_size: int = 32
    ) -> Dict[str, float]:
        """Analyze memory coalescing efficiency.

        Args:
            access_pattern: Array of memory addresses accessed by threads
            warp_size: CUDA warp size

        Returns:
            Dictionary with coalescing metrics
        """
        # Reshape to analyze per-warp access patterns
        num_warps = len(access_pattern) // warp_size
        if num_warps == 0:
            return {"coalescing_efficiency": 0.0, "transactions_per_warp": 0.0}

        warp_accesses = access_pattern[: num_warps * warp_size].reshape(
            num_warps, warp_size
        )

        total_transactions = 0
        coalesced_transactions = 0

        for warp_idx in range(num_warps):
            warp_addresses = warp_accesses[warp_idx]

            # Sort addresses to find contiguous segments
            sorted_addresses = cp.sort(warp_addresses)

            # Count cache line transactions (assuming 128-byte cache lines)
            cache_line_size = 128
            cache_lines = set()

            for addr in sorted_addresses:
                cache_line = int(addr) // cache_line_size
                cache_lines.add(cache_line)

            transactions = len(cache_lines)
            total_transactions += transactions

            # Ideal case: all threads access same cache line
            if transactions == 1:
                coalesced_transactions += 1

        coalescing_efficiency = (
            coalesced_transactions / num_warps if num_warps > 0 else 0.0
        )
        avg_transactions_per_warp = (
            total_transactions / num_warps if num_warps > 0 else 0.0
        )

        return {
            "coalescing_efficiency": coalescing_efficiency,
            "transactions_per_warp": avg_transactions_per_warp,
            "ideal_transactions_per_warp": 1.0,
            "memory_efficiency": (
                1.0 / avg_transactions_per_warp
                if avg_transactions_per_warp > 0
                else 0.0
            ),
        }

    def analyze_bank_conflicts(
        self, shared_memory_accesses: cp.ndarray, num_banks: int = 32
    ) -> Dict[str, float]:
        """Analyze shared memory bank conflicts.

        Args:
            shared_memory_accesses: Array of shared memory addresses
            num_banks: Number of shared memory banks

        Returns:
            Dictionary with bank conflict metrics
        """
        # Group accesses by bank
        bank_accesses = {}
        for addr in shared_memory_accesses:
            bank = int(addr) % num_banks
            if bank not in bank_accesses:
                bank_accesses[bank] = 0
            bank_accesses[bank] += 1

        # Calculate conflicts
        max_conflicts = max(bank_accesses.values()) if bank_accesses else 0
        total_accesses = len(shared_memory_accesses)
        conflict_free_accesses = len(bank_accesses)

        conflict_ratio = (
            (total_accesses - conflict_free_accesses) / total_accesses
            if total_accesses > 0
            else 0.0
        )

        return {
            "bank_conflicts": max_conflicts - 1 if max_conflicts > 1 else 0,
            "conflict_ratio": conflict_ratio,
            "banks_used": len(bank_accesses),
            "max_conflicts_per_bank": max_conflicts,
            "efficiency": 1.0 - conflict_ratio,
        }


class OccupancyOptimizer:
    """Optimizes CUDA kernel occupancy."""

    def __init__(self):
        """Initialize occupancy optimizer."""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        # Get device properties
        device = cp.cuda.Device()
        self.device_props = device.attributes
        self.max_threads_per_block = self.device_props["MaxThreadsPerBlock"]
        self.max_shared_memory_per_block = self.device_props["SharedMemoryPerBlock"]
        self.max_registers_per_block = self.device_props.get(
            "MaxRegistersPerBlock", 65536
        )
        self.warp_size = self.device_props["WarpSize"]
        self.multiprocessor_count = self.device_props["MultiProcessorCount"]

    def calculate_theoretical_occupancy(
        self, block_size: int, shared_memory_per_block: int, registers_per_thread: int
    ) -> float:
        """Calculate theoretical occupancy for given kernel configuration.

        Args:
            block_size: Number of threads per block
            shared_memory_per_block: Shared memory usage per block
            registers_per_thread: Register usage per thread

        Returns:
            Theoretical occupancy (0.0 to 1.0)
        """
        # Occupancy limited by threads per block
        max_blocks_by_threads = self.max_threads_per_block // block_size

        # Occupancy limited by shared memory
        max_blocks_by_shared_mem = (
            self.max_shared_memory_per_block // shared_memory_per_block
            if shared_memory_per_block > 0
            else float("inf")
        )

        # Occupancy limited by registers
        registers_per_block = registers_per_thread * block_size
        max_blocks_by_registers = (
            self.max_registers_per_block // registers_per_block
            if registers_per_block > 0
            else float("inf")
        )

        # Actual occupancy is limited by the most restrictive resource
        max_blocks_per_sm = min(
            max_blocks_by_threads, max_blocks_by_shared_mem, max_blocks_by_registers
        )

        # Theoretical maximum blocks per SM (device-specific)
        theoretical_max_blocks = 32  # Common value for modern GPUs

        occupancy = min(1.0, max_blocks_per_sm / theoretical_max_blocks)

        return occupancy

    def find_optimal_block_size(
        self,
        shared_memory_per_block: int,
        registers_per_thread: int,
        min_block_size: Optional[int] = None,
        max_block_size: Optional[int] = None,
    ) -> Tuple[int, float]:
        """Find optimal block size for maximum occupancy.

        Args:
            shared_memory_per_block: Shared memory usage per block
            registers_per_thread: Register usage per thread
            min_block_size: Minimum block size to consider
            max_block_size: Maximum block size to consider

        Returns:
            Tuple of (optimal_block_size, occupancy)
        """
        if min_block_size is None:
            min_block_size = self.warp_size
        if max_block_size is None:
            max_block_size = self.max_threads_per_block

        best_block_size = min_block_size
        best_occupancy = 0.0

        # Test block sizes in multiples of warp size
        for block_size in range(min_block_size, max_block_size + 1, self.warp_size):
            occupancy = self.calculate_theoretical_occupancy(
                block_size, shared_memory_per_block, registers_per_thread
            )

            if occupancy > best_occupancy:
                best_occupancy = occupancy
                best_block_size = block_size

        return best_block_size, best_occupancy

    def optimize_shared_memory_usage(
        self, current_usage: int, block_size: int
    ) -> Dict[str, Any]:
        """Provide recommendations for optimizing shared memory usage.

        Args:
            current_usage: Current shared memory usage in bytes
            block_size: Current block size

        Returns:
            Dictionary with optimization recommendations
        """
        max_blocks_current = (
            self.max_shared_memory_per_block // current_usage
            if current_usage > 0
            else float("inf")
        )

        recommendations = []

        # Check if shared memory is the limiting factor
        max_blocks_by_threads = self.max_threads_per_block // block_size

        if max_blocks_current < max_blocks_by_threads:
            recommendations.append("Shared memory is limiting occupancy")

            # Suggest optimizations
            if current_usage > self.max_shared_memory_per_block // 2:
                recommendations.append(
                    "Consider reducing shared memory usage or using multiple kernel launches"
                )

            # Suggest bank conflict optimization
            recommendations.append(
                "Optimize shared memory access patterns to avoid bank conflicts"
            )

            # Suggest data layout optimizations
            recommendations.append(
                "Consider padding or restructuring shared memory data layout"
            )

        return {
            "current_usage_bytes": current_usage,
            "max_usage_bytes": self.max_shared_memory_per_block,
            "utilization_percentage": (current_usage / self.max_shared_memory_per_block)
            * 100,
            "max_blocks_by_shared_mem": max_blocks_current,
            "recommendations": recommendations,
        }


class KernelProfiler:
    """Comprehensive CUDA kernel profiler."""

    def __init__(self, memory_manager=None):
        """Initialize kernel profiler."""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self.memory_manager = memory_manager or get_memory_manager()
        self.memory_analyzer = MemoryAccessAnalyzer()
        self.occupancy_optimizer = OccupancyOptimizer()
        self.profile_history = []

    @contextmanager
    def profile_context(self, kernel_name: str):
        """Context manager for profiling kernel execution."""
        # Create CUDA events for timing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        # Record memory state before execution
        mem_info_before = self.memory_manager.get_memory_info()

        start_event.record()
        start_time = time.time()

        try:
            yield
        finally:
            end_event.record()
            end_event.synchronize()

            end_time = time.time()

            # Calculate metrics
            gpu_time = cp.cuda.get_elapsed_time(start_event, end_event)
            cpu_time = (end_time - start_time) * 1000  # Convert to ms

            mem_info_after = self.memory_manager.get_memory_info()
            memory_used = mem_info_after.used - mem_info_before.used

            # Store profiling result
            profile_result = {
                "kernel_name": kernel_name,
                "gpu_time_ms": gpu_time,
                "cpu_time_ms": cpu_time,
                "memory_used_bytes": memory_used,
                "timestamp": time.time(),
            }

            self.profile_history.append(profile_result)

            logger.debug(
                f"Profiled {kernel_name}: {gpu_time:.2f}ms GPU, {cpu_time:.2f}ms CPU"
            )

    def profile_kernel_execution(
        self,
        kernel_func: Callable,
        kernel_args: Tuple,
        kernel_kwargs: Dict,
        data_size: int,
        iterations: int = 10,
    ) -> KernelPerformanceMetrics:
        """Profile kernel execution with multiple iterations.

        Args:
            kernel_func: Kernel function to profile
            kernel_args: Kernel arguments
            kernel_kwargs: Kernel keyword arguments
            data_size: Size of data being processed
            iterations: Number of iterations for averaging

        Returns:
            Performance metrics
        """
        execution_times = []

        # Warm up
        for _ in range(3):
            kernel_func(*kernel_args, **kernel_kwargs)

        cp.cuda.Device().synchronize()

        # Profile multiple iterations
        for i in range(iterations):
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()

            start_event.record()
            kernel_func(*kernel_args, **kernel_kwargs)
            end_event.record()
            end_event.synchronize()

            execution_time = cp.cuda.get_elapsed_time(start_event, end_event)
            execution_times.append(execution_time)

        # Calculate statistics
        avg_time = sum(execution_times) / len(execution_times)
        throughput = data_size / (avg_time / 1000.0)  # Elements per second

        # Estimate memory bandwidth (simplified)
        bytes_per_element = 4  # Assume float32
        memory_bandwidth = (
            (data_size * bytes_per_element * 2) / (avg_time / 1000.0) / (1024**3)
        )  # GB/s

        # Get kernel configuration (simplified - would need actual kernel inspection)
        block_size = kernel_kwargs.get("block_size", 256)
        grid_size = kernel_kwargs.get(
            "grid_size", (data_size + block_size - 1) // block_size
        )

        return KernelPerformanceMetrics(
            kernel_name=(
                kernel_func.__name__ if hasattr(kernel_func, "__name__") else "unknown"
            ),
            execution_time_ms=avg_time,
            throughput_elements_per_sec=throughput,
            memory_bandwidth_gb_per_sec=memory_bandwidth,
            occupancy_percentage=75.0,  # Would need actual occupancy calculation
            block_size=block_size,
            grid_size=grid_size,
            shared_memory_bytes=kernel_kwargs.get("shared_memory", 0),
            register_count=32,  # Would need actual register count
            memory_transactions=data_size // 32,  # Simplified estimate
            cache_hit_rate=0.85,  # Simplified estimate
        )

    def compare_kernel_versions(
        self,
        original_kernel: Callable,
        optimized_kernel: Callable,
        kernel_args: Tuple,
        data_size: int,
    ) -> OptimizationResult:
        """Compare performance between original and optimized kernel versions.

        Args:
            original_kernel: Original kernel function
            optimized_kernel: Optimized kernel function
            kernel_args: Kernel arguments
            data_size: Size of data being processed

        Returns:
            Optimization result with comparison metrics
        """
        # Profile original kernel
        original_metrics = self.profile_kernel_execution(
            original_kernel, kernel_args, {}, data_size
        )

        # Profile optimized kernel
        optimized_metrics = self.profile_kernel_execution(
            optimized_kernel, kernel_args, {}, data_size
        )

        # Calculate speedup
        speedup = (
            original_metrics.execution_time_ms / optimized_metrics.execution_time_ms
        )

        # Generate optimization techniques used (would be provided by caller)
        optimization_techniques = [
            "Coalesced memory access",
            "Shared memory optimization",
            "Warp-level primitives",
        ]

        # Generate recommendations
        recommendations = []
        if speedup < 1.2:
            recommendations.append(
                "Consider additional optimizations - speedup is minimal"
            )
        if optimized_metrics.occupancy_percentage < 50:
            recommendations.append("Low occupancy - consider increasing block size")
        if optimized_metrics.memory_bandwidth_gb_per_sec < 100:
            recommendations.append("Low memory bandwidth utilization")

        return OptimizationResult(
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            speedup_factor=speedup,
            optimization_techniques=optimization_techniques,
            recommendations=recommendations,
        )

    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report."""
        if not self.profile_history:
            return "No profiling data available."

        report = ["CUDA Kernel Optimization Report", "=" * 40, ""]

        # Summary statistics
        total_kernels = len(set(p["kernel_name"] for p in self.profile_history))
        total_executions = len(self.profile_history)
        avg_gpu_time = (
            sum(p["gpu_time_ms"] for p in self.profile_history) / total_executions
        )

        report.extend(
            [
                f"Total kernels profiled: {total_kernels}",
                f"Total executions: {total_executions}",
                f"Average GPU execution time: {avg_gpu_time:.2f} ms",
                "",
            ]
        )

        # Per-kernel analysis
        kernel_stats = {}
        for profile in self.profile_history:
            kernel_name = profile["kernel_name"]
            if kernel_name not in kernel_stats:
                kernel_stats[kernel_name] = []
            kernel_stats[kernel_name].append(profile)

        for kernel_name, profiles in kernel_stats.items():
            report.append(f"Kernel: {kernel_name}")
            report.append("-" * 20)

            gpu_times = [p["gpu_time_ms"] for p in profiles]
            avg_time = sum(gpu_times) / len(gpu_times)
            min_time = min(gpu_times)
            max_time = max(gpu_times)

            report.extend(
                [
                    f"  Executions: {len(profiles)}",
                    f"  Average time: {avg_time:.2f} ms",
                    f"  Min time: {min_time:.2f} ms",
                    f"  Max time: {max_time:.2f} ms",
                    f"  Variance: {max_time - min_time:.2f} ms",
                    "",
                ]
            )

        return "\n".join(report)

    def save_profile_data(self, filename: str) -> None:
        """Save profiling data to JSON file."""
        with open(filename, "w") as f:
            json.dump(self.profile_history, f, indent=2)

        logger.info(f"Saved profiling data to {filename}")

    def load_profile_data(self, filename: str) -> None:
        """Load profiling data from JSON file."""
        with open(filename) as f:
            self.profile_history = json.load(f)

        logger.info(f"Loaded profiling data from {filename}")


class PerformanceRegressionTester:
    """Tests for performance regressions in CUDA kernels."""

    def __init__(self, profiler: KernelProfiler):
        """Initialize regression tester."""
        self.profiler = profiler
        self.baseline_metrics = {}

    def set_baseline(self, kernel_name: str, metrics: KernelPerformanceMetrics) -> None:
        """Set baseline performance metrics for a kernel."""
        self.baseline_metrics[kernel_name] = metrics
        logger.info(
            f"Set baseline for {kernel_name}: {metrics.execution_time_ms:.2f}ms"
        )

    def test_regression(
        self,
        kernel_name: str,
        current_metrics: KernelPerformanceMetrics,
        tolerance: float = 0.1,
    ) -> Dict[str, Any]:
        """Test for performance regression.

        Args:
            kernel_name: Name of kernel to test
            current_metrics: Current performance metrics
            tolerance: Acceptable performance degradation (0.1 = 10%)

        Returns:
            Dictionary with regression test results
        """
        if kernel_name not in self.baseline_metrics:
            return {
                "status": "no_baseline",
                "message": f"No baseline metrics available for {kernel_name}",
            }

        baseline = self.baseline_metrics[kernel_name]

        # Calculate performance change
        time_change = (
            current_metrics.execution_time_ms - baseline.execution_time_ms
        ) / baseline.execution_time_ms
        throughput_change = (
            current_metrics.throughput_elements_per_sec
            - baseline.throughput_elements_per_sec
        ) / baseline.throughput_elements_per_sec

        # Check for regression
        regression_detected = time_change > tolerance or throughput_change < -tolerance

        result = {
            "status": "regression" if regression_detected else "pass",
            "time_change_percent": time_change * 100,
            "throughput_change_percent": throughput_change * 100,
            "baseline_time_ms": baseline.execution_time_ms,
            "current_time_ms": current_metrics.execution_time_ms,
            "baseline_throughput": baseline.throughput_elements_per_sec,
            "current_throughput": current_metrics.throughput_elements_per_sec,
            "tolerance_percent": tolerance * 100,
        }

        if regression_detected:
            result["message"] = f"Performance regression detected in {kernel_name}"
            logger.warning(result["message"])
        else:
            result["message"] = f"No regression detected in {kernel_name}"
            logger.info(result["message"])

        return result

    def run_regression_suite(
        self,
        kernels_to_test: List[Tuple[str, Callable, Tuple, int]],
        tolerance: float = 0.1,
    ) -> Dict[str, Dict[str, Any]]:
        """Run regression tests on multiple kernels.

        Args:
            kernels_to_test: List of (kernel_name, kernel_func, args, data_size) tuples
            tolerance: Acceptable performance degradation

        Returns:
            Dictionary with results for each kernel
        """
        results = {}

        for kernel_name, kernel_func, kernel_args, data_size in kernels_to_test:
            try:
                # Profile current performance
                current_metrics = self.profiler.profile_kernel_execution(
                    kernel_func, kernel_args, {}, data_size
                )

                # Test for regression
                regression_result = self.test_regression(
                    kernel_name, current_metrics, tolerance
                )
                results[kernel_name] = regression_result

            except Exception as e:
                results[kernel_name] = {
                    "status": "error",
                    "message": f"Error testing {kernel_name}: {str(e)}",
                }
                logger.error(f"Error in regression test for {kernel_name}: {e}")

        return results


# Global profiler instance
_kernel_profiler: Optional[KernelProfiler] = None


def get_kernel_profiler() -> KernelProfiler:
    """Get global kernel profiler instance."""
    global _kernel_profiler
    if _kernel_profiler is None:
        _kernel_profiler = KernelProfiler()
    return _kernel_profiler

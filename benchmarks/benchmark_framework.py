"""
Core Benchmarking Framework for ChemiAccel.

This module provides the main benchmarking infrastructure for comparing
ChemiAccel performance against reference implementations like RDKit and Mordred.
"""

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import numpy as np
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

try:
    from ..src.chemi_accel.utils.logging import get_logger
except ImportError:
    # Fallback for standalone execution
    def get_logger():
        return logging.getLogger(__name__)


logger = get_logger()


class BenchmarkStatus(Enum):
    """Status of benchmark execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    name: str
    description: str
    dataset_sizes: List[int] = field(default_factory=lambda: [100, 1000, 10000])
    iterations: int = 5
    warmup_iterations: int = 2
    timeout_seconds: int = 300
    memory_limit_gb: Optional[float] = None
    backends_to_test: List[str] = field(default_factory=lambda: ["cuda", "jax", "cpu"])
    descriptors_to_test: List[str] = field(
        default_factory=lambda: ["molecular_weight", "tpsa", "logp"]
    )
    reference_implementations: List[str] = field(
        default_factory=lambda: ["rdkit", "mordred"]
    )
    statistical_significance_level: float = 0.05
    min_speedup_threshold: float = 2.0
    output_directory: str = "benchmark_results"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    config_name: str
    implementation: str
    backend: str
    descriptor: str
    dataset_size: int
    execution_times: List[float]
    memory_usage_mb: float
    status: BenchmarkStatus
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_time(self) -> float:
        """Mean execution time."""
        return statistics.mean(self.execution_times) if self.execution_times else 0.0

    @property
    def std_time(self) -> float:
        """Standard deviation of execution times."""
        return (
            statistics.stdev(self.execution_times)
            if len(self.execution_times) > 1
            else 0.0
        )

    @property
    def median_time(self) -> float:
        """Median execution time."""
        return statistics.median(self.execution_times) if self.execution_times else 0.0

    @property
    def throughput(self) -> float:
        """Throughput in molecules per second."""
        return self.dataset_size / self.mean_time if self.mean_time > 0 else 0.0


@dataclass
class PerformanceComparison:
    """Comparison between ChemiAccel and reference implementation."""

    chemi_accel_result: BenchmarkResult
    reference_result: BenchmarkResult
    speedup_factor: float
    memory_ratio: float
    statistical_significance: float
    is_significant_improvement: bool

    @property
    def performance_improvement_percent(self) -> float:
        """Performance improvement as percentage."""
        return (self.speedup_factor - 1.0) * 100


class BenchmarkSuite:
    """Main benchmarking suite for ChemiAccel."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark suite.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[PerformanceComparison] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized benchmark suite: {config.name}")

    def run_benchmark(
        self,
        implementation_func: Callable,
        dataset: Any,
        implementation_name: str,
        backend: str,
        descriptor: str,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            implementation_func: Function to benchmark
            dataset: Dataset to process
            implementation_name: Name of implementation (e.g., "chemi_accel", "rdkit")
            backend: Backend name (e.g., "cuda", "cpu")
            descriptor: Descriptor name

        Returns:
            Benchmark result
        """
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else 1000

        result = BenchmarkResult(
            config_name=self.config.name,
            implementation=implementation_name,
            backend=backend,
            descriptor=descriptor,
            dataset_size=dataset_size,
            execution_times=[],
            memory_usage_mb=0.0,
            status=BenchmarkStatus.RUNNING,
        )

        try:
            # Warmup runs
            logger.debug(f"Warming up {implementation_name} {descriptor} on {backend}")
            for _ in range(self.config.warmup_iterations):
                try:
                    _ = implementation_func(dataset)
                except Exception as e:
                    logger.warning(f"Warmup failed: {e}")
                    break

            # Benchmark runs
            execution_times = []
            memory_usage_samples = []

            for iteration in range(self.config.iterations):
                logger.debug(
                    f"Benchmark iteration {iteration + 1}/{self.config.iterations}"
                )

                # Measure memory before
                memory_before = self._get_memory_usage()

                # Time execution
                start_time = time.perf_counter()

                try:
                    output = implementation_func(dataset)

                    end_time = time.perf_counter()
                    execution_time = end_time - start_time

                    # Measure memory after
                    memory_after = self._get_memory_usage()
                    memory_used = memory_after - memory_before

                    execution_times.append(execution_time)
                    memory_usage_samples.append(memory_used)

                    # Validate output (basic check)
                    if output is None:
                        logger.warning(
                            f"Implementation returned None for iteration {iteration}"
                        )

                except Exception as e:
                    logger.error(f"Benchmark iteration {iteration} failed: {e}")
                    result.status = BenchmarkStatus.FAILED
                    result.error_message = str(e)
                    return result

            # Calculate statistics
            result.execution_times = execution_times
            result.memory_usage_mb = (
                statistics.mean(memory_usage_samples) if memory_usage_samples else 0.0
            )
            result.status = BenchmarkStatus.COMPLETED

            logger.info(
                f"Completed benchmark: {implementation_name} {descriptor} - "
                f"{result.mean_time:.3f}s ± {result.std_time:.3f}s"
            )

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)

        return result

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def compare_implementations(
        self, chemi_accel_result: BenchmarkResult, reference_result: BenchmarkResult
    ) -> PerformanceComparison:
        """Compare ChemiAccel result with reference implementation.

        Args:
            chemi_accel_result: ChemiAccel benchmark result
            reference_result: Reference implementation result

        Returns:
            Performance comparison
        """
        # Calculate speedup
        speedup = (
            reference_result.mean_time / chemi_accel_result.mean_time
            if chemi_accel_result.mean_time > 0
            else 0.0
        )

        # Calculate memory ratio
        memory_ratio = (
            chemi_accel_result.memory_usage_mb / reference_result.memory_usage_mb
            if reference_result.memory_usage_mb > 0
            else 1.0
        )

        # Statistical significance test (simplified t-test)
        significance = self._calculate_statistical_significance(
            chemi_accel_result.execution_times, reference_result.execution_times
        )

        is_significant = (
            significance < self.config.statistical_significance_level
            and speedup >= self.config.min_speedup_threshold
        )

        comparison = PerformanceComparison(
            chemi_accel_result=chemi_accel_result,
            reference_result=reference_result,
            speedup_factor=speedup,
            memory_ratio=memory_ratio,
            statistical_significance=significance,
            is_significant_improvement=is_significant,
        )

        self.comparisons.append(comparison)

        logger.info(
            f"Performance comparison: {speedup:.2f}x speedup, "
            f"memory ratio: {memory_ratio:.2f}, "
            f"significant: {is_significant}"
        )

        return comparison

    def _calculate_statistical_significance(
        self, sample1: List[float], sample2: List[float]
    ) -> float:
        """Calculate statistical significance using t-test.

        Args:
            sample1: First sample (ChemiAccel times)
            sample2: Second sample (reference times)

        Returns:
            P-value
        """
        try:
            from scipy import stats

            if len(sample1) < 2 or len(sample2) < 2:
                return 1.0  # Not significant

            # Perform two-sample t-test
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
            return p_value

        except ImportError:
            # Fallback to simple comparison if scipy not available
            mean1 = statistics.mean(sample1)
            mean2 = statistics.mean(sample2)

            # Simple heuristic: if means differ by more than 2 standard deviations
            std1 = statistics.stdev(sample1) if len(sample1) > 1 else 0.0
            std2 = statistics.stdev(sample2) if len(sample2) > 1 else 0.0
            combined_std = (std1 + std2) / 2

            if combined_std == 0:
                return 0.0 if mean1 != mean2 else 1.0

            z_score = abs(mean1 - mean2) / combined_std
            return 0.01 if z_score > 2.0 else 0.1  # Rough approximation

    def run_full_suite(
        self,
        chemi_accel_implementations: Dict[str, Callable],
        reference_implementations: Dict[str, Callable],
        datasets: Dict[int, Any],
    ) -> None:
        """Run the complete benchmark suite.

        Args:
            chemi_accel_implementations: Dict of ChemiAccel implementations by backend
            reference_implementations: Dict of reference implementations
            datasets: Dict of datasets by size
        """
        self.start_time = time.time()

        logger.info(f"Starting benchmark suite: {self.config.name}")

        # Run ChemiAccel benchmarks
        chemi_accel_results = {}

        for backend in self.config.backends_to_test:
            if backend not in chemi_accel_implementations:
                logger.warning(f"Backend {backend} not available, skipping")
                continue

            for descriptor in self.config.descriptors_to_test:
                for dataset_size in self.config.dataset_sizes:
                    if dataset_size not in datasets:
                        logger.warning(
                            f"Dataset size {dataset_size} not available, skipping"
                        )
                        continue

                    logger.info(
                        f"Benchmarking ChemiAccel {backend} {descriptor} size {dataset_size}"
                    )

                    result = self.run_benchmark(
                        chemi_accel_implementations[backend],
                        datasets[dataset_size],
                        "chemi_accel",
                        backend,
                        descriptor,
                    )

                    self.results.append(result)
                    chemi_accel_results[(backend, descriptor, dataset_size)] = result

        # Run reference implementation benchmarks
        reference_results = {}

        for ref_impl in self.config.reference_implementations:
            if ref_impl not in reference_implementations:
                logger.warning(
                    f"Reference implementation {ref_impl} not available, skipping"
                )
                continue

            for descriptor in self.config.descriptors_to_test:
                for dataset_size in self.config.dataset_sizes:
                    if dataset_size not in datasets:
                        continue

                    logger.info(
                        f"Benchmarking {ref_impl} {descriptor} size {dataset_size}"
                    )

                    result = self.run_benchmark(
                        reference_implementations[ref_impl],
                        datasets[dataset_size],
                        ref_impl,
                        "cpu",  # Reference implementations typically use CPU
                        descriptor,
                    )

                    self.results.append(result)
                    reference_results[(ref_impl, descriptor, dataset_size)] = result

        # Generate comparisons
        for backend in self.config.backends_to_test:
            for descriptor in self.config.descriptors_to_test:
                for dataset_size in self.config.dataset_sizes:
                    chemi_key = (backend, descriptor, dataset_size)

                    if chemi_key not in chemi_accel_results:
                        continue

                    chemi_result = chemi_accel_results[chemi_key]

                    for ref_impl in self.config.reference_implementations:
                        ref_key = (ref_impl, descriptor, dataset_size)

                        if ref_key not in reference_results:
                            continue

                        ref_result = reference_results[ref_key]

                        _ = self.compare_implementations(chemi_result, ref_result)

        self.end_time = time.time()

        logger.info(
            f"Completed benchmark suite in {self.end_time - self.start_time:.2f} seconds"
        )

        # Save results
        self.save_results()

    def save_results(self) -> None:
        """Save benchmark results to files."""
        # Save raw results as JSON
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "config_name": result.config_name,
                    "implementation": result.implementation,
                    "backend": result.backend,
                    "descriptor": result.descriptor,
                    "dataset_size": result.dataset_size,
                    "execution_times": result.execution_times,
                    "memory_usage_mb": result.memory_usage_mb,
                    "status": result.status.value,
                    "error_message": result.error_message,
                    "metadata": result.metadata,
                    "mean_time": result.mean_time,
                    "std_time": result.std_time,
                    "throughput": result.throughput,
                }
            )

        results_file = self.output_dir / f"{self.config.name}_results.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        # Save comparisons
        comparisons_data = []
        for comp in self.comparisons:
            comparisons_data.append(
                {
                    "chemi_accel_implementation": comp.chemi_accel_result.implementation,
                    "chemi_accel_backend": comp.chemi_accel_result.backend,
                    "reference_implementation": comp.reference_result.implementation,
                    "descriptor": comp.chemi_accel_result.descriptor,
                    "dataset_size": comp.chemi_accel_result.dataset_size,
                    "speedup_factor": comp.speedup_factor,
                    "memory_ratio": comp.memory_ratio,
                    "statistical_significance": comp.statistical_significance,
                    "is_significant_improvement": bool(comp.is_significant_improvement),
                    "performance_improvement_percent": comp.performance_improvement_percent,
                }
            )

        comparisons_file = self.output_dir / f"{self.config.name}_comparisons.json"
        with open(comparisons_file, "w") as f:
            json.dump(comparisons_data, f, indent=2)

        # Save configuration
        config_file = self.output_dir / f"{self.config.name}_config.json"
        with open(config_file, "w") as f:
            json.dump(
                {
                    "name": self.config.name,
                    "description": self.config.description,
                    "dataset_sizes": self.config.dataset_sizes,
                    "iterations": self.config.iterations,
                    "warmup_iterations": self.config.warmup_iterations,
                    "timeout_seconds": self.config.timeout_seconds,
                    "backends_to_test": self.config.backends_to_test,
                    "descriptors_to_test": self.config.descriptors_to_test,
                    "reference_implementations": self.config.reference_implementations,
                    "statistical_significance_level": self.config.statistical_significance_level,
                    "min_speedup_threshold": self.config.min_speedup_threshold,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved benchmark results to {self.output_dir}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the benchmark suite."""
        if not self.results:
            return {}

        # Group results by implementation
        by_implementation = {}
        for result in self.results:
            key = f"{result.implementation}_{result.backend}"
            if key not in by_implementation:
                by_implementation[key] = []
            by_implementation[key].append(result)

        summary = {
            "total_benchmarks": len(self.results),
            "successful_benchmarks": len(
                [r for r in self.results if r.status == BenchmarkStatus.COMPLETED]
            ),
            "failed_benchmarks": len(
                [r for r in self.results if r.status == BenchmarkStatus.FAILED]
            ),
            "total_comparisons": len(self.comparisons),
            "significant_improvements": len(
                [c for c in self.comparisons if c.is_significant_improvement]
            ),
            "execution_time_seconds": (
                self.end_time - self.start_time
                if self.end_time and self.start_time
                else 0
            ),
            "by_implementation": {},
        }

        for impl_key, impl_results in by_implementation.items():
            impl_summary = {
                "total_benchmarks": len(impl_results),
                "successful_benchmarks": len(
                    [r for r in impl_results if r.status == BenchmarkStatus.COMPLETED]
                ),
                "mean_execution_time": statistics.mean(
                    [
                        r.mean_time
                        for r in impl_results
                        if r.status == BenchmarkStatus.COMPLETED
                    ]
                ),
                "mean_throughput": statistics.mean(
                    [
                        r.throughput
                        for r in impl_results
                        if r.status == BenchmarkStatus.COMPLETED
                    ]
                ),
                "mean_memory_usage_mb": statistics.mean(
                    [
                        r.memory_usage_mb
                        for r in impl_results
                        if r.status == BenchmarkStatus.COMPLETED
                    ]
                ),
            }
            summary["by_implementation"][impl_key] = impl_summary

        return summary


def run_benchmark_suite(
    config: BenchmarkConfig,
    chemi_accel_implementations: Dict[str, Callable],
    reference_implementations: Dict[str, Callable],
    datasets: Dict[int, Any],
) -> BenchmarkSuite:
    """Run a complete benchmark suite.

    Args:
        config: Benchmark configuration
        chemi_accel_implementations: ChemiAccel implementations by backend
        reference_implementations: Reference implementations
        datasets: Datasets by size

    Returns:
        Completed benchmark suite
    """
    suite = BenchmarkSuite(config)
    suite.run_full_suite(
        chemi_accel_implementations, reference_implementations, datasets
    )
    return suite

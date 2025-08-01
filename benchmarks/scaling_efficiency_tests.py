"""
Scaling Efficiency Tests for Distributed Computing.

This module provides comprehensive testing infrastructure for evaluating
scaling efficiency across different distributed computing configurations,
including multi-node, multi-GPU, and federated learning scenarios.
"""

import json
import logging
import statistics
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import dask
    from dask.distributed import Client, LocalCluster

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ScalingType(Enum):
    """Types of scaling to test."""

    STRONG_SCALING = "strong_scaling"  # Fixed problem size, increasing resources
    WEAK_SCALING = "weak_scaling"  # Problem size scales with resources
    MIXED_SCALING = "mixed_scaling"  # Combination of strong and weak scaling


class DistributionType(Enum):
    """Types of distributed computing to test."""

    MULTI_PROCESS = "multi_process"  # Multi-process on single node
    MULTI_NODE = "multi_node"  # Multi-node cluster
    MULTI_GPU = "multi_gpu"  # Multi-GPU scaling
    FEDERATED = "federated"  # Federated learning
    HYBRID = "hybrid"  # Combination of above


@dataclass
class ScalingConfiguration:
    """Configuration for scaling efficiency tests."""

    scaling_type: ScalingType
    distribution_type: DistributionType
    resource_counts: List[int]  # Number of resources to test
    dataset_sizes: List[int]  # Dataset sizes for each test
    iterations: int = 3  # Number of test iterations
    timeout_seconds: int = 3600  # Test timeout
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingResult:
    """Result of scaling efficiency test."""

    test_name: str
    configuration: ScalingConfiguration
    resource_count: int
    dataset_size: int
    execution_time: float
    throughput: float
    efficiency: float
    speedup: float
    memory_usage_mb: float
    cpu_utilization: float
    network_usage_mbps: Optional[float] = None
    gpu_utilization: Optional[float] = None
    passed: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingAnalysis:
    """Analysis of scaling efficiency results."""

    configuration: ScalingConfiguration
    results: List[ScalingResult]
    ideal_speedups: List[float]
    actual_speedups: List[float]
    efficiencies: List[float]
    scalability_limit: Optional[int] = None
    optimal_resource_count: Optional[int] = None
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    overall_score: float = 0.0


class ResourceMonitor:
    """Monitor system resources during scaling tests."""

    def __init__(self, monitoring_interval: float = 1.0):
        """Initialize resource monitor.

        Args:
            monitoring_interval: Interval between measurements in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.active = False
        self.measurements = []
        self.monitor_thread = None

    @contextmanager
    def monitor_resources(self):
        """Context manager for resource monitoring."""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()

    def start_monitoring(self):
        """Start resource monitoring."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, resource monitoring disabled")
            return

        self.active = True
        self.measurements = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        if self.active:
            self.active = False
            if self.monitor_thread:
                self.monitor_thread.join()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.active:
            try:
                measurement = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_used_mb": psutil.virtual_memory().used / 1024**2,
                    "disk_io": psutil.disk_io_counters()._asdict()
                    if psutil.disk_io_counters()
                    else {},
                    "network_io": psutil.net_io_counters()._asdict()
                    if psutil.net_io_counters()
                    else {},
                }

                # Add GPU metrics if available
                if CUPY_AVAILABLE:
                    try:
                        measurement["gpu_memory_used_mb"] = (
                            cp.cuda.runtime.memGetInfo()[1] / 1024**2
                        )
                        measurement["gpu_utilization"] = self._get_gpu_utilization()
                    except Exception:
                        pass

                self.measurements.append(measurement)

            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")

            time.sleep(self.monitoring_interval)

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization (mock implementation)."""
        # In practice, this would use nvidia-ml-py or similar
        return np.random.uniform(0.7, 0.95)  # Mock GPU utilization

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics from measurements."""
        if not self.measurements:
            return {}

        stats = {}

        # CPU statistics
        cpu_values = [m["cpu_percent"] for m in self.measurements]
        stats["avg_cpu_percent"] = statistics.mean(cpu_values)
        stats["max_cpu_percent"] = max(cpu_values)

        # Memory statistics
        memory_values = [m["memory_used_mb"] for m in self.measurements]
        stats["avg_memory_mb"] = statistics.mean(memory_values)
        stats["max_memory_mb"] = max(memory_values)

        # Network statistics (if available)
        if self.measurements[0]["network_io"]:
            bytes_sent_values = [
                m["network_io"].get("bytes_sent", 0) for m in self.measurements
            ]
            bytes_recv_values = [
                m["network_io"].get("bytes_recv", 0) for m in self.measurements
            ]

            if len(bytes_sent_values) > 1:
                sent_rate = (bytes_sent_values[-1] - bytes_sent_values[0]) / len(
                    self.measurements
                )
                recv_rate = (bytes_recv_values[-1] - bytes_recv_values[0]) / len(
                    self.measurements
                )
                stats["network_sent_mbps"] = sent_rate / 1024**2
                stats["network_recv_mbps"] = recv_rate / 1024**2

        # GPU statistics (if available)
        gpu_memory_values = [m.get("gpu_memory_used_mb", 0) for m in self.measurements]
        gpu_util_values = [m.get("gpu_utilization", 0) for m in self.measurements]

        if any(gpu_memory_values):
            stats["avg_gpu_memory_mb"] = statistics.mean(
                [v for v in gpu_memory_values if v > 0]
            )
            stats["max_gpu_memory_mb"] = max(gpu_memory_values)

        if any(gpu_util_values):
            stats["avg_gpu_utilization"] = statistics.mean(
                [v for v in gpu_util_values if v > 0]
            )
            stats["max_gpu_utilization"] = max(gpu_util_values)

        return stats


class ScalingEfficiencyTester:
    """Comprehensive scaling efficiency testing framework."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize scaling efficiency tester.

        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = (
            Path(output_dir) if output_dir else Path("scaling_test_results")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[ScalingResult] = []
        self.analyses: List[ScalingAnalysis] = []

        # Test configurations
        self.standard_configs = self._create_standard_configurations()

    def _create_standard_configurations(self) -> List[ScalingConfiguration]:
        """Create standard scaling test configurations."""
        configs = []

        # Strong scaling tests
        configs.append(
            ScalingConfiguration(
                scaling_type=ScalingType.STRONG_SCALING,
                distribution_type=DistributionType.MULTI_PROCESS,
                resource_counts=[1, 2, 4, 8, 16],
                dataset_sizes=[100000] * 5,  # Fixed dataset size
                iterations=3,
                metadata={"description": "Multi-process strong scaling"},
            )
        )

        # Weak scaling tests
        configs.append(
            ScalingConfiguration(
                scaling_type=ScalingType.WEAK_SCALING,
                distribution_type=DistributionType.MULTI_PROCESS,
                resource_counts=[1, 2, 4, 8, 16],
                dataset_sizes=[10000, 20000, 40000, 80000, 160000],  # Scaling dataset
                iterations=3,
                metadata={"description": "Multi-process weak scaling"},
            )
        )

        # Multi-GPU scaling (if available)
        if CUPY_AVAILABLE:
            configs.append(
                ScalingConfiguration(
                    scaling_type=ScalingType.STRONG_SCALING,
                    distribution_type=DistributionType.MULTI_GPU,
                    resource_counts=[1, 2, 4, 8],
                    dataset_sizes=[50000] * 4,
                    iterations=3,
                    metadata={"description": "Multi-GPU strong scaling"},
                )
            )

        # Dask distributed scaling (if available)
        if DASK_AVAILABLE:
            configs.append(
                ScalingConfiguration(
                    scaling_type=ScalingType.STRONG_SCALING,
                    distribution_type=DistributionType.MULTI_NODE,
                    resource_counts=[1, 2, 4, 8],
                    dataset_sizes=[75000] * 4,
                    iterations=3,
                    metadata={"description": "Dask multi-node strong scaling"},
                )
            )

        return configs

    def run_scaling_test(
        self,
        test_function: Callable,
        configuration: ScalingConfiguration,
        test_name: str,
    ) -> ScalingAnalysis:
        """Run comprehensive scaling efficiency test.

        Args:
            test_function: Function to test (should accept dataset and resource_count)
            configuration: Scaling configuration
            test_name: Name of the test

        Returns:
            Scaling analysis results
        """
        logger.info(f"Starting scaling test: {test_name}")

        test_results = []
        baseline_time = None

        for i, (resource_count, dataset_size) in enumerate(
            zip(configuration.resource_counts, configuration.dataset_sizes)
        ):
            logger.info(f"Testing {resource_count} resources with {dataset_size} items")

            # Generate test dataset
            test_dataset = self._generate_test_dataset(dataset_size)

            # Run test iterations
            iteration_results = []

            for iteration in range(configuration.iterations):
                with ResourceMonitor() as monitor:
                    start_time = time.perf_counter()

                    try:
                        # Execute test function
                        if (
                            configuration.distribution_type
                            == DistributionType.MULTI_PROCESS
                        ):
                            result = self._run_multiprocess_test(
                                test_function, test_dataset, resource_count
                            )
                        elif (
                            configuration.distribution_type
                            == DistributionType.MULTI_GPU
                        ):
                            result = self._run_multigpu_test(
                                test_function, test_dataset, resource_count
                            )
                        elif (
                            configuration.distribution_type
                            == DistributionType.MULTI_NODE
                        ):
                            result = self._run_multinode_test(
                                test_function, test_dataset, resource_count
                            )
                        else:
                            result = test_function(test_dataset, resource_count)

                        end_time = time.perf_counter()
                        execution_time = end_time - start_time

                        # Get resource statistics
                        resource_stats = monitor.get_summary_stats()

                        iteration_results.append(
                            {
                                "execution_time": execution_time,
                                "throughput": dataset_size / execution_time
                                if execution_time > 0
                                else 0,
                                "resource_stats": resource_stats,
                                "result": result,
                            }
                        )

                    except Exception as e:
                        logger.error(f"Test iteration failed: {e}")
                        iteration_results.append(
                            {
                                "execution_time": float("inf"),
                                "throughput": 0,
                                "resource_stats": {},
                                "error": str(e),
                            }
                        )

            # Calculate average metrics
            valid_iterations = [r for r in iteration_results if "error" not in r]

            if valid_iterations:
                avg_time = statistics.mean(
                    [r["execution_time"] for r in valid_iterations]
                )
                avg_throughput = statistics.mean(
                    [r["throughput"] for r in valid_iterations]
                )

                # Calculate scaling metrics
                if baseline_time is None:
                    baseline_time = avg_time
                    speedup = 1.0
                    efficiency = 1.0
                else:
                    if configuration.scaling_type == ScalingType.STRONG_SCALING:
                        # Strong scaling: speedup = baseline_time / current_time
                        speedup = baseline_time / avg_time if avg_time > 0 else 0
                        efficiency = speedup / resource_count
                    else:
                        # Weak scaling: efficiency = baseline_time / current_time
                        efficiency = baseline_time / avg_time if avg_time > 0 else 0
                        speedup = efficiency * resource_count

                # Aggregate resource statistics
                all_stats = [
                    r["resource_stats"] for r in valid_iterations if r["resource_stats"]
                ]
                avg_stats = self._aggregate_resource_stats(all_stats)

                test_result = ScalingResult(
                    test_name=f"{test_name}_{resource_count}_{dataset_size}",
                    configuration=configuration,
                    resource_count=resource_count,
                    dataset_size=dataset_size,
                    execution_time=avg_time,
                    throughput=avg_throughput,
                    efficiency=efficiency,
                    speedup=speedup,
                    memory_usage_mb=avg_stats.get("max_memory_mb", 0),
                    cpu_utilization=avg_stats.get("avg_cpu_percent", 0),
                    network_usage_mbps=avg_stats.get("network_sent_mbps", 0),
                    gpu_utilization=avg_stats.get("avg_gpu_utilization"),
                    passed=True,
                    metadata={
                        "iterations": len(valid_iterations),
                        "baseline_time": baseline_time,
                        "iteration_details": iteration_results,
                    },
                )
            else:
                # All iterations failed
                test_result = ScalingResult(
                    test_name=f"{test_name}_{resource_count}_{dataset_size}",
                    configuration=configuration,
                    resource_count=resource_count,
                    dataset_size=dataset_size,
                    execution_time=float("inf"),
                    throughput=0,
                    efficiency=0,
                    speedup=0,
                    memory_usage_mb=0,
                    cpu_utilization=0,
                    passed=False,
                    error_message="All test iterations failed",
                    metadata={"failed_iterations": iteration_results},
                )

            test_results.append(test_result)
            self.results.append(test_result)

        # Analyze scaling behavior
        analysis = self._analyze_scaling_results(test_results, configuration)
        self.analyses.append(analysis)

        logger.info(f"Completed scaling test: {test_name}")
        return analysis

    def _generate_test_dataset(self, size: int) -> List[str]:
        """Generate test molecular dataset."""
        # Generate realistic-looking SMILES strings
        base_molecules = [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "CCN",
            "C=C",
            "CO",
            "CS",
            "c1ccc2ccccc2c1",
            "C1CCCCC1",
            "CC(C)C",
            "CCCC",
            "c1ccncc1",
        ]

        dataset = []
        for i in range(size):
            base = base_molecules[i % len(base_molecules)]
            # Add some variation
            if i % 10 == 0:
                dataset.append(f"C{base}")
            elif i % 7 == 0:
                dataset.append(f"{base}C")
            else:
                dataset.append(base)

        return dataset

    def _run_multiprocess_test(
        self, test_function: Callable, dataset: List[str], process_count: int
    ) -> Any:
        """Run test using multiprocessing."""
        chunk_size = len(dataset) // process_count
        chunks = [
            dataset[i : i + chunk_size] for i in range(0, len(dataset), chunk_size)
        ]

        with ProcessPoolExecutor(max_workers=process_count) as executor:
            futures = [executor.submit(test_function, chunk, 1) for chunk in chunks]
            results = [future.result() for future in as_completed(futures)]

        # Combine results
        return {"combined_results": results, "total_items": len(dataset)}

    def _run_multigpu_test(
        self, test_function: Callable, dataset: List[str], gpu_count: int
    ) -> Any:
        """Run test using multiple GPUs."""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for multi-GPU testing")

        # Mock multi-GPU execution
        chunk_size = len(dataset) // gpu_count
        chunks = [
            dataset[i : i + chunk_size] for i in range(0, len(dataset), chunk_size)
        ]

        results = []
        for i, chunk in enumerate(chunks):
            # Simulate GPU selection
            with cp.cuda.Device(i % cp.cuda.runtime.getDeviceCount()):
                result = test_function(chunk, 1)
                results.append(result)

        return {"gpu_results": results, "total_items": len(dataset)}

    def _run_multinode_test(
        self, test_function: Callable, dataset: List[str], node_count: int
    ) -> Any:
        """Run test using Dask distributed computing."""
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask not available for multi-node testing")

        # Create local cluster for simulation
        with LocalCluster(n_workers=node_count, threads_per_worker=1) as cluster:
            with Client(cluster) as client:
                # Distribute computation
                chunk_size = len(dataset) // node_count
                chunks = [
                    dataset[i : i + chunk_size]
                    for i in range(0, len(dataset), chunk_size)
                ]

                futures = [client.submit(test_function, chunk, 1) for chunk in chunks]
                results = client.gather(futures)

        return {"distributed_results": results, "total_items": len(dataset)}

    def _aggregate_resource_stats(
        self, stats_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate resource statistics from multiple iterations."""
        if not stats_list:
            return {}

        aggregated = {}

        # Get all unique keys
        all_keys = set()
        for stats in stats_list:
            all_keys.update(stats.keys())

        # Aggregate each metric
        for key in all_keys:
            values = [stats.get(key, 0) for stats in stats_list if key in stats]
            if values:
                if "max" in key:
                    aggregated[key] = max(values)
                elif "avg" in key or "mean" in key:
                    aggregated[key] = statistics.mean(values)
                else:
                    aggregated[key] = statistics.mean(values)

        return aggregated

    def _analyze_scaling_results(
        self, results: List[ScalingResult], configuration: ScalingConfiguration
    ) -> ScalingAnalysis:
        """Analyze scaling efficiency results."""
        valid_results = [r for r in results if r.passed]

        if not valid_results:
            return ScalingAnalysis(
                configuration=configuration,
                results=results,
                ideal_speedups=[],
                actual_speedups=[],
                efficiencies=[],
                bottlenecks=["All tests failed"],
                recommendations=["Investigate test failures"],
                overall_score=0.0,
            )

        # Calculate ideal vs actual performance
        resource_counts = [r.resource_count for r in valid_results]
        actual_speedups = [r.speedup for r in valid_results]
        efficiencies = [r.efficiency for r in valid_results]

        if configuration.scaling_type == ScalingType.STRONG_SCALING:
            ideal_speedups = resource_counts  # Perfect linear scaling
        else:
            ideal_speedups = [1.0] * len(resource_counts)  # Perfect weak scaling

        # Identify bottlenecks and recommendations
        bottlenecks = []
        recommendations = []

        # Check for scaling inefficiencies
        if len(efficiencies) > 1:
            efficiency_decline = efficiencies[0] - efficiencies[-1]
            if efficiency_decline > 0.3:
                bottlenecks.append(
                    "Significant efficiency decline with increased resources"
                )
                recommendations.append(
                    "Investigate communication overhead and load balancing"
                )

        # Check for memory bottlenecks
        memory_usage = [r.memory_usage_mb for r in valid_results]
        if len(memory_usage) > 1:
            memory_growth_rate = (memory_usage[-1] - memory_usage[0]) / memory_usage[0]
            if memory_growth_rate > 1.0:  # More than 100% increase
                bottlenecks.append("Memory usage grows superlinearly with resources")
                recommendations.append(
                    "Optimize memory usage and consider memory-efficient algorithms"
                )

        # Check for CPU utilization
        cpu_utilizations = [
            r.cpu_utilization for r in valid_results if r.cpu_utilization > 0
        ]
        if cpu_utilizations:
            avg_cpu = statistics.mean(cpu_utilizations)
            if avg_cpu < 70:
                bottlenecks.append(
                    "Low CPU utilization indicates underutilized resources"
                )
                recommendations.append(
                    "Increase workload per worker or reduce resource count"
                )

        # Find scalability limit (where efficiency drops below 50%)
        scalability_limit = None
        for i, eff in enumerate(efficiencies):
            if eff < 0.5:
                scalability_limit = resource_counts[i]
                break

        # Find optimal resource count (highest throughput per resource)
        if len(valid_results) > 1:
            throughput_per_resource = [
                r.throughput / r.resource_count for r in valid_results
            ]
            optimal_idx = throughput_per_resource.index(max(throughput_per_resource))
            optimal_resource_count = valid_results[optimal_idx].resource_count
        else:
            optimal_resource_count = resource_counts[0] if resource_counts else None

        # Calculate overall scaling score
        if efficiencies:
            avg_efficiency = statistics.mean(efficiencies)
            efficiency_stability = 1.0 - (
                statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0
            )
            overall_score = (avg_efficiency * 0.7) + (efficiency_stability * 0.3)
        else:
            overall_score = 0.0

        return ScalingAnalysis(
            configuration=configuration,
            results=results,
            ideal_speedups=ideal_speedups,
            actual_speedups=actual_speedups,
            efficiencies=efficiencies,
            scalability_limit=scalability_limit,
            optimal_resource_count=optimal_resource_count,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            overall_score=overall_score,
        )

    def run_all_standard_tests(
        self, test_function: Callable
    ) -> Dict[str, ScalingAnalysis]:
        """Run all standard scaling efficiency tests.

        Args:
            test_function: Function to test

        Returns:
            Dictionary of scaling analyses by test name
        """
        logger.info("Running all standard scaling efficiency tests")

        analyses = {}

        for i, config in enumerate(self.standard_configs):
            test_name = f"standard_scaling_test_{i+1}_{config.scaling_type.value}_{config.distribution_type.value}"

            try:
                analysis = self.run_scaling_test(test_function, config, test_name)
                analyses[test_name] = analysis
            except Exception as e:
                logger.error(f"Standard test {test_name} failed: {e}")
                # Create failed analysis
                analyses[test_name] = ScalingAnalysis(
                    configuration=config,
                    results=[],
                    ideal_speedups=[],
                    actual_speedups=[],
                    efficiencies=[],
                    bottlenecks=[f"Test execution failed: {str(e)}"],
                    recommendations=[
                        "Fix test execution issues before running scaling tests"
                    ],
                    overall_score=0.0,
                )

        # Save results
        self.save_results()

        logger.info(f"Completed all standard scaling tests: {len(analyses)} analyses")
        return analyses

    def generate_scaling_report(self, analyses: Dict[str, ScalingAnalysis]) -> str:
        """Generate comprehensive scaling efficiency report.

        Args:
            analyses: Dictionary of scaling analyses

        Returns:
            Formatted scaling report
        """
        report = ["Scaling Efficiency Test Report", "=" * 40, ""]

        # Executive Summary
        total_tests = len(analyses)
        successful_tests = sum(1 for a in analyses.values() if a.overall_score > 0)
        avg_score = (
            statistics.mean([a.overall_score for a in analyses.values()])
            if analyses
            else 0
        )

        report.extend(
            [
                "EXECUTIVE SUMMARY",
                "-" * 20,
                f"Total Scaling Tests: {total_tests}",
                f"Successful Tests: {successful_tests}",
                f"Average Scaling Score: {avg_score:.2f}",
                f"Overall Assessment: {'EXCELLENT' if avg_score > 0.8 else 'GOOD' if avg_score > 0.6 else 'NEEDS_IMPROVEMENT'}",
                "",
            ]
        )

        # Detailed Results
        for test_name, analysis in analyses.items():
            report.extend(
                [
                    f"TEST: {test_name}",
                    "-" * len(test_name) + "------",
                    f"Configuration: {analysis.configuration.scaling_type.value} / {analysis.configuration.distribution_type.value}",
                    f"Resource Range: {min(analysis.configuration.resource_counts)} - {max(analysis.configuration.resource_counts)}",
                    f"Overall Score: {analysis.overall_score:.2f}",
                    "",
                ]
            )

            if analysis.results:
                successful_results = [r for r in analysis.results if r.passed]

                if successful_results:
                    report.extend(
                        [
                            "Performance Metrics:",
                            f"  Best Efficiency: {max(r.efficiency for r in successful_results):.2f}",
                            f"  Best Speedup: {max(r.speedup for r in successful_results):.2f}",
                            f"  Optimal Resource Count: {analysis.optimal_resource_count}",
                            f"  Scalability Limit: {analysis.scalability_limit or 'Not reached'}",
                            "",
                        ]
                    )

                    # Resource utilization
                    avg_cpu = statistics.mean(
                        [
                            r.cpu_utilization
                            for r in successful_results
                            if r.cpu_utilization > 0
                        ]
                    )
                    avg_memory = statistics.mean(
                        [r.memory_usage_mb for r in successful_results]
                    )

                    report.extend(
                        [
                            "Resource Utilization:",
                            f"  Average CPU: {avg_cpu:.1f}%",
                            f"  Average Memory: {avg_memory:.1f} MB",
                            "",
                        ]
                    )

            # Bottlenecks and Recommendations
            if analysis.bottlenecks:
                report.extend(["Identified Bottlenecks:"])
                for bottleneck in analysis.bottlenecks:
                    report.append(f"  • {bottleneck}")
                report.append("")

            if analysis.recommendations:
                report.extend(["Recommendations:"])
                for rec in analysis.recommendations:
                    report.append(f"  • {rec}")
                report.append("")

            report.append("=" * 50)
            report.append("")

        return "\n".join(report)

    def save_results(self):
        """Save scaling test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "test_name": result.test_name,
                    "resource_count": result.resource_count,
                    "dataset_size": result.dataset_size,
                    "execution_time": result.execution_time,
                    "throughput": result.throughput,
                    "efficiency": result.efficiency,
                    "speedup": result.speedup,
                    "memory_usage_mb": result.memory_usage_mb,
                    "cpu_utilization": result.cpu_utilization,
                    "network_usage_mbps": result.network_usage_mbps,
                    "gpu_utilization": result.gpu_utilization,
                    "passed": result.passed,
                    "error_message": result.error_message,
                    "configuration": {
                        "scaling_type": result.configuration.scaling_type.value,
                        "distribution_type": result.configuration.distribution_type.value,
                        "resource_counts": result.configuration.resource_counts,
                        "dataset_sizes": result.configuration.dataset_sizes,
                    },
                    "metadata": result.metadata,
                }
            )

        results_file = self.output_dir / f"scaling_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        # Save analyses
        analyses_data = []
        for analysis in self.analyses:
            analyses_data.append(
                {
                    "configuration": {
                        "scaling_type": analysis.configuration.scaling_type.value,
                        "distribution_type": analysis.configuration.distribution_type.value,
                        "resource_counts": analysis.configuration.resource_counts,
                        "dataset_sizes": analysis.configuration.dataset_sizes,
                    },
                    "ideal_speedups": analysis.ideal_speedups,
                    "actual_speedups": analysis.actual_speedups,
                    "efficiencies": analysis.efficiencies,
                    "scalability_limit": analysis.scalability_limit,
                    "optimal_resource_count": analysis.optimal_resource_count,
                    "bottlenecks": analysis.bottlenecks,
                    "recommendations": analysis.recommendations,
                    "overall_score": analysis.overall_score,
                }
            )

        analyses_file = self.output_dir / f"scaling_analyses_{timestamp}.json"
        with open(analyses_file, "w") as f:
            json.dump(analyses_data, f, indent=2)

        logger.info(f"Scaling test results saved to {results_file}")
        logger.info(f"Scaling analyses saved to {analyses_file}")


def mock_molecular_descriptor_calculation(
    molecules: List[str], resource_count: int = 1
) -> Dict[str, Any]:
    """Mock molecular descriptor calculation for testing.

    Args:
        molecules: List of molecule SMILES
        resource_count: Number of resources (for compatibility)

    Returns:
        Dictionary with calculation results
    """
    # Simulate computation time based on molecular complexity
    base_time_per_molecule = 0.001  # 1ms per molecule
    computation_time = len(molecules) * base_time_per_molecule

    # Simulate slight performance variation
    computation_time *= np.random.uniform(0.9, 1.1)

    # Actually sleep to simulate real computation
    time.sleep(computation_time)

    # Generate mock descriptors
    descriptors = {
        "molecular_weight": np.random.uniform(50, 500, len(molecules)).tolist(),
        "tpsa": np.random.uniform(20, 200, len(molecules)).tolist(),
        "logp": np.random.uniform(-3, 5, len(molecules)).tolist(),
        "hbd": np.random.randint(0, 10, len(molecules)).tolist(),
        "hba": np.random.randint(0, 15, len(molecules)).tolist(),
    }

    return {
        "molecule_count": len(molecules),
        "descriptors": descriptors,
        "computation_time": computation_time,
        "resource_count": resource_count,
    }


if __name__ == "__main__":
    # Demonstrate scaling efficiency testing
    print("ChemiAccel Scaling Efficiency Testing Framework")
    print("=" * 50)

    # Initialize tester
    tester = ScalingEfficiencyTester()

    # Run standard scaling tests
    print("\nRunning standard scaling efficiency tests...")
    analyses = tester.run_all_standard_tests(mock_molecular_descriptor_calculation)

    # Generate and display report
    report = tester.generate_scaling_report(analyses)

    print("\n" + "=" * 50)
    print("SCALING EFFICIENCY REPORT")
    print("=" * 50)
    print(report)

    # Save detailed report
    report_file = (
        tester.output_dir
        / f"scaling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\nDetailed report saved to: {report_file}")

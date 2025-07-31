"""
Scaling Tests for ChemiAccel Performance Analysis.

This module provides comprehensive scaling tests across different dataset sizes
and hardware configurations to analyze performance characteristics.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    pd = None
    np = None
    plt = None

try:
    from ..src.chemi_accel.utils.logging import get_logger
except ImportError:
    # Fallback for standalone execution
    import logging

    def get_logger():
        return logging.getLogger(__name__)


logger = get_logger()


class ScalingType(Enum):
    """Types of scaling tests."""

    DATASET_SIZE = "dataset_size"
    BATCH_SIZE = "batch_size"
    HARDWARE_CORES = "hardware_cores"
    MEMORY_USAGE = "memory_usage"


@dataclass
class HardwareConfig:
    """Hardware configuration for scaling tests."""

    name: str
    description: str
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 1
    system_memory_gb: float = 8.0
    backend: str = "cpu"
    device_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingResult:
    """Result of a scaling test."""

    test_name: str
    scaling_type: ScalingType
    hardware_config: HardwareConfig
    implementation: str
    descriptor: str
    scale_values: List[float]  # X-axis values (e.g., dataset sizes)
    execution_times: List[float]  # Y-axis values (execution times)
    throughputs: List[float]  # Molecules per second
    memory_usage: List[float]  # Memory usage in MB
    scaling_efficiency: float  # How well it scales (0-1)
    theoretical_speedup: float  # Expected speedup
    actual_speedup: float  # Measured speedup
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScalingTestRunner:
    """Runs comprehensive scaling tests for performance analysis."""

    def __init__(self, output_directory: str = "scaling_results"):
        """Initialize scaling test runner.

        Args:
            output_directory: Directory to save results
        """
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[ScalingResult] = []
        self.hardware_configs: Dict[str, HardwareConfig] = {}

        # Detect available hardware configurations
        self._detect_hardware_configs()

        logger.info(
            f"Initialized scaling test runner with {len(self.hardware_configs)} hardware configs"
        )

    def _detect_hardware_configs(self) -> None:
        """Detect available hardware configurations."""
        # CPU configuration
        cpu_config = HardwareConfig(
            name="cpu",
            description="CPU-only execution",
            cpu_cores=self._get_cpu_count(),
            system_memory_gb=self._get_system_memory_gb(),
            backend="cpu",
        )
        self.hardware_configs["cpu"] = cpu_config

        # Try to detect CUDA configuration
        try:
            import cupy as cp

            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                device = cp.cuda.Device(0)
                device.use()

                # Get GPU memory
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                gpu_memory_gb = total_mem / (1024**3)

                # Get device properties
                device_props = device.attributes

                cuda_config = HardwareConfig(
                    name="cuda",
                    description="CUDA GPU execution",
                    gpu_count=device_count,
                    gpu_memory_gb=gpu_memory_gb,
                    cpu_cores=self._get_cpu_count(),
                    system_memory_gb=self._get_system_memory_gb(),
                    backend="cuda",
                    device_info={
                        "device_name": device_props.get("Name", "Unknown"),
                        "compute_capability": f"{device_props.get('ComputeCapabilityMajor', 0)}.{device_props.get('ComputeCapabilityMinor', 0)}",
                        "multiprocessor_count": device_props.get(
                            "MultiProcessorCount", 0
                        ),
                        "max_threads_per_block": device_props.get(
                            "MaxThreadsPerBlock", 0
                        ),
                    },
                )
                self.hardware_configs["cuda"] = cuda_config

        except ImportError:
            logger.info("CUDA not available for scaling tests")
        except Exception as e:
            logger.warning(f"Failed to detect CUDA configuration: {e}")

        # Try to detect JAX configuration
        try:
            import jax

            devices = jax.devices()
            if devices:
                device_type = devices[0].device_kind

                jax_config = HardwareConfig(
                    name="jax",
                    description=f"JAX {device_type} execution",
                    gpu_count=len([d for d in devices if d.device_kind == "gpu"]),
                    cpu_cores=len([d for d in devices if d.device_kind == "cpu"]),
                    system_memory_gb=self._get_system_memory_gb(),
                    backend="jax",
                    device_info={
                        "devices": [str(d) for d in devices],
                        "backend": jax.lib.xla_bridge.get_backend().platform,
                    },
                )
                self.hardware_configs["jax"] = jax_config

        except ImportError:
            logger.info("JAX not available for scaling tests")
        except Exception as e:
            logger.warning(f"Failed to detect JAX configuration: {e}")

    def _get_cpu_count(self) -> int:
        """Get number of CPU cores."""
        try:
            import multiprocessing

            return multiprocessing.cpu_count()
        except Exception:
            return 1

    def _get_system_memory_gb(self) -> float:
        """Get system memory in GB."""
        try:
            import psutil

            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 8.0  # Default assumption
        except Exception:
            return 8.0

    def run_dataset_size_scaling(
        self,
        implementation_func: callable,
        datasets: Dict[int, Any],
        hardware_config: HardwareConfig,
        implementation_name: str,
        descriptor: str,
        test_name: str = "dataset_size_scaling",
    ) -> ScalingResult:
        """Run dataset size scaling test.

        Args:
            implementation_func: Function to test
            datasets: Dictionary mapping size to dataset
            hardware_config: Hardware configuration
            implementation_name: Name of implementation
            descriptor: Descriptor being tested
            test_name: Name of the test

        Returns:
            Scaling test result
        """
        logger.info(f"Running dataset size scaling test: {test_name}")

        scale_values = []
        execution_times = []
        throughputs = []
        memory_usage = []

        # Sort dataset sizes
        sorted_sizes = sorted(datasets.keys())

        for size in sorted_sizes:
            dataset = datasets[size]

            logger.info(f"Testing dataset size: {size}")

            # Measure memory before
            memory_before = self._get_memory_usage()

            # Time execution
            start_time = time.perf_counter()

            try:
                result = implementation_func(dataset)

                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # Measure memory after
                memory_after = self._get_memory_usage()
                memory_used = memory_after - memory_before

                # Calculate throughput
                throughput = size / execution_time if execution_time > 0 else 0.0

                scale_values.append(float(size))
                execution_times.append(execution_time)
                throughputs.append(throughput)
                memory_usage.append(memory_used)

                logger.info(
                    f"Size {size}: {execution_time:.3f}s, {throughput:.0f} mol/s"
                )

            except Exception as e:
                logger.error(f"Failed to test size {size}: {e}")
                # Add placeholder values to maintain array consistency
                scale_values.append(float(size))
                execution_times.append(float("inf"))
                throughputs.append(0.0)
                memory_usage.append(0.0)

        # Calculate scaling metrics
        scaling_efficiency = self._calculate_scaling_efficiency(
            scale_values, execution_times
        )
        theoretical_speedup = (
            scale_values[-1] / scale_values[0] if scale_values else 1.0
        )
        actual_speedup = (
            execution_times[0] / execution_times[-1]
            if execution_times and execution_times[-1] > 0
            else 1.0
        )

        result = ScalingResult(
            test_name=test_name,
            scaling_type=ScalingType.DATASET_SIZE,
            hardware_config=hardware_config,
            implementation=implementation_name,
            descriptor=descriptor,
            scale_values=scale_values,
            execution_times=execution_times,
            throughputs=throughputs,
            memory_usage=memory_usage,
            scaling_efficiency=scaling_efficiency,
            theoretical_speedup=theoretical_speedup,
            actual_speedup=actual_speedup,
            metadata={
                "dataset_sizes": sorted_sizes,
                "total_test_time": sum(execution_times),
            },
        )

        self.results.append(result)

        logger.info(
            f"Dataset scaling test completed: efficiency={scaling_efficiency:.3f}, "
            f"speedup={actual_speedup:.2f}x"
        )

        return result

    def run_batch_size_scaling(
        self,
        implementation_func: callable,
        dataset: Any,
        batch_sizes: List[int],
        hardware_config: HardwareConfig,
        implementation_name: str,
        descriptor: str,
        test_name: str = "batch_size_scaling",
    ) -> ScalingResult:
        """Run batch size scaling test.

        Args:
            implementation_func: Function to test (should accept batch_size parameter)
            dataset: Dataset to use for testing
            batch_sizes: List of batch sizes to test
            hardware_config: Hardware configuration
            implementation_name: Name of implementation
            descriptor: Descriptor being tested
            test_name: Name of the test

        Returns:
            Scaling test result
        """
        logger.info(f"Running batch size scaling test: {test_name}")

        scale_values = []
        execution_times = []
        throughputs = []
        memory_usage = []

        dataset_size = len(dataset) if hasattr(dataset, "__len__") else 1000

        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")

            # Measure memory before
            memory_before = self._get_memory_usage()

            # Time execution
            start_time = time.perf_counter()

            try:
                # Call function with batch size parameter
                result = implementation_func(dataset, batch_size=batch_size)

                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # Measure memory after
                memory_after = self._get_memory_usage()
                memory_used = memory_after - memory_before

                # Calculate throughput
                throughput = (
                    dataset_size / execution_time if execution_time > 0 else 0.0
                )

                scale_values.append(float(batch_size))
                execution_times.append(execution_time)
                throughputs.append(throughput)
                memory_usage.append(memory_used)

                logger.info(
                    f"Batch size {batch_size}: {execution_time:.3f}s, {throughput:.0f} mol/s"
                )

            except Exception as e:
                logger.error(f"Failed to test batch size {batch_size}: {e}")
                scale_values.append(float(batch_size))
                execution_times.append(float("inf"))
                throughputs.append(0.0)
                memory_usage.append(0.0)

        # Calculate scaling metrics
        scaling_efficiency = self._calculate_scaling_efficiency(
            scale_values, execution_times
        )

        # For batch size scaling, optimal is usually the largest batch size that fits in memory
        best_throughput_idx = throughputs.index(max(throughputs)) if throughputs else 0
        optimal_batch_size = (
            scale_values[best_throughput_idx] if scale_values else batch_sizes[0]
        )

        result = ScalingResult(
            test_name=test_name,
            scaling_type=ScalingType.BATCH_SIZE,
            hardware_config=hardware_config,
            implementation=implementation_name,
            descriptor=descriptor,
            scale_values=scale_values,
            execution_times=execution_times,
            throughputs=throughputs,
            memory_usage=memory_usage,
            scaling_efficiency=scaling_efficiency,
            theoretical_speedup=1.0,  # Batch size doesn't theoretically improve speed
            actual_speedup=(
                max(throughputs) / min(throughputs)
                if throughputs and min(throughputs) > 0
                else 1.0
            ),
            metadata={
                "batch_sizes": batch_sizes,
                "optimal_batch_size": optimal_batch_size,
                "dataset_size": dataset_size,
            },
        )

        self.results.append(result)

        logger.info(
            f"Batch size scaling test completed: optimal_batch_size={optimal_batch_size}"
        )

        return result

    def _calculate_scaling_efficiency(
        self, scale_values: List[float], execution_times: List[float]
    ) -> float:
        """Calculate scaling efficiency.

        Args:
            scale_values: Scale values (e.g., dataset sizes)
            execution_times: Corresponding execution times

        Returns:
            Scaling efficiency (0-1, where 1 is perfect linear scaling)
        """
        if len(scale_values) < 2 or len(execution_times) < 2:
            return 0.0

        # Remove invalid measurements
        valid_pairs = [
            (s, t)
            for s, t in zip(scale_values, execution_times)
            if t > 0 and t != float("inf")
        ]

        if len(valid_pairs) < 2:
            return 0.0

        scale_values, execution_times = zip(*valid_pairs)

        # Calculate correlation coefficient between scale and time
        # Perfect linear scaling would have correlation = 1
        try:
            if np is not None:
                correlation = np.corrcoef(scale_values, execution_times)[0, 1]
                # Convert correlation to efficiency (higher correlation = better scaling)
                efficiency = max(0.0, correlation)
            else:
                # Fallback calculation without numpy
                n = len(scale_values)
                mean_scale = sum(scale_values) / n
                mean_time = sum(execution_times) / n

                numerator = sum(
                    (s - mean_scale) * (t - mean_time)
                    for s, t in zip(scale_values, execution_times)
                )

                scale_var = sum((s - mean_scale) ** 2 for s in scale_values)
                time_var = sum((t - mean_time) ** 2 for t in execution_times)

                denominator = (scale_var * time_var) ** 0.5

                if denominator > 0:
                    correlation = numerator / denominator
                    efficiency = max(0.0, correlation)
                else:
                    efficiency = 0.0

            return min(1.0, efficiency)

        except Exception as e:
            logger.warning(f"Failed to calculate scaling efficiency: {e}")
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def run_comprehensive_scaling_suite(
        self,
        implementations: Dict[str, callable],
        datasets: Dict[int, Any],
        descriptors: List[str],
        batch_sizes: List[int] = None,
    ) -> List[ScalingResult]:
        """Run comprehensive scaling tests across all configurations.

        Args:
            implementations: Dictionary of implementation functions by backend
            datasets: Dictionary of datasets by size
            descriptors: List of descriptors to test
            batch_sizes: List of batch sizes to test (optional)

        Returns:
            List of all scaling results
        """
        if batch_sizes is None:
            batch_sizes = [32, 64, 128, 256, 512, 1024]

        logger.info("Starting comprehensive scaling test suite")

        all_results = []

        for backend_name, impl_func in implementations.items():
            if backend_name not in self.hardware_configs:
                logger.warning(
                    f"No hardware config for backend {backend_name}, skipping"
                )
                continue

            hardware_config = self.hardware_configs[backend_name]

            for descriptor in descriptors:
                logger.info(f"Testing {backend_name} {descriptor}")

                # Dataset size scaling
                try:
                    result = self.run_dataset_size_scaling(
                        impl_func,
                        datasets,
                        hardware_config,
                        backend_name,
                        descriptor,
                        f"{backend_name}_{descriptor}_dataset_scaling",
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error(
                        f"Dataset scaling test failed for {backend_name} {descriptor}: {e}"
                    )

                # Batch size scaling (if implementation supports it)
                if (
                    hasattr(impl_func, "__code__")
                    and "batch_size" in impl_func.__code__.co_varnames
                ):
                    try:
                        # Use medium-sized dataset for batch size testing
                        test_dataset = datasets.get(10000, list(datasets.values())[0])

                        result = self.run_batch_size_scaling(
                            impl_func,
                            test_dataset,
                            batch_sizes,
                            hardware_config,
                            backend_name,
                            descriptor,
                            f"{backend_name}_{descriptor}_batch_scaling",
                        )
                        all_results.append(result)
                    except Exception as e:
                        logger.error(
                            f"Batch scaling test failed for {backend_name} {descriptor}: {e}"
                        )

        logger.info(
            f"Completed comprehensive scaling suite: {len(all_results)} results"
        )

        # Save results
        self.save_results()

        return all_results

    def save_results(self) -> None:
        """Save scaling test results to files."""
        # Save raw results as JSON
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "test_name": result.test_name,
                    "scaling_type": result.scaling_type.value,
                    "hardware_config": {
                        "name": result.hardware_config.name,
                        "description": result.hardware_config.description,
                        "gpu_count": result.hardware_config.gpu_count,
                        "gpu_memory_gb": result.hardware_config.gpu_memory_gb,
                        "cpu_cores": result.hardware_config.cpu_cores,
                        "system_memory_gb": result.hardware_config.system_memory_gb,
                        "backend": result.hardware_config.backend,
                        "device_info": result.hardware_config.device_info,
                    },
                    "implementation": result.implementation,
                    "descriptor": result.descriptor,
                    "scale_values": result.scale_values,
                    "execution_times": result.execution_times,
                    "throughputs": result.throughputs,
                    "memory_usage": result.memory_usage,
                    "scaling_efficiency": result.scaling_efficiency,
                    "theoretical_speedup": result.theoretical_speedup,
                    "actual_speedup": result.actual_speedup,
                    "metadata": result.metadata,
                }
            )

        results_file = self.output_dir / "scaling_results.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved scaling results to {results_file}")

    def generate_scaling_plots(self) -> None:
        """Generate scaling analysis plots."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot generation")
            return

        # Group results by scaling type
        by_scaling_type = {}
        for result in self.results:
            scaling_type = result.scaling_type
            if scaling_type not in by_scaling_type:
                by_scaling_type[scaling_type] = []
            by_scaling_type[scaling_type].append(result)

        # Generate plots for each scaling type
        for scaling_type, results in by_scaling_type.items():
            self._plot_scaling_results(scaling_type, results)

    def _plot_scaling_results(
        self, scaling_type: ScalingType, results: List[ScalingResult]
    ) -> None:
        """Plot scaling results for a specific scaling type."""
        if not PLOTTING_AVAILABLE:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{scaling_type.value.replace("_", " ").title()} Scaling Analysis')

        # Plot 1: Execution time vs scale
        for result in results:
            label = f"{result.implementation}_{result.descriptor}"
            ax1.plot(result.scale_values, result.execution_times, "o-", label=label)

        ax1.set_xlabel("Scale Value")
        ax1.set_ylabel("Execution Time (s)")
        ax1.set_title("Execution Time vs Scale")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Throughput vs scale
        for result in results:
            label = f"{result.implementation}_{result.descriptor}"
            ax2.plot(result.scale_values, result.throughputs, "s-", label=label)

        ax2.set_xlabel("Scale Value")
        ax2.set_ylabel("Throughput (molecules/s)")
        ax2.set_title("Throughput vs Scale")
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Memory usage vs scale
        for result in results:
            label = f"{result.implementation}_{result.descriptor}"
            ax3.plot(result.scale_values, result.memory_usage, "^-", label=label)

        ax3.set_xlabel("Scale Value")
        ax3.set_ylabel("Memory Usage (MB)")
        ax3.set_title("Memory Usage vs Scale")
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Scaling efficiency comparison
        implementations = [result.implementation for result in results]
        efficiencies = [result.scaling_efficiency for result in results]

        ax4.bar(implementations, efficiencies)
        ax4.set_ylabel("Scaling Efficiency")
        ax4.set_title("Scaling Efficiency Comparison")
        ax4.set_ylim(0, 1)

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / f"{scaling_type.value}_scaling_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved scaling plot to {plot_file}")


def run_scaling_tests(
    implementations: Dict[str, callable],
    datasets: Dict[int, Any],
    descriptors: List[str],
    output_directory: str = "scaling_results",
) -> List[ScalingResult]:
    """Run comprehensive scaling tests.

    Args:
        implementations: Dictionary of implementation functions by backend
        datasets: Dictionary of datasets by size
        descriptors: List of descriptors to test
        output_directory: Directory to save results

    Returns:
        List of scaling test results
    """
    runner = ScalingTestRunner(output_directory)
    results = runner.run_comprehensive_scaling_suite(
        implementations, datasets, descriptors
    )
    runner.generate_scaling_plots()
    return results

"""
Distributed computing utilities for ChemiAccel.
"""

import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class PerformanceMetrics:
    """Performance metrics for distributed calculations."""

    total_time: float
    compute_time: float
    communication_time: float
    io_time: float
    memory_usage_mb: float
    cpu_utilization: float
    gpu_utilization: Optional[float] = None
    throughput_molecules_per_sec: float = 0.0
    scaling_efficiency: Optional[float] = None


class PerformanceMonitor:
    """Monitor performance metrics during distributed calculations."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.monitoring_active = False

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.monitoring_active = True
        self.metrics = {"cpu_usage": [], "memory_usage": [], "timestamps": []}
        logger.info("Performance monitoring started")

    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return metrics."""
        self.end_time = time.time()
        self.monitoring_active = False

        total_time = self.end_time - self.start_time

        metrics = PerformanceMetrics(
            total_time=total_time,
            compute_time=total_time,
            communication_time=0.0,
            io_time=0.0,
            memory_usage_mb=0.0,
            cpu_utilization=0.0,
        )

        logger.info(f"Performance monitoring completed: {total_time:.2f}s total")
        return metrics


class LoadBalancer:
    """Load balancing utilities for distributed processing."""

    @staticmethod
    def balance_workload(total_items: int, num_workers: int) -> List[Tuple[int, int]]:
        """Balance workload across workers."""
        items_per_worker = total_items // num_workers
        remainder = total_items % num_workers

        ranges = []
        start = 0

        for i in range(num_workers):
            # First 'remainder' workers get one extra item
            chunk_size = items_per_worker + (1 if i < remainder else 0)
            end = start + chunk_size
            ranges.append((start, end))
            start = end

        return ranges

    @staticmethod
    def calculate_load_balance_factor(work_times: List[float]) -> float:
        """Calculate load balance factor (1.0 = perfect balance)."""
        if not work_times:
            return 0.0

        min_time = min(work_times)
        max_time = max(work_times)

        if max_time == 0:
            return 1.0

        return min_time / max_time


def generate_test_molecules(count: int, complexity: str = "mixed") -> pd.Series:
    """Generate test molecules for benchmarking."""
    molecules = []

    if complexity == "simple":
        base_molecules = ["CCO", "CC(C)O", "c1ccccc1"]
    elif complexity == "complex":
        base_molecules = [
            "CC(C)(C)c1ccc(cc1)C(C)(C)c2ccc(cc2)C(C)(C)C",
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        ]
    else:  # mixed
        base_molecules = [
            "CCO",
            "CC(C)O",
            "c1ccccc1",
            "CCc1ccccc1",
            "CC(C)(C)c1ccc(cc1)C(C)(C)C",
        ]

    while len(molecules) < count:
        molecules.extend(base_molecules)

    return pd.Series(molecules[:count])


def benchmark_scaling_efficiency(
    calculator_func: Callable, test_data: pd.Series, node_counts: List[int]
) -> Dict[str, Any]:
    """Benchmark scaling efficiency across different node counts."""
    results = {}
    baseline_time = None

    for node_count in node_counts:
        monitor = PerformanceMonitor()
        monitor.start_monitoring()

        # Run calculation
        try:
            result = calculator_func(test_data, nodes=node_count)
            metrics = monitor.stop_monitoring()

            if baseline_time is None:
                baseline_time = metrics.total_time

            # Calculate scaling efficiency
            ideal_time = baseline_time / node_count
            efficiency = (
                ideal_time / metrics.total_time if metrics.total_time > 0 else 0
            )

            results[node_count] = {
                "total_time": metrics.total_time,
                "efficiency": efficiency,
                "throughput": (
                    len(test_data) / metrics.total_time if metrics.total_time > 0 else 0
                ),
                "memory_usage": metrics.memory_usage_mb,
            }

        except Exception as e:
            logger.error(f"Benchmark failed for {node_count} nodes: {e}")
            results[node_count] = {
                "total_time": float("inf"),
                "efficiency": 0.0,
                "throughput": 0.0,
                "error": str(e),
            }

    return results


def validate_hpc_environment() -> Dict[str, Any]:
    """Validate HPC environment and available resources."""
    validation = {
        "mpi_available": False,
        "slurm_available": False,
        "lustre_available": False,
        "gpfs_available": False,
        "infiniband_available": False,
        "gpu_count": 0,
        "node_count": 1,
        "recommendations": [],
    }

    # Check MPI
    try:
        result = subprocess.run(["mpirun", "--version"], capture_output=True)
        validation["mpi_available"] = result.returncode == 0
    except FileNotFoundError:
        pass

    # Check SLURM
    try:
        result = subprocess.run(["sinfo", "--version"], capture_output=True)
        validation["slurm_available"] = result.returncode == 0

        if validation["slurm_available"]:
            # Get node count from SLURM
            try:
                result = subprocess.run(
                    ["sinfo", "-h", "-o", "%D"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    validation["node_count"] = sum(
                        int(x) for x in result.stdout.strip().split("\n") if x.isdigit()
                    )
            except:
                pass
    except FileNotFoundError:
        pass

    # Check Lustre
    try:
        result = subprocess.run(["lfs", "--version"], capture_output=True)
        validation["lustre_available"] = result.returncode == 0
    except FileNotFoundError:
        pass

    # Check GPFS
    try:
        result = subprocess.run(["mmlsfs", "all"], capture_output=True)
        validation["gpfs_available"] = result.returncode == 0
    except FileNotFoundError:
        pass

    # Check InfiniBand
    try:
        result = subprocess.run(["ibstat"], capture_output=True)
        validation["infiniband_available"] = result.returncode == 0
    except FileNotFoundError:
        pass

    # Check GPUs
    try:
        import pynvml

        pynvml.nvmlInit()
        validation["gpu_count"] = pynvml.nvmlDeviceGetCount()
    except:
        pass

    # Generate recommendations
    if not validation["mpi_available"]:
        validation["recommendations"].append("Install MPI for distributed processing")

    if validation["lustre_available"]:
        validation["recommendations"].append("Use Lustre striping for large files")

    if validation["gpu_count"] > 0:
        validation["recommendations"].append(
            f"Utilize {validation['gpu_count']} available GPUs"
        )

    if validation["infiniband_available"]:
        validation["recommendations"].append(
            "InfiniBand detected - use for high-speed MPI communication"
        )

    return validation

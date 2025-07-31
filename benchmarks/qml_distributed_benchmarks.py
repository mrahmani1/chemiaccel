"""
Benchmarks and performance validation for distributed QML inference.

This module provides comprehensive benchmarking tools to validate the performance
and scalability of distributed QML inference systems.
"""

import json
import logging
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from chemi_accel.qml.base import BaseQMLModel, QMLPrediction, QuantumProperty
from chemi_accel.qml.distributed_inference import (
    DataTransferMode,
    DistributedConfig,
    DistributionBackend,
    create_distributed_engine,
)
from chemi_accel.qml.on_the_fly import GeometryInput


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""

    # Dataset parameters
    molecule_counts: List[int] = field(default_factory=lambda: [10, 50, 100, 500, 1000])
    molecule_complexities: List[str] = field(
        default_factory=lambda: ["simple", "medium", "complex"]
    )

    # System parameters
    worker_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    backends: List[str] = field(default_factory=lambda: ["multiprocessing", "dask"])

    # Performance parameters
    repetitions: int = 3
    warmup_runs: int = 1
    timeout_seconds: float = 300.0

    # Output parameters
    output_dir: str = "benchmark_results"
    save_plots: bool = True
    save_detailed_logs: bool = True

    # Hardware parameters
    measure_gpu_usage: bool = True
    measure_memory_usage: bool = True
    measure_cpu_usage: bool = True


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    # Configuration
    backend: str
    num_workers: int
    batch_size: int
    molecule_count: int
    molecule_complexity: str

    # Performance metrics
    total_time: float = 0.0
    throughput: float = 0.0  # molecules/second
    latency: float = 0.0  # seconds/molecule

    # Resource usage
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_gpu_percent: float = 0.0

    # Quality metrics
    success_rate: float = 1.0
    error_count: int = 0
    prediction_quality: float = 1.0

    # Scaling metrics
    parallel_efficiency: float = 1.0
    scaling_factor: float = 1.0

    # Metadata
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    system_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MockDistributedQMLModel(BaseQMLModel):
    """Mock QML model with configurable performance characteristics."""

    def __init__(self, complexity_factor: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.complexity_factor = complexity_factor
        self._is_loaded = True

    def load_model(self):
        self._is_loaded = True

    def predict(self, molecules, properties=None, **kwargs):
        # Simulate computation time based on complexity
        if isinstance(molecules, list):
            for mol in molecules:
                self._simulate_computation(mol)
            return [self._create_mock_prediction() for _ in molecules]
        else:
            self._simulate_computation(molecules)
            return self._create_mock_prediction()

    def batch_predict(self, molecules, properties=None, batch_size=32, **kwargs):
        # Simulate batch processing with efficiency gains
        batch_efficiency = min(1.0, len(molecules) / 32.0) * 0.8 + 0.2

        for mol in molecules:
            self._simulate_computation(mol, efficiency=batch_efficiency)

        return [self._create_mock_prediction() for _ in molecules]

    def get_supported_properties(self):
        return [
            QuantumProperty.TOTAL_ENERGY,
            QuantumProperty.HOMO_ENERGY,
            QuantumProperty.LUMO_ENERGY,
        ]

    def validate_molecule(self, molecule):
        return True

    def _simulate_computation(self, molecule, efficiency: float = 1.0):
        """Simulate computation time based on molecule complexity."""
        if isinstance(molecule, GeometryInput):
            atom_count = len(molecule.atomic_numbers)
        elif isinstance(molecule, str):
            atom_count = len(molecule) // 3  # Rough estimate from SMILES
        else:
            atom_count = 10  # Default

        # Computation time scales with atom count and complexity factor
        base_time = atom_count * 0.001 * self.complexity_factor / efficiency

        # Add some randomness
        computation_time = base_time * (0.8 + 0.4 * np.random.random())

        if computation_time > 0:
            time.sleep(computation_time)

    def _create_mock_prediction(self):
        return QMLPrediction(
            properties={
                QuantumProperty.TOTAL_ENERGY: np.random.normal(-100.0, 10.0),
                QuantumProperty.HOMO_ENERGY: np.random.normal(-8.0, 2.0),
                QuantumProperty.LUMO_ENERGY: np.random.normal(-2.0, 1.0),
            },
            confidence=0.85 + 0.1 * np.random.random(),
            model_name="MockDistributedModel",
        )


class SystemMonitor:
    """Monitor system resources during benchmarking."""

    def __init__(self, interval: float = 0.1):
        """
        Initialize system monitor.

        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self._metrics = {
            "cpu_percent": [],
            "memory_mb": [],
            "gpu_percent": [],
            "gpu_memory_mb": [],
            "timestamps": [],
        }

    def start_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring = True
        self._metrics = {key: [] for key in self._metrics.keys()}

        # Start monitoring in separate thread
        import threading

        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False

        if hasattr(self, "_monitor_thread"):
            self._monitor_thread.join(timeout=1.0)

        # Calculate summary statistics
        summary = {}

        if self._metrics["cpu_percent"]:
            summary["avg_cpu_percent"] = np.mean(self._metrics["cpu_percent"])
            summary["peak_cpu_percent"] = np.max(self._metrics["cpu_percent"])

        if self._metrics["memory_mb"]:
            summary["avg_memory_mb"] = np.mean(self._metrics["memory_mb"])
            summary["peak_memory_mb"] = np.max(self._metrics["memory_mb"])

        if self._metrics["gpu_percent"]:
            summary["avg_gpu_percent"] = np.mean(self._metrics["gpu_percent"])
            summary["peak_gpu_percent"] = np.max(self._metrics["gpu_percent"])

        if self._metrics["gpu_memory_mb"]:
            summary["avg_gpu_memory_mb"] = np.mean(self._metrics["gpu_memory_mb"])
            summary["peak_gpu_memory_mb"] = np.max(self._metrics["gpu_memory_mb"])

        summary["raw_metrics"] = self._metrics.copy()

        return summary

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            timestamp = time.time()

            # Monitor CPU
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024 * 1024)

                self._metrics["cpu_percent"].append(cpu_percent)
                self._metrics["memory_mb"].append(memory_mb)

            # Monitor GPU
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Monitor first GPU
                        self._metrics["gpu_percent"].append(gpu.load * 100)
                        self._metrics["gpu_memory_mb"].append(gpu.memoryUsed)
                except (ImportError, AttributeError, IndexError):
                    pass  # GPU monitoring failed

            self._metrics["timestamps"].append(timestamp)

            time.sleep(self.interval)


class DistributedQMLBenchmark:
    """Main benchmark runner for distributed QML inference."""

    def __init__(self, config: BenchmarkConfig = None):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.system_monitor = SystemMonitor()

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logging()

        logging.info(
            f"Initialized distributed QML benchmark with config: {self.config}"
        )

    def _setup_logging(self):
        """Set up logging for benchmark runs."""
        log_file = Path(self.config.output_dir) / "benchmark.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """
        Run complete benchmark suite.

        Returns:
            List of benchmark results
        """
        logging.info("Starting full benchmark suite")

        total_experiments = (
            len(self.config.molecule_counts)
            * len(self.config.molecule_complexities)
            * len(self.config.worker_counts)
            * len(self.config.batch_sizes)
            * len(self.config.backends)
            * self.config.repetitions
        )

        logging.info(f"Total experiments to run: {total_experiments}")

        experiment_count = 0

        for molecule_count in self.config.molecule_counts:
            for complexity in self.config.molecule_complexities:
                for num_workers in self.config.worker_counts:
                    for batch_size in self.config.batch_sizes:
                        for backend in self.config.backends:
                            for rep in range(self.config.repetitions):
                                experiment_count += 1

                                logging.info(
                                    f"Running experiment {experiment_count}/{total_experiments}: "
                                    f"molecules={molecule_count}, complexity={complexity}, "
                                    f"workers={num_workers}, batch_size={batch_size}, "
                                    f"backend={backend}, rep={rep+1}"
                                )

                                try:
                                    result = self._run_single_experiment(
                                        molecule_count=molecule_count,
                                        complexity=complexity,
                                        num_workers=num_workers,
                                        batch_size=batch_size,
                                        backend=backend,
                                    )

                                    self.results.append(result)

                                    logging.info(
                                        f"Experiment completed: throughput={result.throughput:.2f} mol/s, "
                                        f"latency={result.latency:.3f}s, efficiency={result.parallel_efficiency:.2f}"
                                    )

                                except Exception as e:
                                    logging.error(f"Experiment failed: {e}")

                                    # Create error result
                                    error_result = BenchmarkResult(
                                        backend=backend,
                                        num_workers=num_workers,
                                        batch_size=batch_size,
                                        molecule_count=molecule_count,
                                        molecule_complexity=complexity,
                                        success_rate=0.0,
                                        error_count=1,
                                    )

                                    self.results.append(error_result)

        logging.info("Full benchmark suite completed")

        # Save results
        self._save_results()

        # Generate analysis
        self._generate_analysis()

        return self.results

    def _run_single_experiment(
        self,
        molecule_count: int,
        complexity: str,
        num_workers: int,
        batch_size: int,
        backend: str,
    ) -> BenchmarkResult:
        """Run a single benchmark experiment."""

        # Generate test molecules
        test_molecules = self._generate_test_molecules(molecule_count, complexity)

        # Create model with appropriate complexity
        complexity_factors = {"simple": 0.5, "medium": 1.0, "complex": 2.0}
        model = MockDistributedQMLModel(
            complexity_factor=complexity_factors.get(complexity, 1.0)
        )

        # Configure distributed system
        config = DistributedConfig(
            backend=DistributionBackend(backend),
            num_workers=num_workers,
            batch_size=batch_size,
            max_batch_size=min(batch_size * 2, 256),
            data_transfer_mode=DataTransferMode.PIPELINE,
        )

        # Warmup runs
        if self.config.warmup_runs > 0:
            logging.debug(f"Running {self.config.warmup_runs} warmup runs")

            with create_distributed_engine(
                model, backend, num_workers
            ) as warmup_engine:
                for _ in range(self.config.warmup_runs):
                    try:
                        warmup_engine.predict_distributed(
                            test_molecules[: min(10, len(test_molecules))],
                            [QuantumProperty.TOTAL_ENERGY],
                            timeout=30.0,
                        )
                    except Exception as e:
                        logging.warning(f"Warmup run failed: {e}")

        # Start monitoring
        self.system_monitor.start_monitoring()

        # Run main experiment
        start_time = time.time()
        error_count = 0
        predictions = []

        try:
            with create_distributed_engine(
                model, backend, num_workers, batch_size=batch_size
            ) as engine:
                result = engine.predict_distributed(
                    test_molecules,
                    [QuantumProperty.TOTAL_ENERGY, QuantumProperty.HOMO_ENERGY],
                    timeout=self.config.timeout_seconds,
                )

                predictions = result.predictions
                error_count = len(result.errors)

        except Exception as e:
            logging.error(f"Experiment execution failed: {e}")
            error_count = molecule_count  # All molecules failed

        end_time = time.time()
        total_time = end_time - start_time

        # Stop monitoring
        monitoring_results = self.system_monitor.stop_monitoring()

        # Calculate metrics
        successful_predictions = len(predictions) - error_count
        success_rate = (
            successful_predictions / molecule_count if molecule_count > 0 else 0.0
        )

        throughput = successful_predictions / total_time if total_time > 0 else 0.0
        latency = total_time / molecule_count if molecule_count > 0 else 0.0

        # Calculate parallel efficiency (compared to single worker)
        baseline_time = self._estimate_baseline_time(molecule_count, complexity)
        ideal_parallel_time = baseline_time / num_workers
        parallel_efficiency = (
            ideal_parallel_time / total_time if total_time > 0 else 0.0
        )
        parallel_efficiency = min(parallel_efficiency, 1.0)  # Cap at 100%

        # Calculate scaling factor
        scaling_factor = baseline_time / total_time if total_time > 0 else 0.0

        # Create result
        result = BenchmarkResult(
            backend=backend,
            num_workers=num_workers,
            batch_size=batch_size,
            molecule_count=molecule_count,
            molecule_complexity=complexity,
            total_time=total_time,
            throughput=throughput,
            latency=latency,
            peak_memory_mb=monitoring_results.get("peak_memory_mb", 0.0),
            avg_cpu_percent=monitoring_results.get("avg_cpu_percent", 0.0),
            avg_gpu_percent=monitoring_results.get("avg_gpu_percent", 0.0),
            success_rate=success_rate,
            error_count=error_count,
            parallel_efficiency=parallel_efficiency,
            scaling_factor=scaling_factor,
            system_info=self._get_system_info(),
        )

        return result

    def _generate_test_molecules(
        self, count: int, complexity: str
    ) -> List[GeometryInput]:
        """Generate test molecules with specified complexity."""
        molecules = []

        complexity_params = {
            "simple": {"min_atoms": 3, "max_atoms": 8, "heavy_atom_fraction": 0.3},
            "medium": {"min_atoms": 8, "max_atoms": 20, "heavy_atom_fraction": 0.5},
            "complex": {"min_atoms": 20, "max_atoms": 50, "heavy_atom_fraction": 0.7},
        }

        params = complexity_params.get(complexity, complexity_params["medium"])

        for i in range(count):
            num_atoms = np.random.randint(params["min_atoms"], params["max_atoms"] + 1)

            # Generate atomic numbers
            heavy_atoms = int(num_atoms * params["heavy_atom_fraction"])
            hydrogen_atoms = num_atoms - heavy_atoms

            # Heavy atoms (C, N, O, S)
            heavy_atom_types = np.random.choice(
                [6, 7, 8, 16], size=heavy_atoms, p=[0.5, 0.2, 0.2, 0.1]
            )
            hydrogen_types = np.ones(hydrogen_atoms, dtype=int)

            atomic_numbers = np.concatenate([heavy_atom_types, hydrogen_types])
            np.random.shuffle(atomic_numbers)

            # Generate random coordinates
            coordinates = np.random.randn(num_atoms, 3) * 2.0

            # Add some structure for realism
            if num_atoms > 1:
                # Create some bonds by placing atoms closer together
                for j in range(1, min(num_atoms, heavy_atoms + 1)):
                    bond_length = 1.5 + 0.5 * np.random.random()
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    coordinates[j] = coordinates[j - 1] + direction * bond_length

            molecule = GeometryInput(
                coordinates=coordinates,
                atomic_numbers=atomic_numbers,
                charge=0,
                multiplicity=1,
            )

            molecules.append(molecule)

        return molecules

    def _estimate_baseline_time(self, molecule_count: int, complexity: str) -> float:
        """Estimate baseline single-worker execution time."""
        complexity_factors = {"simple": 0.5, "medium": 1.0, "complex": 2.0}
        base_time_per_molecule = 0.01  # 10ms per molecule baseline

        complexity_factor = complexity_factors.get(complexity, 1.0)
        baseline_time = molecule_count * base_time_per_molecule * complexity_factor

        return baseline_time

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        info = {
            "python_version": f"{mp.sys.version_info.major}.{mp.sys.version_info.minor}",
            "cpu_count": mp.cpu_count(),
        }

        if PSUTIL_AVAILABLE:
            info.update(
                {
                    "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                    "cpu_freq_mhz": (
                        psutil.cpu_freq().current if psutil.cpu_freq() else 0
                    ),
                }
            )

        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info.update(
                        {"gpu_name": gpu.name, "gpu_memory_gb": gpu.memoryTotal / 1024}
                    )
            except (ImportError, AttributeError, IndexError):
                pass

        return info

    def _save_results(self):
        """Save benchmark results to files."""
        # Save as JSON
        json_file = Path(self.config.output_dir) / "benchmark_results.json"
        results_data = [result.to_dict() for result in self.results]

        with open(json_file, "w") as f:
            json.dump(results_data, f, indent=2)

        # Save as CSV
        csv_file = Path(self.config.output_dir) / "benchmark_results.csv"
        df = pd.DataFrame(results_data)
        df.to_csv(csv_file, index=False)

        logging.info(f"Results saved to {json_file} and {csv_file}")

    def _generate_analysis(self):
        """Generate analysis and visualizations."""
        if not self.results:
            logging.warning("No results to analyze")
            return

        # Convert to DataFrame for analysis
        df = pd.DataFrame([result.to_dict() for result in self.results])

        # Generate plots
        if self.config.save_plots:
            self._create_performance_plots(df)
            self._create_scaling_plots(df)
            self._create_resource_usage_plots(df)

        # Generate summary statistics
        self._generate_summary_statistics(df)

    def _create_performance_plots(self, df: pd.DataFrame):
        """Create performance visualization plots."""
        # Set up the plotting style
        plt.style.use("seaborn-v0_8")

        # Throughput vs. number of workers
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Throughput vs Workers
        for backend in df["backend"].unique():
            backend_data = df[df["backend"] == backend]
            grouped = backend_data.groupby("num_workers")["throughput"].mean()
            axes[0, 0].plot(grouped.index, grouped.values, marker="o", label=backend)

        axes[0, 0].set_xlabel("Number of Workers")
        axes[0, 0].set_ylabel("Throughput (molecules/sec)")
        axes[0, 0].set_title("Throughput vs Number of Workers")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot 2: Latency vs Batch Size
        for complexity in df["molecule_complexity"].unique():
            complexity_data = df[df["molecule_complexity"] == complexity]
            grouped = complexity_data.groupby("batch_size")["latency"].mean()
            axes[0, 1].plot(grouped.index, grouped.values, marker="s", label=complexity)

        axes[0, 1].set_xlabel("Batch Size")
        axes[0, 1].set_ylabel("Latency (seconds/molecule)")
        axes[0, 1].set_title("Latency vs Batch Size")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot 3: Parallel Efficiency
        for backend in df["backend"].unique():
            backend_data = df[df["backend"] == backend]
            grouped = backend_data.groupby("num_workers")["parallel_efficiency"].mean()
            axes[1, 0].plot(grouped.index, grouped.values, marker="^", label=backend)

        axes[1, 0].set_xlabel("Number of Workers")
        axes[1, 0].set_ylabel("Parallel Efficiency")
        axes[1, 0].set_title("Parallel Efficiency vs Number of Workers")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim(0, 1.1)

        # Plot 4: Success Rate
        success_by_complexity = df.groupby("molecule_complexity")["success_rate"].mean()
        axes[1, 1].bar(success_by_complexity.index, success_by_complexity.values)
        axes[1, 1].set_xlabel("Molecule Complexity")
        axes[1, 1].set_ylabel("Success Rate")
        axes[1, 1].set_title("Success Rate by Complexity")
        axes[1, 1].set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(
            Path(self.config.output_dir) / "performance_plots.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logging.info("Performance plots saved")

    def _create_scaling_plots(self, df: pd.DataFrame):
        """Create scaling analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Strong scaling (fixed problem size, varying workers)
        for molecule_count in [100, 500, 1000]:
            if molecule_count in df["molecule_count"].unique():
                subset = df[df["molecule_count"] == molecule_count]
                grouped = subset.groupby("num_workers")["throughput"].mean()
                axes[0, 0].plot(
                    grouped.index,
                    grouped.values,
                    marker="o",
                    label=f"{molecule_count} molecules",
                )

        axes[0, 0].set_xlabel("Number of Workers")
        axes[0, 0].set_ylabel("Throughput (molecules/sec)")
        axes[0, 0].set_title("Strong Scaling Analysis")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Weak scaling (proportional problem size increase)
        base_molecules_per_worker = 100
        weak_scaling_data = []

        for num_workers in sorted(df["num_workers"].unique()):
            target_molecules = base_molecules_per_worker * num_workers
            # Find closest molecule count in data
            closest_count = min(
                df["molecule_count"].unique(), key=lambda x: abs(x - target_molecules)
            )

            if (
                abs(closest_count - target_molecules) <= target_molecules * 0.2
            ):  # Within 20%
                subset = df[
                    (df["num_workers"] == num_workers)
                    & (df["molecule_count"] == closest_count)
                ]
                if not subset.empty:
                    avg_throughput = subset["throughput"].mean()
                    weak_scaling_data.append((num_workers, avg_throughput))

        if weak_scaling_data:
            workers, throughputs = zip(*weak_scaling_data)
            axes[0, 1].plot(workers, throughputs, marker="s", color="red", linewidth=2)

            # Add ideal scaling line
            ideal_throughput = throughputs[0]
            ideal_line = [ideal_throughput] * len(workers)
            axes[0, 1].plot(workers, ideal_line, "--", color="gray", label="Ideal")

        axes[0, 1].set_xlabel("Number of Workers")
        axes[0, 1].set_ylabel("Throughput (molecules/sec)")
        axes[0, 1].set_title("Weak Scaling Analysis")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Scaling efficiency heatmap
        pivot_data = df.pivot_table(
            values="parallel_efficiency",
            index="num_workers",
            columns="batch_size",
            aggfunc="mean",
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            ax=axes[1, 0],
            cbar_kws={"label": "Parallel Efficiency"},
        )
        axes[1, 0].set_title("Parallel Efficiency Heatmap")

        # Throughput vs Dataset Size
        for backend in df["backend"].unique():
            backend_data = df[df["backend"] == backend]
            grouped = backend_data.groupby("molecule_count")["throughput"].mean()
            axes[1, 1].plot(grouped.index, grouped.values, marker="o", label=backend)

        axes[1, 1].set_xlabel("Dataset Size (molecules)")
        axes[1, 1].set_ylabel("Throughput (molecules/sec)")
        axes[1, 1].set_title("Throughput vs Dataset Size")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_xscale("log")

        plt.tight_layout()
        plt.savefig(
            Path(self.config.output_dir) / "scaling_plots.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logging.info("Scaling plots saved")

    def _create_resource_usage_plots(self, df: pd.DataFrame):
        """Create resource usage analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Memory usage vs dataset size
        for backend in df["backend"].unique():
            backend_data = df[df["backend"] == backend]
            grouped = backend_data.groupby("molecule_count")["peak_memory_mb"].mean()
            axes[0, 0].plot(grouped.index, grouped.values, marker="o", label=backend)

        axes[0, 0].set_xlabel("Dataset Size (molecules)")
        axes[0, 0].set_ylabel("Peak Memory (MB)")
        axes[0, 0].set_title("Memory Usage vs Dataset Size")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # CPU usage vs number of workers
        for backend in df["backend"].unique():
            backend_data = df[df["backend"] == backend]
            grouped = backend_data.groupby("num_workers")["avg_cpu_percent"].mean()
            axes[0, 1].plot(grouped.index, grouped.values, marker="s", label=backend)

        axes[0, 1].set_xlabel("Number of Workers")
        axes[0, 1].set_ylabel("Average CPU Usage (%)")
        axes[0, 1].set_title("CPU Usage vs Number of Workers")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Resource efficiency (throughput per MB)
        df["resource_efficiency"] = df["throughput"] / (
            df["peak_memory_mb"] + 1
        )  # +1 to avoid division by zero

        for complexity in df["molecule_complexity"].unique():
            complexity_data = df[df["molecule_complexity"] == complexity]
            grouped = complexity_data.groupby("batch_size")[
                "resource_efficiency"
            ].mean()
            axes[1, 0].plot(grouped.index, grouped.values, marker="^", label=complexity)

        axes[1, 0].set_xlabel("Batch Size")
        axes[1, 0].set_ylabel("Resource Efficiency (throughput/MB)")
        axes[1, 0].set_title("Resource Efficiency vs Batch Size")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # GPU usage (if available)
        gpu_data = df[df["avg_gpu_percent"] > 0]
        if not gpu_data.empty:
            for backend in gpu_data["backend"].unique():
                backend_data = gpu_data[gpu_data["backend"] == backend]
                grouped = backend_data.groupby("num_workers")["avg_gpu_percent"].mean()
                axes[1, 1].plot(
                    grouped.index, grouped.values, marker="d", label=backend
                )

            axes[1, 1].set_xlabel("Number of Workers")
            axes[1, 1].set_ylabel("Average GPU Usage (%)")
            axes[1, 1].set_title("GPU Usage vs Number of Workers")
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No GPU usage data available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("GPU Usage (No Data)")

        plt.tight_layout()
        plt.savefig(
            Path(self.config.output_dir) / "resource_usage_plots.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logging.info("Resource usage plots saved")

    def _generate_summary_statistics(self, df: pd.DataFrame):
        """Generate summary statistics report."""
        summary_file = Path(self.config.output_dir) / "benchmark_summary.txt"

        with open(summary_file, "w") as f:
            f.write("DISTRIBUTED QML BENCHMARK SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            # Overall statistics
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total experiments: {len(df)}\n")
            f.write(
                f"Average throughput: {df['throughput'].mean():.2f} molecules/sec\n"
            )
            f.write(f"Max throughput: {df['throughput'].max():.2f} molecules/sec\n")
            f.write(f"Average latency: {df['latency'].mean():.3f} sec/molecule\n")
            f.write(f"Min latency: {df['latency'].min():.3f} sec/molecule\n")
            f.write(f"Average success rate: {df['success_rate'].mean():.3f}\n")
            f.write(
                f"Average parallel efficiency: {df['parallel_efficiency'].mean():.3f}\n\n"
            )

            # Performance by backend
            f.write("PERFORMANCE BY BACKEND\n")
            f.write("-" * 25 + "\n")
            backend_stats = (
                df.groupby("backend")
                .agg(
                    {
                        "throughput": ["mean", "max"],
                        "latency": ["mean", "min"],
                        "parallel_efficiency": "mean",
                        "success_rate": "mean",
                    }
                )
                .round(3)
            )
            f.write(str(backend_stats))
            f.write("\n\n")

            # Scaling analysis
            f.write("SCALING ANALYSIS\n")
            f.write("-" * 16 + "\n")

            # Best configurations
            best_throughput = df.loc[df["throughput"].idxmax()]
            best_efficiency = df.loc[df["parallel_efficiency"].idxmax()]

            f.write("Best throughput configuration:\n")
            f.write(f"  Backend: {best_throughput['backend']}\n")
            f.write(f"  Workers: {best_throughput['num_workers']}\n")
            f.write(f"  Batch size: {best_throughput['batch_size']}\n")
            f.write(
                f"  Throughput: {best_throughput['throughput']:.2f} molecules/sec\n\n"
            )

            f.write("Best efficiency configuration:\n")
            f.write(f"  Backend: {best_efficiency['backend']}\n")
            f.write(f"  Workers: {best_efficiency['num_workers']}\n")
            f.write(f"  Batch size: {best_efficiency['batch_size']}\n")
            f.write(f"  Efficiency: {best_efficiency['parallel_efficiency']:.3f}\n\n")

            # Resource usage
            f.write("RESOURCE USAGE\n")
            f.write("-" * 14 + "\n")
            f.write(f"Average peak memory: {df['peak_memory_mb'].mean():.1f} MB\n")
            f.write(f"Max peak memory: {df['peak_memory_mb'].max():.1f} MB\n")
            f.write(f"Average CPU usage: {df['avg_cpu_percent'].mean():.1f}%\n")

            if df["avg_gpu_percent"].sum() > 0:
                f.write(f"Average GPU usage: {df['avg_gpu_percent'].mean():.1f}%\n")

            f.write("\n")

            # System information
            if self.results:
                system_info = self.results[0].system_info
                f.write("SYSTEM INFORMATION\n")
                f.write("-" * 18 + "\n")
                for key, value in system_info.items():
                    f.write(f"{key}: {value}\n")

        logging.info(f"Summary statistics saved to {summary_file}")


def run_quick_benchmark() -> List[BenchmarkResult]:
    """Run a quick benchmark for testing."""
    config = BenchmarkConfig(
        molecule_counts=[10, 50],
        molecule_complexities=["simple", "medium"],
        worker_counts=[1, 2],
        batch_sizes=[16, 32],
        backends=["multiprocessing"],
        repetitions=2,
        warmup_runs=1,
        output_dir="quick_benchmark_results",
    )

    benchmark = DistributedQMLBenchmark(config)
    return benchmark.run_full_benchmark()


def run_scaling_benchmark() -> List[BenchmarkResult]:
    """Run a comprehensive scaling benchmark."""
    config = BenchmarkConfig(
        molecule_counts=[100, 500, 1000, 2000],
        molecule_complexities=["simple", "medium", "complex"],
        worker_counts=[1, 2, 4, 8, 16],
        batch_sizes=[16, 32, 64, 128, 256],
        backends=["multiprocessing", "dask"],
        repetitions=3,
        warmup_runs=2,
        timeout_seconds=600.0,
        output_dir="scaling_benchmark_results",
    )

    benchmark = DistributedQMLBenchmark(config)
    return benchmark.run_full_benchmark()


def run_performance_regression_test() -> bool:
    """Run performance regression test."""
    # Quick benchmark to check for performance regressions
    results = run_quick_benchmark()

    # Define minimum acceptable performance thresholds
    min_throughput = 10.0  # molecules/sec
    min_success_rate = 0.95
    min_parallel_efficiency = 0.3

    # Check results
    passed_tests = 0
    total_tests = 0

    for result in results:
        total_tests += 1

        if (
            result.throughput >= min_throughput
            and result.success_rate >= min_success_rate
            and result.parallel_efficiency >= min_parallel_efficiency
        ):
            passed_tests += 1
        else:
            logging.warning(
                f"Performance regression detected: "
                f"throughput={result.throughput:.2f}, "
                f"success_rate={result.success_rate:.2f}, "
                f"efficiency={result.parallel_efficiency:.2f}"
            )

    success_rate = passed_tests / total_tests if total_tests > 0 else 0.0

    logging.info(
        f"Performance regression test: {passed_tests}/{total_tests} passed ({success_rate:.1%})"
    )

    return success_rate >= 0.8  # 80% of tests must pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run distributed QML benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")
    parser.add_argument("--regression", action="store_true", help="Run regression test")
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Output directory"
    )

    args = parser.parse_args()

    if args.quick:
        print("Running quick benchmark...")
        results = run_quick_benchmark()
        print(f"Quick benchmark completed with {len(results)} results")

    elif args.scaling:
        print("Running scaling benchmark...")
        results = run_scaling_benchmark()
        print(f"Scaling benchmark completed with {len(results)} results")

    elif args.regression:
        print("Running performance regression test...")
        passed = run_performance_regression_test()
        exit_code = 0 if passed else 1
        print(f"Regression test {'PASSED' if passed else 'FAILED'}")
        exit(exit_code)

    else:
        print("Running full benchmark suite...")
        config = BenchmarkConfig(output_dir=args.output_dir)
        benchmark = DistributedQMLBenchmark(config)
        results = benchmark.run_full_benchmark()
        print(f"Full benchmark completed with {len(results)} results")

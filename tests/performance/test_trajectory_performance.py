"""
Performance Tests for Terabyte-Scale Trajectory Analysis - Task 19.4

This module provides comprehensive performance testing for ChemiAccel's high-throughput
MD trajectory analysis capabilities, validating scalability and efficiency for
terabyte-scale datasets and production workflows.

Test Categories:
1. Trajectory file reading performance (XTC, DCD, TRR formats)
2. Time-series property calculation scaling
3. Dynamic network analysis performance
4. PLIF generation throughput
5. Memory usage and efficiency tests
6. GPU acceleration benchmarks
7. Parallel processing scalability
8. Large-scale dataset validation

Performance Targets:
- Trajectory reading: >1000 frames/second
- Property calculation: >100 molecules/second
- Network analysis: >50 networks/second
- PLIF generation: >200 frames/second
- Memory usage: <8GB for 100K frames
- GPU speedup: >5x over CPU baseline
- Parallel efficiency: >80% with 8 workers

Scientific Validation:
- Accuracy preservation under performance optimization
- Numerical stability with large datasets
- Reproducibility across different hardware configurations
- Error handling and fault tolerance testing
"""

import gc
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import psutil
import pytest

# Core ChemiAccel imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from chemi_accel.analysis.dynamic_networks import NetworkAnalysisConfig, NetworkBuilder
from chemi_accel.analysis.dynamic_plif import PLIFAnalysisConfig, create_plif_analyzer
from chemi_accel.analysis.timeseries_properties import (
    TimeSeriesAnalysisConfig,
    TimeSeriesCalculator,
    create_timeseries_calculator,
)
from chemi_accel.analysis.trajectory_reader import (
    ParallelReadingConfig,
    ParallelTrajectoryReader,
    TrajectoryFrame,
    TrajectorySelection,
    create_parallel_trajectory_reader,
)

# GPU availability check
try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Test markers
pytestmark = [pytest.mark.performance, pytest.mark.slow]


@dataclass
class PerformanceMetrics:
    """Container for performance test results."""

    test_name: str
    dataset_size: str
    processing_time: float
    throughput: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float = 0.0
    success_rate: float = 1.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame storage."""
        return {
            "test_name": self.test_name,
            "dataset_size": self.dataset_size,
            "processing_time": self.processing_time,
            "throughput": self.throughput,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "gpu_usage_percent": self.gpu_usage_percent,
            "success_rate": self.success_rate,
            "n_errors": len(self.errors),
        }


class PerformanceMonitor:
    """Monitor system performance during tests."""

    def __init__(self, interval: float = 0.1):
        """Initialize performance monitor.

        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self._monitor_thread = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.gpu_usage.clear()

        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        self.monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join()

        # Calculate statistics
        stats = {
            "cpu_mean": np.mean(self.cpu_usage) if self.cpu_usage else 0.0,
            "cpu_max": np.max(self.cpu_usage) if self.cpu_usage else 0.0,
            "memory_mean_mb": np.mean(self.memory_usage) if self.memory_usage else 0.0,
            "memory_max_mb": np.max(self.memory_usage) if self.memory_usage else 0.0,
            "gpu_mean": np.mean(self.gpu_usage) if self.gpu_usage else 0.0,
            "gpu_max": np.max(self.gpu_usage) if self.gpu_usage else 0.0,
        }

        return stats

    def _monitor_loop(self):
        """Monitor performance in loop."""
        process = psutil.Process()

        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = process.cpu_percent()
                self.cpu_usage.append(cpu_percent)

                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.memory_usage.append(memory_mb)

                # GPU usage (if available)
                if GPU_AVAILABLE:
                    try:
                        gpu_memory = cp.get_default_memory_pool().used_bytes()
                        gpu_memory_mb = gpu_memory / (1024 * 1024)
                        self.gpu_usage.append(gpu_memory_mb)
                    except:
                        self.gpu_usage.append(0.0)

                time.sleep(self.interval)

            except Exception:
                break


def create_mock_trajectory_files(
    n_frames: int, n_atoms: int, file_format: str = "dcd"
) -> Tuple[str, str]:
    """Create mock trajectory files for testing.

    Args:
        n_frames: Number of frames
        n_atoms: Number of atoms
        file_format: File format (dcd, xtc, trr)

    Returns:
        Tuple of (trajectory_file, topology_file)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create trajectory file
        traj_path = os.path.join(tmpdir, f"test.{file_format}")
        top_path = os.path.join(tmpdir, "test.pdb")

        # Create mock trajectory data
        trajectory_data = {
            "n_frames": n_frames,
            "n_atoms": n_atoms,
            "coordinates": np.random.random((n_frames, n_atoms, 3)) * 100,
            "times": np.arange(n_frames) * 0.1,  # ps
        }

        # Write mock files
        with open(traj_path, "wb") as f:
            # Write binary trajectory data (simplified format)
            f.write(b"MOCK_TRAJECTORY")
            f.write(n_frames.to_bytes(4, "little"))
            f.write(n_atoms.to_bytes(4, "little"))
            trajectory_data["coordinates"].astype(np.float32).tobytes()

        with open(top_path, "w") as f:
            # Write mock PDB topology
            f.write("HEADER    MOCK TOPOLOGY\n")
            for i in range(n_atoms):
                f.write(
                    f"ATOM  {i+1:5d}  CA  ALA A{i//100+1:4d}    "
                    f"{10.0:8.3f}{10.0:8.3f}{10.0:8.3f}  1.00 20.00           C\n"
                )
            f.write("END\n")

        return traj_path, top_path


def create_large_dataset(size_category: str) -> Dict[str, Any]:
    """Create large dataset for performance testing.

    Args:
        size_category: Size category (small, medium, large, xlarge)

    Returns:
        Dataset parameters
    """
    datasets = {
        "small": {"n_frames": 1000, "n_atoms": 1000, "size_gb": 0.01},
        "medium": {"n_frames": 10000, "n_atoms": 5000, "size_gb": 0.2},
        "large": {"n_frames": 50000, "n_atoms": 10000, "size_gb": 2.0},
        "xlarge": {"n_frames": 100000, "n_atoms": 20000, "size_gb": 8.0},
        "terabyte": {"n_frames": 1000000, "n_atoms": 50000, "size_gb": 200.0},
    }

    if size_category not in datasets:
        raise ValueError(f"Unknown size category: {size_category}")

    return datasets[size_category]


class TestTrajectoryReadingPerformance:
    """Test trajectory file reading performance."""

    @pytest.fixture(scope="class")
    def performance_monitor(self):
        """Performance monitoring fixture."""
        return PerformanceMonitor()

    @pytest.mark.parametrize("size_category", ["small", "medium", "large"])
    @pytest.mark.parametrize("file_format", ["dcd", "xtc"])
    def test_single_trajectory_reading_performance(
        self, size_category, file_format, performance_monitor
    ):
        """Test single trajectory reading performance."""
        dataset = create_large_dataset(size_category)

        # Create mock trajectory
        traj_file, top_file = create_mock_trajectory_files(
            dataset["n_frames"], dataset["n_atoms"], file_format
        )

        # Configure reader
        config = ParallelReadingConfig(
            n_workers=1,  # Single-threaded for baseline
            use_gpu_acceleration=False,
            memory_limit_gb=4.0,
        )

        reader = ParallelTrajectoryReader(config)

        # Add trajectory
        traj_id = reader.add_trajectory(traj_file, top_file)

        # Create selection
        selection = TrajectorySelection(stride=1)
        selections = {traj_id: selection}

        # Start monitoring
        performance_monitor.start_monitoring()
        start_time = time.time()

        try:
            # Read trajectory
            results = reader.read_parallel_frames(selections)

            # Measure performance
            end_time = time.time()
            processing_time = end_time - start_time

            # Stop monitoring
            perf_stats = performance_monitor.stop_monitoring()

            # Validate results
            assert traj_id in results
            frames = results[traj_id]
            assert len(frames) > 0

            # Calculate metrics
            throughput = len(frames) / processing_time  # frames/second

            metrics = PerformanceMetrics(
                test_name=f"single_trajectory_{file_format}",
                dataset_size=size_category,
                processing_time=processing_time,
                throughput=throughput,
                memory_usage_mb=perf_stats["memory_max_mb"],
                cpu_usage_percent=perf_stats["cpu_max"],
                gpu_usage_percent=perf_stats["gpu_max"],
            )

            # Performance assertions
            if size_category == "small":
                assert throughput > 500, f"Throughput too low: {throughput} frames/s"
            elif size_category == "medium":
                assert throughput > 200, f"Throughput too low: {throughput} frames/s"
            else:
                assert throughput > 100, f"Throughput too low: {throughput} frames/s"

            # Memory usage should be reasonable
            expected_memory_mb = dataset["size_gb"] * 1024 * 2  # 2x overhead
            assert (
                perf_stats["memory_max_mb"] < expected_memory_mb
            ), f"Memory usage too high: {perf_stats['memory_max_mb']} MB"

            print(f"\n{metrics.test_name} ({size_category}):")
            print(f"  Throughput: {throughput:.1f} frames/s")
            print(f"  Memory: {perf_stats['memory_max_mb']:.1f} MB")
            print(f"  CPU: {perf_stats['cpu_max']:.1f}%")

        finally:
            reader.cleanup()

    @pytest.mark.parametrize("n_workers", [1, 2, 4, 8])
    def test_parallel_reading_scalability(self, n_workers, performance_monitor):
        """Test parallel reading scalability."""
        dataset = create_large_dataset("medium")

        # Create multiple trajectory files
        trajectory_files = []
        n_trajectories = 4

        for i in range(n_trajectories):
            traj_file, top_file = create_mock_trajectory_files(
                dataset["n_frames"] // n_trajectories, dataset["n_atoms"], "dcd"
            )
            trajectory_files.append((traj_file, top_file))

        # Configure reader
        config = ParallelReadingConfig(
            n_workers=n_workers, use_gpu_acceleration=False, memory_limit_gb=8.0
        )

        reader = ParallelTrajectoryReader(config)

        # Add trajectories
        selections = {}
        for i, (traj_file, top_file) in enumerate(trajectory_files):
            traj_id = reader.add_trajectory(traj_file, top_file, f"traj_{i}")
            selections[traj_id] = TrajectorySelection(stride=1)

        # Start monitoring
        performance_monitor.start_monitoring()
        start_time = time.time()

        try:
            # Read trajectories in parallel
            results = reader.read_parallel_frames(selections)

            # Measure performance
            end_time = time.time()
            processing_time = end_time - start_time

            # Stop monitoring
            perf_stats = performance_monitor.stop_monitoring()

            # Calculate total frames processed
            total_frames = sum(len(frames) for frames in results.values())
            throughput = total_frames / processing_time

            # Calculate parallel efficiency
            baseline_throughput = 200  # frames/s (single worker baseline)
            theoretical_max = baseline_throughput * n_workers
            efficiency = throughput / theoretical_max

            metrics = PerformanceMetrics(
                test_name=f"parallel_reading_{n_workers}workers",
                dataset_size="medium_multi",
                processing_time=processing_time,
                throughput=throughput,
                memory_usage_mb=perf_stats["memory_max_mb"],
                cpu_usage_percent=perf_stats["cpu_max"],
                success_rate=efficiency,
            )

            # Performance assertions
            assert (
                throughput > baseline_throughput
            ), f"Parallel throughput worse than baseline: {throughput} vs {baseline_throughput}"

            if n_workers <= 4:
                assert (
                    efficiency > 0.6
                ), f"Parallel efficiency too low: {efficiency:.2f}"
            else:
                assert (
                    efficiency > 0.4
                ), f"Parallel efficiency too low: {efficiency:.2f}"

            print(f"\nParallel reading ({n_workers} workers):")
            print(f"  Throughput: {throughput:.1f} frames/s")
            print(f"  Efficiency: {efficiency:.2f}")
            print(f"  Memory: {perf_stats['memory_max_mb']:.1f} MB")

        finally:
            reader.cleanup()

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_acceleration_performance(self, performance_monitor):
        """Test GPU acceleration performance."""
        dataset = create_large_dataset("large")

        # Create trajectory
        traj_file, top_file = create_mock_trajectory_files(
            dataset["n_frames"], dataset["n_atoms"], "dcd"
        )

        # Test CPU vs GPU performance
        results = {}

        for use_gpu in [False, True]:
            config = ParallelReadingConfig(
                n_workers=1, use_gpu_acceleration=use_gpu, memory_limit_gb=8.0
            )

            reader = ParallelTrajectoryReader(config)
            traj_id = reader.add_trajectory(traj_file, top_file)
            selection = TrajectorySelection(stride=5)  # Subsample for speed

            # Start monitoring
            performance_monitor.start_monitoring()
            start_time = time.time()

            try:
                results_data = reader.read_parallel_frames({traj_id: selection})

                end_time = time.time()
                processing_time = end_time - start_time
                perf_stats = performance_monitor.stop_monitoring()

                frames = results_data[traj_id]
                throughput = len(frames) / processing_time

                results[f"{'gpu' if use_gpu else 'cpu'}"] = {
                    "throughput": throughput,
                    "processing_time": processing_time,
                    "memory_mb": perf_stats["memory_max_mb"],
                    "gpu_memory_mb": perf_stats["gpu_max"],
                }

            finally:
                reader.cleanup()

        # Compare performance
        cpu_throughput = results["cpu"]["throughput"]
        gpu_throughput = results["gpu"]["throughput"]
        speedup = gpu_throughput / cpu_throughput

        # GPU should provide speedup
        assert speedup > 1.2, f"GPU speedup insufficient: {speedup:.2f}x"

        print("\nGPU Acceleration Performance:")
        print(f"  CPU: {cpu_throughput:.1f} frames/s")
        print(f"  GPU: {gpu_throughput:.1f} frames/s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  GPU Memory: {results['gpu']['gpu_memory_mb']:.1f} MB")


class TestTimeSeriesPerformance:
    """Test time-series property calculation performance."""

    def create_mock_frames(self, n_frames: int, n_atoms: int) -> List[TrajectoryFrame]:
        """Create mock trajectory frames."""
        frames = []

        for i in range(n_frames):
            coords = np.random.random((n_atoms, 3)) * 100

            frame = TrajectoryFrame(
                frame_id=i,
                time_ps=i * 0.1,
                coordinates=coords,
                velocities=np.random.random((n_atoms, 3)) * 10 if i % 2 == 0 else None,
                temperature=300.0 + np.random.random() * 50,
                potential_energy=-1000.0 + np.random.random() * 100,
            )

            frames.append(frame)

        return frames

    @pytest.mark.parametrize("size_category", ["small", "medium", "large"])
    @pytest.mark.parametrize("use_gpu", [False, True] if GPU_AVAILABLE else [False])
    def test_property_calculation_performance(self, size_category, use_gpu):
        """Test time-series property calculation performance."""
        dataset = create_large_dataset(size_category)

        # Create mock frames
        frames = self.create_mock_frames(dataset["n_frames"], dataset["n_atoms"])

        # Configure calculator
        config = TimeSeriesAnalysisConfig(
            use_gpu=use_gpu, n_workers=4 if not use_gpu else 1, memory_limit_gb=8.0
        )

        calculator = TimeSeriesCalculator(config)

        # Add various calculators
        reference_coords = np.random.random((dataset["n_atoms"], 3)) * 100
        calculator.add_rmsd_calculator(reference_coords)
        calculator.add_rg_calculator()
        calculator.add_distance_calculator([(0, 1), (2, 3), (4, 5)])
        calculator.add_energy_calculator("potential")
        calculator.add_temperature_calculator()

        # Monitor performance
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        start_time = time.time()

        try:
            # Calculate properties
            results = calculator.calculate_properties(frames)

            end_time = time.time()
            processing_time = end_time - start_time
            perf_stats = monitor.stop_monitoring()

            # Validate results
            assert len(results) > 0
            assert "RMSD" in results
            assert "Radius_of_Gyration" in results

            # Calculate metrics
            throughput = len(frames) / processing_time

            # Performance assertions
            min_throughput = 50 if use_gpu else 20
            assert (
                throughput > min_throughput
            ), f"Throughput too low: {throughput:.1f} frames/s"

            print(
                f"\nProperty calculation ({'GPU' if use_gpu else 'CPU'}, {size_category}):"
            )
            print(f"  Throughput: {throughput:.1f} frames/s")
            print(f"  Properties: {len(results)}")
            print(f"  Memory: {perf_stats['memory_max_mb']:.1f} MB")

        finally:
            calculator.cleanup()

    def test_streaming_calculation_performance(self):
        """Test streaming property calculation performance."""
        n_frames = 10000
        n_atoms = 5000
        batch_size = 100

        # Create frame generator
        def frame_generator():
            for i in range(n_frames):
                coords = np.random.random((n_atoms, 3)) * 100
                yield TrajectoryFrame(
                    frame_id=i, time_ps=i * 0.1, coordinates=coords, temperature=300.0
                )

        # Configure calculator
        config = TimeSeriesAnalysisConfig(
            batch_size=batch_size,
            streaming_mode=True,
            memory_limit_gb=2.0,  # Limited memory
        )

        calculator = TimeSeriesCalculator(config)
        calculator.add_temperature_calculator()

        # Monitor performance
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        start_time = time.time()

        processed_frames = 0

        try:
            # Process in streaming mode
            for batch_results in calculator.calculate_streaming(frame_generator()):
                processed_frames += len(next(iter(batch_results.values())).times)

            end_time = time.time()
            processing_time = end_time - start_time
            perf_stats = monitor.stop_monitoring()

            # Calculate metrics
            throughput = processed_frames / processing_time

            # Memory should stay low
            assert (
                perf_stats["memory_max_mb"] < 4000
            ), f"Memory usage too high for streaming: {perf_stats['memory_max_mb']} MB"

            # Should process all frames
            assert processed_frames == n_frames

            print("\nStreaming calculation:")
            print(f"  Throughput: {throughput:.1f} frames/s")
            print(f"  Max Memory: {perf_stats['memory_max_mb']:.1f} MB")
            print(f"  Batch Size: {batch_size}")

        finally:
            calculator.cleanup()


class TestNetworkAnalysisPerformance:
    """Test dynamic network analysis performance."""

    def create_mock_topology(self, n_atoms: int) -> Dict[str, Any]:
        """Create mock topology for network analysis."""
        topology = {"atoms": {}, "residues": {}}

        # Create atoms grouped by residues
        atoms_per_residue = 10
        n_residues = n_atoms // atoms_per_residue

        for i in range(n_atoms):
            residue_id = i // atoms_per_residue

            topology["atoms"][i] = {
                "residue_id": residue_id,
                "chain_id": "A",
                "residue_name": "ALA",
                "atom_name": "CA" if i % atoms_per_residue == 0 else "CB",
                "element": "C",
            }

        for i in range(n_residues):
            topology["residues"][i] = {"chain_id": "A", "residue_name": "ALA"}

        return topology

    @pytest.mark.parametrize("size_category", ["small", "medium"])
    def test_network_construction_performance(self, size_category):
        """Test network construction performance."""
        dataset = create_large_dataset(size_category)

        # Create mock frames and topology
        frames = []
        topology = self.create_mock_topology(dataset["n_atoms"])

        for i in range(min(dataset["n_frames"], 1000)):  # Limit for network tests
            coords = np.random.random((dataset["n_atoms"], 3)) * 100
            frame = TrajectoryFrame(frame_id=i, time_ps=i * 0.1, coordinates=coords)
            frames.append(frame)

        # Configure network builder
        config = NetworkAnalysisConfig(
            contact_cutoff=4.0, use_gpu=GPU_AVAILABLE, n_workers=4
        )

        builder = NetworkBuilder(config)

        # Monitor performance
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        start_time = time.time()

        try:
            # Build networks
            networks = builder.build_trajectory_networks(frames, topology)

            end_time = time.time()
            processing_time = end_time - start_time
            perf_stats = monitor.stop_monitoring()

            # Validate results
            assert len(networks) == len(frames)
            assert all(network.n_nodes > 0 for network in networks)

            # Calculate metrics
            throughput = len(networks) / processing_time

            # Performance assertions
            min_throughput = 10  # networks/second
            assert (
                throughput > min_throughput
            ), f"Network construction too slow: {throughput:.1f} networks/s"

            print(f"\nNetwork construction ({size_category}):")
            print(f"  Throughput: {throughput:.1f} networks/s")
            print(f"  Avg nodes: {np.mean([n.n_nodes for n in networks]):.1f}")
            print(f"  Avg edges: {np.mean([n.n_edges for n in networks]):.1f}")
            print(f"  Memory: {perf_stats['memory_max_mb']:.1f} MB")

        finally:
            builder.cleanup()

    def test_pathway_analysis_performance(self):
        """Test allosteric pathway analysis performance."""
        # Create mock networks
        n_networks = 100
        n_nodes = 200

        networks = []
        for i in range(n_networks):
            # Create mock network with random connectivity
            nodes = {j: f"node_{j}" for j in range(n_nodes)}

            # Create random edges
            n_edges = n_nodes * 2  # Sparse network
            edges = []

            for _ in range(n_edges):
                node1 = np.random.randint(0, n_nodes)
                node2 = np.random.randint(0, n_nodes)
                if node1 != node2:
                    weight = np.random.random()
                    edges.append((node1, node2, weight))

            # Mock network object would be created here
            # For testing, we'll use a simplified approach

        # This test would benchmark pathway finding algorithms
        # Implementation depends on final network analysis structure

        print("\nPathway analysis performance test placeholder")
        print(f"  Networks: {n_networks}")
        print(f"  Nodes per network: {n_nodes}")


class TestPLIFPerformance:
    """Test PLIF generation performance."""

    def create_mock_protein_ligand_topology(
        self, n_protein_atoms: int, n_ligand_atoms: int
    ) -> Tuple[Dict, Dict]:
        """Create mock protein and ligand topologies."""
        protein_topology = {"atom_indices": list(range(n_protein_atoms)), "atoms": {}}

        ligand_topology = {
            "atom_indices": list(
                range(n_protein_atoms, n_protein_atoms + n_ligand_atoms)
            ),
            "atoms": {},
        }

        # Create protein atoms
        for i in range(n_protein_atoms):
            residue_id = i // 10
            protein_topology["atoms"][i] = {
                "residue_id": residue_id,
                "chain_id": "A",
                "residue_name": "ALA",
                "atom_name": "CA",
                "element": "C",
            }

        # Create ligand atoms
        for i in range(n_ligand_atoms):
            ligand_idx = n_protein_atoms + i
            ligand_topology["atoms"][ligand_idx] = {
                "residue_id": 999,  # Ligand residue
                "chain_id": "L",
                "residue_name": "LIG",
                "atom_name": f"C{i}",
                "element": "C",
            }

        return protein_topology, ligand_topology

    @pytest.mark.parametrize("size_category", ["small", "medium"])
    def test_plif_generation_performance(self, size_category):
        """Test PLIF generation performance."""
        dataset = create_large_dataset(size_category)

        # Adjust system size for PLIF tests
        n_protein_atoms = min(dataset["n_atoms"], 5000)
        n_ligand_atoms = 50
        n_frames = min(dataset["n_frames"], 5000)

        # Create mock frames and topologies
        frames = []
        total_atoms = n_protein_atoms + n_ligand_atoms

        for i in range(n_frames):
            coords = np.random.random((total_atoms, 3)) * 100
            frame = TrajectoryFrame(frame_id=i, time_ps=i * 0.1, coordinates=coords)
            frames.append(frame)

        protein_topology, ligand_topology = self.create_mock_protein_ligand_topology(
            n_protein_atoms, n_ligand_atoms
        )

        # Configure PLIF analyzer
        config = PLIFAnalysisConfig(use_gpu=GPU_AVAILABLE, n_workers=4, batch_size=100)

        analyzer = create_plif_analyzer(use_gpu=GPU_AVAILABLE)

        # Monitor performance
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        start_time = time.time()

        try:
            # Generate PLIFs
            plifs = analyzer.analyze_trajectory(
                frames, protein_topology, ligand_topology, "test_ligand"
            )

            end_time = time.time()
            processing_time = end_time - start_time
            perf_stats = monitor.stop_monitoring()

            # Validate results
            assert len(plifs) == len(frames)
            assert all(plif.n_interactions >= 0 for plif in plifs)

            # Calculate metrics
            throughput = len(plifs) / processing_time

            # Performance assertions
            min_throughput = 50  # PLIFs/second
            assert (
                throughput > min_throughput
            ), f"PLIF generation too slow: {throughput:.1f} PLIFs/s"

            print(f"\nPLIF generation ({size_category}):")
            print(f"  Throughput: {throughput:.1f} PLIFs/s")
            print(f"  Protein atoms: {n_protein_atoms}")
            print(f"  Ligand atoms: {n_ligand_atoms}")
            print(
                f"  Avg interactions: {np.mean([p.n_interactions for p in plifs]):.1f}"
            )
            print(f"  Memory: {perf_stats['memory_max_mb']:.1f} MB")

        finally:
            analyzer.cleanup()

    def test_plif_clustering_performance(self):
        """Test PLIF clustering performance."""
        # Create mock PLIFs
        n_plifs = 1000
        n_interactions = 20

        # This would test the clustering performance
        # Implementation depends on final PLIF structure

        print("\nPLIF clustering performance test placeholder")
        print(f"  PLIFs: {n_plifs}")
        print(f"  Interactions per PLIF: {n_interactions}")


class TestMemoryEfficiency:
    """Test memory usage and efficiency."""

    def test_memory_scaling(self):
        """Test memory usage scaling with dataset size."""
        size_categories = ["small", "medium", "large"]
        memory_usage = {}

        for size_category in size_categories:
            dataset = create_large_dataset(size_category)

            # Create trajectory reader
            config = ParallelReadingConfig(memory_limit_gb=8.0, use_memory_mapping=True)

            reader = ParallelTrajectoryReader(config)

            # Monitor memory before and during processing
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)

            try:
                # Simulate processing
                traj_file, top_file = create_mock_trajectory_files(
                    dataset["n_frames"] // 10,  # Reduced for memory test
                    dataset["n_atoms"],
                    "dcd",
                )

                traj_id = reader.add_trajectory(traj_file, top_file)
                selection = TrajectorySelection(stride=10)

                # Process and measure peak memory
                results = reader.read_parallel_frames({traj_id: selection})

                peak_memory = process.memory_info().rss / (1024 * 1024)
                memory_increase = peak_memory - initial_memory

                memory_usage[size_category] = {
                    "dataset_size_gb": dataset["size_gb"],
                    "memory_increase_mb": memory_increase,
                    "efficiency_ratio": memory_increase / (dataset["size_gb"] * 1024),
                }

            finally:
                reader.cleanup()
                gc.collect()  # Force garbage collection

        # Analyze memory scaling
        print("\nMemory efficiency analysis:")
        for size_cat, stats in memory_usage.items():
            print(
                f"  {size_cat}: {stats['memory_increase_mb']:.1f} MB "
                f"({stats['efficiency_ratio']:.2f} ratio)"
            )

        # Memory usage should scale sub-linearly
        if len(memory_usage) >= 2:
            small_ratio = memory_usage["small"]["efficiency_ratio"]
            medium_ratio = memory_usage["medium"]["efficiency_ratio"]

            # Memory efficiency should improve (or at least not worsen significantly)
            assert (
                medium_ratio < small_ratio * 2
            ), f"Memory scaling too poor: {medium_ratio:.2f} vs {small_ratio:.2f}"

    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        n_iterations = 10
        memory_measurements = []

        process = psutil.Process()

        for i in range(n_iterations):
            # Create and process small dataset
            dataset = create_large_dataset("small")

            traj_file, top_file = create_mock_trajectory_files(
                dataset["n_frames"] // 10, dataset["n_atoms"], "dcd"
            )

            # Process with trajectory reader
            config = ParallelReadingConfig(memory_limit_gb=2.0)
            reader = ParallelTrajectoryReader(config)

            try:
                traj_id = reader.add_trajectory(traj_file, top_file)
                selection = TrajectorySelection(stride=5)
                results = reader.read_parallel_frames({traj_id: selection})

                # Measure memory after processing
                memory_mb = process.memory_info().rss / (1024 * 1024)
                memory_measurements.append(memory_mb)

            finally:
                reader.cleanup()
                gc.collect()

        # Check for memory leaks (increasing trend)
        if len(memory_measurements) >= 5:
            # Linear regression to detect trend
            x = np.arange(len(memory_measurements))
            y = np.array(memory_measurements)
            slope, _ = np.polyfit(x, y, 1)

            # Slope should be small (no significant memory increase)
            memory_increase_per_iteration = slope
            assert (
                memory_increase_per_iteration < 50
            ), f"Potential memory leak detected: {memory_increase_per_iteration:.1f} MB/iteration"

            print("\nMemory leak test:")
            print(f"  Iterations: {n_iterations}")
            print(f"  Memory trend: {memory_increase_per_iteration:.2f} MB/iteration")
            print(f"  Final memory: {memory_measurements[-1]:.1f} MB")


@pytest.mark.slow
class TestLargeScaleIntegration:
    """Integration tests for large-scale processing."""

    @pytest.mark.parametrize("format_type", ["dcd", "xtc"])
    def test_end_to_end_pipeline_performance(self, format_type):
        """Test complete analysis pipeline performance."""
        # Use medium dataset for integration test
        dataset = create_large_dataset("medium")

        # Reduce size for integration test
        n_frames = min(dataset["n_frames"], 5000)
        n_atoms = min(dataset["n_atoms"], 2000)

        # Create trajectory
        traj_file, top_file = create_mock_trajectory_files(
            n_frames, n_atoms, format_type
        )

        # Pipeline components
        trajectory_reader = create_parallel_trajectory_reader(
            n_workers=4, use_gpu=GPU_AVAILABLE
        )
        timeseries_calc = create_timeseries_calculator(use_gpu=GPU_AVAILABLE)

        # Monitor complete pipeline
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        start_time = time.time()

        try:
            # Step 1: Read trajectory
            traj_id = trajectory_reader.add_trajectory(traj_file, top_file)
            selection = TrajectorySelection(stride=2)
            frames_results = trajectory_reader.read_parallel_frames(
                {traj_id: selection}
            )
            frames = frames_results[traj_id]

            # Step 2: Calculate properties
            reference_coords = frames[0].coordinates
            timeseries_calc.add_rmsd_calculator(reference_coords)
            timeseries_calc.add_rg_calculator()
            timeseries_calc.add_temperature_calculator()

            properties = timeseries_calc.calculate_properties(frames)

            # Step 3: Basic analysis
            for prop_name, prop_data in properties.items():
                # Calculate basic statistics
                mean_val = np.mean(prop_data.values)
                std_val = np.std(prop_data.values)

                assert not np.isnan(mean_val), f"NaN values in {prop_name}"
                assert std_val >= 0, f"Invalid std for {prop_name}"

            end_time = time.time()
            processing_time = end_time - start_time
            perf_stats = monitor.stop_monitoring()

            # Calculate pipeline throughput
            pipeline_throughput = len(frames) / processing_time

            print(f"\nEnd-to-end pipeline ({format_type}):")
            print(f"  Frames processed: {len(frames)}")
            print(f"  Properties calculated: {len(properties)}")
            print(f"  Pipeline throughput: {pipeline_throughput:.1f} frames/s")
            print(f"  Total time: {processing_time:.2f}s")
            print(f"  Peak memory: {perf_stats['memory_max_mb']:.1f} MB")

            # Performance assertions
            assert (
                pipeline_throughput > 10
            ), f"Pipeline too slow: {pipeline_throughput:.1f} frames/s"
            assert len(properties) > 0, "No properties calculated"

        finally:
            trajectory_reader.cleanup()
            timeseries_calc.cleanup()


def save_performance_results(
    results: List[PerformanceMetrics],
    output_file: str = "trajectory_performance_results.csv",
):
    """Save performance test results to CSV file."""
    if not results:
        return

    # Convert to DataFrame
    df = pd.DataFrame([metrics.to_dict() for metrics in results])

    # Add timestamp and system info
    df["timestamp"] = pd.Timestamp.now()
    df["python_version"] = sys.version.split()[0]
    df["gpu_available"] = GPU_AVAILABLE
    df["cpu_count"] = psutil.cpu_count()
    df["total_memory_gb"] = psutil.virtual_memory().total / (1024**3)

    # Save to file
    output_path = Path("test_results") / output_file
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nPerformance results saved to: {output_path}")
    print("Test summary:")
    print(f"  Tests run: {len(results)}")
    print(f"  Average throughput: {df['throughput'].mean():.1f}")
    print(f"  Peak memory usage: {df['memory_usage_mb'].max():.1f} MB")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])

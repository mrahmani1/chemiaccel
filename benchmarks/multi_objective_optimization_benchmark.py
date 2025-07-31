"""
Performance benchmarks for multi-objective optimization framework.

This script benchmarks the MPOptimizer performance across different
problem sizes, number of objectives, and hardware configurations.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chemi_accel.ai.multi_objective_optimization import (
    CrowdingDistance,
    HypervolumeCalculator,
    MPOptimizer,
    NonDominatedSorting,
    ObjectiveConfig,
)


class MultiObjectiveBenchmark:
    """Benchmark suite for multi-objective optimization."""

    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results = {
            "pareto_front_identification": [],
            "non_dominated_sorting": [],
            "crowding_distance": [],
            "hypervolume_calculation": [],
            "scalability_molecules": [],
            "scalability_objectives": [],
            "gpu_acceleration": [],
        }

    def generate_synthetic_data(
        self, n_molecules: int, n_objectives: int, seed: int = 42
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Generate synthetic molecular data for benchmarking."""
        np.random.seed(seed)

        molecules = [f"mol_{i:06d}" for i in range(n_molecules)]
        properties = {}

        for i in range(n_objectives):
            # Generate correlated objectives to create realistic Pareto fronts
            if i == 0:
                values = np.random.exponential(scale=2.0, size=n_molecules)
            else:
                # Create some correlation with previous objectives
                correlation = 0.3
                noise = np.random.normal(0, 1, n_molecules)
                values = (
                    correlation * properties[f"obj_{i-1}"] + (1 - correlation) * noise
                )
                values = np.abs(values)  # Ensure positive values

            properties[f"obj_{i}"] = values

        return molecules, properties

    def benchmark_pareto_front_identification(self):
        """Benchmark Pareto front identification performance."""
        print("Benchmarking Pareto front identification...")

        problem_sizes = [100, 500, 1000, 5000, 10000, 50000]
        n_objectives = 3

        for n_molecules in problem_sizes:
            molecules, properties = self.generate_synthetic_data(
                n_molecules, n_objectives
            )

            objectives = [
                ObjectiveConfig(name=f"obj_{i}", minimize=True, normalization="minmax")
                for i in range(n_objectives)
            ]

            # CPU benchmark
            start_time = time.time()
            optimizer_cpu = MPOptimizer(objectives=objectives, use_gpu=False)
            result_cpu = optimizer_cpu.optimize(molecules, properties)
            cpu_time = time.time() - start_time

            # GPU benchmark
            start_time = time.time()
            optimizer_gpu = MPOptimizer(objectives=objectives, use_gpu=True)
            result_gpu = optimizer_gpu.optimize(molecules, properties)
            gpu_time = time.time() - start_time

            self.results["pareto_front_identification"].append(
                {
                    "n_molecules": n_molecules,
                    "n_objectives": n_objectives,
                    "cpu_time": cpu_time,
                    "gpu_time": gpu_time,
                    "speedup": cpu_time / gpu_time if gpu_time > 0 else 1.0,
                    "n_pareto_cpu": len(result_cpu.pareto_front.pareto_indices),
                    "n_pareto_gpu": len(result_gpu.pareto_front.pareto_indices),
                    "hypervolume_cpu": result_cpu.pareto_front.hypervolume,
                    "hypervolume_gpu": result_gpu.pareto_front.hypervolume,
                }
            )

            print(
                f"  {n_molecules:6d} molecules: CPU {cpu_time:6.2f}s, "
                f"GPU {gpu_time:6.2f}s, Speedup {cpu_time/gpu_time:5.2f}x"
            )

    def benchmark_non_dominated_sorting(self):
        """Benchmark non-dominated sorting algorithms."""
        print("Benchmarking non-dominated sorting...")

        problem_sizes = [100, 500, 1000, 5000, 10000, 25000]
        n_objectives = 5

        for n_molecules in problem_sizes:
            # Generate random objective values
            objectives = np.random.rand(n_molecules, n_objectives)

            # CPU sorting
            sorter_cpu = NonDominatedSorting(use_gpu=False)
            start_time = time.time()
            fronts_cpu = sorter_cpu.fast_non_dominated_sort(objectives)
            cpu_time = time.time() - start_time

            # GPU sorting
            sorter_gpu = NonDominatedSorting(use_gpu=True)
            start_time = time.time()
            fronts_gpu = sorter_gpu.fast_non_dominated_sort(objectives)
            gpu_time = time.time() - start_time

            self.results["non_dominated_sorting"].append(
                {
                    "n_molecules": n_molecules,
                    "n_objectives": n_objectives,
                    "cpu_time": cpu_time,
                    "gpu_time": gpu_time,
                    "speedup": cpu_time / gpu_time if gpu_time > 0 else 1.0,
                    "n_fronts_cpu": len(fronts_cpu),
                    "n_fronts_gpu": len(fronts_gpu),
                    "first_front_size_cpu": len(fronts_cpu[0]) if fronts_cpu else 0,
                    "first_front_size_gpu": len(fronts_gpu[0]) if fronts_gpu else 0,
                }
            )

            print(
                f"  {n_molecules:6d} molecules: CPU {cpu_time:6.3f}s, "
                f"GPU {gpu_time:6.3f}s, Speedup {cpu_time/gpu_time:5.2f}x"
            )

    def benchmark_crowding_distance(self):
        """Benchmark crowding distance calculation."""
        print("Benchmarking crowding distance calculation...")

        front_sizes = [10, 50, 100, 500, 1000, 2000]
        n_objectives = 4

        for front_size in front_sizes:
            # Generate random objectives for front
            objectives = np.random.rand(front_size * 2, n_objectives)
            front_indices = np.arange(front_size)

            # CPU crowding distance
            crowding_cpu = CrowdingDistance(use_gpu=False)
            start_time = time.time()
            distances_cpu = crowding_cpu.calculate_crowding_distance(
                objectives, front_indices
            )
            cpu_time = time.time() - start_time

            # GPU crowding distance
            crowding_gpu = CrowdingDistance(use_gpu=True)
            start_time = time.time()
            distances_gpu = crowding_gpu.calculate_crowding_distance(
                objectives, front_indices
            )
            gpu_time = time.time() - start_time

            self.results["crowding_distance"].append(
                {
                    "front_size": front_size,
                    "n_objectives": n_objectives,
                    "cpu_time": cpu_time,
                    "gpu_time": gpu_time,
                    "speedup": cpu_time / gpu_time if gpu_time > 0 else 1.0,
                    "max_distance_cpu": (
                        np.max(distances_cpu[np.isfinite(distances_cpu)])
                        if np.any(np.isfinite(distances_cpu))
                        else 0
                    ),
                    "max_distance_gpu": (
                        np.max(distances_gpu[np.isfinite(distances_gpu)])
                        if np.any(np.isfinite(distances_gpu))
                        else 0
                    ),
                }
            )

            print(
                f"  {front_size:4d} front size: CPU {cpu_time:6.4f}s, "
                f"GPU {gpu_time:6.4f}s, Speedup {cpu_time/gpu_time:5.2f}x"
            )

    def benchmark_hypervolume_calculation(self):
        """Benchmark hypervolume calculation."""
        print("Benchmarking hypervolume calculation...")

        pareto_sizes = [5, 10, 20, 50, 100, 200]
        n_objectives_list = [2, 3, 4, 5]

        for n_objectives in n_objectives_list:
            for pareto_size in pareto_sizes:
                # Generate random Pareto front
                objectives = np.random.rand(pareto_size * 2, n_objectives)
                pareto_indices = np.arange(pareto_size)

                # Hypervolume calculation
                hv_calculator = HypervolumeCalculator()
                start_time = time.time()
                hypervolume = hv_calculator.calculate_hypervolume(
                    objectives, pareto_indices
                )
                computation_time = time.time() - start_time

                self.results["hypervolume_calculation"].append(
                    {
                        "pareto_size": pareto_size,
                        "n_objectives": n_objectives,
                        "computation_time": computation_time,
                        "hypervolume": hypervolume,
                    }
                )

                print(
                    f"  {n_objectives}D, {pareto_size:3d} points: {computation_time:6.4f}s, "
                    f"HV: {hypervolume:8.4f}"
                )

    def benchmark_scalability_molecules(self):
        """Benchmark scalability with increasing number of molecules."""
        print("Benchmarking scalability with molecules...")

        molecule_counts = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        n_objectives = 4

        for n_molecules in molecule_counts:
            molecules, properties = self.generate_synthetic_data(
                n_molecules, n_objectives
            )

            objectives = [
                ObjectiveConfig(name=f"obj_{i}", minimize=True, normalization="minmax")
                for i in range(n_objectives)
            ]

            optimizer = MPOptimizer(objectives=objectives, use_gpu=True)

            start_time = time.time()
            result = optimizer.optimize(molecules, properties)
            computation_time = time.time() - start_time

            self.results["scalability_molecules"].append(
                {
                    "n_molecules": n_molecules,
                    "n_objectives": n_objectives,
                    "computation_time": computation_time,
                    "time_per_molecule": computation_time
                    / n_molecules
                    * 1000,  # ms per molecule
                    "n_pareto_solutions": len(result.pareto_front.pareto_indices),
                    "pareto_fraction": len(result.pareto_front.pareto_indices)
                    / n_molecules,
                    "hypervolume": result.pareto_front.hypervolume,
                    "memory_usage_mb": self._get_memory_usage(),
                }
            )

            print(
                f"  {n_molecules:6d} molecules: {computation_time:6.2f}s "
                f"({computation_time/n_molecules*1000:5.2f} ms/mol), "
                f"{len(result.pareto_front.pareto_indices):4d} Pareto solutions"
            )

    def benchmark_scalability_objectives(self):
        """Benchmark scalability with increasing number of objectives."""
        print("Benchmarking scalability with objectives...")

        objective_counts = [2, 3, 4, 5, 6, 8, 10, 15, 20]
        n_molecules = 5000

        for n_objectives in objective_counts:
            molecules, properties = self.generate_synthetic_data(
                n_molecules, n_objectives
            )

            objectives = [
                ObjectiveConfig(name=f"obj_{i}", minimize=True, normalization="minmax")
                for i in range(n_objectives)
            ]

            optimizer = MPOptimizer(objectives=objectives, use_gpu=True)

            start_time = time.time()
            result = optimizer.optimize(molecules, properties)
            computation_time = time.time() - start_time

            self.results["scalability_objectives"].append(
                {
                    "n_molecules": n_molecules,
                    "n_objectives": n_objectives,
                    "computation_time": computation_time,
                    "time_per_objective": computation_time / n_objectives,
                    "n_pareto_solutions": len(result.pareto_front.pareto_indices),
                    "pareto_fraction": len(result.pareto_front.pareto_indices)
                    / n_molecules,
                    "hypervolume": result.pareto_front.hypervolume,
                }
            )

            print(
                f"  {n_objectives:2d} objectives: {computation_time:6.2f}s "
                f"({computation_time/n_objectives:5.2f} s/obj), "
                f"{len(result.pareto_front.pareto_indices):4d} Pareto solutions"
            )

    def benchmark_gpu_acceleration(self):
        """Benchmark GPU acceleration effectiveness."""
        print("Benchmarking GPU acceleration...")

        test_cases = [
            (1000, 3),
            (5000, 3),
            (10000, 3),
            (5000, 5),
            (5000, 8),
            (20000, 4),
        ]

        for n_molecules, n_objectives in test_cases:
            molecules, properties = self.generate_synthetic_data(
                n_molecules, n_objectives
            )

            objectives = [
                ObjectiveConfig(name=f"obj_{i}", minimize=True, normalization="minmax")
                for i in range(n_objectives)
            ]

            # CPU benchmark
            optimizer_cpu = MPOptimizer(objectives=objectives, use_gpu=False)
            start_time = time.time()
            result_cpu = optimizer_cpu.optimize(molecules, properties)
            cpu_time = time.time() - start_time

            # GPU benchmark
            optimizer_gpu = MPOptimizer(objectives=objectives, use_gpu=True)
            start_time = time.time()
            result_gpu = optimizer_gpu.optimize(molecules, properties)
            gpu_time = time.time() - start_time

            self.results["gpu_acceleration"].append(
                {
                    "n_molecules": n_molecules,
                    "n_objectives": n_objectives,
                    "cpu_time": cpu_time,
                    "gpu_time": gpu_time,
                    "speedup": cpu_time / gpu_time if gpu_time > 0 else 1.0,
                    "efficiency": (cpu_time - gpu_time)
                    / cpu_time
                    * 100,  # % improvement
                    "n_pareto_cpu": len(result_cpu.pareto_front.pareto_indices),
                    "n_pareto_gpu": len(result_gpu.pareto_front.pareto_indices),
                    "results_match": abs(
                        len(result_cpu.pareto_front.pareto_indices)
                        - len(result_gpu.pareto_front.pareto_indices)
                    )
                    <= 1,
                }
            )

            print(
                f"  {n_molecules:6d} mol, {n_objectives:2d} obj: "
                f"CPU {cpu_time:6.2f}s, GPU {gpu_time:6.2f}s, "
                f"Speedup {cpu_time/gpu_time:5.2f}x"
            )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def run_all_benchmarks(self):
        """Run all benchmark suites."""
        print("Starting multi-objective optimization benchmarks...")
        print("=" * 60)

        self.benchmark_pareto_front_identification()
        print()

        self.benchmark_non_dominated_sorting()
        print()

        self.benchmark_crowding_distance()
        print()

        self.benchmark_hypervolume_calculation()
        print()

        self.benchmark_scalability_molecules()
        print()

        self.benchmark_scalability_objectives()
        print()

        self.benchmark_gpu_acceleration()
        print()

        print("All benchmarks completed!")

    def save_results(self):
        """Save benchmark results to files."""
        # Save raw results as JSON
        results_file = os.path.join(
            self.output_dir, "multi_objective_benchmark_results.json"
        )
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to {results_file}")

        # Create summary report
        self._create_summary_report()

    def _create_summary_report(self):
        """Create a summary report of benchmark results."""
        report_file = os.path.join(
            self.output_dir, "multi_objective_benchmark_summary.txt"
        )

        with open(report_file, "w") as f:
            f.write("Multi-Objective Optimization Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")

            # Pareto front identification summary
            if self.results["pareto_front_identification"]:
                f.write("Pareto Front Identification Performance:\n")
                f.write("-" * 40 + "\n")

                data = self.results["pareto_front_identification"]
                avg_speedup = np.mean([r["speedup"] for r in data])
                max_molecules = max([r["n_molecules"] for r in data])
                max_time = max([r["cpu_time"] for r in data])

                f.write(f"Average GPU speedup: {avg_speedup:.2f}x\n")
                f.write(f"Largest problem size: {max_molecules:,} molecules\n")
                f.write(f"Maximum computation time: {max_time:.2f}s\n\n")

            # Scalability summary
            if self.results["scalability_molecules"]:
                f.write("Scalability with Molecules:\n")
                f.write("-" * 30 + "\n")

                data = self.results["scalability_molecules"]
                times = [r["time_per_molecule"] for r in data]
                avg_time_per_mol = np.mean(times)

                f.write(f"Average time per molecule: {avg_time_per_mol:.3f} ms\n")
                f.write(
                    f"Time complexity appears: {'Linear' if np.std(times) < np.mean(times) * 0.5 else 'Super-linear'}\n\n"
                )

            # GPU acceleration summary
            if self.results["gpu_acceleration"]:
                f.write("GPU Acceleration Effectiveness:\n")
                f.write("-" * 35 + "\n")

                data = self.results["gpu_acceleration"]
                speedups = [r["speedup"] for r in data]
                avg_speedup = np.mean(speedups)
                max_speedup = max(speedups)

                f.write(f"Average speedup: {avg_speedup:.2f}x\n")
                f.write(f"Maximum speedup: {max_speedup:.2f}x\n")
                f.write(
                    f"GPU acceleration effective: {'Yes' if avg_speedup > 1.5 else 'Limited'}\n\n"
                )

            # Recommendations
            f.write("Recommendations:\n")
            f.write("-" * 15 + "\n")

            if avg_speedup > 2.0:
                f.write("- GPU acceleration provides significant benefits\n")
            elif avg_speedup > 1.2:
                f.write("- GPU acceleration provides moderate benefits\n")
            else:
                f.write("- GPU acceleration benefits are limited\n")

            if max_molecules >= 50000:
                f.write("- Framework scales well to large problem sizes\n")

            f.write("- Consider using population size limits for very large datasets\n")
            f.write("- Normalize objectives for better convergence\n")

        print(f"Summary report saved to {report_file}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Multi-objective optimization benchmarks"
    )
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks with smaller problem sizes",
    )

    args = parser.parse_args()

    benchmark = MultiObjectiveBenchmark(output_dir=args.output_dir)

    if args.quick:
        print("Running quick benchmarks...")
        # Reduce problem sizes for quick testing
        benchmark.benchmark_pareto_front_identification()
        benchmark.benchmark_gpu_acceleration()
    else:
        benchmark.run_all_benchmarks()

    benchmark.save_results()


if __name__ == "__main__":
    main()

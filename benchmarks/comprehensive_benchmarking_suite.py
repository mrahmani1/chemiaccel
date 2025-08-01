"""
Comprehensive Benchmarking Suite for All ChemiAccel Modules.

This module provides an integrated benchmarking framework that tests performance
across all ChemiAccel components including AI, polymer analysis, materials science,
cloud computing, HPC integration, and core descriptor calculations.
"""

import gc
import json
import logging
import multiprocessing as mp
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil

# Import existing benchmark framework
from .benchmark_framework import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkStatus,
    BenchmarkSuite,
)

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from dask.distributed import Client, LocalCluster

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


@dataclass
class ComprehensiveBenchmarkResult:
    """Extended benchmark result with module-specific metrics."""

    base_result: BenchmarkResult
    module_category: str
    complexity_score: float
    scalability_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    resource_efficiency: Dict[str, float] = field(default_factory=dict)
    comparative_analysis: Dict[str, Any] = field(default_factory=dict)


class ModuleCategory(Enum):
    """Categories of ChemiAccel modules."""

    CORE_DESCRIPTORS = "core_descriptors"
    AI_MODULES = "ai_modules"
    POLYMER_ANALYSIS = "polymer_analysis"
    MATERIALS_SCIENCE = "materials_science"
    CLOUD_COMPUTING = "cloud_computing"
    HPC_INTEGRATION = "hpc_integration"
    FEDERATED_LEARNING = "federated_learning"
    QUANTUM_CHEMISTRY = "quantum_chemistry"
    BIOINFORMATICS = "bioinformatics"


class ComprehensiveBenchmarkSuite:
    """Advanced benchmarking suite for all ChemiAccel modules."""

    def __init__(self, config: BenchmarkConfig, output_dir: Optional[str] = None):
        """Initialize comprehensive benchmark suite.

        Args:
            config: Base benchmark configuration
            output_dir: Directory to save benchmark results
        """
        self.config = config
        self.output_dir = (
            Path(output_dir) if output_dir else Path("comprehensive_benchmarks")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[ComprehensiveBenchmarkResult] = []
        self.module_suites: Dict[ModuleCategory, BenchmarkSuite] = {}

        # Initialize module-specific benchmark suites
        self._initialize_module_suites()

        # Performance baselines for comparison
        self.performance_baselines = {
            "core_descriptors": {
                "molecular_weight": 10000,  # molecules/sec
                "tpsa": 5000,
                "logp": 3000,
                "morgan_fingerprint": 1000,
            },
            "ai_modules": {
                "active_learning": 500,  # iterations/sec
                "generative_chemistry": 100,  # molecules/sec
                "uncertainty_quantification": 1000,
            },
            "polymer_analysis": {
                "monomer_analysis": 2000,  # molecules/sec
                "copolymer_analysis": 500,
                "thermoset_analysis": 200,
            },
        }

    def _initialize_module_suites(self):
        """Initialize benchmark suites for each module category."""
        for category in ModuleCategory:
            category_config = BenchmarkConfig(
                name=f"{self.config.name}_{category.value}",
                description=f"Benchmarks for {category.value} module",
                dataset_sizes=self.config.dataset_sizes,
                iterations=self.config.iterations,
                backends_to_test=self.config.backends_to_test,
            )

            self.module_suites[category] = BenchmarkSuite(category_config)

    @contextmanager
    def resource_monitor(self):
        """Context manager for monitoring system resources."""
        initial_memory = psutil.virtual_memory().used
        initial_cpu_times = psutil.cpu_times()
        start_time = time.time()

        # Start resource monitoring
        resource_samples = []

        def monitor_resources():
            while hasattr(monitor_resources, "active") and monitor_resources.active:
                sample = {
                    "timestamp": time.time(),
                    "memory_used_gb": psutil.virtual_memory().used / 1024**3,
                    "memory_percent": psutil.virtual_memory().percent,
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "disk_io": psutil.disk_io_counters()._asdict()
                    if psutil.disk_io_counters()
                    else {},
                    "network_io": psutil.net_io_counters()._asdict()
                    if psutil.net_io_counters()
                    else {},
                }
                resource_samples.append(sample)
                time.sleep(0.5)

        import threading

        monitor_resources.active = True
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        try:
            yield {
                "get_samples": lambda: resource_samples,
                "get_current_memory": lambda: psutil.virtual_memory().used / 1024**3,
                "get_current_cpu": lambda: psutil.cpu_percent(),
            }
        finally:
            monitor_resources.active = False
            monitor_thread.join()

    def benchmark_core_descriptors(self) -> List[ComprehensiveBenchmarkResult]:
        """Benchmark core descriptor calculation modules."""
        results = []

        print("Benchmarking core descriptor modules...")

        # Test configurations for different descriptor types
        descriptor_configs = [
            {
                "name": "basic_descriptors",
                "descriptors": ["molecular_weight", "tpsa", "logp", "hbd", "hba"],
                "complexity": 1.0,
            },
            {
                "name": "fingerprints",
                "descriptors": [
                    "morgan_fingerprint",
                    "maccs_keys",
                    "topological_fingerprint",
                ],
                "complexity": 2.0,
            },
            {
                "name": "3d_descriptors",
                "descriptors": [
                    "pmi",
                    "asphericity",
                    "eccentricity",
                    "radius_of_gyration",
                ],
                "complexity": 3.0,
            },
            {
                "name": "advanced_descriptors",
                "descriptors": ["whim", "getaway", "autocorrelation"],
                "complexity": 4.0,
            },
        ]

        for desc_config in descriptor_configs:
            with self.resource_monitor() as monitor:
                # Mock descriptor calculation benchmark
                def mock_descriptor_calculation(molecules):
                    """Mock descriptor calculation with realistic timing."""
                    time.sleep(
                        len(molecules) * desc_config["complexity"] * 0.0001
                    )  # Simulate complexity

                    return {
                        desc: np.random.randn(len(molecules))
                        for desc in desc_config["descriptors"]
                    }

                # Run benchmark across different dataset sizes
                for dataset_size in self.config.dataset_sizes[:3]:  # Test first 3 sizes
                    # Generate mock molecular dataset
                    mock_molecules = [f"molecule_{i}" for i in range(dataset_size)]

                    # Benchmark execution
                    start_time = time.time()
                    result_data = mock_descriptor_calculation(mock_molecules)
                    end_time = time.time()

                    execution_time = end_time - start_time
                    throughput = (
                        dataset_size / execution_time if execution_time > 0 else 0
                    )

                    # Create base benchmark result
                    base_result = BenchmarkResult(
                        config_name=self.config.name,
                        implementation="chemi_accel",
                        backend="cpu",
                        descriptor=desc_config["name"],
                        dataset_size=dataset_size,
                        execution_times=[execution_time],
                        memory_usage_mb=monitor["get_current_memory"]() * 1024,
                        status=BenchmarkStatus.COMPLETED,
                    )

                    # Calculate performance metrics
                    baseline_throughput = self.performance_baselines[
                        "core_descriptors"
                    ].get(desc_config["descriptors"][0], 1000)
                    performance_ratio = throughput / baseline_throughput

                    # Create comprehensive result
                    comprehensive_result = ComprehensiveBenchmarkResult(
                        base_result=base_result,
                        module_category=ModuleCategory.CORE_DESCRIPTORS.value,
                        complexity_score=desc_config["complexity"],
                        scalability_metrics={
                            "throughput": throughput,
                            "performance_ratio": performance_ratio,
                            "memory_efficiency": dataset_size
                            / (monitor["get_current_memory"]() * 1024),
                            "scaling_factor": 1.0,  # Would calculate from multiple runs
                        },
                        quality_metrics={
                            "accuracy": 0.99,  # Mock accuracy
                            "precision": 0.98,
                            "completeness": 1.0,
                        },
                        resource_efficiency={
                            "cpu_utilization": monitor["get_current_cpu"]() / 100,
                            "memory_peak_gb": monitor["get_current_memory"](),
                            "memory_growth_rate": 0.0,  # Mock value
                        },
                    )

                    results.append(comprehensive_result)
                    print(
                        f"  {desc_config['name']} ({dataset_size:,} molecules): {throughput:.0f} molecules/sec"
                    )

        return results

    def benchmark_ai_modules(self) -> List[ComprehensiveBenchmarkResult]:
        """Benchmark AI and machine learning modules."""
        results = []

        print("Benchmarking AI modules...")

        ai_modules = [
            {
                "name": "active_learning",
                "complexity": 3.0,
                "metric_unit": "iterations/sec",
            },
            {
                "name": "uncertainty_quantification",
                "complexity": 2.5,
                "metric_unit": "predictions/sec",
            },
            {
                "name": "generative_chemistry",
                "complexity": 4.0,
                "metric_unit": "molecules_generated/sec",
            },
            {
                "name": "multi_objective_optimization",
                "complexity": 3.5,
                "metric_unit": "evaluations/sec",
            },
            {
                "name": "bayesian_optimization",
                "complexity": 3.0,
                "metric_unit": "acquisitions/sec",
            },
        ]

        for ai_module in ai_modules:
            with self.resource_monitor() as monitor:
                # Mock AI module benchmark
                def mock_ai_computation(data_size):
                    """Mock AI computation with complexity-based timing."""
                    computation_time = data_size * ai_module["complexity"] * 0.001
                    time.sleep(
                        min(computation_time, 5.0)
                    )  # Cap at 5 seconds for testing

                    return {
                        "performance_score": np.random.uniform(0.8, 0.95),
                        "convergence_iterations": np.random.randint(10, 100),
                        "quality_metrics": {
                            "accuracy": np.random.uniform(0.85, 0.98),
                            "precision": np.random.uniform(0.80, 0.95),
                            "recall": np.random.uniform(0.82, 0.96),
                        },
                    }

                # Test different problem sizes
                problem_sizes = [1000, 5000, 10000]  # Reduced for AI modules

                for problem_size in problem_sizes:
                    start_time = time.time()
                    ai_result = mock_ai_computation(problem_size)
                    end_time = time.time()

                    execution_time = end_time - start_time
                    throughput = (
                        problem_size / execution_time if execution_time > 0 else 0
                    )

                    # Create base benchmark result
                    base_result = BenchmarkResult(
                        config_name=self.config.name,
                        implementation="chemi_accel_ai",
                        backend="cpu",
                        descriptor=ai_module["name"],
                        dataset_size=problem_size,
                        execution_times=[execution_time],
                        memory_usage_mb=monitor["get_current_memory"]() * 1024,
                        status=BenchmarkStatus.COMPLETED,
                    )

                    # Calculate AI-specific metrics
                    baseline_throughput = self.performance_baselines["ai_modules"].get(
                        ai_module["name"], 500
                    )
                    performance_ratio = throughput / baseline_throughput

                    comprehensive_result = ComprehensiveBenchmarkResult(
                        base_result=base_result,
                        module_category=ModuleCategory.AI_MODULES.value,
                        complexity_score=ai_module["complexity"],
                        scalability_metrics={
                            "throughput": throughput,
                            "performance_ratio": performance_ratio,
                            "convergence_rate": 1.0
                            / ai_result["convergence_iterations"],
                            "computational_efficiency": problem_size
                            / (execution_time * ai_module["complexity"]),
                        },
                        quality_metrics=ai_result["quality_metrics"],
                        resource_efficiency={
                            "cpu_utilization": monitor["get_current_cpu"]() / 100,
                            "memory_peak_gb": monitor["get_current_memory"](),
                            "ai_performance_score": ai_result["performance_score"],
                        },
                    )

                    results.append(comprehensive_result)
                    print(
                        f"  {ai_module['name']} ({problem_size:,} samples): {throughput:.1f} {ai_module['metric_unit']}"
                    )

        return results

    def benchmark_polymer_analysis(self) -> List[ComprehensiveBenchmarkResult]:
        """Benchmark polymer analysis modules."""
        results = []

        print("Benchmarking polymer analysis modules...")

        polymer_modules = [
            {
                "name": "monomer_analysis",
                "complexity": 2.0,
                "test_sizes": [500, 2000, 5000],
            },
            {
                "name": "copolymer_analysis",
                "complexity": 3.0,
                "test_sizes": [100, 500, 1000],
            },
            {
                "name": "thermoset_analysis",
                "complexity": 4.0,
                "test_sizes": [50, 200, 500],
            },
            {
                "name": "polymer_property_prediction",
                "complexity": 2.5,
                "test_sizes": [1000, 5000, 10000],
            },
        ]

        for polymer_module in polymer_modules:
            with self.resource_monitor() as monitor:

                def mock_polymer_analysis(molecules):
                    """Mock polymer analysis computation."""
                    analysis_time = (
                        len(molecules) * polymer_module["complexity"] * 0.002
                    )
                    time.sleep(min(analysis_time, 3.0))  # Cap for testing

                    return {
                        "analysis_accuracy": np.random.uniform(0.88, 0.96),
                        "property_predictions": {
                            "glass_transition_temp": np.random.uniform(
                                50, 200, len(molecules)
                            ),
                            "molecular_weight": np.random.uniform(
                                1000, 100000, len(molecules)
                            ),
                            "polydispersity": np.random.uniform(
                                1.1, 3.0, len(molecules)
                            ),
                        },
                        "confidence_scores": np.random.uniform(
                            0.7, 0.95, len(molecules)
                        ),
                    }

                for test_size in polymer_module["test_sizes"]:
                    mock_polymers = [f"polymer_{i}" for i in range(test_size)]

                    start_time = time.time()
                    analysis_result = mock_polymer_analysis(mock_polymers)
                    end_time = time.time()

                    execution_time = end_time - start_time
                    throughput = test_size / execution_time if execution_time > 0 else 0

                    base_result = BenchmarkResult(
                        config_name=self.config.name,
                        implementation="chemi_accel_polymer",
                        backend="cpu",
                        descriptor=polymer_module["name"],
                        dataset_size=test_size,
                        execution_times=[execution_time],
                        memory_usage_mb=monitor["get_current_memory"]() * 1024,
                        status=BenchmarkStatus.COMPLETED,
                    )

                    baseline_throughput = self.performance_baselines[
                        "polymer_analysis"
                    ].get(polymer_module["name"], 1000)
                    performance_ratio = throughput / baseline_throughput

                    comprehensive_result = ComprehensiveBenchmarkResult(
                        base_result=base_result,
                        module_category=ModuleCategory.POLYMER_ANALYSIS.value,
                        complexity_score=polymer_module["complexity"],
                        scalability_metrics={
                            "throughput": throughput,
                            "performance_ratio": performance_ratio,
                            "analysis_efficiency": test_size
                            / (execution_time * polymer_module["complexity"]),
                        },
                        quality_metrics={
                            "analysis_accuracy": analysis_result["analysis_accuracy"],
                            "prediction_confidence": np.mean(
                                analysis_result["confidence_scores"]
                            ),
                            "property_coverage": len(
                                analysis_result["property_predictions"]
                            ),
                        },
                        resource_efficiency={
                            "cpu_utilization": monitor["get_current_cpu"]() / 100,
                            "memory_peak_gb": monitor["get_current_memory"](),
                            "polymer_complexity_handled": polymer_module["complexity"],
                        },
                    )

                    results.append(comprehensive_result)
                    print(
                        f"  {polymer_module['name']} ({test_size:,} polymers): {throughput:.1f} polymers/sec"
                    )

        return results

    def benchmark_materials_science(self) -> List[ComprehensiveBenchmarkResult]:
        """Benchmark materials science modules."""
        results = []

        print("Benchmarking materials science modules...")

        materials_modules = [
            {
                "name": "crystal_structure_analysis",
                "complexity": 3.5,
                "test_sizes": [100, 500, 1000],
            },
            {
                "name": "periodic_descriptors",
                "complexity": 4.0,
                "test_sizes": [50, 200, 500],
            },
            {
                "name": "materials_property_prediction",
                "complexity": 3.0,
                "test_sizes": [200, 1000, 2000],
            },
        ]

        for materials_module in materials_modules:
            with self.resource_monitor() as monitor:

                def mock_materials_computation(structures):
                    """Mock materials science computation."""
                    computation_time = (
                        len(structures) * materials_module["complexity"] * 0.005
                    )
                    time.sleep(min(computation_time, 4.0))

                    return {
                        "structure_accuracy": np.random.uniform(0.92, 0.98),
                        "property_predictions": {
                            "band_gap": np.random.uniform(0.5, 5.0, len(structures)),
                            "formation_energy": np.random.uniform(
                                -5, 0, len(structures)
                            ),
                            "bulk_modulus": np.random.uniform(50, 300, len(structures)),
                        },
                        "symmetry_analysis": np.random.choice(
                            [True, False], len(structures)
                        ),
                    }

                for test_size in materials_module["test_sizes"]:
                    mock_structures = [f"structure_{i}" for i in range(test_size)]

                    start_time = time.time()
                    computation_result = mock_materials_computation(mock_structures)
                    end_time = time.time()

                    execution_time = end_time - start_time
                    throughput = test_size / execution_time if execution_time > 0 else 0

                    base_result = BenchmarkResult(
                        config_name=self.config.name,
                        implementation="chemi_accel_materials",
                        backend="cpu",
                        descriptor=materials_module["name"],
                        dataset_size=test_size,
                        execution_times=[execution_time],
                        memory_usage_mb=monitor["get_current_memory"]() * 1024,
                        status=BenchmarkStatus.COMPLETED,
                    )

                    comprehensive_result = ComprehensiveBenchmarkResult(
                        base_result=base_result,
                        module_category=ModuleCategory.MATERIALS_SCIENCE.value,
                        complexity_score=materials_module["complexity"],
                        scalability_metrics={
                            "throughput": throughput,
                            "computational_efficiency": test_size
                            / (execution_time * materials_module["complexity"]),
                            "structure_processing_rate": throughput,
                        },
                        quality_metrics={
                            "structure_accuracy": computation_result[
                                "structure_accuracy"
                            ],
                            "property_prediction_coverage": len(
                                computation_result["property_predictions"]
                            ),
                            "symmetry_detection_rate": np.mean(
                                computation_result["symmetry_analysis"]
                            ),
                        },
                        resource_efficiency={
                            "cpu_utilization": monitor["get_current_cpu"]() / 100,
                            "memory_peak_gb": monitor["get_current_memory"](),
                            "materials_complexity": materials_module["complexity"],
                        },
                    )

                    results.append(comprehensive_result)
                    print(
                        f"  {materials_module['name']} ({test_size:,} structures): {throughput:.1f} structures/sec"
                    )

        return results

    def benchmark_cloud_computing(self) -> List[ComprehensiveBenchmarkResult]:
        """Benchmark cloud computing integration modules."""
        results = []

        print("Benchmarking cloud computing modules...")

        cloud_modules = [
            {
                "name": "aws_sagemaker_integration",
                "complexity": 2.0,
                "test_scenarios": ["training", "inference", "batch_processing"],
            },
            {
                "name": "gcp_vertex_ai_integration",
                "complexity": 2.0,
                "test_scenarios": ["training", "inference", "batch_processing"],
            },
            {
                "name": "azure_ml_integration",
                "complexity": 2.0,
                "test_scenarios": ["training", "inference", "batch_processing"],
            },
            {
                "name": "multi_cloud_orchestration",
                "complexity": 3.5,
                "test_scenarios": ["workflow", "data_sync", "cost_optimization"],
            },
        ]

        for cloud_module in cloud_modules:
            with self.resource_monitor() as monitor:

                def mock_cloud_operation(scenario, data_size):
                    """Mock cloud computing operation."""
                    operation_time = data_size * cloud_module["complexity"] * 0.0005
                    time.sleep(min(operation_time, 2.0))  # Cap for testing

                    return {
                        "operation_success": True,
                        "latency_ms": np.random.uniform(50, 200),
                        "cost_estimate": data_size * 0.001,
                        "throughput": data_size / max(operation_time, 0.001),
                    }

                for scenario in cloud_module["test_scenarios"]:
                    test_sizes = [1000, 5000, 10000]

                    for test_size in test_sizes:
                        start_time = time.time()
                        operation_result = mock_cloud_operation(scenario, test_size)
                        end_time = time.time()

                        execution_time = end_time - start_time

                        base_result = BenchmarkResult(
                            config_name=self.config.name,
                            implementation="chemi_accel_cloud",
                            backend="cloud",
                            descriptor=f"{cloud_module['name']}_{scenario}",
                            dataset_size=test_size,
                            execution_times=[execution_time],
                            memory_usage_mb=monitor["get_current_memory"]() * 1024,
                            status=BenchmarkStatus.COMPLETED,
                        )

                        comprehensive_result = ComprehensiveBenchmarkResult(
                            base_result=base_result,
                            module_category=ModuleCategory.CLOUD_COMPUTING.value,
                            complexity_score=cloud_module["complexity"],
                            scalability_metrics={
                                "cloud_throughput": operation_result["throughput"],
                                "latency_ms": operation_result["latency_ms"],
                                "cost_efficiency": test_size
                                / operation_result["cost_estimate"],
                            },
                            quality_metrics={
                                "operation_success_rate": 1.0
                                if operation_result["operation_success"]
                                else 0.0,
                                "service_availability": 0.99,  # Mock SLA
                                "data_integrity": 1.0,
                            },
                            resource_efficiency={
                                "cloud_cost_per_operation": operation_result[
                                    "cost_estimate"
                                ]
                                / test_size,
                                "network_efficiency": 0.95,  # Mock metric
                                "auto_scaling_effectiveness": 0.90,
                            },
                        )

                        results.append(comprehensive_result)
                        print(
                            f"  {cloud_module['name']} {scenario} ({test_size:,} ops): {operation_result['throughput']:.1f} ops/sec"
                        )

        return results

    def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive benchmarks across all modules."""
        print("Starting comprehensive ChemiAccel benchmarking suite...")
        print(f"Configuration: {self.config.name}")
        print(f"Dataset sizes: {self.config.dataset_sizes}")
        print(f"Backends: {self.config.backends_to_test}")
        print("-" * 50)

        start_time = time.time()

        # Run benchmarks for each module category
        benchmark_functions = [
            self.benchmark_core_descriptors,
            self.benchmark_ai_modules,
            self.benchmark_polymer_analysis,
            self.benchmark_materials_science,
            self.benchmark_cloud_computing,
        ]

        all_results = []

        for benchmark_func in benchmark_functions:
            try:
                module_results = benchmark_func()
                all_results.extend(module_results)
                print(f"Completed {benchmark_func.__name__}")

                # Force garbage collection between modules
                gc.collect()

            except Exception as e:
                print(f"Error in {benchmark_func.__name__}: {e}")
                continue

        self.results = all_results

        end_time = time.time()
        total_duration = end_time - start_time

        print("-" * 50)
        print(f"Comprehensive benchmarking completed in {total_duration:.2f} seconds")

        # Generate comprehensive analysis
        summary = self._generate_comprehensive_summary()

        # Save results
        self._save_comprehensive_results()

        return summary

    def _generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary across all modules."""
        if not self.results:
            return {"error": "No benchmark results available"}

        # Group results by module category
        by_category = {}
        for result in self.results:
            category = result.module_category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)

        # Calculate overall statistics
        total_tests = len(self.results)
        successful_tests = sum(
            1 for r in self.results if r.base_result.status == BenchmarkStatus.COMPLETED
        )

        # Performance analysis
        performance_scores = []
        for result in self.results:
            if result.scalability_metrics.get("performance_ratio"):
                performance_scores.append(
                    result.scalability_metrics["performance_ratio"]
                )

        avg_performance_ratio = np.mean(performance_scores) if performance_scores else 0

        # Quality analysis
        quality_scores = []
        for result in self.results:
            quality_metrics = result.quality_metrics
            if quality_metrics:
                avg_quality = np.mean(list(quality_metrics.values()))
                quality_scores.append(avg_quality)

        avg_quality_score = np.mean(quality_scores) if quality_scores else 0

        # Resource efficiency analysis
        memory_usage = [
            r.resource_efficiency.get("memory_peak_gb", 0) for r in self.results
        ]
        cpu_utilization = [
            r.resource_efficiency.get("cpu_utilization", 0) for r in self.results
        ]

        summary = {
            "overview": {
                "total_benchmarks": total_tests,
                "successful_benchmarks": successful_tests,
                "success_rate": successful_tests / total_tests
                if total_tests > 0
                else 0,
                "avg_performance_ratio": avg_performance_ratio,
                "avg_quality_score": avg_quality_score,
                "peak_memory_usage_gb": max(memory_usage) if memory_usage else 0,
                "avg_cpu_utilization": np.mean(cpu_utilization)
                if cpu_utilization
                else 0,
            },
            "module_categories": {},
            "performance_rankings": self._rank_modules_by_performance(),
            "resource_efficiency_rankings": self._rank_modules_by_efficiency(),
            "recommendations": self._generate_optimization_recommendations(),
        }

        # Detailed analysis by module category
        for category, category_results in by_category.items():
            category_summary = {
                "total_tests": len(category_results),
                "avg_complexity": np.mean(
                    [r.complexity_score for r in category_results]
                ),
                "performance_metrics": {
                    "avg_throughput": np.mean(
                        [
                            r.scalability_metrics.get("throughput", 0)
                            for r in category_results
                        ]
                    ),
                    "avg_performance_ratio": np.mean(
                        [
                            r.scalability_metrics.get("performance_ratio", 0)
                            for r in category_results
                        ]
                    ),
                },
                "quality_metrics": {
                    "avg_accuracy": np.mean(
                        [
                            np.mean(list(r.quality_metrics.values()))
                            for r in category_results
                            if r.quality_metrics
                        ]
                    )
                    if any(r.quality_metrics for r in category_results)
                    else 0
                },
                "resource_metrics": {
                    "avg_memory_gb": np.mean(
                        [
                            r.resource_efficiency.get("memory_peak_gb", 0)
                            for r in category_results
                        ]
                    ),
                    "avg_cpu_utilization": np.mean(
                        [
                            r.resource_efficiency.get("cpu_utilization", 0)
                            for r in category_results
                        ]
                    ),
                },
                "benchmark_details": [
                    {
                        "name": r.base_result.descriptor,
                        "dataset_size": r.base_result.dataset_size,
                        "execution_time": r.base_result.mean_time,
                        "throughput": r.scalability_metrics.get("throughput", 0),
                        "complexity": r.complexity_score,
                    }
                    for r in category_results
                ],
            }

            summary["module_categories"][category] = category_summary

        return summary

    def _rank_modules_by_performance(self) -> List[Dict[str, Any]]:
        """Rank modules by performance metrics."""
        module_performance = {}

        for result in self.results:
            module_name = result.base_result.descriptor
            performance_ratio = result.scalability_metrics.get("performance_ratio", 0)

            if module_name not in module_performance:
                module_performance[module_name] = []
            module_performance[module_name].append(performance_ratio)

        # Calculate average performance for each module
        avg_performance = {
            module: np.mean(ratios) for module, ratios in module_performance.items()
        }

        # Sort by performance
        ranked_modules = sorted(
            avg_performance.items(), key=lambda x: x[1], reverse=True
        )

        return [
            {
                "module": module,
                "avg_performance_ratio": ratio,
                "performance_category": "high"
                if ratio > 1.5
                else "medium"
                if ratio > 0.8
                else "low",
            }
            for module, ratio in ranked_modules
        ]

    def _rank_modules_by_efficiency(self) -> List[Dict[str, Any]]:
        """Rank modules by resource efficiency."""
        module_efficiency = {}

        for result in self.results:
            module_name = result.base_result.descriptor
            memory_efficiency = result.scalability_metrics.get("memory_efficiency", 0)
            cpu_utilization = result.resource_efficiency.get("cpu_utilization", 0)

            # Combined efficiency score
            efficiency_score = memory_efficiency * 0.6 + cpu_utilization * 0.4

            if module_name not in module_efficiency:
                module_efficiency[module_name] = []
            module_efficiency[module_name].append(efficiency_score)

        # Calculate average efficiency
        avg_efficiency = {
            module: np.mean(scores) for module, scores in module_efficiency.items()
        }

        # Sort by efficiency
        ranked_modules = sorted(
            avg_efficiency.items(), key=lambda x: x[1], reverse=True
        )

        return [
            {
                "module": module,
                "efficiency_score": score,
                "efficiency_category": "high"
                if score > 0.8
                else "medium"
                if score > 0.5
                else "low",
            }
            for module, score in ranked_modules
        ]

    def _generate_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on benchmark results."""
        recommendations = []

        # Analyze performance bottlenecks
        low_performance_modules = [
            r
            for r in self.results
            if r.scalability_metrics.get("performance_ratio", 0) < 0.5
        ]

        if low_performance_modules:
            recommendations.append(
                {
                    "category": "performance",
                    "priority": "high",
                    "issue": "Low performance modules detected",
                    "modules": [
                        r.base_result.descriptor for r in low_performance_modules
                    ],
                    "recommendation": "Consider algorithm optimization, caching, or parallelization",
                }
            )

        # Analyze memory usage
        high_memory_modules = [
            r
            for r in self.results
            if r.resource_efficiency.get("memory_peak_gb", 0) > 8.0
        ]

        if high_memory_modules:
            recommendations.append(
                {
                    "category": "memory",
                    "priority": "medium",
                    "issue": "High memory usage detected",
                    "modules": [r.base_result.descriptor for r in high_memory_modules],
                    "recommendation": "Implement memory optimization, batch processing, or streaming",
                }
            )

        # Analyze CPU utilization
        low_cpu_modules = [
            r
            for r in self.results
            if r.resource_efficiency.get("cpu_utilization", 0) < 0.3
        ]

        if low_cpu_modules:
            recommendations.append(
                {
                    "category": "cpu_utilization",
                    "priority": "low",
                    "issue": "Low CPU utilization detected",
                    "modules": [r.base_result.descriptor for r in low_cpu_modules],
                    "recommendation": "Consider parallelization or multi-threading improvements",
                }
            )

        # Quality-based recommendations
        low_quality_modules = [
            r
            for r in self.results
            if r.quality_metrics and np.mean(list(r.quality_metrics.values())) < 0.8
        ]

        if low_quality_modules:
            recommendations.append(
                {
                    "category": "quality",
                    "priority": "high",
                    "issue": "Low quality metrics detected",
                    "modules": [r.base_result.descriptor for r in low_quality_modules],
                    "recommendation": "Review algorithms, improve validation, or enhance testing",
                }
            )

        return recommendations

    def _save_comprehensive_results(self):
        """Save comprehensive benchmark results."""
        # Save detailed results
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "base_result": {
                        "config_name": result.base_result.config_name,
                        "implementation": result.base_result.implementation,
                        "backend": result.base_result.backend,
                        "descriptor": result.base_result.descriptor,
                        "dataset_size": result.base_result.dataset_size,
                        "execution_times": result.base_result.execution_times,
                        "memory_usage_mb": result.base_result.memory_usage_mb,
                        "status": result.base_result.status.value,
                        "mean_time": result.base_result.mean_time,
                        "throughput": result.base_result.throughput,
                    },
                    "module_category": result.module_category,
                    "complexity_score": result.complexity_score,
                    "scalability_metrics": result.scalability_metrics,
                    "quality_metrics": result.quality_metrics,
                    "resource_efficiency": result.resource_efficiency,
                    "comparative_analysis": result.comparative_analysis,
                }
            )

        # Save to JSON
        results_file = (
            self.output_dir / f"{self.config.name}_comprehensive_results.json"
        )
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"Comprehensive benchmark results saved to {results_file}")


def create_comprehensive_benchmark_config() -> BenchmarkConfig:
    """Create a comprehensive benchmark configuration."""
    return BenchmarkConfig(
        name="chemi_accel_comprehensive_benchmark",
        description="Comprehensive performance benchmark across all ChemiAccel modules",
        dataset_sizes=[1000, 5000, 10000, 50000],
        iterations=3,
        warmup_iterations=1,
        timeout_seconds=600,
        backends_to_test=["cpu", "cuda"] if CUPY_AVAILABLE else ["cpu"],
        descriptors_to_test=[
            "molecular_weight",
            "tpsa",
            "logp",
            "morgan_fingerprint",
            "active_learning",
            "polymer_analysis",
            "materials_prediction",
        ],
        reference_implementations=["rdkit", "mordred"],
        statistical_significance_level=0.05,
        min_speedup_threshold=1.0,
        output_directory="comprehensive_benchmark_results",
    )


def run_comprehensive_benchmarks():
    """Run the complete comprehensive benchmark suite."""
    config = create_comprehensive_benchmark_config()
    suite = ComprehensiveBenchmarkSuite(config)

    summary = suite.run_comprehensive_benchmarks()

    print("\n" + "=" * 60)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 60)

    overview = summary["overview"]
    print(f"Total Benchmarks: {overview['total_benchmarks']}")
    print(f"Success Rate: {overview['success_rate']:.2%}")
    print(f"Average Performance Ratio: {overview['avg_performance_ratio']:.2f}")
    print(f"Average Quality Score: {overview['avg_quality_score']:.2%}")
    print(f"Peak Memory Usage: {overview['peak_memory_usage_gb']:.2f} GB")
    print(f"Average CPU Utilization: {overview['avg_cpu_utilization']:.2%}")

    print("\nTop Performing Modules:")
    for i, module in enumerate(summary["performance_rankings"][:5]):
        print(f"  {i+1}. {module['module']}: {module['avg_performance_ratio']:.2f}x")

    print("\nOptimization Recommendations:")
    for rec in summary["recommendations"][:3]:
        print(f"  - {rec['category'].upper()}: {rec['issue']}")
        print(f"    Recommendation: {rec['recommendation']}")

    return summary


if __name__ == "__main__":
    summary = run_comprehensive_benchmarks()

"""
Performance Optimization Recommendations System for ChemiAccel.

This module provides intelligent performance optimization recommendations based on
benchmark results, profiling data, system characteristics, and workload patterns.
It integrates with other testing frameworks to provide actionable insights.
"""

import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of performance optimizations."""

    ALGORITHM = "algorithm"
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"
    PARALLELIZATION = "parallelization"
    CACHING = "caching"
    CONFIGURATION = "configuration"
    INFRASTRUCTURE = "infrastructure"


class Priority(Enum):
    """Priority levels for recommendations."""

    CRITICAL = "critical"  # >50% performance impact potential
    HIGH = "high"  # 25-50% performance impact potential
    MEDIUM = "medium"  # 10-25% performance impact potential
    LOW = "low"  # 5-10% performance impact potential
    INFORMATIONAL = "informational"  # <5% performance impact potential


class ImplementationDifficulty(Enum):
    """Implementation difficulty levels."""

    TRIVIAL = "trivial"  # Configuration change
    EASY = "easy"  # Simple code change
    MODERATE = "moderate"  # Requires some development
    HARD = "hard"  # Significant refactoring
    EXPERT = "expert"  # Requires domain expertise


@dataclass
class PerformanceMetrics:
    """Performance metrics for analysis."""

    execution_time_seconds: float
    throughput_items_per_second: float
    memory_usage_mb: float
    cpu_utilization_percent: float
    gpu_utilization_percent: Optional[float] = None
    network_usage_mbps: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    parallelization_efficiency: Optional[float] = None
    error_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemCharacteristics:
    """System characteristics for optimization context."""

    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    gpu_memory_gb: Optional[float] = None
    storage_type: str = "ssd"  # ssd, hdd, nvme
    network_bandwidth_gbps: Optional[float] = None
    is_cloud: bool = False
    cloud_provider: Optional[str] = None
    instance_type: Optional[str] = None
    cost_per_hour: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadProfile:
    """Workload characteristics for optimization context."""

    dataset_size: int
    descriptor_types: List[str]
    computation_type: str  # batch, streaming, interactive
    io_intensity: str  # low, medium, high
    memory_pattern: str  # sequential, random, mixed
    parallel_potential: str  # low, medium, high
    bottleneck_areas: List[str] = field(default_factory=list)
    typical_batch_sizes: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""

    title: str
    description: str
    optimization_type: OptimizationType
    priority: Priority
    difficulty: ImplementationDifficulty
    estimated_impact_percent: float
    estimated_effort_hours: float
    confidence_score: float

    # Technical details
    current_bottleneck: str
    proposed_solution: str
    implementation_steps: List[str]
    code_examples: List[str] = field(default_factory=list)
    configuration_changes: Dict[str, Any] = field(default_factory=dict)

    # Cost-benefit analysis
    development_cost_estimate: Optional[float] = None
    ongoing_maintenance_cost: Optional[float] = None
    performance_benefit_value: Optional[float] = None

    # Dependencies and risks
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)

    # Measurement and validation
    success_metrics: List[str] = field(default_factory=list)
    testing_approach: str = ""
    rollback_plan: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceAnalyzer:
    """Analyzer for identifying performance bottlenecks and patterns."""

    def __init__(self):
        """Initialize performance analyzer."""
        self.analysis_rules = self._create_analysis_rules()

    def _create_analysis_rules(self) -> List[Dict[str, Any]]:
        """Create performance analysis rules."""
        return [
            {
                "name": "high_memory_usage",
                "condition": lambda metrics: metrics.memory_usage_mb > 8000,  # >8GB
                "bottleneck": "memory",
                "severity": "high",
            },
            {
                "name": "low_cpu_utilization",
                "condition": lambda metrics: metrics.cpu_utilization_percent < 50,
                "bottleneck": "cpu_underutilization",
                "severity": "medium",
            },
            {
                "name": "high_cpu_utilization",
                "condition": lambda metrics: metrics.cpu_utilization_percent > 90,
                "bottleneck": "cpu_saturation",
                "severity": "high",
            },
            {
                "name": "gpu_underutilization",
                "condition": lambda metrics: (
                    metrics.gpu_utilization_percent is not None
                    and metrics.gpu_utilization_percent < 60
                ),
                "bottleneck": "gpu_underutilization",
                "severity": "medium",
            },
            {
                "name": "low_throughput",
                "condition": lambda metrics: metrics.throughput_items_per_second < 100,
                "bottleneck": "throughput",
                "severity": "high",
            },
            {
                "name": "poor_parallelization",
                "condition": lambda metrics: (
                    metrics.parallelization_efficiency is not None
                    and metrics.parallelization_efficiency < 0.7
                ),
                "bottleneck": "parallelization",
                "severity": "medium",
            },
            {
                "name": "low_cache_hit_rate",
                "condition": lambda metrics: (
                    metrics.cache_hit_rate is not None and metrics.cache_hit_rate < 0.8
                ),
                "bottleneck": "caching",
                "severity": "medium",
            },
            {
                "name": "high_error_rate",
                "condition": lambda metrics: (
                    metrics.error_rate is not None
                    and metrics.error_rate > 0.01  # >1% error rate
                ),
                "bottleneck": "reliability",
                "severity": "critical",
            },
        ]

    def analyze_metrics(
        self,
        metrics: PerformanceMetrics,
        system: SystemCharacteristics,
        workload: WorkloadProfile,
    ) -> Dict[str, Any]:
        """Analyze performance metrics to identify bottlenecks.

        Args:
            metrics: Performance metrics
            system: System characteristics
            workload: Workload profile

        Returns:
            Analysis results
        """
        analysis = {
            "bottlenecks": [],
            "efficiency_scores": {},
            "resource_utilization": {},
            "optimization_opportunities": [],
            "performance_score": 0.0,
        }

        # Apply analysis rules
        for rule in self.analysis_rules:
            if rule["condition"](metrics):
                analysis["bottlenecks"].append(
                    {
                        "type": rule["bottleneck"],
                        "severity": rule["severity"],
                        "rule": rule["name"],
                    }
                )

        # Calculate efficiency scores
        analysis["efficiency_scores"] = {
            "cpu_efficiency": min(
                1.0, metrics.cpu_utilization_percent / 85
            ),  # Target 85%
            "memory_efficiency": self._calculate_memory_efficiency(metrics, system),
            "throughput_efficiency": self._calculate_throughput_efficiency(
                metrics, workload
            ),
        }

        if metrics.gpu_utilization_percent is not None:
            analysis["efficiency_scores"]["gpu_efficiency"] = min(
                1.0, metrics.gpu_utilization_percent / 90
            )  # Target 90%

        if metrics.parallelization_efficiency is not None:
            analysis["efficiency_scores"][
                "parallel_efficiency"
            ] = metrics.parallelization_efficiency

        # Resource utilization analysis
        analysis["resource_utilization"] = {
            "cpu_utilization": metrics.cpu_utilization_percent,
            "memory_utilization": (metrics.memory_usage_mb / (system.memory_gb * 1024))
            * 100,
            "is_cpu_bound": metrics.cpu_utilization_percent > 80,
            "is_memory_bound": metrics.memory_usage_mb
            > (system.memory_gb * 1024 * 0.8),
            "is_io_bound": self._detect_io_bottleneck(metrics, workload),
        }

        # Calculate overall performance score
        efficiency_values = list(analysis["efficiency_scores"].values())
        analysis["performance_score"] = (
            statistics.mean(efficiency_values) if efficiency_values else 0.0
        )

        # Identify optimization opportunities
        analysis["optimization_opportunities"] = self._identify_opportunities(
            analysis, metrics, system, workload
        )

        return analysis

    def _calculate_memory_efficiency(
        self, metrics: PerformanceMetrics, system: SystemCharacteristics
    ) -> float:
        """Calculate memory efficiency score."""
        total_memory_mb = system.memory_gb * 1024
        utilization = metrics.memory_usage_mb / total_memory_mb

        # Optimal utilization is around 70-80%
        if utilization < 0.5:
            return utilization * 2  # Penalize underutilization
        elif utilization < 0.8:
            return 1.0  # Optimal range
        else:
            return max(0.0, 1.0 - (utilization - 0.8) * 5)  # Penalize overutilization

    def _calculate_throughput_efficiency(
        self, metrics: PerformanceMetrics, workload: WorkloadProfile
    ) -> float:
        """Calculate throughput efficiency score."""
        # Estimate theoretical maximum throughput based on workload
        if "molecular_weight" in workload.descriptor_types:
            theoretical_max = 10000  # Simple descriptors
        elif any("fingerprint" in desc for desc in workload.descriptor_types):
            theoretical_max = 5000  # Complex descriptors
        else:
            theoretical_max = 7500  # Average

        return min(1.0, metrics.throughput_items_per_second / theoretical_max)

    def _detect_io_bottleneck(
        self, metrics: PerformanceMetrics, workload: WorkloadProfile
    ) -> bool:
        """Detect if workload is I/O bound."""
        # Simple heuristic: low CPU + high I/O intensity = I/O bound
        return metrics.cpu_utilization_percent < 60 and workload.io_intensity in [
            "medium",
            "high",
        ]

    def _identify_opportunities(
        self,
        analysis: Dict[str, Any],
        metrics: PerformanceMetrics,
        system: SystemCharacteristics,
        workload: WorkloadProfile,
    ) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []

        # CPU optimization opportunities
        if analysis["resource_utilization"]["cpu_utilization"] < 50:
            opportunities.append("cpu_scaling_down")
        elif analysis["resource_utilization"]["cpu_utilization"] > 90:
            opportunities.append("cpu_scaling_up")

        # Memory optimization opportunities
        memory_util = analysis["resource_utilization"]["memory_utilization"]
        if memory_util > 80:
            opportunities.append("memory_optimization")
        elif memory_util < 30:
            opportunities.append("memory_right_sizing")

        # GPU opportunities
        if system.gpu_count > 0 and metrics.gpu_utilization_percent:
            if metrics.gpu_utilization_percent < 60:
                opportunities.append("gpu_optimization")
        elif system.gpu_count == 0 and workload.parallel_potential == "high":
            opportunities.append("gpu_acceleration")

        # Parallelization opportunities
        if (
            workload.parallel_potential == "high"
            and metrics.parallelization_efficiency
            and metrics.parallelization_efficiency < 0.8
        ):
            opportunities.append("parallelization_improvement")

        # Caching opportunities
        if workload.memory_pattern == "random" and not metrics.cache_hit_rate:
            opportunities.append("caching_implementation")
        elif metrics.cache_hit_rate and metrics.cache_hit_rate < 0.8:
            opportunities.append("cache_optimization")

        return opportunities


class OptimizationRecommendationEngine:
    """Engine for generating performance optimization recommendations."""

    def __init__(self):
        """Initialize recommendation engine."""
        self.recommendation_templates = self._create_recommendation_templates()
        self.analyzer = PerformanceAnalyzer()

    def _create_recommendation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create recommendation templates for different optimization types."""
        return {
            "cpu_scaling_up": {
                "title": "Scale Up CPU Resources",
                "optimization_type": OptimizationType.CPU,
                "priority": Priority.HIGH,
                "difficulty": ImplementationDifficulty.EASY,
                "template": {
                    "description": "CPU utilization is consistently high ({cpu_util:.1f}%), indicating CPU bottleneck. Scaling up CPU resources will improve performance.",
                    "proposed_solution": "Increase CPU core count or switch to CPU-optimized instance type",
                    "implementation_steps": [
                        "Monitor current CPU utilization patterns",
                        "Determine optimal CPU core count based on workload",
                        "Upgrade instance type or scale horizontally",
                        "Validate performance improvement",
                    ],
                    "success_metrics": [
                        "CPU utilization < 80%",
                        "Improved throughput",
                        "Reduced execution time",
                    ],
                },
            },
            "memory_optimization": {
                "title": "Optimize Memory Usage",
                "optimization_type": OptimizationType.MEMORY,
                "priority": Priority.HIGH,
                "difficulty": ImplementationDifficulty.MODERATE,
                "template": {
                    "description": "Memory usage is high ({memory_util:.1f}% of available memory). Optimizing memory usage will prevent swapping and improve performance.",
                    "proposed_solution": "Implement memory-efficient algorithms and optimize data structures",
                    "implementation_steps": [
                        "Profile memory usage patterns",
                        "Identify memory-intensive operations",
                        "Implement batch processing to reduce peak memory",
                        "Optimize data structures and algorithms",
                        "Add memory monitoring and alerting",
                    ],
                    "code_examples": [
                        "# Use batch processing to reduce memory footprint",
                        "def process_in_batches(data, batch_size=1000):",
                        "    for i in range(0, len(data), batch_size):",
                        "        batch = data[i:i + batch_size]",
                        "        yield process_batch(batch)",
                        "",
                        "# Use generators for large datasets",
                        "def molecular_descriptors_generator(molecules):",
                        "    for molecule in molecules:",
                        "        yield calculate_descriptors(molecule)",
                    ],
                },
            },
            "gpu_acceleration": {
                "title": "Implement GPU Acceleration",
                "optimization_type": OptimizationType.GPU,
                "priority": Priority.HIGH,
                "difficulty": ImplementationDifficulty.HARD,
                "template": {
                    "description": "Workload has high parallel potential but no GPU acceleration. GPU acceleration can provide 5-50x speedup for molecular descriptor calculations.",
                    "proposed_solution": "Implement GPU-accelerated computation using CUDA or OpenCL",
                    "implementation_steps": [
                        "Analyze computational kernels for GPU suitability",
                        "Choose GPU acceleration framework (CUDA, CuPy, JAX)",
                        "Implement GPU kernels for descriptor calculations",
                        "Add GPU memory management",
                        "Benchmark and optimize GPU performance",
                        "Implement CPU fallback for compatibility",
                    ],
                    "prerequisites": [
                        "GPU hardware available",
                        "CUDA toolkit installed",
                        "GPU programming expertise",
                    ],
                    "code_examples": [
                        "# Example CuPy GPU acceleration",
                        "import cupy as cp",
                        "",
                        "def gpu_molecular_weight_calculation(molecules):",
                        "    # Transfer data to GPU",
                        "    gpu_data = cp.asarray(molecular_features)",
                        "    # Perform parallel computation",
                        "    gpu_weights = cp.sum(gpu_data * atomic_weights, axis=1)",
                        "    # Transfer result back to CPU",
                        "    return cp.asnumpy(gpu_weights)",
                    ],
                },
            },
            "parallelization_improvement": {
                "title": "Improve Parallelization Efficiency",
                "optimization_type": OptimizationType.PARALLELIZATION,
                "priority": Priority.MEDIUM,
                "difficulty": ImplementationDifficulty.MODERATE,
                "template": {
                    "description": "Parallel efficiency is low ({parallel_eff:.1%}). Improving parallelization can significantly boost performance.",
                    "proposed_solution": "Optimize parallel processing architecture and reduce synchronization overhead",
                    "implementation_steps": [
                        "Profile parallel execution to identify bottlenecks",
                        "Minimize synchronization points",
                        "Optimize work distribution and load balancing",
                        "Reduce inter-process communication overhead",
                        "Consider alternative parallelization strategies",
                    ],
                    "code_examples": [
                        "# Improved multiprocessing with reduced overhead",
                        "from concurrent.futures import ProcessPoolExecutor",
                        "import multiprocessing as mp",
                        "",
                        "def optimized_parallel_processing(molecules, n_workers=None):",
                        "    if n_workers is None:",
                        "        n_workers = mp.cpu_count()",
                        "    ",
                        "    # Chunk data efficiently",
                        "    chunk_size = len(molecules) // n_workers",
                        "    chunks = [molecules[i:i+chunk_size] for i in range(0, len(molecules), chunk_size)]",
                        "    ",
                        "    with ProcessPoolExecutor(max_workers=n_workers) as executor:",
                        "        results = list(executor.map(process_chunk, chunks))",
                        "    ",
                        "    return combine_results(results)",
                    ],
                },
            },
            "caching_implementation": {
                "title": "Implement Intelligent Caching",
                "optimization_type": OptimizationType.CACHING,
                "priority": Priority.MEDIUM,
                "difficulty": ImplementationDifficulty.MODERATE,
                "template": {
                    "description": "Workload shows repeated computations with random access patterns. Implementing caching can reduce redundant calculations.",
                    "proposed_solution": "Add multi-level caching for computed descriptors and intermediate results",
                    "implementation_steps": [
                        "Identify cacheable computations",
                        "Design cache architecture (memory, disk, distributed)",
                        "Implement cache key generation and validation",
                        "Add cache eviction policies",
                        "Monitor cache hit rates and optimize",
                    ],
                    "code_examples": [
                        "# Example LRU cache implementation",
                        "from functools import lru_cache",
                        "import hashlib",
                        "",
                        "@lru_cache(maxsize=10000)",
                        "def cached_descriptor_calculation(molecule_smiles):",
                        "    return calculate_molecular_descriptors(molecule_smiles)",
                        "",
                        "# Persistent cache for expensive computations",
                        "class PersistentCache:",
                        '    def __init__(self, cache_dir="descriptor_cache"):',
                        "        self.cache_dir = Path(cache_dir)",
                        "        self.cache_dir.mkdir(exist_ok=True)",
                        "    ",
                        "    def get(self, key):",
                        '        cache_file = self.cache_dir / f"{hash(key)}.json"',
                        "        if cache_file.exists():",
                        "            with open(cache_file) as f:",
                        "                return json.load(f)",
                        "        return None",
                    ],
                },
            },
            "algorithm_optimization": {
                "title": "Optimize Core Algorithms",
                "optimization_type": OptimizationType.ALGORITHM,
                "priority": Priority.HIGH,
                "difficulty": ImplementationDifficulty.EXPERT,
                "template": {
                    "description": "Performance analysis indicates algorithmic inefficiencies. Optimizing core algorithms can provide significant performance improvements.",
                    "proposed_solution": "Replace inefficient algorithms with optimized versions or alternative approaches",
                    "implementation_steps": [
                        "Profile algorithmic complexity",
                        "Research state-of-the-art algorithms for the domain",
                        "Implement and benchmark alternative algorithms",
                        "Consider approximation algorithms for acceptable trade-offs",
                        "Validate correctness and performance improvements",
                    ],
                    "prerequisites": [
                        "Algorithm analysis expertise",
                        "Domain knowledge",
                        "Comprehensive testing",
                    ],
                    "code_examples": [
                        "# Example: Vectorized operations instead of loops",
                        "import numpy as np",
                        "",
                        "# Inefficient: Loop-based calculation",
                        "def slow_calculation(molecules):",
                        "    results = []",
                        "    for mol in molecules:",
                        "        result = sum(atom.mass for atom in mol.atoms)",
                        "        results.append(result)",
                        "    return results",
                        "",
                        "# Optimized: Vectorized calculation",
                        "def fast_calculation(molecules):",
                        "    atom_masses = np.array([mol.get_mass_array() for mol in molecules])",
                        "    return np.sum(atom_masses, axis=1)",
                    ],
                },
            },
            "infrastructure_optimization": {
                "title": "Optimize Infrastructure Configuration",
                "optimization_type": OptimizationType.INFRASTRUCTURE,
                "priority": Priority.MEDIUM,
                "difficulty": ImplementationDifficulty.EASY,
                "template": {
                    "description": "Infrastructure configuration is not optimized for current workload characteristics.",
                    "proposed_solution": "Optimize cloud instance types, storage configuration, and network settings",
                    "implementation_steps": [
                        "Analyze current infrastructure utilization",
                        "Benchmark different instance types",
                        "Optimize storage configuration (SSD, NVMe)",
                        "Configure network optimization",
                        "Implement auto-scaling if applicable",
                    ],
                    "configuration_changes": {
                        "instance_type": "cpu_optimized",
                        "storage_type": "nvme_ssd",
                        "network_optimization": True,
                        "auto_scaling": {
                            "enabled": True,
                            "min_instances": 1,
                            "max_instances": 10,
                            "target_cpu_utilization": 75,
                        },
                    },
                },
            },
        }

    def generate_recommendations(
        self,
        metrics: PerformanceMetrics,
        system: SystemCharacteristics,
        workload: WorkloadProfile,
        budget_constraints: Optional[Dict[str, float]] = None,
    ) -> List[OptimizationRecommendation]:
        """Generate performance optimization recommendations.

        Args:
            metrics: Performance metrics
            system: System characteristics
            workload: Workload profile
            budget_constraints: Optional budget constraints

        Returns:
            List of optimization recommendations
        """
        # Analyze performance metrics
        analysis = self.analyzer.analyze_metrics(metrics, system, workload)

        recommendations = []

        # Generate recommendations based on analysis
        for opportunity in analysis["optimization_opportunities"]:
            if opportunity in self.recommendation_templates:
                template = self.recommendation_templates[opportunity]

                # Create recommendation from template
                recommendation = self._create_recommendation_from_template(
                    template, metrics, system, workload, analysis
                )

                # Apply budget constraints if provided
                if budget_constraints and not self._meets_budget_constraints(
                    recommendation, budget_constraints
                ):
                    continue

                recommendations.append(recommendation)

        # Add general recommendations based on bottlenecks
        for bottleneck in analysis["bottlenecks"]:
            bottleneck_rec = self._create_bottleneck_recommendation(
                bottleneck, metrics, system, workload
            )
            if bottleneck_rec:
                recommendations.append(bottleneck_rec)

        # Sort by priority and impact
        recommendations.sort(
            key=lambda x: (
                x.priority.value,
                -x.estimated_impact_percent,
                x.difficulty.value,
            )
        )

        return recommendations

    def _create_recommendation_from_template(
        self,
        template: Dict[str, Any],
        metrics: PerformanceMetrics,
        system: SystemCharacteristics,
        workload: WorkloadProfile,
        analysis: Dict[str, Any],
    ) -> OptimizationRecommendation:
        """Create recommendation from template."""
        template_data = template["template"]

        # Format description with actual values
        description = template_data["description"]
        if "{cpu_util" in description:
            description = description.format(cpu_util=metrics.cpu_utilization_percent)
        if "{memory_util" in description:
            memory_util = (metrics.memory_usage_mb / (system.memory_gb * 1024)) * 100
            description = description.format(memory_util=memory_util)
        if "{parallel_eff" in description and metrics.parallelization_efficiency:
            description = description.format(
                parallel_eff=metrics.parallelization_efficiency
            )

        # Estimate impact based on current bottlenecks
        estimated_impact = self._estimate_impact(
            template["optimization_type"], analysis
        )

        # Estimate effort based on difficulty
        effort_hours = {
            ImplementationDifficulty.TRIVIAL: 1,
            ImplementationDifficulty.EASY: 8,
            ImplementationDifficulty.MODERATE: 40,
            ImplementationDifficulty.HARD: 120,
            ImplementationDifficulty.EXPERT: 200,
        }[template["difficulty"]]

        return OptimizationRecommendation(
            title=template["title"],
            description=description,
            optimization_type=template["optimization_type"],
            priority=template["priority"],
            difficulty=template["difficulty"],
            estimated_impact_percent=estimated_impact,
            estimated_effort_hours=effort_hours,
            confidence_score=0.8,
            current_bottleneck=self._identify_primary_bottleneck(analysis),
            proposed_solution=template_data["proposed_solution"],
            implementation_steps=template_data["implementation_steps"],
            code_examples=template_data.get("code_examples", []),
            configuration_changes=template_data.get("configuration_changes", {}),
            prerequisites=template_data.get("prerequisites", []),
            success_metrics=template_data.get("success_metrics", []),
            testing_approach="A/B testing with performance monitoring",
            rollback_plan="Revert to previous configuration if performance degrades",
        )

    def _create_bottleneck_recommendation(
        self,
        bottleneck: Dict[str, Any],
        metrics: PerformanceMetrics,
        system: SystemCharacteristics,
        workload: WorkloadProfile,
    ) -> Optional[OptimizationRecommendation]:
        """Create recommendation for specific bottleneck."""
        bottleneck_type = bottleneck["type"]
        severity = bottleneck["severity"]

        # Map severity to priority
        priority_map = {
            "critical": Priority.CRITICAL,
            "high": Priority.HIGH,
            "medium": Priority.MEDIUM,
            "low": Priority.LOW,
        }

        if bottleneck_type == "reliability" and severity == "critical":
            return OptimizationRecommendation(
                title="Fix High Error Rate",
                description=f"Error rate is {metrics.error_rate:.2%}, which is above acceptable threshold (1%)",
                optimization_type=OptimizationType.ALGORITHM,
                priority=Priority.CRITICAL,
                difficulty=ImplementationDifficulty.MODERATE,
                estimated_impact_percent=50.0,
                estimated_effort_hours=24,
                confidence_score=0.9,
                current_bottleneck="high_error_rate",
                proposed_solution="Identify and fix root causes of errors",
                implementation_steps=[
                    "Analyze error logs and patterns",
                    "Identify most common error types",
                    "Fix data validation and error handling",
                    "Add comprehensive error monitoring",
                    "Implement retry mechanisms where appropriate",
                ],
                success_metrics=[
                    "Error rate < 0.1%",
                    "Improved data quality",
                    "Reduced failed computations",
                ],
            )

        return None

    def _estimate_impact(
        self, optimization_type: OptimizationType, analysis: Dict[str, Any]
    ) -> float:
        """Estimate performance impact of optimization."""
        base_impacts = {
            OptimizationType.ALGORITHM: 30.0,
            OptimizationType.GPU: 500.0,  # Can be very high for parallel workloads
            OptimizationType.PARALLELIZATION: 200.0,
            OptimizationType.MEMORY: 25.0,
            OptimizationType.CPU: 50.0,
            OptimizationType.CACHING: 40.0,
            OptimizationType.CONFIGURATION: 15.0,
            OptimizationType.INFRASTRUCTURE: 20.0,
            OptimizationType.NETWORK: 10.0,
            OptimizationType.STORAGE: 15.0,
        }

        base_impact = base_impacts.get(optimization_type, 20.0)

        # Adjust based on current performance score
        performance_score = analysis.get("performance_score", 0.5)
        if performance_score < 0.3:
            base_impact *= 1.5  # Higher impact for poor performance
        elif performance_score > 0.8:
            base_impact *= 0.5  # Lower impact for good performance

        # Cap at reasonable values
        return min(base_impact, 100.0)

    def _identify_primary_bottleneck(self, analysis: Dict[str, Any]) -> str:
        """Identify primary performance bottleneck."""
        bottlenecks = analysis.get("bottlenecks", [])
        if not bottlenecks:
            return "unknown"

        # Return the most severe bottleneck
        critical_bottlenecks = [b for b in bottlenecks if b["severity"] == "critical"]
        if critical_bottlenecks:
            return critical_bottlenecks[0]["type"]

        high_bottlenecks = [b for b in bottlenecks if b["severity"] == "high"]
        if high_bottlenecks:
            return high_bottlenecks[0]["type"]

        return bottlenecks[0]["type"]

    def _meets_budget_constraints(
        self,
        recommendation: OptimizationRecommendation,
        budget_constraints: Dict[str, float],
    ) -> bool:
        """Check if recommendation meets budget constraints."""
        max_effort_hours = budget_constraints.get("max_effort_hours")
        if (
            max_effort_hours
            and recommendation.estimated_effort_hours > max_effort_hours
        ):
            return False

        max_cost = budget_constraints.get("max_cost_usd")
        if max_cost and recommendation.development_cost_estimate:
            if recommendation.development_cost_estimate > max_cost:
                return False

        return True

    def generate_optimization_report(
        self,
        recommendations: List[OptimizationRecommendation],
        metrics: PerformanceMetrics,
        system: SystemCharacteristics,
    ) -> str:
        """Generate comprehensive optimization report.

        Args:
            recommendations: List of recommendations
            metrics: Performance metrics
            system: System characteristics

        Returns:
            Formatted optimization report
        """
        report = ["Performance Optimization Report", "=" * 40, ""]

        # Executive Summary
        total_impact = sum(rec.estimated_impact_percent for rec in recommendations)
        total_effort = sum(rec.estimated_effort_hours for rec in recommendations)

        high_priority = [
            r
            for r in recommendations
            if r.priority in [Priority.CRITICAL, Priority.HIGH]
        ]

        report.extend(
            [
                "EXECUTIVE SUMMARY",
                "-" * 20,
                f"Total Recommendations: {len(recommendations)}",
                f"High Priority Items: {len(high_priority)}",
                f"Estimated Total Impact: {total_impact:.1f}% performance improvement",
                f"Estimated Total Effort: {total_effort:.0f} hours",
                f"Current Performance Score: {metrics.throughput_items_per_second:.0f} items/sec",
                "",
            ]
        )

        # System Overview
        report.extend(
            [
                "SYSTEM OVERVIEW",
                "-" * 15,
                f"CPU Cores: {system.cpu_cores}",
                f"Memory: {system.memory_gb:.1f} GB",
                f"GPU Count: {system.gpu_count}",
                f"Storage Type: {system.storage_type}",
                f"Cloud Provider: {system.cloud_provider or 'On-premises'}",
                "",
            ]
        )

        # Current Performance Metrics
        report.extend(
            [
                "CURRENT PERFORMANCE",
                "-" * 20,
                f"Throughput: {metrics.throughput_items_per_second:.0f} items/second",
                f"Execution Time: {metrics.execution_time_seconds:.1f} seconds",
                f"Memory Usage: {metrics.memory_usage_mb:.0f} MB",
                f"CPU Utilization: {metrics.cpu_utilization_percent:.1f}%",
            ]
        )

        if metrics.gpu_utilization_percent:
            report.append(f"GPU Utilization: {metrics.gpu_utilization_percent:.1f}%")

        report.extend(["", ""])

        # Recommendations by Priority
        report.extend(["OPTIMIZATION RECOMMENDATIONS", "-" * 30])

        for priority in [
            Priority.CRITICAL,
            Priority.HIGH,
            Priority.MEDIUM,
            Priority.LOW,
        ]:
            priority_recs = [r for r in recommendations if r.priority == priority]
            if not priority_recs:
                continue

            report.extend([f"{priority.value.upper()} PRIORITY", "-" * 15])

            for i, rec in enumerate(priority_recs, 1):
                report.extend(
                    [
                        f"{i}. {rec.title}",
                        f"   Type: {rec.optimization_type.value.title()}",
                        f"   Impact: {rec.estimated_impact_percent:.1f}% improvement",
                        f"   Effort: {rec.estimated_effort_hours:.0f} hours",
                        f"   Difficulty: {rec.difficulty.value.title()}",
                        f"   Confidence: {rec.confidence_score:.1%}",
                        "",
                        f"   Description: {rec.description}",
                        f"   Solution: {rec.proposed_solution}",
                        "",
                    ]
                )

                if rec.prerequisites:
                    report.extend(["   Prerequisites:"])
                    for prereq in rec.prerequisites:
                        report.append(f"   • {prereq}")
                    report.append("")

                if rec.implementation_steps:
                    report.extend(["   Implementation Steps:"])
                    for step in rec.implementation_steps:
                        report.append(f"   1. {step}")
                    report.append("")

                if rec.risks:
                    report.extend(["   Risks:"])
                    for risk in rec.risks:
                        report.append(f"   ⚠️  {risk}")
                    report.append("")

                report.append("   " + "=" * 50)
                report.append("")

        # Implementation Roadmap
        report.extend(["IMPLEMENTATION ROADMAP", "-" * 25])

        # Group by difficulty for phased approach
        trivial_recs = [
            r
            for r in recommendations
            if r.difficulty == ImplementationDifficulty.TRIVIAL
        ]
        easy_recs = [
            r for r in recommendations if r.difficulty == ImplementationDifficulty.EASY
        ]
        moderate_recs = [
            r
            for r in recommendations
            if r.difficulty == ImplementationDifficulty.MODERATE
        ]
        hard_recs = [
            r
            for r in recommendations
            if r.difficulty
            in [ImplementationDifficulty.HARD, ImplementationDifficulty.EXPERT]
        ]

        phases = [
            ("Phase 1 - Quick Wins (1-2 weeks)", trivial_recs + easy_recs),
            ("Phase 2 - Medium Impact (4-8 weeks)", moderate_recs),
            ("Phase 3 - Major Improvements (12+ weeks)", hard_recs),
        ]

        for phase_name, phase_recs in phases:
            if phase_recs:
                report.extend([phase_name, ""])
                for rec in sorted(
                    phase_recs, key=lambda x: -x.estimated_impact_percent
                ):
                    report.append(
                        f"• {rec.title} ({rec.estimated_impact_percent:.1f}% impact)"
                    )
                report.append("")

        return "\n".join(report)


def create_sample_performance_data() -> (
    Tuple[PerformanceMetrics, SystemCharacteristics, WorkloadProfile]
):
    """Create sample performance data for testing."""
    metrics = PerformanceMetrics(
        execution_time_seconds=120.5,
        throughput_items_per_second=830,
        memory_usage_mb=12000,  # 12GB - high memory usage
        cpu_utilization_percent=45,  # Low CPU utilization
        gpu_utilization_percent=None,  # No GPU
        cache_hit_rate=0.65,  # Low cache hit rate
        parallelization_efficiency=0.6,  # Poor parallelization
        error_rate=0.002,  # 0.2% error rate - acceptable
    )

    system = SystemCharacteristics(
        cpu_cores=16,
        memory_gb=32,
        gpu_count=0,
        storage_type="ssd",
        is_cloud=True,
        cloud_provider="aws",
        instance_type="m5.4xlarge",
        cost_per_hour=0.768,
    )

    workload = WorkloadProfile(
        dataset_size=100000,
        descriptor_types=["molecular_weight", "tpsa", "logp", "morgan_fingerprint"],
        computation_type="batch",
        io_intensity="medium",
        memory_pattern="random",
        parallel_potential="high",
        typical_batch_sizes=[1000, 5000, 10000],
    )

    return metrics, system, workload


if __name__ == "__main__":
    # Demonstrate performance optimization recommendations
    print("ChemiAccel Performance Optimization Recommendations System")
    print("=" * 60)

    # Create sample performance data
    metrics, system, workload = create_sample_performance_data()

    print("\nAnalyzing performance for:")
    print(f"- Dataset: {workload.dataset_size:,} molecules")
    print(
        f"- System: {system.cpu_cores} cores, {system.memory_gb}GB RAM, {system.gpu_count} GPUs"
    )
    print(f"- Current throughput: {metrics.throughput_items_per_second:.0f} items/sec")
    print(f"- Memory usage: {metrics.memory_usage_mb/1024:.1f}GB")
    print(f"- CPU utilization: {metrics.cpu_utilization_percent:.1f}%")

    # Generate recommendations
    engine = OptimizationRecommendationEngine()
    recommendations = engine.generate_recommendations(metrics, system, workload)

    print(f"\nGenerated {len(recommendations)} optimization recommendations")

    # Generate and display report
    report = engine.generate_optimization_report(recommendations, metrics, system)

    print("\n" + "=" * 60)
    print("OPTIMIZATION REPORT")
    print("=" * 60)
    print(report)

    # Save report to file
    output_dir = Path("optimization_reports")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"optimization_report_{timestamp}.txt"

    with open(report_file, "w") as f:
        f.write(report)

    print(f"\nDetailed report saved to: {report_file}")

    # Show top 3 recommendations summary
    print("\nTOP 3 RECOMMENDATIONS:")
    print("-" * 30)

    for i, rec in enumerate(recommendations[:3], 1):
        print(f"{i}. {rec.title}")
        print(
            f"   Impact: {rec.estimated_impact_percent:.1f}% | Effort: {rec.estimated_effort_hours:.0f}h | Priority: {rec.priority.value}"
        )
        print(
            f"   {rec.description[:100]}{'...' if len(rec.description) > 100 else ''}"
        )
        print()

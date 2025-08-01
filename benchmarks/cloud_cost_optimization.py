"""
Cloud Cost Optimization Analysis for ChemiAccel.

This module provides comprehensive cost analysis and optimization recommendations
for running ChemiAccel workloads across different cloud providers (AWS, GCP, Azure)
with various compute configurations.
"""

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class InstanceType(Enum):
    """Types of compute instances."""

    CPU_OPTIMIZED = "cpu_optimized"
    MEMORY_OPTIMIZED = "memory_optimized"
    GPU_ACCELERATED = "gpu_accelerated"
    GENERAL_PURPOSE = "general_purpose"
    SPOT_INSTANCE = "spot_instance"


class OptimizationObjective(Enum):
    """Cost optimization objectives."""

    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_COST_EFFICIENCY = "maximize_cost_efficiency"
    BALANCED = "balanced"


@dataclass
class InstanceConfiguration:
    """Cloud instance configuration."""

    provider: CloudProvider
    instance_type: str
    instance_category: InstanceType
    vcpus: int
    memory_gb: float
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    storage_gb: int = 100
    hourly_cost_usd: float = 0.0
    spot_price_usd: Optional[float] = None
    availability_zones: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadRequirements:
    """Workload resource requirements."""

    molecules_count: int
    descriptors_count: int
    estimated_memory_gb: float
    estimated_compute_hours: float
    gpu_required: bool = False
    parallel_workers: int = 1
    storage_gb: int = 10
    network_intensive: bool = False
    fault_tolerance_required: bool = True
    deadline_hours: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostEstimate:
    """Cost estimate for a workload configuration."""

    configuration: InstanceConfiguration
    workload: WorkloadRequirements
    estimated_runtime_hours: float
    compute_cost_usd: float
    storage_cost_usd: float
    network_cost_usd: float
    total_cost_usd: float
    cost_per_molecule: float
    utilization_efficiency: float
    spot_savings_potential: Optional[float] = None
    preemption_risk: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation."""

    recommended_config: InstanceConfiguration
    estimated_cost: CostEstimate
    cost_savings_usd: Optional[float] = None
    cost_savings_percent: Optional[float] = None
    performance_impact: str = "neutral"  # "positive", "neutral", "negative"
    confidence_score: float = 0.8
    reasoning: List[str] = field(default_factory=list)
    trade_offs: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)


class CloudCostOptimizer:
    """Cloud cost optimization analyzer for ChemiAccel workloads."""

    def __init__(self, pricing_data_file: Optional[str] = None):
        """Initialize cloud cost optimizer.

        Args:
            pricing_data_file: Path to cloud pricing data file
        """
        self.pricing_data = self._load_pricing_data(pricing_data_file)
        self.instance_configs = self._create_instance_configurations()

        # Performance benchmarks (molecules per hour per vCPU)
        self.performance_benchmarks = {
            InstanceType.CPU_OPTIMIZED: {
                "molecules_per_vcpu_hour": 10000,
                "memory_per_molecule_mb": 2.0,
                "efficiency_factor": 1.2,
            },
            InstanceType.MEMORY_OPTIMIZED: {
                "molecules_per_vcpu_hour": 8000,
                "memory_per_molecule_mb": 4.0,
                "efficiency_factor": 1.0,
            },
            InstanceType.GPU_ACCELERATED: {
                "molecules_per_vcpu_hour": 50000,  # GPU acceleration
                "memory_per_molecule_mb": 1.5,
                "efficiency_factor": 2.0,
            },
            InstanceType.GENERAL_PURPOSE: {
                "molecules_per_vcpu_hour": 7500,
                "memory_per_molecule_mb": 2.5,
                "efficiency_factor": 1.0,
            },
            InstanceType.SPOT_INSTANCE: {
                "molecules_per_vcpu_hour": 7500,  # Same as general purpose
                "memory_per_molecule_mb": 2.5,
                "efficiency_factor": 0.9,  # Slight overhead from interruptions
            },
        }

    def _load_pricing_data(self, pricing_file: Optional[str]) -> Dict[str, Any]:
        """Load cloud pricing data."""
        if pricing_file and Path(pricing_file).exists():
            with open(pricing_file) as f:
                return json.load(f)

        # Default pricing data (approximate, as of 2024)
        return {
            "aws": {
                "compute": {
                    "c5.large": {"vcpus": 2, "memory_gb": 4, "hourly_usd": 0.085},
                    "c5.xlarge": {"vcpus": 4, "memory_gb": 8, "hourly_usd": 0.17},
                    "c5.2xlarge": {"vcpus": 8, "memory_gb": 16, "hourly_usd": 0.34},
                    "c5.4xlarge": {"vcpus": 16, "memory_gb": 32, "hourly_usd": 0.68},
                    "m5.large": {"vcpus": 2, "memory_gb": 8, "hourly_usd": 0.096},
                    "m5.xlarge": {"vcpus": 4, "memory_gb": 16, "hourly_usd": 0.192},
                    "m5.2xlarge": {"vcpus": 8, "memory_gb": 32, "hourly_usd": 0.384},
                    "r5.large": {"vcpus": 2, "memory_gb": 16, "hourly_usd": 0.126},
                    "r5.xlarge": {"vcpus": 4, "memory_gb": 32, "hourly_usd": 0.252},
                    "p3.2xlarge": {
                        "vcpus": 8,
                        "memory_gb": 61,
                        "hourly_usd": 3.06,
                        "gpu_count": 1,
                        "gpu_type": "V100",
                    },
                    "p4d.24xlarge": {
                        "vcpus": 96,
                        "memory_gb": 1152,
                        "hourly_usd": 32.77,
                        "gpu_count": 8,
                        "gpu_type": "A100",
                    },
                },
                "storage_gb_month": 0.10,
                "network_gb": 0.09,
            },
            "gcp": {
                "compute": {
                    "c2-standard-4": {"vcpus": 4, "memory_gb": 16, "hourly_usd": 0.168},
                    "c2-standard-8": {"vcpus": 8, "memory_gb": 32, "hourly_usd": 0.336},
                    "c2-standard-16": {
                        "vcpus": 16,
                        "memory_gb": 64,
                        "hourly_usd": 0.672,
                    },
                    "n1-standard-4": {"vcpus": 4, "memory_gb": 15, "hourly_usd": 0.19},
                    "n1-standard-8": {"vcpus": 8, "memory_gb": 30, "hourly_usd": 0.38},
                    "n1-highmem-4": {"vcpus": 4, "memory_gb": 26, "hourly_usd": 0.237},
                    "n1-highmem-8": {"vcpus": 8, "memory_gb": 52, "hourly_usd": 0.474},
                    "n1-standard-4-gpu": {
                        "vcpus": 4,
                        "memory_gb": 15,
                        "hourly_usd": 0.82,
                        "gpu_count": 1,
                        "gpu_type": "T4",
                    },
                    "a2-highgpu-1g": {
                        "vcpus": 12,
                        "memory_gb": 85,
                        "hourly_usd": 3.67,
                        "gpu_count": 1,
                        "gpu_type": "A100",
                    },
                },
                "storage_gb_month": 0.08,
                "network_gb": 0.12,
            },
            "azure": {
                "compute": {
                    "Standard_F4s_v2": {
                        "vcpus": 4,
                        "memory_gb": 8,
                        "hourly_usd": 0.169,
                    },
                    "Standard_F8s_v2": {
                        "vcpus": 8,
                        "memory_gb": 16,
                        "hourly_usd": 0.338,
                    },
                    "Standard_D4s_v3": {
                        "vcpus": 4,
                        "memory_gb": 16,
                        "hourly_usd": 0.192,
                    },
                    "Standard_D8s_v3": {
                        "vcpus": 8,
                        "memory_gb": 32,
                        "hourly_usd": 0.384,
                    },
                    "Standard_E4s_v3": {
                        "vcpus": 4,
                        "memory_gb": 32,
                        "hourly_usd": 0.252,
                    },
                    "Standard_E8s_v3": {
                        "vcpus": 8,
                        "memory_gb": 64,
                        "hourly_usd": 0.504,
                    },
                    "Standard_NC6s_v3": {
                        "vcpus": 6,
                        "memory_gb": 112,
                        "hourly_usd": 3.06,
                        "gpu_count": 1,
                        "gpu_type": "V100",
                    },
                    "Standard_ND40rs_v2": {
                        "vcpus": 40,
                        "memory_gb": 672,
                        "hourly_usd": 22.03,
                        "gpu_count": 8,
                        "gpu_type": "V100",
                    },
                },
                "storage_gb_month": 0.095,
                "network_gb": 0.087,
            },
        }

    def _create_instance_configurations(self) -> List[InstanceConfiguration]:
        """Create instance configurations from pricing data."""
        configs = []

        for provider_name, provider_data in self.pricing_data.items():
            provider = CloudProvider(provider_name)

            for instance_name, instance_data in provider_data["compute"].items():
                # Determine instance category
                if (
                    "gpu" in instance_name.lower()
                    or instance_data.get("gpu_count", 0) > 0
                ):
                    category = InstanceType.GPU_ACCELERATED
                elif (
                    "highmem" in instance_name.lower()
                    or "memory" in instance_name.lower()
                ):
                    category = InstanceType.MEMORY_OPTIMIZED
                elif instance_name.startswith(("c", "f")):  # Compute optimized naming
                    category = InstanceType.CPU_OPTIMIZED
                else:
                    category = InstanceType.GENERAL_PURPOSE

                config = InstanceConfiguration(
                    provider=provider,
                    instance_type=instance_name,
                    instance_category=category,
                    vcpus=instance_data["vcpus"],
                    memory_gb=instance_data["memory_gb"],
                    gpu_count=instance_data.get("gpu_count", 0),
                    gpu_type=instance_data.get("gpu_type"),
                    hourly_cost_usd=instance_data["hourly_usd"],
                    spot_price_usd=instance_data["hourly_usd"]
                    * 0.3,  # Estimate 70% savings
                    metadata=instance_data,
                )

                configs.append(config)

                # Also create spot instance variant if not GPU
                if config.gpu_count == 0:
                    spot_config = InstanceConfiguration(
                        provider=provider,
                        instance_type=f"{instance_name}-spot",
                        instance_category=InstanceType.SPOT_INSTANCE,
                        vcpus=config.vcpus,
                        memory_gb=config.memory_gb,
                        gpu_count=0,
                        hourly_cost_usd=config.spot_price_usd,
                        spot_price_usd=config.spot_price_usd,
                        metadata={**instance_data, "spot_instance": True},
                    )
                    configs.append(spot_config)

        return configs

    def estimate_workload_cost(
        self, workload: WorkloadRequirements, configuration: InstanceConfiguration
    ) -> CostEstimate:
        """Estimate cost for running workload on specific configuration.

        Args:
            workload: Workload requirements
            configuration: Instance configuration

        Returns:
            Cost estimate
        """
        # Get performance benchmarks
        benchmarks = self.performance_benchmarks[configuration.instance_category]

        # Calculate estimated runtime
        base_molecules_per_hour = (
            benchmarks["molecules_per_vcpu_hour"] * configuration.vcpus
        )

        # Apply GPU acceleration if available
        if configuration.gpu_count > 0 and workload.gpu_required:
            gpu_acceleration_factor = 5.0 * configuration.gpu_count  # 5x per GPU
            base_molecules_per_hour *= gpu_acceleration_factor

        # Apply efficiency factors
        efficiency_factor = benchmarks["efficiency_factor"]
        if workload.parallel_workers > 1:
            # Parallel efficiency factor
            parallel_efficiency = min(1.0, 0.9 ** (workload.parallel_workers - 1))
            efficiency_factor *= parallel_efficiency

        effective_molecules_per_hour = base_molecules_per_hour * efficiency_factor
        estimated_runtime_hours = (
            workload.molecules_count / effective_molecules_per_hour
        )

        # Check if deadline can be met
        if (
            workload.deadline_hours
            and estimated_runtime_hours > workload.deadline_hours
        ):
            # Need to scale up resources
            scale_factor = math.ceil(estimated_runtime_hours / workload.deadline_hours)
            estimated_runtime_hours = workload.deadline_hours
            # Note: This would require multiple instances (not handled in this simple model)

        # Calculate costs
        compute_cost = configuration.hourly_cost_usd * estimated_runtime_hours

        # Storage cost (monthly rate prorated)
        provider_data = self.pricing_data[configuration.provider.value]
        storage_cost = (
            workload.storage_gb
            * provider_data["storage_gb_month"]
            * (estimated_runtime_hours / (30 * 24))  # Convert to monthly fraction
        )

        # Network cost (estimated)
        network_gb = workload.molecules_count * 0.001  # 1KB per molecule
        if workload.network_intensive:
            network_gb *= 10  # 10x more network usage
        network_cost = network_gb * provider_data["network_gb"]

        total_cost = compute_cost + storage_cost + network_cost
        cost_per_molecule = total_cost / workload.molecules_count

        # Calculate utilization efficiency
        required_memory_gb = workload.estimated_memory_gb
        memory_utilization = min(1.0, required_memory_gb / configuration.memory_gb)

        required_vcpus = workload.parallel_workers
        cpu_utilization = min(1.0, required_vcpus / configuration.vcpus)

        utilization_efficiency = (memory_utilization + cpu_utilization) / 2

        # Spot instance considerations
        spot_savings_potential = None
        preemption_risk = None

        if configuration.instance_category == InstanceType.SPOT_INSTANCE:
            on_demand_cost = (
                configuration.hourly_cost_usd / 0.3
            )  # Reverse spot discount
            spot_savings_potential = (
                on_demand_cost - configuration.hourly_cost_usd
            ) * estimated_runtime_hours
            preemption_risk = 0.1  # 10% risk estimate

        return CostEstimate(
            configuration=configuration,
            workload=workload,
            estimated_runtime_hours=estimated_runtime_hours,
            compute_cost_usd=compute_cost,
            storage_cost_usd=storage_cost,
            network_cost_usd=network_cost,
            total_cost_usd=total_cost,
            cost_per_molecule=cost_per_molecule,
            utilization_efficiency=utilization_efficiency,
            spot_savings_potential=spot_savings_potential,
            preemption_risk=preemption_risk,
            metadata={
                "effective_molecules_per_hour": effective_molecules_per_hour,
                "efficiency_factor": efficiency_factor,
                "memory_utilization": memory_utilization,
                "cpu_utilization": cpu_utilization,
            },
        )

    def find_optimal_configurations(
        self,
        workload: WorkloadRequirements,
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_COST,
        max_results: int = 5,
    ) -> List[CostEstimate]:
        """Find optimal configurations for a workload.

        Args:
            workload: Workload requirements
            objective: Optimization objective
            max_results: Maximum number of results to return

        Returns:
            List of optimal cost estimates
        """
        # Generate cost estimates for all configurations
        estimates = []

        for config in self.instance_configs:
            # Skip GPU instances if not required
            if config.gpu_count > 0 and not workload.gpu_required:
                continue

            # Skip if insufficient resources
            if (
                config.memory_gb < workload.estimated_memory_gb
                or config.vcpus < workload.parallel_workers
            ):
                continue

            try:
                estimate = self.estimate_workload_cost(workload, config)
                estimates.append(estimate)
            except Exception as e:
                logger.warning(
                    f"Failed to estimate cost for {config.instance_type}: {e}"
                )

        # Sort based on objective
        if objective == OptimizationObjective.MINIMIZE_COST:
            estimates.sort(key=lambda x: x.total_cost_usd)
        elif objective == OptimizationObjective.MINIMIZE_TIME:
            estimates.sort(key=lambda x: x.estimated_runtime_hours)
        elif objective == OptimizationObjective.MAXIMIZE_COST_EFFICIENCY:
            estimates.sort(key=lambda x: x.cost_per_molecule)
        elif objective == OptimizationObjective.BALANCED:
            # Balanced score: cost efficiency + utilization efficiency
            estimates.sort(
                key=lambda x: (x.cost_per_molecule * (2 - x.utilization_efficiency))
            )

        return estimates[:max_results]

    def generate_optimization_recommendations(
        self,
        workload: WorkloadRequirements,
        current_config: Optional[InstanceConfiguration] = None,
    ) -> List[OptimizationRecommendation]:
        """Generate comprehensive optimization recommendations.

        Args:
            workload: Workload requirements
            current_config: Current configuration (if any)

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Get optimal configurations for different objectives
        cost_optimal = self.find_optimal_configurations(
            workload, OptimizationObjective.MINIMIZE_COST, 3
        )
        time_optimal = self.find_optimal_configurations(
            workload, OptimizationObjective.MINIMIZE_TIME, 3
        )
        balanced_optimal = self.find_optimal_configurations(
            workload, OptimizationObjective.BALANCED, 3
        )

        # Current cost (if provided)
        current_cost = None
        if current_config:
            current_cost = self.estimate_workload_cost(workload, current_config)

        # Cost optimization recommendation
        if cost_optimal:
            best_cost = cost_optimal[0]
            cost_savings = None
            cost_savings_percent = None

            if current_cost:
                cost_savings = current_cost.total_cost_usd - best_cost.total_cost_usd
                cost_savings_percent = (
                    cost_savings / current_cost.total_cost_usd
                ) * 100

            reasoning = [
                f"Lowest cost option at ${best_cost.total_cost_usd:.2f}",
                f"Cost per molecule: ${best_cost.cost_per_molecule:.4f}",
                f"Utilization efficiency: {best_cost.utilization_efficiency:.1%}",
            ]

            if best_cost.spot_savings_potential:
                reasoning.append(
                    f"Spot instance savings: ${best_cost.spot_savings_potential:.2f}"
                )

            trade_offs = []
            if best_cost.estimated_runtime_hours > 24:
                trade_offs.append("Long runtime may increase risk of interruptions")
            if best_cost.utilization_efficiency < 0.7:
                trade_offs.append(
                    "Low resource utilization - consider smaller instance"
                )

            implementation = [
                f"Deploy on {best_cost.configuration.provider.value} {best_cost.configuration.instance_type}",
                f"Allocate {best_cost.configuration.memory_gb}GB memory and {best_cost.configuration.vcpus} vCPUs",
            ]

            if best_cost.configuration.instance_category == InstanceType.SPOT_INSTANCE:
                implementation.append(
                    "Configure spot instance with automatic retry on interruption"
                )

            recommendations.append(
                OptimizationRecommendation(
                    recommended_config=best_cost.configuration,
                    estimated_cost=best_cost,
                    cost_savings_usd=cost_savings,
                    cost_savings_percent=cost_savings_percent,
                    performance_impact="neutral",
                    confidence_score=0.9,
                    reasoning=reasoning,
                    trade_offs=trade_offs,
                    implementation_steps=implementation,
                )
            )

        # Time optimization recommendation
        if time_optimal and time_optimal[0] != cost_optimal[0]:
            best_time = time_optimal[0]

            reasoning = [
                f"Fastest execution at {best_time.estimated_runtime_hours:.1f} hours",
                "High-performance configuration",
                f"Total cost: ${best_time.total_cost_usd:.2f}",
            ]

            if best_time.configuration.gpu_count > 0:
                reasoning.append(
                    f"GPU acceleration with {best_time.configuration.gpu_count} {best_time.configuration.gpu_type} GPUs"
                )

            trade_offs = ["Higher cost than minimum cost option"]
            if current_cost:
                cost_difference = (
                    best_time.total_cost_usd - cost_optimal[0].total_cost_usd
                )
                trade_offs.append(
                    f"${cost_difference:.2f} more expensive than cost-optimal"
                )

            performance_impact = "positive"  # Faster execution

            recommendations.append(
                OptimizationRecommendation(
                    recommended_config=best_time.configuration,
                    estimated_cost=best_time,
                    performance_impact=performance_impact,
                    confidence_score=0.85,
                    reasoning=reasoning,
                    trade_offs=trade_offs,
                    implementation_steps=[
                        f"Deploy on {best_time.configuration.provider.value} {best_time.configuration.instance_type}",
                        "Configure for maximum performance",
                        "Monitor resource utilization during execution",
                    ],
                )
            )

        # Balanced recommendation
        if balanced_optimal and balanced_optimal[0] not in [
            cost_optimal[0],
            time_optimal[0] if time_optimal else None,
        ]:
            best_balanced = balanced_optimal[0]

            reasoning = [
                "Optimal balance of cost and performance",
                f"Good utilization efficiency: {best_balanced.utilization_efficiency:.1%}",
                f"Reasonable runtime: {best_balanced.estimated_runtime_hours:.1f} hours",
                f"Cost per molecule: ${best_balanced.cost_per_molecule:.4f}",
            ]

            recommendations.append(
                OptimizationRecommendation(
                    recommended_config=best_balanced.configuration,
                    estimated_cost=best_balanced,
                    performance_impact="neutral",
                    confidence_score=0.8,
                    reasoning=reasoning,
                    trade_offs=["Compromise between cost and speed"],
                    implementation_steps=[
                        f"Deploy on {best_balanced.configuration.provider.value} {best_balanced.configuration.instance_type}",
                        "Monitor both cost and performance metrics",
                        "Scale horizontally if needed for larger workloads",
                    ],
                )
            )

        return recommendations

    def analyze_cost_trends(
        self, workload_sizes: List[int], configuration: InstanceConfiguration
    ) -> Dict[str, Any]:
        """Analyze cost trends across different workload sizes.

        Args:
            workload_sizes: List of molecule counts to analyze
            configuration: Instance configuration to analyze

        Returns:
            Cost trend analysis
        """
        estimates = []

        for size in workload_sizes:
            workload = WorkloadRequirements(
                molecules_count=size,
                descriptors_count=10,  # Standard descriptor set
                estimated_memory_gb=size * 0.002,  # 2MB per molecule
                estimated_compute_hours=size / 10000,  # Rough estimate
                gpu_required=configuration.gpu_count > 0,
            )

            estimate = self.estimate_workload_cost(workload, configuration)
            estimates.append(
                {
                    "molecules": size,
                    "total_cost": estimate.total_cost_usd,
                    "cost_per_molecule": estimate.cost_per_molecule,
                    "runtime_hours": estimate.estimated_runtime_hours,
                    "utilization_efficiency": estimate.utilization_efficiency,
                }
            )

        # Calculate trends
        costs = [e["total_cost"] for e in estimates]
        per_molecule_costs = [e["cost_per_molecule"] for e in estimates]

        # Linear fit for cost trends
        x = np.array(workload_sizes)
        y_total = np.array(costs)
        y_per_molecule = np.array(per_molecule_costs)

        # Cost scaling analysis
        cost_linearity = np.corrcoef(x, y_total)[0, 1] ** 2  # R-squared

        # Economies of scale
        economies_of_scale = (
            per_molecule_costs[0] - per_molecule_costs[-1]
        ) / per_molecule_costs[0]

        return {
            "estimates": estimates,
            "cost_linearity": cost_linearity,
            "economies_of_scale": economies_of_scale,
            "optimal_batch_size": workload_sizes[np.argmin(per_molecule_costs)],
            "scaling_efficiency": {
                "linear_scaling": cost_linearity > 0.95,
                "economies_present": economies_of_scale > 0.1,
                "recommended_min_batch": max(1000, workload_sizes[0]),
            },
            "configuration": {
                "provider": configuration.provider.value,
                "instance_type": configuration.instance_type,
                "hourly_cost": configuration.hourly_cost_usd,
            },
        }

    def generate_cost_report(
        self,
        workload: WorkloadRequirements,
        recommendations: List[OptimizationRecommendation],
    ) -> str:
        """Generate comprehensive cost optimization report.

        Args:
            workload: Workload requirements
            recommendations: Optimization recommendations

        Returns:
            Formatted cost report
        """
        report = ["Cloud Cost Optimization Report", "=" * 40, ""]

        # Workload Summary
        report.extend(
            [
                "WORKLOAD REQUIREMENTS",
                "-" * 25,
                f"Molecules to Process: {workload.molecules_count:,}",
                f"Descriptors: {workload.descriptors_count}",
                f"Estimated Memory: {workload.estimated_memory_gb:.1f} GB",
                f"Parallel Workers: {workload.parallel_workers}",
                f"GPU Required: {'Yes' if workload.gpu_required else 'No'}",
                f"Deadline: {workload.deadline_hours:.1f} hours"
                if workload.deadline_hours
                else "Deadline: None",
                "",
            ]
        )

        # Recommendations
        report.extend(["OPTIMIZATION RECOMMENDATIONS", "-" * 30])

        for i, rec in enumerate(recommendations, 1):
            config = rec.recommended_config
            cost = rec.estimated_cost

            report.extend(
                [
                    f"{i}. {config.provider.value.upper()} {config.instance_type}",
                    f"   Category: {config.instance_category.value.replace('_', ' ').title()}",
                    f"   Resources: {config.vcpus} vCPUs, {config.memory_gb:.1f} GB RAM",
                    "",
                ]
            )

            if config.gpu_count > 0:
                report.append(f"   GPUs: {config.gpu_count}x {config.gpu_type}")

            report.extend(
                [
                    f"   Estimated Cost: ${cost.total_cost_usd:.2f}",
                    f"   Cost per Molecule: ${cost.cost_per_molecule:.4f}",
                    f"   Runtime: {cost.estimated_runtime_hours:.1f} hours",
                    f"   Utilization Efficiency: {cost.utilization_efficiency:.1%}",
                    "",
                ]
            )

            if rec.cost_savings_usd:
                report.extend(
                    [
                        f"   Cost Savings: ${rec.cost_savings_usd:.2f} ({rec.cost_savings_percent:.1f}%)",
                        "",
                    ]
                )

            if rec.reasoning:
                report.extend(["   Reasoning:"])
                for reason in rec.reasoning:
                    report.append(f"   • {reason}")
                report.append("")

            if rec.trade_offs:
                report.extend(["   Trade-offs:"])
                for trade_off in rec.trade_offs:
                    report.append(f"   • {trade_off}")
                report.append("")

            report.append("   " + "=" * 50)
            report.append("")

        # Cost Comparison
        if len(recommendations) > 1:
            report.extend(["COST COMPARISON", "-" * 20])

            min_cost = min(rec.estimated_cost.total_cost_usd for rec in recommendations)
            max_cost = max(rec.estimated_cost.total_cost_usd for rec in recommendations)

            report.extend(
                [
                    f"Cost Range: ${min_cost:.2f} - ${max_cost:.2f}",
                    f"Maximum Potential Savings: ${max_cost - min_cost:.2f}",
                    "",
                ]
            )

            # Create simple cost table
            report.extend(
                ["Provider    Instance Type         Cost      Runtime    $/Molecule"]
            )
            report.append("-" * 65)

            for rec in recommendations:
                config = rec.recommended_config
                cost = rec.estimated_cost

                provider = config.provider.value.upper()[:8]
                instance = config.instance_type[:20]
                cost_str = f"${cost.total_cost_usd:.2f}"
                runtime_str = f"{cost.estimated_runtime_hours:.1f}h"
                per_mol_str = f"${cost.cost_per_molecule:.4f}"

                report.append(
                    f"{provider:<8} {instance:<20} {cost_str:<8} {runtime_str:<10} {per_mol_str}"
                )

            report.append("")

        return "\n".join(report)

    def save_analysis_results(
        self,
        workload: WorkloadRequirements,
        recommendations: List[OptimizationRecommendation],
        output_dir: str = "cost_analysis_results",
    ):
        """Save cost analysis results to files.

        Args:
            workload: Workload requirements
            recommendations: Optimization recommendations
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results as JSON
        results_data = {
            "timestamp": timestamp,
            "workload": {
                "molecules_count": workload.molecules_count,
                "descriptors_count": workload.descriptors_count,
                "estimated_memory_gb": workload.estimated_memory_gb,
                "estimated_compute_hours": workload.estimated_compute_hours,
                "gpu_required": workload.gpu_required,
                "parallel_workers": workload.parallel_workers,
                "deadline_hours": workload.deadline_hours,
            },
            "recommendations": [],
        }

        for rec in recommendations:
            config = rec.recommended_config
            cost = rec.estimated_cost

            rec_data = {
                "configuration": {
                    "provider": config.provider.value,
                    "instance_type": config.instance_type,
                    "instance_category": config.instance_category.value,
                    "vcpus": config.vcpus,
                    "memory_gb": config.memory_gb,
                    "gpu_count": config.gpu_count,
                    "gpu_type": config.gpu_type,
                    "hourly_cost_usd": config.hourly_cost_usd,
                },
                "cost_estimate": {
                    "total_cost_usd": cost.total_cost_usd,
                    "cost_per_molecule": cost.cost_per_molecule,
                    "estimated_runtime_hours": cost.estimated_runtime_hours,
                    "compute_cost_usd": cost.compute_cost_usd,
                    "storage_cost_usd": cost.storage_cost_usd,
                    "network_cost_usd": cost.network_cost_usd,
                    "utilization_efficiency": cost.utilization_efficiency,
                    "spot_savings_potential": cost.spot_savings_potential,
                    "preemption_risk": cost.preemption_risk,
                },
                "optimization": {
                    "cost_savings_usd": rec.cost_savings_usd,
                    "cost_savings_percent": rec.cost_savings_percent,
                    "performance_impact": rec.performance_impact,
                    "confidence_score": rec.confidence_score,
                    "reasoning": rec.reasoning,
                    "trade_offs": rec.trade_offs,
                    "implementation_steps": rec.implementation_steps,
                },
            }

            results_data["recommendations"].append(rec_data)

        # Save JSON results
        json_file = output_path / f"cost_analysis_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(results_data, f, indent=2)

        # Save human-readable report
        report = self.generate_cost_report(workload, recommendations)
        report_file = output_path / f"cost_report_{timestamp}.txt"

        with open(report_file, "w") as f:
            f.write(report)

        logger.info(f"Cost analysis results saved to {json_file}")
        logger.info(f"Cost report saved to {report_file}")


if __name__ == "__main__":
    # Demonstrate cloud cost optimization
    print("ChemiAccel Cloud Cost Optimization Analysis")
    print("=" * 50)

    # Initialize optimizer
    optimizer = CloudCostOptimizer()

    # Example workload
    workload = WorkloadRequirements(
        molecules_count=100000,
        descriptors_count=15,
        estimated_memory_gb=200,  # 200GB memory requirement
        estimated_compute_hours=10,
        gpu_required=False,
        parallel_workers=8,
        storage_gb=50,
        deadline_hours=24,  # 24 hour deadline
    )

    print("\nAnalyzing workload:")
    print(f"- {workload.molecules_count:,} molecules")
    print(f"- {workload.descriptors_count} descriptors")
    print(f"- {workload.estimated_memory_gb} GB memory required")
    print(f"- {workload.parallel_workers} parallel workers")
    print(f"- {workload.deadline_hours} hour deadline")

    # Generate recommendations
    print("\nGenerating optimization recommendations...")
    recommendations = optimizer.generate_optimization_recommendations(workload)

    # Generate and display report
    report = optimizer.generate_cost_report(workload, recommendations)

    print("\n" + "=" * 60)
    print("COST OPTIMIZATION REPORT")
    print("=" * 60)
    print(report)

    # Save detailed analysis
    optimizer.save_analysis_results(workload, recommendations)

    print("\nDetailed analysis saved to cost_analysis_results/")

    # Example: Analyze cost trends for different workload sizes
    print("\nAnalyzing cost trends across workload sizes...")

    sample_config = recommendations[0].recommended_config
    workload_sizes = [10000, 50000, 100000, 500000, 1000000]

    trend_analysis = optimizer.analyze_cost_trends(workload_sizes, sample_config)

    print(
        f"\nCost Trend Analysis for {sample_config.provider.value} {sample_config.instance_type}:"
    )
    print(f"- Cost Linearity: {trend_analysis['cost_linearity']:.3f}")
    print(f"- Economies of Scale: {trend_analysis['economies_of_scale']:.1%}")
    print(f"- Optimal Batch Size: {trend_analysis['optimal_batch_size']:,} molecules")
    print(
        f"- Linear Scaling: {'Yes' if trend_analysis['scaling_efficiency']['linear_scaling'] else 'No'}"
    )

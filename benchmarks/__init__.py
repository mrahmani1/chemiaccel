"""
ChemiAccel Benchmarking Suite.

This module provides comprehensive benchmarking capabilities for comparing
ChemiAccel performance against RDKit and Mordred implementations.
"""

from .benchmark_framework import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
    PerformanceComparison,
    run_benchmark_suite,
)
from .dataset_generator import (
    BenchmarkDatasetGenerator,
    DatasetConfig,
    MolecularDataset,
    generate_benchmark_datasets,
)
from .reference_implementations import (
    MordredReference,
    RDKitReference,
    ReferenceImplementation,
    get_reference_implementations,
)
from .regression_testing import (
    PerformanceBaseline,
    RegressionResult,
    RegressionTestSuite,
    run_regression_tests,
)
from .report_generator import (
    BenchmarkReportGenerator,
    ReportConfig,
    generate_benchmark_report,
)
from .scaling_tests import (
    HardwareConfig,
    ScalingResult,
    ScalingTestRunner,
    run_scaling_tests,
)

__all__ = [
    # Framework
    "BenchmarkSuite",
    "BenchmarkResult",
    "BenchmarkConfig",
    "PerformanceComparison",
    "run_benchmark_suite",
    # Dataset Generation
    "BenchmarkDatasetGenerator",
    "MolecularDataset",
    "DatasetConfig",
    "generate_benchmark_datasets",
    # Reference Implementations
    "RDKitReference",
    "MordredReference",
    "ReferenceImplementation",
    "get_reference_implementations",
    # Scaling Tests
    "ScalingTestRunner",
    "ScalingResult",
    "HardwareConfig",
    "run_scaling_tests",
    # Regression Testing
    "RegressionTestSuite",
    "RegressionResult",
    "PerformanceBaseline",
    "run_regression_tests",
    # Report Generation
    "BenchmarkReportGenerator",
    "ReportConfig",
    "generate_benchmark_report",
]

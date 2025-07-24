"""
ChemiAccel Benchmarking Suite.

This module provides comprehensive benchmarking capabilities for comparing
ChemiAccel performance against RDKit and Mordred implementations.
"""

from .benchmark_framework import (
    BenchmarkSuite,
    BenchmarkResult,
    BenchmarkConfig,
    PerformanceComparison,
    run_benchmark_suite
)

from .dataset_generator import (
    BenchmarkDatasetGenerator,
    MolecularDataset,
    DatasetConfig,
    generate_benchmark_datasets
)

from .reference_implementations import (
    RDKitReference,
    MordredReference,
    ReferenceImplementation,
    get_reference_implementations
)

from .scaling_tests import (
    ScalingTestRunner,
    ScalingResult,
    HardwareConfig,
    run_scaling_tests
)

from .regression_testing import (
    RegressionTestSuite,
    RegressionResult,
    PerformanceBaseline,
    run_regression_tests
)

from .report_generator import (
    BenchmarkReportGenerator,
    ReportConfig,
    generate_benchmark_report
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
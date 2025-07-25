#!/usr/bin/env python3
"""
Main Benchmark Runner for ChemiAccel.

This script runs comprehensive benchmarks comparing ChemiAccel performance
against RDKit and Mordred implementations across different configurations.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark_framework import BenchmarkConfig, run_benchmark_suite
from dataset_generator import DatasetConfig, generate_benchmark_datasets
from reference_implementations import get_reference_implementations
from regression_testing import run_regression_tests
from report_generator import ReportConfig, generate_benchmark_report
from scaling_tests import run_scaling_tests

# Import ChemiAccel components
try:
    from chemi_accel.backends.cuda import get_optimized_kernel_manager
    from chemi_accel.backends.jax import get_jax_backend
    from chemi_accel.core.calculator import Calculator

    CHEMI_ACCEL_AVAILABLE = True
except ImportError as e:
    print(f"ChemiAccel not available: {e}")
    CHEMI_ACCEL_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_chemi_accel_implementations() -> Dict[str, Any]:
    """Setup ChemiAccel implementations for benchmarking."""
    implementations = {}

    if not CHEMI_ACCEL_AVAILABLE:
        logger.warning("ChemiAccel not available, skipping ChemiAccel benchmarks")
        return implementations

    try:
        # CUDA backend
        cuda_calc = Calculator(backend="cuda")
        implementations["cuda"] = lambda molecules: cuda_calc.pandas(molecules)
        logger.info("Setup CUDA backend for benchmarking")
    except Exception as e:
        logger.warning(f"Failed to setup CUDA backend: {e}")

    try:
        # JAX backend
        jax_calc = Calculator(backend="jax")
        implementations["jax"] = lambda molecules: jax_calc.pandas(molecules)
        logger.info("Setup JAX backend for benchmarking")
    except Exception as e:
        logger.warning(f"Failed to setup JAX backend: {e}")

    try:
        # CPU backend (fallback)
        cpu_calc = Calculator(backend="cpu")
        implementations["cpu"] = lambda molecules: cpu_calc.pandas(molecules)
        logger.info("Setup CPU backend for benchmarking")
    except Exception as e:
        logger.warning(f"Failed to setup CPU backend: {e}")

    return implementations


def create_reference_implementations() -> Dict[str, Any]:
    """Create reference implementation wrappers."""
    ref_implementations = get_reference_implementations()

    # Wrap reference implementations for consistent interface
    wrapped_implementations = {}

    for name, ref_impl in ref_implementations.items():

        def create_wrapper(impl, descriptor_name):
            def wrapper(molecules):
                if hasattr(molecules, "tolist"):
                    mol_list = molecules.tolist()
                elif hasattr(molecules, "__iter__"):
                    mol_list = list(molecules)
                else:
                    mol_list = [molecules]

                result = impl.calculate_descriptor(mol_list, descriptor_name)
                return result.values

            return wrapper

        # Create wrappers for common descriptors
        for descriptor in ["molecular_weight", "logp", "tpsa"]:
            if ref_impl.is_descriptor_supported(descriptor):
                key = f"{name}_{descriptor}"
                wrapped_implementations[key] = create_wrapper(ref_impl, descriptor)

    return wrapped_implementations


def run_performance_benchmarks(args) -> None:
    """Run performance benchmarks."""
    logger.info("Starting performance benchmarks")

    # Generate datasets
    dataset_config = DatasetConfig(
        name="benchmark_dataset", sizes=args.dataset_sizes, random_seed=42
    )

    datasets = generate_benchmark_datasets(dataset_config)
    logger.info(f"Generated {len(datasets)} benchmark datasets")

    # Setup implementations
    chemi_accel_impls = setup_chemi_accel_implementations()
    reference_impls = create_reference_implementations()

    if not chemi_accel_impls:
        logger.error("No ChemiAccel implementations available")
        return

    if not reference_impls:
        logger.error("No reference implementations available")
        return

    # Configure benchmark
    benchmark_config = BenchmarkConfig(
        name="chemi_accel_performance_benchmark",
        description="Comprehensive performance comparison of ChemiAccel vs reference implementations",
        dataset_sizes=args.dataset_sizes,
        iterations=args.iterations,
        descriptors_to_test=args.descriptors,
        backends_to_test=list(chemi_accel_impls.keys()),
        reference_implementations=list(
            set(impl.split("_")[0] for impl in reference_impls.keys())
        ),
        output_directory=args.output_dir,
    )

    # Run benchmark suite
    suite = run_benchmark_suite(
        benchmark_config, chemi_accel_impls, reference_impls, datasets
    )

    logger.info(f"Completed performance benchmarks: {len(suite.results)} results")

    # Generate report
    if args.generate_report:
        report_config = ReportConfig(
            title="ChemiAccel Performance Benchmark Report",
            include_plots=True,
            output_format="html",
        )

        report_path = generate_benchmark_report(
            suite.results,
            suite.comparisons,
            config=report_config,
            output_directory=args.output_dir,
        )

        logger.info(f"Generated benchmark report: {report_path}")


def run_scaling_benchmarks(args) -> None:
    """Run scaling benchmarks."""
    logger.info("Starting scaling benchmarks")

    # Generate datasets for scaling tests
    dataset_config = DatasetConfig(
        name="scaling_dataset", sizes=args.scaling_sizes, random_seed=42
    )

    datasets = generate_benchmark_datasets(dataset_config)

    # Setup implementations
    chemi_accel_impls = setup_chemi_accel_implementations()

    if not chemi_accel_impls:
        logger.error("No ChemiAccel implementations available for scaling tests")
        return

    # Run scaling tests
    scaling_results = run_scaling_tests(
        chemi_accel_impls, datasets, args.descriptors, args.output_dir
    )

    logger.info(f"Completed scaling benchmarks: {len(scaling_results)} results")


def run_regression_benchmarks(args) -> None:
    """Run regression benchmarks."""
    logger.info("Starting regression benchmarks")

    # Generate test dataset
    dataset_config = DatasetConfig(
        name="regression_dataset",
        sizes=[1000],  # Use consistent size for regression testing
        random_seed=42,
    )

    datasets = generate_benchmark_datasets(dataset_config)

    # Setup implementations
    chemi_accel_impls = setup_chemi_accel_implementations()

    if not chemi_accel_impls:
        logger.error("No ChemiAccel implementations available for regression tests")
        return

    # Run regression tests
    regression_results = run_regression_tests(
        chemi_accel_impls,
        datasets,
        args.descriptors,
        baseline_directory=args.baseline_dir,
        results_directory=args.output_dir,
    )

    logger.info(f"Completed regression benchmarks: {len(regression_results)} results")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="ChemiAccel Benchmark Suite")

    # General options
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument(
        "--descriptors",
        nargs="+",
        default=["molecular_weight", "tpsa", "logp"],
        help="Descriptors to benchmark",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--generate-report", action="store_true", help="Generate HTML report"
    )

    # Benchmark type selection
    parser.add_argument(
        "--performance", action="store_true", help="Run performance benchmarks"
    )
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmarks")
    parser.add_argument(
        "--regression", action="store_true", help="Run regression benchmarks"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmark types")

    # Dataset configuration
    parser.add_argument(
        "--dataset-sizes",
        nargs="+",
        type=int,
        default=[100, 1000, 10000],
        help="Dataset sizes for performance benchmarks",
    )
    parser.add_argument(
        "--scaling-sizes",
        nargs="+",
        type=int,
        default=[100, 500, 1000, 5000, 10000, 50000],
        help="Dataset sizes for scaling benchmarks",
    )

    # Regression testing options
    parser.add_argument(
        "--baseline-dir",
        default="performance_baselines",
        help="Directory containing performance baselines",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Determine which benchmarks to run
    run_performance = args.performance or args.all
    run_scaling = args.scaling or args.all
    run_regression = args.regression or args.all

    if not (run_performance or run_scaling or run_regression):
        logger.error(
            "No benchmark type specified. Use --performance, --scaling, --regression, or --all"
        )
        return 1

    try:
        # Run selected benchmarks
        if run_performance:
            run_performance_benchmarks(args)

        if run_scaling:
            run_scaling_benchmarks(args)

        if run_regression:
            run_regression_benchmarks(args)

        logger.info("All benchmarks completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

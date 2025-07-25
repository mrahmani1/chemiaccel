#!/usr/bin/env python3
"""
Test script for ChemiAccel benchmarking framework.

This script tests the benchmarking components to ensure they work correctly.
"""

import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_framework import BenchmarkConfig, BenchmarkSuite
from dataset_generator import BenchmarkDatasetGenerator, DatasetConfig
from reference_implementations import get_reference_implementations

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset_generation():
    """Test dataset generation."""
    logger.info("Testing dataset generation...")

    config = DatasetConfig(
        name="test_dataset",
        description="Test dataset for benchmarking framework",
        sizes=[100, 500],
        random_seed=42,
    )

    generator = BenchmarkDatasetGenerator(config)
    datasets = generator.generate_all_datasets()

    assert len(datasets) == 2
    assert 100 in datasets
    assert 500 in datasets

    # Check dataset properties
    dataset_100 = datasets[100]
    assert len(dataset_100.smiles) == 100
    assert len(dataset_100.properties["complexity"]) == 100

    logger.info("✓ Dataset generation test passed")


def test_reference_implementations():
    """Test reference implementations."""
    logger.info("Testing reference implementations...")

    ref_impls = get_reference_implementations()

    # Should have at least CPU reference
    assert "cpu_reference" in ref_impls

    cpu_ref = ref_impls["cpu_reference"]

    # Test basic descriptor calculation
    test_molecules = ["CCO", "CC(C)O", "c1ccccc1"]

    result = cpu_ref.calculate_descriptor(test_molecules, "molecular_weight")

    assert len(result.values) == 3
    assert result.success_count > 0
    assert result.implementation == "cpu_reference"

    logger.info("✓ Reference implementations test passed")


def test_benchmark_framework():
    """Test benchmark framework."""
    logger.info("Testing benchmark framework...")

    # Create simple test implementations
    def fast_impl(molecules):
        import time

        time.sleep(0.001)  # Simulate fast computation
        return [1.0] * len(molecules)

    def slow_impl(molecules):
        import time

        time.sleep(0.01)  # Simulate slow computation
        return [1.0] * len(molecules)

    # Create test dataset
    test_molecules = ["CCO", "CC(C)O", "c1ccccc1", "CC(=O)O", "c1ccc(O)cc1"]

    # Configure benchmark
    config = BenchmarkConfig(
        name="test_benchmark",
        description="Test benchmark for framework validation",
        dataset_sizes=[5],
        iterations=3,
        warmup_iterations=1,
        descriptors_to_test=["test_descriptor"],
        backends_to_test=["fast", "slow"],
        reference_implementations=["slow"],
        output_directory="test_results",
    )

    # Create benchmark suite
    suite = BenchmarkSuite(config)

    # Run benchmarks
    chemi_accel_impls = {"fast": fast_impl, "slow": slow_impl}

    reference_impls = {"slow": slow_impl}

    datasets = {5: test_molecules}

    suite.run_full_suite(chemi_accel_impls, reference_impls, datasets)

    # Check results
    assert len(suite.results) > 0
    assert len(suite.comparisons) > 0

    # Fast implementation should be faster than slow
    fast_results = [r for r in suite.results if r.backend == "fast"]
    slow_results = [r for r in suite.results if r.backend == "slow"]

    if fast_results and slow_results:
        fast_time = fast_results[0].mean_time
        slow_time = slow_results[0].mean_time
        assert (
            fast_time < slow_time
        ), f"Fast impl ({fast_time}s) should be faster than slow impl ({slow_time}s)"

    logger.info("✓ Benchmark framework test passed")


def main():
    """Run all tests."""
    logger.info("Starting benchmarking framework tests...")

    try:
        test_dataset_generation()
        test_reference_implementations()
        test_benchmark_framework()

        logger.info("✅ All benchmarking tests passed!")
        return 0

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

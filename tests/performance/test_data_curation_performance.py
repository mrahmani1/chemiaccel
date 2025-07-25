"""
Performance Tests for GPU-Accelerated Data Curation Pipeline.

This module provides comprehensive performance tests for the molecular data curation
pipeline, designed to validate performance at scale including 100-million compound
datasets. Tests GPU acceleration, parallel processing, and memory efficiency.
"""

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil
import pytest

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import the data curation modules
from src.chemi_accel.data_curation.gpu_molecular_cleaning import (
    CleaningConfig,
    GPUMolecularCleaningPipeline,
)
from src.chemi_accel.data_curation.inchikey_deduplication import (
    InChIKeyConfig,
    InChIKeyGenerator,
)
from src.chemi_accel.data_curation.pains_substructure_filter import (
    FilterConfig,
    FilterType,
    MolecularFilterEngine,
)
from src.chemi_accel.data_curation.parallel_validation import (
    SMILESValidator,
    ValidationConfig,
)
from src.chemi_accel.data_curation.tautomer_standardization import (
    TautomerStandardizationConfig,
    TautomerStandardizer,
)
from src.chemi_accel.utils.logging import get_logger

logger = get_logger()


@dataclass
class PerformanceMetrics:
    """Performance metrics for data curation operations."""

    molecules_per_second: float
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    processing_time: float
    success_rate: float
    cpu_utilization: float
    gpu_utilization: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class ScalingTest:
    """Configuration for scaling tests."""

    dataset_sizes: List[int]
    expected_speedup: float
    max_memory_gb: float
    timeout_seconds: int
    backends_to_test: List[str]


class DatasetGenerator:
    """Generate synthetic molecular datasets for performance testing."""

    def __init__(self, seed: int = 42):
        """Initialize dataset generator.

        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed

        # Common SMILES patterns for realistic datasets
        self.base_smiles = [
            "CCO",  # Ethanol
            "CC(C)O",  # Isopropanol
            "c1ccccc1",  # Benzene
            "CC(=O)O",  # Acetic acid
            "CCN",  # Ethylamine
            "c1ccc(cc1)O",  # Phenol
            "CC(C)(C)O",  # tert-Butanol
            "CCCCO",  # Butanol
            "c1ccc2c(c1)cccn2",  # Quinoline
            "CC1=CC=CC=C1",  # Toluene
            "CCc1ccccc1",  # Ethylbenzene
            "c1ccc(cc1)N",  # Aniline
            "CC(=O)C",  # Acetone
            "CCCC",  # Butane
            "c1ccc(cc1)C(=O)O",  # Benzoic acid
            "CCc1ccc(cc1)O",  # 4-Ethylphenol
            "CC(C)C",  # Isobutane
            "c1ccc2c(c1)oc3ccccc32",  # Dibenzofuran
            "CCN(CC)CC",  # Triethylamine
            "c1ccc(cc1)C=O",  # Benzaldehyde
        ]

        # Add some problematic SMILES for error handling tests
        self.problematic_smiles = [
            "invalid_smiles",
            "C[C@H]1[C@@H]([C@H]([C@@H]([C@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O[C@@H]3[C@H](OC([C@@H]([C@H]3O)O)O)CO)CO)O)O)O)O",  # Very complex sugar
            "",  # Empty string
            "C" * 1000,  # Very long alkane chain
            "[Na+].[Cl-]",  # Salt
            "C1CCC1",  # Cyclobutane
        ]

        logger.info(f"Initialized dataset generator with seed {seed}")

    def generate_smiles_dataset(
        self,
        size: int,
        error_rate: float = 0.01,
        include_salts: bool = True,
        include_duplicates: bool = True,
        duplicate_rate: float = 0.1,
    ) -> List[str]:
        """Generate a synthetic SMILES dataset.

        Args:
            size: Number of SMILES to generate
            error_rate: Fraction of invalid SMILES
            include_salts: Include salt forms
            include_duplicates: Include duplicate molecules
            duplicate_rate: Fraction of duplicates

        Returns:
            List of SMILES strings
        """
        logger.info(
            f"Generating dataset of {size} SMILES with {error_rate*100:.1f}% error rate"
        )

        smiles_list = []

        # Calculate counts
        num_errors = int(size * error_rate)
        num_duplicates = int(size * duplicate_rate) if include_duplicates else 0
        num_unique = size - num_errors - num_duplicates

        # Generate unique SMILES
        for i in range(num_unique):
            base_smiles = np.random.choice(self.base_smiles)

            # Add variations
            if np.random.random() < 0.3:  # 30% chance of modification
                # Add substituents or functional groups
                modifications = ["C", "O", "N", "Cl", "F", "Br"]
                mod = np.random.choice(modifications)
                if "c1ccccc1" in base_smiles:  # Benzene ring
                    base_smiles = base_smiles.replace("c1ccccc1", f"c1ccc(cc1){mod}")
                else:
                    base_smiles = base_smiles + mod

            # Add salt forms
            if include_salts and np.random.random() < 0.2:  # 20% chance of salt
                salts = ["[Na+].[Cl-]", "[K+].[Br-]", "[NH4+].[Cl-]"]
                salt = np.random.choice(salts)
                base_smiles = f"{base_smiles}.{salt}"

            smiles_list.append(base_smiles)

        # Add duplicates
        if include_duplicates and num_duplicates > 0:
            duplicate_indices = np.random.choice(
                len(smiles_list), num_duplicates, replace=True
            )
            for idx in duplicate_indices:
                smiles_list.append(smiles_list[idx])

        # Add problematic SMILES
        for i in range(num_errors):
            error_smiles = np.random.choice(self.problematic_smiles)
            smiles_list.append(error_smiles)

        # Shuffle the list
        np.random.shuffle(smiles_list)

        logger.info(
            f"Generated dataset: {num_unique} unique, {num_duplicates} duplicates, {num_errors} errors"
        )
        return smiles_list

    def generate_test_datasets(self, sizes: List[int]) -> Dict[int, List[str]]:
        """Generate multiple test datasets of different sizes.

        Args:
            sizes: List of dataset sizes to generate

        Returns:
            Dictionary mapping size to SMILES list
        """
        datasets = {}
        for size in sizes:
            datasets[size] = self.generate_smiles_dataset(size)
        return datasets


class PerformanceTestSuite:
    """Main performance testing suite for data curation pipeline."""

    def __init__(self, output_dir: str = "performance_results"):
        """Initialize performance test suite.

        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_generator = DatasetGenerator()
        self.results = {}

        logger.info(f"Initialized performance test suite, output dir: {output_dir}")

    def measure_performance(
        self, operation_func, *args, **kwargs
    ) -> PerformanceMetrics:
        """Measure performance of an operation.

        Args:
            operation_func: Function to measure
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Performance metrics
        """
        # Force garbage collection before test
        gc.collect()
        if CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        initial_gpu_memory = 0.0
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                initial_gpu_memory = mempool.used_bytes() / (1024 * 1024)  # MB
            except:
                pass

        # Measure CPU usage
        cpu_percent_start = process.cpu_percent()

        # Execute operation
        start_time = time.perf_counter()
        try:
            result = operation_func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            result = None
            success = False
        end_time = time.perf_counter()

        processing_time = end_time - start_time

        # Get final memory usage
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_usage = final_memory - initial_memory

        final_gpu_memory = 0.0
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                final_gpu_memory = mempool.used_bytes() / (1024 * 1024)  # MB
            except:
                pass

        gpu_memory_usage = final_gpu_memory - initial_gpu_memory

        # Get CPU usage
        cpu_percent_end = process.cpu_percent()
        cpu_utilization = (cpu_percent_start + cpu_percent_end) / 2

        # Calculate throughput
        dataset_size = kwargs.get(
            "dataset_size", len(args[0]) if args and hasattr(args[0], "__len__") else 1
        )
        molecules_per_second = (
            dataset_size / processing_time if processing_time > 0 else 0.0
        )

        metrics = PerformanceMetrics(
            molecules_per_second=molecules_per_second,
            memory_usage_mb=memory_usage,
            gpu_memory_usage_mb=gpu_memory_usage,
            processing_time=processing_time,
            success_rate=1.0 if success else 0.0,
            cpu_utilization=cpu_utilization,
            metadata={"result_available": result is not None},
        )

        return metrics

    def test_smiles_validation_performance(
        self, dataset_sizes: List[int]
    ) -> Dict[int, PerformanceMetrics]:
        """Test SMILES validation performance across different dataset sizes.

        Args:
            dataset_sizes: List of dataset sizes to test

        Returns:
            Dictionary mapping dataset size to performance metrics
        """
        logger.info("Testing SMILES validation performance")

        results = {}
        validator = SMILESValidator(ValidationConfig(parallel_backend="threading"))

        for size in dataset_sizes:
            logger.info(f"Testing SMILES validation with {size} molecules")

            dataset = self.dataset_generator.generate_smiles_dataset(size)
            metrics = self.measure_performance(
                validator.validate_batch, dataset, dataset_size=size
            )

            results[size] = metrics

            logger.info(
                f"SMILES validation ({size}): {metrics.molecules_per_second:.1f} mol/s, "
                f"{metrics.memory_usage_mb:.1f} MB, {metrics.processing_time:.2f}s"
            )

        return results

    def test_tautomer_standardization_performance(
        self, dataset_sizes: List[int]
    ) -> Dict[int, PerformanceMetrics]:
        """Test tautomer standardization performance.

        Args:
            dataset_sizes: List of dataset sizes to test

        Returns:
            Dictionary mapping dataset size to performance metrics
        """
        logger.info("Testing tautomer standardization performance")

        results = {}
        config = TautomerStandardizationConfig(
            parallel_backend="threading", max_tautomers=100, cache_enabled=True
        )
        standardizer = TautomerStandardizer(config)

        for size in dataset_sizes:
            logger.info(f"Testing tautomer standardization with {size} molecules")

            dataset = self.dataset_generator.generate_smiles_dataset(
                size, error_rate=0.0
            )
            metrics = self.measure_performance(
                standardizer.standardize_batch, dataset, dataset_size=size
            )

            results[size] = metrics

            logger.info(
                f"Tautomer standardization ({size}): {metrics.molecules_per_second:.1f} mol/s, "
                f"{metrics.memory_usage_mb:.1f} MB, {metrics.processing_time:.2f}s"
            )

        return results

    def test_inchikey_generation_performance(
        self, dataset_sizes: List[int]
    ) -> Dict[int, PerformanceMetrics]:
        """Test InChIKey generation performance.

        Args:
            dataset_sizes: List of dataset sizes to test

        Returns:
            Dictionary mapping dataset size to performance metrics
        """
        logger.info("Testing InChIKey generation performance")

        results = {}
        config = InChIKeyConfig(
            parallel_backend="threading",
            cache_enabled=True,
            use_gpu_acceleration=CUPY_AVAILABLE,
        )
        generator = InChIKeyGenerator(config)

        for size in dataset_sizes:
            logger.info(f"Testing InChIKey generation with {size} molecules")

            dataset = self.dataset_generator.generate_smiles_dataset(
                size, error_rate=0.0
            )
            metrics = self.measure_performance(
                generator.generate_batch, dataset, dataset_size=size
            )

            results[size] = metrics

            logger.info(
                f"InChIKey generation ({size}): {metrics.molecules_per_second:.1f} mol/s, "
                f"{metrics.memory_usage_mb:.1f} MB, {metrics.processing_time:.2f}s"
            )

        return results

    def test_molecular_filtering_performance(
        self, dataset_sizes: List[int]
    ) -> Dict[int, PerformanceMetrics]:
        """Test molecular filtering performance.

        Args:
            dataset_sizes: List of dataset sizes to test

        Returns:
            Dictionary mapping dataset size to performance metrics
        """
        logger.info("Testing molecular filtering performance")

        results = {}
        config = FilterConfig(
            enabled_filters=[FilterType.PAINS_A, FilterType.LIPINSKI_RULE_OF_5],
            parallel_backend="threading",
            cache_enabled=True,
            use_gpu_acceleration=CUPY_AVAILABLE,
        )
        filter_engine = MolecularFilterEngine(config)

        for size in dataset_sizes:
            logger.info(f"Testing molecular filtering with {size} molecules")

            dataset = self.dataset_generator.generate_smiles_dataset(
                size, error_rate=0.0
            )
            metrics = self.measure_performance(
                filter_engine.filter_batch, dataset, dataset_size=size
            )

            results[size] = metrics

            logger.info(
                f"Molecular filtering ({size}): {metrics.molecules_per_second:.1f} mol/s, "
                f"{metrics.memory_usage_mb:.1f} MB, {metrics.processing_time:.2f}s"
            )

        return results

    def test_full_pipeline_performance(
        self, dataset_sizes: List[int]
    ) -> Dict[int, PerformanceMetrics]:
        """Test full data curation pipeline performance.

        Args:
            dataset_sizes: List of dataset sizes to test

        Returns:
            Dictionary mapping dataset size to performance metrics
        """
        logger.info("Testing full pipeline performance")

        results = {}
        config = CleaningConfig(
            parallel_backend="threading",
            use_gpu_acceleration=CUPY_AVAILABLE,
            enable_tautomer_standardization=True,
            enable_inchikey_deduplication=True,
            enable_pains_filtering=True,
        )
        pipeline = GPUMolecularCleaningPipeline(config)

        for size in dataset_sizes:
            logger.info(f"Testing full pipeline with {size} molecules")

            dataset = self.dataset_generator.generate_smiles_dataset(
                size, error_rate=0.05, include_duplicates=True, duplicate_rate=0.1
            )

            metrics = self.measure_performance(
                pipeline.clean_molecular_dataset, dataset, dataset_size=size
            )

            results[size] = metrics

            logger.info(
                f"Full pipeline ({size}): {metrics.molecules_per_second:.1f} mol/s, "
                f"{metrics.memory_usage_mb:.1f} MB, {metrics.processing_time:.2f}s"
            )

        return results

    def test_scaling_behavior(
        self,
        base_size: int = 1000,
        scale_factors: List[int] = [1, 2, 5, 10, 20, 50, 100],
    ) -> Dict[str, Any]:
        """Test scaling behavior of the pipeline.

        Args:
            base_size: Base dataset size
            scale_factors: List of scaling factors to test

        Returns:
            Scaling analysis results
        """
        logger.info(
            f"Testing scaling behavior from {base_size} to {base_size * max(scale_factors)}"
        )

        dataset_sizes = [base_size * factor for factor in scale_factors]

        # Test different components
        smiles_results = self.test_smiles_validation_performance(dataset_sizes)
        inchikey_results = self.test_inchikey_generation_performance(dataset_sizes)
        filter_results = self.test_molecular_filtering_performance(dataset_sizes)
        pipeline_results = self.test_full_pipeline_performance(dataset_sizes)

        # Analyze scaling efficiency
        scaling_analysis = {
            "dataset_sizes": dataset_sizes,
            "scale_factors": scale_factors,
            "components": {
                "smiles_validation": self._analyze_scaling_efficiency(smiles_results),
                "inchikey_generation": self._analyze_scaling_efficiency(
                    inchikey_results
                ),
                "molecular_filtering": self._analyze_scaling_efficiency(filter_results),
                "full_pipeline": self._analyze_scaling_efficiency(pipeline_results),
            },
        }

        return scaling_analysis

    def _analyze_scaling_efficiency(
        self, results: Dict[int, PerformanceMetrics]
    ) -> Dict[str, Any]:
        """Analyze scaling efficiency from performance results.

        Args:
            results: Performance results by dataset size

        Returns:
            Scaling efficiency analysis
        """
        sizes = sorted(results.keys())
        throughputs = [results[size].molecules_per_second for size in sizes]
        processing_times = [results[size].processing_time for size in sizes]
        memory_usage = [results[size].memory_usage_mb for size in sizes]

        # Calculate scaling efficiency
        if len(sizes) >= 2:
            # Theoretical linear scaling
            base_size = sizes[0]
            base_throughput = throughputs[0]

            scaling_efficiency = []
            for i, size in enumerate(sizes[1:], 1):
                expected_time = processing_times[0] * (size / base_size)
                actual_time = processing_times[i]
                efficiency = expected_time / actual_time if actual_time > 0 else 0.0
                scaling_efficiency.append(efficiency)

            avg_efficiency = np.mean(scaling_efficiency) if scaling_efficiency else 0.0
        else:
            avg_efficiency = 1.0
            scaling_efficiency = []

        return {
            "sizes": sizes,
            "throughputs": throughputs,
            "processing_times": processing_times,
            "memory_usage": memory_usage,
            "scaling_efficiency": scaling_efficiency,
            "average_efficiency": avg_efficiency,
            "max_throughput": max(throughputs) if throughputs else 0.0,
            "memory_scaling_factor": memory_usage[-1] / memory_usage[0]
            if len(memory_usage) >= 2
            else 1.0,
        }

    def test_100_million_dataset(self) -> Dict[str, Any]:
        """Test performance with 100 million compound dataset.

        Returns:
            Performance results for 100M dataset
        """
        logger.info("Testing 100 million compound dataset performance")

        # Use a smaller test for development/CI, but support full 100M
        test_size = 100_000_000  # 100 million

        # Check available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        if available_memory_gb < 32:  # Less than 32GB RAM
            logger.warning(
                f"Limited RAM ({available_memory_gb:.1f}GB), using smaller test size"
            )
            test_size = min(
                test_size, int(available_memory_gb * 1_000_000)
            )  # ~1M molecules per GB

        logger.info(f"Testing with {test_size:,} molecules")

        # Test individual components with streaming/batching
        results = {}

        # SMILES validation (most memory efficient)
        logger.info("Testing SMILES validation at scale")
        results["smiles_validation"] = self._test_component_at_scale(
            self.test_smiles_validation_performance, test_size, max_batch_size=1_000_000
        )

        # InChIKey generation
        logger.info("Testing InChIKey generation at scale")
        results["inchikey_generation"] = self._test_component_at_scale(
            self.test_inchikey_generation_performance, test_size, max_batch_size=500_000
        )

        # Molecular filtering
        logger.info("Testing molecular filtering at scale")
        results["molecular_filtering"] = self._test_component_at_scale(
            self.test_molecular_filtering_performance, test_size, max_batch_size=500_000
        )

        # Full pipeline (smaller batches due to memory requirements)
        logger.info("Testing full pipeline at scale")
        results["full_pipeline"] = self._test_component_at_scale(
            self.test_full_pipeline_performance, test_size, max_batch_size=100_000
        )

        # Calculate overall metrics
        results["summary"] = self._calculate_scale_test_summary(results, test_size)

        return results

    def _test_component_at_scale(
        self, test_func, total_size: int, max_batch_size: int
    ) -> Dict[str, Any]:
        """Test a component at scale using batching if necessary.

        Args:
            test_func: Test function to run
            total_size: Total dataset size
            max_batch_size: Maximum batch size

        Returns:
            Aggregated performance results
        """
        if total_size <= max_batch_size:
            # Single batch test
            results = test_func([total_size])
            return {
                "single_batch": True,
                "batch_size": total_size,
                "results": results[total_size],
                "total_time": results[total_size].processing_time,
                "effective_throughput": results[total_size].molecules_per_second,
            }
        else:
            # Multi-batch test
            num_batches = (total_size + max_batch_size - 1) // max_batch_size

            logger.info(
                f"Using {num_batches} batches of ~{max_batch_size:,} molecules each"
            )

            batch_results = []
            total_time = 0.0
            total_memory = 0.0

            for batch_idx in range(num_batches):
                batch_start = batch_idx * max_batch_size
                batch_end = min(batch_start + max_batch_size, total_size)
                batch_size = batch_end - batch_start

                logger.info(
                    f"Processing batch {batch_idx + 1}/{num_batches} ({batch_size:,} molecules)"
                )

                batch_result = test_func([batch_size])
                batch_metrics = batch_result[batch_size]

                batch_results.append(batch_metrics)
                total_time += batch_metrics.processing_time
                total_memory = max(total_memory, batch_metrics.memory_usage_mb)

            # Calculate aggregate metrics
            effective_throughput = total_size / total_time if total_time > 0 else 0.0
            avg_success_rate = np.mean([r.success_rate for r in batch_results])

            return {
                "single_batch": False,
                "num_batches": num_batches,
                "batch_size": max_batch_size,
                "total_time": total_time,
                "effective_throughput": effective_throughput,
                "peak_memory_mb": total_memory,
                "average_success_rate": avg_success_rate,
                "batch_results": [
                    {
                        "throughput": r.molecules_per_second,
                        "memory_mb": r.memory_usage_mb,
                        "time": r.processing_time,
                        "success_rate": r.success_rate,
                    }
                    for r in batch_results
                ],
            }

    def _calculate_scale_test_summary(
        self, results: Dict[str, Any], total_size: int
    ) -> Dict[str, Any]:
        """Calculate summary metrics for scale test.

        Args:
            results: Component test results
            total_size: Total dataset size tested

        Returns:
            Summary metrics
        """
        summary = {"total_molecules": total_size, "components": {}}

        for component, result in results.items():
            if component == "summary":
                continue

            if "effective_throughput" in result:
                summary["components"][component] = {
                    "throughput_mol_per_sec": result["effective_throughput"],
                    "total_time_hours": result["total_time"] / 3600,
                    "peak_memory_gb": result.get("peak_memory_mb", 0) / 1024,
                    "success_rate": result.get("average_success_rate", 0.0),
                }

        # Calculate overall performance
        if summary["components"]:
            avg_throughput = np.mean(
                [
                    comp["throughput_mol_per_sec"]
                    for comp in summary["components"].values()
                ]
            )
            max_memory = max(
                [comp["peak_memory_gb"] for comp in summary["components"].values()]
            )

            summary["overall"] = {
                "average_throughput_mol_per_sec": avg_throughput,
                "peak_memory_gb": max_memory,
                "estimated_100m_time_hours": 100_000_000 / avg_throughput / 3600
                if avg_throughput > 0
                else float("inf"),
            }

        return summary

    def save_results(
        self, results: Dict[str, Any], filename: str = "performance_results.json"
    ):
        """Save performance test results to file.

        Args:
            results: Test results to save
            filename: Output filename
        """
        output_file = self.output_dir / filename

        # Convert PerformanceMetrics objects to dictionaries
        serializable_results = self._make_serializable(results)

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Saved performance results to {output_file}")

    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, PerformanceMetrics):
            return {
                "molecules_per_second": obj.molecules_per_second,
                "memory_usage_mb": obj.memory_usage_mb,
                "gpu_memory_usage_mb": obj.gpu_memory_usage_mb,
                "processing_time": obj.processing_time,
                "success_rate": obj.success_rate,
                "cpu_utilization": obj.cpu_utilization,
                "gpu_utilization": obj.gpu_utilization,
                "metadata": obj.metadata,
            }
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


# Pytest test functions for integration with pytest
class TestDataCurationPerformance:
    """Pytest test class for data curation performance."""

    @pytest.fixture(scope="class")
    def performance_suite(self):
        """Create performance test suite."""
        return PerformanceTestSuite()

    @pytest.mark.performance
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_smiles_validation_small_scale(self, performance_suite, size):
        """Test SMILES validation performance at small scale."""
        results = performance_suite.test_smiles_validation_performance([size])

        metrics = results[size]
        assert metrics.success_rate > 0.9  # At least 90% success rate
        assert metrics.molecules_per_second > 100  # At least 100 mol/s
        assert metrics.memory_usage_mb < 1000  # Less than 1GB memory

    @pytest.mark.performance
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_inchikey_generation_small_scale(self, performance_suite, size):
        """Test InChIKey generation performance at small scale."""
        results = performance_suite.test_inchikey_generation_performance([size])

        metrics = results[size]
        assert metrics.success_rate > 0.9
        assert metrics.molecules_per_second > 50  # At least 50 mol/s
        assert metrics.memory_usage_mb < 2000  # Less than 2GB memory

    @pytest.mark.performance
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_molecular_filtering_small_scale(self, performance_suite, size):
        """Test molecular filtering performance at small scale."""
        results = performance_suite.test_molecular_filtering_performance([size])

        metrics = results[size]
        assert metrics.success_rate > 0.9
        assert metrics.molecules_per_second > 20  # At least 20 mol/s
        assert metrics.memory_usage_mb < 2000  # Less than 2GB memory

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.parametrize("size", [1000, 10000, 100000])
    def test_full_pipeline_medium_scale(self, performance_suite, size):
        """Test full pipeline performance at medium scale."""
        results = performance_suite.test_full_pipeline_performance([size])

        metrics = results[size]
        assert (
            metrics.success_rate > 0.8
        )  # At least 80% success rate (some failures expected)
        assert metrics.molecules_per_second > 10  # At least 10 mol/s
        assert metrics.memory_usage_mb < 8000  # Less than 8GB memory

    @pytest.mark.performance
    @pytest.mark.slow
    def test_scaling_behavior(self, performance_suite):
        """Test scaling behavior of the pipeline."""
        results = performance_suite.test_scaling_behavior(
            base_size=1000, scale_factors=[1, 2, 5, 10]
        )

        # Check that all components show reasonable scaling
        for component, analysis in results["components"].items():
            assert (
                analysis["average_efficiency"] > 0.5
            )  # At least 50% scaling efficiency
            assert analysis["max_throughput"] > 0  # Positive throughput

    @pytest.mark.performance
    @pytest.mark.very_slow
    @pytest.mark.skipif(
        psutil.virtual_memory().available < 16 * 1024**3,
        reason="Requires at least 16GB available RAM",
    )
    def test_100_million_dataset_simulation(self, performance_suite):
        """Test 100 million dataset simulation (scaled down for CI)."""
        # Use a smaller dataset that represents the scaling characteristics
        test_size = min(
            1_000_000, int(psutil.virtual_memory().available / (1024**3) * 100_000)
        )

        results = performance_suite._test_component_at_scale(
            performance_suite.test_smiles_validation_performance,
            test_size,
            max_batch_size=100_000,
        )

        assert results["effective_throughput"] > 100  # At least 100 mol/s
        assert results["average_success_rate"] > 0.9  # At least 90% success

        # Extrapolate to 100M
        estimated_100m_time_hours = 100_000_000 / results["effective_throughput"] / 3600
        assert estimated_100m_time_hours < 24  # Should complete within 24 hours


if __name__ == "__main__":
    # Run performance tests standalone
    suite = PerformanceTestSuite()

    # Test different scales
    small_scale_sizes = [100, 1000, 10000]
    medium_scale_sizes = [10000, 50000, 100000]

    print("Running small scale tests...")
    smiles_results = suite.test_smiles_validation_performance(small_scale_sizes)
    suite.save_results(
        {"smiles_validation_small": smiles_results}, "smiles_small_scale.json"
    )

    print("Running medium scale tests...")
    pipeline_results = suite.test_full_pipeline_performance(medium_scale_sizes)
    suite.save_results(
        {"full_pipeline_medium": pipeline_results}, "pipeline_medium_scale.json"
    )

    print("Running scaling analysis...")
    scaling_results = suite.test_scaling_behavior(
        base_size=1000, scale_factors=[1, 2, 5, 10, 20]
    )
    suite.save_results({"scaling_analysis": scaling_results}, "scaling_analysis.json")

    print("Running 100M dataset simulation...")
    large_scale_results = suite.test_100_million_dataset()
    suite.save_results(
        {"100m_simulation": large_scale_results}, "100m_dataset_simulation.json"
    )

    print("Performance testing complete! Results saved to performance_results/")

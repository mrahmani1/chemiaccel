"""
Performance benchmarks for RAPIDS integration in ChemiAccel.

These benchmarks compare CuPy vs NumPy operations and measure
the performance benefits of GPU acceleration for molecular computations.
"""

import pytest
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any

# Import ChemiAccel components
from chemi_accel.backends.rapids import (
    CupyArrayProcessor,
    check_cupy_availability,
    molecular_weight_cupy,
    tanimoto_similarity_cupy
)

# Benchmark configuration
BENCHMARK_SIZES = [100, 1000, 10000]  # Different dataset sizes
NUM_ITERATIONS = 5  # Number of iterations for timing
WARMUP_ITERATIONS = 2  # Warmup iterations to exclude from timing


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.numpy_times: List[float] = []
        self.cupy_times: List[float] = []
        self.numpy_mean: float = 0.0
        self.cupy_mean: float = 0.0
        self.speedup: float = 0.0
        self.memory_usage: Dict[str, Any] = {}
    
    def calculate_statistics(self):
        """Calculate timing statistics."""
        if self.numpy_times:
            self.numpy_mean = np.mean(self.numpy_times)
        if self.cupy_times:
            self.cupy_mean = np.mean(self.cupy_times)
        
        if self.cupy_mean > 0:
            self.speedup = self.numpy_mean / self.cupy_mean
    
    def __str__(self):
        return (f"{self.name} (size={self.size}): "
                f"NumPy={self.numpy_mean:.4f}s, "
                f"CuPy={self.cupy_mean:.4f}s, "
                f"Speedup={self.speedup:.2f}x")


class RapidsBenchmarkSuite:
    """Benchmark suite for RAPIDS operations."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.cupy_available = check_cupy_availability()
        
        if self.cupy_available:
            self.processor = CupyArrayProcessor()
    
    def time_function(self, func: Callable, *args, **kwargs) -> float:
        """Time a function execution."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Ensure result is computed (important for lazy evaluation)
        if hasattr(result, 'get'):  # CuPy array
            result.get()
        
        return end_time - start_time
    
    def benchmark_array_operations(self, size: int) -> BenchmarkResult:
        """Benchmark basic array operations."""
        result = BenchmarkResult("Array Operations", size)
        
        # Generate test data
        test_data = np.random.rand(size, 100).astype(np.float32)
        
        def numpy_operation(data):
            return np.sum(data ** 2 + np.sqrt(data), axis=1)
        
        def cupy_operation(data):
            import cupy as cp
            return cp.sum(data ** 2 + cp.sqrt(data), axis=1)
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            numpy_operation(test_data)
            if self.cupy_available:
                cupy_data = self.processor.numpy_to_cupy(test_data)
                cupy_operation(cupy_data)
        
        # Benchmark NumPy
        for _ in range(NUM_ITERATIONS):
            elapsed = self.time_function(numpy_operation, test_data)
            result.numpy_times.append(elapsed)
        
        # Benchmark CuPy
        if self.cupy_available:
            cupy_data = self.processor.numpy_to_cupy(test_data)
            
            for _ in range(NUM_ITERATIONS):
                elapsed = self.time_function(cupy_operation, cupy_data)
                result.cupy_times.append(elapsed)
        
        result.calculate_statistics()
        return result
    
    def benchmark_molecular_weight_calculation(self, size: int) -> BenchmarkResult:
        """Benchmark molecular weight calculations."""
        result = BenchmarkResult("Molecular Weight", size)
        
        # Generate molecular weight data (simulated)
        max_atoms = 50
        atom_weights = np.random.rand(size, max_atoms).astype(np.float32) * 20  # Atomic weights 0-20
        
        def numpy_molecular_weight(weights):
            return np.sum(weights, axis=1)
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            numpy_molecular_weight(atom_weights)
            if self.cupy_available:
                cupy_weights = self.processor.numpy_to_cupy(atom_weights)
                molecular_weight_cupy(cupy_weights)
        
        # Benchmark NumPy
        for _ in range(NUM_ITERATIONS):
            elapsed = self.time_function(numpy_molecular_weight, atom_weights)
            result.numpy_times.append(elapsed)
        
        # Benchmark CuPy
        if self.cupy_available:
            cupy_weights = self.processor.numpy_to_cupy(atom_weights)
            
            for _ in range(NUM_ITERATIONS):
                elapsed = self.time_function(molecular_weight_cupy, cupy_weights)
                result.cupy_times.append(elapsed)
        
        result.calculate_statistics()
        return result
    
    def benchmark_tanimoto_similarity(self, size: int) -> BenchmarkResult:
        """Benchmark Tanimoto similarity calculations."""
        result = BenchmarkResult("Tanimoto Similarity", size)
        
        # Generate binary fingerprint data
        fingerprint_length = 1024
        fingerprints1 = np.random.randint(0, 2, (size, fingerprint_length), dtype=bool)
        fingerprints2 = np.random.randint(0, 2, (size, fingerprint_length), dtype=bool)
        
        def numpy_tanimoto(fp1, fp2):
            intersection = np.sum(fp1 & fp2, axis=1)
            union = np.sum(fp1 | fp2, axis=1)
            return np.where(union > 0, intersection / union, 0.0)
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            numpy_tanimoto(fingerprints1, fingerprints2)
            if self.cupy_available:
                cupy_fp1 = self.processor.numpy_to_cupy(fingerprints1)
                cupy_fp2 = self.processor.numpy_to_cupy(fingerprints2)
                tanimoto_similarity_cupy(cupy_fp1, cupy_fp2)
        
        # Benchmark NumPy
        for _ in range(NUM_ITERATIONS):
            elapsed = self.time_function(numpy_tanimoto, fingerprints1, fingerprints2)
            result.numpy_times.append(elapsed)
        
        # Benchmark CuPy
        if self.cupy_available:
            cupy_fp1 = self.processor.numpy_to_cupy(fingerprints1)
            cupy_fp2 = self.processor.numpy_to_cupy(fingerprints2)
            
            for _ in range(NUM_ITERATIONS):
                elapsed = self.time_function(tanimoto_similarity_cupy, cupy_fp1, cupy_fp2)
                result.cupy_times.append(elapsed)
        
        result.calculate_statistics()
        return result
    
    def benchmark_matrix_operations(self, size: int) -> BenchmarkResult:
        """Benchmark matrix operations for molecular graphs."""
        result = BenchmarkResult("Matrix Operations", size)
        
        # Generate adjacency matrices (sparse)
        density = 0.1  # 10% density for sparse matrices
        matrices = []
        for _ in range(size):
            matrix_size = np.random.randint(10, 50)  # Variable size molecules
            matrix = np.random.rand(matrix_size, matrix_size) < density
            matrix = matrix.astype(np.float32)
            # Make symmetric (undirected graph)
            matrix = (matrix + matrix.T) > 0
            matrices.append(matrix)
        
        def numpy_graph_properties(matrices_list):
            results = []
            for matrix in matrices_list:
                # Calculate basic graph properties
                degrees = np.sum(matrix, axis=1)
                num_edges = np.sum(matrix) // 2  # Undirected
                num_nodes = matrix.shape[0]
                
                properties = np.array([
                    num_nodes,
                    num_edges,
                    np.mean(degrees),
                    np.std(degrees),
                    np.max(degrees)
                ])
                results.append(properties)
            
            return np.array(results)
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            numpy_graph_properties(matrices)
            if self.cupy_available:
                self.processor.sparse_matrix_operations(matrices)
        
        # Benchmark NumPy
        for _ in range(NUM_ITERATIONS):
            elapsed = self.time_function(numpy_graph_properties, matrices)
            result.numpy_times.append(elapsed)
        
        # Benchmark CuPy
        if self.cupy_available:
            for _ in range(NUM_ITERATIONS):
                elapsed = self.time_function(self.processor.sparse_matrix_operations, matrices)
                result.cupy_times.append(elapsed)
        
        result.calculate_statistics()
        return result
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks for different sizes."""
        all_results = []
        
        for size in BENCHMARK_SIZES:
            print(f"\nRunning benchmarks for size {size}...")
            
            # Array operations
            result = self.benchmark_array_operations(size)
            all_results.append(result)
            print(f"  {result}")
            
            # Molecular weight
            result = self.benchmark_molecular_weight_calculation(size)
            all_results.append(result)
            print(f"  {result}")
            
            # Tanimoto similarity
            result = self.benchmark_tanimoto_similarity(size)
            all_results.append(result)
            print(f"  {result}")
            
            # Matrix operations
            result = self.benchmark_matrix_operations(size)
            all_results.append(result)
            print(f"  {result}")
        
        self.results = all_results
        return all_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report = ["RAPIDS Performance Benchmark Report", "=" * 40, ""]
        
        if not self.cupy_available:
            report.append("WARNING: CuPy not available - only NumPy timings shown")
            report.append("")
        
        # Group results by benchmark type
        benchmark_groups = {}
        for result in self.results:
            if result.name not in benchmark_groups:
                benchmark_groups[result.name] = []
            benchmark_groups[result.name].append(result)
        
        for benchmark_name, results in benchmark_groups.items():
            report.append(f"{benchmark_name} Benchmark:")
            report.append("-" * (len(benchmark_name) + 11))
            
            for result in results:
                report.append(f"  Size {result.size:>6}: "
                            f"NumPy {result.numpy_mean:>8.4f}s, "
                            f"CuPy {result.cupy_mean:>8.4f}s, "
                            f"Speedup {result.speedup:>6.2f}x")
            
            report.append("")
        
        # Summary statistics
        if self.cupy_available:
            speedups = [r.speedup for r in self.results if r.speedup > 0]
            if speedups:
                report.append("Summary:")
                report.append(f"  Average speedup: {np.mean(speedups):.2f}x")
                report.append(f"  Maximum speedup: {np.max(speedups):.2f}x")
                report.append(f"  Minimum speedup: {np.min(speedups):.2f}x")
        
        return "\n".join(report)


# Test classes for pytest integration
@pytest.mark.skipif(not check_cupy_availability(), reason="CuPy not available")
class TestRapidsPerformanceBenchmarks:
    """Test class for RAPIDS performance benchmarks."""
    
    @pytest.fixture(scope="class")
    def benchmark_suite(self):
        """Create benchmark suite for testing."""
        return RapidsBenchmarkSuite()
    
    def test_array_operations_benchmark(self, benchmark_suite):
        """Test array operations benchmark."""
        result = benchmark_suite.benchmark_array_operations(1000)
        
        assert isinstance(result, BenchmarkResult)
        assert result.name == "Array Operations"
        assert result.size == 1000
        assert len(result.numpy_times) == NUM_ITERATIONS
        
        if benchmark_suite.cupy_available:
            assert len(result.cupy_times) == NUM_ITERATIONS
            assert result.speedup > 0
    
    def test_molecular_weight_benchmark(self, benchmark_suite):
        """Test molecular weight calculation benchmark."""
        result = benchmark_suite.benchmark_molecular_weight_calculation(1000)
        
        assert isinstance(result, BenchmarkResult)
        assert result.name == "Molecular Weight"
        assert len(result.numpy_times) == NUM_ITERATIONS
        
        if benchmark_suite.cupy_available:
            assert len(result.cupy_times) == NUM_ITERATIONS
    
    def test_tanimoto_similarity_benchmark(self, benchmark_suite):
        """Test Tanimoto similarity benchmark."""
        result = benchmark_suite.benchmark_tanimoto_similarity(500)  # Smaller size for similarity
        
        assert isinstance(result, BenchmarkResult)
        assert result.name == "Tanimoto Similarity"
        assert len(result.numpy_times) == NUM_ITERATIONS
        
        if benchmark_suite.cupy_available:
            assert len(result.cupy_times) == NUM_ITERATIONS
    
    def test_matrix_operations_benchmark(self, benchmark_suite):
        """Test matrix operations benchmark."""
        result = benchmark_suite.benchmark_matrix_operations(100)  # Smaller size for matrix ops
        
        assert isinstance(result, BenchmarkResult)
        assert result.name == "Matrix Operations"
        assert len(result.numpy_times) == NUM_ITERATIONS
        
        if benchmark_suite.cupy_available:
            assert len(result.cupy_times) == NUM_ITERATIONS
    
    def test_full_benchmark_suite(self, benchmark_suite):
        """Test running the full benchmark suite."""
        # Use smaller sizes for testing
        original_sizes = BENCHMARK_SIZES.copy()
        global BENCHMARK_SIZES
        BENCHMARK_SIZES = [100, 500]  # Smaller sizes for testing
        
        try:
            results = benchmark_suite.run_all_benchmarks()
            
            assert isinstance(results, list)
            assert len(results) > 0
            
            # Check that we have results for each benchmark type
            benchmark_names = set(r.name for r in results)
            expected_names = {"Array Operations", "Molecular Weight", "Tanimoto Similarity", "Matrix Operations"}
            assert benchmark_names == expected_names
            
            # Generate report
            report = benchmark_suite.generate_report()
            assert isinstance(report, str)
            assert "Benchmark Report" in report
            
        finally:
            # Restore original sizes
            BENCHMARK_SIZES = original_sizes


class TestBenchmarkResultsAnalysis:
    """Test benchmark results analysis."""
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation and statistics."""
        result = BenchmarkResult("Test Benchmark", 1000)
        
        assert result.name == "Test Benchmark"
        assert result.size == 1000
        assert result.numpy_times == []
        assert result.cupy_times == []
    
    def test_benchmark_result_statistics(self):
        """Test benchmark statistics calculation."""
        result = BenchmarkResult("Test", 100)
        
        # Add some timing data
        result.numpy_times = [1.0, 1.1, 0.9, 1.05, 0.95]
        result.cupy_times = [0.2, 0.25, 0.18, 0.22, 0.19]
        
        result.calculate_statistics()
        
        assert abs(result.numpy_mean - 1.0) < 0.1
        assert abs(result.cupy_mean - 0.208) < 0.05
        assert result.speedup > 4.0  # Should be around 5x speedup


if __name__ == "__main__":
    # Run benchmarks directly
    print("Running RAPIDS Performance Benchmarks...")
    
    suite = RapidsBenchmarkSuite()
    
    if not suite.cupy_available:
        print("WARNING: CuPy not available. Only NumPy benchmarks will be run.")
    
    results = suite.run_all_benchmarks()
    
    print("\n" + "=" * 60)
    print(suite.generate_report())
    
    # Also run pytest
    pytest.main([__file__, "-v", "-s"])
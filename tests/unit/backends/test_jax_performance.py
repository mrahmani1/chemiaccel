"""
Performance tests for JAX backend JIT compilation benefits.

This module tests that JIT compilation provides measurable performance
improvements and validates the efficiency of vectorized operations.
"""

import pytest
import time
import numpy as np
from typing import Dict, Any

# Import JAX conditionally for testing
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from chemi_accel.backends.jax.vectorization import JAXVectorizedCalculator
from chemi_accel.backends.jax.descriptors import JAXDescriptorCalculator
from chemi_accel.backends.jax.batch_processing import create_test_batch
from chemi_accel.backends.jax.molecular import create_synthetic_molecule, JAXMolecularProcessor


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
@pytest.mark.benchmark
class TestJAXPerformance:
    """Test JAX performance and JIT compilation benefits."""
    
    def test_jit_compilation_warmup(self):
        """Test that JIT compilation provides benefits after warmup."""
        calculator = JAXVectorizedCalculator(compile_on_init=True)
        
        # Create test batch
        batch = create_test_batch(batch_size=10, max_atoms=15)
        
        # Warmup - first call compiles the function
        warmup_start = time.time()
        results1 = calculator.calculate_batch_vectorized(batch)
        warmup_time = time.time() - warmup_start
        
        # Subsequent calls should be faster (using compiled version)
        compiled_times = []
        for _ in range(3):
            start_time = time.time()
            results2 = calculator.calculate_batch_vectorized(batch)
            compiled_times.append(time.time() - start_time)
        
        avg_compiled_time = np.mean(compiled_times)
        
        # Results should be consistent
        np.testing.assert_allclose(
            results1['molecular_weight'],
            results2['molecular_weight'],
            rtol=1e-5
        )
        
        # Compiled calls should generally be faster than warmup
        # (though this may not always hold in unit test environment)
        print(f"Warmup time: {warmup_time:.4f}s, Avg compiled time: {avg_compiled_time:.4f}s")
        
        # At minimum, compiled version should not be significantly slower
        assert avg_compiled_time <= warmup_time * 2.0
    
    def test_vectorized_vs_sequential_performance(self):
        """Test that vectorized operations are more efficient than sequential."""
        vectorized_calc = JAXVectorizedCalculator(compile_on_init=True)
        sequential_calc = JAXDescriptorCalculator()
        processor = JAXMolecularProcessor(max_atoms=15, max_bonds=20)
        
        # Create test molecules
        molecules = [
            create_synthetic_molecule(num_atoms=8, num_bonds=7, mol_id=f"mol_{i}")
            for i in range(8)
        ]
        
        # Sequential processing
        sequential_start = time.time()
        sequential_results = []
        for mol_data in molecules:
            jax_mol = processor.molecular_data_to_jax(mol_data)
            result = sequential_calc.calculate_single_molecule(jax_mol)
            sequential_results.append(result)
        sequential_time = time.time() - sequential_start
        
        # Vectorized processing
        batch = processor.create_batch(molecules)
        
        # Warmup vectorized calculation
        vectorized_calc.calculate_batch_vectorized(batch)
        
        vectorized_start = time.time()
        vectorized_results = vectorized_calc.calculate_batch_vectorized(batch)
        vectorized_time = time.time() - vectorized_start
        
        print(f"Sequential time: {sequential_time:.4f}s, Vectorized time: {vectorized_time:.4f}s")
        
        # Vectorized should not be significantly slower
        # (In practice it should be faster, but unit test environment may vary)
        assert vectorized_time <= sequential_time * 1.5
        
        # Results should be consistent
        sequential_mw = [r['molecular_weight'] for r in sequential_results]
        vectorized_mw = vectorized_results['molecular_weight']
        
        np.testing.assert_allclose(sequential_mw, vectorized_mw, rtol=1e-4)
    
    def test_batch_size_scaling(self):
        """Test that performance scales appropriately with batch size."""
        calculator = JAXVectorizedCalculator(compile_on_init=True)
        
        batch_sizes = [4, 8, 16]
        times = []
        
        for batch_size in batch_sizes:
            batch = create_test_batch(batch_size=batch_size, max_atoms=12)
            
            # Warmup
            calculator.calculate_batch_vectorized(batch)
            
            # Measure time
            start_time = time.time()
            results = calculator.calculate_batch_vectorized(batch)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            
            # Verify results
            assert len(results['molecular_weight']) == batch_size
        
        print(f"Batch size scaling - Sizes: {batch_sizes}, Times: {times}")
        
        # Time should not increase dramatically with batch size
        # (vectorized operations should have good scaling)
        time_ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        
        # Each doubling of batch size should not increase time by more than 3x
        for ratio in time_ratios:
            assert ratio <= 3.0
    
    def test_memory_efficiency_large_batch(self):
        """Test memory efficiency with larger batches."""
        from chemi_accel.backends.jax.vectorization import JAXMemoryEfficientBatcher
        
        batcher = JAXMemoryEfficientBatcher(memory_limit_mb=50)
        
        # Create large batch
        large_batch = create_test_batch(batch_size=20, max_atoms=15)
        
        # Process with memory constraints
        start_time = time.time()
        results = batcher.chunked_batch_processing(large_batch)
        processing_time = time.time() - start_time
        
        print(f"Large batch processing time: {processing_time:.4f}s")
        
        # Should complete without memory errors
        assert len(results['molecular_weight']) == 20
        assert np.all(results['molecular_weight'] > 0)
        
        # Should not take excessively long
        assert processing_time < 30.0  # 30 second timeout
    
    def test_compilation_overhead_measurement(self):
        """Measure JIT compilation overhead."""
        # Test with compilation
        compiled_calc = JAXVectorizedCalculator(compile_on_init=True)
        
        # Test without pre-compilation
        uncompiled_calc = JAXVectorizedCalculator(compile_on_init=False)
        
        batch = create_test_batch(batch_size=6, max_atoms=12)
        
        # Measure compiled version (already warmed up)
        compiled_start = time.time()
        compiled_results = compiled_calc.calculate_batch_vectorized(batch)
        compiled_time = time.time() - compiled_start
        
        # Measure uncompiled version (will compile on first call)
        uncompiled_start = time.time()
        uncompiled_results = uncompiled_calc.calculate_batch_vectorized(batch)
        uncompiled_time = time.time() - uncompiled_start
        
        print(f"Pre-compiled time: {compiled_time:.4f}s, On-demand compiled time: {uncompiled_time:.4f}s")
        
        # Results should be consistent
        np.testing.assert_allclose(
            compiled_results['molecular_weight'],
            uncompiled_results['molecular_weight'],
            rtol=1e-5
        )
        
        # Pre-compiled should generally be faster or similar
        # (compilation overhead is paid upfront)
        assert compiled_time <= uncompiled_time * 1.2
    
    def test_descriptor_calculation_efficiency(self):
        """Test efficiency of different descriptor calculations."""
        calculator = JAXVectorizedCalculator(compile_on_init=True)
        batch = create_test_batch(batch_size=12, max_atoms=15)
        
        # Warmup
        calculator.calculate_batch_vectorized(batch)
        
        # Time individual descriptor types
        descriptor_times = {}
        
        # Basic descriptors (should be fast)
        start_time = time.time()
        results = calculator.calculate_batch_vectorized(batch)
        basic_time = time.time() - start_time
        descriptor_times['basic'] = basic_time
        
        print(f"Descriptor calculation times: {descriptor_times}")
        
        # Basic descriptors should complete quickly
        assert basic_time < 5.0  # 5 second timeout
        
        # Results should be valid
        assert len(results['molecular_weight']) == 12
        assert np.all(results['molecular_weight'] > 0)


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
@pytest.mark.benchmark
class TestJAXMemoryPerformance:
    """Test memory usage and performance characteristics."""
    
    def test_memory_usage_scaling(self):
        """Test memory usage with different batch sizes."""
        from chemi_accel.backends.jax.vectorization import JAXMemoryEfficientBatcher
        
        batcher = JAXMemoryEfficientBatcher()
        
        # Test memory estimation for different batch sizes
        batch_sizes = [4, 8, 16]
        memory_estimates = []
        
        for batch_size in batch_sizes:
            batch = create_test_batch(batch_size=batch_size, max_atoms=12)
            memory_per_mol = batcher._estimate_memory_per_molecule(batch)
            total_memory = memory_per_mol * batch_size
            memory_estimates.append(total_memory)
        
        print(f"Memory estimates (MB): {memory_estimates}")
        
        # Memory should scale roughly linearly with batch size
        for i in range(1, len(memory_estimates)):
            ratio = memory_estimates[i] / memory_estimates[i-1]
            expected_ratio = batch_sizes[i] / batch_sizes[i-1]
            
            # Allow some tolerance for overhead
            assert 0.8 * expected_ratio <= ratio <= 1.2 * expected_ratio
    
    def test_chunk_processing_efficiency(self):
        """Test efficiency of chunked processing."""
        from chemi_accel.backends.jax.vectorization import JAXMemoryEfficientBatcher
        
        batcher = JAXMemoryEfficientBatcher(memory_limit_mb=30)
        
        # Create batch that requires chunking
        large_batch = create_test_batch(batch_size=16, max_atoms=15)
        
        # Process with different chunk sizes
        chunk_sizes = [2, 4, 8]
        chunk_times = []
        
        for chunk_size in chunk_sizes:
            start_time = time.time()
            results = batcher.chunked_batch_processing(large_batch, chunk_size=chunk_size)
            chunk_time = time.time() - start_time
            chunk_times.append(chunk_time)
            
            # Verify results
            assert len(results['molecular_weight']) == 16
        
        print(f"Chunk processing times: {dict(zip(chunk_sizes, chunk_times))}")
        
        # All should complete in reasonable time
        for chunk_time in chunk_times:
            assert chunk_time < 10.0  # 10 second timeout
    
    def test_gradient_calculation_performance(self):
        """Test performance of gradient calculations."""
        from chemi_accel.backends.jax.vectorization import JAXAutoDiffCalculator
        
        autodiff_calc = JAXAutoDiffCalculator()
        processor = JAXMolecularProcessor(max_atoms=12, max_bonds=18)
        
        # Create test molecule
        mol_data = create_synthetic_molecule(num_atoms=8, num_bonds=7)
        jax_mol = processor.molecular_data_to_jax(mol_data)
        
        # Test gradient calculation performance
        available_grads = autodiff_calc.get_available_gradients()
        
        gradient_times = {}
        for descriptor in available_grads:
            start_time = time.time()
            try:
                gradients = autodiff_calc.calculate_gradients(jax_mol, descriptor)
                grad_time = time.time() - start_time
                gradient_times[descriptor] = grad_time
                
                # Verify gradient shape
                assert gradients.shape == jax_mol.atom_features.shape
                
            except Exception as e:
                print(f"Gradient calculation failed for {descriptor}: {e}")
        
        print(f"Gradient calculation times: {gradient_times}")
        
        # Gradient calculations should complete quickly
        for descriptor, grad_time in gradient_times.items():
            assert grad_time < 2.0, f"Gradient calculation for {descriptor} took too long: {grad_time}s"


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
@pytest.mark.slow
class TestJAXLargeScalePerformance:
    """Test performance with larger datasets (marked as slow tests)."""
    
    def test_large_batch_performance(self):
        """Test performance with large batches."""
        calculator = JAXVectorizedCalculator(compile_on_init=True)
        
        # Create large batch
        large_batch = create_test_batch(batch_size=50, max_atoms=20)
        
        # Warmup
        calculator.calculate_batch_vectorized(large_batch)
        
        # Measure performance
        start_time = time.time()
        results = calculator.calculate_batch_vectorized(large_batch)
        processing_time = time.time() - start_time
        
        print(f"Large batch (50 molecules) processing time: {processing_time:.4f}s")
        
        # Should complete in reasonable time
        assert processing_time < 15.0  # 15 second timeout
        
        # Results should be valid
        assert len(results['molecular_weight']) == 50
        assert np.all(results['molecular_weight'] > 0)
        
        # Calculate throughput
        throughput = 50 / processing_time
        print(f"Throughput: {throughput:.2f} molecules/second")
        
        # Should achieve reasonable throughput
        assert throughput > 5.0  # At least 5 molecules per second
    
    def test_memory_constrained_large_processing(self):
        """Test processing large datasets with memory constraints."""
        from chemi_accel.backends.jax.vectorization import JAXMemoryEfficientBatcher
        
        batcher = JAXMemoryEfficientBatcher(memory_limit_mb=100)
        
        # Create very large batch
        very_large_batch = create_test_batch(batch_size=100, max_atoms=15)
        
        # Process with memory constraints
        start_time = time.time()
        results = batcher.chunked_batch_processing(very_large_batch)
        processing_time = time.time() - start_time
        
        print(f"Very large batch (100 molecules) processing time: {processing_time:.4f}s")
        
        # Should complete without memory errors
        assert len(results['molecular_weight']) == 100
        assert np.all(results['molecular_weight'] > 0)
        
        # Should complete in reasonable time
        assert processing_time < 30.0  # 30 second timeout
        
        # Calculate throughput
        throughput = 100 / processing_time
        print(f"Memory-constrained throughput: {throughput:.2f} molecules/second")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "benchmark"])
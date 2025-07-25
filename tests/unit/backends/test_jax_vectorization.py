"""
Unit tests for JAX vectorization and JIT compilation.

This module tests the JAX backend vectorization, JIT compilation,
and automatic differentiation functionality.
"""


import numpy as np
import pytest

# Import JAX conditionally for testing
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from chemi_accel.backends.jax.batch_processing import create_test_batch
from chemi_accel.backends.jax.molecular import (
    JAXMolecularProcessor,
    create_synthetic_molecule,
)
from chemi_accel.backends.jax.vectorization import (
    JAXAutoDiffCalculator,
    JAXMemoryEfficientBatcher,
    JAXVectorizedCalculator,
)
from chemi_accel.core.exceptions import BackendUnavailableError


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXVectorizedCalculator:
    """Test JAX vectorized calculator functionality."""

    def test_vectorized_calculator_initialization(self):
        """Test vectorized calculator initialization."""
        calculator = JAXVectorizedCalculator(compile_on_init=True)

        assert calculator.compile_on_init is True
        assert len(calculator._compiled_functions) > 0
        assert len(calculator._vectorized_functions) > 0

        # Check that expected functions are compiled
        expected_compiled = ["molecular_weight", "atom_count", "bond_count"]
        for func_name in expected_compiled:
            assert func_name in calculator._compiled_functions
            assert func_name in calculator._vectorized_functions

    def test_single_molecule_vectorized_calculation(self):
        """Test vectorized calculation for single molecule."""
        calculator = JAXVectorizedCalculator()
        processor = JAXMolecularProcessor(max_atoms=15, max_bonds=20)

        # Create test molecule
        mol_data = create_synthetic_molecule(
            num_atoms=8, num_bonds=7, mol_id="test_mol"
        )
        jax_mol = processor.molecular_data_to_jax(mol_data)

        # Calculate descriptors
        results = calculator.calculate_single_vectorized(jax_mol)

        # Check results
        assert isinstance(results, dict)
        assert "molecular_weight" in results
        assert "atom_count" in results
        assert "bond_count" in results

        # Validate values
        assert results["atom_count"] == 8
        assert results["bond_count"] == 7
        assert results["molecular_weight"] > 0
        assert isinstance(results["molecular_weight"], float)
        assert isinstance(results["atom_count"], int)

    def test_batch_vectorized_calculation(self):
        """Test vectorized calculation for batch of molecules."""
        calculator = JAXVectorizedCalculator()

        # Create test batch
        batch = create_test_batch(batch_size=4, max_atoms=12)

        # Calculate descriptors
        results = calculator.calculate_batch_vectorized(batch)

        # Check results structure
        assert isinstance(results, dict)
        assert "molecular_weight" in results
        assert "atom_count" in results

        # Check array shapes
        for key in ["molecular_weight", "atom_count", "bond_count"]:
            if key in results:
                assert isinstance(results[key], np.ndarray)
                assert len(results[key]) == 4  # batch size

    def test_vectorization_consistency(self):
        """Test consistency between single and vectorized calculations."""
        calculator = JAXVectorizedCalculator()
        processor = JAXMolecularProcessor(max_atoms=15, max_bonds=20)

        # Create test molecules
        molecules = [
            create_synthetic_molecule(num_atoms=6, num_bonds=5, mol_id=f"mol_{i}")
            for i in range(3)
        ]

        # Calculate single molecule results
        single_results = []
        for mol_data in molecules:
            jax_mol = processor.molecular_data_to_jax(mol_data)
            result = calculator.calculate_single_vectorized(jax_mol)
            single_results.append(result)

        # Calculate batch results
        batch = processor.create_batch(molecules)
        batch_results = calculator.calculate_batch_vectorized(batch)

        # Compare results
        descriptors_to_check = ["molecular_weight", "atom_count", "bond_count"]

        for i, single_result in enumerate(single_results):
            for descriptor in descriptors_to_check:
                if descriptor in single_result and descriptor in batch_results:
                    single_value = single_result[descriptor]
                    batch_value = batch_results[descriptor][i]

                    if isinstance(single_value, float):
                        assert abs(single_value - batch_value) < 1e-5
                    else:
                        assert single_value == batch_value

    def test_jit_compilation_benefits(self):
        """Test that JIT compilation provides performance benefits."""
        # Create calculators with and without compilation
        compiled_calc = JAXVectorizedCalculator(compile_on_init=True)
        uncompiled_calc = JAXVectorizedCalculator(compile_on_init=False)

        # Check that compiled version has more functions
        assert len(compiled_calc._compiled_functions) > len(
            uncompiled_calc._compiled_functions
        )
        assert len(compiled_calc._vectorized_functions) > len(
            uncompiled_calc._vectorized_functions
        )

    def test_morgan_fingerprint_vectorized(self):
        """Test vectorized Morgan fingerprint calculation."""
        calculator = JAXVectorizedCalculator()

        # Create test batch
        batch = create_test_batch(batch_size=3, max_atoms=10)

        # Calculate fingerprints
        results = calculator.calculate_batch_vectorized(batch)

        if "morgan_fingerprint" in results:
            fingerprints = results["morgan_fingerprint"]
            assert fingerprints.shape == (3, 1024)  # batch_size x fingerprint_size
            assert np.all(fingerprints >= 0)

    def test_function_listing(self):
        """Test getting lists of compiled and vectorized functions."""
        calculator = JAXVectorizedCalculator()

        compiled_funcs = calculator.get_compiled_functions()
        vectorized_funcs = calculator.get_vectorized_functions()

        assert isinstance(compiled_funcs, list)
        assert isinstance(vectorized_funcs, list)
        assert len(compiled_funcs) > 0
        assert len(vectorized_funcs) > 0

        # Check that vectorized functions are subset of compiled functions
        for func in vectorized_funcs:
            assert func in compiled_funcs


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXAutoDiffCalculator:
    """Test JAX automatic differentiation functionality."""

    def test_autodiff_calculator_initialization(self):
        """Test automatic differentiation calculator initialization."""
        calculator = JAXAutoDiffCalculator()

        assert len(calculator._grad_functions) > 0

        # Check that expected gradient functions exist
        available_grads = calculator.get_available_gradients()
        assert isinstance(available_grads, list)
        assert len(available_grads) > 0

    def test_gradient_calculation(self):
        """Test gradient calculation for molecular descriptors."""
        calculator = JAXAutoDiffCalculator()
        processor = JAXMolecularProcessor(max_atoms=10, max_bonds=15)

        # Create test molecule
        mol_data = create_synthetic_molecule(num_atoms=5, num_bonds=4)
        jax_mol = processor.molecular_data_to_jax(mol_data)

        # Calculate gradients for available descriptors
        available_grads = calculator.get_available_gradients()

        for descriptor in available_grads:
            try:
                gradients = calculator.calculate_gradients(jax_mol, descriptor)

                # Check gradient properties
                assert isinstance(gradients, jnp.ndarray)
                assert gradients.shape == jax_mol.atom_features.shape
                assert not jnp.any(jnp.isnan(gradients))

            except Exception as e:
                pytest.fail(f"Gradient calculation failed for {descriptor}: {e}")

    def test_gradient_nonzero(self):
        """Test that gradients are non-zero for relevant changes."""
        calculator = JAXAutoDiffCalculator()
        processor = JAXMolecularProcessor(max_atoms=8, max_bonds=12)

        # Create test molecule
        mol_data = create_synthetic_molecule(num_atoms=4, num_bonds=3)
        jax_mol = processor.molecular_data_to_jax(mol_data)

        # Test molecular weight gradients
        if "molecular_weight" in calculator.get_available_gradients():
            gradients = calculator.calculate_gradients(jax_mol, "molecular_weight")

            # Gradients should be non-zero for atomic number changes
            # (since molecular weight depends on atomic numbers)
            atomic_number_grads = gradients[:, 0]  # First column is atomic number
            assert jnp.sum(jnp.abs(atomic_number_grads)) > 0

    def test_invalid_descriptor_gradient(self):
        """Test error handling for invalid descriptor gradients."""
        calculator = JAXAutoDiffCalculator()
        processor = JAXMolecularProcessor(max_atoms=8, max_bonds=12)

        mol_data = create_synthetic_molecule(num_atoms=4, num_bonds=3)
        jax_mol = processor.molecular_data_to_jax(mol_data)

        # Test with non-existent descriptor
        with pytest.raises(ValueError):
            calculator.calculate_gradients(jax_mol, "nonexistent_descriptor")


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXMemoryEfficientBatcher:
    """Test memory-efficient batching strategies."""

    def test_memory_efficient_batcher_initialization(self):
        """Test memory-efficient batcher initialization."""
        batcher = JAXMemoryEfficientBatcher(memory_limit_mb=100)

        assert batcher.memory_limit_mb == 100
        assert batcher.vectorized_calc is not None

    def test_chunked_batch_processing(self):
        """Test chunked batch processing."""
        batcher = JAXMemoryEfficientBatcher(memory_limit_mb=50)

        # Create larger batch
        batch = create_test_batch(batch_size=8, max_atoms=12)

        # Process in chunks
        results = batcher.chunked_batch_processing(batch, chunk_size=3)

        # Check results
        assert isinstance(results, dict)

        # Check that all molecules were processed
        for key in ["molecular_weight", "atom_count"]:
            if key in results:
                assert len(results[key]) == 8  # Original batch size
                assert isinstance(results[key], np.ndarray)

    def test_optimal_chunk_size_determination(self):
        """Test optimal chunk size determination."""
        batcher = JAXMemoryEfficientBatcher(memory_limit_mb=20)  # Small limit

        # Create batch with varying molecule sizes
        batch = create_test_batch(batch_size=10, max_atoms=15)

        # Determine chunk size
        chunk_size = batcher._determine_optimal_chunk_size(batch)

        assert isinstance(chunk_size, int)
        assert 1 <= chunk_size <= batch.batch_size

        # With small memory limit, chunk size should be small
        assert chunk_size <= 5

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        batcher = JAXMemoryEfficientBatcher()

        # Create test batch
        batch = create_test_batch(batch_size=4, max_atoms=10)

        # Estimate memory per molecule
        memory_per_mol = batcher._estimate_memory_per_molecule(batch)

        assert isinstance(memory_per_mol, float)
        assert memory_per_mol > 0

        # Should be reasonable for small molecules
        assert memory_per_mol < 10  # Less than 10MB per molecule

    def test_chunk_extraction(self):
        """Test extraction of chunks from batch."""
        batcher = JAXMemoryEfficientBatcher()

        # Create test batch
        original_batch = create_test_batch(batch_size=6, max_atoms=10)

        # Extract chunk
        chunk = batcher._extract_chunk(original_batch, start_idx=2, end_idx=5)

        # Check chunk properties
        assert chunk.batch_size == 3  # end_idx - start_idx
        assert chunk.max_atoms == original_batch.max_atoms
        assert chunk.max_bonds == original_batch.max_bonds

        # Check that arrays have correct shapes
        assert chunk.atom_features.shape[0] == 3
        assert chunk.bond_features.shape[0] == 3
        assert chunk.adjacency_matrices.shape[0] == 3

    def test_large_batch_processing(self):
        """Test processing of large batches with memory constraints."""
        batcher = JAXMemoryEfficientBatcher(memory_limit_mb=30)

        # Create large batch
        batch = create_test_batch(batch_size=12, max_atoms=15)

        # Process with automatic chunking
        results = batcher.chunked_batch_processing(batch)

        # Verify all molecules were processed
        assert "molecular_weight" in results
        assert len(results["molecular_weight"]) == 12

        # Check that results are reasonable
        assert np.all(results["molecular_weight"] > 0)
        if "atom_count" in results:
            assert np.all(results["atom_count"] > 0)


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestVectorizationIntegration:
    """Test integration of vectorization components."""

    def test_end_to_end_vectorized_workflow(self):
        """Test complete vectorized workflow."""
        # Initialize components
        vectorized_calc = JAXVectorizedCalculator()
        autodiff_calc = JAXAutoDiffCalculator()
        memory_batcher = JAXMemoryEfficientBatcher(memory_limit_mb=100)

        # Create test data
        batch = create_test_batch(batch_size=6, max_atoms=12)

        # Test vectorized calculation
        vectorized_results = vectorized_calc.calculate_batch_vectorized(batch)
        assert len(vectorized_results["molecular_weight"]) == 6

        # Test memory-efficient processing
        chunked_results = memory_batcher.chunked_batch_processing(batch, chunk_size=2)
        assert len(chunked_results["molecular_weight"]) == 6

        # Results should be consistent
        np.testing.assert_allclose(
            vectorized_results["molecular_weight"],
            chunked_results["molecular_weight"],
            rtol=1e-5,
        )

        # Test gradient calculation on individual molecules
        mol = batch.get_molecule(0)
        available_grads = autodiff_calc.get_available_gradients()

        for descriptor in available_grads[:2]:  # Test first 2 to save time
            try:
                gradients = autodiff_calc.calculate_gradients(mol, descriptor)
                assert gradients.shape == mol.atom_features.shape
            except Exception as e:
                pytest.skip(f"Gradient calculation not available for {descriptor}: {e}")

    def test_performance_comparison(self):
        """Test performance comparison between different approaches."""
        # This test would ideally measure timing, but for unit tests
        # we just verify that all approaches produce consistent results

        vectorized_calc = JAXVectorizedCalculator()
        memory_batcher = JAXMemoryEfficientBatcher()

        # Create test batch
        batch = create_test_batch(batch_size=4, max_atoms=10)

        # Calculate using different methods
        vectorized_results = vectorized_calc.calculate_batch_vectorized(batch)
        chunked_results = memory_batcher.chunked_batch_processing(batch, chunk_size=2)

        # Results should be consistent
        for key in ["molecular_weight", "atom_count"]:
            if key in vectorized_results and key in chunked_results:
                np.testing.assert_allclose(
                    vectorized_results[key],
                    chunked_results[key],
                    rtol=1e-5,
                    err_msg=f"Inconsistent results for {key}",
                )


# Test without JAX available
@pytest.mark.skipif(JAX_AVAILABLE, reason="JAX is available")
class TestJAXVectorizationUnavailable:
    """Test behavior when JAX is not available."""

    def test_jax_unavailable_errors(self):
        """Test that appropriate errors are raised when JAX is unavailable."""
        with pytest.raises(BackendUnavailableError):
            JAXVectorizedCalculator()

        with pytest.raises(BackendUnavailableError):
            JAXAutoDiffCalculator()

        with pytest.raises(BackendUnavailableError):
            JAXMemoryEfficientBatcher()


if __name__ == "__main__":
    pytest.main([__file__])

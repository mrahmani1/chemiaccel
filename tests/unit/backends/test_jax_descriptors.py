"""
Unit tests for JAX descriptor calculations.

This module tests the JAX backend descriptor calculation functions
with synthetic molecular data.
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
from chemi_accel.backends.jax.descriptors import (
    JAXDescriptorCalculator,
    batch_fingerprints,
    calculate_molecular_fingerprint,
    calculate_topological_indices,
)
from chemi_accel.backends.jax.molecular import (
    JAXMolecularProcessor,
    create_synthetic_molecule,
)
from chemi_accel.core.exceptions import BackendUnavailableError


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXDescriptorCalculator:
    """Test JAX descriptor calculator."""

    def test_calculator_initialization(self):
        """Test descriptor calculator initialization."""
        calculator = JAXDescriptorCalculator()

        assert calculator._compiled_functions is not None
        assert len(calculator._compiled_functions) > 0

        # Check that basic functions are compiled
        expected_functions = [
            "molecular_weight",
            "atom_count",
            "bond_count",
            "ring_count",
            "aromatic_atom_count",
            "formal_charge",
        ]

        for func_name in expected_functions:
            assert func_name in calculator._compiled_functions

    def test_single_molecule_descriptors(self):
        """Test descriptor calculation for single molecule."""
        calculator = JAXDescriptorCalculator()
        processor = JAXMolecularProcessor(max_atoms=20, max_bonds=30)

        # Create test molecule
        mol_data = create_synthetic_molecule(
            num_atoms=8, num_bonds=7, mol_id="test_mol"
        )
        jax_mol = processor.molecular_data_to_jax(mol_data)

        # Calculate descriptors
        results = calculator.calculate_single_molecule(jax_mol)

        # Check that all expected descriptors are present
        expected_descriptors = [
            "molecular_weight",
            "atom_count",
            "bond_count",
            "ring_atom_count",
            "aromatic_atom_count",
            "formal_charge",
        ]

        for descriptor in expected_descriptors:
            assert descriptor in results
            assert isinstance(results[descriptor], (int, float))

        # Check metadata
        assert results["mol_id"] == "test_mol"
        assert results["smiles"] is not None

        # Validate specific values
        assert results["atom_count"] == 8
        assert results["bond_count"] == 7
        assert results["molecular_weight"] > 0

    def test_batch_descriptors(self):
        """Test descriptor calculation for batch of molecules."""
        calculator = JAXDescriptorCalculator()

        # Create test batch
        batch = create_test_batch(batch_size=4, max_atoms=15)

        # Calculate batch descriptors
        results = calculator.calculate_batch(batch)

        # Check result structure
        assert isinstance(results, dict)
        assert "batch_size" in results
        assert results["batch_size"] == 4

        # Check descriptor arrays
        expected_descriptors = [
            "molecular_weight",
            "atom_count",
            "bond_count",
            "ring_atom_count",
            "aromatic_atom_count",
            "formal_charge",
        ]

        for descriptor in expected_descriptors:
            assert descriptor in results
            assert isinstance(results[descriptor], np.ndarray)
            assert len(results[descriptor]) == 4

        # Check metadata
        assert "mol_ids" in results
        assert "smiles_list" in results
        assert len(results["mol_ids"]) == 4

    def test_descriptor_consistency(self):
        """Test consistency between single and batch calculations."""
        calculator = JAXDescriptorCalculator()
        processor = JAXMolecularProcessor(max_atoms=20, max_bonds=30)

        # Create test molecules
        molecules = [
            create_synthetic_molecule(num_atoms=6, num_bonds=5, mol_id=f"mol_{i}")
            for i in range(3)
        ]

        # Calculate single molecule descriptors
        single_results = []
        for mol_data in molecules:
            jax_mol = processor.molecular_data_to_jax(mol_data)
            result = calculator.calculate_single_molecule(jax_mol)
            single_results.append(result)

        # Calculate batch descriptors
        batch = processor.create_batch(molecules)
        batch_results = calculator.calculate_batch(batch)

        # Compare results
        descriptors_to_check = ["molecular_weight", "atom_count", "bond_count"]

        for i, single_result in enumerate(single_results):
            for descriptor in descriptors_to_check:
                single_value = single_result[descriptor]
                batch_value = batch_results[descriptor][i]

                # Allow small numerical differences due to compilation
                if isinstance(single_value, float):
                    assert abs(single_value - batch_value) < 1e-5
                else:
                    assert single_value == batch_value

    def test_available_descriptors(self):
        """Test getting available descriptors."""
        calculator = JAXDescriptorCalculator()

        descriptors = calculator.get_available_descriptors()

        assert isinstance(descriptors, dict)
        assert len(descriptors) > 0

        # Check that all descriptors have descriptions
        for name, description in descriptors.items():
            assert isinstance(name, str)
            assert isinstance(description, str)
            assert len(description) > 0

    def test_error_handling(self):
        """Test error handling in descriptor calculations."""
        calculator = JAXDescriptorCalculator()

        # Test with invalid molecule (this would need to be constructed carefully)
        # For now, we test that the calculator handles exceptions gracefully

        # This test ensures the error handling structure is in place
        assert hasattr(calculator, "calculate_single_molecule")
        assert hasattr(calculator, "calculate_batch")


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestTopologicalIndices:
    """Test topological indices calculations."""

    def test_topological_indices_calculation(self):
        """Test calculation of topological indices."""
        processor = JAXMolecularProcessor(max_atoms=15, max_bonds=20)

        # Create test molecule with known structure
        mol_data = create_synthetic_molecule(num_atoms=6, num_bonds=5)
        jax_mol = processor.molecular_data_to_jax(mol_data)

        # Calculate topological indices
        indices = calculate_topological_indices(jax_mol)

        # Check that all expected indices are present
        expected_indices = ["wiener_index", "zagreb_1", "zagreb_2", "randic_index"]

        for index_name in expected_indices:
            assert index_name in indices
            assert isinstance(indices[index_name], float)
            assert not np.isnan(indices[index_name])
            assert not np.isinf(indices[index_name])

    def test_topological_indices_properties(self):
        """Test properties of topological indices."""
        processor = JAXMolecularProcessor(max_atoms=10, max_bonds=15)

        # Create simple chain molecule
        mol_data = create_synthetic_molecule(num_atoms=5, num_bonds=4)
        jax_mol = processor.molecular_data_to_jax(mol_data)

        indices = calculate_topological_indices(jax_mol)

        # Basic sanity checks
        assert indices["wiener_index"] >= 0
        assert indices["zagreb_1"] >= 0
        assert indices["zagreb_2"] >= 0
        assert indices["randic_index"] >= 0


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestMolecularFingerprints:
    """Test molecular fingerprint calculations."""

    def test_single_molecule_fingerprint(self):
        """Test fingerprint calculation for single molecule."""
        processor = JAXMolecularProcessor(max_atoms=15, max_bonds=20)

        # Create test molecule
        mol_data = create_synthetic_molecule(num_atoms=8, num_bonds=7)
        jax_mol = processor.molecular_data_to_jax(mol_data)

        # Calculate fingerprint
        fingerprint = calculate_molecular_fingerprint(jax_mol, radius=2)

        # Check fingerprint properties
        assert isinstance(fingerprint, jnp.ndarray)
        assert fingerprint.shape == (1024,)  # Default fingerprint size
        assert jnp.sum(fingerprint) > 0  # Should have some non-zero elements
        assert jnp.all(fingerprint >= 0)  # All elements should be non-negative

    def test_batch_fingerprints(self):
        """Test fingerprint calculation for batch of molecules."""
        # Create test batch
        batch = create_test_batch(batch_size=3, max_atoms=12)

        # Calculate batch fingerprints
        fingerprints = batch_fingerprints(batch, radius=2)

        # Check batch fingerprint properties
        assert isinstance(fingerprints, jnp.ndarray)
        assert fingerprints.shape == (3, 1024)  # batch_size x fingerprint_size
        assert jnp.all(fingerprints >= 0)

        # Check that different molecules have different fingerprints
        # (with high probability for random synthetic molecules)
        fp1, fp2, fp3 = fingerprints[0], fingerprints[1], fingerprints[2]
        assert not jnp.array_equal(fp1, fp2)
        assert not jnp.array_equal(fp2, fp3)

    def test_fingerprint_radius_effect(self):
        """Test effect of different radius values on fingerprints."""
        processor = JAXMolecularProcessor(max_atoms=15, max_bonds=20)

        # Create test molecule
        mol_data = create_synthetic_molecule(num_atoms=10, num_bonds=9)
        jax_mol = processor.molecular_data_to_jax(mol_data)

        # Calculate fingerprints with different radii
        fp_r1 = calculate_molecular_fingerprint(jax_mol, radius=1)
        fp_r2 = calculate_molecular_fingerprint(jax_mol, radius=2)
        fp_r3 = calculate_molecular_fingerprint(jax_mol, radius=3)

        # Fingerprints should be different for different radii
        assert not jnp.array_equal(fp_r1, fp_r2)
        assert not jnp.array_equal(fp_r2, fp_r3)

        # All should have the same shape
        assert fp_r1.shape == fp_r2.shape == fp_r3.shape


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXCompilation:
    """Test JAX JIT compilation functionality."""

    def test_jit_compilation(self):
        """Test that JIT compilation works correctly."""
        calculator = JAXDescriptorCalculator()

        # Check that compiled functions exist
        assert len(calculator._compiled_functions) > 0

        # Test that compiled functions can be called
        processor = JAXMolecularProcessor(max_atoms=10, max_bonds=15)
        mol_data = create_synthetic_molecule(num_atoms=5, num_bonds=4)
        jax_mol = processor.molecular_data_to_jax(mol_data)

        # This should work without errors if compilation succeeded
        if "molecular_weight" in calculator._compiled_functions:
            mw = calculator._compiled_functions["molecular_weight"](jax_mol)
            assert isinstance(mw, (float, jnp.ndarray))
            assert not np.isnan(float(mw))

    def test_compilation_fallback(self):
        """Test fallback when compilation fails."""
        # This test ensures that even if compilation fails,
        # the calculator can still function
        calculator = JAXDescriptorCalculator()

        # Clear compiled functions to test fallback
        original_functions = calculator._compiled_functions.copy()
        calculator._compiled_functions.clear()

        processor = JAXMolecularProcessor(max_atoms=10, max_bonds=15)
        mol_data = create_synthetic_molecule(num_atoms=5, num_bonds=4)
        jax_mol = processor.molecular_data_to_jax(mol_data)

        # Should still work with non-compiled functions
        results = calculator.calculate_single_molecule(jax_mol)
        assert "molecular_weight" in results
        assert isinstance(results["molecular_weight"], (int, float))

        # Restore compiled functions
        calculator._compiled_functions = original_functions


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestDescriptorValidation:
    """Test descriptor calculation validation and edge cases."""

    def test_small_molecule_descriptors(self):
        """Test descriptors for very small molecules."""
        calculator = JAXDescriptorCalculator()
        processor = JAXMolecularProcessor(max_atoms=10, max_bonds=15)

        # Create minimal molecule (single atom)
        mol_data = create_synthetic_molecule(num_atoms=1, num_bonds=0)
        jax_mol = processor.molecular_data_to_jax(mol_data)

        results = calculator.calculate_single_molecule(jax_mol)

        assert results["atom_count"] == 1
        assert results["bond_count"] == 0
        assert results["molecular_weight"] > 0

    def test_descriptor_ranges(self):
        """Test that descriptors are within reasonable ranges."""
        calculator = JAXDescriptorCalculator()
        processor = JAXMolecularProcessor(max_atoms=20, max_bonds=30)

        # Create test molecule
        mol_data = create_synthetic_molecule(num_atoms=10, num_bonds=9)
        jax_mol = processor.molecular_data_to_jax(mol_data)

        results = calculator.calculate_single_molecule(jax_mol)

        # Check reasonable ranges
        assert 0 < results["molecular_weight"] < 10000  # Reasonable MW range
        assert results["atom_count"] == 10
        assert results["bond_count"] == 9
        assert -10 <= results["formal_charge"] <= 10  # Reasonable charge range
        assert 0 <= results["aromatic_atom_count"] <= 10
        assert 0 <= results["ring_atom_count"] <= 10


# Test without JAX available
@pytest.mark.skipif(JAX_AVAILABLE, reason="JAX is available")
class TestJAXDescriptorsUnavailable:
    """Test behavior when JAX is not available."""

    def test_jax_unavailable_error(self):
        """Test that appropriate errors are raised when JAX is unavailable."""
        with pytest.raises(BackendUnavailableError):
            JAXDescriptorCalculator()


if __name__ == "__main__":
    pytest.main([__file__])

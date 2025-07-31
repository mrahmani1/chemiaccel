"""
Unit tests for JAX batch processing utilities.

This module tests the JAX backend batch processing functionality
including automatic shape handling and memory management.
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

from chemi_accel.backends.jax.batch_processing import (
    JAXBatchProcessor,
    JAXShapeHandler,
    create_test_batch,
)
from chemi_accel.backends.jax.molecular import create_synthetic_molecule
from chemi_accel.core.exceptions import BackendUnavailableError, ValidationError


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXBatchProcessor:
    """Test JAX batch processor functionality."""

    def test_batch_processor_initialization(self):
        """Test batch processor initialization."""
        processor = JAXBatchProcessor(
            max_atoms=50, max_bonds=100, default_batch_size=16, memory_limit_mb=1000
        )

        assert processor.max_atoms == 50
        assert processor.max_bonds == 100
        assert processor.default_batch_size == 16
        assert processor.memory_limit_mb == 1000
        assert processor.processed_batches == 0
        assert processor.total_molecules == 0

    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        processor = JAXBatchProcessor(max_atoms=20, max_bonds=30)

        # Test memory estimation for different batch sizes
        memory_small = processor.estimate_memory_usage(batch_size=4)
        memory_large = processor.estimate_memory_usage(batch_size=16)

        assert memory_small > 0
        assert memory_large > memory_small
        assert isinstance(memory_small, float)
        assert isinstance(memory_large, float)

    def test_batch_size_optimization(self):
        """Test batch size optimization based on molecule characteristics."""
        processor = JAXBatchProcessor(max_atoms=50, max_bonds=100)

        # Create molecules of different sizes
        small_molecules = [create_synthetic_molecule(num_atoms=5) for _ in range(20)]
        medium_molecules = [create_synthetic_molecule(num_atoms=25) for _ in range(20)]
        large_molecules = [create_synthetic_molecule(num_atoms=45) for _ in range(20)]

        # Test optimization
        small_batch_size = processor.optimize_batch_size(small_molecules)
        medium_batch_size = processor.optimize_batch_size(medium_molecules)
        large_batch_size = processor.optimize_batch_size(large_molecules)

        # Larger molecules should have smaller batch sizes
        assert large_batch_size <= medium_batch_size
        assert medium_batch_size <= small_batch_size

        # All should be positive and reasonable
        assert 1 <= small_batch_size <= 64
        assert 1 <= medium_batch_size <= 32
        assert 1 <= large_batch_size <= 16

    def test_batch_creation(self):
        """Test creation of batches from molecule list."""
        processor = JAXBatchProcessor(max_atoms=20, max_bonds=30)

        # Create test molecules
        molecules = [
            create_synthetic_molecule(num_atoms=5 + i, mol_id=f"mol_{i}")
            for i in range(8)
        ]

        # Create batches with specific batch size
        batches = processor.create_batches(molecules, batch_size=3)

        # Should create 3 batches: [3, 3, 2]
        assert len(batches) == 3
        assert batches[0].batch_size == 3
        assert batches[1].batch_size == 3
        assert batches[2].batch_size == 2

        # Check that all molecules are included
        total_molecules = sum(batch.batch_size for batch in batches)
        assert total_molecules == 8

    def test_batch_iterator(self):
        """Test batch processing iterator."""
        processor = JAXBatchProcessor(max_atoms=15, max_bonds=25)

        # Create test molecules
        molecules = [
            create_synthetic_molecule(num_atoms=6, mol_id=f"mol_{i}") for i in range(10)
        ]

        # Test iterator
        batches_processed = 0
        total_molecules_processed = 0

        for batch in processor.process_batches_iterator(molecules, batch_size=4):
            batches_processed += 1
            total_molecules_processed += batch.batch_size

            # Validate batch
            assert batch.batch_size <= 4
            assert batch.max_atoms == 15
            assert batch.max_bonds == 25

        # Should process all molecules
        assert total_molecules_processed == 10
        assert batches_processed == 3  # [4, 4, 2]

        # Check processor statistics
        assert processor.processed_batches == 3
        assert processor.total_molecules == 10

    def test_adaptive_batch_processing(self):
        """Test adaptive batch processing with automatic adjustment."""
        processor = JAXBatchProcessor(max_atoms=20, max_bonds=30)

        # Create test molecules
        molecules = [
            create_synthetic_molecule(num_atoms=8, mol_id=f"mol_{i}") for i in range(12)
        ]

        # Define a simple processing function
        def simple_process_function(batch):
            return {
                "batch_size": batch.batch_size,
                "total_atoms": jnp.sum(batch.molecule_sizes),
                "processed": True,
            }

        # Run adaptive processing
        results = processor.adaptive_batch_processing(
            molecules, simple_process_function, initial_batch_size=5
        )

        # Check results
        assert len(results) > 0
        total_processed = sum(result["batch_size"] for result in results)
        assert total_processed == 12

        # All results should be successful
        for result in results:
            assert result["processed"] is True
            assert result["batch_size"] > 0

    def test_processing_statistics(self):
        """Test processing statistics tracking."""
        processor = JAXBatchProcessor()

        # Initial stats
        stats = processor.get_processing_stats()
        assert stats["processed_batches"] == 0
        assert stats["total_molecules"] == 0
        assert stats["avg_molecules_per_batch"] == 0

        # Process some batches
        molecules = [create_synthetic_molecule(num_atoms=5) for _ in range(6)]

        batch_count = 0
        for batch in processor.process_batches_iterator(molecules, batch_size=2):
            batch_count += 1

        # Check updated stats
        stats = processor.get_processing_stats()
        assert stats["processed_batches"] == 3
        assert stats["total_molecules"] == 6
        assert stats["avg_molecules_per_batch"] == 2.0

        # Reset stats
        processor.reset_stats()
        stats = processor.get_processing_stats()
        assert stats["processed_batches"] == 0
        assert stats["total_molecules"] == 0

    def test_empty_molecule_list(self):
        """Test handling of empty molecule list."""
        processor = JAXBatchProcessor()

        # Empty list should return empty batches
        batches = processor.create_batches([])
        assert len(batches) == 0

        # Iterator should not yield anything
        batch_count = 0
        for batch in processor.process_batches_iterator([]):
            batch_count += 1
        assert batch_count == 0

    def test_memory_constrained_optimization(self):
        """Test batch size optimization with memory constraints."""
        processor = JAXBatchProcessor(
            max_atoms=100, max_bonds=200, memory_limit_mb=50  # Small memory limit
        )

        # Create large molecules
        molecules = [create_synthetic_molecule(num_atoms=80) for _ in range(10)]

        # Optimization should respect memory limit
        batch_size = processor.optimize_batch_size(molecules)
        estimated_memory = processor.estimate_memory_usage(batch_size)

        assert batch_size >= 1
        assert estimated_memory <= processor.memory_limit_mb


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXShapeHandler:
    """Test JAX shape handling utilities."""

    def test_pad_to_shape(self):
        """Test array padding to target shape."""
        # Create test array
        original_array = np.random.rand(3, 4)
        target_shape = (5, 6)

        # Pad array
        padded = JAXShapeHandler.pad_to_shape(
            original_array, target_shape, pad_value=0.0
        )

        # Check result
        assert padded.shape == target_shape
        assert jnp.allclose(padded[:3, :4], original_array)
        assert jnp.all(padded[3:, :] == 0.0)
        assert jnp.all(padded[:, 4:] == 0.0)

    def test_pad_to_shape_validation(self):
        """Test padding validation."""
        original_array = np.random.rand(3, 4)

        # Test dimension mismatch
        with pytest.raises(ValidationError):
            JAXShapeHandler.pad_to_shape(
                original_array, (5,)
            )  # Wrong number of dimensions

        # Test size exceeding target
        with pytest.raises(ValidationError):
            JAXShapeHandler.pad_to_shape(
                original_array, (2, 6)
            )  # First dimension too small

    def test_create_mask(self):
        """Test mask creation."""
        mask = JAXShapeHandler.create_mask(actual_size=5, max_size=10)

        assert mask.shape == (10,)
        assert jnp.sum(mask) == 5
        assert jnp.all(mask[:5] == True)
        assert jnp.all(mask[5:] == False)

    def test_create_mask_validation(self):
        """Test mask creation validation."""
        with pytest.raises(ValidationError):
            JAXShapeHandler.create_mask(actual_size=15, max_size=10)  # Actual > max

    def test_apply_mask(self):
        """Test mask application."""
        # Create test array and mask
        array = jnp.ones((5, 3))
        mask = jnp.array([True, True, False, True, False])

        # Apply mask
        masked_array = JAXShapeHandler.apply_mask(array, mask, axis=0)

        # Check result
        assert masked_array.shape == array.shape
        assert jnp.allclose(masked_array[0], array[0])  # True in mask
        assert jnp.allclose(masked_array[1], array[1])  # True in mask
        assert jnp.allclose(masked_array[2], jnp.zeros(3))  # False in mask
        assert jnp.allclose(masked_array[3], array[3])  # True in mask
        assert jnp.allclose(masked_array[4], jnp.zeros(3))  # False in mask

    def test_unpad_array(self):
        """Test array unpadding."""
        # Create padded array
        padded_array = jnp.array(
            [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        )

        # Unpad
        unpadded = JAXShapeHandler.unpad_array(padded_array, actual_size=2, axis=0)

        # Check result
        assert unpadded.shape == (2, 5)
        assert jnp.array_equal(unpadded, padded_array[:2])

        # Test unpadding along different axis
        unpadded_axis1 = JAXShapeHandler.unpad_array(
            padded_array, actual_size=3, axis=1
        )
        assert unpadded_axis1.shape == (4, 3)
        assert jnp.array_equal(unpadded_axis1, padded_array[:, :3])


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestCreateTestBatch:
    """Test test batch creation utility."""

    def test_create_test_batch(self):
        """Test creation of test batch."""
        batch = create_test_batch(batch_size=5, max_atoms=15)

        assert batch.batch_size == 5
        assert batch.max_atoms == 15
        assert batch.atom_features.shape[0] == 5  # batch dimension
        assert batch.atom_features.shape[1] == 15  # max_atoms dimension

        # Check that molecules have different sizes
        molecule_sizes = [batch.get_molecule(i).num_atoms for i in range(5)]
        assert len(set(molecule_sizes)) > 1  # Should have variety in sizes

        # All sizes should be reasonable
        for size in molecule_sizes:
            assert 5 <= size < 15

    def test_test_batch_properties(self):
        """Test properties of created test batch."""
        batch = create_test_batch(batch_size=3, max_atoms=12)

        # Check individual molecules
        for i in range(3):
            mol = batch.get_molecule(i)
            assert mol.mol_id == f"test_mol_{i}"
            assert mol.smiles == f"test_smiles_{i}"
            assert mol.num_atoms > 0
            assert mol.actual_num_atoms == mol.num_atoms

        # Check batch consistency
        assert len(batch.mol_ids) == 3
        assert len(batch.smiles_list) == 3
        assert batch.molecule_sizes.shape == (3,)


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestBatchProcessingIntegration:
    """Test integration of batch processing components."""

    def test_end_to_end_batch_processing(self):
        """Test complete batch processing workflow."""
        processor = JAXBatchProcessor(max_atoms=20, max_bonds=30)

        # Create diverse molecules
        molecules = []
        for i in range(15):
            num_atoms = 5 + (i % 10)  # Varying sizes
            mol = create_synthetic_molecule(
                num_atoms=num_atoms, num_bonds=max(0, num_atoms - 1), mol_id=f"mol_{i}"
            )
            molecules.append(mol)

        # Process with adaptive batching
        def dummy_processor(batch):
            return {
                "batch_id": id(batch),
                "batch_size": batch.batch_size,
                "avg_atoms": float(jnp.mean(batch.molecule_sizes)),
                "success": True,
            }

        results = processor.adaptive_batch_processing(molecules, dummy_processor)

        # Validate results
        assert len(results) > 0
        total_processed = sum(r["batch_size"] for r in results)
        assert total_processed == 15

        # All should be successful
        for result in results:
            assert result["success"] is True
            assert result["batch_size"] > 0
            assert result["avg_atoms"] > 0

    def test_memory_aware_processing(self):
        """Test memory-aware batch processing."""
        # Create processor with tight memory constraints
        processor = JAXBatchProcessor(
            max_atoms=30, max_bonds=50, memory_limit_mb=10  # Very tight limit
        )

        # Create molecules that would exceed memory if batched too large
        molecules = [create_synthetic_molecule(num_atoms=25) for _ in range(8)]

        # Should automatically use small batch sizes
        batches = processor.create_batches(molecules)

        # Verify that batches respect memory constraints
        for batch in batches:
            estimated_memory = processor.estimate_memory_usage(batch.batch_size)
            assert (
                estimated_memory <= processor.memory_limit_mb * 1.1
            )  # Allow small tolerance


# Test without JAX available
@pytest.mark.skipif(JAX_AVAILABLE, reason="JAX is available")
class TestJAXBatchProcessingUnavailable:
    """Test behavior when JAX is not available."""

    def test_jax_unavailable_error(self):
        """Test that appropriate errors are raised when JAX is unavailable."""
        with pytest.raises(BackendUnavailableError):
            JAXBatchProcessor()

        with pytest.raises(BackendUnavailableError):
            JAXShapeHandler.pad_to_shape(np.array([1, 2, 3]), (5,))


if __name__ == "__main__":
    pytest.main([__file__])

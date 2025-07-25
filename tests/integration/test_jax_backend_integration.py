"""
Integration tests for JAX backend functionality.

This module tests the complete JAX backend workflow including
molecular processing, descriptor calculations, and vectorization.
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

from chemi_accel.backends.jax import (
    JAXBatchProcessor,
    JAXDescriptorCalculator,
    JAXMemoryEfficientBatcher,
    JAXMolecularProcessor,
    JAXVectorizedCalculator,
)
from chemi_accel.backends.jax.molecular import create_synthetic_molecule


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
@pytest.mark.integration
class TestJAXBackendIntegration:
    """Test complete JAX backend integration."""

    def test_end_to_end_molecular_processing(self):
        """Test complete molecular processing workflow."""
        # Initialize components
        processor = JAXMolecularProcessor(max_atoms=20, max_bonds=30)
        descriptor_calc = JAXDescriptorCalculator()
        vectorized_calc = JAXVectorizedCalculator()

        # Create test molecules
        molecules = [
            create_synthetic_molecule(num_atoms=8, num_bonds=7, mol_id=f"mol_{i}")
            for i in range(6)
        ]

        # Step 1: Convert to JAX format
        jax_molecules = [processor.molecular_data_to_jax(mol) for mol in molecules]

        # Step 2: Calculate descriptors for individual molecules
        individual_results = []
        for jax_mol in jax_molecules:
            result = descriptor_calc.calculate_single_molecule(jax_mol)
            individual_results.append(result)

        # Step 3: Create batch and calculate descriptors
        batch = processor.create_batch(molecules)
        batch_results = vectorized_calc.calculate_batch_vectorized(batch)

        # Step 4: Verify consistency
        for i, individual_result in enumerate(individual_results):
            for descriptor in ["molecular_weight", "atom_count", "bond_count"]:
                if descriptor in individual_result and descriptor in batch_results:
                    individual_value = individual_result[descriptor]
                    batch_value = batch_results[descriptor][i]

                    if isinstance(individual_value, float):
                        assert abs(individual_value - batch_value) < 1e-4
                    else:
                        assert individual_value == batch_value

        # Step 5: Verify all molecules were processed
        assert len(batch_results["molecular_weight"]) == 6
        assert all(mw > 0 for mw in batch_results["molecular_weight"])

    def test_memory_efficient_large_batch_processing(self):
        """Test memory-efficient processing of large batches."""
        # Initialize memory-efficient batcher
        batcher = JAXMemoryEfficientBatcher(memory_limit_mb=50)

        # Create large dataset
        molecules = [
            create_synthetic_molecule(
                num_atoms=np.random.randint(5, 15),
                num_bonds=np.random.randint(4, 14),
                mol_id=f"large_mol_{i}",
            )
            for i in range(20)
        ]

        # Create batch
        processor = JAXMolecularProcessor(max_atoms=20, max_bonds=25)
        large_batch = processor.create_batch(molecules)

        # Process with memory constraints
        results = batcher.chunked_batch_processing(large_batch)

        # Verify results
        assert len(results["molecular_weight"]) == 20
        assert all(mw > 0 for mw in results["molecular_weight"])
        assert all(ac > 0 for ac in results["atom_count"])

        # Verify molecular IDs are preserved
        if "mol_ids" in results:
            assert len(results["mol_ids"]) == 20

    def test_batch_size_optimization(self):
        """Test automatic batch size optimization."""
        batch_processor = JAXBatchProcessor(max_atoms=25, max_bonds=40)

        # Create molecules of different sizes
        small_molecules = [create_synthetic_molecule(num_atoms=5) for _ in range(10)]
        large_molecules = [create_synthetic_molecule(num_atoms=20) for _ in range(10)]

        # Test optimization
        small_batch_size = batch_processor.optimize_batch_size(small_molecules)
        large_batch_size = batch_processor.optimize_batch_size(large_molecules)

        # Large molecules should have smaller batch sizes
        assert large_batch_size <= small_batch_size

        # Create batches with optimized sizes
        small_batches = batch_processor.create_batches(small_molecules)
        large_batches = batch_processor.create_batches(large_molecules)

        # Verify batches were created
        assert len(small_batches) > 0
        assert len(large_batches) > 0

        # Verify total molecules are preserved
        total_small = sum(batch.batch_size for batch in small_batches)
        total_large = sum(batch.batch_size for batch in large_batches)

        assert total_small == 10
        assert total_large == 10

    def test_descriptor_calculation_consistency(self):
        """Test consistency across different calculation methods."""
        # Initialize calculators
        standard_calc = JAXDescriptorCalculator()
        vectorized_calc = JAXVectorizedCalculator()
        processor = JAXMolecularProcessor(max_atoms=15, max_bonds=20)

        # Create test molecules
        molecules = [
            create_synthetic_molecule(
                num_atoms=8, num_bonds=7, mol_id=f"consistency_mol_{i}"
            )
            for i in range(4)
        ]

        # Method 1: Standard single molecule calculations
        standard_results = []
        for mol in molecules:
            jax_mol = processor.molecular_data_to_jax(mol)
            result = standard_calc.calculate_single_molecule(jax_mol)
            standard_results.append(result)

        # Method 2: Vectorized single molecule calculations
        vectorized_single_results = []
        for mol in molecules:
            jax_mol = processor.molecular_data_to_jax(mol)
            result = vectorized_calc.calculate_single_vectorized(jax_mol)
            vectorized_single_results.append(result)

        # Method 3: Batch vectorized calculations
        batch = processor.create_batch(molecules)
        batch_results = vectorized_calc.calculate_batch_vectorized(batch)

        # Compare results across methods
        descriptors_to_check = ["molecular_weight", "atom_count", "bond_count"]

        for i in range(len(molecules)):
            for descriptor in descriptors_to_check:
                if descriptor in standard_results[i]:
                    standard_val = standard_results[i][descriptor]
                    vectorized_val = vectorized_single_results[i][descriptor]
                    batch_val = batch_results[descriptor][i]

                    # All methods should give consistent results
                    if isinstance(standard_val, float):
                        assert abs(standard_val - vectorized_val) < 1e-4
                        assert abs(standard_val - batch_val) < 1e-4
                    else:
                        assert standard_val == vectorized_val
                        assert standard_val == batch_val

    def test_error_handling_and_recovery(self):
        """Test error handling and graceful recovery."""
        processor = JAXMolecularProcessor(max_atoms=10, max_bonds=15)
        vectorized_calc = JAXVectorizedCalculator()

        # Create mix of valid and problematic molecules
        molecules = []

        # Add valid molecules
        for i in range(3):
            mol = create_synthetic_molecule(
                num_atoms=6, num_bonds=5, mol_id=f"valid_mol_{i}"
            )
            molecules.append(mol)

        # Add molecule that might cause issues (very small)
        tiny_mol = create_synthetic_molecule(
            num_atoms=1, num_bonds=0, mol_id="tiny_mol"
        )
        molecules.append(tiny_mol)

        # Process batch - should handle edge cases gracefully
        try:
            batch = processor.create_batch(molecules)
            results = vectorized_calc.calculate_batch_vectorized(batch)

            # Should get results for all molecules
            assert len(results["molecular_weight"]) == 4
            assert all(mw > 0 for mw in results["molecular_weight"])

        except Exception as e:
            pytest.fail(f"Batch processing failed with error: {e}")

    def test_performance_scaling(self):
        """Test that performance scales reasonably with batch size."""
        import time

        vectorized_calc = JAXVectorizedCalculator()
        processor = JAXMolecularProcessor(max_atoms=15, max_bonds=20)

        # Test different batch sizes
        batch_sizes = [4, 8, 16]
        processing_times = []

        for batch_size in batch_sizes:
            # Create molecules
            molecules = [
                create_synthetic_molecule(
                    num_atoms=8, num_bonds=7, mol_id=f"perf_mol_{i}"
                )
                for i in range(batch_size)
            ]

            # Create batch
            batch = processor.create_batch(molecules)

            # Warmup
            vectorized_calc.calculate_batch_vectorized(batch)

            # Measure processing time
            start_time = time.time()
            results = vectorized_calc.calculate_batch_vectorized(batch)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Verify results
            assert len(results["molecular_weight"]) == batch_size

        # Performance should not degrade dramatically with batch size
        # (This is a loose check since performance can vary in test environments)
        for i in range(1, len(processing_times)):
            time_ratio = processing_times[i] / processing_times[i - 1]
            batch_ratio = batch_sizes[i] / batch_sizes[i - 1]

            # Time increase should not be much worse than linear
            assert time_ratio <= batch_ratio * 2.0

    def test_molecular_data_preservation(self):
        """Test that molecular metadata is preserved through processing."""
        processor = JAXMolecularProcessor(max_atoms=12, max_bonds=18)
        vectorized_calc = JAXVectorizedCalculator()

        # Create molecules with specific metadata
        molecules = []
        for i in range(5):
            mol = create_synthetic_molecule(
                num_atoms=6 + i,
                num_bonds=5 + i,
                mol_id=f"metadata_mol_{i}",
                smiles=f"test_smiles_{i}",
            )
            molecules.append(mol)

        # Process batch
        batch = processor.create_batch(molecules)
        results = vectorized_calc.calculate_batch_vectorized(batch)

        # Check that metadata is preserved
        if "mol_ids" in results:
            assert len(results["mol_ids"]) == 5
            for i, mol_id in enumerate(results["mol_ids"]):
                assert mol_id == f"metadata_mol_{i}"

        if "smiles_list" in results:
            assert len(results["smiles_list"]) == 5
            for i, smiles in enumerate(results["smiles_list"]):
                assert smiles == f"test_smiles_{i}"

        # Check that molecular sizes are correct
        expected_atom_counts = [6, 7, 8, 9, 10]
        actual_atom_counts = results["atom_count"].tolist()
        assert actual_atom_counts == expected_atom_counts


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
@pytest.mark.integration
class TestJAXBackendRealWorldScenarios:
    """Test JAX backend with realistic scenarios."""

    def test_drug_discovery_workflow(self):
        """Test workflow similar to drug discovery pipeline."""
        # Initialize components for drug discovery
        processor = JAXMolecularProcessor(max_atoms=50, max_bonds=80)
        vectorized_calc = JAXVectorizedCalculator()
        memory_batcher = JAXMemoryEfficientBatcher(memory_limit_mb=100)

        # Create diverse molecular library (simulating drug compounds)
        compound_library = []
        for i in range(25):
            # Vary molecular sizes to simulate diverse compounds
            num_atoms = np.random.randint(10, 30)
            num_bonds = max(num_atoms - 5, num_atoms - 1)

            compound = create_synthetic_molecule(
                num_atoms=num_atoms,
                num_bonds=num_bonds,
                mol_id=f"compound_{i:03d}",
                smiles=f"drug_like_smiles_{i}",
            )
            compound_library.append(compound)

        # Process library with memory constraints (simulating large datasets)
        batch = processor.create_batch(compound_library)
        results = memory_batcher.chunked_batch_processing(batch)

        # Verify all compounds were processed
        assert len(results["molecular_weight"]) == 25

        # Check for reasonable drug-like properties
        molecular_weights = results["molecular_weight"]
        atom_counts = results["atom_count"]

        # Basic drug-likeness checks
        assert all(100 < mw < 1000 for mw in molecular_weights)  # Reasonable MW range
        assert all(5 < ac < 50 for ac in atom_counts)  # Reasonable atom count range

        # Check that processing was efficient
        assert results is not None

    def test_materials_science_workflow(self):
        """Test workflow for materials science applications."""
        # Initialize for larger molecules (materials/polymers)
        processor = JAXMolecularProcessor(max_atoms=100, max_bonds=150)
        batch_processor = JAXBatchProcessor(max_atoms=100, max_bonds=150)

        # Create materials-like molecules (larger, more complex)
        materials = []
        for i in range(10):
            # Larger molecules for materials
            num_atoms = np.random.randint(30, 80)
            num_bonds = max(num_atoms - 10, num_atoms - 1)

            material = create_synthetic_molecule(
                num_atoms=num_atoms,
                num_bonds=num_bonds,
                mol_id=f"material_{i:02d}",
                smiles=f"polymer_unit_{i}",
            )
            materials.append(material)

        # Process with optimized batching for large molecules
        batches = batch_processor.create_batches(materials)

        # Should create smaller batches due to large molecule size
        assert len(batches) > 1  # Should split into multiple batches

        # Verify all materials are accounted for
        total_processed = sum(batch.batch_size for batch in batches)
        assert total_processed == 10

        # Process each batch
        all_results = []
        vectorized_calc = JAXVectorizedCalculator()

        for batch in batches:
            batch_results = vectorized_calc.calculate_batch_vectorized(batch)
            all_results.append(batch_results)

        # Combine results
        combined_molecular_weights = []
        for result in all_results:
            combined_molecular_weights.extend(result["molecular_weight"])

        assert len(combined_molecular_weights) == 10
        assert all(mw > 0 for mw in combined_molecular_weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

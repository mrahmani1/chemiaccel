"""
Unit tests for JAX molecular processing functions.

This module tests the JAX backend molecular representations and
processing utilities with synthetic data.
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

from chemi_accel.backends.jax.molecular import (
    JAXMolecularBatch,
    JAXMolecularProcessor,
    JAXMolecule,
    create_synthetic_molecule,
)
from chemi_accel.core.exceptions import BackendUnavailableError, ValidationError
from chemi_accel.core.molecular import MolecularData


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXMolecule:
    """Test JAX molecular representation."""

    def test_jax_molecule_creation(self):
        """Test creating JAX molecule with valid data."""
        max_atoms, max_bonds = 10, 15
        num_atoms, num_bonds = 5, 4

        # Create test arrays
        atom_features = jnp.zeros((max_atoms, 6))
        bond_features = jnp.zeros((max_bonds, 4))
        adjacency_matrix = jnp.zeros((max_atoms, max_atoms))
        atom_mask = jnp.zeros(max_atoms, dtype=bool).at[:num_atoms].set(True)
        bond_mask = jnp.zeros(max_bonds, dtype=bool).at[:num_bonds].set(True)

        molecule = JAXMolecule(
            atom_features=atom_features,
            bond_features=bond_features,
            adjacency_matrix=adjacency_matrix,
            atom_mask=atom_mask,
            bond_mask=bond_mask,
            num_atoms=num_atoms,
            num_bonds=num_bonds,
            mol_id="test_mol",
            smiles="CCO",
        )

        assert molecule.num_atoms == num_atoms
        assert molecule.num_bonds == num_bonds
        assert molecule.mol_id == "test_mol"
        assert molecule.smiles == "CCO"
        assert molecule.actual_num_atoms == num_atoms
        assert molecule.actual_num_bonds == num_bonds

    def test_jax_molecule_validation(self):
        """Test JAX molecule validation."""
        max_atoms = 10

        # Test shape mismatch for adjacency matrix
        with pytest.raises(ValidationError):
            JAXMolecule(
                atom_features=jnp.zeros((max_atoms, 6)),
                bond_features=jnp.zeros((15, 4)),
                adjacency_matrix=jnp.zeros((5, 5)),  # Wrong shape
                atom_mask=jnp.zeros(max_atoms, dtype=bool),
                bond_mask=jnp.zeros(15, dtype=bool),
                num_atoms=5,
                num_bonds=4,
            )

        # Test shape mismatch for atom mask
        with pytest.raises(ValidationError):
            JAXMolecule(
                atom_features=jnp.zeros((max_atoms, 6)),
                bond_features=jnp.zeros((15, 4)),
                adjacency_matrix=jnp.zeros((max_atoms, max_atoms)),
                atom_mask=jnp.zeros(5, dtype=bool),  # Wrong shape
                bond_mask=jnp.zeros(15, dtype=bool),
                num_atoms=5,
                num_bonds=4,
            )

    def test_valid_features_extraction(self):
        """Test extraction of valid atom/bond features."""
        max_atoms, max_bonds = 10, 15
        num_atoms, num_bonds = 5, 4

        # Create test data with some non-zero values
        atom_features = jnp.zeros((max_atoms, 6))
        atom_features = atom_features.at[:num_atoms, 0].set(
            jnp.array([6, 7, 8, 6, 1])
        )  # Atomic numbers

        bond_features = jnp.zeros((max_bonds, 4))
        bond_features = bond_features.at[:num_bonds, 0].set(
            jnp.array([1, 1, 2, 1])
        )  # Bond types

        adjacency_matrix = jnp.zeros((max_atoms, max_atoms))
        atom_mask = jnp.zeros(max_atoms, dtype=bool).at[:num_atoms].set(True)
        bond_mask = jnp.zeros(max_bonds, dtype=bool).at[:num_bonds].set(True)

        molecule = JAXMolecule(
            atom_features=atom_features,
            bond_features=bond_features,
            adjacency_matrix=adjacency_matrix,
            atom_mask=atom_mask,
            bond_mask=bond_mask,
            num_atoms=num_atoms,
            num_bonds=num_bonds,
        )

        # Test valid atom features
        valid_atoms = molecule.get_valid_atom_features()
        assert valid_atoms.shape == (num_atoms, 6)
        assert jnp.array_equal(valid_atoms[:, 0], jnp.array([6, 7, 8, 6, 1]))

        # Test valid bond features
        valid_bonds = molecule.get_valid_bond_features()
        assert valid_bonds.shape == (num_bonds, 4)
        assert jnp.array_equal(valid_bonds[:, 0], jnp.array([1, 1, 2, 1]))


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXMolecularProcessor:
    """Test JAX molecular processor."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = JAXMolecularProcessor(max_atoms=50, max_bonds=100)

        assert processor.max_atoms == 50
        assert processor.max_bonds == 100
        assert processor.num_atom_features == 6
        assert processor.num_bond_features == 4

    def test_create_padded_arrays(self):
        """Test creation of padded arrays."""
        processor = JAXMolecularProcessor(max_atoms=10, max_bonds=15)

        # Create test molecular data
        num_atoms, num_bonds = 5, 4
        atom_features = np.random.rand(num_atoms, 6)
        bond_features = np.random.rand(num_bonds, 4)
        adjacency_matrix = np.random.rand(num_atoms, num_atoms)

        (
            padded_atoms,
            padded_bonds,
            padded_adjacency,
            atom_mask,
            bond_mask,
        ) = processor.create_padded_arrays(
            atom_features, bond_features, adjacency_matrix
        )

        # Check shapes
        assert padded_atoms.shape == (10, 6)
        assert padded_bonds.shape == (15, 4)
        assert padded_adjacency.shape == (10, 10)
        assert atom_mask.shape == (10,)
        assert bond_mask.shape == (15,)

        # Check masks
        assert jnp.sum(atom_mask) == num_atoms
        assert jnp.sum(bond_mask) == num_bonds

        # Check that original data is preserved
        assert jnp.allclose(padded_atoms[:num_atoms], atom_features)
        assert jnp.allclose(padded_bonds[:num_bonds], bond_features)
        assert jnp.allclose(padded_adjacency[:num_atoms, :num_atoms], adjacency_matrix)

    def test_molecular_data_conversion(self):
        """Test conversion from MolecularData to JAX format."""
        processor = JAXMolecularProcessor(max_atoms=20, max_bonds=30)

        # Create synthetic molecular data
        mol_data = create_synthetic_molecule(
            num_atoms=8, num_bonds=7, mol_id="test_mol"
        )

        # Convert to JAX format
        jax_mol = processor.molecular_data_to_jax(mol_data)

        assert isinstance(jax_mol, JAXMolecule)
        assert jax_mol.num_atoms == 8
        assert jax_mol.num_bonds == 7
        assert jax_mol.mol_id == "test_mol"
        assert jax_mol.atom_features.shape == (20, 6)
        assert jax_mol.bond_features.shape == (30, 4)
        assert jax_mol.actual_num_atoms == 8
        assert jax_mol.actual_num_bonds == 7

    def test_batch_creation(self):
        """Test creation of molecular batches."""
        processor = JAXMolecularProcessor(max_atoms=15, max_bonds=25)

        # Create multiple synthetic molecules
        molecules = []
        for i in range(4):
            mol = create_synthetic_molecule(
                num_atoms=5 + i, num_bonds=4 + i, mol_id=f"mol_{i}"
            )
            molecules.append(mol)

        # Create batch
        batch = processor.create_batch(molecules)

        assert isinstance(batch, JAXMolecularBatch)
        assert batch.batch_size == 4
        assert batch.max_atoms == 15
        assert batch.max_bonds == 25
        assert batch.atom_features.shape == (4, 15, 6)
        assert batch.bond_features.shape == (4, 25, 4)
        assert batch.adjacency_matrices.shape == (4, 15, 15)
        assert len(batch.mol_ids) == 4

        # Test individual molecule extraction
        mol_0 = batch.get_molecule(0)
        assert mol_0.mol_id == "mol_0"
        assert mol_0.num_atoms == 5

    def test_optimize_batch_size(self):
        """Test batch size optimization."""
        processor = JAXMolecularProcessor()

        # Create molecules with different sizes
        small_molecules = [create_synthetic_molecule(num_atoms=5) for _ in range(10)]
        large_molecules = [create_synthetic_molecule(num_atoms=80) for _ in range(10)]

        # Test optimization for small molecules
        small_batch_size = processor.optimize_batch_size(small_molecules)
        assert small_batch_size >= 10  # Should allow larger batches

        # Test optimization for large molecules
        large_batch_size = processor.optimize_batch_size(large_molecules)
        assert large_batch_size <= small_batch_size  # Should use smaller batches

    def test_empty_molecule_list(self):
        """Test handling of empty molecule list."""
        processor = JAXMolecularProcessor()

        with pytest.raises(ValidationError):
            processor.create_batch([])

    def test_molecule_size_validation(self):
        """Test validation of molecule sizes against limits."""
        processor = JAXMolecularProcessor(max_atoms=5, max_bonds=8)

        # Create molecule that exceeds limits
        large_mol = create_synthetic_molecule(num_atoms=10, num_bonds=15)

        with pytest.raises(ValidationError):
            processor.molecular_data_to_jax(large_mol)


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestSyntheticMoleculeGeneration:
    """Test synthetic molecule generation utilities."""

    def test_create_synthetic_molecule(self):
        """Test creation of synthetic molecules."""
        mol = create_synthetic_molecule(num_atoms=10, num_bonds=9, mol_id="test")

        assert isinstance(mol, MolecularData)
        assert mol.num_atoms == 10
        assert mol.num_bonds == 9
        assert mol.mol_id == "test"
        assert mol.atom_features is not None
        assert mol.bond_features is not None
        assert mol.adjacency_matrix is not None
        assert mol.atom_features.shape == (10, 6)
        assert mol.bond_features.shape == (9, 4)
        assert mol.adjacency_matrix.shape == (10, 10)
        assert mol.is_valid()

    def test_synthetic_molecule_features(self):
        """Test that synthetic molecule features are reasonable."""
        mol = create_synthetic_molecule(num_atoms=5, num_bonds=4)

        # Check atom features
        atomic_numbers = mol.atom_features[:, 0]
        assert np.all(atomic_numbers >= 1)
        assert np.all(atomic_numbers <= 18)

        formal_charges = mol.atom_features[:, 1]
        assert np.all(formal_charges >= -2)
        assert np.all(formal_charges <= 2)

        # Check bond features
        bond_types = mol.bond_features[:, 0]
        assert np.all(bond_types >= 1)
        assert np.all(bond_types <= 3)

        # Check adjacency matrix symmetry
        assert np.allclose(mol.adjacency_matrix, mol.adjacency_matrix.T)

    def test_synthetic_molecule_defaults(self):
        """Test synthetic molecule creation with default parameters."""
        mol = create_synthetic_molecule()

        assert mol.num_atoms == 10
        assert mol.num_bonds == 9
        assert mol.smiles == "synthetic_10atoms"
        assert mol.molecular_weight == 120.0  # 10 * 12.0


@pytest.mark.jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXMolecularBatch:
    """Test JAX molecular batch functionality."""

    def test_batch_indexing(self):
        """Test batch indexing and molecule extraction."""
        processor = JAXMolecularProcessor(max_atoms=15, max_bonds=20)

        # Create test molecules
        molecules = [
            create_synthetic_molecule(num_atoms=5, mol_id=f"mol_{i}") for i in range(3)
        ]

        batch = processor.create_batch(molecules)

        # Test length
        assert len(batch) == 3

        # Test indexing
        mol_1 = batch[1]
        assert mol_1.mol_id == "mol_1"
        assert mol_1.num_atoms == 5

        # Test get_molecule method
        mol_2 = batch.get_molecule(2)
        assert mol_2.mol_id == "mol_2"

        # Test index out of range
        with pytest.raises(IndexError):
            batch.get_molecule(5)

    def test_batch_validation(self):
        """Test batch validation during creation."""
        # This test ensures that batch validation catches dimension mismatches
        # The actual validation is tested through the processor creation process
        processor = JAXMolecularProcessor(max_atoms=10, max_bonds=15)

        molecules = [create_synthetic_molecule(num_atoms=5) for _ in range(2)]
        batch = processor.create_batch(molecules)

        # Batch should be created successfully with valid molecules
        assert batch.batch_size == 2
        assert batch.max_atoms == 10
        assert batch.max_bonds == 15


# Test without JAX available
@pytest.mark.skipif(JAX_AVAILABLE, reason="JAX is available")
class TestJAXUnavailable:
    """Test behavior when JAX is not available."""

    def test_jax_unavailable_error(self):
        """Test that appropriate errors are raised when JAX is unavailable."""
        with pytest.raises(BackendUnavailableError):
            JAXMolecularProcessor()


if __name__ == "__main__":
    pytest.main([__file__])

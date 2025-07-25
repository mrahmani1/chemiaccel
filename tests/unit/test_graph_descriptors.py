"""
Tests for molecular graph representations and graph-based descriptors.

This module tests the MolecularGraphBuilder and GraphDescriptorCalculator
to ensure correct graph construction and descriptor calculations.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from chemi_accel.descriptors.graph import (
    GraphDescriptorCalculator,
    MolecularGraphBuilder,
    build_molecular_graphs,
    calculate_graph_descriptors,
)
from chemi_accel.utils.rdkit_integration import RDKitMolecularParser

# Test molecules for graph analysis
GRAPH_TEST_MOLECULES = [
    ("C", "methane"),  # Single atom
    ("CC", "ethane"),  # Simple bond
    ("CCC", "propane"),  # Linear chain
    ("C1CCC1", "cyclobutane"),  # Simple ring
    ("c1ccccc1", "benzene"),  # Aromatic ring
    ("C1CCCCC1", "cyclohexane"),  # Saturated ring
    ("CC1CCCCC1", "methylcyclohexane"),  # Substituted ring
    ("c1ccc2ccccc2c1", "naphthalene"),  # Fused rings
    ("CCc1ccccc1", "ethylbenzene"),  # Aromatic with substituent
]


@pytest.fixture
def graph_builder():
    """Create MolecularGraphBuilder instance."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    return MolecularGraphBuilder()


@pytest.fixture
def graph_calculator():
    """Create GraphDescriptorCalculator instance."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    return GraphDescriptorCalculator()


@pytest.fixture
def parser():
    """Create RDKitMolecularParser instance."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")
    return RDKitMolecularParser()


@pytest.fixture
def test_molecules(parser):
    """Parse test molecules."""
    molecules = []
    for smiles, name in GRAPH_TEST_MOLECULES:
        try:
            mol_data = parser.parse_smiles(smiles, name)
            molecules.append(mol_data)
        except Exception as e:
            pytest.fail(f"Failed to parse test molecule {name} ({smiles}): {e}")
    return molecules


class TestMolecularGraphBuilder:
    """Test MolecularGraphBuilder functionality."""

    def test_initialization(self, graph_builder):
        """Test graph builder initialization."""
        assert graph_builder is not None
        assert hasattr(graph_builder, "atom_features")
        assert hasattr(graph_builder, "bond_features")
        assert not graph_builder.include_hydrogens  # Default

    def test_initialization_with_hydrogens(self):
        """Test graph builder with hydrogen inclusion."""
        if not RDKIT_AVAILABLE:
            pytest.skip("RDKit not available")

        builder = MolecularGraphBuilder(include_hydrogens=True)
        assert builder.include_hydrogens

    def test_feature_dimensions(self, graph_builder):
        """Test feature dimension calculation."""
        dims = graph_builder.get_feature_dimensions()

        assert "atom_features" in dims
        assert "bond_features" in dims
        assert dims["atom_features"] > 0
        assert dims["bond_features"] > 0

    def test_atom_feature_extraction(self, graph_builder, test_molecules):
        """Test atom feature extraction."""
        mol_data = test_molecules[6]  # methylcyclohexane
        rdkit_mol = mol_data.backend_data["rdkit_mol"]

        for atom in rdkit_mol.GetAtoms():
            features = graph_builder.get_atom_features(atom)

            assert isinstance(features, np.ndarray)
            assert features.dtype == np.float32
            assert len(features) > 0

            # Check that features are reasonable
            assert np.all(np.isfinite(features))
            assert np.all(features >= 0)  # Most features should be non-negative

    def test_bond_feature_extraction(self, graph_builder, test_molecules):
        """Test bond feature extraction."""
        mol_data = test_molecules[6]  # methylcyclohexane
        rdkit_mol = mol_data.backend_data["rdkit_mol"]

        for bond in rdkit_mol.GetBonds():
            features = graph_builder.get_bond_features(bond)

            assert isinstance(features, np.ndarray)
            assert features.dtype == np.float32
            assert len(features) > 0

            # Check that features are reasonable
            assert np.all(np.isfinite(features))

    def test_adjacency_matrix_construction(self, graph_builder, test_molecules):
        """Test adjacency matrix construction."""
        for mol_data in test_molecules:
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            adj_matrix = graph_builder.build_adjacency_matrix(rdkit_mol)

            num_atoms = rdkit_mol.GetNumAtoms()

            # Check matrix properties
            assert adj_matrix.shape == (num_atoms, num_atoms)
            assert adj_matrix.dtype == np.int32

            # Should be symmetric
            assert np.array_equal(adj_matrix, adj_matrix.T)

            # Diagonal should be zero (no self-loops)
            assert np.all(np.diag(adj_matrix) == 0)

            # Values should be 0 or 1
            assert np.all((adj_matrix == 0) | (adj_matrix == 1))

            # Check that number of bonds matches
            num_edges = np.sum(adj_matrix) // 2  # Divide by 2 for undirected graph
            assert num_edges == rdkit_mol.GetNumBonds()

    def test_bond_adjacency_matrix(self, graph_builder, test_molecules):
        """Test bond adjacency matrix construction."""
        for mol_data in test_molecules:
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            bond_adj = graph_builder.build_bond_adjacency_matrix(rdkit_mol)

            num_bonds = rdkit_mol.GetNumBonds()

            if num_bonds == 0:
                assert bond_adj.shape == (0, 0)
                continue

            # Check matrix properties
            assert bond_adj.shape == (num_bonds, num_bonds)
            assert bond_adj.dtype == np.int32

            # Should be symmetric
            assert np.array_equal(bond_adj, bond_adj.T)

            # Diagonal should be zero
            assert np.all(np.diag(bond_adj) == 0)

            # Values should be 0 or 1
            assert np.all((bond_adj == 0) | (bond_adj == 1))

    def test_molecular_graph_construction(self, graph_builder, test_molecules):
        """Test complete molecular graph construction."""
        for mol_data in test_molecules:
            graph_data = graph_builder.build_molecular_graph(mol_data)

            # Check required keys
            required_keys = [
                "num_atoms",
                "num_bonds",
                "atom_features",
                "bond_features",
                "adjacency_matrix",
                "adjacency_sparse",
                "bond_adjacency_matrix",
                "bond_adjacency_sparse",
                "atom_feature_names",
                "bond_feature_names",
            ]

            for key in required_keys:
                assert key in graph_data

            # Check data types and shapes
            assert isinstance(graph_data["num_atoms"], int)
            assert isinstance(graph_data["num_bonds"], int)
            assert isinstance(graph_data["atom_features"], np.ndarray)
            assert isinstance(graph_data["bond_features"], np.ndarray)
            assert isinstance(graph_data["adjacency_matrix"], np.ndarray)
            assert isinstance(graph_data["adjacency_sparse"], csr_matrix)
            assert isinstance(graph_data["atom_feature_names"], list)
            assert isinstance(graph_data["bond_feature_names"], list)

            # Check consistency
            rdkit_mol = mol_data.backend_data["rdkit_mol"]
            assert graph_data["num_atoms"] == rdkit_mol.GetNumAtoms()
            assert graph_data["num_bonds"] == rdkit_mol.GetNumBonds()

            if graph_data["num_atoms"] > 0:
                assert graph_data["atom_features"].shape[0] == graph_data["num_atoms"]

            if graph_data["num_bonds"] > 0:
                assert graph_data["bond_features"].shape[0] == graph_data["num_bonds"]

    def test_feature_names(self, graph_builder):
        """Test feature name generation."""
        atom_names = graph_builder._get_atom_feature_names()
        bond_names = graph_builder._get_bond_feature_names()

        assert isinstance(atom_names, list)
        assert isinstance(bond_names, list)
        assert len(atom_names) > 0
        assert len(bond_names) > 0

        # Check that names are strings
        assert all(isinstance(name, str) for name in atom_names)
        assert all(isinstance(name, str) for name in bond_names)


class TestGraphDescriptorCalculator:
    """Test GraphDescriptorCalculator functionality."""

    def test_initialization(self, graph_calculator):
        """Test graph descriptor calculator initialization."""
        assert graph_calculator is not None
        descriptors = graph_calculator.get_available_descriptors()
        assert len(descriptors) > 0
        assert "wiener_index" in descriptors
        assert "zagreb_index_1" in descriptors

    def test_wiener_index_calculation(self, graph_calculator, test_molecules):
        """Test Wiener index calculation."""
        # Test on simple molecules with known Wiener indices

        # Methane (single atom) - Wiener index should be 0
        methane = test_molecules[0]
        descriptors = graph_calculator.calculate_graph_descriptors(methane)
        assert descriptors["wiener_index"] == 0.0

        # Ethane (2 atoms) - Wiener index should be 1
        ethane = test_molecules[1]
        descriptors = graph_calculator.calculate_graph_descriptors(ethane)
        assert descriptors["wiener_index"] == 1.0

        # Propane (3 atoms in line) - Wiener index should be 4
        propane = test_molecules[2]
        descriptors = graph_calculator.calculate_graph_descriptors(propane)
        assert descriptors["wiener_index"] == 4.0

    def test_zagreb_indices(self, graph_calculator, test_molecules):
        """Test Zagreb index calculations."""
        for mol_data in test_molecules:
            descriptors = graph_calculator.calculate_graph_descriptors(mol_data)

            # Zagreb indices should be non-negative
            assert descriptors["zagreb_index_1"] >= 0
            assert descriptors["zagreb_index_2"] >= 0

            # For molecules with atoms, Zagreb 1 should be positive
            if mol_data.num_atoms > 1:
                assert descriptors["zagreb_index_1"] > 0

    def test_randic_index(self, graph_calculator, test_molecules):
        """Test Randic connectivity index."""
        for mol_data in test_molecules:
            descriptors = graph_calculator.calculate_graph_descriptors(mol_data)

            # Randic index should be non-negative
            assert descriptors["randic_index"] >= 0

            # Should be finite
            assert np.isfinite(descriptors["randic_index"])

    def test_graph_density(self, graph_calculator, test_molecules):
        """Test graph density calculation."""
        for mol_data in test_molecules:
            descriptors = graph_calculator.calculate_graph_descriptors(mol_data)

            # Graph density should be between 0 and 1
            assert 0 <= descriptors["graph_density"] <= 1

            # Single atoms should have density 0
            if mol_data.num_atoms <= 1:
                assert descriptors["graph_density"] == 0

    def test_degree_descriptors(self, graph_calculator, test_molecules):
        """Test degree-based descriptors."""
        for mol_data in test_molecules:
            descriptors = graph_calculator.calculate_graph_descriptors(mol_data)

            # Average and max degree should be non-negative
            assert descriptors["average_degree"] >= 0
            assert descriptors["max_degree"] >= 0

            # Max degree should be >= average degree
            assert descriptors["max_degree"] >= descriptors["average_degree"]

            # For molecules with bonds, average degree should be > 0
            if mol_data.num_bonds > 0:
                assert descriptors["average_degree"] > 0
                assert descriptors["max_degree"] > 0

    def test_cyclomatic_number(self, graph_calculator, test_molecules):
        """Test cyclomatic number calculation."""
        # Test specific molecules with known ring counts

        # Linear molecules should have cyclomatic number 0
        linear_molecules = [0, 1, 2]  # methane, ethane, propane
        for i in linear_molecules:
            mol_data = test_molecules[i]
            descriptors = graph_calculator.calculate_graph_descriptors(mol_data)
            assert descriptors["cyclomatic_number"] == 0

        # Cyclic molecules should have positive cyclomatic number
        cyclic_molecules = [3, 4, 5, 6, 7]  # cyclobutane, benzene, cyclohexane, etc.
        for i in cyclic_molecules:
            mol_data = test_molecules[i]
            descriptors = graph_calculator.calculate_graph_descriptors(mol_data)
            assert descriptors["cyclomatic_number"] > 0

    def test_all_descriptors(self, graph_calculator, test_molecules):
        """Test calculation of all graph descriptors."""
        for mol_data in test_molecules:
            descriptors = graph_calculator.calculate_graph_descriptors(mol_data)

            available_descriptors = graph_calculator.get_available_descriptors()

            # Should have all descriptors
            for desc_name in available_descriptors:
                assert desc_name in descriptors
                assert isinstance(descriptors[desc_name], (int, float))
                assert np.isfinite(descriptors[desc_name])

    def test_error_handling(self, graph_calculator):
        """Test error handling for invalid inputs."""
        from chemi_accel.core.molecular import MolecularData

        # Test with molecule without RDKit mol object
        invalid_mol = MolecularData(smiles="CCO", mol_id="test")
        descriptors = graph_calculator.calculate_graph_descriptors(invalid_mol)

        # Should return NaN values for all descriptors
        for desc_name in graph_calculator.get_available_descriptors():
            assert np.isnan(descriptors[desc_name])


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_build_molecular_graphs(self, test_molecules):
        """Test build_molecular_graphs function."""
        graphs = build_molecular_graphs(test_molecules)

        assert len(graphs) == len(test_molecules)

        for i, graph in enumerate(graphs):
            if graph:  # Non-empty graph
                assert "num_atoms" in graph
                assert "num_bonds" in graph
                assert "adjacency_matrix" in graph

                # Check consistency with original molecule
                mol_data = test_molecules[i]
                assert graph["num_atoms"] == mol_data.num_atoms
                assert graph["num_bonds"] == mol_data.num_bonds

    def test_calculate_graph_descriptors(self, test_molecules):
        """Test calculate_graph_descriptors function."""
        df = calculate_graph_descriptors(test_molecules)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(test_molecules)
        assert "mol_id" in df.columns
        assert "smiles" in df.columns

        # Check that descriptor columns exist
        calculator = GraphDescriptorCalculator()
        available_descriptors = calculator.get_available_descriptors()

        for desc in available_descriptors:
            assert desc in df.columns

    def test_build_graphs_with_hydrogens(self, test_molecules):
        """Test building graphs with hydrogen atoms included."""
        graphs_no_h = build_molecular_graphs(test_molecules, include_hydrogens=False)
        graphs_with_h = build_molecular_graphs(test_molecules, include_hydrogens=True)

        assert len(graphs_no_h) == len(graphs_with_h)

        for i, (graph_no_h, graph_with_h) in enumerate(zip(graphs_no_h, graphs_with_h)):
            if graph_no_h and graph_with_h:
                # Graph with hydrogens should have more or equal atoms
                assert graph_with_h["num_atoms"] >= graph_no_h["num_atoms"]


class TestSpecificMolecules:
    """Test specific molecules with known graph properties."""

    def test_benzene_properties(self, graph_calculator, parser):
        """Test benzene graph properties."""
        benzene = parser.parse_smiles("c1ccccc1", "benzene")
        descriptors = graph_calculator.calculate_graph_descriptors(benzene)

        # Benzene has 6 atoms, 6 bonds, 1 ring
        assert descriptors["cyclomatic_number"] == 1.0

        # All atoms have degree 2 in benzene
        assert descriptors["average_degree"] == 2.0
        assert descriptors["max_degree"] == 2.0

        # Zagreb index 1 = sum of squared degrees = 6 * 2^2 = 24
        assert descriptors["zagreb_index_1"] == 24.0

        # Zagreb index 2 = sum of products of adjacent degrees = 6 * 2 * 2 = 24
        assert descriptors["zagreb_index_2"] == 24.0

    def test_naphthalene_properties(self, graph_calculator, parser):
        """Test naphthalene graph properties."""
        naphthalene = parser.parse_smiles("c1ccc2ccccc2c1", "naphthalene")
        descriptors = graph_calculator.calculate_graph_descriptors(naphthalene)

        # Naphthalene has 2 fused rings
        assert descriptors["cyclomatic_number"] == 2.0

        # Should have positive Wiener index
        assert descriptors["wiener_index"] > 0

    def test_linear_alkane_properties(self, graph_calculator, parser):
        """Test linear alkane properties."""
        # Test butane (4 carbons in chain)
        butane = parser.parse_smiles("CCCC", "butane")
        descriptors = graph_calculator.calculate_graph_descriptors(butane)

        # Linear alkanes have no rings
        assert descriptors["cyclomatic_number"] == 0.0

        # Butane Wiener index should be 10
        # (1+2+3) + (1+2) + (1) = 6 + 3 + 1 = 10
        assert descriptors["wiener_index"] == 10.0


@pytest.mark.parametrize("smiles,expected_name", GRAPH_TEST_MOLECULES)
def test_individual_graph_molecules(smiles, expected_name):
    """Test individual molecules parametrically for graph properties."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")

    parser = RDKitMolecularParser()
    calculator = GraphDescriptorCalculator()

    try:
        mol_data = parser.parse_smiles(smiles, expected_name)
        descriptors = calculator.calculate_graph_descriptors(mol_data)

        # Basic sanity checks
        assert descriptors["wiener_index"] >= 0
        assert descriptors["zagreb_index_1"] >= 0
        assert descriptors["zagreb_index_2"] >= 0
        assert 0 <= descriptors["graph_density"] <= 1
        assert descriptors["average_degree"] >= 0
        assert descriptors["max_degree"] >= 0
        assert descriptors["cyclomatic_number"] >= 0

        # All values should be finite
        for desc_name, value in descriptors.items():
            assert np.isfinite(
                value
            ), f"Non-finite value for {desc_name} in {expected_name}"

    except Exception as e:
        pytest.fail(f"Failed to process molecule {expected_name} ({smiles}): {e}")


if __name__ == "__main__":
    pytest.main([__file__])

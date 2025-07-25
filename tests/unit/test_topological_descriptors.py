"""
Unit tests for topological molecular descriptors.
"""


import numpy as np
import pandas as pd
import pytest

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.chemi_accel.core.molecular import MolecularData
from src.chemi_accel.descriptors.topological import (
    TopologicalCUDAKernels,
    TopologicalDescriptorCalculator,
    calculate_topological_descriptors,
    get_topological_descriptor_info,
)


@pytest.fixture
def sample_molecules():
    """Create sample molecules for testing."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")

    molecules = []

    # Benzene
    benzene_mol = Chem.MolFromSmiles("c1ccccc1")
    benzene_data = MolecularData(
        mol_id="benzene", smiles="c1ccccc1", backend_data={"rdkit_mol": benzene_mol}
    )
    molecules.append(benzene_data)

    # Ethane
    ethane_mol = Chem.MolFromSmiles("CC")
    ethane_data = MolecularData(
        mol_id="ethane", smiles="CC", backend_data={"rdkit_mol": ethane_mol}
    )
    molecules.append(ethane_data)

    # Propane
    propane_mol = Chem.MolFromSmiles("CCC")
    propane_data = MolecularData(
        mol_id="propane", smiles="CCC", backend_data={"rdkit_mol": propane_mol}
    )
    molecules.append(propane_data)

    # Methanol
    methanol_mol = Chem.MolFromSmiles("CO")
    methanol_data = MolecularData(
        mol_id="methanol", smiles="CO", backend_data={"rdkit_mol": methanol_mol}
    )
    molecules.append(methanol_data)

    return molecules


@pytest.fixture
def topological_calculator():
    """Create topological descriptor calculator."""
    return TopologicalDescriptorCalculator(use_gpu=False)


class TestTopologicalDescriptorCalculator:
    """Test topological descriptor calculator."""

    def test_initialization(self):
        """Test calculator initialization."""
        calc = TopologicalDescriptorCalculator(use_gpu=False)
        assert calc is not None
        assert not calc.use_gpu
        assert len(calc.descriptor_functions) > 0
        assert len(calc.atomic_properties) == 4

    def test_initialization_with_gpu(self):
        """Test calculator initialization with GPU."""
        calc = TopologicalDescriptorCalculator(use_gpu=True)
        assert calc is not None
        # GPU availability depends on system

    def test_get_available_descriptors(self, topological_calculator):
        """Test getting available descriptor names."""
        descriptors = topological_calculator.get_available_descriptors()
        assert isinstance(descriptors, list)
        assert len(descriptors) > 0

        # Check for key descriptor categories
        connectivity_descriptors = [
            d for d in descriptors if "wiener" in d or "zagreb" in d or "randic" in d
        ]
        assert len(connectivity_descriptors) > 0

        topological_charge_descriptors = [
            d for d in descriptors if "topological_charge" in d
        ]
        assert len(topological_charge_descriptors) > 0

        kappa_descriptors = [d for d in descriptors if "kappa" in d]
        assert len(kappa_descriptors) > 0

        path_descriptors = [d for d in descriptors if "path_count" in d]
        assert len(path_descriptors) > 0

        autocorr_descriptors = [d for d in descriptors if "autocorr" in d]
        assert len(autocorr_descriptors) > 0

    def test_get_descriptor_info(self, topological_calculator):
        """Test getting descriptor information."""
        info = topological_calculator.get_descriptor_info()
        assert isinstance(info, dict)
        assert len(info) > 0

        # Check structure of descriptor info
        for desc_name, desc_info in info.items():
            assert "category" in desc_info
            assert "description" in desc_info
            assert isinstance(desc_info["category"], str)
            assert isinstance(desc_info["description"], str)

    def test_atomic_properties(self, topological_calculator):
        """Test atomic property dictionaries."""
        # Test that all property dictionaries have common elements
        common_elements = [1, 6, 7, 8]  # H, C, N, O

        for prop_name, prop_dict in topological_calculator.atomic_properties.items():
            assert isinstance(prop_dict, dict)
            for element in common_elements:
                assert element in prop_dict
                assert isinstance(prop_dict[element], (int, float))
                assert prop_dict[element] > 0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_calculate_connectivity_indices(
        self, topological_calculator, sample_molecules
    ):
        """Test connectivity index calculations."""
        for mol_data in sample_molecules:
            descriptors = topological_calculator.calculate_topological_descriptors(
                mol_data
            )

            # Check that connectivity indices are calculated
            assert "wiener_index" in descriptors
            assert "zagreb_index_1" in descriptors
            assert "zagreb_index_2" in descriptors
            assert "randic_index" in descriptors
            assert "balaban_j" in descriptors
            assert "harary_index" in descriptors

            # Check that values are reasonable
            assert descriptors["wiener_index"] >= 0
            assert descriptors["zagreb_index_1"] >= 0
            assert descriptors["zagreb_index_2"] >= 0
            assert descriptors["randic_index"] >= 0
            assert descriptors["balaban_j"] >= 0
            assert descriptors["harary_index"] >= 0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_calculate_topological_charge_indices(
        self, topological_calculator, sample_molecules
    ):
        """Test topological charge index calculations."""
        for mol_data in sample_molecules:
            descriptors = topological_calculator.calculate_topological_descriptors(
                mol_data
            )

            # Check that topological charge indices are calculated
            for i in range(1, 11):
                tc_name = f"topological_charge_{i}"
                assert tc_name in descriptors
                assert descriptors[tc_name] >= 0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_calculate_estate_descriptors(
        self, topological_calculator, sample_molecules
    ):
        """Test E-state descriptor calculations."""
        for mol_data in sample_molecules:
            descriptors = topological_calculator.calculate_topological_descriptors(
                mol_data
            )

            # Check that E-state VSA descriptors are calculated
            for i in range(1, 12):
                estate_name = f"estate_vsa_{i}"
                assert estate_name in descriptors
                assert descriptors[estate_name] >= 0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_calculate_kappa_indices(self, topological_calculator, sample_molecules):
        """Test Kappa shape index calculations."""
        for mol_data in sample_molecules:
            descriptors = topological_calculator.calculate_topological_descriptors(
                mol_data
            )

            # Check that Kappa indices are calculated
            assert "kappa_1" in descriptors
            assert "kappa_2" in descriptors
            assert "kappa_3" in descriptors
            assert "kappa_alpha_1" in descriptors
            assert "kappa_alpha_2" in descriptors
            assert "kappa_alpha_3" in descriptors
            assert "flexibility_index" in descriptors

            # Check that values are reasonable
            assert descriptors["kappa_1"] >= 0
            assert descriptors["kappa_2"] >= 0
            assert descriptors["kappa_3"] >= 0
            assert 0 <= descriptors["flexibility_index"] <= 1

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_calculate_path_counts(self, topological_calculator, sample_molecules):
        """Test path count calculations."""
        for mol_data in sample_molecules:
            descriptors = topological_calculator.calculate_topological_descriptors(
                mol_data
            )

            # Check that path counts are calculated
            for i in range(1, 7):
                path_name = f"path_count_{i}"
                cluster_name = f"cluster_count_{i}"
                assert path_name in descriptors
                assert cluster_name in descriptors
                assert descriptors[path_name] >= 0
                assert descriptors[cluster_name] >= 0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_calculate_autocorrelation_descriptors(
        self, topological_calculator, sample_molecules
    ):
        """Test autocorrelation descriptor calculations."""
        for mol_data in sample_molecules:
            descriptors = topological_calculator.calculate_topological_descriptors(
                mol_data
            )

            # Check that autocorrelation descriptors are calculated
            properties = ["mass", "volume", "electronegativity", "polarizability"]
            for prop in properties:
                moran_name = f"moran_autocorr_{prop}"
                geary_name = f"geary_autocorr_{prop}"
                assert moran_name in descriptors
                assert geary_name in descriptors
                # Autocorrelation values can be negative
                assert isinstance(descriptors[moran_name], (int, float))
                assert isinstance(descriptors[geary_name], (int, float))

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_calculate_additional_indices(
        self, topological_calculator, sample_molecules
    ):
        """Test additional topological index calculations."""
        for mol_data in sample_molecules:
            descriptors = topological_calculator.calculate_topological_descriptors(
                mol_data
            )

            # Check additional indices
            additional_indices = [
                "schultz_index",
                "gutman_index",
                "xu_index",
                "eccentric_connectivity_index",
                "total_adjacency_index",
                "symmetric_division_index",
                "harmonic_topological_index",
                "geometric_topological_index",
                "arithmetic_topological_index",
            ]

            for index_name in additional_indices:
                assert index_name in descriptors
                assert descriptors[index_name] >= 0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_benzene_specific_values(self, topological_calculator):
        """Test specific descriptor values for benzene."""
        benzene_mol = Chem.MolFromSmiles("c1ccccc1")
        benzene_data = MolecularData(
            mol_id="benzene", smiles="c1ccccc1", backend_data={"rdkit_mol": benzene_mol}
        )

        descriptors = topological_calculator.calculate_topological_descriptors(
            benzene_data
        )

        # Benzene has 6 atoms, 6 bonds
        assert descriptors["path_count_1"] == 6  # 6 bonds
        assert descriptors["cluster_count_1"] == 6  # 6 atoms
        assert descriptors["cluster_count_2"] == 6  # 6 bonds

        # Zagreb indices for benzene (each vertex has degree 2)
        assert descriptors["zagreb_index_1"] == 24  # 6 * 2^2 = 24
        assert descriptors["zagreb_index_2"] == 24  # 6 * 2 * 2 = 24

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_ethane_specific_values(self, topological_calculator):
        """Test specific descriptor values for ethane."""
        ethane_mol = Chem.MolFromSmiles("CC")
        ethane_data = MolecularData(
            mol_id="ethane", smiles="CC", backend_data={"rdkit_mol": ethane_mol}
        )

        descriptors = topological_calculator.calculate_topological_descriptors(
            ethane_data
        )

        # Ethane has 2 atoms, 1 bond
        assert descriptors["path_count_1"] == 1  # 1 bond
        assert descriptors["cluster_count_1"] == 2  # 2 atoms
        assert descriptors["cluster_count_2"] == 1  # 1 bond
        assert descriptors["wiener_index"] == 1  # Distance between the two atoms is 1

    def test_error_handling_no_rdkit_mol(self, topological_calculator):
        """Test error handling when RDKit mol is missing."""
        mol_data = MolecularData(mol_id="test", smiles="CC", backend_data={})

        descriptors = topological_calculator.calculate_topological_descriptors(mol_data)

        # Should return NaN values for all descriptors
        for desc_name in topological_calculator.get_available_descriptors():
            assert np.isnan(descriptors[desc_name])

    def test_distance_matrix_cpu(self, topological_calculator):
        """Test CPU distance matrix calculation."""
        # Simple 3-node path: 0-1-2
        adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        dist_matrix = topological_calculator._get_distance_matrix_cpu(adj_matrix)

        expected = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

        np.testing.assert_array_equal(dist_matrix, expected)

    def test_distance_matrix_gpu_fallback(self, topological_calculator):
        """Test GPU distance matrix calculation fallback."""
        # Set use_gpu to True to test GPU path
        topological_calculator.use_gpu = True

        # Mock the GPU calculation to raise an exception
        original_method = topological_calculator._get_distance_matrix_gpu

        def mock_gpu_method(adj_matrix):
            # Simulate GPU error and fallback
            try:
                raise Exception("GPU error")
            except Exception:
                return topological_calculator._get_distance_matrix_cpu(adj_matrix)

        topological_calculator._get_distance_matrix_gpu = mock_gpu_method

        adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        # Should fall back to CPU calculation
        dist_matrix = topological_calculator._get_distance_matrix(adj_matrix)

        expected = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

        np.testing.assert_array_equal(dist_matrix, expected)

        # Restore original method
        topological_calculator._get_distance_matrix_gpu = original_method


class TestTopologicalCUDAKernels:
    """Test CUDA kernels for topological descriptors."""

    def test_initialization(self):
        """Test CUDA kernels initialization."""
        kernels = TopologicalCUDAKernels()
        assert kernels is not None
        # kernels_available depends on system

    def test_wiener_index_batch_no_cuda(self):
        """Test Wiener index batch calculation without CUDA."""
        kernels = TopologicalCUDAKernels()
        kernels.kernels_available = False

        adj_matrices = [np.array([[0, 1], [1, 0]])]

        with pytest.raises(Exception):
            kernels.calculate_wiener_index_batch(adj_matrices)

    def test_zagreb_indices_batch_no_cuda(self):
        """Test Zagreb indices batch calculation without CUDA."""
        kernels = TopologicalCUDAKernels()
        kernels.kernels_available = False

        adj_matrices = [np.array([[0, 1], [1, 0]])]

        with pytest.raises(Exception):
            kernels.calculate_zagreb_indices_batch(adj_matrices)


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_calculate_topological_descriptors(self, sample_molecules):
        """Test convenience function for calculating topological descriptors."""
        df = calculate_topological_descriptors(sample_molecules, use_gpu=False)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_molecules)
        assert "mol_id" in df.columns
        assert "smiles" in df.columns

        # Check that descriptor columns are present
        desc_cols = [col for col in df.columns if col not in ["mol_id", "smiles"]]
        assert len(desc_cols) > 0

        # Check that all values are numeric
        for col in desc_cols:
            assert df[col].dtype in [np.float64, np.int64] or df[col].isna().all()

    def test_get_topological_descriptor_info(self):
        """Test getting topological descriptor information."""
        info = get_topological_descriptor_info()

        assert isinstance(info, dict)
        assert len(info) > 0

        # Check structure
        for desc_name, desc_info in info.items():
            assert "category" in desc_info
            assert "description" in desc_info

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_empty_molecule_list(self):
        """Test handling of empty molecule list."""
        df = calculate_topological_descriptors([], use_gpu=False)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_descriptor_consistency(self, sample_molecules):
        """Test that descriptors are consistent across multiple calculations."""
        df1 = calculate_topological_descriptors(sample_molecules, use_gpu=False)
        df2 = calculate_topological_descriptors(sample_molecules, use_gpu=False)

        # Results should be identical
        pd.testing.assert_frame_equal(df1, df2)

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_descriptor_ranges(self, sample_molecules):
        """Test that descriptor values are in reasonable ranges."""
        df = calculate_topological_descriptors(sample_molecules, use_gpu=False)

        # Most topological descriptors should be non-negative
        non_negative_descriptors = [
            "wiener_index",
            "zagreb_index_1",
            "zagreb_index_2",
            "randic_index",
            "balaban_j",
            "harary_index",
            "kappa_1",
            "kappa_2",
            "kappa_3",
        ]

        for desc in non_negative_descriptors:
            if desc in df.columns:
                assert (df[desc] >= 0).all() or df[desc].isna().all()

        # Flexibility index should be between 0 and 1
        if "flexibility_index" in df.columns:
            assert (
                (df["flexibility_index"] >= 0) & (df["flexibility_index"] <= 1)
            ).all() or df["flexibility_index"].isna().all()


if __name__ == "__main__":
    pytest.main([__file__])

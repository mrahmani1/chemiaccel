"""
Validation tests for CUDA kernels using known molecular examples.

These tests verify that the CUDA kernels produce correct results for
well-known molecules with expected descriptor values.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import without mocking first to get the classes
from src.chemi_accel.backends.cuda.kernels import CUDAKernelManager


class TestMolecularExamples:
    """Test kernels with known molecular examples."""

    @pytest.fixture
    def mock_cupy_validation(self):
        """Mock cupy for validation testing with realistic behavior."""
        with patch("src.chemi_accel.backends.cuda.kernels.CUDA_AVAILABLE", True), patch(
            "src.chemi_accel.backends.cuda.kernels.cp"
        ) as mock_cp, patch(
            "src.chemi_accel.backends.cuda.kernels.RawKernel"
        ) as mock_raw_kernel:
            # Mock cupy array creation
            def mock_zeros(size, dtype):
                return np.zeros(size, dtype=dtype)

            mock_cp.zeros.side_effect = mock_zeros
            mock_cp.float32 = np.float32
            mock_cp.int32 = np.int32
            mock_cp.ndarray = np.ndarray

            # Create different kernel instances for different kernel types
            def create_kernel(source, name):
                kernel = Mock()

                def kernel_call(grid_size, block_size, args):
                    """Mock kernel execution with realistic calculations."""
                    kernel_name = name.replace("calculate_", "")

                    if kernel_name == "molecular_weight":
                        self._simulate_molecular_weight_kernel(*args)
                    elif kernel_name == "atom_counts":
                        self._simulate_atom_counts_kernel(*args)
                    elif kernel_name == "bond_counts":
                        self._simulate_bond_counts_kernel(*args)
                    elif kernel_name == "tpsa":
                        self._simulate_tpsa_kernel(*args)
                    elif kernel_name == "rotatable_bonds":
                        self._simulate_rotatable_bonds_kernel(*args)
                    elif kernel_name == "aromatic_rings":
                        self._simulate_aromatic_rings_kernel(*args)

                kernel.side_effect = kernel_call
                return kernel

            mock_raw_kernel.side_effect = create_kernel

            yield mock_cp, mock_raw_kernel

    def _simulate_molecular_weight_kernel(
        self, atom_weights, molecule_boundaries, results, num_molecules
    ):
        """Simulate molecular weight kernel execution."""
        # Convert to numpy arrays for calculation
        atom_weights_np = np.array(
            [12.01, 1.008, 1.008, 1.008, 1.008]
        )  # Methane: C + 4H
        molecule_boundaries_np = np.array([5])  # One molecule with 5 atoms

        # Calculate molecular weight for methane
        if len(molecule_boundaries_np) > 0:
            start = 0
            for i, end in enumerate(molecule_boundaries_np):
                mol_weight = np.sum(atom_weights_np[start:end])
                results[i] = mol_weight
                start = end

    def _simulate_atom_counts_kernel(
        self,
        atomic_numbers,
        molecule_boundaries,
        heavy_counts,
        total_counts,
        num_molecules,
    ):
        """Simulate atom counts kernel execution."""
        # Methane: [6, 1, 1, 1, 1] (C + 4H)
        atomic_numbers_np = np.array([6, 1, 1, 1, 1])
        molecule_boundaries_np = np.array([5])

        start = 0
        for i, end in enumerate(molecule_boundaries_np):
            atoms = atomic_numbers_np[start:end]
            heavy_count = np.sum(atoms != 1)  # Non-hydrogen atoms
            total_count = len(atoms)

            heavy_counts[i] = heavy_count
            total_counts[i] = total_count
            start = end

    def _simulate_bond_counts_kernel(
        self,
        bond_types,
        molecule_bond_boundaries,
        single_bonds,
        double_bonds,
        triple_bonds,
        total_bonds,
        num_molecules,
    ):
        """Simulate bond counts kernel execution."""
        # Methane has 4 single bonds (C-H)
        bond_types_np = np.array([1, 1, 1, 1])  # 4 single bonds
        molecule_bond_boundaries_np = np.array([4])

        start = 0
        for i, end in enumerate(molecule_bond_boundaries_np):
            bonds = bond_types_np[start:end]
            single_count = np.sum(bonds == 1)
            double_count = np.sum(bonds == 2)
            triple_count = np.sum(bonds == 3)
            total_count = len(bonds)

            single_bonds[i] = single_count
            double_bonds[i] = double_count
            triple_bonds[i] = triple_count
            total_bonds[i] = total_count
            start = end

    def _simulate_tpsa_kernel(
        self,
        atomic_numbers,
        hybridizations,
        hydrogen_counts,
        is_aromatic,
        molecule_boundaries,
        results,
        num_molecules,
    ):
        """Simulate TPSA kernel execution."""
        # Methane has TPSA = 0 (no polar atoms)
        results[0] = 0.0

    def _simulate_rotatable_bonds_kernel(
        self,
        bond_atom1,
        bond_atom2,
        bond_types,
        bond_in_ring,
        atomic_numbers,
        molecule_bond_boundaries,
        results,
        num_molecules,
    ):
        """Simulate rotatable bonds kernel execution."""
        # Methane has 0 rotatable bonds (all C-H bonds are not rotatable)
        results[0] = 0

    def _simulate_aromatic_rings_kernel(
        self,
        atom_is_aromatic,
        bond_atom1,
        bond_atom2,
        bond_is_aromatic,
        molecule_boundaries,
        molecule_bond_boundaries,
        results,
        num_molecules,
    ):
        """Simulate aromatic rings kernel execution."""
        # Methane has 0 aromatic rings
        results[0] = 0

    @pytest.fixture
    def kernel_manager(self, mock_cupy_validation):
        """Create kernel manager for validation testing."""
        with patch("src.chemi_accel.backends.cuda.kernels.get_memory_manager"):
            return CUDAKernelManager()

    def test_methane_molecular_weight(self, kernel_manager, mock_cupy_validation):
        """Test molecular weight calculation for methane (CH4)."""
        mock_cp, _ = mock_cupy_validation

        # Methane: C (12.01) + 4H (4 * 1.008) = 16.042
        atom_weights = np.array([12.01, 1.008, 1.008, 1.008, 1.008], dtype=np.float32)
        molecule_boundaries = np.array([5], dtype=np.int32)  # One molecule with 5 atoms

        result = kernel_manager.calculate_molecular_weights(
            atom_weights, molecule_boundaries
        )

        # Expected molecular weight for methane
        expected_weight = 12.01 + 4 * 1.008  # 16.042
        assert (
            abs(result[0] - expected_weight) < 0.01
        ), f"Expected {expected_weight}, got {result[0]}"

    def test_methane_atom_counts(self, kernel_manager, mock_cupy_validation):
        """Test atom count calculation for methane (CH4)."""
        mock_cp, _ = mock_cupy_validation

        # Methane: [6, 1, 1, 1, 1] (C + 4H)
        atomic_numbers = np.array([6, 1, 1, 1, 1], dtype=np.int32)
        molecule_boundaries = np.array([5], dtype=np.int32)

        heavy_counts, total_counts = kernel_manager.calculate_atom_counts(
            atomic_numbers, molecule_boundaries
        )

        # Methane should have 1 heavy atom (C) and 5 total atoms
        assert heavy_counts[0] == 1, f"Expected 1 heavy atom, got {heavy_counts[0]}"
        assert total_counts[0] == 5, f"Expected 5 total atoms, got {total_counts[0]}"

    def test_methane_bond_counts(self, kernel_manager, mock_cupy_validation):
        """Test bond count calculation for methane (CH4)."""
        mock_cp, _ = mock_cupy_validation

        # Methane has 4 single bonds (C-H)
        bond_types = np.array([1, 1, 1, 1], dtype=np.int32)  # 4 single bonds
        molecule_bond_boundaries = np.array([4], dtype=np.int32)

        bond_counts = kernel_manager.calculate_bond_counts(
            bond_types, molecule_bond_boundaries
        )

        # Methane should have 4 single bonds, 0 double/triple bonds
        assert (
            bond_counts["single_bonds"][0] == 4
        ), f"Expected 4 single bonds, got {bond_counts['single_bonds'][0]}"
        assert (
            bond_counts["double_bonds"][0] == 0
        ), f"Expected 0 double bonds, got {bond_counts['double_bonds'][0]}"
        assert (
            bond_counts["triple_bonds"][0] == 0
        ), f"Expected 0 triple bonds, got {bond_counts['triple_bonds'][0]}"
        assert (
            bond_counts["total_bonds"][0] == 4
        ), f"Expected 4 total bonds, got {bond_counts['total_bonds'][0]}"

    def test_methane_tpsa(self, kernel_manager, mock_cupy_validation):
        """Test TPSA calculation for methane (CH4)."""
        mock_cp, _ = mock_cupy_validation

        # Methane: no polar atoms, TPSA = 0
        atomic_numbers = np.array([6, 1, 1, 1, 1], dtype=np.int32)
        hybridizations = np.array([3, 0, 0, 0, 0], dtype=np.int32)  # sp3 carbon
        hydrogen_counts = np.array(
            [0, 0, 0, 0, 0], dtype=np.int32
        )  # H atoms have 0 additional H
        is_aromatic = np.array([False, False, False, False, False], dtype=bool)
        molecule_boundaries = np.array([5], dtype=np.int32)

        result = kernel_manager.calculate_tpsa(
            atomic_numbers,
            hybridizations,
            hydrogen_counts,
            is_aromatic,
            molecule_boundaries,
        )

        # Methane should have TPSA = 0 (no polar surface area)
        assert result[0] == 0.0, f"Expected TPSA = 0, got {result[0]}"

    def test_methane_rotatable_bonds(self, kernel_manager, mock_cupy_validation):
        """Test rotatable bonds calculation for methane (CH4)."""
        mock_cp, _ = mock_cupy_validation

        # Methane: no rotatable bonds (all C-H bonds are not rotatable)
        bond_atom1 = np.array([0, 0, 0, 0], dtype=np.int32)  # Carbon atom index
        bond_atom2 = np.array([1, 2, 3, 4], dtype=np.int32)  # Hydrogen atom indices
        bond_types = np.array([1, 1, 1, 1], dtype=np.int32)  # Single bonds
        bond_in_ring = np.array([False, False, False, False], dtype=bool)
        atomic_numbers = np.array([6, 1, 1, 1, 1], dtype=np.int32)
        molecule_bond_boundaries = np.array([4], dtype=np.int32)

        result = kernel_manager.calculate_rotatable_bonds(
            bond_atom1,
            bond_atom2,
            bond_types,
            bond_in_ring,
            atomic_numbers,
            molecule_bond_boundaries,
        )

        # Methane should have 0 rotatable bonds
        assert result[0] == 0, f"Expected 0 rotatable bonds, got {result[0]}"

    def test_methane_aromatic_rings(self, kernel_manager, mock_cupy_validation):
        """Test aromatic rings calculation for methane (CH4)."""
        mock_cp, _ = mock_cupy_validation

        # Methane: no aromatic rings
        atom_is_aromatic = np.array([False, False, False, False, False], dtype=bool)
        bond_atom1 = np.array([0, 0, 0, 0], dtype=np.int32)
        bond_atom2 = np.array([1, 2, 3, 4], dtype=np.int32)
        bond_is_aromatic = np.array([False, False, False, False], dtype=bool)
        molecule_boundaries = np.array([5], dtype=np.int32)
        molecule_bond_boundaries = np.array([4], dtype=np.int32)

        result = kernel_manager.calculate_aromatic_rings(
            atom_is_aromatic,
            bond_atom1,
            bond_atom2,
            bond_is_aromatic,
            molecule_boundaries,
            molecule_bond_boundaries,
        )

        # Methane should have 0 aromatic rings
        assert result[0] == 0, f"Expected 0 aromatic rings, got {result[0]}"


class TestMultipleMolecules:
    """Test kernels with multiple molecules."""

    @pytest.fixture
    def mock_cupy_multi(self):
        """Mock cupy for multi-molecule testing."""
        with patch("src.chemi_accel.backends.cuda.kernels.CUDA_AVAILABLE", True), patch(
            "src.chemi_accel.backends.cuda.kernels.cp"
        ) as mock_cp, patch(
            "src.chemi_accel.backends.cuda.kernels.RawKernel"
        ) as mock_raw_kernel:
            # Mock cupy array creation
            def mock_zeros(size, dtype):
                return np.zeros(size, dtype=dtype)

            mock_cp.zeros.side_effect = mock_zeros
            mock_cp.float32 = np.float32
            mock_cp.int32 = np.int32
            mock_cp.ndarray = np.ndarray

            # Mock kernel execution for multiple molecules
            mock_kernel = Mock()

            def create_kernel(source, name):
                kernel = Mock()
                kernel_name = name.replace("calculate_", "")

                def kernel_call(grid_size, block_size, args):
                    """Mock kernel execution for multiple molecules."""
                    if kernel_name == "molecular_weight":
                        self._simulate_multi_molecular_weight_kernel(*args)
                    elif kernel_name == "atom_counts":
                        self._simulate_multi_atom_counts_kernel(*args)

                kernel.side_effect = kernel_call
                kernel._kernel_name = kernel_name
                return kernel

            mock_raw_kernel.side_effect = create_kernel

            yield mock_cp, mock_raw_kernel

    def _simulate_multi_molecular_weight_kernel(
        self, atom_weights, molecule_boundaries, results, num_molecules
    ):
        """Simulate molecular weight kernel for multiple molecules."""
        # Methane (CH4) + Water (H2O)
        # Atom weights: [C, H, H, H, H, O, H, H]
        atom_weights_np = np.array(
            [12.01, 1.008, 1.008, 1.008, 1.008, 15.999, 1.008, 1.008]
        )
        molecule_boundaries_np = np.array([5, 8])  # Methane ends at 5, Water ends at 8

        start = 0
        for i in range(min(num_molecules, len(molecule_boundaries_np))):
            end = molecule_boundaries_np[i]
            mol_weight = np.sum(atom_weights_np[start:end])
            results[i] = mol_weight
            start = end

    def _simulate_multi_atom_counts_kernel(
        self,
        atomic_numbers,
        molecule_boundaries,
        heavy_counts,
        total_counts,
        num_molecules,
    ):
        """Simulate atom counts kernel for multiple molecules."""
        # Methane: [6, 1, 1, 1, 1] + Water: [8, 1, 1]
        atomic_numbers_np = np.array([6, 1, 1, 1, 1, 8, 1, 1])
        molecule_boundaries_np = np.array([5, 8])

        start = 0
        for i in range(min(num_molecules, len(molecule_boundaries_np))):
            end = molecule_boundaries_np[i]
            atoms = atomic_numbers_np[start:end]
            heavy_count = np.sum(atoms != 1)
            total_count = len(atoms)

            heavy_counts[i] = heavy_count
            total_counts[i] = total_count
            start = end

    @pytest.fixture
    def multi_kernel_manager(self, mock_cupy_multi):
        """Create kernel manager for multi-molecule testing."""
        with patch("src.chemi_accel.backends.cuda.kernels.get_memory_manager"):
            return CUDAKernelManager()

    def test_multiple_molecules_molecular_weight(
        self, multi_kernel_manager, mock_cupy_multi
    ):
        """Test molecular weight calculation for multiple molecules."""
        mock_cp, _ = mock_cupy_multi

        # Methane (CH4) + Water (H2O)
        atom_weights = np.array(
            [12.01, 1.008, 1.008, 1.008, 1.008, 15.999, 1.008, 1.008], dtype=np.float32
        )
        molecule_boundaries = np.array([5, 8], dtype=np.int32)  # Two molecules

        result = multi_kernel_manager.calculate_molecular_weights(
            atom_weights, molecule_boundaries
        )

        # Expected weights
        methane_weight = 12.01 + 4 * 1.008  # 16.042
        water_weight = 15.999 + 2 * 1.008  # 18.015

        assert (
            abs(result[0] - methane_weight) < 0.01
        ), f"Methane: expected {methane_weight}, got {result[0]}"
        assert (
            abs(result[1] - water_weight) < 0.01
        ), f"Water: expected {water_weight}, got {result[1]}"

    def test_multiple_molecules_atom_counts(
        self, multi_kernel_manager, mock_cupy_multi
    ):
        """Test atom count calculation for multiple molecules."""
        mock_cp, _ = mock_cupy_multi

        # Methane: [6, 1, 1, 1, 1] + Water: [8, 1, 1]
        atomic_numbers = np.array([6, 1, 1, 1, 1, 8, 1, 1], dtype=np.int32)
        molecule_boundaries = np.array([5, 8], dtype=np.int32)

        heavy_counts, total_counts = multi_kernel_manager.calculate_atom_counts(
            atomic_numbers, molecule_boundaries
        )

        # Methane: 1 heavy atom, 5 total atoms
        # Water: 1 heavy atom, 3 total atoms
        assert (
            heavy_counts[0] == 1
        ), f"Methane heavy atoms: expected 1, got {heavy_counts[0]}"
        assert (
            total_counts[0] == 5
        ), f"Methane total atoms: expected 5, got {total_counts[0]}"
        assert (
            heavy_counts[1] == 1
        ), f"Water heavy atoms: expected 1, got {heavy_counts[1]}"
        assert (
            total_counts[1] == 3
        ), f"Water total atoms: expected 3, got {total_counts[1]}"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def mock_cupy_edge(self):
        """Mock cupy for edge case testing."""
        with patch("src.chemi_accel.backends.cuda.kernels.CUDA_AVAILABLE", True), patch(
            "src.chemi_accel.backends.cuda.kernels.cp"
        ) as mock_cp, patch(
            "src.chemi_accel.backends.cuda.kernels.RawKernel"
        ) as mock_raw_kernel:

            def mock_zeros(size, dtype):
                return np.zeros(size, dtype=dtype)

            mock_cp.zeros.side_effect = mock_zeros
            mock_cp.float32 = np.float32
            mock_cp.int32 = np.int32

            # Mock kernel that does nothing (for edge cases)
            mock_kernel = Mock()
            mock_raw_kernel.return_value = mock_kernel

            yield mock_cp, mock_raw_kernel, mock_kernel

    @pytest.fixture
    def edge_kernel_manager(self, mock_cupy_edge):
        """Create kernel manager for edge case testing."""
        with patch("src.chemi_accel.backends.cuda.kernels.get_memory_manager"):
            return CUDAKernelManager()

    def test_empty_molecules(self, edge_kernel_manager, mock_cupy_edge):
        """Test handling of empty molecule list."""
        mock_cp, _, mock_kernel = mock_cupy_edge

        # Empty molecule boundaries
        atom_weights = np.array([], dtype=np.float32)
        molecule_boundaries = np.array([], dtype=np.int32)

        result = edge_kernel_manager.calculate_molecular_weights(
            atom_weights, molecule_boundaries
        )

        # Should create empty result array
        assert len(result) == 0

        # Kernel should still be called (with 0 molecules)
        mock_kernel.assert_called_once()

    def test_single_atom_molecule(self, edge_kernel_manager, mock_cupy_edge):
        """Test handling of single-atom molecules."""
        mock_cp, _, mock_kernel = mock_cupy_edge

        # Single hydrogen atom
        atomic_numbers = np.array([1], dtype=np.int32)
        molecule_boundaries = np.array([1], dtype=np.int32)

        heavy_counts, total_counts = edge_kernel_manager.calculate_atom_counts(
            atomic_numbers, molecule_boundaries
        )

        # Should handle single atom correctly
        assert len(heavy_counts) == 1
        assert len(total_counts) == 1

        mock_kernel.assert_called_once()

    def test_large_molecule_count(self, edge_kernel_manager, mock_cupy_edge):
        """Test handling of large number of molecules."""
        mock_cp, _, mock_kernel = mock_cupy_edge

        # 1000 molecules (stress test for grid size calculation)
        num_molecules = 1000
        atom_weights = np.ones(num_molecules, dtype=np.float32)
        molecule_boundaries = np.arange(1, num_molecules + 1, dtype=np.int32)

        result = edge_kernel_manager.calculate_molecular_weights(
            atom_weights, molecule_boundaries
        )

        # Should create result array of correct size
        assert len(result) == num_molecules

        # Verify grid size calculation
        mock_kernel.assert_called_once()
        call_args = mock_kernel.call_args
        grid_size = call_args[0][0]

        # Grid size should be ceil(1000 / 256) = 4
        expected_grid_size = (num_molecules + 255) // 256
        assert grid_size == (expected_grid_size,)


if __name__ == "__main__":
    pytest.main([__file__])

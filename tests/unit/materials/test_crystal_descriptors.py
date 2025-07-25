"""
Tests for crystal descriptor calculations.
"""

import numpy as np
import pandas as pd

from src.chemi_accel.materials.crystal_descriptors import (
    CrystalDescriptorCalculator,
    CrystalDescriptors,
)
from src.chemi_accel.materials.crystal_parser import CrystalStructure


class TestCrystalDescriptors:
    """Test CrystalDescriptors data class."""

    def test_crystal_descriptors_creation(self):
        """Test basic crystal descriptors creation."""
        descriptors = CrystalDescriptors(
            volume=27.0,
            density=2.0,
            packing_efficiency=0.5,
            void_fraction=0.3,
            accessible_void_fraction=0.2,
            pore_limiting_diameter=2.0,
            largest_cavity_diameter=3.0,
            surface_area_geometric=100.0,
            surface_area_accessible=80.0,
            surface_area_volumetric=400.0,
            coordination_numbers=np.array([4, 4]),
            average_coordination=4.0,
            coordination_variance=0.0,
            lattice_parameters=np.array([3.0, 3.0, 3.0]),
            lattice_angles=np.array([90.0, 90.0, 90.0]),
            lattice_volume=27.0,
        )

        assert descriptors.volume == 27.0
        assert descriptors.density == 2.0
        assert len(descriptors.coordination_numbers) == 2

    def test_descriptors_to_dict(self):
        """Test conversion to dictionary."""
        descriptors = CrystalDescriptors(
            volume=27.0,
            density=2.0,
            packing_efficiency=0.5,
            void_fraction=0.3,
            accessible_void_fraction=0.2,
            pore_limiting_diameter=2.0,
            largest_cavity_diameter=3.0,
            surface_area_geometric=100.0,
            surface_area_accessible=80.0,
            surface_area_volumetric=400.0,
            coordination_numbers=np.array([4, 4, 6]),
            average_coordination=4.67,
            coordination_variance=0.89,
            lattice_parameters=np.array([3.0, 3.0, 3.0]),
            lattice_angles=np.array([90.0, 90.0, 90.0]),
            lattice_volume=27.0,
        )

        result_dict = descriptors.to_dict()

        assert "volume" in result_dict
        assert "density" in result_dict
        assert "coordination_numbers_mean" in result_dict
        assert "coordination_numbers_std" in result_dict
        assert "lattice_parameters_mean" in result_dict


class TestCrystalDescriptorCalculator:
    """Test crystal descriptor calculator."""

    def create_simple_cubic_structure(self) -> CrystalStructure:
        """Create a simple cubic crystal structure for testing."""
        lattice_vectors = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])

        atomic_numbers = np.array([6, 6])
        atomic_symbols = ["C", "C"]
        fractional_coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        cartesian_coords = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])

        return CrystalStructure(
            lattice_vectors=lattice_vectors,
            lattice_parameters=None,
            atomic_numbers=atomic_numbers,
            atomic_symbols=atomic_symbols,
            fractional_coords=fractional_coords,
            cartesian_coords=cartesian_coords,
        )

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = CrystalDescriptorCalculator(use_gpu=False)
        assert not calc.use_gpu
        assert calc.probe_radius == 1.4
        assert 6 in calc.vdw_radii  # Carbon

    def test_basic_descriptors_calculation(self):
        """Test basic descriptor calculations."""
        calc = CrystalDescriptorCalculator(use_gpu=False)
        crystal = self.create_simple_cubic_structure()

        basic_desc = calc._calculate_basic_descriptors(crystal)

        assert "volume" in basic_desc
        assert "density" in basic_desc
        assert "packing_efficiency" in basic_desc
        assert basic_desc["volume"] == 27.0
        assert basic_desc["packing_efficiency"] > 0

    def test_supercell_creation(self):
        """Test supercell creation."""
        calc = CrystalDescriptorCalculator(use_gpu=False)
        crystal = self.create_simple_cubic_structure()

        supercell = calc._create_supercell(crystal, (2, 2, 2))

        # Should have 8 times as many atoms
        assert len(supercell.atomic_numbers) == 8 * len(crystal.atomic_numbers)

        # Volume should be 8 times larger (with small tolerance for floating point)
        assert abs(supercell.volume - 8 * crystal.volume) < 1e-10

    def test_coordination_numbers(self):
        """Test coordination number calculation."""
        calc = CrystalDescriptorCalculator(use_gpu=False)
        crystal = self.create_simple_cubic_structure()

        coord_numbers = calc._calculate_coordination_numbers(crystal)

        assert len(coord_numbers) == len(crystal.atomic_numbers)
        assert all(cn >= 0 for cn in coord_numbers)

    def test_void_mask_calculation(self):
        """Test void mask calculation."""
        calc = CrystalDescriptorCalculator(use_gpu=False)

        # Simple test with grid points and atoms
        grid_points = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [1.0, 1.0, 1.0]])
        atom_coords = np.array([[0.0, 0.0, 0.0]])
        atomic_numbers = np.array([6])

        void_mask = calc._calculate_void_mask(grid_points, atom_coords, atomic_numbers)

        assert len(void_mask) == len(grid_points)
        assert isinstance(void_mask[0], (bool, np.bool_))
        # Point at atom location should not be void
        assert not void_mask[0]
        # Distant point should be void
        assert void_mask[1]

    def test_full_descriptor_calculation(self):
        """Test full descriptor calculation."""
        calc = CrystalDescriptorCalculator(use_gpu=False)
        crystal = self.create_simple_cubic_structure()

        # Use small supercell and coarse grid for fast testing
        descriptors = calc.calculate_descriptors(
            crystal, supercell_size=(1, 1, 1), grid_resolution=1.0
        )

        assert isinstance(descriptors, CrystalDescriptors)
        assert descriptors.volume > 0
        assert descriptors.density > 0
        assert 0 <= descriptors.void_fraction <= 1
        assert descriptors.packing_efficiency >= 0  # Can be > 1 for overlapping spheres
        assert len(descriptors.coordination_numbers) > 0

    def test_batch_calculation(self):
        """Test batch descriptor calculation."""
        calc = CrystalDescriptorCalculator(use_gpu=False)

        # Create multiple structures
        structures = [self.create_simple_cubic_structure() for _ in range(3)]

        df = calc.calculate_batch_descriptors(
            structures, supercell_size=(1, 1, 1), grid_resolution=1.0
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "crystal_id" in df.columns
        assert "volume" in df.columns

    def test_surface_area_calculation(self):
        """Test surface area calculations."""
        calc = CrystalDescriptorCalculator(use_gpu=False)
        crystal = self.create_simple_cubic_structure()

        surface_desc = calc._calculate_surface_descriptors(crystal, grid_resolution=1.0)

        assert "surface_area_geometric" in surface_desc
        assert "surface_area_accessible" in surface_desc
        assert "surface_area_volumetric" in surface_desc
        assert all(sa >= 0 for sa in surface_desc.values())

    def test_geometric_surface_area(self):
        """Test geometric surface area calculation."""
        calc = CrystalDescriptorCalculator(use_gpu=False)
        crystal = self.create_simple_cubic_structure()

        geometric_sa = calc._calculate_geometric_surface_area(crystal)

        # For a 3x3x3 cube, surface area should be 6 * 9 = 54
        expected_sa = 6 * 9  # 6 faces, each 3x3
        assert abs(geometric_sa - expected_sa) < 1e-10

    def test_error_handling(self):
        """Test error handling in calculations."""
        calc = CrystalDescriptorCalculator(use_gpu=False)

        # Create invalid structure (no atoms)
        invalid_crystal = CrystalStructure(
            lattice_vectors=np.eye(3),
            lattice_parameters=None,
            atomic_numbers=np.array([]),
            atomic_symbols=[],
            fractional_coords=np.array([]).reshape(0, 3),
            cartesian_coords=np.array([]).reshape(0, 3),
        )

        # Should handle gracefully
        df = calc.calculate_batch_descriptors([invalid_crystal])
        assert len(df) == 1
        assert "crystal_id" in df.columns

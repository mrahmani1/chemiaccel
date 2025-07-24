"""
Tests for crystal structure parsers.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from src.chemi_accel.materials.crystal_parser import (
    CIFParser, POSCARParser, CrystalStructure
)


class TestCrystalStructure:
    """Test CrystalStructure data class."""
    
    def test_crystal_structure_creation(self):
        """Test basic crystal structure creation."""
        lattice_vectors = np.array([
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        
        atomic_numbers = np.array([6, 6])
        atomic_symbols = ['C', 'C']
        fractional_coords = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ])
        cartesian_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 1.5, 1.5]
        ])
        
        crystal = CrystalStructure(
            lattice_vectors=lattice_vectors,
            lattice_parameters=None,
            atomic_numbers=atomic_numbers,
            atomic_symbols=atomic_symbols,
            fractional_coords=fractional_coords,
            cartesian_coords=cartesian_coords
        )
        
        assert crystal.volume == 27.0
        assert len(crystal.atomic_numbers) == 2
        assert crystal.lattice_parameters is not None
        assert len(crystal.lattice_parameters) == 6
    
    def test_crystal_structure_validation(self):
        """Test crystal structure validation."""
        # Invalid lattice vectors (not 3x3)
        with pytest.raises(ValueError, match="Lattice vectors must be a 3x3 matrix"):
            CrystalStructure(
                lattice_vectors=np.array([[1, 0], [0, 1]]),
                lattice_parameters=None,
                atomic_numbers=np.array([6]),
                atomic_symbols=['C'],
                fractional_coords=np.array([[0, 0, 0]]),
                cartesian_coords=np.array([[0, 0, 0]])
            )
    
    def test_density_calculation(self):
        """Test density calculation."""
        # Simple cubic carbon structure
        lattice_vectors = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        
        crystal = CrystalStructure(
            lattice_vectors=lattice_vectors,
            lattice_parameters=None,
            atomic_numbers=np.array([6]),
            atomic_symbols=['C'],
            fractional_coords=np.array([[0.0, 0.0, 0.0]]),
            cartesian_coords=np.array([[0.0, 0.0, 0.0]])
        )
        
        assert crystal.density > 0
        assert abs(crystal.volume - 8.0) < 1e-10


class TestCIFParser:
    """Test CIF file parser."""
    
    def create_test_cif(self, content: str) -> str:
        """Create a temporary CIF file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_basic_cif_parsing(self):
        """Test basic CIF file parsing."""
        cif_content = """
data_test
_cell_length_a    3.567
_cell_length_b    3.567
_cell_length_c    3.567
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 0.0 0.0 0.0
C2 0.25 0.25 0.25
"""
        
        cif_file = self.create_test_cif(cif_content)
        
        try:
            parser = CIFParser(use_pymatgen=False)  # Use basic parser
            structures = parser.parse_file(cif_file)
            
            assert len(structures) == 1
            structure = structures[0]
            
            assert len(structure.atomic_numbers) == 2
            assert all(z == 6 for z in structure.atomic_numbers)  # Carbon
            assert structure.volume > 0
            
        finally:
            os.unlink(cif_file)
    
    def test_cif_parser_file_not_found(self):
        """Test CIF parser with non-existent file."""
        parser = CIFParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse_file("nonexistent.cif")
    
    def test_invalid_cif_content(self):
        """Test CIF parser with invalid content."""
        cif_content = "invalid cif content"
        cif_file = self.create_test_cif(cif_content)
        
        try:
            parser = CIFParser(use_pymatgen=False)
            
            with pytest.raises(ValueError):
                parser.parse_file(cif_file)
                
        finally:
            os.unlink(cif_file)


class TestPOSCARParser:
    """Test POSCAR file parser."""
    
    def create_test_poscar(self, content: str) -> str:
        """Create a temporary POSCAR file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vasp', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_basic_poscar_parsing(self):
        """Test basic POSCAR file parsing."""
        poscar_content = """Test structure
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
C
2
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        poscar_file = self.create_test_poscar(poscar_content)
        
        try:
            parser = POSCARParser(use_pymatgen=False)
            structure = parser.parse_file(poscar_file)
            
            assert len(structure.atomic_numbers) == 2
            assert all(z == 6 for z in structure.atomic_numbers)  # Carbon
            assert structure.volume == 27.0
            
            # Check coordinates
            expected_cart = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
            np.testing.assert_allclose(structure.cartesian_coords, expected_cart)
            
        finally:
            os.unlink(poscar_file)
    
    def test_poscar_cartesian_coordinates(self):
        """Test POSCAR with Cartesian coordinates."""
        poscar_content = """Test structure
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
C
1
Cartesian
0.0 0.0 0.0
"""
        
        poscar_file = self.create_test_poscar(poscar_content)
        
        try:
            parser = POSCARParser(use_pymatgen=False)
            structure = parser.parse_file(poscar_file)
            
            assert len(structure.atomic_numbers) == 1
            assert structure.cartesian_coords[0][0] == 0.0
            
        finally:
            os.unlink(poscar_file)
    
    def test_poscar_parser_file_not_found(self):
        """Test POSCAR parser with non-existent file."""
        parser = POSCARParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse_file("nonexistent.vasp")
    
    def test_poscar_with_scaling(self):
        """Test POSCAR with scaling factor."""
        poscar_content = """Test structure
2.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
C
1
Direct
0.0 0.0 0.0
"""
        
        poscar_file = self.create_test_poscar(poscar_content)
        
        try:
            parser = POSCARParser(use_pymatgen=False)
            structure = parser.parse_file(poscar_file)
            
            # Lattice should be scaled by 2.0
            expected_lattice = np.array([
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0]
            ])
            np.testing.assert_allclose(structure.lattice_vectors, expected_lattice)
            assert abs(structure.volume - 8.0) < 1e-10
            
        finally:
            os.unlink(poscar_file)
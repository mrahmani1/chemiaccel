"""
Tests for materials calculator.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

from src.chemi_accel.materials.materials_calculator import MaterialsCalculator
from src.chemi_accel.materials.crystal_parser import CrystalStructure


class TestMaterialsCalculator:
    """Test materials calculator."""
    
    def create_test_cif(self, content: str) -> str:
        """Create a temporary CIF file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
            f.write(content)
            return f.name
    
    def create_test_poscar(self, content: str) -> str:
        """Create a temporary POSCAR file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vasp', delete=False) as f:
            f.write(content)
            return f.name
    
    def create_simple_crystal(self) -> CrystalStructure:
        """Create a simple crystal structure for testing."""
        lattice_vectors = np.array([
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        
        return CrystalStructure(
            lattice_vectors=lattice_vectors,
            lattice_parameters=None,
            atomic_numbers=np.array([6, 6]),
            atomic_symbols=['C', 'C'],
            fractional_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            cartesian_coords=np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
        )
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = MaterialsCalculator(use_gpu=False)
        
        assert not calc.use_gpu
        assert calc.cif_parser is not None
        assert calc.poscar_parser is not None
        assert calc.crystal_calculator is not None
        assert calc.periodic_calculator is not None
    
    def test_from_cif_file(self):
        """Test loading from CIF file."""
        cif_content = """
data_test
_cell_length_a    3.0
_cell_length_b    3.0
_cell_length_c    3.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 0.0 0.0 0.0
C2 0.5 0.5 0.5
"""
        
        cif_file = self.create_test_cif(cif_content)
        
        try:
            calc = MaterialsCalculator(use_gpu=False, use_pymatgen=False)
            
            # Test without descriptor calculation
            structures = calc.from_cif_file(cif_file, calculate_descriptors=False)
            assert isinstance(structures, list)
            assert len(structures) == 1
            assert isinstance(structures[0], CrystalStructure)
            
            # Test with descriptor calculation
            df = calc.from_cif_file(
                cif_file, 
                calculate_descriptors=True,
                supercell_size=(1, 1, 1),
                grid_resolution=1.0
            )
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            
        finally:
            os.unlink(cif_file)
    
    def test_from_poscar_file(self):
        """Test loading from POSCAR file."""
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
            calc = MaterialsCalculator(use_gpu=False, use_pymatgen=False)
            
            # Test without descriptor calculation
            structure = calc.from_poscar_file(poscar_file, calculate_descriptors=False)
            assert isinstance(structure, CrystalStructure)
            
            # Test with descriptor calculation
            df = calc.from_poscar_file(
                poscar_file,
                calculate_descriptors=True,
                supercell_size=(1, 1, 1),
                grid_resolution=1.0
            )
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            
        finally:
            os.unlink(poscar_file)
    
    def test_calculate_descriptors(self):
        """Test descriptor calculation."""
        calc = MaterialsCalculator(use_gpu=False)
        structures = [self.create_simple_crystal()]
        
        df = calc.calculate_descriptors(
            structures,
            supercell_size=(1, 1, 1),
            grid_resolution=1.0
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'structure_id' in df.columns
        
        # Check for crystal descriptors
        crystal_cols = [col for col in df.columns if col.startswith('crystal_')]
        assert len(crystal_cols) > 0
        
        # Check for periodic descriptors
        periodic_cols = [col for col in df.columns if col.startswith('periodic_')]
        assert len(periodic_cols) > 0
    
    def test_calculate_descriptors_options(self):
        """Test descriptor calculation with different options."""
        calc = MaterialsCalculator(use_gpu=False)
        structures = [self.create_simple_crystal()]
        
        # Only crystal descriptors
        df1 = calc.calculate_descriptors(
            structures,
            include_crystal_descriptors=True,
            include_periodic_descriptors=False,
            supercell_size=(1, 1, 1),
            grid_resolution=1.0
        )
        
        crystal_cols = [col for col in df1.columns if col.startswith('crystal_')]
        periodic_cols = [col for col in df1.columns if col.startswith('periodic_')]
        
        assert len(crystal_cols) > 0
        assert len(periodic_cols) == 0
        
        # Only periodic descriptors
        df2 = calc.calculate_descriptors(
            structures,
            include_crystal_descriptors=False,
            include_periodic_descriptors=True,
            supercell_size=(1, 1, 1)
        )
        
        crystal_cols = [col for col in df2.columns if col.startswith('crystal_')]
        periodic_cols = [col for col in df2.columns if col.startswith('periodic_')]
        
        assert len(crystal_cols) == 0
        assert len(periodic_cols) > 0
    
    def test_batch_process_directory(self):
        """Test batch processing of directory."""
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test CIF files
            cif_content = """
data_test
_cell_length_a    3.0
_cell_length_b    3.0
_cell_length_c    3.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 0.0 0.0 0.0
"""
            
            for i in range(3):
                cif_file = temp_path / f"test_{i}.cif"
                with open(cif_file, 'w') as f:
                    f.write(cif_content)
            
            calc = MaterialsCalculator(use_gpu=False, use_pymatgen=False)
            
            df = calc.batch_process_directory(
                temp_dir,
                file_pattern="*.cif",
                supercell_size=(1, 1, 1),
                grid_resolution=1.0
            )
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert 'filename' in df.columns
            assert 'filepath' in df.columns
            assert 'file_index' in df.columns
    
    def test_batch_process_empty_directory(self):
        """Test batch processing of empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            calc = MaterialsCalculator(use_gpu=False)
            
            df = calc.batch_process_directory(temp_dir, file_pattern="*.cif")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
    
    def test_batch_process_nonexistent_directory(self):
        """Test batch processing of non-existent directory."""
        calc = MaterialsCalculator(use_gpu=False)
        
        with pytest.raises(FileNotFoundError):
            calc.batch_process_directory("nonexistent_directory")
    
    def test_get_basic_structure_info(self):
        """Test getting basic structure information."""
        calc = MaterialsCalculator(use_gpu=False)
        crystal = self.create_simple_crystal()
        
        info = calc._get_basic_structure_info(crystal)
        
        assert 'n_atoms' in info
        assert 'volume' in info
        assert 'density' in info
        assert info['n_atoms'] == 2
        assert info['volume'] == 27.0
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        calc = MaterialsCalculator(use_gpu=False)
        
        formats = calc.get_supported_formats()
        
        assert isinstance(formats, list)
        assert 'CIF' in formats
        assert 'POSCAR/CONTCAR' in formats
    
    def test_validate_structure(self):
        """Test structure validation."""
        calc = MaterialsCalculator(use_gpu=False)
        crystal = self.create_simple_crystal()
        
        validation = calc.validate_structure(crystal)
        
        assert 'is_valid' in validation
        assert 'warnings' in validation
        assert 'errors' in validation
        assert isinstance(validation['is_valid'], bool)
        assert isinstance(validation['warnings'], list)
        assert isinstance(validation['errors'], list)
    
    def test_invalid_structure_validation(self):
        """Test validation of invalid structure."""
        calc = MaterialsCalculator(use_gpu=False)
        
        # Create invalid structure (zero volume)
        invalid_crystal = CrystalStructure(
            lattice_vectors=np.zeros((3, 3)),
            lattice_parameters=None,
            atomic_numbers=np.array([6]),
            atomic_symbols=['C'],
            fractional_coords=np.array([[0.0, 0.0, 0.0]]),
            cartesian_coords=np.array([[0.0, 0.0, 0.0]])
        )
        
        validation = calc.validate_structure(invalid_crystal)
        
        assert not validation['is_valid']
        assert len(validation['errors']) > 0
    
    def test_error_handling_invalid_file(self):
        """Test error handling with invalid files."""
        calc = MaterialsCalculator(use_gpu=False)
        
        with pytest.raises(ValueError, match="Failed to process"):
            calc.from_cif_file("nonexistent.cif")
        
        with pytest.raises(ValueError, match="Failed to process"):
            calc.from_poscar_file("nonexistent.vasp")
    
    def test_descriptor_calculation_error_handling(self):
        """Test error handling in descriptor calculations."""
        calc = MaterialsCalculator(use_gpu=False)
        
        # Create structure that might cause calculation issues
        problematic_crystal = CrystalStructure(
            lattice_vectors=np.array([
                [0.1, 0.0, 0.0],  # Very small lattice
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1]
            ]),
            lattice_parameters=None,
            atomic_numbers=np.array([6]),
            atomic_symbols=['C'],
            fractional_coords=np.array([[0.0, 0.0, 0.0]]),
            cartesian_coords=np.array([[0.0, 0.0, 0.0]])
        )
        
        # Should handle gracefully
        df = calc.calculate_descriptors([problematic_crystal])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
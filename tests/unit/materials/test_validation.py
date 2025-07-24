"""
Tests for materials validation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.chemi_accel.materials.validation import MaterialsValidator
from src.chemi_accel.materials.crystal_parser import CrystalStructure


class TestMaterialsValidator:
    """Test materials validator."""
    
    def create_diamond_structure(self) -> CrystalStructure:
        """Create a diamond-like structure for testing."""
        lattice_vectors = np.array([
            [3.567, 0.0, 0.0],
            [0.0, 3.567, 0.0],
            [0.0, 0.0, 3.567]
        ])
        
        return CrystalStructure(
            lattice_vectors=lattice_vectors,
            lattice_parameters=np.array([3.567, 3.567, 3.567, 90.0, 90.0, 90.0]),
            atomic_numbers=np.array([6, 6]),
            atomic_symbols=['C', 'C'],
            fractional_coords=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
            cartesian_coords=np.array([[0.0, 0.0, 0.0], [0.89175, 0.89175, 0.89175]]),
            formula='C2',
            density=3.515
        )
    
    def create_invalid_structure(self) -> CrystalStructure:
        """Create an invalid structure for testing."""
        return CrystalStructure(
            lattice_vectors=np.zeros((3, 3)),  # Invalid lattice
            lattice_parameters=None,
            atomic_numbers=np.array([]),  # No atoms
            atomic_symbols=[],
            fractional_coords=np.array([]).reshape(0, 3),
            cartesian_coords=np.array([]).reshape(0, 3)
        )
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = MaterialsValidator(tolerance=0.1)
        
        assert validator.tolerance == 0.1
        assert validator.calculator is not None
        assert len(validator.reference_data) > 0
        assert 'diamond' in validator.reference_data
    
    def test_reference_data_loading(self):
        """Test reference data loading."""
        validator = MaterialsValidator()
        
        # Check diamond reference data
        diamond_ref = validator.reference_data['diamond']
        assert diamond_ref['formula'] == 'C'
        assert diamond_ref['space_group'] == 'Fd-3m'
        assert diamond_ref['density'] == 3.515
        assert len(diamond_ref['lattice_parameters']) == 6
    
    def test_basic_structure_validation(self):
        """Test basic structure validation."""
        validator = MaterialsValidator()
        
        # Valid structure
        valid_structure = self.create_diamond_structure()
        validation = validator._validate_basic_structure(valid_structure)
        
        assert validation['structure_valid']
        assert len(validation['errors']) == 0
        
        # Invalid structure
        invalid_structure = self.create_invalid_structure()
        validation = validator._validate_basic_structure(invalid_structure)
        
        assert not validation['structure_valid']
        assert len(validation['errors']) > 0
    
    def test_min_distance_calculation(self):
        """Test minimum distance calculation."""
        validator = MaterialsValidator()
        
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        min_dist = validator._calculate_min_distance(coords)
        assert abs(min_dist - 1.0) < 1e-10
    
    def test_reference_comparison(self):
        """Test comparison with reference data."""
        validator = MaterialsValidator()
        diamond_structure = self.create_diamond_structure()
        
        comparison = validator._compare_with_reference(diamond_structure, 'diamond')
        
        assert 'reference_name' in comparison
        assert comparison['reference_name'] == 'diamond'
        assert 'matches' in comparison
        assert 'deviations' in comparison
        assert 'overall_match' in comparison
        
        # Check lattice parameter comparisons
        assert 'lattice_a' in comparison['deviations']
        assert 'reference' in comparison['deviations']['lattice_a']
        assert 'calculated' in comparison['deviations']['lattice_a']
        assert 'relative_error' in comparison['deviations']['lattice_a']
    
    def test_descriptor_validation(self):
        """Test descriptor validation."""
        validator = MaterialsValidator()
        structure = self.create_diamond_structure()
        
        # This will calculate descriptors and validate them
        validation = validator._validate_descriptors(structure)
        
        assert 'descriptors_valid' in validation
        assert 'descriptor_checks' in validation
        assert isinstance(validation['descriptors_valid'], bool)
    
    def test_full_structure_validation(self):
        """Test full structure validation."""
        validator = MaterialsValidator()
        structure = self.create_diamond_structure()
        
        validation = validator.validate_structure(structure, reference_name='diamond')
        
        assert 'structure_valid' in validation
        assert 'reference_comparison' in validation
        assert 'descriptor_validation' in validation
        assert 'warnings' in validation
        assert 'errors' in validation
        
        # Should be valid
        assert validation['structure_valid']
    
    def test_validation_without_reference(self):
        """Test validation without reference comparison."""
        validator = MaterialsValidator()
        structure = self.create_diamond_structure()
        
        validation = validator.validate_structure(structure)
        
        assert 'structure_valid' in validation
        assert validation['reference_comparison'] == {}
    
    def test_batch_validation(self):
        """Test batch validation."""
        validator = MaterialsValidator()
        
        structures = [
            self.create_diamond_structure(),
            self.create_diamond_structure(),
            self.create_invalid_structure()
        ]
        
        df = validator.validate_batch(structures)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'structure_id' in df.columns
        assert 'structure_valid' in df.columns
        assert 'descriptors_valid' in df.columns
        
        # First two should be valid, third should be invalid
        assert df.iloc[0]['structure_valid']
        assert df.iloc[1]['structure_valid']
        assert not df.iloc[2]['structure_valid']
    
    def test_batch_validation_with_references(self):
        """Test batch validation with reference names."""
        validator = MaterialsValidator()
        
        structures = [self.create_diamond_structure()]
        reference_names = ['diamond']
        
        df = validator.validate_batch(structures, reference_names)
        
        assert len(df) == 1
        assert 'reference_match' in df.columns
        assert df.iloc[0]['reference_match'] is not None
    
    def test_benchmark_against_database(self):
        """Test benchmarking against database."""
        validator = MaterialsValidator()
        
        # Create temporary directory with test CIF files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test CIF file
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
            
            cif_file = temp_path / "test.cif"
            with open(cif_file, 'w') as f:
                f.write(cif_content)
            
            # Run benchmark
            results = validator.benchmark_against_database(temp_dir, max_structures=1)
            
            assert 'n_structures' in results
            assert 'n_valid_structures' in results
            assert 'structure_success_rate' in results
            assert 'descriptor_success_rate' in results
            assert 'validation_details' in results
            
            assert results['n_structures'] >= 1
            assert isinstance(results['validation_details'], pd.DataFrame)
    
    def test_benchmark_nonexistent_database(self):
        """Test benchmark with non-existent database."""
        validator = MaterialsValidator()
        
        with pytest.raises(FileNotFoundError):
            validator.benchmark_against_database("nonexistent_directory")
    
    def test_validation_report_generation(self):
        """Test validation report generation."""
        validator = MaterialsValidator()
        
        # Create mock validation results
        validation_results = {
            'n_structures': 10,
            'n_valid_structures': 9,
            'structure_success_rate': 0.9,
            'descriptor_success_rate': 0.85,
            'average_warnings': 1.2,
            'average_errors': 0.1,
            'validation_details': pd.DataFrame({
                'structure_id': range(10),
                'structure_valid': [True] * 9 + [False],
                'n_errors': [0] * 9 + [1],
                'n_warnings': [1] * 10
            })
        }
        
        report = validator.generate_validation_report(validation_results)
        
        assert isinstance(report, str)
        assert 'ChemiAccel Materials Validation Report' in report
        assert 'Total structures: 10' in report
        assert 'Valid structures: 9' in report
    
    def test_validation_report_with_file_output(self):
        """Test validation report with file output."""
        validator = MaterialsValidator()
        
        validation_results = {
            'n_structures': 5,
            'n_valid_structures': 5,
            'structure_success_rate': 1.0,
            'descriptor_success_rate': 1.0,
            'average_warnings': 0.0,
            'average_errors': 0.0
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name
        
        try:
            report = validator.generate_validation_report(validation_results, output_path)
            
            # Check file was created
            assert Path(output_path).exists()
            
            # Check file content
            with open(output_path, 'r') as f:
                file_content = f.read()
            
            assert file_content == report
            
        finally:
            Path(output_path).unlink()
    
    def test_get_reference_materials(self):
        """Test getting reference materials."""
        validator = MaterialsValidator()
        
        materials = validator.get_reference_materials()
        
        assert isinstance(materials, list)
        assert 'diamond' in materials
        assert 'silicon' in materials
        assert 'quartz' in materials
    
    def test_add_reference_material(self):
        """Test adding reference material."""
        validator = MaterialsValidator()
        
        new_material = {
            'formula': 'NaCl',
            'lattice_parameters': [5.64, 5.64, 5.64, 90.0, 90.0, 90.0],
            'density': 2.165,
            'space_group': 'Fm-3m'
        }
        
        validator.add_reference_material('test_nacl', new_material)
        
        assert 'test_nacl' in validator.reference_data
        assert validator.reference_data['test_nacl']['formula'] == 'NaCl'
    
    def test_add_reference_material_missing_required(self):
        """Test adding reference material with missing required properties."""
        validator = MaterialsValidator()
        
        incomplete_material = {
            'formula': 'NaCl'
            # Missing lattice_parameters and density
        }
        
        with pytest.raises(ValueError, match="Missing required property"):
            validator.add_reference_material('incomplete', incomplete_material)
    
    def test_validation_metrics(self):
        """Test validation metrics configuration."""
        validator = MaterialsValidator()
        
        assert 'volume' in validator.validation_metrics
        assert 'density' in validator.validation_metrics
        assert 'lattice_parameters' in validator.validation_metrics
        
        # Check metric structure
        volume_metric = validator.validation_metrics['volume']
        assert 'tolerance' in volume_metric
        assert 'units' in volume_metric
        assert isinstance(volume_metric['tolerance'], float)
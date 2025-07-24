"""
Validation tests for materials science calculations against known databases.

These tests validate the accuracy of descriptor calculations by comparing
against reference implementations and known crystal structure databases.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import warnings

from src.chemi_accel.materials.materials_calculator import MaterialsCalculator
from src.chemi_accel.materials.validation import MaterialsValidator
from src.chemi_accel.materials.crystal_parser import CrystalStructure


class TestMaterialsValidation:
    """Validation tests for materials science calculations."""
    
    def create_reference_structures(self) -> dict:
        """Create reference crystal structures for validation."""
        structures = {}
        
        # Diamond structure
        diamond_lattice = np.array([
            [3.567, 0.0, 0.0],
            [0.0, 3.567, 0.0],
            [0.0, 0.0, 3.567]
        ])
        
        structures['diamond'] = CrystalStructure(
            lattice_vectors=diamond_lattice,
            lattice_parameters=np.array([3.567, 3.567, 3.567, 90.0, 90.0, 90.0]),
            atomic_numbers=np.array([6, 6, 6, 6, 6, 6, 6, 6]),
            atomic_symbols=['C'] * 8,
            fractional_coords=np.array([
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.0],
                [0.75, 0.75, 0.25],
                [0.5, 0.0, 0.5],
                [0.75, 0.25, 0.75],
                [0.0, 0.5, 0.5],
                [0.25, 0.75, 0.75]
            ]),
            cartesian_coords=np.array([
                [0.0, 0.0, 0.0],
                [0.89175, 0.89175, 0.89175],
                [1.7835, 1.7835, 0.0],
                [2.67525, 2.67525, 0.89175],
                [1.7835, 0.0, 1.7835],
                [2.67525, 0.89175, 2.67525],
                [0.0, 1.7835, 1.7835],
                [0.89175, 2.67525, 2.67525]
            ]),
            formula='C8',
            density=3.515,
            space_group='Fd-3m',
            space_group_number=227
        )
        
        # Silicon structure (same as diamond but with Si)
        structures['silicon'] = CrystalStructure(
            lattice_vectors=np.array([
                [5.431, 0.0, 0.0],
                [0.0, 5.431, 0.0],
                [0.0, 0.0, 5.431]
            ]),
            lattice_parameters=np.array([5.431, 5.431, 5.431, 90.0, 90.0, 90.0]),
            atomic_numbers=np.array([14, 14]),
            atomic_symbols=['Si', 'Si'],
            fractional_coords=np.array([
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25]
            ]),
            cartesian_coords=np.array([
                [0.0, 0.0, 0.0],
                [1.35775, 1.35775, 1.35775]
            ]),
            formula='Si2',
            density=2.329,
            space_group='Fd-3m',
            space_group_number=227
        )
        
        # NaCl structure
        structures['nacl'] = CrystalStructure(
            lattice_vectors=np.array([
                [5.640, 0.0, 0.0],
                [0.0, 5.640, 0.0],
                [0.0, 0.0, 5.640]
            ]),
            lattice_parameters=np.array([5.640, 5.640, 5.640, 90.0, 90.0, 90.0]),
            atomic_numbers=np.array([11, 17, 11, 17, 11, 17, 11, 17]),
            atomic_symbols=['Na', 'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Na', 'Cl'],
            fractional_coords=np.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            ]),
            cartesian_coords=np.array([
                [0.0, 0.0, 0.0],
                [2.82, 0.0, 0.0],
                [0.0, 2.82, 0.0],
                [2.82, 2.82, 0.0],
                [0.0, 0.0, 2.82],
                [2.82, 0.0, 2.82],
                [0.0, 2.82, 2.82],
                [2.82, 2.82, 2.82]
            ]),
            formula='Na4Cl4',
            density=2.165,
            space_group='Fm-3m',
            space_group_number=225
        )
        
        return structures
    
    def test_diamond_structure_validation(self):
        """Test validation against diamond reference structure."""
        validator = MaterialsValidator()
        structures = self.create_reference_structures()
        
        diamond = structures['diamond']
        validation = validator.validate_structure(diamond, reference_name='diamond')
        
        # Structure should be valid
        assert validation['structure_valid']
        assert len(validation['errors']) == 0
        
        # Reference comparison should show good match
        ref_comp = validation['reference_comparison']
        assert ref_comp['overall_match'] or len(validation['warnings']) == 0
        
        # Check specific parameter matches
        if 'deviations' in ref_comp:
            for param, dev_info in ref_comp['deviations'].items():
                if 'relative_error' in dev_info:
                    # Allow some tolerance for numerical differences
                    assert dev_info['relative_error'] < 0.1, f"Large error in {param}: {dev_info['relative_error']}"
    
    def test_silicon_structure_validation(self):
        """Test validation against silicon reference structure."""
        validator = MaterialsValidator()
        structures = self.create_reference_structures()
        
        silicon = structures['silicon']
        validation = validator.validate_structure(silicon, reference_name='silicon')
        
        assert validation['structure_valid']
        
        # Check lattice parameter validation
        ref_comp = validation['reference_comparison']
        if 'deviations' in ref_comp and 'lattice_a' in ref_comp['deviations']:
            lattice_error = ref_comp['deviations']['lattice_a']['relative_error']
            assert lattice_error < 0.01, f"Silicon lattice parameter error too large: {lattice_error}"
    
    def test_nacl_structure_validation(self):
        """Test validation against NaCl reference structure."""
        validator = MaterialsValidator()
        structures = self.create_reference_structures()
        
        nacl = structures['nacl']
        validation = validator.validate_structure(nacl, reference_name='nacl')
        
        assert validation['structure_valid']
        
        # NaCl should have coordination number 6
        desc_validation = validation.get('descriptor_validation', {})
        if 'descriptor_checks' in desc_validation:
            coord_check = desc_validation['descriptor_checks'].get('average_coordination')
            if coord_check:
                # Should be close to 6 for NaCl
                assert 5.0 <= coord_check['value'] <= 7.0
    
    def test_descriptor_accuracy_validation(self):
        """Test accuracy of descriptor calculations."""
        calculator = MaterialsCalculator(use_gpu=False)
        structures = self.create_reference_structures()
        
        # Calculate descriptors for all reference structures
        structure_list = list(structures.values())
        df = calculator.calculate_descriptors(
            structure_list,
            supercell_size=(1, 1, 1),  # Small for fast testing
            grid_resolution=1.0
        )
        
        assert len(df) == len(structures)
        
        # Check basic descriptor validity
        for idx, row in df.iterrows():
            # Volume should be positive
            assert row['volume'] > 0
            
            # Density should be reasonable
            if 'density' in row and pd.notna(row['density']):
                assert 0.1 < row['density'] < 25.0
            
            # Void fraction should be between 0 and 1
            if 'crystal_void_fraction' in row and pd.notna(row['crystal_void_fraction']):
                assert 0 <= row['crystal_void_fraction'] <= 1
            
            # Packing efficiency should be between 0 and 1
            if 'crystal_packing_efficiency' in row and pd.notna(row['crystal_packing_efficiency']):
                assert 0 <= row['crystal_packing_efficiency'] <= 1
            
            # Surface area should be non-negative
            if 'crystal_surface_area_accessible' in row and pd.notna(row['crystal_surface_area_accessible']):
                assert row['crystal_surface_area_accessible'] >= 0
    
    def test_coordination_number_validation(self):
        """Test coordination number calculations."""
        calculator = MaterialsCalculator(use_gpu=False)
        structures = self.create_reference_structures()
        
        # Test diamond structure (should have coordination 4)
        diamond_df = calculator.calculate_descriptors(
            [structures['diamond']],
            supercell_size=(1, 1, 1),
            grid_resolution=1.0
        )
        
        if 'crystal_average_coordination' in diamond_df.columns:
            avg_coord = diamond_df.iloc[0]['crystal_average_coordination']
            if pd.notna(avg_coord):
                # Diamond should have coordination close to 4
                assert 3.0 <= avg_coord <= 5.0, f"Diamond coordination number {avg_coord} not close to 4"
        
        # Test NaCl structure (should have coordination 6)
        nacl_df = calculator.calculate_descriptors(
            [structures['nacl']],
            supercell_size=(1, 1, 1),
            grid_resolution=1.0
        )
        
        if 'crystal_average_coordination' in nacl_df.columns:
            avg_coord = nacl_df.iloc[0]['crystal_average_coordination']
            if pd.notna(avg_coord):
                # NaCl should have coordination close to 6
                assert 4.0 <= avg_coord <= 8.0, f"NaCl coordination number {avg_coord} not close to 6"
    
    def test_density_consistency_validation(self):
        """Test density calculation consistency."""
        calculator = MaterialsCalculator(use_gpu=False)
        structures = self.create_reference_structures()
        
        for name, structure in structures.items():
            df = calculator.calculate_descriptors(
                [structure],
                supercell_size=(1, 1, 1),
                grid_resolution=1.0
            )
            
            if 'density' in df.columns and structure.density:
                calc_density = df.iloc[0]['density']
                ref_density = structure.density
                
                if pd.notna(calc_density):
                    rel_error = abs(calc_density - ref_density) / ref_density
                    assert rel_error < 0.1, f"Density error for {name}: {rel_error:.3f}"
    
    def test_batch_validation_performance(self):
        """Test batch validation performance and accuracy."""
        validator = MaterialsValidator()
        structures = self.create_reference_structures()
        
        structure_list = list(structures.values())
        reference_names = list(structures.keys())
        
        # Run batch validation
        df = validator.validate_batch(structure_list, reference_names)
        
        assert len(df) == len(structures)
        
        # All structures should be valid
        assert df['structure_valid'].all()
        
        # Most should have successful descriptor calculations
        desc_success_rate = df['descriptors_valid'].mean()
        assert desc_success_rate >= 0.8, f"Descriptor success rate too low: {desc_success_rate}"
        
        # Check reference matches
        if 'reference_match' in df.columns:
            # At least some should match their references
            match_rate = df['reference_match'].sum() / df['reference_match'].notna().sum()
            assert match_rate >= 0.5, f"Reference match rate too low: {match_rate}"
    
    def test_periodic_descriptor_validation(self):
        """Test periodic descriptor calculations."""
        calculator = MaterialsCalculator(use_gpu=False)
        structures = self.create_reference_structures()
        
        # Test with periodic descriptors enabled
        df = calculator.calculate_descriptors(
            [structures['diamond']],
            include_periodic_descriptors=True,
            supercell_size=(2, 2, 2),  # Larger supercell for periodic analysis
            rdf_max_distance=10.0,
            rdf_bins=50
        )
        
        # Check for periodic descriptor columns
        periodic_cols = [col for col in df.columns if col.startswith('periodic_')]
        assert len(periodic_cols) > 0, "No periodic descriptors calculated"
        
        # Check specific periodic descriptors
        row = df.iloc[0]
        
        if 'periodic_symmetry_index' in row:
            symmetry_idx = row['periodic_symmetry_index']
            if pd.notna(symmetry_idx):
                assert 0 <= symmetry_idx <= 1, f"Invalid symmetry index: {symmetry_idx}"
        
        if 'periodic_connectivity_index' in row:
            connectivity_idx = row['periodic_connectivity_index']
            if pd.notna(connectivity_idx):
                assert 0 <= connectivity_idx <= 1, f"Invalid connectivity index: {connectivity_idx}"
    
    def test_error_handling_validation(self):
        """Test error handling in validation calculations."""
        calculator = MaterialsCalculator(use_gpu=False)
        validator = MaterialsValidator()
        
        # Create problematic structure
        problematic_structure = CrystalStructure(
            lattice_vectors=np.array([
                [0.01, 0.0, 0.0],  # Very small lattice
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01]
            ]),
            lattice_parameters=None,
            atomic_numbers=np.array([6, 6]),
            atomic_symbols=['C', 'C'],
            fractional_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            cartesian_coords=np.array([[0.0, 0.0, 0.0], [0.005, 0.005, 0.005]])
        )
        
        # Should handle gracefully without crashing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            df = calculator.calculate_descriptors([problematic_structure])
            assert len(df) == 1
            
            validation = validator.validate_structure(problematic_structure)
            assert 'structure_valid' in validation
    
    def test_comprehensive_validation_report(self):
        """Test comprehensive validation report generation."""
        validator = MaterialsValidator()
        structures = self.create_reference_structures()
        
        # Run comprehensive validation
        structure_list = list(structures.values())
        reference_names = list(structures.keys())
        
        validation_df = validator.validate_batch(structure_list, reference_names)
        
        # Create benchmark results
        benchmark_results = {
            'n_structures': len(structure_list),
            'n_valid_structures': validation_df['structure_valid'].sum(),
            'n_valid_descriptors': validation_df['descriptors_valid'].sum(),
            'structure_success_rate': validation_df['structure_valid'].mean(),
            'descriptor_success_rate': validation_df['descriptors_valid'].mean(),
            'average_warnings': validation_df['n_warnings'].mean(),
            'average_errors': validation_df['n_errors'].mean(),
            'validation_details': validation_df
        }
        
        # Generate report
        report = validator.generate_validation_report(benchmark_results)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'ChemiAccel Materials Validation Report' in report
        assert f'Total structures: {len(structure_list)}' in report
        
        # Check that success rates are reasonable
        assert benchmark_results['structure_success_rate'] >= 0.8
        assert benchmark_results['descriptor_success_rate'] >= 0.6
    
    @pytest.mark.slow
    def test_large_structure_validation(self):
        """Test validation with larger structures (marked as slow test)."""
        calculator = MaterialsCalculator(use_gpu=False)
        
        # Create larger diamond supercell
        diamond_base = self.create_reference_structures()['diamond']
        
        # Calculate descriptors with larger supercell
        df = calculator.calculate_descriptors(
            [diamond_base],
            supercell_size=(3, 3, 3),  # Larger supercell
            grid_resolution=0.5,       # Finer grid
            rdf_max_distance=20.0,
            rdf_bins=200
        )
        
        assert len(df) == 1
        
        # Results should still be physically reasonable
        row = df.iloc[0]
        
        if 'crystal_void_fraction' in row and pd.notna(row['crystal_void_fraction']):
            # Diamond has known void fraction around 0.34
            void_fraction = row['crystal_void_fraction']
            assert 0.2 <= void_fraction <= 0.5, f"Diamond void fraction {void_fraction} outside expected range"
        
        if 'crystal_packing_efficiency' in row and pd.notna(row['crystal_packing_efficiency']):
            # Diamond has known packing efficiency around 0.34
            packing_eff = row['crystal_packing_efficiency']
            assert 0.2 <= packing_eff <= 0.5, f"Diamond packing efficiency {packing_eff} outside expected range"
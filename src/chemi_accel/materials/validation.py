"""
Validation module for materials science calculations.

This module provides validation against known crystal structure databases
and reference implementations to ensure accuracy of descriptor calculations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import json

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.local_env import CrystalNN
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

from .crystal_parser import CrystalStructure
from .materials_calculator import MaterialsCalculator


class MaterialsValidator:
    """Validator for materials science calculations against reference databases."""
    
    def __init__(self, tolerance: float = 0.1):
        """Initialize materials validator.
        
        Args:
            tolerance: Relative tolerance for validation comparisons
        """
        self.tolerance = tolerance
        self.calculator = MaterialsCalculator()
        
        # Reference data for common materials
        self.reference_data = self._load_reference_data()
        
        # Validation metrics
        self.validation_metrics = {
            'volume': {'tolerance': 0.05, 'units': 'Ų'},
            'density': {'tolerance': 0.1, 'units': 'g/cm³'},
            'lattice_parameters': {'tolerance': 0.02, 'units': 'Å'},
            'coordination_numbers': {'tolerance': 0.5, 'units': 'count'},
            'void_fraction': {'tolerance': 0.15, 'units': 'fraction'},
            'surface_area': {'tolerance': 0.2, 'units': 'm²/g'}
        }
    
    def _load_reference_data(self) -> Dict[str, Dict[str, Any]]:
        """Load reference data for common crystal structures."""
        # Reference data for common crystal structures
        reference_data = {
            'diamond': {
                'formula': 'C',
                'space_group': 'Fd-3m',
                'space_group_number': 227,
                'lattice_parameters': [3.567, 3.567, 3.567, 90.0, 90.0, 90.0],
                'density': 3.515,  # g/cm³
                'volume_per_atom': 5.67,  # Ų
                'coordination_number': 4,
                'void_fraction': 0.34,
                'crystal_system': 'cubic'
            },
            'silicon': {
                'formula': 'Si',
                'space_group': 'Fd-3m',
                'space_group_number': 227,
                'lattice_parameters': [5.431, 5.431, 5.431, 90.0, 90.0, 90.0],
                'density': 2.329,
                'volume_per_atom': 20.0,
                'coordination_number': 4,
                'void_fraction': 0.34,
                'crystal_system': 'cubic'
            },
            'quartz': {
                'formula': 'SiO2',
                'space_group': 'P3221',
                'space_group_number': 154,
                'lattice_parameters': [4.916, 4.916, 5.405, 90.0, 90.0, 120.0],
                'density': 2.648,
                'volume_per_atom': 37.8,
                'coordination_number': 4,  # Si coordination
                'void_fraction': 0.0,
                'crystal_system': 'hexagonal'
            },
            'nacl': {
                'formula': 'NaCl',
                'space_group': 'Fm-3m',
                'space_group_number': 225,
                'lattice_parameters': [5.640, 5.640, 5.640, 90.0, 90.0, 90.0],
                'density': 2.165,
                'volume_per_atom': 44.7,
                'coordination_number': 6,
                'void_fraction': 0.26,
                'crystal_system': 'cubic'
            },
            'ice': {
                'formula': 'H2O',
                'space_group': 'P63/mmc',
                'space_group_number': 194,
                'lattice_parameters': [4.518, 4.518, 7.355, 90.0, 90.0, 120.0],
                'density': 0.917,
                'volume_per_atom': 32.8,
                'coordination_number': 4,
                'void_fraction': 0.09,
                'crystal_system': 'hexagonal'
            }
        }
        
        return reference_data
    
    def validate_structure(self, structure: CrystalStructure,
                          reference_name: Optional[str] = None) -> Dict[str, Any]:
        """Validate crystal structure against reference data.
        
        Args:
            structure: Crystal structure to validate
            reference_name: Name of reference structure to compare against
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'structure_valid': True,
            'reference_comparison': {},
            'descriptor_validation': {},
            'warnings': [],
            'errors': []
        }
        
        # Basic structure validation
        basic_validation = self._validate_basic_structure(structure)
        validation_results.update(basic_validation)
        
        # Reference comparison if specified
        if reference_name and reference_name in self.reference_data:
            ref_comparison = self._compare_with_reference(structure, reference_name)
            validation_results['reference_comparison'] = ref_comparison
        
        # Descriptor validation
        descriptor_validation = self._validate_descriptors(structure)
        validation_results['descriptor_validation'] = descriptor_validation
        
        return validation_results
    
    def _validate_basic_structure(self, structure: CrystalStructure) -> Dict[str, Any]:
        """Validate basic structural properties."""
        results = {
            'structure_valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check for valid lattice
            if structure.volume <= 0:
                results['errors'].append("Invalid lattice volume")
                results['structure_valid'] = False
            
            # Check lattice vectors
            det = np.linalg.det(structure.lattice_vectors)
            if abs(det) < 1e-10:
                results['errors'].append("Degenerate lattice vectors")
                results['structure_valid'] = False
            
            # Check atomic positions
            if len(structure.atomic_numbers) == 0:
                results['errors'].append("No atoms in structure")
                results['structure_valid'] = False
            
            # Check for reasonable atomic distances
            coords = structure.cartesian_coords
            if len(coords) > 1:
                min_distance = self._calculate_min_distance(coords)
                if min_distance < 0.5:
                    results['warnings'].append(f"Very short interatomic distance: {min_distance:.3f} Å")
                elif min_distance > 10.0:
                    results['warnings'].append(f"Very long minimum distance: {min_distance:.3f} Å")
            
            # Check fractional coordinates
            frac_coords = structure.fractional_coords
            if np.any(frac_coords < -0.01) or np.any(frac_coords > 1.01):
                results['warnings'].append("Fractional coordinates outside unit cell")
            
            # Check density
            if structure.density and (structure.density < 0.1 or structure.density > 25.0):
                results['warnings'].append(f"Unusual density: {structure.density:.3f} g/cm³")
            
        except Exception as e:
            results['errors'].append(f"Structure validation error: {e}")
            results['structure_valid'] = False
        
        return results
    
    def _calculate_min_distance(self, coords: np.ndarray) -> float:
        """Calculate minimum interatomic distance."""
        min_dist = float('inf')
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                min_dist = min(min_dist, dist)
        return min_dist
    
    def _compare_with_reference(self, structure: CrystalStructure,
                               reference_name: str) -> Dict[str, Any]:
        """Compare structure with reference data."""
        ref_data = self.reference_data[reference_name]
        comparison = {
            'reference_name': reference_name,
            'matches': {},
            'deviations': {},
            'overall_match': True
        }
        
        try:
            # Compare lattice parameters
            if structure.lattice_parameters is not None:
                ref_params = np.array(ref_data['lattice_parameters'])
                struct_params = structure.lattice_parameters
                
                param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
                for i, (name, ref_val, struct_val) in enumerate(zip(param_names, ref_params, struct_params)):
                    rel_error = abs(struct_val - ref_val) / ref_val
                    tolerance = self.validation_metrics['lattice_parameters']['tolerance']
                    
                    comparison['matches'][f'lattice_{name}'] = rel_error < tolerance
                    comparison['deviations'][f'lattice_{name}'] = {
                        'reference': ref_val,
                        'calculated': struct_val,
                        'relative_error': rel_error,
                        'within_tolerance': rel_error < tolerance
                    }
                    
                    if rel_error >= tolerance:
                        comparison['overall_match'] = False
            
            # Compare density
            if structure.density and 'density' in ref_data:
                ref_density = ref_data['density']
                rel_error = abs(structure.density - ref_density) / ref_density
                tolerance = self.validation_metrics['density']['tolerance']
                
                comparison['matches']['density'] = rel_error < tolerance
                comparison['deviations']['density'] = {
                    'reference': ref_density,
                    'calculated': structure.density,
                    'relative_error': rel_error,
                    'within_tolerance': rel_error < tolerance
                }
                
                if rel_error >= tolerance:
                    comparison['overall_match'] = False
            
            # Compare volume per atom
            if 'volume_per_atom' in ref_data:
                volume_per_atom = structure.volume / len(structure.atomic_numbers)
                ref_volume = ref_data['volume_per_atom']
                rel_error = abs(volume_per_atom - ref_volume) / ref_volume
                tolerance = self.validation_metrics['volume']['tolerance']
                
                comparison['matches']['volume_per_atom'] = rel_error < tolerance
                comparison['deviations']['volume_per_atom'] = {
                    'reference': ref_volume,
                    'calculated': volume_per_atom,
                    'relative_error': rel_error,
                    'within_tolerance': rel_error < tolerance
                }
                
                if rel_error >= tolerance:
                    comparison['overall_match'] = False
            
        except Exception as e:
            comparison['error'] = str(e)
            comparison['overall_match'] = False
        
        return comparison
    
    def _validate_descriptors(self, structure: CrystalStructure) -> Dict[str, Any]:
        """Validate calculated descriptors for physical reasonableness."""
        validation = {
            'descriptors_valid': True,
            'descriptor_checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Calculate descriptors
            df = self.calculator.calculate_descriptors([structure])
            
            if len(df) == 0:
                validation['errors'].append("Failed to calculate descriptors")
                validation['descriptors_valid'] = False
                return validation
            
            row = df.iloc[0]
            
            # Validate void fraction
            if 'crystal_void_fraction' in row:
                void_fraction = row['crystal_void_fraction']
                if void_fraction < 0 or void_fraction > 1:
                    validation['errors'].append(f"Invalid void fraction: {void_fraction}")
                    validation['descriptors_valid'] = False
                elif void_fraction > 0.8:
                    validation['warnings'].append(f"Very high void fraction: {void_fraction:.3f}")
                
                validation['descriptor_checks']['void_fraction'] = {
                    'value': void_fraction,
                    'valid_range': [0, 1],
                    'is_valid': 0 <= void_fraction <= 1
                }
            
            # Validate packing efficiency
            if 'crystal_packing_efficiency' in row:
                packing_eff = row['crystal_packing_efficiency']
                if packing_eff < 0 or packing_eff > 1:
                    validation['errors'].append(f"Invalid packing efficiency: {packing_eff}")
                    validation['descriptors_valid'] = False
                elif packing_eff > 0.9:
                    validation['warnings'].append(f"Very high packing efficiency: {packing_eff:.3f}")
                
                validation['descriptor_checks']['packing_efficiency'] = {
                    'value': packing_eff,
                    'valid_range': [0, 1],
                    'is_valid': 0 <= packing_eff <= 1
                }
            
            # Validate surface area
            if 'crystal_surface_area_accessible' in row:
                surface_area = row['crystal_surface_area_accessible']
                if surface_area < 0:
                    validation['errors'].append(f"Negative surface area: {surface_area}")
                    validation['descriptors_valid'] = False
                
                validation['descriptor_checks']['surface_area'] = {
                    'value': surface_area,
                    'valid_range': [0, float('inf')],
                    'is_valid': surface_area >= 0
                }
            
            # Validate coordination numbers
            if 'crystal_average_coordination' in row:
                avg_coord = row['crystal_average_coordination']
                if avg_coord < 0 or avg_coord > 12:
                    validation['warnings'].append(f"Unusual average coordination: {avg_coord:.2f}")
                
                validation['descriptor_checks']['average_coordination'] = {
                    'value': avg_coord,
                    'typical_range': [2, 12],
                    'is_typical': 2 <= avg_coord <= 12
                }
            
            # Validate density consistency
            if 'density' in row and structure.density:
                calc_density = row['density']
                struct_density = structure.density
                rel_error = abs(calc_density - struct_density) / struct_density
                
                if rel_error > 0.01:  # 1% tolerance
                    validation['warnings'].append(f"Density inconsistency: {rel_error:.3f}")
                
                validation['descriptor_checks']['density_consistency'] = {
                    'structure_density': struct_density,
                    'calculated_density': calc_density,
                    'relative_error': rel_error,
                    'consistent': rel_error <= 0.01
                }
            
        except Exception as e:
            validation['errors'].append(f"Descriptor validation error: {e}")
            validation['descriptors_valid'] = False
        
        return validation
    
    def validate_batch(self, structures: List[CrystalStructure],
                      reference_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Validate multiple crystal structures.
        
        Args:
            structures: List of crystal structures to validate
            reference_names: Optional list of reference names for comparison
            
        Returns:
            DataFrame with validation results
        """
        results = []
        
        for i, structure in enumerate(structures):
            ref_name = reference_names[i] if reference_names and i < len(reference_names) else None
            
            try:
                validation = self.validate_structure(structure, ref_name)
                
                result = {
                    'structure_id': i,
                    'structure_valid': validation['structure_valid'],
                    'descriptors_valid': validation.get('descriptor_validation', {}).get('descriptors_valid', True),
                    'n_warnings': len(validation['warnings']),
                    'n_errors': len(validation['errors']),
                    'reference_match': validation.get('reference_comparison', {}).get('overall_match', None)
                }
                
                # Add specific validation metrics
                if 'reference_comparison' in validation:
                    ref_comp = validation['reference_comparison']
                    if 'deviations' in ref_comp:
                        for param, dev_info in ref_comp['deviations'].items():
                            result[f'{param}_error'] = dev_info.get('relative_error', None)
                            result[f'{param}_within_tolerance'] = dev_info.get('within_tolerance', None)
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'structure_id': i,
                    'structure_valid': False,
                    'descriptors_valid': False,
                    'n_warnings': 0,
                    'n_errors': 1,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def benchmark_against_database(self, database_path: Union[str, Path],
                                  max_structures: Optional[int] = None) -> Dict[str, Any]:
        """Benchmark calculations against a crystal structure database.
        
        Args:
            database_path: Path to database directory or file
            max_structures: Maximum number of structures to test
            
        Returns:
            Dictionary with benchmark results
        """
        database_path = Path(database_path)
        
        if not database_path.exists():
            raise FileNotFoundError(f"Database path not found: {database_path}")
        
        # Process database files
        if database_path.is_dir():
            # Directory of CIF files
            cif_files = list(database_path.glob("*.cif"))
            if max_structures:
                cif_files = cif_files[:max_structures]
            
            structures = []
            for cif_file in cif_files:
                try:
                    parsed_structures = self.calculator.cif_parser.parse_file(cif_file)
                    structures.extend(parsed_structures)
                except Exception as e:
                    warnings.warn(f"Failed to parse {cif_file}: {e}")
        else:
            # Single file
            try:
                structures = self.calculator.cif_parser.parse_file(database_path)
            except Exception as e:
                raise ValueError(f"Failed to parse database file: {e}")
        
        if max_structures:
            structures = structures[:max_structures]
        
        # Run validation
        validation_df = self.validate_batch(structures)
        
        # Calculate benchmark statistics
        if len(validation_df) > 0:
            benchmark_results = {
                'n_structures': len(structures),
                'n_valid_structures': validation_df['structure_valid'].sum(),
                'n_valid_descriptors': validation_df['descriptors_valid'].sum(),
                'structure_success_rate': validation_df['structure_valid'].mean(),
                'descriptor_success_rate': validation_df['descriptors_valid'].mean(),
                'average_warnings': validation_df['n_warnings'].mean(),
                'average_errors': validation_df['n_errors'].mean(),
                'validation_details': validation_df
            }
        else:
            benchmark_results = {
                'n_structures': len(structures),
                'n_valid_structures': 0,
                'n_valid_descriptors': 0,
                'structure_success_rate': 0.0,
                'descriptor_success_rate': 0.0,
                'average_warnings': 0.0,
                'average_errors': 0.0,
                'validation_details': validation_df
            }
        
        return benchmark_results
    
    def generate_validation_report(self, validation_results: Dict[str, Any],
                                 output_path: Optional[Union[str, Path]] = None) -> str:
        """Generate a comprehensive validation report.
        
        Args:
            validation_results: Results from validation methods
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("ChemiAccel Materials Validation Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Summary statistics
        if 'n_structures' in validation_results:
            report_lines.append("Summary Statistics:")
            report_lines.append(f"  Total structures: {validation_results['n_structures']}")
            report_lines.append(f"  Valid structures: {validation_results['n_valid_structures']}")
            report_lines.append(f"  Structure success rate: {validation_results['structure_success_rate']:.2%}")
            report_lines.append(f"  Descriptor success rate: {validation_results['descriptor_success_rate']:.2%}")
            report_lines.append(f"  Average warnings per structure: {validation_results['average_warnings']:.2f}")
            report_lines.append(f"  Average errors per structure: {validation_results['average_errors']:.2f}")
            report_lines.append("")
        
        # Detailed validation results
        if 'validation_details' in validation_results:
            df = validation_results['validation_details']
            
            report_lines.append("Detailed Results:")
            report_lines.append(f"  Structures with errors: {(df['n_errors'] > 0).sum()}")
            report_lines.append(f"  Structures with warnings: {(df['n_warnings'] > 0).sum()}")
            
            if 'reference_match' in df.columns:
                matches = df['reference_match'].notna()
                if matches.any():
                    match_rate = df.loc[matches, 'reference_match'].mean()
                    report_lines.append(f"  Reference match rate: {match_rate:.2%}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.append("Recommendations:")
        if validation_results.get('structure_success_rate', 1.0) < 0.95:
            report_lines.append("  - Review structure parsing and validation logic")
        if validation_results.get('descriptor_success_rate', 1.0) < 0.90:
            report_lines.append("  - Check descriptor calculation algorithms")
        if validation_results.get('average_errors', 0) > 0.1:
            report_lines.append("  - Investigate common error sources")
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def get_reference_materials(self) -> List[str]:
        """Get list of available reference materials."""
        return list(self.reference_data.keys())
    
    def add_reference_material(self, name: str, properties: Dict[str, Any]):
        """Add a new reference material.
        
        Args:
            name: Name of the reference material
            properties: Dictionary of reference properties
        """
        required_keys = ['formula', 'lattice_parameters', 'density']
        for key in required_keys:
            if key not in properties:
                raise ValueError(f"Missing required property: {key}")
        
        self.reference_data[name] = properties
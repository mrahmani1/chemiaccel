"""
Main materials calculator interface integrating all materials science functionality.

This module provides a unified interface for crystalline materials analysis,
combining CIF/POSCAR parsing, descriptor calculations, and integration with
materials science workflows.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False

from .crystal_parser import CIFParser, POSCARParser, CrystalStructure
from .crystal_descriptors import CrystalDescriptorCalculator, CrystalDescriptors
from .periodic_descriptors import PeriodicDescriptorCalculator, PeriodicDescriptors


class MaterialsCalculator:
    """Unified calculator for crystalline materials analysis."""
    
    def __init__(self, 
                 use_gpu: bool = True,
                 use_pymatgen: bool = True,
                 probe_radius: float = 1.4,
                 max_coordination_shells: int = 10):
        """Initialize materials calculator.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            use_pymatgen: Whether to use pymatgen for parsing
            probe_radius: Probe radius for accessibility calculations (Å)
            max_coordination_shells: Maximum coordination shells to analyze
        """
        self.use_gpu = use_gpu
        self.use_pymatgen = use_pymatgen and PYMATGEN_AVAILABLE
        
        # Initialize parsers
        self.cif_parser = CIFParser(use_pymatgen=self.use_pymatgen)
        self.poscar_parser = POSCARParser(use_pymatgen=self.use_pymatgen)
        
        # Initialize descriptor calculators
        self.crystal_calculator = CrystalDescriptorCalculator(
            use_gpu=use_gpu, probe_radius=probe_radius
        )
        self.periodic_calculator = PeriodicDescriptorCalculator(
            use_gpu=use_gpu, max_shells=max_coordination_shells
        )
        
        # Configuration
        self.default_supercell_size = (2, 2, 2)
        self.default_grid_resolution = 0.2
        self.default_rdf_max_distance = 15.0
        self.default_rdf_bins = 150
    
    def from_cif_file(self, filepath: Union[str, Path],
                     calculate_descriptors: bool = True,
                     **kwargs) -> Union[List[CrystalStructure], pd.DataFrame]:
        """Load and analyze crystal structure from CIF file.
        
        Args:
            filepath: Path to CIF file
            calculate_descriptors: Whether to calculate descriptors
            **kwargs: Additional arguments for descriptor calculation
            
        Returns:
            List of CrystalStructure objects or DataFrame with descriptors
        """
        try:
            structures = self.cif_parser.parse_file(filepath)
            
            if not calculate_descriptors:
                return structures
            
            return self.calculate_descriptors(structures, **kwargs)
            
        except Exception as e:
            raise ValueError(f"Failed to process CIF file {filepath}: {e}")
    
    def from_poscar_file(self, filepath: Union[str, Path],
                        calculate_descriptors: bool = True,
                        **kwargs) -> Union[CrystalStructure, pd.DataFrame]:
        """Load and analyze crystal structure from POSCAR file.
        
        Args:
            filepath: Path to POSCAR file
            calculate_descriptors: Whether to calculate descriptors
            **kwargs: Additional arguments for descriptor calculation
            
        Returns:
            CrystalStructure object or DataFrame with descriptors
        """
        try:
            structure = self.poscar_parser.parse_file(filepath)
            
            if not calculate_descriptors:
                return structure
            
            return self.calculate_descriptors([structure], **kwargs)
            
        except Exception as e:
            raise ValueError(f"Failed to process POSCAR file {filepath}: {e}")
    
    def from_pymatgen_structure(self, structure: 'Structure',
                              calculate_descriptors: bool = True,
                              **kwargs) -> Union[CrystalStructure, pd.DataFrame]:
        """Convert pymatgen Structure and optionally calculate descriptors.
        
        Args:
            structure: Pymatgen Structure object
            calculate_descriptors: Whether to calculate descriptors
            **kwargs: Additional arguments for descriptor calculation
            
        Returns:
            CrystalStructure object or DataFrame with descriptors
        """
        if not PYMATGEN_AVAILABLE:
            raise ImportError("pymatgen is required for this functionality")
        
        try:
            crystal_struct = self._pymatgen_to_crystal_structure(structure)
            
            if not calculate_descriptors:
                return crystal_struct
            
            return self.calculate_descriptors([crystal_struct], **kwargs)
            
        except Exception as e:
            raise ValueError(f"Failed to process pymatgen structure: {e}")
    
    def from_ase_atoms(self, atoms: 'Atoms',
                      calculate_descriptors: bool = True,
                      **kwargs) -> Union[CrystalStructure, pd.DataFrame]:
        """Convert ASE Atoms object and optionally calculate descriptors.
        
        Args:
            atoms: ASE Atoms object
            calculate_descriptors: Whether to calculate descriptors
            **kwargs: Additional arguments for descriptor calculation
            
        Returns:
            CrystalStructure object or DataFrame with descriptors
        """
        if not ASE_AVAILABLE:
            raise ImportError("ASE is required for this functionality")
        
        try:
            crystal_struct = self._ase_to_crystal_structure(atoms)
            
            if not calculate_descriptors:
                return crystal_struct
            
            return self.calculate_descriptors([crystal_struct], **kwargs)
            
        except Exception as e:
            raise ValueError(f"Failed to process ASE atoms: {e}")
    
    def calculate_descriptors(self, structures: List[CrystalStructure],
                            include_crystal_descriptors: bool = True,
                            include_periodic_descriptors: bool = True,
                            supercell_size: Optional[Tuple[int, int, int]] = None,
                            grid_resolution: Optional[float] = None,
                            rdf_max_distance: Optional[float] = None,
                            rdf_bins: Optional[int] = None,
                            return_cudf: bool = False) -> pd.DataFrame:
        """Calculate comprehensive descriptors for crystal structures.
        
        Args:
            structures: List of CrystalStructure objects
            include_crystal_descriptors: Whether to include crystal descriptors
            include_periodic_descriptors: Whether to include periodic descriptors
            supercell_size: Size of supercell for calculations
            grid_resolution: Grid resolution for void calculations
            rdf_max_distance: Maximum distance for RDF calculation
            rdf_bins: Number of bins for RDF
            return_cudf: Whether to return cudf DataFrame (if available)
            
        Returns:
            DataFrame with descriptor results
        """
        # Set defaults
        supercell_size = supercell_size or self.default_supercell_size
        grid_resolution = grid_resolution or self.default_grid_resolution
        rdf_max_distance = rdf_max_distance or self.default_rdf_max_distance
        rdf_bins = rdf_bins or self.default_rdf_bins
        
        results = []
        
        for i, structure in enumerate(structures):
            try:
                result_dict = {'structure_id': i}
                
                # Add basic structure information
                result_dict.update(self._get_basic_structure_info(structure))
                
                # Calculate crystal descriptors
                if include_crystal_descriptors:
                    crystal_desc = self.crystal_calculator.calculate_descriptors(
                        structure, supercell_size, grid_resolution
                    )
                    crystal_dict = crystal_desc.to_dict()
                    # Prefix keys to avoid conflicts
                    crystal_dict = {f"crystal_{k}": v for k, v in crystal_dict.items()}
                    result_dict.update(crystal_dict)
                
                # Calculate periodic descriptors
                if include_periodic_descriptors:
                    periodic_desc = self.periodic_calculator.calculate_descriptors(
                        structure, supercell_size, rdf_max_distance, rdf_bins
                    )
                    periodic_dict = periodic_desc.to_dict()
                    # Prefix keys to avoid conflicts
                    periodic_dict = {f"periodic_{k}": v for k, v in periodic_dict.items()}
                    result_dict.update(periodic_dict)
                
                results.append(result_dict)
                
            except Exception as e:
                warnings.warn(f"Failed to calculate descriptors for structure {i}: {e}")
                results.append({'structure_id': i, 'error': str(e)})
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Convert to cudf if requested and available
        if return_cudf and CUDF_AVAILABLE:
            try:
                df = cudf.from_pandas(df)
            except Exception as e:
                warnings.warn(f"Failed to convert to cudf: {e}")
        
        return df
    
    def batch_process_directory(self, directory: Union[str, Path],
                              file_pattern: str = "*.cif",
                              max_files: Optional[int] = None,
                              **kwargs) -> pd.DataFrame:
        """Process all crystal structure files in a directory.
        
        Args:
            directory: Directory containing crystal structure files
            file_pattern: File pattern to match (e.g., "*.cif", "*.vasp")
            max_files: Maximum number of files to process
            **kwargs: Additional arguments for descriptor calculation
            
        Returns:
            DataFrame with results for all processed files
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find matching files
        files = list(directory.glob(file_pattern))
        if max_files:
            files = files[:max_files]
        
        if not files:
            warnings.warn(f"No files matching pattern '{file_pattern}' found in {directory}")
            return pd.DataFrame()
        
        all_results = []
        
        for i, filepath in enumerate(files):
            try:
                # Determine file type and parse
                if filepath.suffix.lower() in ['.cif']:
                    structures = self.cif_parser.parse_file(filepath)
                elif filepath.suffix.lower() in ['.vasp', '.poscar', '.contcar'] or filepath.name.upper() in ['POSCAR', 'CONTCAR']:
                    structures = [self.poscar_parser.parse_file(filepath)]
                else:
                    warnings.warn(f"Unknown file type: {filepath}")
                    continue
                
                # Calculate descriptors
                df = self.calculate_descriptors(structures, **kwargs)
                
                # Add file information
                df['filename'] = filepath.name
                df['filepath'] = str(filepath)
                df['file_index'] = i
                
                all_results.append(df)
                
            except Exception as e:
                warnings.warn(f"Failed to process file {filepath}: {e}")
                # Add error record
                error_df = pd.DataFrame([{
                    'filename': filepath.name,
                    'filepath': str(filepath),
                    'file_index': i,
                    'error': str(e)
                }])
                all_results.append(error_df)
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _pymatgen_to_crystal_structure(self, structure: 'Structure') -> CrystalStructure:
        """Convert pymatgen Structure to CrystalStructure."""
        lattice = structure.lattice
        
        # Get space group information if available
        try:
            sga = SpacegroupAnalyzer(structure)
            space_group_number = sga.get_space_group_number()
            space_group_symbol = sga.get_space_group_symbol()
        except:
            space_group_number = None
            space_group_symbol = None
        
        return CrystalStructure(
            lattice_vectors=lattice.matrix,
            lattice_parameters=np.array([
                lattice.a, lattice.b, lattice.c,
                lattice.alpha, lattice.beta, lattice.gamma
            ]),
            atomic_numbers=np.array([site.specie.Z for site in structure]),
            atomic_symbols=[str(site.specie) for site in structure],
            fractional_coords=np.array([site.frac_coords for site in structure]),
            cartesian_coords=np.array([site.coords for site in structure]),
            space_group=space_group_symbol,
            space_group_number=space_group_number,
            formula=structure.formula,
            volume=structure.volume,
            density=structure.density
        )
    
    def _ase_to_crystal_structure(self, atoms: 'Atoms') -> CrystalStructure:
        """Convert ASE Atoms to CrystalStructure."""
        cell = atoms.get_cell()
        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()
        symbols = atoms.get_chemical_symbols()
        
        # Convert positions to fractional coordinates
        if np.any(cell):
            fractional_coords = np.linalg.solve(cell.T, positions.T).T
        else:
            fractional_coords = positions
        
        return CrystalStructure(
            lattice_vectors=cell,
            lattice_parameters=None,  # Will be computed
            atomic_numbers=numbers,
            atomic_symbols=symbols,
            fractional_coords=fractional_coords,
            cartesian_coords=positions,
            formula=atoms.get_chemical_formula(),
            volume=atoms.get_volume() if np.any(cell) else 0.0
        )
    
    def _get_basic_structure_info(self, structure: CrystalStructure) -> Dict[str, Any]:
        """Get basic structure information."""
        return {
            'formula': structure.formula,
            'n_atoms': len(structure.atomic_numbers),
            'volume': structure.volume,
            'density': structure.density,
            'space_group': structure.space_group,
            'space_group_number': structure.space_group_number,
            'lattice_a': structure.lattice_parameters[0] if structure.lattice_parameters is not None else None,
            'lattice_b': structure.lattice_parameters[1] if structure.lattice_parameters is not None else None,
            'lattice_c': structure.lattice_parameters[2] if structure.lattice_parameters is not None else None,
            'lattice_alpha': structure.lattice_parameters[3] if structure.lattice_parameters is not None else None,
            'lattice_beta': structure.lattice_parameters[4] if structure.lattice_parameters is not None else None,
            'lattice_gamma': structure.lattice_parameters[5] if structure.lattice_parameters is not None else None,
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        formats = ['CIF']
        formats.append('POSCAR/CONTCAR')
        
        if PYMATGEN_AVAILABLE:
            formats.extend(['pymatgen Structure'])
        
        if ASE_AVAILABLE:
            formats.extend(['ASE Atoms'])
        
        return formats
    
    def validate_structure(self, structure: CrystalStructure) -> Dict[str, Any]:
        """Validate crystal structure and return validation results.
        
        Args:
            structure: Crystal structure to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check basic structure validity
            if len(structure.atomic_numbers) == 0:
                validation_results['errors'].append("No atoms in structure")
                validation_results['is_valid'] = False
            
            if structure.volume <= 0:
                validation_results['errors'].append("Invalid volume")
                validation_results['is_valid'] = False
            
            # Check lattice vectors
            if np.linalg.det(structure.lattice_vectors) <= 0:
                validation_results['errors'].append("Invalid lattice vectors")
                validation_results['is_valid'] = False
            
            # Check coordinate ranges
            frac_coords = structure.fractional_coords
            if np.any(frac_coords < -0.1) or np.any(frac_coords > 1.1):
                validation_results['warnings'].append("Fractional coordinates outside [0,1] range")
            
            # Check for overlapping atoms
            coords = structure.cartesian_coords
            min_distance = float('inf')
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    min_distance = min(min_distance, dist)
            
            if min_distance < 0.5:  # Less than 0.5 Angstrom
                validation_results['warnings'].append(f"Very close atoms detected (min distance: {min_distance:.3f} Å)")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {e}")
            validation_results['is_valid'] = False
        
        return validation_results
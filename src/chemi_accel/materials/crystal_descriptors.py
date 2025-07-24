"""
GPU-accelerated geometric and topological descriptor calculations for crystals.

This module provides comprehensive descriptor calculations for crystalline materials,
including void fraction, pore limiting diameter, surface area, and other geometric
and topological properties.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

try:
    from scipy.spatial import Voronoi, ConvexHull
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some functionality will be limited.")

from .crystal_parser import CrystalStructure


@dataclass
class CrystalDescriptors:
    """Container for crystal descriptor results."""
    
    # Basic geometric descriptors
    volume: float
    density: float
    packing_efficiency: float
    
    # Void and pore descriptors
    void_fraction: float
    accessible_void_fraction: float
    pore_limiting_diameter: float
    largest_cavity_diameter: float
    
    # Surface area descriptors
    surface_area_geometric: float
    surface_area_accessible: float
    surface_area_volumetric: float  # m²/g
    
    # Topological descriptors
    coordination_numbers: np.ndarray
    average_coordination: float
    coordination_variance: float
    
    # Structural descriptors
    lattice_parameters: np.ndarray
    lattice_angles: np.ndarray
    lattice_volume: float
    
    # Symmetry descriptors
    space_group_number: Optional[int] = None
    crystal_system: Optional[str] = None
    point_group: Optional[str] = None
    
    # Additional properties
    framework_density: Optional[float] = None
    gravimetric_surface_area: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, np.ndarray):
                if field_value.size == 1:
                    result[field_name] = float(field_value)
                else:
                    # For arrays, add statistics
                    result[f"{field_name}_mean"] = float(np.mean(field_value))
                    result[f"{field_name}_std"] = float(np.std(field_value))
                    result[f"{field_name}_min"] = float(np.min(field_value))
                    result[f"{field_name}_max"] = float(np.max(field_value))
            else:
                result[field_name] = field_value
        return result


class CrystalDescriptorCalculator:
    """GPU-accelerated calculator for crystal descriptors."""
    
    def __init__(self, use_gpu: bool = True, probe_radius: float = 1.4):
        """Initialize crystal descriptor calculator.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            probe_radius: Probe radius for accessibility calculations (Å)
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.probe_radius = probe_radius
        
        # Van der Waals radii (Å)
        self.vdw_radii = {
            1: 1.20,   # H
            6: 1.70,   # C
            7: 1.55,   # N
            8: 1.52,   # O
            9: 1.47,   # F
            11: 2.27,  # Na
            12: 1.73,  # Mg
            13: 1.84,  # Al
            14: 2.10,  # Si
            15: 1.80,  # P
            16: 1.80,  # S
            17: 1.75,  # Cl
            19: 2.75,  # K
            20: 2.31,  # Ca
        }
        
        if self.use_gpu:
            self.xp = cp
        else:
            self.xp = np
    
    def calculate_descriptors(self, crystal: CrystalStructure,
                            supercell_size: Tuple[int, int, int] = (2, 2, 2),
                            grid_resolution: float = 0.2) -> CrystalDescriptors:
        """Calculate comprehensive crystal descriptors.
        
        Args:
            crystal: Crystal structure to analyze
            supercell_size: Size of supercell for calculations
            grid_resolution: Grid resolution for void calculations (Å)
            
        Returns:
            CrystalDescriptors object
        """
        # Create supercell for accurate calculations
        supercell = self._create_supercell(crystal, supercell_size)
        
        # Calculate basic geometric descriptors
        basic_descriptors = self._calculate_basic_descriptors(crystal)
        
        # Calculate void and pore descriptors
        void_descriptors = self._calculate_void_descriptors(
            supercell, grid_resolution
        )
        
        # Calculate surface area descriptors
        surface_descriptors = self._calculate_surface_descriptors(
            supercell, grid_resolution
        )
        
        # Calculate topological descriptors
        topo_descriptors = self._calculate_topological_descriptors(supercell)
        
        # Combine all descriptors
        return CrystalDescriptors(
            **basic_descriptors,
            **void_descriptors,
            **surface_descriptors,
            **topo_descriptors
        )
    
    def _create_supercell(self, crystal: CrystalStructure,
                         size: Tuple[int, int, int]) -> CrystalStructure:
        """Create supercell for more accurate calculations."""
        nx, ny, nz = size
        
        # Replicate atomic positions
        new_coords = []
        new_symbols = []
        new_numbers = []
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    offset = np.array([i, j, k])
                    shifted_coords = crystal.fractional_coords + offset
                    new_coords.extend(shifted_coords)
                    new_symbols.extend(crystal.atomic_symbols)
                    new_numbers.extend(crystal.atomic_numbers)
        
        new_coords = np.array(new_coords)
        
        # Scale lattice vectors
        new_lattice = crystal.lattice_vectors.copy()
        new_lattice[0] *= nx
        new_lattice[1] *= ny
        new_lattice[2] *= nz
        
        # Convert to cartesian
        new_cart_coords = np.dot(new_coords, new_lattice)
        
        return CrystalStructure(
            lattice_vectors=new_lattice,
            lattice_parameters=None,
            atomic_numbers=np.array(new_numbers),
            atomic_symbols=new_symbols,
            fractional_coords=new_coords,
            cartesian_coords=new_cart_coords
        )
    
    def _calculate_basic_descriptors(self, crystal: CrystalStructure) -> Dict[str, float]:
        """Calculate basic geometric descriptors."""
        volume = crystal.volume
        density = crystal.density if crystal.density else 0.0
        
        # Calculate packing efficiency
        atomic_volumes = []
        for z in crystal.atomic_numbers:
            radius = self.vdw_radii.get(z, 1.5)
            atomic_volumes.append(4/3 * np.pi * radius**3)
        
        total_atomic_volume = sum(atomic_volumes)
        packing_efficiency = total_atomic_volume / volume
        
        return {
            'volume': volume,
            'density': density,
            'packing_efficiency': packing_efficiency,
            'lattice_parameters': crystal.lattice_parameters[:3],
            'lattice_angles': crystal.lattice_parameters[3:],
            'lattice_volume': volume
        }
    
    def _calculate_void_descriptors(self, crystal: CrystalStructure,
                                  grid_resolution: float) -> Dict[str, float]:
        """Calculate void fraction and pore descriptors using grid-based method."""
        # Create 3D grid
        lattice = crystal.lattice_vectors
        
        # Determine grid dimensions
        max_dim = np.max(np.linalg.norm(lattice, axis=1))
        n_points = int(max_dim / grid_resolution)
        
        # Create grid points in fractional coordinates
        x = np.linspace(0, 1, n_points)
        y = np.linspace(0, 1, n_points)
        z = np.linspace(0, 1, n_points)
        
        grid_frac = np.array(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T
        
        # Convert to cartesian
        grid_cart = np.dot(grid_frac, lattice)
        
        if self.use_gpu:
            grid_cart = cp.asarray(grid_cart)
            atom_coords = cp.asarray(crystal.cartesian_coords)
            atomic_numbers = cp.asarray(crystal.atomic_numbers)
        else:
            atom_coords = crystal.cartesian_coords
            atomic_numbers = crystal.atomic_numbers
        
        # Calculate distances from grid points to atoms
        void_mask = self._calculate_void_mask(
            grid_cart, atom_coords, atomic_numbers
        )
        
        # Calculate accessible void (considering probe radius)
        accessible_mask = self._calculate_accessible_void_mask(
            grid_cart, atom_coords, atomic_numbers
        )
        
        if self.use_gpu:
            void_mask = cp.asnumpy(void_mask)
            accessible_mask = cp.asnumpy(accessible_mask)
        
        # Calculate fractions
        total_points = len(grid_cart)
        void_fraction = np.sum(void_mask) / total_points
        accessible_void_fraction = np.sum(accessible_mask) / total_points
        
        # Calculate pore limiting diameter
        pld = self._calculate_pore_limiting_diameter(
            grid_cart, atom_coords, atomic_numbers, accessible_mask
        )
        
        # Calculate largest cavity diameter
        lcd = self._calculate_largest_cavity_diameter(
            grid_cart, accessible_mask, grid_resolution
        )
        
        return {
            'void_fraction': void_fraction,
            'accessible_void_fraction': accessible_void_fraction,
            'pore_limiting_diameter': pld,
            'largest_cavity_diameter': lcd
        }
    
    def _calculate_void_mask(self, grid_points, atom_coords, atomic_numbers):
        """Calculate which grid points are in void space."""
        if self.use_gpu:
            # GPU implementation
            void_mask = cp.ones(len(grid_points), dtype=bool)
            
            for i, (coord, z) in enumerate(zip(atom_coords, atomic_numbers)):
                radius = self.vdw_radii.get(int(z), 1.5)
                distances = cp.linalg.norm(grid_points - coord, axis=1)
                void_mask &= (distances > radius)
            
            return void_mask
        else:
            # CPU implementation
            void_mask = np.ones(len(grid_points), dtype=bool)
            
            for coord, z in zip(atom_coords, atomic_numbers):
                radius = self.vdw_radii.get(z, 1.5)
                distances = np.linalg.norm(grid_points - coord, axis=1)
                void_mask &= (distances > radius)
            
            return void_mask
    
    def _calculate_accessible_void_mask(self, grid_points, atom_coords, atomic_numbers):
        """Calculate accessible void considering probe radius."""
        if self.use_gpu:
            accessible_mask = cp.ones(len(grid_points), dtype=bool)
            
            for i, (coord, z) in enumerate(zip(atom_coords, atomic_numbers)):
                radius = self.vdw_radii.get(int(z), 1.5) + self.probe_radius
                distances = cp.linalg.norm(grid_points - coord, axis=1)
                accessible_mask &= (distances > radius)
            
            return accessible_mask
        else:
            accessible_mask = np.ones(len(grid_points), dtype=bool)
            
            for coord, z in zip(atom_coords, atomic_numbers):
                radius = self.vdw_radii.get(z, 1.5) + self.probe_radius
                distances = np.linalg.norm(grid_points - coord, axis=1)
                accessible_mask &= (distances > radius)
            
            return accessible_mask
    
    def _calculate_pore_limiting_diameter(self, grid_points, atom_coords,
                                        atomic_numbers, accessible_mask) -> float:
        """Calculate pore limiting diameter."""
        if not SCIPY_AVAILABLE:
            return 0.0
        
        # Convert to numpy if needed
        if self.use_gpu:
            grid_points = cp.asnumpy(grid_points)
            atom_coords = cp.asnumpy(atom_coords)
            atomic_numbers = cp.asnumpy(atomic_numbers)
        
        accessible_points = grid_points[accessible_mask]
        
        if len(accessible_points) == 0:
            return 0.0
        
        # Find the largest sphere that can fit through the pore network
        min_clearance = float('inf')
        
        # Sample subset of accessible points for efficiency
        n_sample = min(1000, len(accessible_points))
        sample_indices = np.random.choice(len(accessible_points), n_sample, replace=False)
        sample_points = accessible_points[sample_indices]
        
        for point in sample_points:
            # Find distance to nearest atom surface
            distances_to_atoms = []
            for coord, z in zip(atom_coords, atomic_numbers):
                radius = self.vdw_radii.get(z, 1.5)
                dist_to_surface = np.linalg.norm(point - coord) - radius
                distances_to_atoms.append(dist_to_surface)
            
            clearance = min(distances_to_atoms)
            min_clearance = min(min_clearance, clearance)
        
        return max(0.0, 2 * min_clearance)  # Diameter
    
    def _calculate_largest_cavity_diameter(self, grid_points, accessible_mask,
                                         grid_resolution: float) -> float:
        """Calculate largest cavity diameter."""
        if self.use_gpu:
            grid_points = cp.asnumpy(grid_points)
        
        accessible_points = grid_points[accessible_mask]
        
        if len(accessible_points) == 0:
            return 0.0
        
        # Find the largest distance between accessible points
        if len(accessible_points) > 5000:
            # Sample for large datasets
            sample_indices = np.random.choice(len(accessible_points), 5000, replace=False)
            sample_points = accessible_points[sample_indices]
        else:
            sample_points = accessible_points
        
        if SCIPY_AVAILABLE and len(sample_points) > 1:
            distances = cdist(sample_points, sample_points)
            return np.max(distances)
        else:
            # Fallback method
            max_dist = 0.0
            for i in range(min(100, len(sample_points))):
                for j in range(i+1, min(100, len(sample_points))):
                    dist = np.linalg.norm(sample_points[i] - sample_points[j])
                    max_dist = max(max_dist, dist)
            return max_dist
    
    def _calculate_surface_descriptors(self, crystal: CrystalStructure,
                                     grid_resolution: float) -> Dict[str, float]:
        """Calculate surface area descriptors."""
        # Geometric surface area (simplified calculation)
        geometric_sa = self._calculate_geometric_surface_area(crystal)
        
        # Accessible surface area
        accessible_sa = self._calculate_accessible_surface_area(
            crystal, grid_resolution
        )
        
        # Volumetric surface area (m²/g)
        density_g_cm3 = crystal.density if crystal.density else 1.0
        volumetric_sa = accessible_sa / (density_g_cm3 * 1000)  # Convert to m²/g
        
        return {
            'surface_area_geometric': geometric_sa,
            'surface_area_accessible': accessible_sa,
            'surface_area_volumetric': volumetric_sa
        }
    
    def _calculate_geometric_surface_area(self, crystal: CrystalStructure) -> float:
        """Calculate geometric surface area."""
        # Simplified calculation based on unit cell surface area
        lattice = crystal.lattice_vectors
        
        # Calculate face areas
        face1_area = np.linalg.norm(np.cross(lattice[0], lattice[1]))
        face2_area = np.linalg.norm(np.cross(lattice[1], lattice[2]))
        face3_area = np.linalg.norm(np.cross(lattice[0], lattice[2]))
        
        # Total surface area (6 faces)
        total_sa = 2 * (face1_area + face2_area + face3_area)
        
        return total_sa
    
    def _calculate_accessible_surface_area(self, crystal: CrystalStructure,
                                         grid_resolution: float) -> float:
        """Calculate accessible surface area using probe rolling."""
        # Simplified implementation - in practice, more sophisticated methods
        # like the Shrake-Rupley algorithm would be used
        
        # Create surface grid points
        lattice = crystal.lattice_vectors
        max_dim = np.max(np.linalg.norm(lattice, axis=1))
        n_points = int(max_dim / grid_resolution)
        
        # Sample points on unit cell faces
        surface_points = []
        
        # Face 1 (xy plane)
        for i in np.linspace(0, 1, n_points):
            for j in np.linspace(0, 1, n_points):
                point = i * lattice[0] + j * lattice[1]
                surface_points.append(point)
                point = i * lattice[0] + j * lattice[1] + lattice[2]
                surface_points.append(point)
        
        surface_points = np.array(surface_points)
        
        # Check accessibility
        accessible_count = 0
        for point in surface_points:
            is_accessible = True
            for coord, z in zip(crystal.cartesian_coords, crystal.atomic_numbers):
                radius = self.vdw_radii.get(z, 1.5) + self.probe_radius
                if np.linalg.norm(point - coord) < radius:
                    is_accessible = False
                    break
            if is_accessible:
                accessible_count += 1
        
        # Estimate surface area
        face_area = np.linalg.norm(np.cross(lattice[0], lattice[1]))
        accessible_fraction = accessible_count / len(surface_points)
        
        return accessible_fraction * face_area * 6  # 6 faces
    
    def _calculate_topological_descriptors(self, crystal: CrystalStructure) -> Dict[str, Any]:
        """Calculate topological descriptors."""
        # Calculate coordination numbers
        coordination_numbers = self._calculate_coordination_numbers(crystal)
        
        return {
            'coordination_numbers': coordination_numbers,
            'average_coordination': np.mean(coordination_numbers),
            'coordination_variance': np.var(coordination_numbers)
        }
    
    def _calculate_coordination_numbers(self, crystal: CrystalStructure,
                                     cutoff_factor: float = 1.2) -> np.ndarray:
        """Calculate coordination numbers for each atom."""
        coords = crystal.cartesian_coords
        atomic_numbers = crystal.atomic_numbers
        
        coordination_numbers = []
        
        for i, (coord_i, z_i) in enumerate(zip(coords, atomic_numbers)):
            radius_i = self.vdw_radii.get(z_i, 1.5)
            coord_count = 0
            
            for j, (coord_j, z_j) in enumerate(zip(coords, atomic_numbers)):
                if i == j:
                    continue
                
                radius_j = self.vdw_radii.get(z_j, 1.5)
                cutoff_distance = (radius_i + radius_j) * cutoff_factor
                
                distance = np.linalg.norm(coord_i - coord_j)
                if distance < cutoff_distance:
                    coord_count += 1
            
            coordination_numbers.append(coord_count)
        
        return np.array(coordination_numbers)
    
    def calculate_batch_descriptors(self, crystals: List[CrystalStructure],
                                  **kwargs) -> pd.DataFrame:
        """Calculate descriptors for multiple crystals.
        
        Args:
            crystals: List of crystal structures
            **kwargs: Additional arguments for descriptor calculation
            
        Returns:
            DataFrame with descriptor results
        """
        results = []
        
        for i, crystal in enumerate(crystals):
            try:
                descriptors = self.calculate_descriptors(crystal, **kwargs)
                result_dict = descriptors.to_dict()
                result_dict['crystal_id'] = i
                results.append(result_dict)
            except Exception as e:
                warnings.warn(f"Failed to calculate descriptors for crystal {i}: {e}")
                # Add empty result with crystal_id
                results.append({'crystal_id': i})
        
        return pd.DataFrame(results)
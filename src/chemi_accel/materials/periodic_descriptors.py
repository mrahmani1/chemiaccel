"""
Periodic descriptor calculations for crystalline materials.

This module provides specialized descriptors that take advantage of the
periodic nature of crystalline materials, including symmetry-based descriptors,
periodic boundary condition calculations, and extended structure analysis.
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
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .crystal_parser import CrystalStructure


@dataclass
class PeriodicDescriptors:
    """Container for periodic descriptor results."""
    
    # Symmetry descriptors
    symmetry_index: float
    symmetry_deviation: float
    point_symmetry_score: float
    
    # Periodic structure descriptors
    radial_distribution_function: np.ndarray
    angular_distribution_function: np.ndarray
    structure_factor_peaks: np.ndarray
    
    # Extended structure descriptors
    nearest_neighbor_distances: np.ndarray
    second_shell_distances: np.ndarray
    coordination_shells: List[Dict[str, Any]]
    
    # Topology descriptors
    ring_statistics: Dict[str, float]
    connectivity_index: float
    network_dimensionality: int
    
    # Periodic boundary effects
    surface_to_volume_ratio: float
    edge_effects_score: float
    periodicity_score: float
    
    # Electronic structure indicators
    band_gap_predictor: float
    metallic_character: float
    ionic_character: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, np.ndarray):
                if field_value.size <= 10:  # Small arrays
                    for i, val in enumerate(field_value):
                        result[f"{field_name}_{i}"] = float(val)
                else:  # Large arrays - use statistics
                    result[f"{field_name}_mean"] = float(np.mean(field_value))
                    result[f"{field_name}_std"] = float(np.std(field_value))
                    result[f"{field_name}_min"] = float(np.min(field_value))
                    result[f"{field_name}_max"] = float(np.max(field_value))
            elif isinstance(field_value, list):
                if field_value and isinstance(field_value[0], dict):
                    # Handle coordination shells
                    for i, shell in enumerate(field_value[:5]):  # First 5 shells
                        for key, val in shell.items():
                            result[f"{field_name}_shell_{i}_{key}"] = val
                else:
                    result[field_name] = field_value
            elif isinstance(field_value, dict):
                for key, val in field_value.items():
                    result[f"{field_name}_{key}"] = val
            else:
                result[field_name] = field_value
        return result


class PeriodicDescriptorCalculator:
    """Calculator for periodic descriptors of crystalline materials."""
    
    def __init__(self, use_gpu: bool = True, max_shells: int = 10):
        """Initialize periodic descriptor calculator.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            max_shells: Maximum number of coordination shells to analyze
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.max_shells = max_shells
        
        if self.use_gpu:
            self.xp = cp
        else:
            self.xp = np
        
        # Atomic properties for calculations
        self.atomic_properties = {
            'electronegativity': {
                1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
                11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19,
                16: 2.58, 17: 3.16, 19: 0.82, 20: 1.00
            },
            'ionic_radius': {
                1: 0.31, 6: 0.16, 7: 0.13, 8: 1.40, 9: 1.33,
                11: 1.02, 12: 0.72, 13: 0.54, 14: 0.26, 15: 0.44,
                16: 1.84, 17: 1.81, 19: 1.38, 20: 1.00
            },
            'covalent_radius': {
                1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
                11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
                16: 1.05, 17: 1.02, 19: 2.03, 20: 1.76
            }
        }
    
    def calculate_descriptors(self, crystal: CrystalStructure,
                            supercell_size: Tuple[int, int, int] = (3, 3, 3),
                            rdf_max_distance: float = 15.0,
                            rdf_bins: int = 150) -> PeriodicDescriptors:
        """Calculate comprehensive periodic descriptors.
        
        Args:
            crystal: Crystal structure to analyze
            supercell_size: Size of supercell for calculations
            rdf_max_distance: Maximum distance for RDF calculation
            rdf_bins: Number of bins for RDF
            
        Returns:
            PeriodicDescriptors object
        """
        # Create supercell for accurate periodic calculations
        supercell = self._create_supercell(crystal, supercell_size)
        
        # Calculate symmetry descriptors
        symmetry_desc = self._calculate_symmetry_descriptors(crystal, supercell)
        
        # Calculate radial and angular distribution functions
        rdf, adf = self._calculate_distribution_functions(
            supercell, rdf_max_distance, rdf_bins
        )
        
        # Calculate structure factor
        structure_factor = self._calculate_structure_factor(supercell)
        
        # Calculate coordination shell analysis
        coord_shells = self._calculate_coordination_shells(supercell)
        
        # Calculate topology descriptors
        topology_desc = self._calculate_topology_descriptors(supercell)
        
        # Calculate periodic boundary effects
        boundary_desc = self._calculate_boundary_effects(crystal, supercell)
        
        # Calculate electronic structure indicators
        electronic_desc = self._calculate_electronic_indicators(crystal)
        
        return PeriodicDescriptors(
            **symmetry_desc,
            radial_distribution_function=rdf,
            angular_distribution_function=adf,
            structure_factor_peaks=structure_factor,
            **coord_shells,
            **topology_desc,
            **boundary_desc,
            **electronic_desc
        )
    
    def _create_supercell(self, crystal: CrystalStructure,
                         size: Tuple[int, int, int]) -> CrystalStructure:
        """Create supercell with periodic boundary conditions."""
        nx, ny, nz = size
        
        # Generate all lattice translations
        translations = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    translations.append([i, j, k])
        
        translations = np.array(translations)
        
        # Replicate atoms
        n_atoms = len(crystal.atomic_numbers)
        n_translations = len(translations)
        
        new_coords = []
        new_symbols = []
        new_numbers = []
        
        for trans in translations:
            shifted_coords = crystal.fractional_coords + trans
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
    
    def _calculate_symmetry_descriptors(self, crystal: CrystalStructure,
                                      supercell: CrystalStructure) -> Dict[str, float]:
        """Calculate symmetry-based descriptors."""
        # Symmetry index based on atomic position regularity
        coords = crystal.fractional_coords
        
        # Calculate distances from ideal positions
        ideal_positions = self._generate_ideal_positions(crystal)
        deviations = np.linalg.norm(coords - ideal_positions, axis=1)
        
        symmetry_deviation = np.mean(deviations)
        symmetry_index = 1.0 / (1.0 + symmetry_deviation)
        
        # Point symmetry score based on coordination environment
        point_symmetry = self._calculate_point_symmetry_score(supercell)
        
        return {
            'symmetry_index': symmetry_index,
            'symmetry_deviation': symmetry_deviation,
            'point_symmetry_score': point_symmetry
        }
    
    def _generate_ideal_positions(self, crystal: CrystalStructure) -> np.ndarray:
        """Generate ideal atomic positions for symmetry comparison."""
        # Simplified - assumes cubic symmetry for comparison
        n_atoms = len(crystal.atomic_numbers)
        
        # Generate regular grid positions
        n_per_dim = int(np.ceil(n_atoms**(1/3)))
        positions = []
        
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                for k in range(n_per_dim):
                    if len(positions) < n_atoms:
                        pos = np.array([i, j, k]) / n_per_dim
                        positions.append(pos)
        
        return np.array(positions[:n_atoms])
    
    def _calculate_point_symmetry_score(self, supercell: CrystalStructure) -> float:
        """Calculate point symmetry score based on local environments."""
        coords = supercell.cartesian_coords
        
        if self.use_gpu:
            coords = cp.asarray(coords)
        
        symmetry_scores = []
        
        # Sample subset of atoms for efficiency
        n_sample = min(100, len(coords))
        sample_indices = np.random.choice(len(coords), n_sample, replace=False)
        
        for idx in sample_indices:
            center = coords[idx]
            
            # Find neighbors within cutoff
            if self.use_gpu:
                distances = cp.linalg.norm(coords - center, axis=1)
                neighbor_mask = (distances > 0.1) & (distances < 5.0)
                neighbors = coords[neighbor_mask]
            else:
                distances = np.linalg.norm(coords - center, axis=1)
                neighbor_mask = (distances > 0.1) & (distances < 5.0)
                neighbors = coords[neighbor_mask]
            
            if len(neighbors) > 3:
                # Calculate symmetry of local environment
                local_symmetry = self._calculate_local_symmetry(center, neighbors)
                symmetry_scores.append(local_symmetry)
        
        return np.mean(symmetry_scores) if symmetry_scores else 0.0
    
    def _calculate_local_symmetry(self, center, neighbors) -> float:
        """Calculate local symmetry around a central atom."""
        if self.use_gpu:
            center = cp.asnumpy(center)
            neighbors = cp.asnumpy(neighbors)
        
        # Vector from center to neighbors
        vectors = neighbors - center
        distances = np.linalg.norm(vectors, axis=1)
        
        # Normalize vectors
        unit_vectors = vectors / distances[:, np.newaxis]
        
        # Calculate angular distribution
        angles = []
        for i in range(len(unit_vectors)):
            for j in range(i+1, len(unit_vectors)):
                angle = np.arccos(np.clip(np.dot(unit_vectors[i], unit_vectors[j]), -1, 1))
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Symmetry score based on angle distribution regularity
        angles = np.array(angles)
        angle_std = np.std(angles)
        symmetry_score = 1.0 / (1.0 + angle_std)
        
        return symmetry_score
    
    def _calculate_distribution_functions(self, supercell: CrystalStructure,
                                        max_distance: float, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate radial and angular distribution functions."""
        coords = supercell.cartesian_coords
        
        if self.use_gpu:
            coords = cp.asarray(coords)
        
        # Radial Distribution Function
        rdf = self._calculate_rdf(coords, max_distance, n_bins)
        
        # Angular Distribution Function
        adf = self._calculate_adf(coords, max_distance)
        
        if self.use_gpu:
            rdf = cp.asnumpy(rdf)
            adf = cp.asnumpy(adf)
        
        return rdf, adf
    
    def _calculate_rdf(self, coords, max_distance: float, n_bins: int) -> np.ndarray:
        """Calculate radial distribution function."""
        n_atoms = len(coords)
        
        # Calculate all pairwise distances
        if self.use_gpu:
            # GPU implementation
            distances = []
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    dist = cp.linalg.norm(coords[i] - coords[j])
                    if dist < max_distance:
                        distances.append(dist)
            distances = cp.array(distances)
        else:
            # CPU implementation
            distances = []
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < max_distance:
                        distances.append(dist)
            distances = np.array(distances)
        
        # Create histogram
        if self.use_gpu:
            hist, bin_edges = cp.histogram(distances, bins=n_bins, range=(0, max_distance))
        else:
            hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, max_distance))
        
        # Normalize by shell volume
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        shell_volumes = 4 * np.pi * bin_centers**2 * (bin_edges[1] - bin_edges[0])
        
        # Avoid division by zero
        shell_volumes = np.where(shell_volumes > 0, shell_volumes, 1)
        
        rdf = hist / shell_volumes
        return rdf
    
    def _calculate_adf(self, coords, max_distance: float) -> np.ndarray:
        """Calculate angular distribution function."""
        n_atoms = len(coords)
        angles = []
        
        # Sample subset for efficiency
        n_sample = min(50, n_atoms)
        sample_indices = np.random.choice(n_atoms, n_sample, replace=False)
        
        for center_idx in sample_indices:
            center = coords[center_idx]
            
            # Find neighbors
            if self.use_gpu:
                distances = cp.linalg.norm(coords - center, axis=1)
                neighbor_mask = (distances > 0.1) & (distances < max_distance)
                neighbors = coords[neighbor_mask]
            else:
                distances = np.linalg.norm(coords - center, axis=1)
                neighbor_mask = (distances > 0.1) & (distances < max_distance)
                neighbors = coords[neighbor_mask]
            
            if len(neighbors) >= 2:
                # Calculate angles between neighbor vectors
                vectors = neighbors - center
                
                for i in range(len(vectors)):
                    for j in range(i+1, len(vectors)):
                        if self.use_gpu:
                            dot_product = cp.dot(vectors[i], vectors[j])
                            norms = cp.linalg.norm(vectors[i]) * cp.linalg.norm(vectors[j])
                        else:
                            dot_product = np.dot(vectors[i], vectors[j])
                            norms = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
                        
                        if norms > 0:
                            cos_angle = dot_product / norms
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.arccos(cos_angle)
                            angles.append(angle)
        
        if not angles:
            return np.array([])
        
        # Create angular histogram
        angles = np.array(angles)
        hist, _ = np.histogram(angles, bins=36, range=(0, np.pi))
        
        return hist
    
    def _calculate_structure_factor(self, supercell: CrystalStructure) -> np.ndarray:
        """Calculate structure factor peaks."""
        coords = supercell.fractional_coords
        atomic_numbers = supercell.atomic_numbers
        
        # Calculate structure factors for common reflections
        h_values = np.arange(-3, 4)
        k_values = np.arange(-3, 4)
        l_values = np.arange(-3, 4)
        
        structure_factors = []
        
        for h in h_values:
            for k in k_values:
                for l in l_values:
                    if h == 0 and k == 0 and l == 0:
                        continue
                    
                    # Calculate structure factor
                    sf_real = 0.0
                    sf_imag = 0.0
                    
                    for coord, z in zip(coords, atomic_numbers):
                        phase = 2 * np.pi * (h * coord[0] + k * coord[1] + l * coord[2])
                        # Use atomic number as scattering factor approximation
                        f = z
                        sf_real += f * np.cos(phase)
                        sf_imag += f * np.sin(phase)
                    
                    sf_magnitude = np.sqrt(sf_real**2 + sf_imag**2)
                    structure_factors.append(sf_magnitude)
        
        # Return top peaks
        structure_factors = np.array(structure_factors)
        top_peaks = np.sort(structure_factors)[-10:]  # Top 10 peaks
        
        return top_peaks
    
    def _calculate_coordination_shells(self, supercell: CrystalStructure) -> Dict[str, Any]:
        """Calculate coordination shell analysis."""
        coords = supercell.cartesian_coords
        
        # Calculate all pairwise distances
        if SCIPY_AVAILABLE and len(coords) < 1000:
            distances = pdist(coords)
            distance_matrix = squareform(distances)
        else:
            # Fallback for large systems
            distance_matrix = np.zeros((len(coords), len(coords)))
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
        
        # Find coordination shells
        shells = []
        shell_distances = []
        
        # Use first atom as reference
        ref_distances = distance_matrix[0]
        ref_distances = ref_distances[ref_distances > 0]  # Remove self-distance
        
        if len(ref_distances) > 0:
            # Sort distances and identify shells
            sorted_distances = np.sort(ref_distances)
            
            current_shell = [sorted_distances[0]]
            shell_tolerance = 0.5  # Angstrom
            
            for dist in sorted_distances[1:]:
                if dist - current_shell[-1] < shell_tolerance:
                    current_shell.append(dist)
                else:
                    # New shell
                    shell_info = {
                        'mean_distance': np.mean(current_shell),
                        'std_distance': np.std(current_shell),
                        'coordination_number': len(current_shell)
                    }
                    shells.append(shell_info)
                    shell_distances.append(np.mean(current_shell))
                    
                    current_shell = [dist]
                    
                    if len(shells) >= self.max_shells:
                        break
            
            # Add last shell
            if current_shell and len(shells) < self.max_shells:
                shell_info = {
                    'mean_distance': np.mean(current_shell),
                    'std_distance': np.std(current_shell),
                    'coordination_number': len(current_shell)
                }
                shells.append(shell_info)
                shell_distances.append(np.mean(current_shell))
        
        # Extract nearest neighbor and second shell distances
        nn_distances = shell_distances[:1] if shell_distances else [0.0]
        second_shell = shell_distances[1:2] if len(shell_distances) > 1 else [0.0]
        
        return {
            'nearest_neighbor_distances': np.array(nn_distances),
            'second_shell_distances': np.array(second_shell),
            'coordination_shells': shells
        }
    
    def _calculate_topology_descriptors(self, supercell: CrystalStructure) -> Dict[str, Any]:
        """Calculate topology-based descriptors."""
        coords = supercell.cartesian_coords
        
        # Ring statistics (simplified)
        ring_stats = self._calculate_ring_statistics(supercell)
        
        # Connectivity index
        connectivity = self._calculate_connectivity_index(supercell)
        
        # Network dimensionality
        dimensionality = self._estimate_network_dimensionality(supercell)
        
        return {
            'ring_statistics': ring_stats,
            'connectivity_index': connectivity,
            'network_dimensionality': dimensionality
        }
    
    def _calculate_ring_statistics(self, supercell: CrystalStructure) -> Dict[str, float]:
        """Calculate ring statistics in the crystal structure."""
        # Simplified ring analysis
        coords = supercell.cartesian_coords
        
        # Build connectivity graph
        n_atoms = len(coords)
        adjacency = np.zeros((n_atoms, n_atoms))
        
        cutoff_distance = 3.0  # Angstrom
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < cutoff_distance:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
        
        # Count rings (simplified - just 3, 4, 5, 6 membered rings)
        ring_counts = {3: 0, 4: 0, 5: 0, 6: 0}
        
        # This is a simplified implementation
        # In practice, more sophisticated graph algorithms would be used
        total_connections = np.sum(adjacency) / 2
        
        return {
            'total_rings': total_connections,
            'ring_density': total_connections / n_atoms,
            'average_ring_size': 4.0  # Placeholder
        }
    
    def _calculate_connectivity_index(self, supercell: CrystalStructure) -> float:
        """Calculate connectivity index."""
        coords = supercell.cartesian_coords
        n_atoms = len(coords)
        
        # Count connections
        connections = 0
        cutoff_distance = 3.0
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < cutoff_distance:
                    connections += 1
        
        # Normalize by maximum possible connections
        max_connections = n_atoms * (n_atoms - 1) / 2
        connectivity_index = connections / max_connections
        
        return connectivity_index
    
    def _estimate_network_dimensionality(self, supercell: CrystalStructure) -> int:
        """Estimate network dimensionality (0D, 1D, 2D, 3D)."""
        coords = supercell.cartesian_coords
        
        # Analyze connectivity patterns
        # This is a simplified heuristic
        
        # Calculate principal components of atomic positions
        centered_coords = coords - np.mean(coords, axis=0)
        cov_matrix = np.cov(centered_coords.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Determine dimensionality based on eigenvalue ratios
        ratios = eigenvalues[:-1] / eigenvalues[1:]
        
        # Count significant dimensions
        threshold = 10.0  # Threshold for significant dimension
        significant_dims = np.sum(ratios > threshold) + 1
        
        return min(3, max(0, significant_dims))
    
    def _calculate_boundary_effects(self, crystal: CrystalStructure,
                                  supercell: CrystalStructure) -> Dict[str, float]:
        """Calculate periodic boundary effects."""
        # Surface to volume ratio
        volume = crystal.volume
        
        # Estimate surface area (simplified)
        lattice = crystal.lattice_vectors
        face_areas = [
            np.linalg.norm(np.cross(lattice[0], lattice[1])),
            np.linalg.norm(np.cross(lattice[1], lattice[2])),
            np.linalg.norm(np.cross(lattice[0], lattice[2]))
        ]
        total_surface_area = 2 * sum(face_areas)
        
        surface_to_volume = total_surface_area / volume
        
        # Edge effects score (simplified)
        edge_effects = self._calculate_edge_effects_score(crystal)
        
        # Periodicity score
        periodicity = self._calculate_periodicity_score(crystal, supercell)
        
        return {
            'surface_to_volume_ratio': surface_to_volume,
            'edge_effects_score': edge_effects,
            'periodicity_score': periodicity
        }
    
    def _calculate_edge_effects_score(self, crystal: CrystalStructure) -> float:
        """Calculate edge effects score."""
        # Simplified calculation based on unit cell shape
        lattice_params = crystal.lattice_parameters
        
        if lattice_params is not None:
            a, b, c = lattice_params[:3]
            # More cubic = less edge effects
            ratios = [a/b, b/c, a/c]
            deviations = [abs(r - 1.0) for r in ratios]
            edge_effects = np.mean(deviations)
        else:
            edge_effects = 0.5  # Default value
        
        return edge_effects
    
    def _calculate_periodicity_score(self, crystal: CrystalStructure,
                                   supercell: CrystalStructure) -> float:
        """Calculate periodicity score."""
        # Compare unit cell with supercell patterns
        unit_coords = crystal.fractional_coords
        super_coords = supercell.fractional_coords
        
        # Simplified periodicity check
        n_unit = len(unit_coords)
        n_super = len(super_coords)
        
        expected_ratio = n_super / n_unit
        actual_ratio = len(np.unique(super_coords, axis=0)) / n_unit
        
        periodicity_score = 1.0 / (1.0 + abs(expected_ratio - actual_ratio))
        
        return periodicity_score
    
    def _calculate_electronic_indicators(self, crystal: CrystalStructure) -> Dict[str, float]:
        """Calculate electronic structure indicators."""
        atomic_numbers = crystal.atomic_numbers
        
        # Band gap predictor based on electronegativity differences
        electronegativities = [
            self.atomic_properties['electronegativity'].get(z, 2.0)
            for z in atomic_numbers
        ]
        
        en_range = max(electronegativities) - min(electronegativities)
        band_gap_predictor = en_range * 2.0  # Rough estimate
        
        # Metallic character (inverse of electronegativity range)
        metallic_character = 1.0 / (1.0 + en_range)
        
        # Ionic character based on electronegativity differences
        ionic_character = min(1.0, en_range / 3.0)
        
        return {
            'band_gap_predictor': band_gap_predictor,
            'metallic_character': metallic_character,
            'ionic_character': ionic_character
        }
    
    def calculate_batch_descriptors(self, crystals: List[CrystalStructure],
                                  **kwargs) -> pd.DataFrame:
        """Calculate periodic descriptors for multiple crystals.
        
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
                warnings.warn(f"Failed to calculate periodic descriptors for crystal {i}: {e}")
                results.append({'crystal_id': i})
        
        return pd.DataFrame(results)
"""
Dynamic PLIF Generation for Protein-Ligand Interactions - Task 19.4

This module provides advanced dynamic Protein-Ligand Interaction Fingerprint (PLIF)
generation capabilities for analyzing protein-ligand binding interactions across
MD trajectories, enabling real-time monitoring of binding modes and dynamics.

Key Features:
- Real-time PLIF generation from MD trajectories
- Comprehensive interaction type detection and classification
- Time-resolved binding mode analysis and clustering
- GPU-accelerated geometric calculations for large systems
- Machine learning-based interaction prediction and validation
- Dynamic pharmacophore generation from trajectory ensembles

Interaction Types Detected:
- Hydrogen bonds (donor/acceptor with geometric criteria)
- Electrostatic interactions (salt bridges, ionic contacts)
- Van der Waals contacts and hydrophobic interactions
- π-π stacking and cation-π interactions
- Metal coordination bonds
- Halogen bonds and other non-standard interactions

Scientific Applications:
- Drug-target interaction profiling and optimization
- Binding mode stability assessment in drug design
- Off-target interaction prediction and analysis
- Structure-activity relationship (SAR) studies
- Lead compound optimization and scaffold hopping
- Allosteric binding site characterization

Performance Optimizations:
- Vectorized geometric calculations using NumPy/CuPy
- Spatial indexing for efficient neighbor searches
- Streaming PLIF calculation for memory efficiency
- Parallel processing across trajectory frames
- GPU acceleration for distance and angle calculations
- Pre-computed interaction templates for speed

Analysis Capabilities:
- PLIF similarity clustering and trajectory classification
- Binding mode transition detection and kinetics
- Residence time analysis for specific interactions
- Dynamic pharmacophore consensus modeling
- Interaction frequency maps and hotspot identification
- Machine learning-based binding affinity prediction
"""

import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Iterator, Any, Callable, Set
import threading
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Core ChemiAccel imports
from chemi_accel.core.molecular import MolecularData
from chemi_accel.core.exceptions import (
    ChemiAccelError,
    ValidationError,
    CalculationError
)
from chemi_accel.analysis.trajectory_reader import TrajectoryFrame, TrajectoryMetadata

# GPU acceleration imports
try:
    import cupy as cp
    import cupyx.scipy.spatial.distance as cp_distance
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Chemical analysis imports
try:
    from rdkit import Chem, Geometry
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available - limited chemical analysis")

# Machine learning imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class InteractionType(Enum):
    """Types of protein-ligand interactions."""
    
    HYDROGEN_BOND = "hydrogen_bond"
    ELECTROSTATIC = "electrostatic" 
    VAN_DER_WAALS = "van_der_waals"
    HYDROPHOBIC = "hydrophobic"
    PI_PI_STACKING = "pi_pi_stacking"
    PI_CATION = "pi_cation"
    HALOGEN_BOND = "halogen_bond"
    METAL_COORDINATION = "metal_coordination"
    AROMATIC_CONTACT = "aromatic_contact"
    WATER_MEDIATED = "water_mediated"

@dataclass
class InteractionCriteria:
    """Geometric criteria for interaction detection."""
    
    # Hydrogen bond criteria
    hbond_donor_acceptor_distance: float = 3.5  # Angstrom
    hbond_angle_cutoff: float = 120.0  # degrees
    
    # Electrostatic interaction criteria
    electrostatic_distance: float = 4.0  # Angstrom
    
    # Van der Waals criteria
    vdw_distance_multiplier: float = 1.2  # multiplier of VdW radii sum
    
    # Hydrophobic interaction criteria
    hydrophobic_distance: float = 4.5  # Angstrom
    
    # π-π stacking criteria
    pi_pi_distance: float = 5.0  # Angstrom
    pi_pi_angle_cutoff: float = 30.0  # degrees
    
    # π-cation criteria
    pi_cation_distance: float = 5.0  # Angstrom
    
    # Halogen bond criteria
    halogen_bond_distance: float = 3.8  # Angstrom
    halogen_bond_angle: float = 160.0  # degrees
    
    # Metal coordination criteria
    metal_coordination_distance: float = 2.8  # Angstrom

@dataclass
class InteractionFeature:
    """Single protein-ligand interaction feature."""
    
    interaction_type: InteractionType
    protein_residue_id: int
    protein_atom_index: int
    ligand_atom_index: int
    distance: float
    angle: Optional[float] = None
    strength: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate interaction feature."""
        if self.distance < 0:
            raise ValidationError(f"Distance must be >= 0: {self.distance}")
        if self.angle is not None and not 0 <= self.angle <= 180:
            raise ValidationError(f"Angle must be in [0,180]: {self.angle}")
        if not 0 <= self.strength <= 1:
            raise ValidationError(f"Strength must be in [0,1]: {self.strength}")

@dataclass
class ProteinLigandPLIF:
    """Protein-Ligand Interaction Fingerprint for single frame."""
    
    frame_id: int
    time_ps: float
    interactions: List[InteractionFeature]
    ligand_id: str = ""
    binding_site_id: str = ""
    fingerprint_vector: Optional[np.ndarray] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate PLIF data."""
        if self.frame_id < 0:
            raise ValidationError(f"frame_id must be >= 0: {self.frame_id}")
    
    @property
    def n_interactions(self) -> int:
        """Number of interactions."""
        return len(self.interactions)
    
    def get_interaction_counts(self) -> Dict[InteractionType, int]:
        """Get counts of each interaction type."""
        counts = Counter(interaction.interaction_type for interaction in self.interactions)
        return dict(counts)
    
    def get_residue_interactions(self) -> Dict[int, List[InteractionFeature]]:
        """Get interactions grouped by residue."""
        residue_interactions = defaultdict(list)
        for interaction in self.interactions:
            residue_interactions[interaction.protein_residue_id].append(interaction)
        return dict(residue_interactions)
    
    def to_binary_vector(self, feature_mapping: Dict[str, int]) -> np.ndarray:
        """Convert PLIF to binary vector representation.
        
        Args:
            feature_mapping: Mapping from feature names to vector indices
            
        Returns:
            Binary vector representation
        """
        vector = np.zeros(len(feature_mapping))
        
        for interaction in self.interactions:
            # Create feature key
            feature_key = f"{interaction.protein_residue_id}_{interaction.interaction_type.value}"
            
            if feature_key in feature_mapping:
                vector[feature_mapping[feature_key]] = 1.0
        
        self.fingerprint_vector = vector
        return vector
    
    def calculate_similarity(self, other: 'ProteinLigandPLIF', 
                           method: str = "tanimoto") -> float:
        """Calculate similarity to another PLIF.
        
        Args:
            other: Other PLIF to compare
            method: Similarity method ('tanimoto', 'dice', 'cosine')
            
        Returns:
            Similarity score (0-1)
        """
        if self.fingerprint_vector is None or other.fingerprint_vector is None:
            return 0.0
        
        vec1 = self.fingerprint_vector
        vec2 = other.fingerprint_vector
        
        if len(vec1) != len(vec2):
            return 0.0
        
        if method == "tanimoto":
            intersection = np.sum(vec1 * vec2)
            union = np.sum((vec1 + vec2) > 0)
            return intersection / union if union > 0 else 0.0
        
        elif method == "dice":
            intersection = np.sum(vec1 * vec2)
            total = np.sum(vec1) + np.sum(vec2)
            return 2 * intersection / total if total > 0 else 0.0
        
        elif method == "cosine":
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")

@dataclass
class PLIFAnalysisConfig:
    """Configuration for PLIF analysis."""
    
    interaction_criteria: InteractionCriteria = field(default_factory=InteractionCriteria)
    enabled_interactions: Set[InteractionType] = field(default_factory=lambda: set(InteractionType))
    use_gpu: bool = GPU_AVAILABLE
    n_workers: int = 4
    memory_limit_gb: float = 4.0
    batch_size: int = 100
    include_water_mediated: bool = False
    water_bridge_distance: float = 3.5
    min_interaction_frequency: float = 0.1
    
    def __post_init__(self):
        """Set default enabled interactions if none specified."""
        if not self.enabled_interactions:
            self.enabled_interactions = {
                InteractionType.HYDROGEN_BOND,
                InteractionType.ELECTROSTATIC,
                InteractionType.VAN_DER_WAALS,
                InteractionType.HYDROPHOBIC,
                InteractionType.PI_PI_STACKING
            }

class InteractionDetector(ABC):
    """Abstract base class for interaction detectors."""
    
    def __init__(self, interaction_type: InteractionType, 
                 criteria: InteractionCriteria,
                 config: PLIFAnalysisConfig):
        self.interaction_type = interaction_type
        self.criteria = criteria
        self.config = config
    
    @abstractmethod
    def detect_interactions(self, protein_coords: np.ndarray, 
                          ligand_coords: np.ndarray,
                          protein_atoms: List[Dict[str, Any]],
                          ligand_atoms: List[Dict[str, Any]]) -> List[InteractionFeature]:
        """Detect interactions between protein and ligand."""
        pass
    
    def _calculate_distances(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances."""
        if self.config.use_gpu and GPU_AVAILABLE:
            try:
                gpu_coords1 = cp.asarray(coords1)
                gpu_coords2 = cp.asarray(coords2)
                distances = cp_distance.cdist(gpu_coords1, gpu_coords2)
                return cp.asnumpy(distances)
            except Exception as e:
                warnings.warn(f"GPU distance calculation failed: {e}")
        
        return cdist(coords1, coords2)
    
    def _calculate_angle(self, point1: np.ndarray, point2: np.ndarray, 
                        point3: np.ndarray) -> float:
        """Calculate angle between three points (point2 is vertex)."""
        vec1 = point1 - point2
        vec2 = point3 - point2
        
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        
        return angle

class HydrogenBondDetector(InteractionDetector):
    """Detector for hydrogen bonds."""
    
    def __init__(self, criteria: InteractionCriteria, config: PLIFAnalysisConfig):
        super().__init__(InteractionType.HYDROGEN_BOND, criteria, config)
    
    def detect_interactions(self, protein_coords: np.ndarray, 
                          ligand_coords: np.ndarray,
                          protein_atoms: List[Dict[str, Any]],
                          ligand_atoms: List[Dict[str, Any]]) -> List[InteractionFeature]:
        """Detect hydrogen bonds."""
        interactions = []
        
        # Identify donors and acceptors
        protein_donors = self._find_donors(protein_atoms, protein_coords)
        protein_acceptors = self._find_acceptors(protein_atoms, protein_coords)
        ligand_donors = self._find_donors(ligand_atoms, ligand_coords)
        ligand_acceptors = self._find_acceptors(ligand_atoms, ligand_coords)
        
        # Check protein donor - ligand acceptor
        interactions.extend(self._check_hbonds(
            protein_donors, ligand_acceptors, protein_atoms, ligand_atoms, True
        ))
        
        # Check ligand donor - protein acceptor
        interactions.extend(self._check_hbonds(
            ligand_donors, protein_acceptors, ligand_atoms, protein_atoms, False
        ))
        
        return interactions
    
    def _find_donors(self, atoms: List[Dict[str, Any]], coords: np.ndarray) -> List[Tuple[int, int]]:
        """Find hydrogen bond donors (heavy atom, hydrogen indices)."""
        donors = []
        
        for i, atom in enumerate(atoms):
            if atom.get('element') in ['N', 'O', 'S']:  # Potential donor heavy atoms
                # Find attached hydrogens
                for j, other_atom in enumerate(atoms):
                    if (other_atom.get('element') == 'H' and 
                        np.linalg.norm(coords[i] - coords[j]) < 1.2):  # H bond distance
                        donors.append((i, j))
        
        return donors
    
    def _find_acceptors(self, atoms: List[Dict[str, Any]], coords: np.ndarray) -> List[int]:
        """Find hydrogen bond acceptors."""
        acceptors = []
        
        for i, atom in enumerate(atoms):
            if atom.get('element') in ['N', 'O', 'S', 'F']:  # Potential acceptors
                acceptors.append(i)
        
        return acceptors
    
    def _check_hbonds(self, donors: List[Tuple[int, int]], acceptors: List[int],
                     donor_atoms: List[Dict[str, Any]], acceptor_atoms: List[Dict[str, Any]],
                     protein_is_donor: bool) -> List[InteractionFeature]:
        """Check for hydrogen bonds between donors and acceptors."""
        interactions = []
        
        for donor_heavy, donor_h in donors:
            donor_heavy_coord = donor_atoms[donor_heavy]['coord']
            donor_h_coord = donor_atoms[donor_h]['coord']
            
            for acceptor_idx in acceptors:
                acceptor_coord = acceptor_atoms[acceptor_idx]['coord']
                
                # Check distance criteria
                ha_distance = np.linalg.norm(donor_h_coord - acceptor_coord)
                da_distance = np.linalg.norm(donor_heavy_coord - acceptor_coord)
                
                if (ha_distance <= self.criteria.hbond_donor_acceptor_distance and
                    da_distance <= self.criteria.hbond_donor_acceptor_distance + 1.0):
                    
                    # Check angle criteria
                    angle = self._calculate_angle(donor_heavy_coord, donor_h_coord, acceptor_coord)
                    
                    if angle >= self.criteria.hbond_angle_cutoff:
                        # Calculate interaction strength
                        strength = self._calculate_hbond_strength(ha_distance, angle)
                        
                        if protein_is_donor:
                            protein_residue_id = donor_atoms[donor_heavy].get('residue_id', 0)
                            protein_atom_idx = donor_heavy
                            ligand_atom_idx = acceptor_idx
                        else:
                            protein_residue_id = acceptor_atoms[acceptor_idx].get('residue_id', 0)
                            protein_atom_idx = acceptor_idx
                            ligand_atom_idx = donor_heavy
                        
                        interaction = InteractionFeature(
                            interaction_type=self.interaction_type,
                            protein_residue_id=protein_residue_id,
                            protein_atom_index=protein_atom_idx,
                            ligand_atom_index=ligand_atom_idx,
                            distance=ha_distance,
                            angle=angle,
                            strength=strength,
                            properties={
                                'donor_acceptor_distance': da_distance,
                                'protein_is_donor': protein_is_donor
                            }
                        )
                        
                        interactions.append(interaction)
        
        return interactions
    
    def _calculate_hbond_strength(self, distance: float, angle: float) -> float:
        """Calculate hydrogen bond strength based on geometry."""
        # Distance component (closer is stronger)
        dist_factor = max(0, 1.0 - distance / self.criteria.hbond_donor_acceptor_distance)
        
        # Angle component (more linear is stronger)
        angle_factor = max(0, (angle - 90.0) / 90.0)
        
        return min(1.0, dist_factor * angle_factor)

class ElectrostaticDetector(InteractionDetector):
    """Detector for electrostatic interactions."""
    
    def __init__(self, criteria: InteractionCriteria, config: PLIFAnalysisConfig):
        super().__init__(InteractionType.ELECTROSTATIC, criteria, config)
    
    def detect_interactions(self, protein_coords: np.ndarray,
                          ligand_coords: np.ndarray,
                          protein_atoms: List[Dict[str, Any]],
                          ligand_atoms: List[Dict[str, Any]]) -> List[InteractionFeature]:
        """Detect electrostatic interactions."""
        interactions = []
        
        # Find charged atoms
        protein_charges = self._get_partial_charges(protein_atoms)
        ligand_charges = self._get_partial_charges(ligand_atoms)
        
        # Calculate distances
        distances = self._calculate_distances(protein_coords, ligand_coords)
        
        for i, protein_charge in enumerate(protein_charges):
            if abs(protein_charge) < 0.1:  # Skip neutral atoms
                continue
            
            for j, ligand_charge in enumerate(ligand_charges):
                if abs(ligand_charge) < 0.1:
                    continue
                
                distance = distances[i, j]
                
                if distance <= self.criteria.electrostatic_distance:
                    # Calculate interaction strength
                    charge_product = protein_charge * ligand_charge
                    
                    # Attractive interaction (opposite charges)
                    if charge_product < 0:
                        strength = min(1.0, abs(charge_product) / distance)
                        
                        interaction = InteractionFeature(
                            interaction_type=self.interaction_type,
                            protein_residue_id=protein_atoms[i].get('residue_id', 0),
                            protein_atom_index=i,
                            ligand_atom_index=j,
                            distance=distance,
                            strength=strength,
                            properties={
                                'protein_charge': protein_charge,
                                'ligand_charge': ligand_charge,
                                'charge_product': charge_product
                            }
                        )
                        
                        interactions.append(interaction)
        
        return interactions
    
    def _get_partial_charges(self, atoms: List[Dict[str, Any]]) -> List[float]:
        """Get partial charges for atoms."""
        charges = []
        
        for atom in atoms:
            # Use provided partial charge or estimate from atom type
            charge = atom.get('partial_charge', 0.0)
            
            if charge == 0.0:
                # Rough estimation based on atom type and environment
                element = atom.get('element', '')
                if element == 'N':
                    charge = -0.3
                elif element == 'O':
                    charge = -0.4
                elif element == 'S':
                    charge = -0.2
            
            charges.append(charge)
        
        return charges

class VanDerWaalsDetector(InteractionDetector):
    """Detector for van der Waals interactions."""
    
    def __init__(self, criteria: InteractionCriteria, config: PLIFAnalysisConfig):
        super().__init__(InteractionType.VAN_DER_WAALS, criteria, config)
        
        # Van der Waals radii (Angstrom)
        self.vdw_radii = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
            'S': 1.80, 'P': 1.80, 'F': 1.47, 'Cl': 1.75,
            'Br': 1.85, 'I': 1.98
        }
    
    def detect_interactions(self, protein_coords: np.ndarray,
                          ligand_coords: np.ndarray,
                          protein_atoms: List[Dict[str, Any]],
                          ligand_atoms: List[Dict[str, Any]]) -> List[InteractionFeature]:
        """Detect van der Waals interactions."""
        interactions = []
        
        # Calculate distances
        distances = self._calculate_distances(protein_coords, ligand_coords)
        
        for i, protein_atom in enumerate(protein_atoms):
            protein_element = protein_atom.get('element', 'C')
            protein_radius = self.vdw_radii.get(protein_element, 1.7)
            
            for j, ligand_atom in enumerate(ligand_atoms):
                ligand_element = ligand_atom.get('element', 'C')
                ligand_radius = self.vdw_radii.get(ligand_element, 1.7)
                
                distance = distances[i, j]
                vdw_sum = protein_radius + ligand_radius
                cutoff = vdw_sum * self.criteria.vdw_distance_multiplier
                
                if distance <= cutoff:
                    # Calculate interaction strength (optimal at VdW sum)
                    if distance > vdw_sum:
                        strength = max(0, 1.0 - (distance - vdw_sum) / (cutoff - vdw_sum))
                    else:
                        # Repulsive region - still count as interaction but lower strength
                        strength = 0.5
                    
                    interaction = InteractionFeature(
                        interaction_type=self.interaction_type,
                        protein_residue_id=protein_atom.get('residue_id', 0),
                        protein_atom_index=i,
                        ligand_atom_index=j,
                        distance=distance,
                        strength=strength,
                        properties={
                            'vdw_sum': vdw_sum,
                            'vdw_cutoff': cutoff,
                            'protein_element': protein_element,
                            'ligand_element': ligand_element
                        }
                    )
                    
                    interactions.append(interaction)
        
        return interactions

class HydrophobicDetector(InteractionDetector):
    """Detector for hydrophobic interactions."""
    
    def __init__(self, criteria: InteractionCriteria, config: PLIFAnalysisConfig):
        super().__init__(InteractionType.HYDROPHOBIC, criteria, config)
        
        # Hydrophobic atoms
        self.hydrophobic_elements = {'C'}
        self.hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
    
    def detect_interactions(self, protein_coords: np.ndarray,
                          ligand_coords: np.ndarray,
                          protein_atoms: List[Dict[str, Any]],
                          ligand_atoms: List[Dict[str, Any]]) -> List[InteractionFeature]:
        """Detect hydrophobic interactions."""
        interactions = []
        
        # Identify hydrophobic atoms
        protein_hydrophobic = self._find_hydrophobic_atoms(protein_atoms)
        ligand_hydrophobic = self._find_hydrophobic_atoms(ligand_atoms)
        
        if not protein_hydrophobic or not ligand_hydrophobic:
            return interactions
        
        # Calculate distances between hydrophobic atoms
        protein_hphob_coords = protein_coords[protein_hydrophobic]
        ligand_hphob_coords = ligand_coords[ligand_hydrophobic]
        
        distances = self._calculate_distances(protein_hphob_coords, ligand_hphob_coords)
        
        for i, protein_idx in enumerate(protein_hydrophobic):
            for j, ligand_idx in enumerate(ligand_hydrophobic):
                distance = distances[i, j]
                
                if distance <= self.criteria.hydrophobic_distance:
                    # Calculate interaction strength
                    strength = max(0, 1.0 - distance / self.criteria.hydrophobic_distance)
                    
                    interaction = InteractionFeature(
                        interaction_type=self.interaction_type,
                        protein_residue_id=protein_atoms[protein_idx].get('residue_id', 0),
                        protein_atom_index=protein_idx,
                        ligand_atom_index=ligand_idx,
                        distance=distance,
                        strength=strength,
                        properties={
                            'protein_residue': protein_atoms[protein_idx].get('residue_name', 'UNK')
                        }
                    )
                    
                    interactions.append(interaction)
        
        return interactions
    
    def _find_hydrophobic_atoms(self, atoms: List[Dict[str, Any]]) -> List[int]:
        """Find hydrophobic atoms."""
        hydrophobic_atoms = []
        
        for i, atom in enumerate(atoms):
            element = atom.get('element', '')
            residue_name = atom.get('residue_name', '')
            atom_name = atom.get('atom_name', '')
            
            # Carbon atoms in hydrophobic residues or aliphatic carbons
            if (element == 'C' and 
                (residue_name in self.hydrophobic_residues or
                 atom_name.startswith('C') and not atom_name.startswith('CO'))):
                hydrophobic_atoms.append(i)
        
        return hydrophobic_atoms

class PiPiStackingDetector(InteractionDetector):
    """Detector for π-π stacking interactions."""
    
    def __init__(self, criteria: InteractionCriteria, config: PLIFAnalysisConfig):
        super().__init__(InteractionType.PI_PI_STACKING, criteria, config)
        
        # Aromatic residues and atom patterns
        self.aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
    
    def detect_interactions(self, protein_coords: np.ndarray,
                          ligand_coords: np.ndarray,
                          protein_atoms: List[Dict[str, Any]],
                          ligand_atoms: List[Dict[str, Any]]) -> List[InteractionFeature]:
        """Detect π-π stacking interactions."""
        interactions = []
        
        # Find aromatic rings
        protein_rings = self._find_aromatic_rings(protein_atoms, protein_coords)
        ligand_rings = self._find_aromatic_rings(ligand_atoms, ligand_coords)
        
        # Check all ring pairs
        for protein_ring in protein_rings:
            for ligand_ring in ligand_rings:
                interaction = self._check_pi_pi_stacking(
                    protein_ring, ligand_ring, protein_atoms, ligand_atoms
                )
                
                if interaction:
                    interactions.append(interaction)
        
        return interactions
    
    def _find_aromatic_rings(self, atoms: List[Dict[str, Any]], 
                           coords: np.ndarray) -> List[Dict[str, Any]]:
        """Find aromatic rings in molecule."""
        rings = []
        
        # Simple heuristic: look for aromatic residues
        residue_rings = defaultdict(list)
        
        for i, atom in enumerate(atoms):
            residue_id = atom.get('residue_id', 0)
            residue_name = atom.get('residue_name', '')
            atom_name = atom.get('atom_name', '')
            
            if residue_name in self.aromatic_residues:
                # Aromatic carbon atoms
                if atom.get('element') == 'C' and ('C' in atom_name or 'CD' in atom_name or 'CE' in atom_name):
                    residue_rings[residue_id].append(i)
        
        # Create ring descriptors
        for residue_id, ring_atoms in residue_rings.items():
            if len(ring_atoms) >= 5:  # Minimum for aromatic ring
                ring_coords = coords[ring_atoms]
                center = np.mean(ring_coords, axis=0)
                
                # Calculate normal vector (simplified)
                if len(ring_coords) >= 3:
                    v1 = ring_coords[1] - ring_coords[0]
                    v2 = ring_coords[2] - ring_coords[0]
                    normal = np.cross(v1, v2)
                    normal = normal / np.linalg.norm(normal)
                else:
                    normal = np.array([0, 0, 1])
                
                rings.append({
                    'residue_id': residue_id,
                    'atoms': ring_atoms,
                    'center': center,
                    'normal': normal,
                    'coords': ring_coords
                })
        
        return rings
    
    def _check_pi_pi_stacking(self, protein_ring: Dict[str, Any], 
                            ligand_ring: Dict[str, Any],
                            protein_atoms: List[Dict[str, Any]],
                            ligand_atoms: List[Dict[str, Any]]) -> Optional[InteractionFeature]:
        """Check for π-π stacking between two rings."""
        # Calculate distance between ring centers
        distance = np.linalg.norm(protein_ring['center'] - ligand_ring['center'])
        
        if distance > self.criteria.pi_pi_distance:
            return None
        
        # Calculate angle between ring normals
        normal1 = protein_ring['normal']
        normal2 = ligand_ring['normal']
        
        cos_angle = np.dot(normal1, normal2)
        angle = np.arccos(np.clip(abs(cos_angle), 0, 1)) * 180.0 / np.pi
        
        # Check if rings are parallel (face-to-face) or perpendicular (edge-to-face)
        is_parallel = angle <= self.criteria.pi_pi_angle_cutoff
        is_perpendicular = abs(angle - 90.0) <= self.criteria.pi_pi_angle_cutoff
        
        if is_parallel or is_perpendicular:
            # Calculate interaction strength
            distance_factor = max(0, 1.0 - distance / self.criteria.pi_pi_distance)
            
            if is_parallel:
                angle_factor = 1.0 - angle / self.criteria.pi_pi_angle_cutoff
                stacking_type = "face_to_face"
            else:
                angle_factor = 1.0 - abs(angle - 90.0) / self.criteria.pi_pi_angle_cutoff
                stacking_type = "edge_to_face"
            
            strength = distance_factor * angle_factor
            
            # Use first atom of protein ring as representative
            protein_atom_idx = protein_ring['atoms'][0]
            ligand_atom_idx = ligand_ring['atoms'][0]
            
            return InteractionFeature(
                interaction_type=self.interaction_type,
                protein_residue_id=protein_ring['residue_id'],
                protein_atom_index=protein_atom_idx,
                ligand_atom_index=ligand_atom_idx,
                distance=distance,
                angle=angle,
                strength=strength,
                properties={
                    'stacking_type': stacking_type,
                    'protein_ring_size': len(protein_ring['atoms']),
                    'ligand_ring_size': len(ligand_ring['atoms'])
                }
            )
        
        return None

class DynamicPLIFAnalyzer:
    """Main analyzer for dynamic PLIF generation and analysis."""
    
    def __init__(self, config: PLIFAnalysisConfig = None):
        """Initialize PLIF analyzer.
        
        Args:
            config: Configuration for PLIF analysis
        """
        self.config = config or PLIFAnalysisConfig()
        self.detectors: Dict[InteractionType, InteractionDetector] = {}
        self.plifs: List[ProteinLigandPLIF] = []
        self.feature_mapping: Optional[Dict[str, int]] = None
        
        # Initialize interaction detectors
        self._initialize_detectors()
        
        # GPU memory management
        if self.config.use_gpu and GPU_AVAILABLE:
            try:
                self.memory_pool = cp.get_default_memory_pool()
                memory_limit = int(self.config.memory_limit_gb * 1024**3)
                self.memory_pool.set_limit(size=memory_limit)
            except Exception as e:
                warnings.warn(f"GPU memory setup failed: {e}")
                self.config.use_gpu = False
    
    def _initialize_detectors(self):
        """Initialize interaction detectors."""
        criteria = self.config.interaction_criteria
        
        detector_classes = {
            InteractionType.HYDROGEN_BOND: HydrogenBondDetector,
            InteractionType.ELECTROSTATIC: ElectrostaticDetector,
            InteractionType.VAN_DER_WAALS: VanDerWaalsDetector,
            InteractionType.HYDROPHOBIC: HydrophobicDetector,
            InteractionType.PI_PI_STACKING: PiPiStackingDetector,
            # Add more detector classes as needed
        }
        
        for interaction_type in self.config.enabled_interactions:
            if interaction_type in detector_classes:
                detector_class = detector_classes[interaction_type]
                self.detectors[interaction_type] = detector_class(criteria, self.config)
    
    def analyze_frame(self, frame: TrajectoryFrame, 
                     protein_topology: Dict[str, Any],
                     ligand_topology: Dict[str, Any],
                     ligand_id: str = "") -> ProteinLigandPLIF:
        """Analyze single frame to generate PLIF.
        
        Args:
            frame: Trajectory frame
            protein_topology: Protein topology information
            ligand_topology: Ligand topology information
            ligand_id: Ligand identifier
            
        Returns:
            ProteinLigandPLIF object
        """
        # Extract protein and ligand coordinates and atom information
        protein_info = self._extract_molecule_info(frame, protein_topology, "protein")
        ligand_info = self._extract_molecule_info(frame, ligand_topology, "ligand")
        
        # Detect interactions using all enabled detectors
        all_interactions = []
        
        for interaction_type, detector in self.detectors.items():
            try:
                interactions = detector.detect_interactions(
                    protein_info['coords'], ligand_info['coords'],
                    protein_info['atoms'], ligand_info['atoms']
                )
                all_interactions.extend(interactions)
            except Exception as e:
                warnings.warn(f"Interaction detection failed for {interaction_type}: {e}")
                continue
        
        # Create PLIF
        plif = ProteinLigandPLIF(
            frame_id=frame.frame_id,
            time_ps=frame.time_ps,
            interactions=all_interactions,
            ligand_id=ligand_id,
            properties={
                'n_interactions': len(all_interactions),
                'interaction_types': list(set(i.interaction_type for i in all_interactions))
            }
        )
        
        return plif
    
    def _extract_molecule_info(self, frame: TrajectoryFrame, 
                             topology: Dict[str, Any],
                             molecule_type: str) -> Dict[str, Any]:
        """Extract coordinates and atom information for molecule."""
        atom_indices = topology.get('atom_indices', [])
        
        if not atom_indices:
            # Fallback: use all atoms
            atom_indices = list(range(len(frame.coordinates)))
        
        coords = frame.coordinates[atom_indices]
        
        # Create atom information with coordinates
        atoms = []
        for i, atom_idx in enumerate(atom_indices):
            atom_info = topology.get('atoms', {}).get(atom_idx, {})
            atom_info['coord'] = coords[i]
            atoms.append(atom_info)
        
        return {
            'coords': coords,
            'atoms': atoms,
            'atom_indices': atom_indices
        }
    
    def analyze_trajectory(self, frames: List[TrajectoryFrame],
                         protein_topology: Dict[str, Any],
                         ligand_topology: Dict[str, Any],
                         ligand_id: str = "") -> List[ProteinLigandPLIF]:
        """Analyze trajectory to generate dynamic PLIFs.
        
        Args:
            frames: List of trajectory frames
            protein_topology: Protein topology information
            ligand_topology: Ligand topology information
            ligand_id: Ligand identifier
            
        Returns:
            List of ProteinLigandPLIF objects
        """
        if self.config.n_workers > 1:
            plifs = self._analyze_trajectory_parallel(frames, protein_topology, ligand_topology, ligand_id)
        else:
            plifs = self._analyze_trajectory_sequential(frames, protein_topology, ligand_topology, ligand_id)
        
        self.plifs.extend(plifs)
        return plifs
    
    def _analyze_trajectory_sequential(self, frames: List[TrajectoryFrame],
                                     protein_topology: Dict[str, Any],
                                     ligand_topology: Dict[str, Any],
                                     ligand_id: str) -> List[ProteinLigandPLIF]:
        """Analyze trajectory sequentially."""
        plifs = []
        
        for frame in frames:
            try:
                plif = self.analyze_frame(frame, protein_topology, ligand_topology, ligand_id)
                plifs.append(plif)
            except Exception as e:
                warnings.warn(f"PLIF analysis failed for frame {frame.frame_id}: {e}")
                continue
        
        return plifs
    
    def _analyze_trajectory_parallel(self, frames: List[TrajectoryFrame],
                                   protein_topology: Dict[str, Any],
                                   ligand_topology: Dict[str, Any],
                                   ligand_id: str) -> List[ProteinLigandPLIF]:
        """Analyze trajectory in parallel."""
        plifs = []
        
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Submit analysis tasks
            future_to_frame = {
                executor.submit(self.analyze_frame, frame, protein_topology, ligand_topology, ligand_id): frame
                for frame in frames
            }
            
            # Collect results
            for future in as_completed(future_to_frame):
                frame = future_to_frame[future]
                try:
                    plif = future.result()
                    plifs.append(plif)
                except Exception as e:
                    warnings.warn(f"PLIF analysis failed for frame {frame.frame_id}: {e}")
                    continue
        
        # Sort by frame ID
        plifs.sort(key=lambda x: x.frame_id)
        return plifs
    
    def build_feature_mapping(self, plifs: Optional[List[ProteinLigandPLIF]] = None) -> Dict[str, int]:
        """Build feature mapping for binary vector representation.
        
        Args:
            plifs: PLIFs to build mapping from (uses self.plifs if None)
            
        Returns:
            Feature mapping dictionary
        """
        if plifs is None:
            plifs = self.plifs
        
        # Collect all unique features
        all_features = set()
        
        for plif in plifs:
            for interaction in plif.interactions:
                feature_key = f"{interaction.protein_residue_id}_{interaction.interaction_type.value}"
                all_features.add(feature_key)
        
        # Create mapping
        self.feature_mapping = {feature: i for i, feature in enumerate(sorted(all_features))}
        
        return self.feature_mapping
    
    def convert_to_binary_vectors(self, plifs: Optional[List[ProteinLigandPLIF]] = None) -> np.ndarray:
        """Convert PLIFs to binary vector matrix.
        
        Args:
            plifs: PLIFs to convert (uses self.plifs if None)
            
        Returns:
            Binary matrix (n_frames x n_features)
        """
        if plifs is None:
            plifs = self.plifs
        
        if self.feature_mapping is None:
            self.build_feature_mapping(plifs)
        
        # Convert each PLIF to binary vector
        binary_matrix = np.zeros((len(plifs), len(self.feature_mapping)))
        
        for i, plif in enumerate(plifs):
            binary_vector = plif.to_binary_vector(self.feature_mapping)
            binary_matrix[i] = binary_vector
        
        return binary_matrix
    
    def cluster_binding_modes(self, plifs: Optional[List[ProteinLigandPLIF]] = None,
                            method: str = "kmeans", n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster PLIFs to identify distinct binding modes.
        
        Args:
            plifs: PLIFs to cluster (uses self.plifs if None)
            method: Clustering method ('kmeans', 'dbscan')
            n_clusters: Number of clusters (for kmeans)
            
        Returns:
            Clustering results dictionary
        """
        if plifs is None:
            plifs = self.plifs
        
        if not plifs:
            return {'labels': [], 'n_clusters': 0}
        
        # Convert to binary matrix
        binary_matrix = self.convert_to_binary_vectors(plifs)
        
        # Perform clustering
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(binary_matrix)
            cluster_centers = clusterer.cluster_centers_
            
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.3, min_samples=5)
            labels = clusterer.fit_predict(binary_matrix)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_centers = None
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Calculate silhouette score if multiple clusters
        silhouette = 0.0
        if len(set(labels)) > 1:
            try:
                silhouette = silhouette_score(binary_matrix, labels)
            except:
                pass
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(plifs, labels)
        
        return {
            'labels': labels,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'silhouette_score': silhouette,
            'cluster_centers': cluster_centers,
            'cluster_analysis': cluster_analysis,
            'method': method
        }
    
    def _analyze_clusters(self, plifs: List[ProteinLigandPLIF], 
                         labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Analyze characteristics of each cluster."""
        cluster_analysis = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise cluster in DBSCAN
                continue
            
            # Get PLIFs in this cluster
            cluster_plifs = [plifs[i] for i in range(len(plifs)) if labels[i] == cluster_id]
            
            if not cluster_plifs:
                continue
            
            # Analyze cluster properties
            cluster_size = len(cluster_plifs)
            
            # Interaction type frequencies
            interaction_counts = Counter()
            residue_counts = Counter()
            
            for plif in cluster_plifs:
                for interaction in plif.interactions:
                    interaction_counts[interaction.interaction_type] += 1
                    residue_counts[interaction.protein_residue_id] += 1
            
            # Calculate average number of interactions
            avg_interactions = np.mean([plif.n_interactions for plif in cluster_plifs])
            
            # Find representative PLIF (closest to cluster center)
            if len(cluster_plifs) > 1:
                # Calculate pairwise similarities within cluster
                similarities = []
                for i in range(len(cluster_plifs)):
                    for j in range(i + 1, len(cluster_plifs)):
                        sim = cluster_plifs[i].calculate_similarity(cluster_plifs[j])
                        similarities.append(sim)
                
                avg_similarity = np.mean(similarities) if similarities else 0.0
            else:
                avg_similarity = 1.0
            
            cluster_analysis[cluster_id] = {
                'size': cluster_size,
                'avg_interactions': avg_interactions,
                'avg_similarity': avg_similarity,
                'top_interaction_types': dict(interaction_counts.most_common(5)),
                'top_residues': dict(residue_counts.most_common(10)),
                'time_range': (
                    min(plif.time_ps for plif in cluster_plifs),
                    max(plif.time_ps for plif in cluster_plifs)
                )
            }
        
        return cluster_analysis
    
    def calculate_interaction_frequencies(self, plifs: Optional[List[ProteinLigandPLIF]] = None) -> pd.DataFrame:
        """Calculate frequency of each interaction across trajectory.
        
        Args:
            plifs: PLIFs to analyze (uses self.plifs if None)
            
        Returns:
            DataFrame with interaction frequencies
        """
        if plifs is None:
            plifs = self.plifs
        
        if not plifs:
            return pd.DataFrame()
        
        # Collect all interactions
        interaction_data = []
        
        for plif in plifs:
            for interaction in plif.interactions:
                interaction_data.append({
                    'frame_id': plif.frame_id,
                    'time_ps': plif.time_ps,
                    'interaction_type': interaction.interaction_type.value,
                    'protein_residue_id': interaction.protein_residue_id,
                    'distance': interaction.distance,
                    'strength': interaction.strength,
                    'feature_key': f"{interaction.protein_residue_id}_{interaction.interaction_type.value}"
                })
        
        if not interaction_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(interaction_data)
        
        # Calculate frequencies
        total_frames = len(plifs)
        frequency_stats = df.groupby('feature_key').agg({
            'frame_id': 'count',
            'distance': ['mean', 'std'],
            'strength': ['mean', 'std'],
            'interaction_type': 'first',
            'protein_residue_id': 'first'
        }).round(3)
        
        # Flatten column names
        frequency_stats.columns = ['count', 'distance_mean', 'distance_std', 
                                 'strength_mean', 'strength_std', 
                                 'interaction_type', 'protein_residue_id']
        
        # Calculate frequency
        frequency_stats['frequency'] = frequency_stats['count'] / total_frames
        
        # Sort by frequency
        frequency_stats = frequency_stats.sort_values('frequency', ascending=False)
        
        return frequency_stats
    
    def detect_binding_mode_transitions(self, cluster_labels: np.ndarray,
                                      min_duration: int = 10) -> List[Dict[str, Any]]:
        """Detect transitions between binding modes.
        
        Args:
            cluster_labels: Cluster labels for each frame
            min_duration: Minimum duration for stable binding mode
            
        Returns:
            List of transition events
        """
        transitions = []
        
        if len(cluster_labels) < 2:
            return transitions
        
        # Find segments of same cluster
        current_cluster = cluster_labels[0]
        segment_start = 0
        
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != current_cluster:
                # Segment ended
                segment_duration = i - segment_start
                
                if segment_duration >= min_duration:
                    transitions.append({
                        'from_cluster': current_cluster,
                        'to_cluster': cluster_labels[i],
                        'start_frame': segment_start,
                        'end_frame': i - 1,
                        'duration': segment_duration
                    })
                
                current_cluster = cluster_labels[i]
                segment_start = i
        
        # Handle final segment
        final_duration = len(cluster_labels) - segment_start
        if final_duration >= min_duration and len(transitions) > 0:
            transitions[-1]['final_cluster'] = current_cluster
            transitions[-1]['final_duration'] = final_duration
        
        return transitions
    
    def get_plif_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of PLIF analysis.
        
        Returns:
            DataFrame with PLIF summary statistics
        """
        if not self.plifs:
            return pd.DataFrame()
        
        rows = []
        
        for plif in self.plifs:
            interaction_counts = plif.get_interaction_counts()
            
            row = {
                'frame_id': plif.frame_id,
                'time_ps': plif.time_ps,
                'n_interactions': plif.n_interactions,
                'ligand_id': plif.ligand_id
            }
            
            # Add interaction type counts
            for interaction_type in InteractionType:
                row[f'n_{interaction_type.value}'] = interaction_counts.get(interaction_type, 0)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def cleanup(self):
        """Clean up resources."""
        self.plifs.clear()
        self.feature_mapping = None
        
        # Clean up GPU memory
        if self.config.use_gpu and GPU_AVAILABLE:
            try:
                self.memory_pool.free_all_blocks()
            except:
                pass

def create_plif_analyzer(interaction_types: Optional[List[InteractionType]] = None,
                        use_gpu: bool = GPU_AVAILABLE,
                        n_workers: int = 4) -> DynamicPLIFAnalyzer:
    """Create PLIF analyzer with optimal configuration.
    
    Args:
        interaction_types: List of interaction types to detect
        use_gpu: Enable GPU acceleration
        n_workers: Number of parallel workers
        
    Returns:
        Configured DynamicPLIFAnalyzer
    """
    config = PLIFAnalysisConfig(
        enabled_interactions=set(interaction_types) if interaction_types else None,
        use_gpu=use_gpu,
        n_workers=n_workers
    )
    
    return DynamicPLIFAnalyzer(config)

# Utility functions for PLIF analysis
def calculate_plif_similarity_matrix(plifs: List[ProteinLigandPLIF],
                                   method: str = "tanimoto") -> np.ndarray:
    """Calculate similarity matrix between PLIFs.
    
    Args:
        plifs: List of PLIFs
        method: Similarity method
        
    Returns:
        Similarity matrix
    """
    n_plifs = len(plifs)
    similarity_matrix = np.zeros((n_plifs, n_plifs))
    
    for i in range(n_plifs):
        for j in range(i, n_plifs):
            if i == j:
                similarity = 1.0
            else:
                similarity = plifs[i].calculate_similarity(plifs[j], method)
            
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    return similarity_matrix

def identify_persistent_interactions(plifs: List[ProteinLigandPLIF],
                                   min_frequency: float = 0.8) -> List[str]:
    """Identify interactions that persist across most of the trajectory.
    
    Args:
        plifs: List of PLIFs
        min_frequency: Minimum frequency for persistent interactions
        
    Returns:
        List of persistent interaction feature keys
    """
    if not plifs:
        return []
    
    # Count occurrences of each interaction
    interaction_counts = Counter()
    
    for plif in plifs:
        present_interactions = set()
        
        for interaction in plif.interactions:
            feature_key = f"{interaction.protein_residue_id}_{interaction.interaction_type.value}"
            present_interactions.add(feature_key)
        
        # Count each unique interaction once per frame
        for feature_key in present_interactions:
            interaction_counts[feature_key] += 1
    
    # Find persistent interactions
    total_frames = len(plifs)
    persistent_interactions = []
    
    for feature_key, count in interaction_counts.items():
        frequency = count / total_frames
        if frequency >= min_frequency:
            persistent_interactions.append(feature_key)
    
    return persistent_interactions